// SPDX-License-Identifier: MIT
//! Resource and model policy independent of the runtime implementation.
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Read, os::unix::fs::MetadataExt, path::Path};

pub const BACKENDS: [&str; 4] = ["cpu", "vulkan", "sycl", "cuda"];

pub fn get_u64(config: &Value, key: &str, default: u64) -> u64 {
    config.get(key).and_then(Value::as_u64).unwrap_or(default)
}
pub fn context(config: &Value) -> u64 {
    get_u64(config, "context", 4608)
}
pub fn memory_mib(config: &Value) -> f64 {
    config
        .get("memory_mib")
        .and_then(Value::as_f64)
        .unwrap_or(1900.0)
}

/// Python's `json.dumps(value, sort_keys=True)`: ", " and ": " separators and
/// ASCII escapes. Calibration receipts bind hardware with this exact encoding.
pub fn python_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            let items = keys
                .iter()
                .map(|k| {
                    format!(
                        "{}: {}",
                        python_json(&json!(k)),
                        python_json(&map[k.as_str()])
                    )
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", items.join(", "))
        }
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        other => String::from_utf8(enzimos_platform::json::canonical(other).unwrap_or_default())
            .unwrap_or_default(),
    }
}

pub fn fingerprint(hw: &Value) -> String {
    format!("{:x}", Sha256::digest(python_json(hw).as_bytes()))
}

pub fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut chunk = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            return Ok(format!("{:x}", digest.finalize()));
        }
        digest.update(&chunk[..n]);
    }
}

/// File identity that changes with any write or metadata change; `ctime`
/// cannot be set back by the owner.
fn identity(path: &Path) -> std::io::Result<[i64; 7]> {
    let m = fs::symlink_metadata(path)?;
    Ok([
        m.dev() as i64,
        m.ino() as i64,
        m.len() as i64,
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ])
}

/// Whether `path` still has digest `sha`. A file whose identity matches an
/// earlier full check is not read again, so a service start does not hash
/// several gigabytes of models. Verified identities persist in the service
/// state directory when systemd provides one.
pub fn digest_matches(path: &Path, sha: &str) -> std::io::Result<bool> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<Map<String, Value>>> = Mutex::new(None);
    let store =
        std::env::var_os("STATE_DIRECTORY").map(|d| Path::new(&d).join("model-digests.json"));
    let key = path.to_string_lossy().into_owned();
    let before = identity(path)?;
    let entry = json!({"identity": before, "sha256": sha});
    {
        let mut cache = CACHE.lock().unwrap();
        let cache = cache.get_or_insert_with(|| {
            store
                .as_ref()
                .and_then(|p| fs::read(p).ok())
                .and_then(|d| serde_json::from_slice::<Map<String, Value>>(&d).ok())
                .unwrap_or_default()
        });
        if cache.get(&key) == Some(&entry) {
            return Ok(true);
        }
    }
    let matches = file_sha256(path)? == sha;
    if matches && identity(path)? == before {
        let mut cache = CACHE.lock().unwrap();
        let cache = cache.get_or_insert_with(Map::new);
        cache.insert(key, entry);
        if cache.len() > 64 {
            let first = cache.keys().next().cloned().unwrap_or_default();
            cache.remove(&first);
        }
        if let Some(store) = store {
            let temporary = store.with_extension(format!("{}.tmp", std::process::id()));
            if fs::write(&temporary, Value::Object(cache.clone()).to_string())
                .and_then(|()| fs::rename(&temporary, &store))
                .is_err()
            {
                let _ = fs::remove_file(&temporary);
            }
        }
    }
    Ok(matches)
}

#[allow(
    unsafe_code,
    reason = "Linux syscall ABI (CPU_ISSET, sched_getaffinity)."
)]
pub fn allowed_cpus() -> Vec<u32> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } != 0
    {
        return vec![0];
    }
    (0..libc::CPU_SETSIZE as u32)
        .filter(|c| unsafe { libc::CPU_ISSET(*c as usize, &set) })
        .collect()
}

fn siblings(cpu: u32) -> String {
    fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    ))
    .map_or_else(|_| cpu.to_string(), |s| s.trim().to_owned())
}

/// At most two worker CPUs, avoiding both SMT siblings of one core.
pub fn cpu_budget_with(
    available: &[u32],
    requested: Option<u64>,
    core_of: &dyn Fn(u32) -> String,
) -> Vec<u32> {
    let mut available = available.to_vec();
    available.sort();
    let wanted = requested.unwrap_or(available.len() as u64 / 4) as usize;
    let count = wanted.min(2).min(available.len()).max(1);
    let mut chosen = Vec::new();
    let mut seen = BTreeSet::new();
    for cpu in available.iter().rev() {
        if seen.insert(core_of(*cpu)) {
            chosen.push(*cpu);
        }
        if chosen.len() == count {
            break;
        }
    }
    chosen.sort();
    chosen
}

pub fn cpu_budget(requested: Option<u64>) -> Vec<u32> {
    cpu_budget_with(&allowed_cpus(), requested, &siblings)
}

#[allow(unsafe_code, reason = "Linux syscall ABI (uname).")]
pub fn hardware() -> Value {
    let mut devices = Vec::new();
    let mut nodes = fs::read_dir("/sys/class/drm")
        .map(|d| {
            d.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("renderD"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    nodes.sort();
    for node in nodes {
        let device = Path::new("/sys/class/drm").join(&node).join("device");
        let read = |f: &str| fs::read_to_string(device.join(f)).map(|s| s.trim().to_owned());
        if let (Ok(vendor), Ok(id), Ok(driver)) = (
            read("vendor"),
            read("device"),
            fs::canonicalize(device.join("driver")),
        ) {
            devices.push(json!({"render_node": format!("/dev/dri/{node}"), "vendor": vendor, "device": id, "driver": driver.file_name().map(|n| n.to_string_lossy().into_owned())}));
        }
    }
    let mut drivers = Map::new();
    for directory in ["/usr/lib", "/usr/lib/x86_64-linux-gnu"] {
        for name in [
            "libvulkan_intel.so",
            "libze_intel_gpu.so.1",
            "libigdrcl.so",
            "libcuda.so.1",
        ] {
            let path = Path::new(directory).join(name);
            if let (Ok(m), Ok(resolved)) = (fs::metadata(&path), fs::canonicalize(&path)) {
                drivers.insert(resolved.to_string_lossy().into_owned(), json!({"size": m.len(), "mtime_ns": m.mtime() * 1_000_000_000 + m.mtime_nsec()}));
            }
        }
    }
    let cpu_model = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".into());
    let mut uname: libc::utsname = unsafe { std::mem::zeroed() };
    unsafe { libc::uname(&mut uname) };
    let release = unsafe { std::ffi::CStr::from_ptr(uname.release.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    json!({"cpu_model": cpu_model, "kernel_release": release, "driver_files": drivers, "allowed_cpus": allowed_cpus(), "worker_cpus": cpu_budget(None), "devices": devices})
}

fn alias(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
}

pub fn hex64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub fn validate(config: &Value, verify: bool) -> Result<(), String> {
    if config.get("schema") != Some(&json!(1)) {
        return Err("unsupported configuration schema".into());
    }
    if !(4096..=8192).contains(&context(config)) {
        return Err("context outside supported policy".into());
    }
    if !(1..=3600).contains(&get_u64(config, "idle_seconds", 60)) {
        return Err("invalid idle interval".into());
    }
    if !(256.0..=2048.0).contains(&memory_mib(config)) {
        return Err("memory budget above 2 GiB".into());
    }
    if !matches!(config.get("cpu_threads"), None | Some(Value::Null))
        && !matches!(config["cpu_threads"].as_u64(), Some(1 | 2))
    {
        return Err("CPU thread override must be1 or2".into());
    }
    let runtimes = config["runtimes"].as_object().cloned().unwrap_or_default();
    for (name, item) in &runtimes {
        if !BACKENDS.contains(&name.as_str())
            || !item["server"]
                .as_str()
                .is_some_and(|s| Path::new(s).is_file())
        {
            return Err("invalid runtime".into());
        }
    }
    if !runtimes.contains_key("cpu") {
        return Err("CPU fallback required".into());
    }
    let models = config["models"]
        .as_object()
        .filter(|m| !m.is_empty())
        .ok_or("explicit model registry required")?;
    for (name, item) in models {
        if !alias(name) {
            return Err("invalid model alias".into());
        }
        let kind = item.get("kind").and_then(Value::as_str).unwrap_or("llm");
        if !["llm", "speech"].contains(&kind) {
            return Err("unsupported model kind".into());
        }
        // A sherpa-onnx speech model is a folder of files, each with a digest.
        if kind == "speech" && item.get("files").is_some() {
            crate::models::folder::validate(item, verify)?;
            continue;
        }
        if !matches!(
            item.get("architecture").and_then(Value::as_str),
            None | Some("whisper")
        ) && kind == "speech"
        {
            return Err("unsupported speech model architecture".into());
        }
        let path = Path::new(item["path"].as_str().unwrap_or_default());
        let regular = fs::symlink_metadata(path).is_ok_and(|m| m.is_file());
        if !path.is_absolute() || !regular {
            return Err("invalid model path".into());
        }
        let sha = item["sha256"].as_str().unwrap_or_default();
        if !hex64(sha) {
            return Err("model digest required".into());
        }
        if verify && !digest_matches(path, sha).map_err(|_| "invalid model path")? {
            return Err("model digest mismatch".into());
        }
    }
    if !config["default_model"]
        .as_str()
        .is_some_and(|m| models.contains_key(m))
    {
        return Err("default model absent".into());
    }
    if let Some(speech) = config.get("default_speech_model")
        && !speech
            .as_str()
            .is_some_and(|m| models.contains_key(m) && kind(config, m) == "speech")
    {
        return Err("default speech model absent".into());
    }
    Ok(())
}

pub fn kind(config: &Value, model: &str) -> String {
    config["models"][model]
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("llm")
        .to_owned()
}

/// Calibration rows that exactly match this model, runtime build, hardware and limits.
pub fn calibrated(
    config: &Value,
    model: &str,
    hw: &Value,
    allow_profiles: bool,
) -> Vec<(f64, String, Value)> {
    let item = &config["models"][model];
    let mut scores = Vec::new();
    for row in config["calibration"].as_array().into_iter().flatten() {
        let backend = row["backend"].as_str().unwrap_or_default();
        let Some(runtime) = config["runtimes"].get(backend) else {
            continue;
        };
        if row["hardware_sha256"] != fingerprint(hw).as_str()
            || row["model_sha256"] != item["sha256"]
        {
            continue;
        }
        if row.get("runtime_revision").unwrap_or(&Value::Null)
            != runtime.get("revision").unwrap_or(&Value::Null)
            || row.get("runtime_artifact_sha256").unwrap_or(&Value::Null)
                != runtime.get("artifact_sha256").unwrap_or(&Value::Null)
        {
            continue;
        }
        if let Some(expected) = row
            .get("runtime_server_sha256")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            && file_sha256(Path::new(runtime["server"].as_str().unwrap_or_default()))
                .ok()
                .as_deref()
                != Some(expected)
        {
            continue;
        }
        if row["context"] != context(config)
            || row["threads"] != hw["worker_cpus"].as_array().map_or(0, Vec::len)
        {
            continue;
        }
        let options = &item["backend_options"][backend];
        let profile = json!({"cache_type": row.get("cache_type").cloned().unwrap_or(json!("q8_0")), "repack": row.get("repack").cloned().unwrap_or(json!(true))});
        if !["q8_0", "f16"].contains(&profile["cache_type"].as_str().unwrap_or_default())
            || !profile["repack"].is_boolean()
        {
            continue;
        }
        let configured = (
            options.get("cache_type").cloned().unwrap_or(json!("q8_0")),
            options.get("repack").cloned().unwrap_or(json!(true)),
        );
        if !allow_profiles
            && (profile["cache_type"] != configured.0 || profile["repack"] != configured.1)
        {
            continue;
        }
        if row["correctness_pass"] != true
            || row
                .get("peak_total_mib")
                .and_then(Value::as_f64)
                .unwrap_or(99999.0)
                > memory_mib(config)
        {
            continue;
        }
        let Some(score) = row["estimated_request_seconds"]
            .as_f64()
            .filter(|s| *s > 0.0 && *s < 3600.0)
        else {
            continue;
        };
        scores.push((score, backend.to_owned(), profile));
    }
    scores
}

fn best(mut scores: Vec<(f64, String, Value)>) -> Option<(f64, String, Value)> {
    scores.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scores.into_iter().next()
}

pub fn execution_plan(config: &Value, model: &str, hw: &Value) -> Result<Value, String> {
    let mut backend = config
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or("auto")
        .to_owned();
    if backend == "auto" {
        if let Some((_, backend, profile)) = best(calibrated(config, model, hw, true)) {
            return Ok(
                json!({"backend": backend, "cache_type": profile["cache_type"], "repack": profile["repack"], "calibrated": true}),
            );
        }
        backend = "cpu".into();
    }
    if config["runtimes"].get(&backend).is_none() {
        return Err("requested runtime not installed".into());
    }
    let options = &config["models"][model]["backend_options"][&backend];
    Ok(
        json!({"backend": backend, "cache_type": options.get("cache_type").cloned().unwrap_or(json!("q8_0")), "repack": options.get("repack").cloned().unwrap_or(json!(true)), "calibrated": false}),
    )
}

pub fn server_command(
    config: &Value,
    model: &str,
    backend: &str,
    socket: &Path,
    profile: Option<&Value>,
    cpus: usize,
) -> Result<Vec<String>, String> {
    let runtime = &config["runtimes"][backend];
    let item = &config["models"][model];
    let options = profile.cloned().unwrap_or_else(|| {
        item["backend_options"]
            .get(backend)
            .cloned()
            .unwrap_or(json!({}))
    });
    let cache = options
        .get("cache_type")
        .and_then(Value::as_str)
        .unwrap_or("q8_0");
    if !["f16", "q8_0"].contains(&cache) {
        return Err("unqualified KV cache type".into());
    }
    let s = |v: &str| v.to_owned();
    let mut args = vec![
        s(runtime["server"].as_str().unwrap_or_default()),
        s("--model"),
        s(item["path"].as_str().unwrap_or_default()),
        s("--alias"),
        s(model),
        s("--host"),
        socket.to_string_lossy().into_owned(),
        s("--ctx-size"),
        context(config).to_string(),
        s("--parallel"),
        s("1"),
        s("--threads"),
        cpus.to_string(),
        s("--threads-batch"),
        cpus.to_string(),
        s("--threads-http"),
        s("2"),
        s("--poll"),
        s("0"),
        s("--poll-batch"),
        s("0"),
        s("--batch-size"),
        s("256"),
        s("--ubatch-size"),
        s("128"),
        s("--cache-type-k"),
        s(cache),
        s("--cache-type-v"),
        s(cache),
        s("--flash-attn"),
        s("on"),
        s("--cache-ram"),
        s("0"),
        s("--no-webui"),
        s("--no-ui-mcp-proxy"),
        s("--jinja"),
        s("--min-p"),
        s("0"),
        s("--temp"),
        s("0.7"),
        s("--top-p"),
        s("0.95"),
        s("--fit"),
        s("off"),
        s("--n-gpu-layers"),
        if backend == "cpu" {
            s("0")
        } else {
            runtime
                .get("gpu_layers")
                .and_then(Value::as_u64)
                .unwrap_or(99)
                .to_string()
        },
    ];
    if options.get("repack") == Some(&json!(false)) {
        args.push(s("--no-repack"));
    }
    let device = if backend == "cpu" {
        "none".to_owned()
    } else {
        runtime
            .get("device")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
    };
    if device.is_empty() {
        return Err("GPU device must be explicit".into());
    }
    args.extend([s("--device"), device]);
    Ok(args)
}

const RESERVED: [&str; 14] = [
    "id_slot",
    "slot_id",
    "save_prompt",
    "prompt_cache",
    "n_keep",
    "n_ctx",
    "timings_per_token",
    "lora",
    "max_completion_tokens",
    "n",
    "best_of",
    "n_probs",
    "logprobs",
    "response_fields",
];

/// Bound a text request: known endpoint and model, 1–512 output tokens, no
/// prompt-cache writes or slot overrides, thinking off by default.
pub fn check_request(
    path: &str,
    body: &mut Map<String, Value>,
    config: &Value,
) -> Result<String, String> {
    if !["/v1/chat/completions", "/completion", "/tokenize"].contains(&path) {
        return Err("unsupported endpoint".into());
    }
    let model = body
        .get("model")
        .cloned()
        .unwrap_or_else(|| config["default_model"].clone());
    let model = model
        .as_str()
        .filter(|m| config["models"].get(*m).is_some() && kind(config, m) == "llm")
        .ok_or("unknown text model")?
        .to_owned();
    if path != "/tokenize" {
        if path == "/v1/chat/completions" && body.contains_key("n_predict") {
            return Err("native output override is reserved".into());
        }
        let key = if path == "/completion" {
            "n_predict"
        } else {
            "max_tokens"
        };
        let count = body.get(key).cloned().unwrap_or(json!(256));
        if !count
            .as_u64()
            .is_some_and(|c| count.is_u64() && (1..=512).contains(&c))
        {
            return Err("output limit is 1–512 tokens".into());
        }
        body.insert(key.into(), count);
        if path == "/v1/chat/completions" && !body.get("messages").is_some_and(Value::is_array) {
            return Err("messages required".into());
        }
        if path == "/completion"
            && !body
                .get("prompt")
                .is_some_and(|p| p.is_string() || p.is_array())
        {
            return Err("prompt required".into());
        }
        body.insert("cache_prompt".into(), json!(false));
        if RESERVED.iter().any(|k| body.contains_key(*k)) {
            return Err("request contains a reserved runtime option".into());
        }
        body.entry("chat_template_kwargs")
            .or_insert(json!({"enable_thinking": false}));
    }
    body.insert("model".into(), json!(model));
    Ok(model)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    pub(crate) fn fixture(dir: &Path) -> Value {
        let model = dir.join("model.gguf");
        fs::write(&model, b"weights").unwrap();
        let server = dir.join("llama-server");
        fs::write(&server, "#!/bin/sh\n").unwrap();
        fs::set_permissions(&server, fs::Permissions::from_mode(0o755)).unwrap();
        json!({"schema": 1, "default_model": "m", "runtimes": {"cpu": {"server": server, "revision": "r1", "artifact_sha256": "a".repeat(64)}},
               "models": {"m": {"path": model, "sha256": file_sha256(&model).unwrap()}}})
    }

    #[test]
    fn digest_tampering_and_bad_paths_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = fixture(dir.path());
        validate(&config, true).unwrap();
        fs::write(dir.path().join("model.gguf"), b"other").unwrap();
        assert_eq!(
            validate(&config, true).unwrap_err(),
            "model digest mismatch"
        );
        let mut relative = config.clone();
        relative["models"]["m"]["path"] = json!("model.gguf");
        assert_eq!(
            validate(&relative, false).unwrap_err(),
            "invalid model path"
        );
        let mut no_cpu = config.clone();
        no_cpu["runtimes"] = json!({});
        assert_eq!(
            validate(&no_cpu, false).unwrap_err(),
            "CPU fallback required"
        );
        let mut big = config;
        big["memory_mib"] = json!(4096);
        assert_eq!(
            validate(&big, false).unwrap_err(),
            "memory budget above 2 GiB"
        );
    }

    #[test]
    fn cpu_caps_avoid_smt_siblings() {
        let core = |c: u32| (c / 2).to_string();
        assert_eq!(
            cpu_budget_with(&(0..16).collect::<Vec<_>>(), None, &core),
            [13, 15]
        );
        assert_eq!(cpu_budget_with(&[0, 1], None, &core), [1]);
        assert_eq!(
            cpu_budget_with(&(0..16).collect::<Vec<_>>(), Some(1), &core),
            [15]
        );
    }

    #[test]
    fn output_limits_and_reserved_options() {
        let dir = tempfile::tempdir().unwrap();
        let config = fixture(dir.path());
        let mut body = json!({"messages": []}).as_object().unwrap().clone();
        assert_eq!(
            check_request("/v1/chat/completions", &mut body, &config).unwrap(),
            "m"
        );
        assert_eq!(
            (body["max_tokens"].as_u64(), body["cache_prompt"].as_bool()),
            (Some(256), Some(false))
        );
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        for bad in [
            json!({"messages": [], "max_tokens": 513}),
            json!({"messages": [], "n_predict": 5}),
            json!({"messages": [], "slot_id": 0}),
            json!({"model": "x", "messages": []}),
        ] {
            assert!(
                check_request(
                    "/v1/chat/completions",
                    &mut bad.as_object().unwrap().clone(),
                    &config
                )
                .is_err(),
                "{bad}"
            );
        }
        assert!(
            check_request(
                "/completion",
                &mut json!({"prompt": [1, 2]}).as_object().unwrap().clone(),
                &config
            )
            .is_ok()
        );
        assert!(check_request("/props", &mut Map::new(), &config).is_err());
    }

    #[test]
    fn explicit_gpu_device_and_cpu_offload() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = fixture(dir.path());
        let cpu = server_command(&config, "m", "cpu", Path::new("/run/e.sock"), None, 2).unwrap();
        assert!(
            cpu.windows(2).any(|w| w == ["--n-gpu-layers", "0"])
                && cpu.ends_with(&["--device".to_owned(), "none".to_owned()])
        );
        config["runtimes"]["vulkan"] = json!({"server": config["runtimes"]["cpu"]["server"]});
        assert_eq!(
            server_command(&config, "m", "vulkan", Path::new("/e"), None, 2).unwrap_err(),
            "GPU device must be explicit"
        );
        config["runtimes"]["vulkan"]["device"] = json!("Vulkan0");
        assert!(
            server_command(&config, "m", "vulkan", Path::new("/e"), None, 2)
                .unwrap()
                .ends_with(&["--device".to_owned(), "Vulkan0".to_owned()])
        );
    }

    #[test]
    fn calibration_is_bound_to_exact_inputs_and_memory() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = fixture(dir.path());
        config["runtimes"]["vulkan"] = json!({"server": config["runtimes"]["cpu"]["server"], "revision": "v1", "device": "Vulkan0"});
        let hw = json!({"cpu_model": "x", "worker_cpus": [2, 3]});
        let row = json!({"backend": "vulkan", "hardware_sha256": fingerprint(&hw), "model_sha256": config["models"]["m"]["sha256"], "runtime_revision": "v1",
            "context": 4608, "threads": 2, "correctness_pass": true, "peak_total_mib": 1500, "estimated_request_seconds": 4.0});
        config["calibration"] = json!([row]);
        assert_eq!(
            execution_plan(&config, "m", &hw).unwrap()["backend"],
            "vulkan"
        );
        config["calibration"][0]["peak_total_mib"] = json!(1950);
        assert_eq!(execution_plan(&config, "m", &hw).unwrap()["backend"], "cpu");
        config["calibration"][0]["peak_total_mib"] = json!(1500);
        config["calibration"][0]["runtime_revision"] = json!("v0");
        assert_eq!(execution_plan(&config, "m", &hw).unwrap()["backend"], "cpu");
        assert_eq!(
            execution_plan(&json!({"backend": "cuda", "runtimes": {}}), "m", &hw).unwrap_err(),
            "requested runtime not installed"
        );
    }

    #[test]
    fn hardware_fingerprint_preserves_the_persisted_json_encoding() {
        assert_eq!(
            python_json(&json!({"b": [1, 2], "a": "\u{e9}"})),
            "{\"a\": \"\\u00e9\", \"b\": [1, 2]}"
        );
    }
}
