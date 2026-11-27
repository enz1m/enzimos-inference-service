// SPDX-License-Identifier: MIT
//! Explicit local CPU calibration and hardware probe; no downloads, service
//! activation or GPU performance claims.
use crate::{policy, supervisor::Supervisor};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

const MAX_FILE: u64 = 2 * 1024 * 1024;

fn read_value(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_owned())
        .map_err(|_| "resource limits unavailable".into())
}

/// The calibration must run inside a bounded cgroup v2 scope.
pub fn check_limits(group: &Path) -> Result<(), String> {
    let memory = read_value(&group.join("memory.max"))?;
    let swap = read_value(&group.join("memory.swap.max"))?;
    let tasks = read_value(&group.join("pids.max"))?;
    let cpu = read_value(&group.join("cpu.max"))?;
    let cpu = cpu.split_whitespace().collect::<Vec<_>>();
    if !memory
        .parse::<u64>()
        .is_ok_and(|m| m > 0 && m <= 2 * 1024 * 1024 * 1024)
        || swap != "0"
    {
        return Err("calibration requires MemoryMax<=2GiB and MemorySwapMax=0".into());
    }
    let quota = cpu.first().and_then(|q| q.parse::<u64>().ok());
    let period = cpu.get(1).and_then(|p| p.parse::<u64>().ok());
    if !tasks.parse::<u64>().is_ok_and(|t| t > 0 && t <= 64)
        || !matches!((quota, period), (Some(q), Some(p)) if q > 0 && q <= p)
    {
        return Err("calibration requires TasksMax<=64 and CPUQuota<=100%".into());
    }
    Ok(())
}

pub fn resource_group() -> Result<PathBuf, String> {
    let text = std::fs::read_to_string("/proc/self/cgroup")
        .map_err(|_| "a cgroup v2 resource scope is required")?;
    let entry = text
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .ok_or("a cgroup v2 resource scope is required")?;
    let root = std::fs::canonicalize("/sys/fs/cgroup").map_err(|_| "invalid resource scope")?;
    let path = std::fs::canonicalize(root.join(entry.trim_start_matches('/')))
        .map_err(|_| "invalid resource scope")?;
    if !path.starts_with(&root) {
        return Err("invalid resource scope".into());
    }
    check_limits(&path)?;
    Ok(path)
}

/// The last `limit` bytes of `text`, moved forward to a character boundary.
fn tail(text: &str, limit: usize) -> String {
    let mut start = text.len().saturating_sub(limit);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_owned()
}

#[allow(unsafe_code, reason = "Linux syscall ABI (getuid).")]
pub fn read_private(path: &Path) -> Result<Value, String> {
    let refuse = || "configuration must be an owned private regular file".to_owned();
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| refuse())?;
    let m = file.metadata().map_err(|_| refuse())?;
    if !m.is_file()
        || m.uid() != unsafe { libc::getuid() }
        || m.mode() & 0o7777 != 0o600
        || m.nlink() != 1
        || m.len() > MAX_FILE
    {
        return Err(refuse());
    }
    let mut raw = Vec::new();
    (&mut file)
        .take(MAX_FILE + 1)
        .read_to_end(&mut raw)
        .map_err(|_| refuse())?;
    if raw.len() as u64 > MAX_FILE {
        return Err(refuse());
    }
    serde_json::from_slice(&raw).map_err(|_| "invalid configuration".into())
}

/// A new owned 0600 file in an owned 0700 directory; never an overwrite.
#[allow(unsafe_code, reason = "Linux syscall ABI (getuid).")]
pub fn create_private(path: &Path, value: &Value) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("output directory must be owned and mode0700")?;
    let m = std::fs::symlink_metadata(parent)
        .map_err(|_| "output directory must be owned and mode0700")?;
    if !m.is_dir() || m.uid() != unsafe { libc::getuid() } || m.mode() & 0o7777 != 0o700 {
        return Err("output directory must be owned and mode0700".into());
    }
    let mut raw = serde_json::to_vec_pretty(value).map_err(|_| "invalid receipt")?;
    raw.push(b'\n');
    if raw.len() as u64 > MAX_FILE {
        return Err("receipt/configuration exceeds limit".into());
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|_| "output exists or is unavailable")?;
    if file.write_all(&raw).and_then(|()| file.sync_all()).is_err() {
        let _ = std::fs::remove_file(path);
        return Err("output could not be written".into());
    }
    Ok(())
}

/// One non-streaming JSON request to the native runtime, with bounded framing.
pub async fn native_request(socket: &Path, endpoint: &str, body: &Value) -> Result<Value, String> {
    let stream = tokio::net::UnixStream::connect(socket)
        .await
        .map_err(|_| "native runtime unavailable")?;
    let (reader, mut writer) = stream.into_split();
    writer
        .write_all(&crate::http::post(endpoint, "localhost", &[], body))
        .await
        .map_err(|_| "native runtime unavailable")?;
    let mut reader = BufReader::new(reader);
    let mut status = String::new();
    reader
        .read_line(&mut status)
        .await
        .map_err(|_| "native runtime unavailable")?;
    if !status.starts_with("HTTP/1.1 200 ") {
        return Err("native runtime refused calibration".into());
    }
    let mut length = None;
    let mut size = 0usize;
    loop {
        let mut line = String::new();
        if reader
            .read_line(&mut line)
            .await
            .map_err(|_| "native runtime unavailable")?
            == 0
        {
            break;
        }
        size += line.len();
        if size > 16384 {
            return Err("native headers exceed limit".into());
        }
        if line == "\r\n" {
            break;
        }
        let (key, value) = line.split_once(':').ok_or("invalid native header")?;
        match key.to_ascii_lowercase().as_str() {
            "transfer-encoding" => return Err("unsupported calibration framing".into()),
            "content-length" => length = value.trim().parse::<u64>().ok(),
            _ => {}
        }
    }
    let length = length
        .filter(|l| *l <= MAX_FILE)
        .ok_or("native reply exceeds limit")?;
    let mut body = vec![0u8; length as usize];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|_| "native reply incomplete")?;
    serde_json::from_slice(&body).map_err(|_| "invalid native reply".into())
}

pub fn qualified_row(
    config: &Value,
    model: &str,
    hw: &Value,
    backend: Option<&str>,
    failure: Option<&str>,
    reply: &Value,
    peak: f64,
) -> Result<Value, String> {
    let timings = &reply["timings"];
    let (prompt_n, predicted_n) = (
        timings["prompt_n"].as_u64(),
        timings["predicted_n"].as_u64(),
    );
    if prompt_n != Some(4096) || !predicted_n.is_some_and(|n| (1..=32).contains(&n)) {
        return Err("runtime did not process the requested 4k workload".into());
    }
    if !reply["content"].as_str().is_some_and(|c| !c.is_empty())
        || failure.is_some()
        || backend != Some("cpu")
    {
        return Err("runtime calibration failed".into());
    }
    let (prompt_ms, predicted_ms) = (
        timings["prompt_ms"].as_f64().unwrap_or(-1.0),
        timings["predicted_ms"].as_f64().unwrap_or(-1.0),
    );
    if [prompt_ms, predicted_ms, peak]
        .iter()
        .any(|n| !n.is_finite() || *n <= 0.0)
    {
        return Err("invalid measurements".into());
    }
    if peak > policy::memory_mib(config) {
        return Err("calibration exceeds configured RAM budget".into());
    }
    let predicted_n = predicted_n.unwrap() as f64;
    let score = prompt_ms / 1000.0 + predicted_ms / 1000.0 * 256.0 / predicted_n;
    if !(score > 0.0 && score < 3600.0) {
        return Err("request estimate exceeds the supported calibration range".into());
    }
    let runtime = &config["runtimes"]["cpu"];
    let options = &config["models"][model]["backend_options"]["cpu"];
    let server = policy::file_sha256(Path::new(runtime["server"].as_str().unwrap_or_default()))
        .map_err(|_| "runtime unavailable")?;
    // A runtime/schema/full-context smoke gate, not model quality.
    Ok(
        json!({"backend": "cpu", "hardware_sha256": policy::fingerprint(hw), "model_sha256": config["models"][model]["sha256"],
        "runtime_revision": runtime["revision"], "runtime_artifact_sha256": runtime.get("artifact_sha256").cloned().unwrap_or(Value::Null), "runtime_server_sha256": server,
        "context": policy::context_for_model(config, model), "threads": hw["worker_cpus"].as_array().map_or(0, Vec::len),
        "cache_type": options.get("cache_type").cloned().unwrap_or(json!("q8_0")), "repack": options.get("repack").cloned().unwrap_or(json!(true)),
        "correctness_pass": true, "correctness_scope": "CPU native response/full 4096-token input; not semantic model quality",
        "peak_total_mib": peak, "estimated_request_seconds": score,
        "workload": {"prompt_tokens": 4096, "measured_output_tokens": predicted_n as u64, "estimated_output_tokens": 256},
        "memory_accounting": "max(cgroup memory peak, sampled native+supervisor RSS); CPU only"}),
    )
}

fn self_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map_or(0.0, |k| k / 1024.0)
}

#[allow(unsafe_code, reason = "Linux syscall ABI (kill).")]
pub async fn calibrate(config: &Value, model: &str, group: &Path) -> Result<Value, String> {
    policy::validate(config, true)?;
    if config["models"].get(model).is_none() || policy::kind(config, model) != "llm" {
        return Err("registered text model required".into());
    }
    if policy::context_for_model(config, model) < 4128 {
        return Err("4k prompt plus output headroom required".into());
    }
    let runtime = &config["runtimes"]["cpu"];
    if !runtime["revision"].as_str().is_some_and(|r| !r.is_empty())
        || !runtime["artifact_sha256"]
            .as_str()
            .is_some_and(policy::hex64)
    {
        return Err("runtime revision and artifact digest required".into());
    }
    let server = PathBuf::from(runtime["server"].as_str().unwrap_or_default());
    let before = policy::file_sha256(&server).map_err(|_| "runtime unavailable")?;
    // A new private runtime directory: the service, its jobs and configuration
    // stay untouched, and no generated text is saved.
    let mut selected = config.clone();
    selected["backend"] = json!("cpu");
    selected["calibration"] = json!([]);
    let base =
        std::env::var("XDG_RUNTIME_DIR").map_or_else(|_| std::env::temp_dir(), PathBuf::from);
    let directory = base.join(format!(
        "enzimos-calibrate-{}",
        enzimos_platform::files::random_hex(8).map_err(|_| "random")?
    ));
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&directory)
        .map_err(|_| "calibration directory unavailable")?;
    let result = async {
        let supervisor = Supervisor::new(selected.clone(), &directory, None)?;
        let peak = std::sync::Arc::new(std::sync::Mutex::new(0.0f64));
        let monitor = {
            let (supervisor, peak, group, budget) = (supervisor.clone(), peak.clone(), group.to_path_buf(), policy::memory_mib(config));
            tokio::spawn(async move {
                loop {
                    let cgroup = std::fs::read_to_string(group.join("memory.peak")).ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0) / (1024.0 * 1024.0);
                    let pid = supervisor.state.lock().unwrap().pid;
                    let native = pid.map_or(0.0, |p| std::fs::read_to_string(format!("/proc/{p}/status")).ok().and_then(|t| t.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok())).map_or(0.0, |k| k / 1024.0));
                    let total = cgroup.max(native + self_rss_mib());
                    {
                        let mut p = peak.lock().unwrap();
                        *p = p.max(total);
                    }
                    if total > budget {
                        supervisor.state.lock().unwrap().failure = Some("calibration exceeded RAM budget".into());
                        if let Some(pid) = supervisor.state.lock().unwrap().pid {
                            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
                        }
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            })
        };
        let run = async {
            let mut engine = supervisor.engine.lock().await;
            supervisor.start(&mut engine, model).await.map_err(|e| format!("{e:?}"))?;
            let socket = supervisor.engine_socket();
            let tokenized = native_request(&socket, "/tokenize", &json!({"content": "The quick brown fox jumps over the lazy dog. ", "add_special": false})).await?;
            let tokens = tokenized["tokens"].as_array().filter(|t| !t.is_empty() && t.len() <= 256 && t.iter().all(serde_json::Value::is_u64)).ok_or("invalid tokenizer reply")?.clone();
            let prompt = tokens.iter().cycle().take(4096).cloned().collect::<Vec<_>>();
            let reply = native_request(&socket, "/completion", &json!({"prompt": prompt, "n_predict": 32, "temperature": 0, "seed": 0, "cache_prompt": false, "stream": false})).await?;
            tokio::time::sleep(Duration::from_millis(60)).await;
            policy::validate(config, true)?;
            if policy::file_sha256(&server).map_err(|_| "runtime unavailable")? != before {
                return Err("runtime changed during calibration".into());
            }
            let (hw, backend, failure) = {
                let s = supervisor.state.lock().unwrap();
                (s.hw.clone(), s.backend.clone(), s.failure.clone())
            };
            let peak = *peak.lock().unwrap();
            let row = qualified_row(&selected, model, &hw, backend.as_deref(), failure.as_deref(), &reply, peak);
            supervisor.stop(&mut engine).await;
            row
        };
        let result = tokio::time::timeout(Duration::from_secs(1800), run).await.map_err(|_| "calibration timed out".to_owned()).and_then(|r| r);
        monitor.abort();
        let mut engine = supervisor.engine.lock().await;
        supervisor.stop(&mut engine).await;
        result
    }
    .await;
    let _ = std::fs::remove_dir_all(&directory);
    result
}

/// Enumerate runtimes and devices; enumeration is not a performance qualification.
pub fn probe(config: &Value) -> Value {
    let hw = policy::hardware();
    let mut runtimes = Vec::new();
    for backend in policy::BACKENDS {
        let Some(runtime) = config["runtimes"].get(backend) else {
            continue;
        };
        let mut command =
            std::process::Command::new(runtime["server"].as_str().unwrap_or_default());
        command
            .arg("--list-devices")
            .env_clear()
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        for key in ["PATH", "HOME", "LANG", "XDG_RUNTIME_DIR"] {
            if let Ok(v) = std::env::var(key) {
                command.env(key, v);
            }
        }
        for (k, v) in runtime["environment"].as_object().into_iter().flatten() {
            command.env(k, v.as_str().unwrap_or_default());
        }
        let output = run_bounded(command, Duration::from_secs(20));
        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let devices = stdout.lines().filter_map(|l| {
                    let (id, description) = l.trim_start().split_once(':')?;
                    let digits = ["Vulkan", "SYCL", "CUDA"].iter().find_map(|p| id.strip_prefix(p))?;
                    (!digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit())).then(|| json!({"id": id, "description": description.trim()}))
                }).collect::<Vec<_>>();
                let code = output.status.code();
                let stderr = String::from_utf8_lossy(&output.stderr);
                runtimes.push(json!({"backend": backend, "revision": runtime.get("revision"), "available": code == Some(0) && (backend == "cpu" || !devices.is_empty()),
                    "devices": devices, "exit_code": code, "error": (code != Some(0)).then(|| tail(&stderr, 1024))}));
            }
            Err(_) => runtimes.push(json!({"backend": backend, "available": false, "devices": [], "error": "runtime or driver unavailable"})),
        }
    }
    json!({"hardware": hw, "hardware_sha256": policy::fingerprint(&hw), "runtimes": runtimes, "default_without_calibration": "cpu", "note": "Enumeration is not a performance/memory qualification."})
}

/// Run with a deadline, killing the child when it expires.
#[allow(unsafe_code, reason = "Linux syscall ABI (kill).")]
fn run_bounded(
    mut command: std::process::Command,
    limit: Duration,
) -> std::io::Result<std::process::Output> {
    // Its own group, so helpers that inherited the output pipes die too and
    // the waiter can finish reading.
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let child = command.spawn()?;
    let pid = child.id();
    let (done, wait) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let output = child.wait_with_output();
        let _ = done.send(());
        output
    });
    if wait.recv_timeout(limit).is_err() {
        unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
        let _ = waiter.join();
        return Err(std::io::Error::from(std::io::ErrorKind::TimedOut));
    }
    waiter
        .join()
        .map_err(|_| std::io::Error::other("probe failed"))?
}

trait Mode {
    fn mode(&mut self, mode: u32) -> &mut Self;
}
impl Mode for std::fs::DirBuilder {
    fn mode(&mut self, mode: u32) -> &mut Self {
        std::os::unix::fs::DirBuilderExt::mode(self, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn required_aggregate_limits() {
        let dir = tempfile::tempdir().unwrap();
        let write = |f: &str, v: &str| std::fs::write(dir.path().join(f), v).unwrap();
        write("memory.max", "2147483648\n");
        write("memory.swap.max", "0\n");
        write("pids.max", "64\n");
        write("cpu.max", "100000 100000\n");
        check_limits(dir.path()).unwrap();
        write("memory.max", "max\n");
        assert!(check_limits(dir.path()).is_err());
        write("memory.max", "2147483648\n");
        write("cpu.max", "max 100000\n");
        assert!(check_limits(dir.path()).is_err());
        write("cpu.max", "100000 100000\n");
        write("memory.swap.max", "max\n");
        assert!(check_limits(dir.path()).is_err());
    }

    #[test]
    fn private_files_refuse_overwrite_links_and_public_configs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = dir.path().join("receipt.json");
        create_private(&path, &json!({"a": 1})).unwrap();
        assert!(create_private(&path, &json!({"a": 2})).is_err());
        assert_eq!(read_private(&path).unwrap()["a"], 1);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_private(&path).is_err());
        std::os::unix::fs::symlink(&path, dir.path().join("link")).unwrap();
        assert!(read_private(&dir.path().join("link")).is_err());
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(create_private(&dir.path().join("other.json"), &json!({})).is_err());
    }

    #[test]
    fn only_full_context_cpu_receipts_qualify() {
        let config = json!({"memory_mib": 1900, "runtimes": {"cpu": {"server": "/proc/self/exe", "revision": "r"}}, "models": {"m": {"sha256": "s"}}});
        let hw = json!({"worker_cpus": [1, 2]});
        let reply = |prompt: u64| json!({"content": "x", "timings": {"prompt_n": prompt, "predicted_n": 32, "prompt_ms": 2000.0, "predicted_ms": 1000.0}});
        let row = qualified_row(&config, "m", &hw, Some("cpu"), None, &reply(4096), 800.0).unwrap();
        assert_eq!(row["estimated_request_seconds"], 10.0);
        assert!(qualified_row(&config, "m", &hw, Some("cpu"), None, &reply(100), 800.0).is_err());
        assert!(
            qualified_row(&config, "m", &hw, Some("vulkan"), None, &reply(4096), 800.0).is_err()
        );
        assert!(qualified_row(&config, "m", &hw, Some("cpu"), None, &reply(4096), 1950.0).is_err());
    }
}
