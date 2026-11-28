// SPDX-License-Identifier: MIT
//! On-demand per-user native inference supervisor, private Unix HTTP only.
use crate::{
    http,
    policy::{self, check_request, cpu_budget, execution_plan, server_command},
    preferences::{self, Preferences},
};
use base64::Engine as _;
use serde_json::{Map, Value, json};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream, unix::OwnedWriteHalf},
    process::{Child, Command},
};

#[derive(Debug)]
pub enum Failure {
    Bad(String),
    Conflict(String),
    Unavailable(String),
    /// A model load failure with a public, machine-readable code.
    ModelLoad(&'static str),
    Closed,
}
const MEMORY_BUDGET_EXCEEDED: &str = "model exceeded the configured RAM budget";

fn inference_failure(failure: Option<String>, fallback: &str) -> Failure {
    match failure.as_deref() {
        Some(MEMORY_BUDGET_EXCEEDED) => Failure::ModelLoad("model-out-of-memory"),
        Some(message) => unavailable(message),
        None => unavailable(fallback),
    }
}

fn bad(m: impl Into<String>) -> Failure {
    Failure::Bad(m.into())
}
fn unavailable(m: impl Into<String>) -> Failure {
    Failure::Unavailable(m.into())
}

/// Shared observations; never held across an await.
pub struct State {
    pub config: Value,
    pub hw: Value,
    pub failure: Option<String>,
    pub jobs: usize,
    pub clients: usize,
    pub last_used: Instant,
    pub peak_rss_mib: f64,
    pub peak_driver_ram_mib: f64,
    pub pid: Option<u32>,
    pub loaded: Option<String>,
    pub backend: Option<String>,
    pub profile: Value,
}

/// The native worker; held across a request, so one job runs at a time.
#[derive(Default)]
pub struct Engine {
    child: Option<Child>,
    cache_key: Option<CacheKey>,
    preloaded: bool,
}

struct CacheKey {
    session: Option<String>,
    prefix: Option<Value>,
}

fn static_prefix(path: &str, body: &Map<String, Value>) -> Option<Value> {
    if path != "/v1/chat/completions" {
        return None;
    }
    let messages = body.get("messages")?.as_array()?;
    let system = messages.first()?;
    if system["role"] != "system"
        || !system["content"].is_string()
        || messages.get(1)?["role"] != "user"
        || !messages[1]["content"].is_string()
    {
        return None;
    }
    let mut prefix = json!({
        "model": body["model"],
        "messages": [
            {"role": "system", "content": system["content"]},
            {"role": "user", "content": ""},
        ],
        "chat_template_kwargs": body["chat_template_kwargs"],
        "add_generation_prompt": false,
        "cache_prompt": true,
        "max_tokens": 1,
        "stream": false,
    });
    for key in ["tools", "tool_choice"] {
        if let Some(value) = body.get(key) {
            prefix[key] = value.clone();
        }
    }
    (prefix.to_string().len() <= 8192).then_some(prefix)
}

pub struct Supervisor {
    pub state: Mutex<State>,
    pub engine: tokio::sync::Mutex<Engine>,
    pub directory: PathBuf,
    base: Value,
    preferences: Mutex<Preferences>,
    uid: u32,
    /// Wakes the monitor as soon as a worker starts.
    started: tokio::sync::Notify,
}

const IDLE_MEMORY_RESERVE_MIB: f64 = 768.0;
const PRELOAD_IDLE_SECONDS: u64 = 30;

fn available_memory_mib() -> Option<f64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?;
    Some(kib.split_whitespace().next()?.parse::<f64>().ok()? / 1024.0)
}

const PREFERENCE_KEYS: [&str; 6] = [
    "cpu_threads",
    "memory_mib",
    "idle_seconds",
    "context",
    "backend",
    "default_model",
];

mod power_boost;
use power_boost::Boost;
mod process;
use process::{driver_ram_mib, rss_mib, spawn};
mod benchmark;
mod speech;

/// Decrements the job count however the request ends, including cancellation.
struct Job<'a>(&'a Supervisor);
impl Drop for Job<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        state.jobs -= 1;
        state.last_used = Instant::now();
    }
}

/// Kills the worker if a request is abandoned mid-flight (caller disconnect).
struct Abort<'a> {
    supervisor: &'a Supervisor,
    armed: bool,
}
impl Drop for Abort<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.supervisor.kill_worker();
        }
    }
}

/// Removes a speech job's scratch audio and worker identity, even on cancel.
struct Speech<'a> {
    supervisor: &'a Supervisor,
    scratch: PathBuf,
}
impl Drop for Speech<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.scratch);
        let mut state = self.supervisor.state.lock().unwrap();
        state.pid = None;
        state.loaded = None;
    }
}

impl Supervisor {
    #[allow(unsafe_code, reason = "Linux syscall ABI (getuid).")]
    pub fn new(
        config: Value,
        directory: &Path,
        preferences: Option<&Path>,
    ) -> Result<Arc<Self>, String> {
        policy::validate(&config, true)?;
        let base = config.clone();
        let defaults = json!({"cpu_threads": config.get("cpu_threads").cloned().unwrap_or(Value::Null), "memory_mib": config.get("memory_mib").cloned().unwrap_or(json!(1900)),
            "idle_seconds": config.get("idle_seconds").cloned().unwrap_or_else(|| json!(60)), "context": config["context"].clone(),
            "backend": config.get("backend").cloned().unwrap_or(json!("auto")), "default_model": config["default_model"]});
        let checked = config.clone();
        let validate: preferences::Validate = Box::new(move |value| {
            if value.len() != PREFERENCE_KEYS.len()
                || !PREFERENCE_KEYS.iter().all(|k| value.contains_key(*k))
            {
                return Err("unknown inference preference".into());
            }
            if !value["cpu_threads"].is_null()
                && !matches!(value["cpu_threads"].as_u64(), Some(1 | 2))
            {
                return Err("CPU threads must be1 or2 or null".into());
            }
            for (key, low, high) in [("memory_mib", 256, 2048), ("idle_seconds", 1, 3600)] {
                if !value[key]
                    .as_u64()
                    .is_some_and(|v| value[key].is_u64() && (low..=high).contains(&v))
                {
                    return Err("resource preference outside supported policy".into());
                }
            }
            if !value["context"].is_null()
                && !value["context"]
                    .as_u64()
                    .is_some_and(|value| (4096..=8192).contains(&value))
            {
                return Err("context outside supported policy".into());
            }
            let backend = value["backend"].as_str().unwrap_or_default();
            if backend != "auto" && checked["runtimes"].get(backend).is_none() {
                return Err("requested runtime not installed".into());
            }
            if !policy::valid_text_default(&checked, &value["default_model"]) {
                return Err("registered text model required".into());
            }
            Ok(value)
        });
        let preferences =
            Preferences::new(defaults.as_object().unwrap().clone(), preferences, validate)
                .map_err(|e| format!("{e:?}"))?;
        let mut effective = config;
        for (k, v) in &preferences.value {
            effective[k] = v.clone();
        }
        let mut hw = policy::hardware();
        hw["worker_cpus"] = json!(cpu_budget(effective["cpu_threads"].as_u64()));
        Ok(Arc::new(Self {
            state: Mutex::new(State {
                config: effective,
                hw,
                failure: None,
                jobs: 0,
                clients: 0,
                last_used: Instant::now(),
                peak_rss_mib: 0.0,
                peak_driver_ram_mib: 0.0,
                pid: None,
                loaded: None,
                backend: None,
                profile: Value::Null,
            }),
            engine: tokio::sync::Mutex::new(Engine::default()),
            directory: directory.to_path_buf(),
            base,
            preferences: Mutex::new(preferences),
            uid: unsafe { libc::getuid() },
            started: tokio::sync::Notify::new(),
        }))
    }

    pub fn engine_socket(&self) -> PathBuf {
        self.directory.join("engine.sock")
    }

    fn worker_cpus(&self) -> Vec<u32> {
        let state = self.state.lock().unwrap();
        state.hw["worker_cpus"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| c.as_u64().map(|c| c as u32))
            .collect()
    }

    /// SIGKILL the worker without waiting for it; used where nothing can be awaited.
    #[allow(unsafe_code, reason = "Linux syscall ABI (kill).")]
    fn kill_worker(&self) {
        let mut state = self.state.lock().unwrap();
        if let Some(pid) = state.pid.take() {
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        }
        state.loaded = None;
    }

    #[allow(unsafe_code, reason = "Linux syscall ABI (kill).")]
    pub async fn stop(&self, engine: &mut Engine) {
        let child = engine.child.take();
        engine.cache_key = None;
        engine.preloaded = false;
        {
            let mut state = self.state.lock().unwrap();
            state.pid = None;
            state.loaded = None;
        }
        if let Some(mut child) = child
            && child.try_wait().ok().flatten().is_none()
        {
            if let Some(pid) = child.id() {
                unsafe { libc::kill(pid as i32, libc::SIGTERM) };
            }
            if tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .is_err()
            {
                let _ = child.kill().await;
            }
        }
        let _ = std::fs::remove_file(self.engine_socket());
    }

    async fn healthy(&self) -> bool {
        let Ok(mut stream) = UnixStream::connect(self.engine_socket()).await else {
            return false;
        };
        if stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .is_err()
        {
            return false;
        }
        let mut line = [0u8; 64];
        let Ok(Ok(n)) = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut line)).await
        else {
            return false;
        };
        line[..n].windows(5).any(|w| w == b" 200 ")
    }

    /// Load `model`, falling back from a GPU backend to CPU if it cannot start.
    pub async fn start(&self, engine: &mut Engine, model: &str) -> Result<(), Failure> {
        let alive = engine
            .child
            .as_mut()
            .is_some_and(|c| c.try_wait().ok().flatten().is_none());
        if alive && self.state.lock().unwrap().loaded.as_deref() == Some(model) {
            return Ok(());
        }
        self.stop(engine).await;
        let (config, hw) = {
            let s = self.state.lock().unwrap();
            (s.config.clone(), s.hw.clone())
        };
        // The runtime opens the model by path; confirm it is still the
        // verified file (cheap while its identity is unchanged).
        let entry = &config["models"][model];
        if !policy::digest_matches(
            Path::new(entry["path"].as_str().unwrap_or_default()),
            entry["sha256"].as_str().unwrap_or_default(),
        )
        .unwrap_or(false)
        {
            return Err(unavailable("model file changed since it was verified"));
        }
        let plan = execution_plan(&config, model, &hw).map_err(bad)?;
        let selected = plan["backend"].as_str().unwrap_or("cpu").to_owned();
        let mut candidates = vec![selected.clone()];
        if selected != "cpu" {
            candidates.push("cpu".into());
        }
        let cpus = self.worker_cpus();
        let mut load_error = None;
        for backend in candidates {
            let profile = if backend == selected {
                plan.clone()
            } else {
                let mut cpu_only = config.clone();
                cpu_only["backend"] = json!("auto");
                cpu_only["calibration"] = json!(
                    config["calibration"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|r| r["backend"] == "cpu")
                        .cloned()
                        .collect::<Vec<_>>()
                );
                execution_plan(&cpu_only, model, &hw).map_err(bad)?
            };
            {
                let mut s = self.state.lock().unwrap();
                s.failure = None;
                s.backend = Some(backend.clone());
                s.profile = profile.clone();
            }
            let args = server_command(
                &config,
                model,
                &backend,
                &self.engine_socket(),
                Some(&profile),
                cpus.len(),
            )
            .map_err(bad)?;
            let mut env = [
                "PATH",
                "LANG",
                "LC_ALL",
                "LD_LIBRARY_PATH",
                "HOME",
                "XDG_RUNTIME_DIR",
            ]
            .iter()
            .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
            .collect::<Vec<_>>();
            for (k, v) in config["runtimes"][&backend]["environment"]
                .as_object()
                .into_iter()
                .flatten()
            {
                env.push((k.clone(), v.as_str().unwrap_or_default().to_owned()));
            }
            let threads = cpus.len().to_string();
            env.extend([
                ("OMP_NUM_THREADS".into(), threads.clone()),
                ("GGML_VK_DISABLE_F16".into(), "0".into()),
                ("OMP_WAIT_POLICY".into(), "PASSIVE".into()),
                ("MKL_NUM_THREADS".into(), threads),
                ("MESA_SHADER_CACHE_MAX_SIZE".into(), "256M".into()),
            ]);
            // The runtime directory is RAM charged to this service and wiped at
            // every stop; keep the GPU shader cache on disk or not at all.
            match std::env::var_os("STATE_DIRECTORY") {
                Some(state) => env.push((
                    "XDG_CACHE_HOME".into(),
                    Path::new(&state)
                        .join("shader-cache")
                        .to_string_lossy()
                        .into_owned(),
                )),
                None => env.push(("MESA_SHADER_CACHE_DISABLE".into(), "true".into())),
            }
            let mut child = spawn(&args, &env, cpus.clone(), None, true)
                .map_err(|_| unavailable("runtime could not start"))?;
            let diagnostic = child
                .stderr
                .take()
                .map(|stderr| tokio::spawn(process::model_load_error(stderr)));
            self.state.lock().unwrap().pid = child.id();
            engine.child = Some(child);
            self.started.notify_one();
            let deadline = Instant::now() + Duration::from_secs(120);
            while Instant::now() < deadline
                && engine
                    .child
                    .as_mut()
                    .is_some_and(|c| c.try_wait().ok().flatten().is_none())
                && self.state.lock().unwrap().pid.is_some()
            {
                if self.engine_socket().exists() && self.healthy().await {
                    self.state.lock().unwrap().loaded = Some(model.into());
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
            self.stop(engine).await;
            load_error = if let Some(diagnostic) = diagnostic {
                tokio::time::timeout(Duration::from_millis(250), diagnostic)
                    .await
                    .ok()
                    .and_then(Result::ok)
                    .flatten()
            } else {
                None
            };
        }
        let failure = self.state.lock().unwrap().failure.clone();
        Err(load_error.map_or_else(
            || inference_failure(failure, "runtime could not load model within budget"),
            Failure::ModelLoad,
        ))
    }

    fn admit(&self) -> Result<Job<'_>, ()> {
        let mut state = self.state.lock().unwrap();
        if state.jobs >= 2 {
            return Err(());
        }
        state.jobs += 1;
        Ok(Job(self))
    }

    async fn warmup(&self, mut body: Map<String, Value>) -> Result<Option<(u16, Value)>, Failure> {
        if body
            .keys()
            .any(|key| !matches!(key.as_str(), "model" | "prefix"))
        {
            return Err(bad("invalid warm-up request"));
        }
        let seed = body.remove("prefix");
        let skipped = Ok(Some((200, json!({"warmed": false}))));
        let Ok(mut engine) = self.engine.try_lock() else {
            return skipped;
        };
        let config = {
            let state = self.state.lock().unwrap();
            // Speculative work must not evict an active model or its conversation cache.
            if state.jobs != 0 {
                return skipped;
            }
            state.config.clone()
        };
        let model = check_request("/v1/models/warmup", &mut body, &config).map_err(bad)?;
        let loaded = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .loaded
            .clone();
        let same_model = loaded.as_deref() == Some(model.as_str());
        let prefix = if let Some(Value::Object(mut seed)) = seed {
            seed.insert("model".into(), json!(model));
            static_prefix("/v1/chat/completions", &seed).filter(|_| {
                seed["messages"]
                    .as_array()
                    .is_some_and(|messages| messages.len() == 2 && messages[1]["content"] == "")
            })
        } else {
            None
        };
        if same_model
            && (prefix.is_none()
                || engine
                    .cache_key
                    .as_ref()
                    .is_some_and(|key| key.prefix == prefix))
        {
            return skipped;
        }
        if !same_model
            && !available_memory_mib().is_some_and(|available| {
                available >= policy::memory_mib(&config) + IDLE_MEMORY_RESERVE_MIB
            })
        {
            return skipped;
        }
        let Ok(_job) = self.admit() else {
            return skipped;
        };
        let mut abort = Abort {
            supervisor: self,
            armed: true,
        };
        self.start(&mut engine, &model).await?;
        abort.armed = false;
        let prefix_requested = prefix.is_some();
        let mut prefetched = false;
        if let Some(prefix) = prefix {
            engine.cache_key = None;
            prefetched = tokio::time::timeout(
                Duration::from_secs(15),
                crate::calibrate::native_request(
                    &self.engine_socket(),
                    "/v1/chat/completions",
                    &prefix,
                ),
            )
            .await
            .is_ok_and(|result| result.is_ok());
            if prefetched {
                engine.cache_key = Some(CacheKey {
                    session: None,
                    prefix: Some(prefix),
                });
            }
        }
        engine.preloaded = !same_model || engine.preloaded;
        let mut result = json!({"warmed": true, "model": model});
        if prefix_requested {
            result["prefix_cached"] = json!(prefetched);
        }
        Ok(Some((200, result)))
    }

    pub async fn proxy(
        &self,
        path: &str,
        mut body: Map<String, Value>,
        writer: &mut OwnedWriteHalf,
        started: &mut bool,
    ) -> Result<Option<(u16, Value)>, Failure> {
        let Ok(_job) = self.admit() else {
            return Ok(Some((429, json!({"error": "local inference busy"}))));
        };
        let session = body.remove("conversation_id");
        let session = match session {
            None => None,
            Some(Value::String(s)) if (1..=128).contains(&s.chars().count()) => Some(s),
            Some(_) => return Err(bad("invalid conversation ID")),
        };
        let config = self.state.lock().unwrap().config.clone();
        let model = check_request(path, &mut body, &config).map_err(bad)?;
        let mut engine = self.engine.lock().await;
        // Cancelled mid-load the half-started worker is killed. Once loaded,
        // dropping the runtime connection cancels the runtime's task, so a
        // caller pressing stop does not cost the next request a model reload.
        let abort = Abort {
            supervisor: self,
            armed: true,
        };
        self.start(&mut engine, &model).await?;
        let mut abort = abort;
        abort.armed = false;
        engine.preloaded = false;
        let key = match path {
            "/tokenize" => None,
            _ => session.map(|session| CacheKey {
                session: Some(session),
                prefix: static_prefix(path, &body),
            }),
        };
        let _boost = (path != "/tokenize").then(Boost::while_processing);
        if path != "/tokenize" {
            let mut reuse = false;
            if let (Some(current), Some(previous)) = (&key, engine.cache_key.take()) {
                if previous.session == current.session {
                    reuse = true;
                } else if let Some(prefix) = &current.prefix
                    && previous.prefix.as_ref() == Some(prefix)
                {
                    // Replace the old slot's private tail before comparing it to a different chat.
                    reuse = previous.session.is_none()
                        || tokio::time::timeout(
                            Duration::from_secs(2),
                            crate::calibrate::native_request(&self.engine_socket(), path, prefix),
                        )
                        .await
                        .is_ok_and(|result| result.is_ok());
                }
            }
            body.insert("cache_prompt".into(), json!(reuse));
        }
        let relay = async {
            let mut native = UnixStream::connect(self.engine_socket())
                .await
                .map_err(|_| unavailable("runtime unavailable"))?;
            native
                .write_all(&http::post(path, "localhost", &[], &Value::Object(body)))
                .await
                .map_err(|_| unavailable("runtime unavailable"))?;
            // Relay native HTTP framing and SSE incrementally.
            const OK: &[u8] = b"HTTP/1.1 200 ";
            let mut status = Vec::with_capacity(OK.len());
            let mut total = 0usize;
            let mut chunk = vec![0u8; 65536];
            loop {
                let n = tokio::time::timeout(Duration::from_secs(600), native.read(&mut chunk))
                    .await
                    .map_err(|_| unavailable("runtime timed out"))?
                    .map_err(|_| unavailable("runtime unavailable"))?;
                if n == 0 {
                    break;
                }
                let prefix = n.min(OK.len() - status.len());
                status.extend_from_slice(&chunk[..prefix]);
                total += n;
                if total > 4 * 1024 * 1024 {
                    return Err(unavailable("runtime response exceeds limit"));
                }
                *started = true;
                writer
                    .write_all(&chunk[..n])
                    .await
                    .map_err(|_| Failure::Closed)?;
            }
            if total == 0 {
                let failure = self.state.lock().unwrap().failure.clone();
                return Err(inference_failure(failure, "runtime exited before response"));
            }
            Ok(status == OK)
        }
        .await;
        drop(_boost);
        match relay {
            Ok(accepted) => {
                if accepted && path != "/tokenize" {
                    engine.cache_key = key;
                }
                Ok(None)
            }
            Err(Failure::Closed) => Err(Failure::Closed),
            Err(failure) => {
                self.stop(&mut engine).await;
                Err(failure)
            }
        }
    }

    pub async fn transcribe(
        &self,
        body: Map<String, Value>,
        benchmark: bool,
    ) -> Result<Option<(u16, Value)>, Failure> {
        if self.state.lock().unwrap().jobs >= 2 {
            return Ok(Some((429, json!({"error": "local inference busy"}))));
        }
        let config = self.state.lock().unwrap().config.clone();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if config["models"].get(&model).and_then(|m| m.get("kind")) != Some(&json!("speech")) {
            return Err(bad("unknown speech model"));
        }
        let language = body.get("language").cloned().unwrap_or(json!("auto"));
        let language = language
            .as_str()
            .filter(|l| {
                *l == "auto"
                    || ((2..=3).contains(&l.len()) && l.bytes().all(|c| c.is_ascii_lowercase()))
            })
            .ok_or_else(|| bad("invalid speech language code"))?
            .to_owned();
        let plan = speech::Plan::new(&config, &model, &language)?;
        let audio = base64::engine::general_purpose::STANDARD
            .decode(
                body.get("wav_base64")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .map_err(|_| bad("invalid audio encoding"))?;
        if audio.len() > 2 * 1024 * 1024 {
            return Err(bad("audio too large"));
        }
        wav_shape(&audio)?;
        let Ok(_job) = self.admit() else {
            return Ok(Some((429, json!({"error": "local inference busy"}))));
        };
        let cli = plan.cli().to_owned();
        let mut engine = self.engine.lock().await;
        self.stop(&mut engine).await;
        let scratch = tempfile_dir(&self.directory)
            .map_err(|_| unavailable("speech workspace unavailable"))?;
        // Cancellation drops this future mid-wait: the guard still deletes the
        // user's audio and clears the worker identity the monitor watches.
        let _cleanup = Speech {
            supervisor: self,
            scratch: scratch.clone(),
        };
        async {
            let wav = scratch.join("audio.wav");
            std::fs::write(&wav, &audio)
                .map_err(|_| unavailable("speech workspace unavailable"))?;
            let cpus = self.worker_cpus();
            let args = plan.args(&config["models"][&model], &wav, &scratch, cpus.len());
            let env = [
                (
                    "LD_LIBRARY_PATH".to_owned(),
                    Path::new(&cli)
                        .parent()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                ),
                (
                    "PATH".to_owned(),
                    std::env::var("PATH").unwrap_or_else(|_| "/usr/bin".into()),
                ),
                ("OMP_NUM_THREADS".to_owned(), cpus.len().to_string()),
            ];
            let mut child = spawn(&args, &env, cpus, plan.output(&scratch).as_deref(), false)
                .map_err(|_| unavailable("speech runtime could not start"))?;
            {
                let mut s = self.state.lock().unwrap();
                s.pid = child.id();
                s.loaded = Some(model.clone());
                s.backend = Some("cpu".into());
            }
            let started = Instant::now();
            let wait = tokio::time::timeout(Duration::from_secs(120), child.wait());
            let (status, memory) = if benchmark {
                benchmark::measure_memory(self, wait).await
            } else {
                (wait.await, 0.0)
            };
            {
                let mut s = self.state.lock().unwrap();
                s.pid = None;
                s.loaded = None;
            }
            let status = match status {
                Ok(Ok(status)) => status,
                _ => {
                    let _ = child.kill().await;
                    return Err(unavailable("speech inference timed out"));
                }
            };
            let failure = self.state.lock().unwrap().failure.clone();
            let text = status
                .success()
                .then(|| plan.transcript(&scratch))
                .flatten()
                .ok_or_else(|| {
                    unavailable(failure.unwrap_or_else(|| "speech inference failed".into()))
                })?;
            if benchmark {
                return Ok(Some((
                    200,
                    json!({
                        "audio_seconds": (audio.len() - 44) as f64 / 32_000.0,
                        "elapsed_seconds": started.elapsed().as_secs_f64(),
                        "memory_mib": memory,
                    }),
                )));
            }
            Ok(Some((200, json!({"text": text.trim(), "model": model}))))
        }
        .await
    }

    /// Enforce the RAM budget and unload an idle worker.
    pub async fn monitor(self: Arc<Self>) {
        loop {
            let pid = self.state.lock().unwrap().pid;
            if pid.is_some() {
                tokio::time::sleep(Duration::from_secs(1)).await;
            } else {
                tokio::select! {
                    () = tokio::time::sleep(Duration::from_secs(5)) => {}
                    () = self.started.notified() => continue,
                }
            }
            let Some(pid) = self.state.lock().unwrap().pid else {
                // A worker killed by an aborted request is reaped once the engine is free.
                if let Ok(mut engine) = self.engine.try_lock()
                    && engine.child.is_some()
                {
                    self.stop(&mut engine).await;
                }
                continue;
            };
            let (rss, driver) = (rss_mib(pid), driver_ram_mib(pid));
            let (over, idle_seconds) = {
                let mut s = self.state.lock().unwrap();
                s.peak_rss_mib = s.peak_rss_mib.max(rss);
                s.peak_driver_ram_mib = s.peak_driver_ram_mib.max(driver);
                // Summed conservatively: some driver buffers may also appear in RSS.
                let over = rss + driver > policy::memory_mib(&s.config);
                if over {
                    s.failure = Some(MEMORY_BUDGET_EXCEEDED.into());
                }
                (over, policy::get_u64(&s.config, "idle_seconds", 60))
            };
            if over {
                self.kill_worker();
                // Reap the killed worker and remove its socket when no job holds the engine.
                if let Ok(mut engine) = self.engine.try_lock() {
                    self.stop(&mut engine).await;
                }
            } else if let Ok(mut engine) = self.engine.try_lock() {
                let idle_limit = if engine.preloaded {
                    idle_seconds.min(PRELOAD_IDLE_SECONDS)
                } else {
                    idle_seconds
                };
                let should_stop = {
                    let state = self.state.lock().unwrap();
                    state.jobs == 0
                        && (state.last_used.elapsed() > Duration::from_secs(idle_limit)
                            || available_memory_mib()
                                .is_some_and(|mib| mib < IDLE_MEMORY_RESERVE_MIB))
                };
                if should_stop {
                    self.stop(&mut engine).await;
                }
            }
        }
    }

    async fn configure(&self, body: &Map<String, Value>) -> Result<Option<(u16, Value)>, Failure> {
        let busy = || {
            Ok(Some((
                409,
                json!({"error": "inference jobs active; retry after completion"}),
            )))
        };
        if self.state.lock().unwrap().jobs > 0 {
            return busy();
        }
        let mut engine = self.engine.lock().await;
        if self.state.lock().unwrap().jobs > 0 {
            return busy();
        }
        let (snapshot, value) = {
            let mut preferences = self.preferences.lock().unwrap();
            let snapshot = preferences.commit(body).map_err(|e| match e {
                preferences::Error::Conflict(m) => Failure::Conflict(m),
                preferences::Error::Invalid(m) => Failure::Bad(m),
            })?;
            (snapshot, preferences.value.clone())
        };
        {
            let mut s = self.state.lock().unwrap();
            let mut config = self.base.clone();
            for (k, v) in &value {
                config[k] = v.clone();
            }
            s.hw["worker_cpus"] = json!(cpu_budget(config["cpu_threads"].as_u64()));
            s.config = config;
        }
        self.stop(&mut engine).await;
        let mut s = self.state.lock().unwrap();
        s.backend = None;
        s.profile = Value::Null;
        s.failure = None;
        Ok(Some((200, snapshot)))
    }

    fn status(&self) -> Value {
        let s = self.state.lock().unwrap();
        json!({"ready": true, "loaded_model": s.loaded, "backend": s.backend, "installed_runtimes": s.config["runtimes"].as_object().map(|r| r.keys().cloned().collect::<Vec<_>>()).unwrap_or_default(),
               "profile": s.profile, "jobs": s.jobs, "worker_cpus": s.hw["worker_cpus"], "peak_worker_rss_mib": s.peak_rss_mib, "peak_driver_ram_mib": s.peak_driver_ram_mib, "failure": s.failure})
    }

    pub async fn serve(self: Arc<Self>, listener: UnixListener) {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                // Back off on descriptor exhaustion instead of spinning.
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            };
            let supervisor = self.clone();
            tokio::spawn(async move { supervisor.client(stream).await });
        }
    }

    async fn client(&self, stream: UnixStream) {
        let clients = {
            let mut s = self.state.lock().unwrap();
            s.clients += 1;
            s.clients
        };
        let uid = stream.peer_cred().map(|c| c.uid()).ok();
        let (mut reader, mut writer) = stream.into_split();
        let mut started = false;
        let result = if clients > 16 {
            Ok(Some((429, json!({"error": "too many connections"}))))
        } else if uid != Some(self.uid) {
            Ok(Some((403, json!({"error": "peer UID refused"}))))
        } else {
            self.handle(&mut reader, &mut writer, &mut started).await
        };
        let response = match result {
            Ok(response) => response,
            Err(Failure::Closed) => None,
            Err(Failure::Conflict(m)) => Some((409, json!({"error": m}))),
            Err(Failure::Bad(m)) => Some((400, json!({"error": m}))),
            Err(Failure::Unavailable(m)) => Some((503, json!({"error": m}))),
            Err(Failure::ModelLoad(code)) => Some((503, json!({"error": {"code":code}}))),
        };
        if let Some((status, body)) = response.filter(|_| !started) {
            let _ = http::reply(&mut writer, status, &body).await;
        }
        let _ = writer.shutdown().await;
        self.state.lock().unwrap().clients -= 1;
    }

    async fn handle(
        &self,
        reader: &mut tokio::net::unix::OwnedReadHalf,
        writer: &mut OwnedWriteHalf,
        started: &mut bool,
    ) -> Result<Option<(u16, Value)>, Failure> {
        let request = http::read_head(reader).await.map_err(|e| {
            if e == "connection closed" {
                Failure::Closed
            } else {
                bad(e)
            }
        })?;
        if request.headers.contains_key("transfer-encoding") {
            return Err(bad("chunked input unsupported"));
        }
        let path = request.path.as_str();
        if request.method == "GET" {
            return match path {
                "/v1/preferences" => Ok(Some((200, self.preferences.lock().unwrap().snapshot()))),
                "/health" | "/status" => Ok(Some((200, self.status()))),
                "/v1/models" => {
                    let config = self
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .config
                        .clone();
                    // `default` marks the text model the service answers with
                    // and the speech model the user chose for dictation.
                    let data = config["models"].as_object().into_iter().flatten().map(|(id, m)| {
                        let mut entry = json!({"id": id, "object": "model", "kind": m.get("kind").cloned().unwrap_or_else(|| json!("llm"))});
                        if entry["kind"] == "llm" {
                            entry["context"] = json!(policy::context_for_model(&config, id));
                        }
                        let title = match m.get("title").and_then(Value::as_str) {
                            // Existing registries store the artifact stem for the bundled Whisper model.
                            Some("ggml-base-q5_1")
                                if id == "whisper-base"
                                    && entry["kind"] == "speech"
                                    && m["architecture"] == "whisper" =>
                            {
                                Some("Whisper base")
                            }
                            title => title,
                        };
                        if let Some(title) = title {
                            entry["title"] = json!(title);
                        }
                        if let Some(architecture) = m.get("architecture") {
                            entry["architecture"] = architecture.clone();
                        }
                        if config["default_model"] == *id || config["default_speech_model"] == *id {
                            entry["default"] = json!(true);
                        }
                        entry
                    }).collect::<Vec<_>>();
                    Ok(Some((200, json!({"object": "list", "data": data}))))
                }
                _ => Err(bad("unsupported request")),
            };
        }
        if request.method != "POST" {
            return Err(bad("unsupported request"));
        }
        let limit = if matches!(path, "/v1/preferences" | "/v1/models/benchmark") {
            4096
        } else if path == "/v1/models/warmup" {
            8192
        } else {
            3 * 1024 * 1024
        };
        let body = http::read_body(reader, &request, limit)
            .await
            .map_err(|e| {
                if e == "connection closed" {
                    Failure::Closed
                } else {
                    bad(e)
                }
            })?;
        let body = body.as_object().cloned().unwrap_or_default();
        if path == "/v1/preferences" {
            return self.configure(&body).await;
        }
        // The caller disconnecting cancels the job; Abort and Job guards clean up.
        let operation = async {
            match path {
                "/v1/audio/transcriptions" => self.transcribe(body, false).await,
                "/v1/models/benchmark" => self.benchmark(body).await,
                "/v1/models/warmup" => self.warmup(body).await,
                _ => self.proxy(path, body, writer, started).await,
            }
        };
        tokio::select! {
            result = tokio::time::timeout(Duration::from_secs(600), operation) => result.map_err(|_| unavailable("request timed out"))?,
            () = client_gone(reader) => Err(Failure::Closed),
        }
    }
}

/// Resolves when the caller has really gone. A half-close (`SHUT_WR`) or
/// trailing bytes after the body are not cancellation; after end of input
/// the socket is polled for a full hangup instead.
#[allow(unsafe_code, reason = "Linux syscall ABI (poll).")]
async fn client_gone(reader: &mut tokio::net::unix::OwnedReadHalf) {
    use std::os::fd::AsRawFd;
    let mut discard = [0u8; 4096];
    let mut extra = 0usize;
    loop {
        match reader.read(&mut discard).await {
            Ok(0) => break,
            Ok(n) if extra + n <= 65536 => extra += n,
            _ => return,
        }
    }
    let fd = reader.as_ref().as_raw_fd();
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let mut poll = libc::pollfd {
            fd,
            events: 0,
            revents: 0,
        };
        if unsafe { libc::poll(&mut poll, 1, 0) } != 0
            && poll.revents & (libc::POLLHUP | libc::POLLERR) != 0
        {
            return;
        }
    }
}

/// 16 kHz mono 16-bit PCM WAV, at most 60 seconds.
pub fn wav_shape(audio: &[u8]) -> Result<(), Failure> {
    let refuse = || bad("16kHz mono PCM WAV, at most 60s required");
    if audio.len() < 12 || &audio[..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return Err(refuse());
    }
    let container = u32::from_le_bytes(audio[4..8].try_into().unwrap()) as usize;
    if container.checked_add(8) != Some(audio.len()) {
        return Err(refuse());
    }
    let (mut offset, mut format, mut frames) = (12usize, None, None);
    while offset < audio.len() {
        let header = audio
            .get(offset..offset.checked_add(8).ok_or_else(refuse)?)
            .ok_or_else(refuse)?;
        let id = &header[..4];
        let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let end = start.checked_add(size).ok_or_else(refuse)?;
        let data = audio.get(start..end).ok_or_else(refuse)?;
        if id == b"fmt " {
            if format.is_some() || data.len() < 16 {
                return Err(refuse());
            }
            let u16_at = |i: usize| u16::from_le_bytes([data[i], data[i + 1]]);
            let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
            format = Some((u16_at(0), u16_at(2), u32_at(4), u16_at(14)));
            if u32_at(8) != 32_000 || u16_at(12) != 2 {
                return Err(refuse());
            }
        } else if id == b"data" {
            if frames.is_some() || size == 0 || size % 2 != 0 {
                return Err(refuse());
            }
            frames = Some(size / 2);
        }
        offset = end.checked_add(size & 1).ok_or_else(refuse)?;
        if offset > audio.len() {
            return Err(refuse());
        }
    }
    match (format, frames) {
        (Some((1, 1, 16000, 16)), Some(frames)) if frames <= 16000 * 60 => Ok(()),
        _ => Err(refuse()),
    }
}

fn tempfile_dir(parent: &Path) -> std::io::Result<PathBuf> {
    let path = parent.join(format!(
        "speech-{}",
        enzimos_platform::files::random_hex(8).map_err(|_| std::io::Error::other("random"))?
    ));
    std::fs::DirBuilder::new().mode_private().create(&path)?;
    Ok(path)
}

trait Private {
    fn mode_private(&mut self) -> &mut Self;
}
impl Private for std::fs::DirBuilder {
    fn mode_private(&mut self) -> &mut Self {
        std::os::unix::fs::DirBuilderExt::mode(self, 0o700)
    }
}

/// Bind a new socket inside a private owned directory.
#[allow(unsafe_code, reason = "Linux syscall ABI (getuid, umask).")]
pub fn bind(path: &Path) -> Result<UnixListener, String> {
    let parent = path.parent().ok_or("socket directory required")?;
    let _ = std::fs::DirBuilder::new()
        .recursive(true)
        .mode_private()
        .create(parent);
    let m =
        std::fs::symlink_metadata(parent).map_err(|_| "private owned socket directory required")?;
    use std::os::unix::fs::MetadataExt;
    if !m.is_dir() || m.uid() != unsafe { libc::getuid() } || m.mode() & 0o077 != 0 {
        return Err("private owned socket directory required".into());
    }
    if path.exists() {
        return Err("socket already exists; refusing replacement".into());
    }
    unsafe { libc::umask(0o077) };
    let listener = UnixListener::bind(path).map_err(|_| "socket unavailable")?;
    std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
        .map_err(|_| "socket unavailable")?;
    Ok(listener)
}
