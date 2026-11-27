// SPDX-License-Identifier: MIT
//! The supervisor over its real socket, with a fake native runtime.
use enzimos_inference_service::supervisor::{Supervisor, bind};
use serde_json::{Value, json};
use std::{
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
};

fn fake_engine() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    exe.parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/fake_engine")
}

fn config(dir: &Path) -> Value {
    let model = dir.join("model.gguf");
    std::fs::write(&model, b"weights").unwrap();
    let sha = enzimos_inference_service::policy::file_sha256(&model).unwrap();
    json!({"schema": 1, "default_model": "m", "idle_seconds": 3600,
           "runtimes": {"cpu": {"server": fake_engine(), "environment": {"FAKE_LOG": dir.join("requests.log")}}},
           "models": {"m": {"path": model, "sha256": sha}}})
}

struct Running {
    socket: PathBuf,
    supervisor: Arc<Supervisor>,
    _dir: tempfile::TempDir,
}

async fn start(edit: impl FnOnce(&mut Value, &Path)) -> Running {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut config = config(dir.path());
    edit(&mut config, dir.path());
    let socket = dir.path().join("run/service.sock");
    let listener = bind(&socket).unwrap();
    let supervisor = Supervisor::new(config, socket.parent().unwrap(), None).unwrap();
    tokio::spawn(supervisor.clone().serve(listener));
    tokio::spawn(supervisor.clone().monitor());
    Running {
        socket,
        supervisor,
        _dir: dir,
    }
}

async fn send(socket: &Path, raw: String) -> String {
    let mut stream = UnixStream::connect(socket).await.unwrap();
    stream.write_all(raw.as_bytes()).await.unwrap();
    let mut out = String::new();
    stream.read_to_string(&mut out).await.unwrap();
    out
}
fn post(path: &str, body: &Value) -> String {
    let data = body.to_string();
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{data}",
        data.len()
    )
}
async fn status(socket: &Path) -> Value {
    let out = send(socket, "GET /status HTTP/1.1\r\n\r\n".into()).await;
    serde_json::from_str(out.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}
fn logged(dir: &Path) -> Vec<Value> {
    std::fs::read_to_string(dir.join("requests.log"))
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn health_does_not_load_and_streaming_applies_model_policy() {
    let run = start(|_, _| {}).await;
    assert!(status(&run.socket).await["loaded_model"].is_null());
    let out = send(
        &run.socket,
        post(
            "/v1/chat/completions",
            &json!({"messages": [{"role": "user", "content": "hi"}], "stream": true}),
        ),
    )
    .await;
    assert!(
        out.starts_with("HTTP/1.1 200 OK") && out.contains("\"two\""),
        "{out}"
    );
    let body = &logged(run._dir.path())[0];
    assert_eq!(
        (
            body["max_tokens"].as_u64(),
            body["cache_prompt"].as_bool(),
            body["model"].as_str()
        ),
        (Some(256), Some(false), Some("m"))
    );
    assert_eq!(status(&run.socket).await["loaded_model"], "m");
    let prefetch = send(
        &run.socket,
        post("/v1/models/warmup", &json!({"model": "m"})),
    )
    .await;
    assert!(prefetch.contains("\"warmed\":false"), "{prefetch}");
    assert_eq!(logged(run._dir.path()).len(), 1);
    let refused = send(
        &run.socket,
        post(
            "/v1/chat/completions",
            &json!({"messages": [], "max_tokens": 4096}),
        ),
    )
    .await;
    assert!(refused.starts_with("HTTP/1.1 400") && refused.contains("output limit"));
    let dup = send(
        &run.socket,
        "GET /status HTTP/1.1\r\nA: 1\r\na: 2\r\n\r\n".into(),
    )
    .await;
    assert!(dup.starts_with("HTTP/1.1 400"));
}

#[tokio::test(flavor = "current_thread")]
async fn prompt_cache_shares_only_the_static_prefix_between_conversations() {
    let run = start(|_, _| {}).await;
    for (conversation, system, prompt) in [
        (Some("a"), "shared guidance", "private A"),
        (Some("a"), "shared guidance", "follow-up A"),
        (Some("b"), "shared guidance", "private B"),
        (Some("c"), "different guidance", "rejected by native"),
        (Some("c"), "different guidance", "private C"),
        (None, "different guidance", "unspecified conversation"),
    ] {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        });
        if let Some(id) = conversation {
            body["conversation_id"] = json!(id);
        }
        send(&run.socket, post("/v1/chat/completions", &body)).await;
    }
    let requests = logged(run._dir.path());
    let flags = requests
        .iter()
        .map(|b| b["cache_prompt"].as_bool().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(flags, [false, true, true, true, false, false, false]);
    let prefix = &requests[2];
    assert_eq!(prefix["max_tokens"], 1);
    assert_eq!(prefix["add_generation_prompt"], false);
    assert_eq!(prefix["messages"][0], requests[1]["messages"][0]);
    assert_eq!(
        prefix["messages"][1],
        json!({"role": "user", "content": ""})
    );
    assert_eq!(prefix["tools"], requests[1]["tools"]);
    assert_eq!(requests[3]["messages"][1]["content"], "private B");
    assert!(
        requests
            .iter()
            .all(|body| body.get("conversation_id").is_none())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn disconnect_releases_the_job_and_keeps_the_model_loaded() {
    let run =
        start(|c, _| c["runtimes"]["cpu"]["environment"]["FAKE_DELAY_MS"] = json!("3000")).await;
    let mut stream = UnixStream::connect(&run.socket).await.unwrap();
    stream
        .write_all(post("/v1/chat/completions", &json!({"messages": []})).as_bytes())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert_eq!(status(&run.socket).await["jobs"], 1);
    drop(stream);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let s = status(&run.socket).await;
    assert_eq!(
        (s["jobs"].as_u64(), s["loaded_model"].as_str()),
        (Some(0), Some("m")),
        "{s}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn half_closed_caller_still_gets_the_reply() {
    let run = start(|_, _| {}).await;
    let mut stream = UnixStream::connect(&run.socket).await.unwrap();
    stream
        .write_all(post("/v1/chat/completions", &json!({"messages": []})).as_bytes())
        .await
        .unwrap();
    stream.shutdown().await.unwrap();
    let mut out = String::new();
    stream.read_to_string(&mut out).await.unwrap();
    assert!(out.contains("\"two\""), "{out}");
}

#[tokio::test(flavor = "current_thread")]
async fn at_most_two_jobs_are_admitted() {
    let run =
        start(|c, _| c["runtimes"]["cpu"]["environment"]["FAKE_DELAY_MS"] = json!("1500")).await;
    let socket = run.socket.clone();
    let slow = (0..2)
        .map(|_| tokio::spawn(send_owned(socket.clone())))
        .collect::<Vec<_>>();
    tokio::time::sleep(Duration::from_millis(600)).await;
    let third = send(
        &run.socket,
        post("/v1/chat/completions", &json!({"messages": []})),
    )
    .await;
    assert!(third.starts_with("HTTP/1.1 429"), "{third}");
    for task in slow {
        assert!(task.await.unwrap().starts_with("HTTP/1.1 200"));
    }
}
async fn send_owned(socket: PathBuf) -> String {
    send(
        &socket,
        post("/v1/chat/completions", &json!({"messages": []})),
    )
    .await
}

#[tokio::test(flavor = "current_thread")]
async fn gpu_startup_failure_falls_back_to_cpu() {
    let run = start(|c, _| {
        c["runtimes"]["vulkan"] = json!({"server": fake_engine(), "device": "Vulkan0", "environment": {"FAKE_MODE": "fail"}});
        c["backend"] = json!("vulkan");
    })
    .await;
    let out = send(
        &run.socket,
        post("/v1/chat/completions", &json!({"messages": []})),
    )
    .await;
    assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    assert_eq!(status(&run.socket).await["backend"], "cpu");
}

#[tokio::test(flavor = "current_thread")]
async fn idle_worker_is_unloaded() {
    let run = start(|c, _| c["idle_seconds"] = json!(1)).await;
    send(
        &run.socket,
        post("/v1/chat/completions", &json!({"messages": []})),
    )
    .await;
    assert_eq!(status(&run.socket).await["loaded_model"], "m");
    tokio::time::sleep(Duration::from_secs(4)).await;
    let s = status(&run.socket).await;
    assert!(
        s["loaded_model"].is_null(),
        "{s} pid={:?}",
        run.supervisor.state.lock().unwrap().pid
    );
    let _ = &run.supervisor;
}

fn wav(seconds: u32, rate: u32, channels: u16) -> Vec<u8> {
    let data = vec![0u8; (seconds * rate * u32::from(channels) * 2) as usize];
    let mut out = b"RIFF".to_vec();
    out.extend((36 + data.len() as u32).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(channels.to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * u32::from(channels) * 2).to_le_bytes());
    out.extend((channels * 2).to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    out
}

#[tokio::test(flavor = "current_thread")]
async fn local_speech_is_bounded_and_transcribed() {
    use base64::Engine as _;
    let run = start(|c, dir| {
        // A scripted whisper-cli: writes the JSON transcript named by -of.
        let cli = dir.join("whisper-cli");
        std::fs::write(&cli, "#!/bin/sh\nwhile [ $# -gt 0 ]; do [ \"$1\" = -of ] && out=$2; shift; done\nprintf '{\"transcription\":[{\"text\":\" hello\"},{\"text\":\" world \"}]}' > \"$out.json\"\n").unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        let model = dir.join("speech.bin");
        std::fs::write(&model, b"speech").unwrap();
        c["models"]["voice"] = json!({"path": model, "kind": "speech", "sha256": enzimos_inference_service::policy::file_sha256(&model).unwrap()});
        c["speech"] = json!({"cli": cli});
    })
    .await;
    let encode = |a: &[u8]| base64::engine::general_purpose::STANDARD.encode(a);
    let ok = send(
        &run.socket,
        post(
            "/v1/audio/transcriptions",
            &json!({"model": "voice", "wav_base64": encode(&wav(1, 16000, 1))}),
        ),
    )
    .await;
    assert!(
        ok.starts_with("HTTP/1.1 200") && ok.contains("\"text\":\"hello world\""),
        "{ok}"
    );
    let valid = wav(1, 16000, 1);
    let mut padded = valid.clone();
    padded.extend(b"JUNK\x01\x00\x00\x00x\x00");
    let size = (padded.len() - 8) as u32;
    padded[4..8].copy_from_slice(&size.to_le_bytes());
    assert!(enzimos_inference_service::supervisor::wav_shape(&padded).is_ok());
    let mut truncated = valid[..44].to_vec();
    truncated[4..8].copy_from_slice(&36u32.to_le_bytes());
    let mut misaligned = valid.clone();
    misaligned[40..44].copy_from_slice(&31_999u32.to_le_bytes());
    for (audio, model) in [
        (truncated, "voice"),
        (misaligned, "voice"),
        (wav(1, 44100, 1), "voice"),
        (wav(1, 16000, 2), "voice"),
        (wav(61, 16000, 1), "voice"),
        (wav(1, 16000, 1), "m"),
    ] {
        let refused = send(
            &run.socket,
            post(
                "/v1/audio/transcriptions",
                &json!({"model": model, "wav_base64": encode(&audio)}),
            ),
        )
        .await;
        assert!(refused.starts_with("HTTP/1.1 400"), "{refused}");
    }
}

/// A Moonshine folder runs on sherpa-onnx, whose result line on standard
/// output is the transcript; it is asked for English only, and the default
/// speech model is marked in the model list.
#[tokio::test(flavor = "current_thread")]
async fn sherpa_speech_is_read_from_its_output_line() {
    use base64::Engine as _;
    let run = start(|c, dir| {
        // A scripted sherpa-onnx-offline: logs, then the JSON result line,
        // only when it was given Moonshine's parts and the audio.
        let cli = dir.join("sherpa-onnx-offline");
        std::fs::write(&cli, "#!/bin/sh\necho \"$0 $*\" >&2\ncase \"$*\" in *--moonshine-cached-decoder=*cached_decode.int8.onnx*--num-threads=*audio.wav) ;; *) exit 3;; esac\necho '{\"lang\": \"\", \"text\": \" After early nightfall.\", \"tokens\": []}'\n").unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        let folder = dir.join("moonshine");
        std::fs::create_dir(&folder).unwrap();
        let mut files = serde_json::Map::new();
        for (role, name) in [("preprocessor", "preprocess.onnx"), ("encoder", "encode.int8.onnx"), ("uncached_decoder", "uncached_decode.int8.onnx"), ("cached_decoder", "cached_decode.int8.onnx"), ("tokens", "tokens.txt")] {
            std::fs::write(folder.join(name), name).unwrap();
            files.insert(role.into(), json!({"name": name, "sha256": enzimos_inference_service::policy::file_sha256(&folder.join(name)).unwrap()}));
        }
        let manifest = enzimos_inference_service::models::folder::manifest(&files);
        c["models"]["moonshine"] = json!({"path": folder, "kind": "speech", "architecture": "moonshine", "files": files, "sha256": manifest});
        c["speech"] = json!({"onnx": cli});
        c["default_speech_model"] = json!("moonshine");
    })
    .await;
    let models = send(&run.socket, "GET /v1/models HTTP/1.1\r\n\r\n".into()).await;
    assert!(
        models.contains(r#"{"architecture":"moonshine","default":true,"id":"moonshine","kind":"speech","object":"model"}"#),
        "{models}"
    );
    let audio = base64::engine::general_purpose::STANDARD.encode(wav(1, 16000, 1));
    let ok = send(
        &run.socket,
        post(
            "/v1/audio/transcriptions",
            &json!({"model": "moonshine", "wav_base64": audio}),
        ),
    )
    .await;
    assert!(
        ok.starts_with("HTTP/1.1 200") && ok.contains(r#""text":"After early nightfall.""#),
        "{ok}"
    );
    let german = send(
        &run.socket,
        post(
            "/v1/audio/transcriptions",
            &json!({"model": "moonshine", "language": "de", "wav_base64": audio}),
        ),
    )
    .await;
    assert!(
        german.starts_with("HTTP/1.1 400") && german.contains("English only"),
        "{german}"
    );
    assert_eq!(
        std::fs::read_dir(run.socket.parent().unwrap())
            .unwrap()
            .filter(|e| e
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("speech-"))
            .count(),
        0,
        "the audio and output are removed"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn stale_stored_preference_falls_back_per_field() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let prefs = dir.path().join("preferences.json");
    std::fs::write(&prefs, br#"{"backend":"vulkan","context":8192}"#).unwrap();
    std::fs::set_permissions(&prefs, std::fs::Permissions::from_mode(0o600)).unwrap();
    let socket = dir.path().join("run/service.sock");
    let listener = bind(&socket).unwrap();
    let supervisor =
        Supervisor::new(config(dir.path()), socket.parent().unwrap(), Some(&prefs)).unwrap();
    tokio::spawn(supervisor.clone().serve(listener));
    let out = send(&socket, "GET /v1/preferences HTTP/1.1\r\n\r\n".into()).await;
    let body: Value = serde_json::from_str(out.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        (
            body["preferences"]["backend"].as_str(),
            body["preferences"]["context"].as_u64()
        ),
        (Some("auto"), Some(8192)),
        "{body}"
    );
}
