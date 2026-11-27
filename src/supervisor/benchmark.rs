// SPDX-License-Identifier: MIT
use super::{Abort, Failure, Supervisor, bad, driver_ram_mib, rss_mib, unavailable};
use base64::Engine as _;
use serde_json::{Map, Value, json};
use std::{future::Future, time::Duration};

const PROMPT: &str = "Continue in plain English: A walk through the park on a sunny day";

// Synthetic English speech generated with eSpeak NG and resampled to 16 kHz mono PCM.
const AUDIO: &[u8] = include_bytes!("benchmark.wav");

impl Supervisor {
    pub(super) async fn benchmark(
        &self,
        body: Map<String, Value>,
    ) -> Result<Option<(u16, Value)>, Failure> {
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .filter(|_| body.len() == 1)
            .ok_or_else(|| bad("Benchmark requires a model ID."))?;
        let kind = {
            let state = self.state.lock().unwrap();
            if state.config["models"].get(model).is_none() {
                return Err(bad("Unknown model."));
            }
            crate::policy::kind(&state.config, model) == "speech"
        };
        if kind {
            let sample = base64::engine::general_purpose::STANDARD.encode(AUDIO);
            return self
                .transcribe(
                    Map::from_iter([
                        ("model".into(), json!(model)),
                        ("language".into(), json!("en")),
                        ("wav_base64".into(), json!(sample)),
                    ]),
                    true,
                )
                .await;
        }

        let mut engine = self
            .engine
            .try_lock()
            .map_err(|_| unavailable("Local AI is busy."))?;
        let _job = self
            .admit()
            .map_err(|()| unavailable("Local AI is busy."))?;
        let mut abort = Abort {
            supervisor: self,
            armed: true,
        };
        self.start(&mut engine, model).await?;
        abort.armed = false;
        engine.cache_key = None;
        engine.preloaded = false;
        let request = json!({"prompt": PROMPT, "n_predict": 48, "temperature": 0,
            "cache_prompt": false, "stream": false});
        let socket = self.engine_socket();
        let (reply, memory) = measure_memory(
            self,
            crate::calibrate::native_request(&socket, "/completion", &request),
        )
        .await;
        let reply = reply.map_err(|_| unavailable("The model could not finish the benchmark."))?;
        let timings = &reply["timings"];
        let (Some(first_ms), Some(rate)) = (
            timings["prompt_ms"].as_f64(),
            timings["predicted_per_second"]
                .as_f64()
                .filter(|rate| *rate > 0.0),
        ) else {
            return Err(unavailable("The runtime did not report benchmark timings."));
        };
        Ok(Some((
            200,
            json!({
                "first_token_ms": first_ms,
                "tokens_per_second": rate,
                "memory_mib": memory,
            }),
        )))
    }
}

pub(super) async fn measure_memory<T>(
    supervisor: &Supervisor,
    operation: impl Future<Output = T>,
) -> (T, f64) {
    let pid = supervisor.state.lock().unwrap().pid;
    let sample = || pid.map_or(0.0, |pid| rss_mib(pid) + driver_ram_mib(pid));
    let mut peak = sample();
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    tokio::pin!(operation);
    loop {
        tokio::select! {
            result = &mut operation => return (result, peak.max(sample())),
            _ = interval.tick() => peak = peak.max(sample()),
        }
    }
}
