// SPDX-License-Identifier: MIT
//! enzimos-inference-probe --config FILE: hardware and runtime enumeration only.
use std::process::ExitCode;

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [flag, path] = args.as_slice() else {
        eprintln!("usage: enzimos-inference-probe --config FILE");
        return ExitCode::from(2);
    };
    if flag != "--config" {
        return ExitCode::from(2);
    }
    let Ok(Ok(config)) =
        std::fs::read(path).map(|raw| serde_json::from_slice::<serde_json::Value>(&raw))
    else {
        return ExitCode::FAILURE;
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&enzimos_inference_service::calibrate::probe(&config))
            .unwrap_or_default()
    );
    ExitCode::SUCCESS
}
