// SPDX-License-Identifier: MIT
//! Opt-in CPU calibration: processes a private 4096-token test and saves no generated text.
use enzimos_inference_service::calibrate;
use serde_json::json;
use std::{os::unix::process::CommandExt, path::PathBuf, process::ExitCode};

const USAGE: &str = "usage: enzimos-inference-calibrate --config FILE --model ALIAS --receipt FILE [--output-config FILE] --run";

#[allow(unsafe_code, reason = "Linux syscall ABI (setpriority).")]
fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (mut config, mut model, mut receipt, mut output, mut run, mut managed) =
        (None, None, None, None, false, false);
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--config" => config = rest.next().map(PathBuf::from),
            "--model" => model = rest.next().cloned(),
            "--receipt" => receipt = rest.next().map(PathBuf::from),
            "--output-config" => output = rest.next().map(PathBuf::from),
            "--run" => run = true,
            "--managed" => managed = true,
            _ => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(config), Some(model), Some(receipt)) = (config, model, receipt) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    if !run {
        eprintln!("--run is required to load a model");
        return ExitCode::from(2);
    }
    if !managed {
        // Re-run inside a bounded transient scope.
        let Ok(exe) = std::env::current_exe() else {
            return ExitCode::FAILURE;
        };
        let unit = format!(
            "--unit=enzimos-calibrate-{}",
            enzimos_platform::files::random_hex(16).unwrap_or_default()
        );
        let error = std::process::Command::new("/usr/bin/systemd-run")
            .args([
                "--user",
                "--scope",
                "--collect",
                "--quiet",
                &unit,
                "--property=MemoryMax=2G",
                "--property=MemorySwapMax=0",
                "--property=TasksMax=64",
                "--property=CPUQuota=100%",
                "--",
            ])
            .arg(exe)
            .args(&args)
            .arg("--managed")
            .exec();
        eprintln!("calibration scope unavailable: {error}");
        return ExitCode::FAILURE;
    }
    let result = (|| {
        let group = calibrate::resource_group()?;
        unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 19) };
        let value = calibrate::read_private(&config)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| "runtime unavailable")?;
        let row = runtime.block_on(calibrate::calibrate(&value, &model, &group))?;
        calibrate::create_private(
            &receipt,
            &json!({"schema": 1, "calibration": row, "configuration_changed": false, "services_enabled": false}),
        )?;
        if let Some(output) = &output {
            // A new opt-in configuration, never an overwrite of running settings.
            let mut updated = value.clone();
            let mut rows = value["calibration"].as_array().cloned().unwrap_or_default();
            rows.retain(|r| !(r["backend"] == "cpu" && r["model_sha256"] == row["model_sha256"]));
            rows.push(row);
            updated["calibration"] = json!(rows);
            calibrate::create_private(output, &updated)?;
        }
        Ok::<_, String>(())
    })();
    match result {
        Ok(()) => {
            println!(
                "{}",
                json!({"status": "CPU-calibration-complete", "receipt": receipt, "output_config": output, "services_enabled": false})
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("calibration refused: {error}");
            ExitCode::FAILURE
        }
    }
}
