// SPDX-License-Identifier: MIT
//! enzimos-inference-models list | catalog | add PATH [--id ID] [--sha256 HEX] [--default] [--system]
//!     | fetch ID [--default] | default ID | remove ID
//!
//! Manages the account's model registry (`~/.config/enzimos/inference.json`):
//! a model is added from a file or folder the user chose (a MiniCPM or other
//! GGUF, a whisper.cpp speech model, or a sherpa-onnx Parakeet TDT or Moonshine
//! folder), checked, kept read-only under `~/.local/share/enzimos/models` and
//! registered by its digest. `--system` registers a model the system installed
//! (root-owned, such as /usr/share/enzimos/models) where it is, without a copy.
//! Only `fetch` downloads: a speech model from the built-in `catalog`, checked
//! against the digest pinned there. `--default` and `default` choose the text
//! model the service answers with, or the speech model dictation uses. The
//! running service is restarted to pick the change up.
use enzimos_inference_service::models;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: enzimos-inference-models setup | list | catalog | add PATH [--id ID] [--sha256 HEX] [--default] [--system] | fetch ID [--default] | default ID | remove ID";

fn home(var: &str, fallback: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(fallback)))
}

/// Downloads `url` to `file` over HTTPS only, with curl.
fn download(url: &str, file: &std::path::Path) -> Result<(), String> {
    let status = std::process::Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--max-filesize",
            "2000000000",
            "--output",
        ])
        .arg(file)
        .arg(url)
        .stdin(std::process::Stdio::null())
        .status()
        .map_err(|error| format!("curl: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("the download failed".to_owned())
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(config_home), Some(data_home)) = (
        home("XDG_CONFIG_HOME", ".config"),
        home("XDG_DATA_HOME", ".local/share"),
    ) else {
        eprintln!("enzimos-inference-models: no home directory");
        return ExitCode::FAILURE;
    };
    let config_dir = config_home.join("enzimos");
    if args == ["setup"] {
        let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .ok_or_else(|| "no absolute XDG_RUNTIME_DIR".to_owned());
        return match runtime_dir.and_then(|dir| models::setup(&config_dir, &dir)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("enzimos-inference-models: {error}");
                ExitCode::FAILURE
            }
        };
    }
    let path = config_dir.join("inference.json");
    let store = data_home.join("enzimos/models");
    let mut config = match std::fs::read(&path) {
        Ok(raw) => match serde_json::from_slice(&raw) {
            Ok(config) => config,
            Err(_) => {
                eprintln!(
                    "enzimos-inference-models: {} is not valid JSON",
                    path.display()
                );
                return ExitCode::FAILURE;
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            models::fresh_config(std::path::Path::new(models::RUNTIMES))
        }
        Err(error) => {
            eprintln!("enzimos-inference-models: cannot read registry: {error}");
            return ExitCode::FAILURE;
        }
    };
    models::adopt_runtimes(&mut config, std::path::Path::new(models::RUNTIMES));
    let result = match args.first().map(String::as_str) {
        Some("list") if args.len() == 1 => {
            let mut out = std::io::stdout();
            for (name, item) in config["models"].as_object().into_iter().flatten() {
                let default = if config["default_model"] == *name
                    || config["default_speech_model"] == *name
                {
                    " (default)"
                } else {
                    ""
                };
                let _ = writeln!(
                    out,
                    "{name}\t{}\t{}{default}",
                    item["kind"].as_str().unwrap_or("llm"),
                    item["path"].as_str().unwrap_or_default()
                );
            }
            return ExitCode::SUCCESS;
        }
        Some("add") if args.len() >= 2 => {
            let mut id = None;
            let mut sha = None;
            let mut default = false;
            let mut system = false;
            let mut rest = args[2..].iter();
            let mut bad = false;
            while let Some(flag) = rest.next() {
                match flag.as_str() {
                    "--id" => id = rest.next().cloned(),
                    "--sha256" => sha = rest.next().cloned(),
                    "--default" => default = true,
                    "--system" => system = true,
                    _ => bad = true,
                }
            }
            if bad {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            models::add(
                &mut config,
                if system {
                    models::Keep::Installed { owner: 0 }
                } else {
                    models::Keep::Copy(&store)
                },
                std::path::Path::new(&args[1]),
                id.as_deref(),
                sha.as_deref(),
                default,
            )
            .map(|alias| format!("added {alias}"))
        }
        Some("catalog") if args.len() == 1 => {
            let mut out = std::io::stdout();
            for offer in &models::folder::CATALOG {
                let state = if config["models"].get(offer.id).is_some() {
                    "added"
                } else {
                    "available"
                };
                let _ = writeln!(
                    out,
                    "{}\t{}\t{} MB\t{}\t{state}",
                    offer.id,
                    offer.title,
                    offer.bytes / 1_000_000,
                    offer.languages
                );
            }
            return ExitCode::SUCCESS;
        }
        Some("fetch") if (2..=3).contains(&args.len()) => {
            let default = match args.get(2).map(String::as_str) {
                None => false,
                Some("--default") => true,
                Some(_) => {
                    eprintln!("{USAGE}");
                    return ExitCode::from(2);
                }
            };
            if let Some(offer) = models::folder::offer(&args[1]) {
                let _ = writeln!(
                    std::io::stdout(),
                    "downloading {} ({} MB)",
                    offer.title,
                    offer.bytes / 1_000_000
                );
            }
            let work = store.join(format!(".fetch-{}", args[1].replace('/', "-")));
            models::fetch(&mut config, &store, &work, &args[1], default, download)
                .map(|alias| format!("added {alias}"))
        }
        Some("default") if args.len() == 2 => {
            models::set_default(&mut config, &args[1]).map(|()| format!("default {}", args[1]))
        }
        Some("remove") if args.len() == 2 => {
            models::remove(&mut config, &args[1]).map(|()| format!("removed {}", args[1]))
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result.and_then(|message| models::save(&config, &path).map(|()| message)) {
        Ok(message) => {
            let _ = writeln!(std::io::stdout(), "{message}");
            // The service reads its registry at start; it may not have been
            // able to start before it had a model.
            let _ = std::process::Command::new("systemctl")
                .args(["--user", "restart", "enzimos-inference.service"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("enzimos-inference-models: {error}");
            ExitCode::FAILURE
        }
    }
}
