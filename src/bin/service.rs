// SPDX-License-Identifier: MIT
//! enzimos-inference --config FILE --socket PATH [--preferences FILE]
use std::{path::PathBuf, process::ExitCode};

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (mut config, mut socket, mut preferences) = (None, None, None);
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        let value = rest.next().cloned();
        match flag.as_str() {
            "--config" => config = value,
            "--socket" => socket = value,
            "--preferences" => preferences = value,
            _ => return usage(),
        }
    }
    let (Some(config), Some(socket)) = (config, socket) else {
        return usage();
    };
    let Ok(Ok(config)) =
        std::fs::read(&config).map(|raw| serde_json::from_slice::<serde_json::Value>(&raw))
    else {
        eprintln!("inference: configuration unavailable");
        return ExitCode::FAILURE;
    };
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return ExitCode::FAILURE;
    };
    runtime.block_on(async move {
        let socket = PathBuf::from(socket);
        let listener = match enzimos_inference_service::supervisor::bind(&socket) {
            Ok(listener) => listener,
            Err(error) => {
                eprintln!("inference: {error}");
                return ExitCode::FAILURE;
            }
        };
        let supervisor = match enzimos_inference_service::supervisor::Supervisor::new(
            config,
            socket.parent().unwrap(),
            preferences.as_deref().map(std::path::Path::new),
        ) {
            Ok(supervisor) => supervisor,
            Err(error) => {
                let _ = std::fs::remove_file(&socket);
                eprintln!("inference: {error}");
                return ExitCode::FAILURE;
            }
        };
        let (Ok(mut term), Ok(mut int)) = (
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()),
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()),
        ) else {
            return ExitCode::FAILURE;
        };
        tokio::spawn(supervisor.clone().monitor());
        tokio::select! {
            () = supervisor.clone().serve(listener) => {}
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
        let mut engine = supervisor.engine.lock().await;
        supervisor.stop(&mut engine).await;
        let _ = std::fs::remove_file(&socket);
        ExitCode::SUCCESS
    })
}

fn usage() -> ExitCode {
    eprintln!("usage: enzimos-inference --config FILE --socket PATH [--preferences FILE]");
    ExitCode::from(2)
}
