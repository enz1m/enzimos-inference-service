// SPDX-License-Identifier: MIT
//! Test double for llama-server: serves `--host SOCKET`, logs request bodies to
//! FAKE_LOG, streams two SSE events. FAKE_MODE=fail exits at once; FAKE_DELAY_MS
//! delays each response.
use std::{
    io::{Read, Write},
    os::unix::net::UnixListener,
};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if std::env::var("FAKE_MODE").as_deref() == Ok("fail") {
        std::process::exit(1);
    }
    let socket = args
        .iter()
        .position(|a| a == "--host")
        .map(|i| args[i + 1].clone())
        .expect("--host");
    let delay = std::env::var("FAKE_DELAY_MS")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(0);
    let log = std::env::var("FAKE_LOG").ok();
    let listener = UnixListener::bind(&socket).unwrap();
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let mut data = Vec::new();
        let mut chunk = [0u8; 65536];
        loop {
            let n = stream.read(&mut chunk).unwrap_or(0);
            data.extend_from_slice(&chunk[..n]);
            let text = String::from_utf8_lossy(&data).into_owned();
            if let Some(end) = text.find("\r\n\r\n") {
                let length = text
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                if data.len() >= end + 4 + length || n == 0 {
                    break;
                }
            }
            if n == 0 {
                break;
            }
        }
        let text = String::from_utf8_lossy(&data).into_owned();
        if text.starts_with("GET /health") {
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}");
            continue;
        }
        if let Some(log) = &log {
            let body = text
                .split_once("\r\n\r\n")
                .map(|(_, b)| b)
                .unwrap_or_default();
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log)
                .unwrap();
            let _ = writeln!(file, "{body}");
        }
        std::thread::sleep(std::time::Duration::from_millis(delay));
        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"content\":\"one\"}\n\n");
        let _ = stream.write_all(b"data: {\"content\":\"two\"}\n\n");
    }
}
