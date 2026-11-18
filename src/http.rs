// SPDX-License-Identifier: MIT
//! Minimal one-request-per-connection HTTP/1.1 on a private Unix socket.
use serde_json::Value;
use std::{collections::HashMap, time::Duration};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const HEAD_LIMIT: usize = 16384;

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    /// Bytes already read past the head (the start of the body).
    pub rest: Vec<u8>,
}

/// Read the request head within 5 seconds; duplicate headers are refused.
pub async fn read_head<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Request, String> {
    let mut data = Vec::new();
    let mut chunk = [0u8; 4096];
    let end = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(end) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                return Ok(end);
            }
            if data.len() > HEAD_LIMIT {
                return Err("request head exceeds limit".to_owned());
            }
            let n = reader
                .read(&mut chunk)
                .await
                .map_err(|_| "connection closed".to_owned())?;
            if n == 0 {
                return Err("connection closed".into());
            }
            data.extend_from_slice(&chunk[..n]);
        }
    })
    .await
    .map_err(|_| "request head timed out".to_owned())??;
    let head = std::str::from_utf8(&data[..end]).map_err(|_| "invalid request head".to_owned())?;
    let mut lines = head.split("\r\n");
    let parts = lines
        .next()
        .unwrap_or_default()
        .split(' ')
        .collect::<Vec<_>>();
    let [method, path, _] = parts.as_slice() else {
        return Err("invalid request line".into());
    };
    let mut headers = HashMap::new();
    for line in lines.filter(|l| !l.is_empty()) {
        let (key, value) = line.split_once(':').ok_or("invalid header")?;
        if headers
            .insert(key.to_ascii_lowercase(), value.trim().to_owned())
            .is_some()
        {
            return Err("duplicate header".into());
        }
    }
    Ok(Request {
        method: method.to_string(),
        path: path.to_string(),
        headers,
        rest: data[end + 4..].to_vec(),
    })
}

/// Read exactly the declared body within 10 seconds.
pub async fn read_body<R: AsyncRead + Unpin>(
    reader: &mut R,
    request: &Request,
    limit: usize,
) -> Result<Value, String> {
    if request.headers.contains_key("transfer-encoding") {
        return Err("unsupported request".into());
    }
    let length = request
        .headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    if length == 0 || length > limit || request.rest.len() > length {
        return Err("invalid body length".into());
    }
    let mut body = request.rest.clone();
    body.resize(length, 0);
    tokio::time::timeout(
        Duration::from_secs(10),
        reader.read_exact(&mut body[request.rest.len()..]),
    )
    .await
    .map_err(|_| "request body timed out".to_owned())?
    .map_err(|_| "connection closed".to_owned())?;
    match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(object)) => Ok(Value::Object(object)),
        _ => Err("JSON object required".into()),
    }
}

pub async fn reply<W: AsyncWrite + Unpin>(
    writer: &mut W,
    status: u16,
    body: &Value,
) -> std::io::Result<()> {
    let data = body.to_string();
    let head = format!(
        "HTTP/1.1 {status} Result\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        data.len()
    );
    writer.write_all(head.as_bytes()).await?;
    writer.write_all(data.as_bytes()).await?;
    writer.flush().await
}

/// One JSON POST over a fresh connection, headers in the given order.
pub fn post(path: &str, host: &str, headers: &[(String, String)], body: &Value) -> Vec<u8> {
    let data = body.to_string();
    let mut head = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        data.len()
    );
    for (key, value) in headers {
        head.push_str(&format!("{key}: {value}\r\n"));
    }
    head.push_str("\r\n");
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(data.as_bytes());
    bytes
}
