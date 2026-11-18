// SPDX-License-Identifier: MIT
//! Per-user supervisor for bundled, pinned llama.cpp runtimes and the whisper.cpp
//! and sherpa-onnx speech runtimes.
//! It executes local models only; provider routing belongs to enzimos-ai-gateway.
pub mod calibrate;
pub mod http;
pub mod models;
pub mod policy;
pub mod preferences;
pub mod supervisor;
