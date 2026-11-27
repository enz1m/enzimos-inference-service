// SPDX-License-Identifier: MIT
//! Versioned downloads offered by the model registry.

use serde_json::Value;

/// How a downloaded model is registered.
#[derive(Clone, Copy, Debug)]
pub enum Format {
    /// A GGUF used for text generation.
    Text,
    /// A sherpa-onnx speech model in a compressed directory.
    Speech {
        /// Directory inside the archive.
        folder: &'static str,
        /// Model architecture checked at import.
        architecture: &'static str,
    },
}

/// One model offered for download.
#[derive(Debug)]
pub struct Offer {
    /// Registry ID.
    pub id: &'static str,
    /// Display name.
    pub title: &'static str,
    /// Catalog description.
    pub description: &'static str,
    /// Model format.
    pub format: Format,
    /// HTTPS artifact URL.
    pub url: &'static str,
    /// SHA-256 of the downloaded bytes.
    pub sha256: &'static str,
    /// Download size in bytes.
    pub bytes: u64,
    required_architecture: Option<&'static str>,
}

impl Offer {
    /// Whether the installed runtime can register the offer.
    #[must_use]
    pub fn available(&self, config: &Value) -> bool {
        self.base_runtime_available(config)
            && self.required_architecture.is_none_or(|architecture| {
                config["runtimes"]["cpu"]["server"]
                    .as_str()
                    .and_then(|path| std::path::Path::new(path).parent())
                    .is_some_and(|folder| {
                        folder
                            .join("supported-architectures")
                            .join(architecture)
                            .is_file()
                    })
            })
    }

    /// Whether the main runtime is installed, before checking model-specific capabilities.
    #[must_use]
    pub fn base_runtime_available(&self, config: &Value) -> bool {
        config["runtimes"]["cpu"].is_object()
            && match self.format {
                Format::Text => true,
                Format::Speech { .. } => config["speech"]["onnx"].is_string(),
            }
    }
}

/// Downloads shown in Settings when their runtime is available.
pub const CATALOG: [Offer; 4] = [
    Offer {
        id: "lfm3-vl-2b",
        title: "LFM3-VL-2B",
        description: "General-purpose text model",
        format: Format::Text,
        url: "https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/1add221a5ea6c047fcf1eaaf2d247939d7cf8849/LFM3-VL-2B-Q4_K_M.gguf",
        sha256: "56dac2e192ff0d9ab8507f4f1aeb4aed0463449bbbefaac79497f041c95f21eb",
        bytes: 1_099_627_808,
        required_architecture: Some("lfm3"),
    },
    Offer {
        id: "minicpm5-1b",
        title: "MiniCPM5 1B",
        description: "Smaller text model for low-memory computers",
        format: Format::Text,
        url: "https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/resolve/5385560dd3e232bcc3577fd9442c332cad2facbd/MiniCPM5-1B-Q4_K_M.gguf",
        sha256: "81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa",
        bytes: 688_065_920,
        required_architecture: None,
    },
    Offer {
        id: "parakeet-tdt-0.6b-v3",
        title: "Parakeet TDT 0.6B v3 (int8)",
        description: "25 European languages, detected automatically",
        format: Format::Speech {
            folder: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            architecture: "parakeet-tdt",
        },
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        sha256: "1495e91b59d7fc53ddca91f3431758ec99f8fa253fdb0e1e851be0f72b14b0f9",
        bytes: 487_211_443,
        required_architecture: None,
    },
    Offer {
        id: "moonshine-base",
        title: "Moonshine base (int8)",
        description: "English",
        format: Format::Speech {
            folder: "sherpa-onnx-moonshine-base-en-int8",
            architecture: "moonshine",
        },
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2",
        sha256: "21870cecaa2e44e4e2bf63e02d1072bed183ccd10284871353bd9d24dad14e5e",
        bytes: 250_807_309,
        required_architecture: None,
    },
];

/// Looks up an exact catalog ID.
#[must_use]
pub fn offer(id: &str) -> Option<&'static Offer> {
    CATALOG.iter().find(|offer| offer.id == id)
}
