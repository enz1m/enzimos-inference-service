//! Which speech runtime runs a model, with which arguments, and how its
//! transcript is read: whisper.cpp's `whisper-cli` for whisper models,
//! sherpa-onnx's `sherpa-onnx-offline` for Parakeet TDT and Moonshine folders.

use super::*;
use std::io::Read as _;

/// The languages Parakeet TDT 0.6B v3 transcribes (it detects which itself).
const PARAKEET_LANGUAGES: [&str; 25] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];

/// How a model is run, decided before the job is admitted.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Plan {
    /// whisper-cli writes `<scratch>/transcript.json`.
    Whisper { cli: String, language: String },
    /// sherpa-onnx-offline prints one JSON result line on standard output.
    Sherpa {
        cli: String,
        architecture: &'static str,
    },
}

impl Plan {
    /// The plan for `model` asked for `language` (`auto` or an ISO 639 code).
    ///
    /// # Errors
    /// The model's runtime is not installed, its architecture is unknown, or
    /// it cannot transcribe the language.
    pub(super) fn new(config: &Value, model: &str, language: &str) -> Result<Self, Failure> {
        let item = &config["models"][model];
        let runtime = |key: &str| {
            config["speech"][key]
                .as_str()
                .filter(|c| Path::new(c).is_file())
                .map(str::to_owned)
                .ok_or_else(|| bad("speech runtime absent"))
        };
        match item.get("architecture").and_then(Value::as_str) {
            None | Some("whisper") => Ok(Self::Whisper {
                cli: runtime("cli")?,
                language: language.to_owned(),
            }),
            Some("parakeet-tdt") => {
                if language != "auto" && !PARAKEET_LANGUAGES.contains(&language) {
                    return Err(bad(format!(
                        "Parakeet does not transcribe language '{language}'; it detects one of {}",
                        PARAKEET_LANGUAGES.join(", ")
                    )));
                }
                Ok(Self::Sherpa {
                    cli: runtime("onnx")?,
                    architecture: "parakeet-tdt",
                })
            }
            Some("moonshine") => {
                if !["auto", "en"].contains(&language) {
                    return Err(bad(format!(
                        "Moonshine transcribes English only, not language '{language}'"
                    )));
                }
                Ok(Self::Sherpa {
                    cli: runtime("onnx")?,
                    architecture: "moonshine",
                })
            }
            Some(_) => Err(bad("unsupported speech model architecture")),
        }
    }

    pub(super) fn cli(&self) -> &str {
        match self {
            Self::Whisper { cli, .. } | Self::Sherpa { cli, .. } => cli,
        }
    }

    /// The command line for transcribing `wav` with `item` on `threads`
    /// threads, its temporary files in `scratch`.
    pub(super) fn args(
        &self,
        item: &Value,
        wav: &Path,
        scratch: &Path,
        threads: usize,
    ) -> Vec<String> {
        let path = item["path"].as_str().unwrap_or_default();
        match self {
            Self::Whisper { cli, language } => vec![
                cli.clone(),
                "-m".into(),
                path.into(),
                "-f".into(),
                wav.to_string_lossy().into_owned(),
                "-t".into(),
                threads.to_string(),
                "-ng".into(),
                "-l".into(),
                language.clone(),
                "-oj".into(),
                "-of".into(),
                scratch.join("transcript").to_string_lossy().into_owned(),
            ],
            Self::Sherpa { cli, architecture } => {
                let file = |role: &str| {
                    Path::new(path)
                        .join(item["files"][role]["name"].as_str().unwrap_or_default())
                        .to_string_lossy()
                        .into_owned()
                };
                let mut args = vec![cli.clone(), format!("--tokens={}", file("tokens"))];
                if *architecture == "moonshine" {
                    for (flag, role) in [
                        ("moonshine-preprocessor", "preprocessor"),
                        ("moonshine-encoder", "encoder"),
                        ("moonshine-uncached-decoder", "uncached_decoder"),
                        ("moonshine-cached-decoder", "cached_decoder"),
                    ] {
                        args.push(format!("--{flag}={}", file(role)));
                    }
                } else {
                    for role in ["encoder", "decoder", "joiner"] {
                        args.push(format!("--{role}={}", file(role)));
                    }
                    args.push("--model-type=nemo_transducer".into());
                }
                args.extend([
                    format!("--num-threads={threads}"),
                    "--provider=cpu".into(),
                    "--debug=0".into(),
                    wav.to_string_lossy().into_owned(),
                ]);
                args
            }
        }
    }

    /// Where the runtime's standard output goes: a file for sherpa-onnx,
    /// which prints its result there; nowhere for whisper-cli.
    pub(super) fn output(&self, scratch: &Path) -> Option<PathBuf> {
        matches!(self, Self::Sherpa { .. }).then(|| scratch.join("output.txt"))
    }

    /// The transcript the runtime left in `scratch`.
    pub(super) fn transcript(&self, scratch: &Path) -> Option<String> {
        match self {
            Self::Whisper { .. } => {
                let raw = std::fs::read(scratch.join("transcript.json")).ok()?;
                let value: Value = serde_json::from_slice(&raw).ok()?;
                Some(
                    value["transcription"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|x| x["text"].as_str())
                        .collect::<String>(),
                )
            }
            Self::Sherpa { .. } => {
                let mut raw = Vec::new();
                std::fs::File::open(scratch.join("output.txt"))
                    .ok()?
                    .take(1 << 20)
                    .read_to_end(&mut raw)
                    .ok()?;
                sherpa_text(&String::from_utf8_lossy(&raw))
            }
        }
    }
}

/// The text of sherpa-onnx-offline's result: its last JSON line's `text`.
pub(super) fn sherpa_text(output: &str) -> Option<String> {
    output
        .lines()
        .rev()
        .filter(|line| line.trim_start().starts_with('{'))
        .find_map(|line| serde_json::from_str::<Value>(line).ok())
        .and_then(|value| value["text"].as_str().map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(dir: &Path) -> Value {
        let whisper = dir.join("whisper-cli");
        let onnx = dir.join("sherpa-onnx-offline");
        std::fs::write(&whisper, b"").unwrap();
        std::fs::write(&onnx, b"").unwrap();
        let files = |roles: &[&str]| {
            roles
                .iter()
                .map(|role| {
                    (
                        (*role).to_owned(),
                        json!({"name": format!("{role}.int8.onnx")}),
                    )
                })
                .collect::<Map<_, _>>()
        };
        json!({"speech": {"cli": whisper, "onnx": onnx}, "models": {
            "whisper-base": {"path": "/m/ggml-base.bin", "kind": "speech"},
            "parakeet": {"path": "/m/p", "kind": "speech", "architecture": "parakeet-tdt",
                "files": files(&["encoder", "decoder", "joiner", "tokens"])},
            "moonshine": {"path": "/m/m", "kind": "speech", "architecture": "moonshine",
                "files": files(&["preprocessor", "encoder", "uncached_decoder", "cached_decoder", "tokens"])},
            "odd": {"path": "/m/o", "kind": "speech", "architecture": "wav2vec"}}})
    }

    /// Each architecture runs its own runtime with its own files and thread
    /// count; Parakeet is told it is a NeMo transducer, whisper gets the
    /// language and writes JSON beside the audio.
    #[test]
    fn speech_arguments_follow_the_architecture() {
        let dir = tempfile::tempdir().unwrap();
        let config = config(dir.path());
        let (wav, scratch) = (Path::new("/s/audio.wav"), Path::new("/s"));
        let args = |model: &str, language: &str| {
            Plan::new(&config, model, language).unwrap().args(
                &config["models"][model],
                wav,
                scratch,
                2,
            )
        };
        let onnx = config["speech"]["onnx"].as_str().unwrap().to_owned();
        assert_eq!(
            args("parakeet", "auto"),
            [
                onnx.as_str(),
                "--tokens=/m/p/tokens.int8.onnx",
                "--encoder=/m/p/encoder.int8.onnx",
                "--decoder=/m/p/decoder.int8.onnx",
                "--joiner=/m/p/joiner.int8.onnx",
                "--model-type=nemo_transducer",
                "--num-threads=2",
                "--provider=cpu",
                "--debug=0",
                "/s/audio.wav"
            ]
        );
        assert_eq!(
            args("moonshine", "en"),
            [
                onnx.as_str(),
                "--tokens=/m/m/tokens.int8.onnx",
                "--moonshine-preprocessor=/m/m/preprocessor.int8.onnx",
                "--moonshine-encoder=/m/m/encoder.int8.onnx",
                "--moonshine-uncached-decoder=/m/m/uncached_decoder.int8.onnx",
                "--moonshine-cached-decoder=/m/m/cached_decoder.int8.onnx",
                "--num-threads=2",
                "--provider=cpu",
                "--debug=0",
                "/s/audio.wav"
            ]
        );
        let whisper = args("whisper-base", "de");
        assert_eq!(whisper[0], config["speech"]["cli"].as_str().unwrap());
        assert_eq!(
            whisper[1..],
            [
                "-m",
                "/m/ggml-base.bin",
                "-f",
                "/s/audio.wav",
                "-t",
                "2",
                "-ng",
                "-l",
                "de",
                "-oj",
                "-of",
                "/s/transcript"
            ]
        );
        assert!(
            Plan::new(&config, "odd", "auto").is_err(),
            "an unknown architecture"
        );
        let mut missing = config.clone();
        missing["speech"]["onnx"] = json!(dir.path().join("absent"));
        assert!(
            Plan::new(&missing, "parakeet", "auto").is_err(),
            "no sherpa-onnx runtime"
        );
        assert!(Plan::new(&missing, "whisper-base", "auto").is_ok());
    }

    /// Moonshine refuses any language but English, saying so; Parakeet takes
    /// its own languages and refuses others; whisper takes any.
    #[test]
    fn speech_language_is_refused_where_the_model_cannot() {
        let dir = tempfile::tempdir().unwrap();
        let config = config(dir.path());
        let refused = |model: &str, language: &str| match Plan::new(&config, model, language) {
            Err(Failure::Bad(message)) => Some(message),
            _ => None,
        };
        assert_eq!(
            refused("moonshine", "de").as_deref(),
            Some("Moonshine transcribes English only, not language 'de'")
        );
        assert!(refused("moonshine", "en").is_none() && refused("moonshine", "auto").is_none());
        assert!(refused("parakeet", "ja").is_some_and(|m| m.contains("'ja'")));
        assert!(refused("parakeet", "de").is_none() && refused("parakeet", "auto").is_none());
        assert!(refused("whisper-base", "ja").is_none());
    }

    /// The transcript is the last JSON line's text, after any log lines.
    #[test]
    fn sherpa_transcript_is_its_json_line() {
        let output = "/x/sherpa-onnx-offline --tokens=...\n{\"lang\": \"\", \"text\": \"Ask not what your country can do.\", \"tokens\": [\" A\"]}\n";
        assert_eq!(
            sherpa_text(output).as_deref(),
            Some("Ask not what your country can do.")
        );
        assert_eq!(sherpa_text("Done!\n"), None);
    }
}
