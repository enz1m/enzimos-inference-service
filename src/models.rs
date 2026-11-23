// SPDX-License-Identifier: MIT
//! The model registry a user manages: adding a model file (a llama.cpp GGUF
//! such as MiniCPM, or a whisper.cpp speech model) or a sherpa-onnx speech
//! model folder (Parakeet TDT, Moonshine) checks its format, names it from its
//! own metadata, records its digests and keeps a read-only copy, then
//! rewrites the service's configuration only if it still validates.

use crate::policy;
use serde_json::{Value, json};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

pub mod catalog;
pub mod folder;

/// Where the installed packages put the runtimes.
pub const RUNTIMES: &str = "/usr/lib/enzimos-inference";

/// What a model file is, read from its header.
#[derive(Debug, PartialEq, Eq)]
pub struct Identity {
    /// `llm` or `speech`.
    pub kind: &'static str,
    /// The model's own name (`general.name`), if it gives one.
    pub name: Option<String>,
    /// `general.architecture` of a GGUF (`minicpm`, `llama`...).
    pub architecture: Option<String>,
    /// `general.size_label` (`1B`...), if given.
    pub size: Option<String>,
}

/// Reads a model file's header: a GGUF (text model) or a whisper.cpp model;
/// or a speech model folder's layout.
///
/// # Errors
/// The file cannot be read or is no known format.
pub fn identify(path: &Path) -> Result<Identity, String> {
    if path.is_dir() {
        let layout = folder::identify(path)?;
        return Ok(Identity {
            kind: "speech",
            name: None,
            architecture: Some(layout.architecture.to_owned()),
            size: None,
        });
    }
    let file = std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0_u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|_| "the file is too short to be a model".to_owned())?;
    match &magic {
        b"GGUF" => gguf(&mut reader),
        // whisper.cpp's model magic, 0x67676d6c little-endian.
        b"lmgg" => Ok(Identity {
            kind: "speech",
            name: None,
            architecture: Some("whisper".to_owned()),
            size: None,
        }),
        _ => Err("not a GGUF or whisper.cpp model".to_owned()),
    }
}

fn gguf(reader: &mut impl Read) -> Result<Identity, String> {
    let bad = |_| "the GGUF header is damaged".to_owned();
    let version = read_u32(reader).map_err(bad)?;
    if !(2..=3).contains(&version) {
        return Err(format!("GGUF version {version} is not supported"));
    }
    let _tensors = read_u64(reader).map_err(bad)?;
    let entries = read_u64(reader).map_err(bad)?;
    if entries > 100_000 {
        return Err("the GGUF header is damaged".to_owned());
    }
    let mut found = std::collections::BTreeMap::new();
    for _ in 0..entries {
        let key = read_string(reader).map_err(bad)?;
        let kind = read_u32(reader).map_err(bad)?;
        let wanted = matches!(
            key.as_str(),
            "general.name" | "general.architecture" | "general.size_label"
        );
        if let Some(value) = read_value(reader, kind, wanted).map_err(bad)? {
            found.insert(key, value);
        }
        if found.len() == 3 {
            break;
        }
    }
    let architecture = found.remove("general.architecture");
    // A whisper model converted to GGUF is still speech.
    let kind = if architecture.as_deref() == Some("whisper") {
        "speech"
    } else {
        "llm"
    };
    Ok(Identity {
        kind,
        name: found.remove("general.name"),
        architecture,
        size: found.remove("general.size_label"),
    })
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0; 4];
    reader.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0; 8];
    reader.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(reader: &mut impl Read) -> std::io::Result<String> {
    let length = read_u64(reader)?;
    if length > 1 << 20 {
        return Err(std::io::ErrorKind::InvalidData.into());
    }
    let mut bytes = vec![0; usize::try_from(length).map_err(|_| std::io::ErrorKind::InvalidData)?];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Reads (or skips) one GGUF value; returns it when `keep` and a string.
fn read_value(reader: &mut impl Read, kind: u32, keep: bool) -> std::io::Result<Option<String>> {
    let fixed = |kind| match kind {
        0 | 1 | 7 => Some(1_u64),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    };
    match kind {
        8 => {
            let text = read_string(reader)?;
            Ok(keep.then_some(text))
        }
        9 => {
            let inner = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > 1 << 24 {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            for _ in 0..count {
                read_value(reader, inner, false)?;
            }
            Ok(None)
        }
        other => {
            let size = fixed(other).ok_or(std::io::ErrorKind::InvalidData)?;
            std::io::copy(&mut reader.take(size), &mut std::io::sink())?;
            Ok(None)
        }
    }
}

/// A registry alias from the model's own name, or its file name.
pub fn alias_for(identity: &Identity, path: &Path) -> String {
    let base = identity.name.clone().unwrap_or_else(|| {
        path.file_stem().map_or_else(
            || "model".to_owned(),
            |stem| stem.to_string_lossy().into_owned(),
        )
    });
    let mut alias: String = base
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if let Some(size) = &identity.size {
        let size = size.to_ascii_lowercase();
        if !alias.contains(&size) {
            alias = format!("{alias}-{size}");
        }
    }
    let alias: String = alias
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    alias.chars().take(64).collect()
}

/// A new configuration for the runtimes installed under `runtimes`.
pub fn fresh_config(runtimes: &Path) -> Value {
    let mut config = json!({"schema": 1, "runtimes": {}, "models": {}});
    for backend in policy::BACKENDS {
        let server = runtimes.join(backend).join("llama-server");
        if server.is_file() {
            config["runtimes"][backend] = json!({"server": server, "revision": "installed"});
        }
    }
    let cli = runtimes.join("speech/whisper-cli");
    if cli.is_file() {
        config["speech"]["cli"] = json!(cli);
    }
    let onnx = runtimes.join(ONNX_RUNTIME);
    if onnx.is_file() {
        config["speech"]["onnx"] = json!(onnx);
    }
    config
}

/// sherpa-onnx's offline recogniser, under the runtimes.
pub const ONNX_RUNTIME: &str = "speech-onnx/sherpa-onnx-offline";

/// Fills in speech runtimes installed under `runtimes` that `config` does
/// not name yet (a registry written before they were installed); a runtime
/// it names is kept.
pub fn adopt_runtimes(config: &mut Value, runtimes: &Path) {
    let fresh = fresh_config(runtimes);
    for key in ["cli", "onnx"] {
        if config["speech"].get(key).is_none()
            && let Some(path) = fresh["speech"].get(key)
        {
            if !config["speech"].is_object() {
                config["speech"] = json!({});
            }
            config["speech"][key] = path.clone();
        }
    }
}

/// Where an added model's file stays.
#[derive(Clone, Copy, Debug)]
pub enum Keep<'a> {
    /// A read-only copy in this account's store.
    Copy(&'a Path),
    /// The file itself: a model the system installed, owned by `owner`
    /// (root) and writable by no one else, in a folder likewise owned.
    Installed { owner: u32 },
}

/// Adds the model at `source` to `config`, kept as `keep` says; `id` names
/// it (else its own name does), `default` makes a text model the default
/// (the first one always is). Returns its alias.
///
/// # Errors
/// The file is no model, its digest is not `expected`, it cannot be kept,
/// an installed model is not protected, or the configuration would no
/// longer validate; `config` is then unchanged.
pub fn add(
    config: &mut Value,
    keep: Keep<'_>,
    source: &Path,
    id: Option<&str>,
    expected: Option<&str>,
    default: bool,
) -> Result<String, String> {
    if source.is_dir() {
        return add_folder(config, keep, source, id, expected, default);
    }
    let identity = identify(source)?;
    if !config["runtimes"]["cpu"].is_object() {
        return Err("the local AI runtime is not installed (enzimos-llama-cpu)".to_owned());
    }
    let alias = id.map_or_else(|| alias_for(&identity, source), str::to_owned);
    let sha =
        policy::file_sha256(source).map_err(|error| format!("{}: {error}", source.display()))?;
    if let Some(expected) = expected
        && !expected.eq_ignore_ascii_case(&sha)
    {
        return Err("the file's digest does not match the expected one".to_owned());
    }
    let kept: PathBuf = match keep {
        Keep::Copy(store) => {
            std::fs::create_dir_all(store)
                .map_err(|error| format!("{}: {error}", store.display()))?;
            let extension = if identity.kind == "speech"
                && identity.architecture.as_deref() == Some("whisper")
                && source.extension().is_some_and(|e| e == "bin")
            {
                "bin"
            } else {
                "gguf"
            };
            let kept = store.join(format!("{sha}.{extension}"));
            if !kept.is_file() {
                let partial = store.join(format!(".{sha}.partial"));
                std::fs::copy(source, &partial)
                    .map_err(|error| format!("copying the model: {error}"))?;
                set_mode(&partial, 0o444)?;
                std::fs::rename(&partial, &kept).map_err(|error| error.to_string())?;
            }
            kept
        }
        Keep::Installed { owner } => installed(source, owner)?,
    };
    let mut next = config.clone();
    if !next["models"].is_object() {
        next["models"] = json!({});
    }
    next["models"][&alias] = json!({"path": kept, "sha256": sha, "kind": identity.kind});
    if let Some(architecture) = identity
        .architecture
        .as_deref()
        .filter(|_| identity.kind == "speech")
    {
        next["models"][&alias]["architecture"] = json!(architecture);
    }
    if identity.kind == "llm" {
        next["models"][&alias]["backend_options"] = json!({"cpu": cpu_options(source)});
    }
    let has_default = next["default_model"]
        .as_str()
        .is_some_and(|m| next["models"].get(m).is_some());
    if identity.kind == "llm" && (default || !has_default) {
        next["default_model"] = json!(alias);
    }
    if identity.kind == "speech" && default {
        next["default_speech_model"] = json!(alias);
    }
    policy::validate(&next, false)?;
    *config = next;
    Ok(alias)
}

/// [`add`] for a sherpa-onnx speech model folder: its layout's files are
/// recorded with their digests and kept together; `expected` is the digest
/// over them ([`folder::manifest`]).
fn add_folder(
    config: &mut Value,
    keep: Keep<'_>,
    source: &Path,
    id: Option<&str>,
    expected: Option<&str>,
    default: bool,
) -> Result<String, String> {
    let layout = folder::identify(source)?;
    if !config["runtimes"]["cpu"].is_object() {
        return Err("the local AI runtime is not installed (enzimos-llama-cpu)".to_owned());
    }
    if !config["speech"]["onnx"].is_string() {
        return Err(
            "the speech runtime for this model is not installed (enzimos-sherpa-onnx-cpu)"
                .to_owned(),
        );
    }
    let alias = id.map_or_else(
        || {
            source.file_name().map_or_else(
                || layout.architecture.to_owned(),
                |name| {
                    let name = name.to_string_lossy();
                    let name = name.strip_prefix("sherpa-onnx-").unwrap_or(&name);
                    alias_for(
                        &Identity {
                            kind: "speech",
                            name: Some(name.to_owned()),
                            architecture: None,
                            size: None,
                        },
                        source,
                    )
                },
            )
        },
        str::to_owned,
    );
    let files = folder::digests(source, &layout)?;
    let manifest = folder::manifest(&files);
    if let Some(expected) = expected
        && !expected.eq_ignore_ascii_case(&manifest)
    {
        return Err("the folder's digest does not match the expected one".to_owned());
    }
    let kept = match keep {
        Keep::Copy(store) => {
            std::fs::create_dir_all(store)
                .map_err(|error| format!("{}: {error}", store.display()))?;
            folder::keep_copy(store, source, &layout, &manifest)?
        }
        Keep::Installed { owner } => folder::installed(source, &layout, owner)?,
    };
    let mut next = config.clone();
    if !next["models"].is_object() {
        next["models"] = json!({});
    }
    next["models"][&alias] = json!({"path": kept, "sha256": manifest, "kind": "speech",
        "architecture": layout.architecture, "files": files});
    if default {
        next["default_speech_model"] = json!(alias);
    }
    policy::validate(&next, false)?;
    *config = next;
    Ok(alias)
}

/// Downloads and verifies a catalog model before registering it.
///
/// # Errors
/// The offer or its runtime is unavailable, or the download fails validation.
pub fn fetch(
    config: &mut Value,
    store: &Path,
    work: &Path,
    id: &str,
    default: bool,
    download: impl FnOnce(&str, &Path) -> Result<(), String>,
) -> Result<String, String> {
    let offer = catalog::offer(id).ok_or_else(|| format!("no catalog model named {id}"))?;
    if !offer.available(config) {
        return Err("the runtime for this model is not installed".into());
    }
    struct Scratch<'a>(&'a Path);
    impl Drop for Scratch<'_> {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.0);
        }
    }
    std::fs::create_dir_all(work).map_err(|error| error.to_string())?;
    let _scratch = Scratch(work);
    set_mode(work, 0o700)?;
    let downloaded = work.join("download");
    download(offer.url, &downloaded)?;
    let source = match offer.format {
        catalog::Format::Text => {
            return add(
                config,
                Keep::Copy(store),
                &downloaded,
                Some(offer.id),
                Some(offer.sha256),
                default,
            );
        }
        catalog::Format::Speech {
            folder: name,
            architecture,
        } => {
            let sha = policy::file_sha256(&downloaded)
                .map_err(|error| format!("reading the download: {error}"))?;
            if sha != offer.sha256 {
                return Err(format!(
                    "the download's SHA-256 does not match the catalog entry for {}; no model was added",
                    offer.title
                ));
            }
            let unpacked = work.join("unpacked");
            std::fs::create_dir(&unpacked).map_err(|error| error.to_string())?;
            folder::unpack(&downloaded, &unpacked)?;
            let source = unpacked.join(name);
            if identify(&source)?.architecture.as_deref() != Some(architecture) {
                return Err("the download is not the speech model the catalog names".into());
            }
            source
        }
    };
    add(
        config,
        Keep::Copy(store),
        &source,
        Some(offer.id),
        None,
        default,
    )
}

/// Makes model `id` the default of its kind: the text model the service
/// answers with, or the speech model dictation uses.
///
/// # Errors
/// There is no such model.
pub fn set_default(config: &mut Value, id: &str) -> Result<(), String> {
    if config["models"].get(id).is_none() {
        return Err(format!("no model named {id}"));
    }
    let key = if policy::kind(config, id) == "speech" {
        "default_speech_model"
    } else {
        "default_model"
    };
    let mut next = config.clone();
    next[key] = json!(id);
    policy::validate(&next, false)?;
    *config = next;
    Ok(())
}

/// An installed model's own path, if nobody but `owner` can change it or
/// the folder it is in.
fn installed(source: &Path, owner: u32) -> Result<PathBuf, String> {
    use std::os::unix::fs::MetadataExt as _;
    let path = std::fs::canonicalize(source).map_err(|error| error.to_string())?;
    let protected = |p: &Path, file: bool| {
        std::fs::symlink_metadata(p).is_ok_and(|m| {
            (if file { m.is_file() } else { m.is_dir() })
                && m.uid() == owner
                && m.mode() & 0o022 == 0
        })
    };
    let parent = path.parent().ok_or("an installed model needs a folder")?;
    if !protected(&path, true) || !protected(parent, false) {
        return Err("an installed model must be a system file nobody else can change".to_owned());
    }
    Ok(path)
}

// Weight repacking adds a copy of part of the model to its memory footprint.
fn cpu_options(path: &Path) -> Value {
    let size = std::fs::metadata(path).map_or(u64::MAX, |m| m.len());
    json!({"cache_type": "f16", "repack": size <= 1 << 30})
}

/// Removes model `id`; the default moves to another text model if it was it.
///
/// # Errors
/// There is no such model, or the configuration would no longer validate.
pub fn remove(config: &mut Value, id: &str) -> Result<(), String> {
    let mut next = config.clone();
    if next["models"]
        .as_object_mut()
        .and_then(|m| m.remove(id))
        .is_none()
    {
        return Err(format!("no model named {id}"));
    }
    if next["default_model"] == json!(id) {
        let other = next["models"]
            .as_object()
            .into_iter()
            .flatten()
            .find(|(name, _)| policy::kind(&next, name) == "llm")
            .map(|(name, _)| name.clone());
        next["default_model"] = other.map_or(Value::Null, Value::from);
    }
    if next["default_speech_model"] == json!(id)
        && let Some(object) = next.as_object_mut()
    {
        object.remove("default_speech_model");
    }
    policy::validate(&next, false)?;
    *config = next;
    Ok(())
}

fn set_mode(path: &Path, mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .map_err(|error| error.to_string())
}

/// Writes `config` to `path` (owner only), atomically.
///
/// # Errors
/// The file cannot be written.
pub fn save(config: &Value, path: &Path) -> Result<(), String> {
    write_config(config, path, true)
}

fn write_config(config: &Value, path: &Path, replace: bool) -> Result<(), String> {
    let error = |error: std::io::Error| error.to_string();
    let parent = path.parent().ok_or("configuration path has no directory")?;
    std::fs::create_dir_all(parent).map_err(error)?;
    let file = tempfile::NamedTempFile::new_in(parent).map_err(error)?;
    serde_json::to_writer_pretty(file.as_file(), config).map_err(|error| error.to_string())?;
    file.as_file().sync_all().map_err(error)?;
    let published = if replace {
        file.persist(path)
    } else {
        file.persist_noclobber(path)
    };
    match published {
        Ok(_) => std::fs::File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(error),
        Err(failure) if !replace && failure.error.kind() == std::io::ErrorKind::AlreadyExists => {
            Ok(())
        }
        Err(failure) => Err(failure.error.to_string()),
    }
}

/// Initializes the installed model registry and local gateway configuration.
/// Existing user configuration is preserved.
///
/// # Errors
/// Returns an error if a shipped model or its manifest is invalid, or new
/// configuration cannot be published.
pub fn setup(config_dir: &Path, socket_dir: &Path) -> Result<(), String> {
    seed_registry(
        &config_dir.join("inference.json"),
        Path::new("/usr/share/enzimos/models"),
        Path::new(RUNTIMES),
        0,
    )?;
    let providers = config_dir.join("ai-providers.json");
    if !providers.try_exists().map_err(|error| error.to_string())? {
        write_config(
            &json!({
                "default_provider": "local",
                "providers": {"local": {
                    "kind": "local",
                    "socket": socket_dir.join("enzimos-inference/service.sock"),
                }},
            }),
            &providers,
            false,
        )?;
    }
    Ok(())
}

fn seed_registry(
    registry: &Path,
    shipped: &Path,
    runtimes: &Path,
    owner: u32,
) -> Result<(), String> {
    if registry.try_exists().map_err(|error| error.to_string())? {
        return Ok(());
    }
    let manifest = match std::fs::File::open(shipped.join("models.tsv")) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.to_string()),
    };
    const MANIFEST_LIMIT: u64 = 16 * 1024;
    let mut rows = String::new();
    manifest
        .take(MANIFEST_LIMIT + 1)
        .read_to_string(&mut rows)
        .map_err(|error| error.to_string())?;
    if rows.len() as u64 > MANIFEST_LIMIT {
        return Err("shipped model manifest exceeds its size limit".into());
    }
    let mut config = fresh_config(runtimes);
    for row in rows
        .lines()
        .filter(|x| !x.is_empty() && !x.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let [id, file, digest, default] = fields.as_slice() else {
            return Err("invalid shipped model entry".into());
        };
        if Path::new(file).file_name().and_then(|v| v.to_str()) != Some(file)
            || !matches!(*default, "yes" | "no")
            || config["models"].get(*id).is_some()
        {
            return Err("invalid shipped model entry".into());
        }
        add(
            &mut config,
            Keep::Installed { owner },
            &shipped.join(file),
            Some(id),
            Some(digest),
            *default == "yes",
        )?;
    }
    if config["models"]
        .as_object()
        .is_some_and(|models| !models.is_empty())
    {
        write_config(&config, registry, false)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF header with `general.*` strings among other keys, an array
    /// before them as a tokenizer has, and no tensors.
    fn gguf_file(path: &Path, name: &str, architecture: &str) {
        let mut b = b"GGUF".to_vec();
        b.extend(3_u32.to_le_bytes());
        b.extend(0_u64.to_le_bytes());
        b.extend(5_u64.to_le_bytes());
        let string = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        };
        string(&mut b, "tokenizer.ggml.tokens");
        b.extend(9_u32.to_le_bytes());
        b.extend(8_u32.to_le_bytes());
        b.extend(2_u64.to_le_bytes());
        string(&mut b, "<s>");
        string(&mut b, "</s>");
        string(&mut b, "general.context_length");
        b.extend(4_u32.to_le_bytes());
        b.extend(4096_u32.to_le_bytes());
        for (key, value) in [
            ("general.architecture", architecture),
            ("general.name", name),
            ("general.size_label", "1B"),
        ] {
            string(&mut b, key);
            b.extend(8_u32.to_le_bytes());
            string(&mut b, value);
        }
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn first_login_publishes_only_a_complete_registry_and_keeps_user_edits() {
        use std::os::unix::fs::MetadataExt as _;
        let fixture = tempfile::tempdir().unwrap();
        let dir = fixture.path();
        let runtimes = dir.join("runtimes");
        std::fs::create_dir_all(runtimes.join("cpu")).unwrap();
        std::fs::write(runtimes.join("cpu/llama-server"), b"#!").unwrap();
        let model = dir.join("text.gguf");
        gguf_file(&model, "MiniCPM5", "minicpm");
        let speech = dir.join("speech.bin");
        std::fs::write(&speech, b"lmggfixture").unwrap();
        let owner = std::fs::metadata(dir).unwrap().uid();
        let registry = dir.join("config/inference.json");
        let manifest = dir.join("models.tsv");
        let text_digest = policy::file_sha256(&model).unwrap();
        let rows = |speech_digest| {
            format!(
                "text\ttext.gguf\t{text_digest}\tyes\nspeech\tspeech.bin\t{speech_digest}\tyes\n"
            )
        };
        std::fs::write(&manifest, rows("wrong-digest")).unwrap();
        assert!(seed_registry(&registry, dir, &runtimes, owner).is_err());
        assert!(!registry.exists());
        let speech_digest = policy::file_sha256(&speech).unwrap();
        std::fs::write(&manifest, rows(&speech_digest)).unwrap();
        seed_registry(&registry, dir, &runtimes, owner).unwrap();
        let mut saved: Value = serde_json::from_slice(&std::fs::read(&registry).unwrap()).unwrap();
        assert_eq!(saved["default_model"], "text");
        assert_eq!(saved["default_speech_model"], "speech");
        assert_eq!(std::fs::metadata(&registry).unwrap().mode() & 0o777, 0o600);
        saved["default_speech_model"] = Value::Null;
        save(&saved, &registry).unwrap();
        write_config(&json!({"unexpected":true}), &registry, false).unwrap();
        seed_registry(&registry, dir, &runtimes, owner).unwrap();
        let current: Value = serde_json::from_slice(&std::fs::read(&registry).unwrap()).unwrap();
        assert_eq!(current, saved);
    }

    /// A MiniCPM GGUF is recognised from its own header and enrolled under
    /// its name with its digest, as a read-only copy, and becomes the
    /// default; a speech model joins without displacing it; a file that is
    /// no model, or a digest that does not match, changes nothing.
    #[test]
    fn a_model_file_is_checked_named_kept_and_registered() {
        let fixture = tempfile::tempdir().unwrap();
        let dir = fixture.path();
        std::fs::create_dir_all(dir.join("runtimes/cpu")).unwrap();
        std::fs::write(dir.join("runtimes/cpu/llama-server"), b"#!").unwrap();
        let gguf = dir.join("download.gguf");
        gguf_file(&gguf, "MiniCPM5", "minicpm");
        let whisper = dir.join("ggml-base.bin");
        std::fs::write(&whisper, b"lmggrest of a whisper model").unwrap();
        let store = dir.join("store");
        let mut config = fresh_config(&dir.join("runtimes"));
        config["context"] = json!(4608);

        assert_eq!(
            identify(&gguf).unwrap().architecture.as_deref(),
            Some("minicpm")
        );
        let alias = add(&mut config, Keep::Copy(&store), &gguf, None, None, false).unwrap();
        assert_eq!(alias, "minicpm5-1b");
        assert_eq!(config["default_model"], "minicpm5-1b");
        assert_eq!(
            config["models"]["minicpm5-1b"]["backend_options"]["cpu"],
            json!({"cache_type": "f16", "repack": true}),
            "a small model keeps repacking, with the faster cache"
        );
        let kept = PathBuf::from(config["models"]["minicpm5-1b"]["path"].as_str().unwrap());
        assert!(kept.starts_with(&store));
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(
            std::fs::metadata(&kept).unwrap().permissions().mode() & 0o777,
            0o444
        );
        policy::validate(&config, true).unwrap();

        let speech = add(
            &mut config,
            Keep::Copy(&store),
            &whisper,
            Some("whisper-base"),
            None,
            true,
        )
        .unwrap();
        assert_eq!(config["models"][&speech]["kind"], "speech");
        assert_eq!(
            config["default_model"], "minicpm5-1b",
            "a speech model is never the default"
        );

        let before = config.clone();
        let junk = dir.join("notes.txt");
        std::fs::write(&junk, b"hello").unwrap();
        assert!(add(&mut config, Keep::Copy(&store), &junk, None, None, false).is_err());
        assert!(
            add(
                &mut config,
                Keep::Copy(&store),
                &gguf,
                Some("other"),
                Some(&"0".repeat(64)),
                false
            )
            .is_err()
        );
        assert_eq!(config, before);

        remove(&mut config, "whisper-base").unwrap();
        assert!(config["models"].get("whisper-base").is_none());

        // An installed model is used where it is, only when protected.
        use std::os::unix::fs::MetadataExt as _;
        let system = dir.join("system");
        std::fs::create_dir_all(&system).unwrap();
        let shipped = system.join("MiniCPM5-1B.gguf");
        std::fs::copy(&gguf, &shipped).unwrap();
        let owner = std::fs::metadata(&shipped).unwrap().uid();
        std::fs::set_permissions(&system, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(
            add(
                &mut config,
                Keep::Installed { owner },
                &shipped,
                Some("shipped"),
                None,
                false
            )
            .is_err(),
            "a folder others can write"
        );
        std::fs::set_permissions(&system, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            add(
                &mut config,
                Keep::Installed { owner: owner + 1 },
                &shipped,
                Some("shipped"),
                None,
                false
            )
            .is_err(),
            "another owner"
        );
        let alias = add(
            &mut config,
            Keep::Installed { owner },
            &shipped,
            Some("shipped"),
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            config["models"][&alias]["path"],
            json!(std::fs::canonicalize(&shipped).unwrap())
        );
    }

    /// A runtime folder with the llama.cpp and sherpa-onnx runtimes.
    fn runtimes(dir: &Path) -> Value {
        for runtime in ["cpu/llama-server", ONNX_RUNTIME] {
            let path = dir.join("runtimes").join(runtime);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"#!").unwrap();
        }
        let mut config = fresh_config(&dir.join("runtimes"));
        config["context"] = json!(4608);
        config
    }

    /// A sherpa-onnx model folder: Parakeet's files with its encoder's
    /// export metadata at the end, or Moonshine's.
    fn speech_folder(dir: &Path, architecture: &str) -> PathBuf {
        let folder = dir.join(architecture);
        std::fs::create_dir_all(folder.join("test_wavs")).unwrap();
        let names: &[&str] = if architecture == "moonshine" {
            &[
                "preprocess.onnx",
                "encode.int8.onnx",
                "uncached_decode.int8.onnx",
                "cached_decode.int8.onnx",
            ]
        } else {
            &["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"]
        };
        for name in names {
            let mut bytes = vec![7_u8; 100_000];
            if *name == "encoder.int8.onnx" {
                bytes.extend(b"model_type\x12EncDecRNNTBPEModel\x0aurl\x12https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3");
            }
            std::fs::write(folder.join(name), bytes).unwrap();
        }
        std::fs::write(folder.join("tokens.txt"), b"<unk> 0\n").unwrap();
        std::fs::write(folder.join("test_wavs/en.wav"), b"RIFF").unwrap();
        folder
    }

    /// Parakeet TDT and Moonshine folders are told apart by their files (a
    /// transducer only when its encoder says it is Parakeet TDT, the int8
    /// file preferred); a folder of neither is refused.
    #[test]
    fn speech_folder_layouts_are_identified() {
        let dir = tempfile::tempdir().unwrap();
        let parakeet = speech_folder(dir.path(), "parakeet-tdt");
        std::fs::write(parakeet.join("decoder.onnx"), b"float weights").unwrap();
        assert_eq!(
            folder::identify(&parakeet).unwrap(),
            folder::Layout {
                architecture: "parakeet-tdt",
                files: vec![
                    ("encoder", "encoder.int8.onnx".to_owned()),
                    ("decoder", "decoder.int8.onnx".to_owned()),
                    ("joiner", "joiner.int8.onnx".to_owned()),
                    ("tokens", "tokens.txt".to_owned()),
                ]
            }
        );
        let moonshine = speech_folder(dir.path(), "moonshine");
        let layout = folder::identify(&moonshine).unwrap();
        assert_eq!(layout.architecture, "moonshine");
        assert_eq!(
            layout
                .files
                .iter()
                .map(|(role, _)| *role)
                .collect::<Vec<_>>(),
            [
                "preprocessor",
                "encoder",
                "uncached_decoder",
                "cached_decoder",
                "tokens"
            ]
        );
        assert_eq!(
            identify(&moonshine).unwrap().architecture.as_deref(),
            Some("moonshine")
        );

        let zipformer = speech_folder(dir.path(), "zipformer");
        std::fs::write(zipformer.join("encoder.int8.onnx"), vec![7_u8; 1000]).unwrap();
        assert!(
            folder::identify(&zipformer).is_err(),
            "a transducer that is not Parakeet TDT"
        );
        std::fs::remove_file(moonshine.join("cached_decode.int8.onnx")).unwrap();
        assert!(folder::identify(&moonshine).is_err(), "a part missing");
    }

    /// A folder model is kept as a read-only copy of its layout's files (not
    /// its samples), registered with each file's digest, and chosen for
    /// dictation only when asked; without the sherpa-onnx runtime, or with
    /// another digest, nothing changes; a changed file fails verification.
    #[test]
    fn a_speech_folder_is_kept_and_registered_with_its_digests() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        let mut config = runtimes(dir.path());
        let gguf = dir.path().join("model.gguf");
        gguf_file(&gguf, "MiniCPM5", "minicpm");
        add(&mut config, Keep::Copy(&store), &gguf, None, None, false).unwrap();
        let parakeet = speech_folder(dir.path(), "parakeet-tdt");

        let mut bare = config.clone();
        bare.as_object_mut().unwrap().remove("speech");
        assert!(
            add(&mut bare, Keep::Copy(&store), &parakeet, None, None, false)
                .unwrap_err()
                .contains("enzimos-sherpa-onnx-cpu")
        );
        let before = config.clone();
        assert!(
            add(
                &mut config,
                Keep::Copy(&store),
                &parakeet,
                None,
                Some(&"0".repeat(64)),
                false
            )
            .is_err()
        );
        assert_eq!(config, before);

        let alias = add(
            &mut config,
            Keep::Copy(&store),
            &parakeet,
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(alias, "parakeet-tdt");
        let item = config["models"][&alias].clone();
        assert_eq!(
            (item["kind"].as_str(), item["architecture"].as_str()),
            (Some("speech"), Some("parakeet-tdt"))
        );
        assert_eq!(
            item["files"]["encoder"]["sha256"],
            policy::file_sha256(&parakeet.join("encoder.int8.onnx")).unwrap()
        );
        assert_eq!(
            item["sha256"],
            folder::manifest(item["files"].as_object().unwrap())
        );
        let kept = PathBuf::from(item["path"].as_str().unwrap());
        assert!(kept.starts_with(&store));
        let mut names: Vec<_> = std::fs::read_dir(&kept)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "decoder.int8.onnx",
                "encoder.int8.onnx",
                "joiner.int8.onnx",
                "tokens.txt"
            ]
        );
        assert_eq!(
            std::fs::metadata(kept.join("joiner.int8.onnx"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o444
        );
        assert!(
            config.get("default_speech_model").is_none(),
            "not chosen unless asked"
        );
        policy::validate(&config, true).unwrap();

        let moonshine = speech_folder(dir.path(), "moonshine");
        let alias = add(
            &mut config,
            Keep::Copy(&store),
            &moonshine,
            Some("moonshine-base"),
            None,
            true,
        )
        .unwrap();
        assert_eq!(config["default_speech_model"], json!(alias));
        assert_eq!(config["default_model"], "minicpm5-1b");
        set_default(&mut config, "parakeet-tdt").unwrap();
        assert_eq!(config["default_speech_model"], "parakeet-tdt");
        remove(&mut config, "parakeet-tdt").unwrap();
        assert!(
            config.get("default_speech_model").is_none(),
            "the choice goes with its model"
        );

        let copy = PathBuf::from(config["models"]["moonshine-base"]["path"].as_str().unwrap());
        std::fs::set_permissions(&copy, std::fs::Permissions::from_mode(0o755)).unwrap();
        let tokens = copy.join("tokens.txt");
        std::fs::set_permissions(&tokens, std::fs::Permissions::from_mode(0o644)).unwrap();
        std::fs::write(&tokens, b"<unk> 1\n").unwrap();
        assert_eq!(
            policy::validate(&config, true).unwrap_err(),
            "model digest mismatch"
        );
    }

    /// A shipped folder is used in place only when it, its folder and its
    /// files are the owner's and nobody else can change them.
    #[test]
    fn an_installed_speech_folder_must_be_protected() {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        let dir = tempfile::tempdir().unwrap();
        let mut config = runtimes(dir.path());
        let gguf = dir.path().join("model.gguf");
        gguf_file(&gguf, "MiniCPM5", "minicpm");
        add(
            &mut config,
            Keep::Copy(&dir.path().join("store")),
            &gguf,
            None,
            None,
            false,
        )
        .unwrap();
        let moonshine = speech_folder(dir.path(), "moonshine");
        let owner = std::fs::metadata(&moonshine).unwrap().uid();
        let tokens = moonshine.join("tokens.txt");
        std::fs::set_permissions(&tokens, std::fs::Permissions::from_mode(0o666)).unwrap();
        assert!(
            add(
                &mut config,
                Keep::Installed { owner },
                &moonshine,
                None,
                None,
                false
            )
            .is_err(),
            "a file others can write"
        );
        std::fs::set_permissions(&tokens, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(
            add(
                &mut config,
                Keep::Installed { owner: owner + 1 },
                &moonshine,
                None,
                None,
                false
            )
            .is_err(),
            "another owner"
        );
        let alias = add(
            &mut config,
            Keep::Installed { owner },
            &moonshine,
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            config["models"][&alias]["path"],
            json!(std::fs::canonicalize(&moonshine).unwrap())
        );
    }

    /// `fetch` downloads only catalog models, needs the runtime before it
    /// downloads, and refuses a download whose digest is not the catalog's:
    /// nothing is added or kept, and its scratch folder is gone.
    #[test]
    fn a_fetched_model_with_another_digest_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        let work = store.join(".fetch-moonshine-base");
        let mut config = runtimes(dir.path());
        let gguf = dir.path().join("model.gguf");
        gguf_file(&gguf, "MiniCPM5", "minicpm");
        add(&mut config, Keep::Copy(&store), &gguf, None, None, false).unwrap();
        let before = config.clone();
        let mut asked = Vec::new();
        let error = fetch(
            &mut config,
            &store,
            &work,
            "moonshine-base",
            true,
            |url, file| {
                asked.push(url.to_owned());
                std::fs::write(file, b"not the tarball").map_err(|e| e.to_string())
            },
        )
        .unwrap_err();
        assert!(error.contains("does not match the catalog"), "{error}");
        assert_eq!(asked, [catalog::offer("moonshine-base").unwrap().url]);
        assert_eq!(config, before);
        assert!(!work.exists(), "the download is deleted");
        assert_eq!(
            std::fs::read_dir(&store).unwrap().count(),
            1,
            "only the text model is kept"
        );

        let never = |_: &str, _: &Path| -> Result<(), String> { panic!("nothing is downloaded") };
        assert!(fetch(&mut config, &store, &work, "whisper-huge", false, never).is_err());
        let mut bare = config.clone();
        bare.as_object_mut().unwrap().remove("speech");
        assert!(
            fetch(
                &mut bare,
                &store,
                &work,
                "parakeet-tdt-0.6b-v3",
                false,
                never
            )
            .is_err()
        );
        for offer in &catalog::CATALOG {
            assert!(
                policy::hex64(offer.sha256) && offer.url.starts_with("https://"),
                "{offer:?}"
            );
        }
    }
}
