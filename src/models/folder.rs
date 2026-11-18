// SPDX-License-Identifier: MIT
//! Speech models that are folders of ONNX files run by sherpa-onnx: a NeMo
//! Parakeet TDT transducer (encoder, decoder, joiner, tokens) or Moonshine
//! (preprocessor, encoder, uncached and cached decoders, tokens).

use crate::policy;
use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};

/// The parts of each folder layout: the role sherpa-onnx gives a file and
/// the file's stem (`encoder` is `encoder.int8.onnx` or `encoder.onnx`).
const PARAKEET: [(&str, &str); 4] = [
    ("encoder", "encoder"),
    ("decoder", "decoder"),
    ("joiner", "joiner"),
    ("tokens", "tokens"),
];
const MOONSHINE: [(&str, &str); 5] = [
    ("preprocessor", "preprocess"),
    ("encoder", "encode"),
    ("uncached_decoder", "uncached_decode"),
    ("cached_decoder", "cached_decode"),
    ("tokens", "tokens"),
];

/// The architectures a speech model folder can have.
pub const ARCHITECTURES: [&str; 2] = ["parakeet-tdt", "moonshine"];

/// The roles a folder of `architecture` must provide.
pub fn roles(architecture: &str) -> Option<Vec<&'static str>> {
    match architecture {
        "parakeet-tdt" => Some(PARAKEET.iter().map(|(role, _)| *role).collect()),
        "moonshine" => Some(MOONSHINE.iter().map(|(role, _)| *role).collect()),
        _ => None,
    }
}

/// A speech model folder: its architecture and the file that plays each role.
#[derive(Debug, PartialEq, Eq)]
pub struct Layout {
    /// `parakeet-tdt` or `moonshine`.
    pub architecture: &'static str,
    /// Role and file name inside the folder, in the layout's order.
    pub files: Vec<(&'static str, String)>,
}

/// The file in `dir` for `stem`: the int8 ONNX file if there is one, else
/// the plain one; `tokens.txt` for the tokens.
fn part(dir: &Path, stem: &str) -> Option<String> {
    let names: &[String] = if stem == "tokens" {
        &["tokens.txt".to_owned()]
    } else {
        &[format!("{stem}.int8.onnx"), format!("{stem}.onnx")]
    };
    names
        .iter()
        .find(|name| std::fs::symlink_metadata(dir.join(name)).is_ok_and(|m| m.is_file()))
        .cloned()
}

fn parts(dir: &Path, layout: &[(&'static str, &str)]) -> Option<Vec<(&'static str, String)>> {
    layout
        .iter()
        .map(|(role, stem)| part(dir, stem).map(|name| (*role, name)))
        .collect()
}

/// Whether a transducer encoder is a NeMo Parakeet TDT export: sherpa-onnx's
/// export writes its metadata (`model_type`, the model's URL) after the
/// weights, at the end of the file.
fn parakeet_tdt(encoder: &Path) -> Result<bool, String> {
    let mut file = std::fs::File::open(encoder).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    file.seek(SeekFrom::Start(length.saturating_sub(64 * 1024)))
        .map_err(|error| error.to_string())?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)
        .map_err(|error| error.to_string())?;
    let has = |needle: &[u8]| tail.windows(needle.len()).any(|w| w == needle);
    Ok(has(b"EncDecRNNTBPEModel") && has(b"parakeet-tdt"))
}

/// Reads a folder's layout.
///
/// # Errors
/// The folder holds neither layout, or its transducer is not Parakeet TDT.
pub fn identify(dir: &Path) -> Result<Layout, String> {
    if let Some(files) = parts(dir, &MOONSHINE) {
        return Ok(Layout {
            architecture: "moonshine",
            files,
        });
    }
    if let Some(files) = parts(dir, &PARAKEET) {
        if !parakeet_tdt(&dir.join(&files[0].1))? {
            return Err("only NeMo Parakeet TDT transducers are supported".to_owned());
        }
        return Ok(Layout {
            architecture: "parakeet-tdt",
            files,
        });
    }
    Err(
        "not a speech model folder: Parakeet TDT (encoder, decoder, joiner, tokens.txt) \
         or Moonshine (preprocess, encode, uncached_decode, cached_decode, tokens.txt)"
            .to_owned(),
    )
}

/// The digest that names a folder model: SHA-256 over its `role\tsha256`
/// lines in role order, so it changes with any file.
pub fn manifest(files: &Map<String, Value>) -> String {
    let mut roles: Vec<&String> = files.keys().collect();
    roles.sort();
    let mut digest = Sha256::new();
    for role in roles {
        digest.update(format!(
            "{role}\t{}\n",
            files[role]["sha256"].as_str().unwrap_or_default()
        ));
    }
    format!("{:x}", digest.finalize())
}

/// The registry's `files` for a layout in `dir`: each role's name and digest.
///
/// # Errors
/// A file cannot be read.
pub fn digests(dir: &Path, layout: &Layout) -> Result<Map<String, Value>, String> {
    let mut files = Map::new();
    for (role, name) in &layout.files {
        let path = dir.join(name);
        let sha =
            policy::file_sha256(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        files.insert((*role).to_owned(), json!({"name": name, "sha256": sha}));
    }
    Ok(files)
}

/// Copies the layout's files into `store/<manifest>`, read-only, unless an
/// earlier add already did. Returns the folder.
///
/// # Errors
/// The store cannot be written.
pub fn keep_copy(
    store: &Path,
    source: &Path,
    layout: &Layout,
    manifest: &str,
) -> Result<PathBuf, String> {
    let kept = store.join(manifest);
    if kept.is_dir() {
        return Ok(kept);
    }
    let partial = store.join(format!(".{manifest}.partial"));
    if partial.exists() {
        set_tree_writable(&partial);
        std::fs::remove_dir_all(&partial).map_err(|error| error.to_string())?;
    }
    std::fs::create_dir_all(&partial).map_err(|error| error.to_string())?;
    for (_, name) in &layout.files {
        let target = partial.join(name);
        std::fs::copy(source.join(name), &target)
            .map_err(|error| format!("copying the model: {error}"))?;
        super::set_mode(&target, 0o444)?;
    }
    super::set_mode(&partial, 0o555)?;
    std::fs::rename(&partial, &kept).map_err(|error| error.to_string())?;
    Ok(kept)
}

fn set_tree_writable(dir: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
}

/// A system-installed folder, used in place: it, its parent and each of
/// its files must be owned by `owner` and writable by no one else.
///
/// # Errors
/// Something in it is not protected.
pub fn installed(source: &Path, layout: &Layout, owner: u32) -> Result<PathBuf, String> {
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
    if !protected(&path, false)
        || !protected(parent, false)
        || !layout
            .files
            .iter()
            .all(|(_, name)| protected(&path.join(name), true))
    {
        return Err("an installed model must be a system folder nobody else can change".to_owned());
    }
    Ok(path)
}

/// Checks a registered folder model: a known architecture, every role a
/// plain regular file in the folder with a digest, the manifest matching,
/// and with `verify` each file's contents.
///
/// # Errors
/// Says what is wrong.
pub fn validate(item: &Value, verify: bool) -> Result<(), String> {
    let dir = Path::new(item["path"].as_str().unwrap_or_default());
    if !dir.is_absolute() || !std::fs::symlink_metadata(dir).is_ok_and(|m| m.is_dir()) {
        return Err("invalid model path".into());
    }
    let roles = item["architecture"]
        .as_str()
        .and_then(roles)
        .ok_or("unsupported speech model architecture")?;
    let files = item["files"].as_object().ok_or("model files required")?;
    if files.len() != roles.len() || !roles.iter().all(|role| files.contains_key(*role)) {
        return Err("model files do not match the architecture".into());
    }
    for file in files.values() {
        let name = file["name"].as_str().unwrap_or_default();
        let sha = file["sha256"].as_str().unwrap_or_default();
        let plain = !name.is_empty() && !name.contains('/') && name != "." && name != "..";
        let path = dir.join(name);
        if !plain || !std::fs::symlink_metadata(&path).is_ok_and(|m| m.is_file()) {
            return Err("invalid model path".into());
        }
        if !policy::hex64(sha) {
            return Err("model digest required".into());
        }
        if verify && !policy::digest_matches(&path, sha).map_err(|_| "invalid model path")? {
            return Err("model digest mismatch".into());
        }
    }
    if item["sha256"].as_str() != Some(manifest(files).as_str()) {
        return Err("model digest mismatch".into());
    }
    Ok(())
}

/// One speech model the catalog offers to fetch: a sherpa-onnx release
/// tarball, pinned by the digest it was verified with.
#[derive(Debug)]
pub struct Offer {
    /// The registry id it gets.
    pub id: &'static str,
    /// What Settings shows.
    pub title: &'static str,
    /// `parakeet-tdt` or `moonshine`.
    pub architecture: &'static str,
    /// The languages it understands.
    pub languages: &'static str,
    /// The tarball (tar.bz2).
    pub url: &'static str,
    /// The tarball's SHA-256.
    pub sha256: &'static str,
    /// The tarball's size in bytes.
    pub bytes: u64,
    /// The model folder inside the tarball.
    pub folder: &'static str,
}

/// The speech models `enzimos-inference-models fetch` can download.
pub const CATALOG: [Offer; 2] = [
    Offer {
        id: "parakeet-tdt-0.6b-v3",
        title: "Parakeet TDT 0.6B v3 (int8)",
        architecture: "parakeet-tdt",
        languages: "25 European languages, detected automatically",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        sha256: "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
        bytes: 487_170_055,
        folder: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    },
    Offer {
        id: "moonshine-base",
        title: "Moonshine base (int8)",
        architecture: "moonshine",
        languages: "English",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2",
        sha256: "21870cecaa2e44e4e2bf63e02d1072bed183ccd10284871353bd9d24dad14e5e",
        bytes: 250_807_309,
        folder: "sherpa-onnx-moonshine-base-en-int8",
    },
];

/// The catalog entry `id`.
pub fn offer(id: &str) -> Option<&'static Offer> {
    CATALOG.iter().find(|offer| offer.id == id)
}

/// Unpacks a tar.bz2 `archive` into `into` with the system `tar`.
///
/// # Errors
/// `tar` fails.
pub fn unpack(archive: &Path, into: &Path) -> Result<(), String> {
    let status = std::process::Command::new("tar")
        .arg("-xjf")
        .arg(archive)
        .arg("-C")
        .arg(into)
        .args(["--no-same-owner", "--no-same-permissions"])
        .stdin(std::process::Stdio::null())
        .status()
        .map_err(|error| format!("tar: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("the download could not be unpacked".to_owned())
    }
}
