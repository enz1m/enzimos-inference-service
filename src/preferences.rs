// SPDX-License-Identifier: MIT
//! Private, revision-bound preference storage for a local service.
use serde_json::{Map, Value, json};
use std::{
    ffi::CString,
    fs::File,
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The caller's revision no longer names the active preferences.
    Conflict(String),
    Invalid(String),
}

fn invalid<T>(message: &str) -> Result<T, Error> {
    Err(Error::Invalid(message.into()))
}

pub type Validate =
    Box<dyn Fn(Map<String, Value>) -> Result<Map<String, Value>, String> + Send + Sync>;

pub fn encoded(value: &Map<String, Value>) -> Vec<u8> {
    enzimos_platform::json::canonical(&Value::Object(value.clone())).unwrap_or_default()
}

pub fn digest(value: &Map<String, Value>) -> String {
    enzimos_platform::json::sha256(&encoded(value))
}

#[allow(unsafe_code, reason = "Linux syscall ABI (openat).")]
fn open_at(directory: &File, name: &str, flags: i32, mode: u32) -> std::io::Result<File> {
    let name =
        CString::new(name).map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_CLOEXEC,
            mode,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[allow(unsafe_code, reason = "Linux syscall ABI (getuid).")]
fn private_file(file: &File) -> Result<(), Error> {
    let m = file
        .metadata()
        .map_err(|_| Error::Invalid("preference file unavailable".into()))?;
    if !m.is_file()
        || m.uid() != unsafe { libc::getuid() }
        || m.mode() & 0o077 != 0
        || m.nlink() != 1
    {
        return invalid("private unlinked-to-other-paths preference file required");
    }
    Ok(())
}

pub struct Preferences {
    path: Option<PathBuf>,
    validate: Validate,
    pub value: Map<String, Value>,
    disk: Option<Vec<u8>>,
    pub revision: String,
}

impl Preferences {
    pub fn new(
        defaults: Map<String, Value>,
        path: Option<&Path>,
        validate: Validate,
    ) -> Result<Self, Error> {
        let mut value = validate(defaults.clone()).map_err(Error::Invalid)?;
        let mut preferences = Self {
            path: path.map(Path::to_path_buf),
            validate,
            value: Map::new(),
            disk: None,
            revision: String::new(),
        };
        if preferences.path.is_some() {
            let directory = preferences.directory()?;
            preferences.disk = preferences.read(&directory)?;
            if let Some(disk) = &preferences.disk {
                let stored: Map<String, Value> = serde_json::from_slice(disk)
                    .map_err(|_| Error::Invalid("preference object required".into()))?;
                let mut merged = defaults;
                merged.extend(stored.clone());
                value = match (preferences.validate)(merged) {
                    Ok(value) => value,
                    // A stored value the current configuration no longer
                    // accepts (a renamed model, a removed GPU runtime) falls
                    // back to its default field by field instead of keeping
                    // the service down. The file stays as written so later
                    // commits still detect outside edits.
                    Err(e) => {
                        eprintln!("enzimos-inference: stored preferences partly ignored: {e}");
                        let mut kept = value;
                        for (key, stored) in stored {
                            let mut candidate = kept.clone();
                            candidate.insert(key, stored);
                            if let Ok(candidate) = (preferences.validate)(candidate) {
                                kept = candidate;
                            }
                        }
                        kept
                    }
                };
            }
        }
        preferences.revision = digest(&value);
        preferences.value = value;
        Ok(preferences)
    }

    fn name(&self) -> &str {
        self.path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or_default()
    }

    #[allow(unsafe_code, reason = "Linux syscall ABI (getuid).")]
    fn directory(&self) -> Result<File, Error> {
        let parent = self
            .path
            .as_ref()
            .and_then(|p| p.parent())
            .ok_or_else(|| Error::Invalid("preference path required".into()))?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags_directory()
            .open(parent)
            .map_err(|_| Error::Invalid("private owned preference directory required".into()))?;
        let m = file
            .metadata()
            .map_err(|_| Error::Invalid("private owned preference directory required".into()))?;
        if !m.is_dir() || m.uid() != unsafe { libc::getuid() } || m.mode() & 0o077 != 0 {
            return invalid("private owned preference directory required");
        }
        Ok(file)
    }

    fn read(&self, directory: &File) -> Result<Option<Vec<u8>>, Error> {
        let mut file = match open_at(
            directory,
            self.name(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            0,
        ) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return invalid("private unlinked-to-other-paths preference file required"),
        };
        private_file(&file)?;
        let mut data = Vec::new();
        (&mut file)
            .take(16385)
            .read_to_end(&mut data)
            .map_err(|_| Error::Invalid("preferences unavailable".into()))?;
        if data.len() > 16384 {
            return invalid("preferences exceed size limit");
        }
        match serde_json::from_slice::<Value>(&data) {
            Ok(Value::Object(_)) => Ok(Some(data)),
            _ => invalid("preference object required"),
        }
    }

    pub fn snapshot(&self) -> Value {
        json!({"schema_version": 1, "revision": self.revision, "persistent": self.path.is_some(), "preferences": self.value})
    }

    #[allow(unsafe_code, reason = "Linux syscall ABI (flock, renameat, unlinkat).")]
    pub fn commit(&mut self, body: &Map<String, Value>) -> Result<Value, Error> {
        let fields = ["schema_version", "revision", "preferences"];
        if body.len() != 3
            || !fields.iter().all(|f| body.contains_key(*f))
            || !body["schema_version"].is_u64()
            || body["schema_version"] != 1
        {
            return invalid("versioned preference mutation required");
        }
        if body["revision"].as_str() != Some(self.revision.as_str()) {
            return Err(Error::Conflict("preference revision changed".into()));
        }
        let changes = body["preferences"]
            .as_object()
            .filter(|c| !c.is_empty() && c.keys().all(|k| self.value.contains_key(k)));
        let Some(changes) = changes else {
            return invalid("unknown or empty preference mutation");
        };
        let mut merged = self.value.clone();
        merged.extend(changes.clone());
        let value = (self.validate)(merged).map_err(Error::Invalid)?;
        if self.path.is_none() {
            return invalid("persistent preference control is not configured");
        }
        let directory = self.directory()?;
        let lock = open_at(
            &directory,
            &format!("{}.lock", self.name()),
            libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            0o600,
        )
        .map_err(|_| Error::Invalid("preference lock unavailable".into()))?;
        private_file(&lock)?;
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(Error::Conflict("preferences are being changed".into()));
        }
        if self.read(&directory)? != self.disk {
            return Err(Error::Conflict(
                "stored preferences changed; restart and review again".into(),
            ));
        }
        let data = encoded(&value);
        let temporary = format!(
            ".{}.{}",
            self.name(),
            enzimos_platform::files::random_hex(12)
                .map_err(|_| Error::Invalid("randomness unavailable".into()))?
        );
        let written = (|| -> std::io::Result<()> {
            let mut file = open_at(
                &directory,
                &temporary,
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW,
                0o600,
            )?;
            file.write_all(&data)?;
            file.sync_all()?;
            let (from, to) = (
                CString::new(temporary.as_str()).unwrap(),
                CString::new(self.name()).unwrap(),
            );
            if unsafe {
                libc::renameat(
                    directory.as_raw_fd(),
                    from.as_ptr(),
                    directory.as_raw_fd(),
                    to.as_ptr(),
                )
            } != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            directory.sync_all()
        })();
        if written.is_err() {
            let name = CString::new(temporary.as_str()).unwrap();
            unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) };
            return invalid("preferences could not be saved");
        }
        self.disk = Some(data);
        self.revision = digest(&value);
        self.value = value;
        Ok(self.snapshot())
    }
}

trait DirectoryFlags {
    fn custom_flags_directory(&mut self) -> &mut Self;
}
impl DirectoryFlags for std::fs::OpenOptions {
    fn custom_flags_directory(&mut self) -> &mut Self {
        use std::os::unix::fs::OpenOptionsExt;
        self.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    pub(crate) fn private_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        dir
    }
    fn any() -> Validate {
        Box::new(|v| {
            if v.get("x").is_some_and(serde_json::Value::is_string) {
                Ok(v)
            } else {
                Err("x must be text".into())
            }
        })
    }
    fn defaults() -> Map<String, Value> {
        json!({"x": "a"}).as_object().unwrap().clone()
    }

    #[test]
    fn restart_external_edit_and_stale_revision() {
        let dir = private_dir();
        let path = dir.path().join("prefs.json");
        let mut p = Preferences::new(defaults(), Some(&path), any()).unwrap();
        let revision = p.revision.clone();
        let body = json!({"schema_version": 1, "revision": revision, "preferences": {"x": "b"}});
        let snapshot = p.commit(body.as_object().unwrap()).unwrap();
        assert_eq!(snapshot["preferences"]["x"], "b");
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert_eq!(
            p.commit(body.as_object().unwrap()).unwrap_err(),
            Error::Conflict("preference revision changed".into())
        );
        let reloaded = Preferences::new(defaults(), Some(&path), any()).unwrap();
        assert_eq!(
            (reloaded.value["x"].as_str(), reloaded.revision.as_str()),
            (Some("b"), p.revision.as_str())
        );
        std::fs::write(&path, br#"{"x":"external"}"#).unwrap();
        let body = json!({"schema_version": 1, "revision": p.revision, "preferences": {"x": "c"}});
        assert!(matches!(
            p.commit(body.as_object().unwrap()),
            Err(Error::Conflict(_))
        ));
    }

    #[test]
    fn links_public_directories_unknown_keys_and_nonpersistent_refused() {
        let dir = private_dir();
        std::fs::write(dir.path().join("real.json"), "{}").unwrap();
        std::os::unix::fs::symlink(dir.path().join("real.json"), dir.path().join("prefs.json"))
            .unwrap();
        assert!(Preferences::new(defaults(), Some(&dir.path().join("prefs.json")), any()).is_err());
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(Preferences::new(defaults(), Some(&dir.path().join("other.json")), any()).is_err());
        let mut p = Preferences::new(defaults(), None, any()).unwrap();
        let unknown = json!({"schema_version": 1, "revision": p.revision, "preferences": {"y": 1}});
        assert!(matches!(
            p.commit(unknown.as_object().unwrap()),
            Err(Error::Invalid(_))
        ));
        let ok = json!({"schema_version": 1, "revision": p.revision, "preferences": {"x": "b"}});
        assert_eq!(
            p.commit(ok.as_object().unwrap()),
            Err(Error::Invalid(
                "persistent preference control is not configured".into()
            ))
        );
    }
}
