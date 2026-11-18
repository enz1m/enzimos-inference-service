//! Confined child spawning and native memory observations.

use super::*;

/// Applied in the child before exec: limits exist before native threads start.
#[allow(
    unsafe_code,
    reason = "Linux syscall ABI (CPU_SET, getppid, nice, prctl and others)."
)]
fn harden(cpus: Vec<u32>, parent: i32) -> impl FnMut() -> std::io::Result<()> {
    move || unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for cpu in &cpus {
            libc::CPU_SET(*cpu as usize, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        libc::nice(15);
        libc::umask(0o077);
        let zero = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        libc::setrlimit(libc::RLIMIT_CORE, &zero);
        // GPU drivers reserve large virtual ranges; RSS and cgroup limits enforce RAM.
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0);
        if libc::getppid() != parent {
            return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
        }
        libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
        Ok(())
    }
}

#[allow(unsafe_code, reason = "Fork-child setup must run before exec.")]
pub(super) fn spawn(
    args: &[String],
    env: &[(String, String)],
    cpus: Vec<u32>,
    output: Option<&Path>,
) -> std::io::Result<Child> {
    let stdout = match output {
        Some(path) => {
            use std::os::unix::fs::OpenOptionsExt as _;
            Stdio::from(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(path)?,
            )
        }
        None => Stdio::null(),
    };
    let mut command = Command::new(&args[0]);
    command
        .args(&args[1..])
        .env_clear()
        .envs(env.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(Stdio::null())
        .kill_on_drop(true);
    unsafe { command.pre_exec(harden(cpus, libc::getpid())) };
    command.spawn()
}

pub(super) fn rss_mib(pid: u32) -> f64 {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map_or(0.0, |kib| kib / 1024.0)
}

/// GPU memory the driver attributes to the worker's DRM clients.
pub(super) fn driver_ram_mib(pid: u32) -> f64 {
    let mut clients = std::collections::HashMap::new();
    for entry in std::fs::read_dir(format!("/proc/{pid}/fdinfo"))
        .into_iter()
        .flatten()
        .flatten()
    {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let rows = text
            .lines()
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.to_owned(), v.trim().to_owned()))
            .collect::<std::collections::HashMap<_, _>>();
        let Some(client) = rows.get("drm-client-id") else {
            continue;
        };
        let total = rows
            .iter()
            .filter(|(k, _)| {
                [
                    "drm-resident-system",
                    "drm-resident-stolen-system",
                    "drm-resident-gtt",
                ]
                .iter()
                .any(|p| k.starts_with(p))
            })
            .map(|(_, v)| {
                let mut parts = v.split_whitespace();
                let n = parts
                    .next()
                    .and_then(|n| n.parse::<f64>().ok())
                    .unwrap_or(0.0);
                if parts.next() == Some("KiB") {
                    n * 1024.0
                } else {
                    n
                }
            })
            .sum::<f64>();
        clients.insert((rows.get("drm-pdev").cloned(), client.clone()), total);
    }
    clients.values().sum::<f64>() / (1024.0 * 1024.0)
}
