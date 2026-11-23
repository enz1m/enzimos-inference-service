# enzimOS local inference service

A per-user supervisor for pinned llama.cpp, whisper.cpp and sherpa-onnx runtimes.
Runtime binaries and model weights are packaged separately.
The process executes local models only. Provider routing, account login and cloud
credentials belong to the separate `enzimos-ai-gateway` repository.

Rust binaries: `enzimos-inference` (service), `enzimos-inference-calibrate`,
`enzimos-inference-models` (the account's model registry) and `enzimos-inference-probe`. Private mode-0600 Unix HTTP socket,
verified model registry, lazy loading, one native worker, two admitted jobs maximum (one active, one waiting),
streaming relay, disconnect cancellation, CPU affinity/priority, idle unloading,
RAM monitoring, GPU startup fallback and CPU/Vulkan/SYCL/CUDA runtime slots.
Automatic selection uses exact model/runtime/hardware/context calibration receipts;
unmeasured hardware conservatively uses CPU. Merely finding an Intel/NVIDIA GPU
is insufficient evidence that its backend is faster or fits the memory budget.

The native binary is bundled in an isolated runtime directory. Upstream llama.cpp
and voice model licenses are retained. No PyTorch/Python model stack is required. The worker is launched directly
with CPU affinity, nice 15, no core dumps, a parent-death signal and
no-new-privileges applied before exec.
No models or cloud credentials are included in this repository.

## Run locally

```sh
enzimos-inference --config /path/inference.json \
  --socket /path/private-directory/service.sock
curl --unix-socket /path/private-directory/service.sock http://localhost/status
```

The socket directory must be owned by the user and mode 0700. Engine sockets live
inside it; llama.cpp is never exposed on a TCP port. Do not enable the sample unit
until a verified model registry and qualified runtime paths are configured.

## Adding models

```sh
enzimos-inference-models add ~/Downloads/MiniCPM5-1B-Q4_K_M.gguf   # alias from the GGUF: minicpm5-1b
enzimos-inference-models add ggml-base-q5_1.bin --id whisper-base  # whisper.cpp speech model
enzimos-inference-models list
enzimos-inference-models remove ID
enzimos-inference-models add /usr/share/enzimos/models/X.gguf --system  # a shipped model, used in place
enzimos-inference-models add ~/sherpa-onnx-moonshine-base-en-int8 --id moonshine-base  # a speech model folder
enzimos-inference-models catalog                         # text and speech downloads
enzimos-inference-models fetch parakeet-tdt-0.6b-v3 --default   # download, verify, add, use for dictation
enzimos-inference-models default whisper-base            # the speech (or text) model to use
```

The header is checked (GGUF v2/v3 or whisper.cpp), the alias comes from
`general.name` and `general.size_label`, `--sha256` enforces an expected digest,
and a read-only copy is kept under `~/.local/share/enzimos/models/<sha256>`. The
`--system` registers a model the system installed where it is, without a copy,
only when root owns it and its folder and nobody else can change them. The
registry is created from the installed runtimes when missing and rewritten
atomically only when it still validates. Settings › Agents can import a model
file or a speech model folder. The service reloads the registry on restart.

A speech model folder in the sherpa-onnx layout is identified by its files:
Parakeet TDT (`encoder`, `decoder`, `joiner` ONNX files and `tokens.txt`; the
encoder's export metadata must name a NeMo Parakeet TDT model) or Moonshine
(`preprocess`, `encode`, `uncached_decode`, `cached_decode` and `tokens.txt`).
The int8 file is taken where both exist. Only those files are kept, read-only,
under `~/.local/share/enzimos/models/<digest>/`, each recorded with its SHA-256;
the model's digest (what `--sha256` checks) is SHA-256 over the `role<TAB>sha256`
lines in role order. `--system` uses a root-owned folder in place when it, its
parent and its files are protected. Folder models need the sherpa-onnx runtime
(`enzimos-sherpa-onnx-cpu`, `/usr/lib/enzimos-inference/speech-onnx/`).

`fetch ID` downloads an explicit choice from the [built-in catalog](src/models/catalog.rs)
over HTTPS and verifies its pinned SHA-256 before importing it. The catalog
offers LFM3-VL-2B for text, Qwen3.5 0.8B, Parakeet TDT and Moonshine. LFM3
requires a CPU runtime that declares support for its architecture. Downloads
use private scratch space on the same filesystem as the model store.

`--default` on a speech model, or `default ID`, records `default_speech_model`;
`GET /v1/models` marks it, and the default text model, with `"default": true`,
and dictation uses it. Removing the model removes the choice (dictation then
falls back to `whisper-base`).

## First login

The `enzimos-ai-setup` user unit runs `enzimos-inference-models setup` before
the inference and gateway services start. It validates every entry in
`/usr/share/enzimos/models/models.tsv` before publishing a new model registry.
The models remain in their installed location. Setup also creates the local
gateway configuration when it is absent. Existing configuration files are
preserved. The units use the user manager's XDG configuration directory.

## Request contract

- `GET /health`, `/status`, `/v1/models` do not load a model.
- `POST /v1/chat/completions`: native chat/tool schema, output limit 512 tokens;
  `chat_template_kwargs.enable_thinking=false` is the interactive default.
- `POST /completion`: native prompt/token-array interface, same output limit.
- `POST /tokenize`: tokenization for context budgeting.
- Streaming uses native SSE. One request per HTTP connection, bounded input/output.
- `POST /v1/audio/transcriptions`: **local** JSON `{model, wav_base64, language?}`;
  mono 16-bit PCM WAV at 16 kHz, up to 60 seconds. One serial CPU speech worker per job,
  chosen by the model's architecture: whisper-cli for whisper models,
  `sherpa-onnx-offline` for Parakeet TDT (detects one of its 25 European languages;
  other codes are refused) and Moonshine (English only; another language code is
  refused). Same limits, timeout, confinement and removal of temporary data for both.
  This is not the cloud multipart transcription API and is not transparently routed.
- Unsupported runtime controls and model paths are rejected. MCP tool execution is
  separate and remains subject to the central broker’s consent and capabilities.

The default context of 4,608 tokens leaves room for a 4,096-token input and 512
output tokens. The runtime uses one slot, a batch of 256 and a microbatch of
128. The cache format can vary with the selected model and backend.
Conversation-scoped `conversation_id` permits in-memory prefix reuse only on
consecutive requests to the same model/session; unrelated requests reset it.
No session transcript or disk prompt cache is written.

The default inference memory budget is 1,900 MiB. The user unit also sets
`MemoryHigh=2400M`, `MemoryMax=2688M`, `MemorySwapMax=0`, `CPUQuota=200%`,
`CPUWeight=20` and `Nice=15`. Workers use at most two threads. GPU allocations need separate driver accounting and installed cgroup tests;
RSS alone is not a total-memory guarantee. Unqualified GPU calibration is refused.

## Build and test

Requires Linux and Rust 1.95 or newer.

```sh
cargo build --release --locked
cargo build --example fake_engine --locked
cargo test --locked
cargo clippy --all-targets --locked
cargo fmt --check
```

Transport tests drive the actual supervisor through a private socket with a small
fixture runtime. They do not download models or establish native CPU/GPU model
quality, speed or memory qualification. Test installed runtimes and cgroup policy
in a disposable guest before deploying the sample units.

## Opt-in CPU calibration

`enzimos-inference-calibrate` measures a registered local text model using an
actual4096-token prompt and up to32 generated tokens, with no network downloads.
It creates a transient user scope (2GiB memory, no swap,64 tasks, one CPU quota),
uses at most two worker CPUs at low priority, and refuses an unbounded fallback.
Existing inference services and their jobs/configuration remain untouched.

The input config must be an owned mode0600 regular file. Runtime revision and
artifact SHA256 are required; model/runtime changes invalidate the measurement.
Create an owned mode0700 output directory, then explicitly authorize the run:

```sh
enzimos-inference-calibrate --run --config /private/inference.json \
  --model my-model --receipt /private/calibration.json \
  --output-config /private/inference-calibrated.json
```

Outputs are new private files; existing files are never overwritten. The optional
new config enrolls the CPU receipt. Apply it explicitly in the service/settings
workflow; calibration does not enable or restart services. It saves measurements
and hashes, never model text or prompts. Memory is conservatively the maximum of
the scope memory peak and sampled native-plus-supervisor RSS. GPU profiling remains
unqualified without independent driver/correctness accounting. The CPU correctness
flag is a native response/full-context smoke gate, not semantic model quality.

## Typed preference control

`GET /v1/preferences` returns schema1, a content-derived `revision`, `persistent`
and redacted `preferences`. `POST /v1/preferences` accepts exactly
`{"schema_version":1,"revision":"<read revision>","preferences":{...}}`.
Unknown fields, unsupported values and unavailable providers/runtimes are refused.
HTTP409 means active work or a stale revision; re-read and review before retrying.
Preferences never interrupt an admitted job. An idle inference worker is unloaded
when its configuration changes; the next request uses the new resource policy.
Gateway selection never sends a request or falls back to another provider.

Inference supports `cpu_threads` (null,1,2), `memory_mib` (256–2048),
`idle_seconds` (1–3600), `context` (4096–8192), installed `backend` or `auto`, and a
registered text `default_model`. Gateway supports only `default_provider`, selected
from already configured, enabled chat providers. This API cannot add runtimes,
model paths, credentials, endpoints or an OAuth registration.

The optional `--preferences FILE` enables persistent mutations; its existing parent
must be owned mode0700. Files are owned mode0600, regular, single-link and opened
without following symlinks. Writes use an exclusive lock, atomic rename and fsync;
external edits cause a conflict until restart. Without this option the read API
works but mutation is unavailable. Sample units use a private `StateDirectory`
and `%S/<service>/preferences.json`; configuration/credentials stay read-only.

Settings must use its existing broker consent and call-liveness checks before
forwarding agent changes. The socket authenticates the account UID, not a specific
application: this API is not a replacement for sandbox admission or MCP consent.
Reads do not reveal model paths, credentials or endpoint URLs and do not load models.
No service is enabled or provisioned with models/accounts by this interface.

`GET /status` lists `installed_runtimes` as at most four backend identifiers from
the validated service configuration. This exposes installed choices for Settings,
not hardware compatibility or performance qualification. It does not probe devices,
start inference or expose executable paths. `auto` remains a preference, rather
than an installed runtime. Explicit selection still uses the existing resource
limits and error handling; calibration is required for automatic GPU selection.

## License

MIT; see [LICENSE](LICENSE). External runtimes and models retain their own licenses.
