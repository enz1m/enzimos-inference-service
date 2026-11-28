# Streaming speech

The resident process shares a single one-thread sherpa-onnx recognizer across
capture clients. Each take has its own decoder stream and Silero VAD.
Recognition uses Kroko Community 64-L English. The worker reads PipeWire audio
in memory.

Build with the sherpa-onnx v2.0.3 shared C API. With the header in `include/`
and the library in `lib/`, compile the worker from this directory:

```sh
cc -std=c11 -O2 -Wall -Wextra -pthread -I include worker.c -L lib \
  -Wl,-rpath,'$ORIGIN' -lsherpa-onnx-c-api -lsystemd -o enzimos-streaming-speech
```

On Arch, build the package with `makepkg` from this directory. It includes
the pinned English model and starts the user service with the Shell session.
For a manual installation, install the executable, the shared C API and its runtime libraries, and
`encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`, and
`silero_vad.onnx` in `/usr/lib/enzimos-inference/streaming-speech`. The unit
uses that directory as its working directory. Kroko Community assets are
distributed separately under CC BY-SA by Banafo. Silero VAD is MIT and
sherpa-onnx is Apache-2.0. Install the unit in the user unit directory and
enable it with `systemctl --user enable --now enzimos-streaming-speech.service`
in the desktop account.

The private user socket accepts a two-byte header. Send ASCII `m` for manual
or `a` for automatic, followed by one unsigned byte with the maximum speech
duration from 1 to 60 seconds. Send `a` to enable automatic endpointing in an
open take or `f` to finish. Closing the socket cancels capture.

Replies are UTF-8 lines containing an event name, a tab and an optional
payload. After `ready`, `level` reports the PCM peak (0–32768) over the last
64 ms, and `partial` reports tentative text. `transcribing` announces
finalization after capture. The terminal event is `final` with the completed
text or `error` with the reason.
