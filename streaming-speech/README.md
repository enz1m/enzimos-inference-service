# Streaming speech

The resident process shares a single one-thread sherpa-onnx recognizer across
the desktop's capture clients. Each take has its own decoder stream and
Silero VAD. It uses Kroko Community 64-L English without vocabulary hints.
There is no second recognizer. Audio is read from PipeWire in memory.

Apply sherpa-onnx-v2.0.3-alternatives.patch to sherpa-onnx v2.0.3 before
building the C API library and worker.c. Keep the shared runtime,
encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt and silero_vad.onnx beside
the executable. The unit uses that directory as its working directory. Kroko
Community assets are distributed separately under CC BY-SA by Banafo. Silero
VAD is MIT and sherpa-onnx is Apache-2.0.

The user-owned Unix socket accepts a two-byte header. Send ASCII m for manual
or a for automatic, followed by one unsigned byte with the maximum speech
duration, from 1 to 60 seconds. Send a to enable automatic endpointing in an
open take, or f to finish. Closing the socket cancels capture. Replies are UTF-8
lines containing an event name, a tab and optional text. The terminal event is
final or error. Partial events are for display and only final text is committed.
The level event gives the PCM peak (0–32768) over the last 64 ms.
The transcribing event marks the start of finalization.

When ready advertises alternatives-v1, a client can send n during capture.
The worker then sends an alternatives event before final, containing JSON
pairs of text and length-normalized decoder log score from the same beam.
Without n the worker sends the usual transcript events. Voice Typing uses
final directly.

Assistant can run `context-rescore.cc` as `enzimos-speech-context` before a
voice request. Build it as C++20 against the same ONNX Runtime that is
installed beside the speech worker. It reads the transcript and one preceding
utterance on two lines of stdin and writes one proposed transcript. This
short-lived process masks one word at a time and combines contextual token
scores with pronunciation distance. Assistant uses close Kroko alternatives
to decide whether it can apply the proposed word or needs transcript review.

The executable loads `context-model.onnx` and `context-vocab.txt` from its own
directory. These are `onnx/model_quantized.onnx` and `vocab.txt` from
`Xenova/distilbert-base-uncased@5d73105e39e322b779ff7a8fcd11530fa579165b`.
For `context-phones.txt`, join lowercase alphabetic vocabulary entries to
CMUdict revision `74790861f652b15e4ac49015a90074ad62a27690`. Strip alternate
pronunciation suffixes from words and stress digits from phones. Write one
line per pronunciation with the zero-based vocabulary ID followed by its
space-separated phones. Retain CMUdict's license beside the derived table.
