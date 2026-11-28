/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "c-api.h"

enum { RATE = 16000, FRAME = 512, PREROLL = 16, MAX_CLIENTS = 3, MAX_TEXT = 4096 };
extern char **environ;

static const SherpaOnnxOnlineRecognizer *recognizer;
static SherpaOnnxVadModelConfig vad_config;
static pthread_mutex_t decoder_lock = PTHREAD_MUTEX_INITIALIZER;
static atomic_int clients;

static int send_event(int fd, const char *kind, const char *text) {
    char message[MAX_TEXT + 24];
    size_t n = strlen(kind);
    memcpy(message, kind, n);
    message[n++] = '\t';
    for (size_t i = 0; text[i] && i < MAX_TEXT; ++i) {
        unsigned char c = (unsigned char)text[i];
        message[n++] = c < 32 || c == 127 ? ' ' : (char)c;
    }
    message[n++] = '\n';
    for (size_t sent = 0; sent < n;) {
        ssize_t count = send(fd, message + sent, n - sent, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return -1;
        sent += (size_t)count;
    }
    return 0;
}

static int record_audio(int *read_end, pid_t *child) {
    int pipes[2];
    if (pipe2(pipes, O_CLOEXEC) != 0) return -1;
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error != 0) {
        close(pipes[0]);
        close(pipes[1]);
        return -1;
    }
    if (!(error = posix_spawn_file_actions_addclose(&actions, pipes[0])) &&
        !(error = posix_spawn_file_actions_adddup2(&actions, pipes[1], STDOUT_FILENO)) &&
        !(error = posix_spawn_file_actions_addclose(&actions, pipes[1]))) {
        char *argv[] = {"/usr/bin/pw-record", "--raw", "--rate", "16000", "--channels", "1",
                        "--format", "s16", "--latency", "60ms", "--media-role", "Communication",
                        "--properties", "{ node.name = enzimos-speech-capture }", "-", NULL};
        error = posix_spawn(child, argv[0], &actions, NULL, argv, environ);
    }
    posix_spawn_file_actions_destroy(&actions);
    close(pipes[1]);
    if (error != 0) {
        close(pipes[0]);
        return -1;
    }
    *read_end = pipes[0];
    return 0;
}

static void decode(const SherpaOnnxOnlineStream *stream) {
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream))
        SherpaOnnxDecodeOnlineStream(recognizer, stream);
}

static void feed(const SherpaOnnxOnlineStream *stream, const float *frame) {
    SherpaOnnxOnlineStreamAcceptWaveform(stream, RATE, frame, FRAME);
    decode(stream);
}

static void take(int fd) {
    unsigned char header[2];
    if (recv(fd, header, sizeof(header), MSG_WAITALL) != sizeof(header)) return;
    if ((header[0] != 'a' && header[0] != 'm') || header[1] == 0 || header[1] > 60) {
        send_event(fd, "error", "Invalid capture request");
        return;
    }

    SherpaOnnxVadModelConfig take_config = vad_config;
    take_config.silero_vad.max_speech_duration = header[1];
    const SherpaOnnxVoiceActivityDetector *vad =
        SherpaOnnxCreateVoiceActivityDetector(&take_config, header[1] + 2);
    pthread_mutex_lock(&decoder_lock);
    const SherpaOnnxOnlineStream *stream = SherpaOnnxCreateOnlineStream(recognizer);
    pthread_mutex_unlock(&decoder_lock);

    int mic = -1;
    pid_t recorder = -1;
    int automatic = header[0] == 'a';
    int max_frames = header[1] * RATE / FRAME;
    int total = 0, voiced = 0, finishing = 0, result = 0;
    int want_alternatives = 0;
    int started = 0, peak = 0;
    float prior[PREROLL][FRAME];
    unsigned char pcm[FRAME * 2];
    size_t filled = 0;
    char previous[MAX_TEXT + 1] = {0};
    const char *error = NULL;
    if (!vad || !stream || record_audio(&mic, &recorder) != 0) {
        error = "Microphone capture could not start";
        goto done;
    }
    if (send_event(fd, "ready", "alternatives-v1") != 0) goto done;
    while (total < (automatic && !started ? 60 * RATE / FRAME : max_frames + started)) {
        struct pollfd watched[] = {{fd, POLLIN, 0}, {mic, POLLIN, 0}};
        int n = poll(watched, 2, 3500);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            error = "Microphone stopped delivering audio";
            goto done;
        }
        if (watched[0].revents) {
            char cmd[32];
            ssize_t count = recv(fd, cmd, sizeof(cmd), 0);
            if (count <= 0) goto done;
            for (ssize_t i = 0; i < count; ++i) {
                switch (cmd[i]) {
                case 'f': if (!finishing) finishing = 4; break;
                case 'a': automatic = 1; break;
                case 'n': want_alternatives = 1; break;
                default: goto done;
                }
            }
        }
        if (!watched[1].revents) continue;
        ssize_t bytes = read(mic, pcm + filled, sizeof(pcm) - filled);
        if (bytes < 0 && errno == EINTR) continue;
        if (bytes <= 0) {
            error = "Microphone capture ended unexpectedly";
            goto done;
        }
        filled += (size_t)bytes;
        if (filled < sizeof(pcm)) continue;
        filled = 0;
        float frame[FRAME];
        for (int i = 0; i < FRAME; ++i) {
            int sample = (int16_t)(pcm[2*i] | pcm[2*i+1] << 8);
            if (abs(sample) > peak) peak = abs(sample);
            frame[i] = sample / 32768.0f;
        }
        ++total;
        if (total % 2 == 0) {
            char level[6];
            snprintf(level, sizeof(level), "%u", (unsigned)peak);
            if (send_event(fd, "level", level) != 0) goto done;
            peak = 0;
        }

        SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, frame, FRAME);
        pthread_mutex_lock(&decoder_lock);
        if (started) {
            feed(stream, frame);
        } else {
            memcpy(prior[voiced % PREROLL], frame, sizeof(frame));
            ++voiced;
            if (SherpaOnnxVoiceActivityDetectorDetected(vad)) {
                started = total;
                for (int i = voiced > PREROLL ? voiced - PREROLL : 0; i < voiced; ++i)
                    feed(stream, prior[i % PREROLL]);
            }
        }
        const SherpaOnnxOnlineRecognizerResult *text =
            started ? SherpaOnnxGetOnlineStreamResult(recognizer, stream) : NULL;
        pthread_mutex_unlock(&decoder_lock);
        if (text) {
            char next[MAX_TEXT + 1];
            snprintf(next, sizeof(next), "%s", text->text);
            SherpaOnnxDestroyOnlineRecognizerResult(text);
            if (strcmp(previous, next)) {
                memcpy(previous, next, strlen(next) + 1);
                if (send_event(fd, "partial", previous) != 0) goto done;
            }
        }
        int endpoint = !SherpaOnnxVoiceActivityDetectorEmpty(vad);
        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad))
            SherpaOnnxVoiceActivityDetectorPop(vad);
        if ((automatic && endpoint) || (finishing && --finishing == 0)) break;
    }
    if (send_event(fd, "transcribing", "") != 0) goto done;
    result = 1;

done:
    if (mic >= 0) close(mic);
    if (recorder > 0) {
        kill(recorder, SIGKILL);
        while (waitpid(recorder, NULL, 0) < 0 && errno == EINTR) {}
    }
    if (error) send_event(fd, "error", error);
    else if (result && started) {
        pthread_mutex_lock(&decoder_lock);
        const float zeros[FRAME] = {0};
        for (int i = 0; i < 25; ++i) feed(stream, zeros);
        SherpaOnnxOnlineStreamInputFinished(stream);
        decode(stream);
        const SherpaOnnxOnlineRecognizerResult *text =
            SherpaOnnxGetOnlineStreamResult(recognizer, stream);
        const char *alternatives = want_alternatives
            ? SherpaOnnxGetOnlineStreamAlternativesAsJson(recognizer, stream)
            : NULL;
        pthread_mutex_unlock(&decoder_lock);
        if (alternatives && strlen(alternatives) <= MAX_TEXT)
            send_event(fd, "alternatives", alternatives);
        send_event(fd, "final", text ? text->text : "");
        SherpaOnnxDestroyOnlineStreamResultJson(alternatives);
        if (text) SherpaOnnxDestroyOnlineRecognizerResult(text);
    } else if (result) send_event(fd, "final", "");
    if (stream) {
        pthread_mutex_lock(&decoder_lock);
        SherpaOnnxDestroyOnlineStream(stream);
        pthread_mutex_unlock(&decoder_lock);
    }
    if (vad) SherpaOnnxDestroyVoiceActivityDetector(vad);
}

static void *client(void *argument) {
    int fd = *(int *)argument;
    free(argument);
    struct timeval timeout = {.tv_sec = 3};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    take(fd);
    close(fd);
    atomic_fetch_sub(&clients, 1);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 2 || strlen(argv[1]) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        fputs("usage: enzimos-streaming-speech SOCKET_PATH\n", stderr);
        return 2;
    }
    const SherpaOnnxOnlineRecognizerConfig config = {
        .feat_config = {.sample_rate = RATE, .feature_dim = 80},
        .model_config = {
            .transducer = {.encoder = "encoder.onnx", .decoder = "decoder.onnx", .joiner = "joiner.onnx"},
            .tokens = "tokens.txt", .num_threads = 1, .provider = "cpu",
        },
        .decoding_method = "modified_beam_search", .max_active_paths = 8,
    };
    vad_config = (SherpaOnnxVadModelConfig){
        .silero_vad = {.model = "silero_vad.onnx", .threshold = 0.5,
                       .min_speech_duration = 0.2, .min_silence_duration = 0.9,
                       .max_speech_duration = 12, .window_size = FRAME},
        .sample_rate = RATE, .num_threads = 1, .provider = "cpu",
    };
    recognizer = SherpaOnnxCreateOnlineRecognizer(&config);
    if (!recognizer) return 1;
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    strcpy(addr.sun_path, argv[1]);
    umask(0077);
    int server = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (server < 0 || bind(server, (const void *)&addr, sizeof(addr)) != 0 || listen(server, 8) != 0) {
        perror("streaming speech socket");
        return 1;
    }
    fputs("Kroko streaming speech ready\n", stderr);
    for (;;) {
        int fd = accept4(server, NULL, NULL, SOCK_CLOEXEC);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            return 1;
        }
        if (atomic_load(&clients) >= MAX_CLIENTS) {
            send_event(fd, "error", "Speech recognition is busy");
            close(fd);
            continue;
        }
        int *argument = malloc(sizeof(*argument));
        if (!argument) {
            close(fd);
            continue;
        }
        *argument = fd;
        pthread_t thread;
        atomic_fetch_add(&clients, 1);
        if (pthread_create(&thread, NULL, client, argument) == 0) {
            pthread_detach(thread);
        } else {
            atomic_fetch_sub(&clients, 1);
            free(argument);
            close(fd);
        }
    }
}
