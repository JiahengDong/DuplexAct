#!/usr/bin/env python3
"""
direct_replay/inference/infer_vita15.py
==========================================
VITA-MLLM/VITA-1.5 client -- drives the model's OWN official real-time
Socket.IO server (`web_demo/server.py`), not a bespoke single-process
adaptation. See `model_docs/opensource/vita-1.5.md` for the full rationale
and setup steps; this docstring covers only what the client itself does.

Why the full official server (supersedes an earlier single-GPU draft):
VITA-1.5's authors' own live-interaction design runs TWO GPU-resident
copies of the 8B LLM (`model_1_process`/`model_2_process` in server.py,
pinned to `CUDA_VISIBLE_DEVICES` "0"/"1") plus a separate TTS worker
process, gated by their own bundled Silero-VAD turn-detector
(`wakeup_and_vad/wakeup_and_vad.py`, threshold=0.7, min_silence_duration_ms
=500). The dual-copy design isn't just about barge-in-while-the-model-is-
talking -- while ONE copy is busy generating/decoding a response, the
OTHER copy's process is free to keep consuming the live PCM stream
(`send_pcm()` runs continuously per connection, independent of either LLM
worker's state), so the model's own architecture never has a "blind spot"
window where incoming audio goes unprocessed just because it's mid-response.
A single busy worker (the earlier single-GPU draft) cannot replicate that:
it just doesn't observe input at all while generating. Since several of
our L2 actions (`backchannel`, `wait: continuous_floor_holding`) are
specifically about behavior WHILE the user's audio keeps streaming, not
just single-turn-then-silence, this distinction is real for us, not just
faithfulness for its own sake -- hence standing up the full 2-GPU server
rather than extracting only the VAD module onto our existing 1-GPU
generation path.

Protocol (see FD-Bench's `examples/vita/vita/web_demo/client-audio.py`,
which drives this exact server for the same kind of batch replay
evaluation, and `web_demo/server.py`'s `@socketio.on(...)` handlers):
  - Session is HTTPS/WSS (server generates a self-signed cert on first
    launch) -- connect with `ssl_verify=False`.
  - `emit('reset_state')` then `emit('recording-started')` open a turn.
  - Input audio is sent as `emit('audio', json.dumps({'sample_rate':
    16000, 'audio': list(<int16 PCM bytes>)}))`, one chunk at a time,
    paced in real wall-clock time (200ms chunks, matching FD-Bench's own
    client) -- `target_sample_rate=16000` is hardcoded server-side
    (VAD + wakeup detector), NOT read from the payload's `sample_rate`
    field, so this must always be 16kHz regardless of what's sent there.
  - Turn-end is decided ENTIRELY by the server's own VAD (500ms silence),
    not by us -- we just keep streaming.
  - Model audio (TTS output, 24kHz int16 PCM) is pushed back via
    `emit('audio', <raw bytes>)` -- but ONLY as a side effect INSIDE the
    server's `handle_audio()` callback, i.e. only in response to OUR
    'audio' emits, never autonomously. This is a pull-on-write protocol,
    not a push stream: if we stop sending input chunks, we stop receiving
    output too, even if the model has audio queued up server-side ready
    to go. So after real user content we keep emitting silence chunks to
    (a) let Silero see >=500ms of zeros so VAD can fire `end`, and (b)
    drain TTS during the observation window.
  - Dataset wavs bake a 15s trailing pad into `input.wav`. VITA's official
    server does NOT treat that pad as an observation window: it is still
    "recording", and `recording-stopped` then `interrupt()` + `stop_tts` +
    `reset()`. Streaming the pad as user audio and stopping immediately
    pinned every onset to clip_end+15.00s. Strip the pad from the user
    turn, then wait the same 15s as every other client — by pulling
    silence chunks, not a dead sleep — THEN `recording-stopped`.

Standalone usage (single sample, for debugging):
    python infer_vita15.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger
from local_common import LocalServerManager, opensource_root_for

VITA15_SLUG = "vita-1.5"
DEFAULT_MODEL = "VITA-MLLM/VITA-1.5"

INPUT_SR  = 16_000   # server.py's own `target_sample_rate` (VAD + wakeup detector) -- fixed, not negotiable via the payload
OUTPUT_SR = 24_000   # TTS codec output rate (tts_worker()'s `seg.shape[-1]/24000`)

SEND_CHUNK_SAMPLES = int(0.2 * INPUT_SR)   # 200ms chunks -- matches FD-Bench's official client-audio.py cadence
SEND_CHUNK_SEC = SEND_CHUNK_SAMPLES / INPUT_SR

# Every Direct Replay input.wav ends with this much trailing silence.
# Other clients stream that pad as the observation window. VITA's official
# server treats incoming PCM as still-recording (Silero VAD) and
# `recording-stopped` as interrupt+reset, so we strip the pad from the user
# turn and instead wait DATASET_PAD_S after real content — same 15s window,
# not as user audio. We cannot asyncio.sleep through it: handle_audio is
# pull-on-write, so the wait is paced zero-chunks (kind=pull_silence).
DATASET_PAD_S = 15.0
# Silero min_silence_duration_ms=500. Nested inside the 15s wait; not extra.
VAD_SILENCE_TAIL_S = 1.0

# server.py's OWN `handle_disconnect()` and `disconnect_user()` both do
# `...release(); time.sleep(3); del connected_users[sid]` -- a deliberate,
# hard-coded 3s cooldown the authors themselves put between releasing a
# session's GPU/thread resources and actually freeing its `connected_users`
# slot. With `--max_users 1` (our deployment, one server shared across the
# whole job's samples -- see LocalServerManager.ensure() call below),
# `handle_connect()` checks `len(connected_users) >= args.max_users` BEFORE
# accepting a new connection -- reconnecting for the next sample inside
# that 3s window (which our per-sample connect/disconnect cycle did, with
# no pacing) hits this still-occupied slot and produces `BadNamespaceError:
# / is not a connected namespace.` client-side, exactly matching the ~50%
# alternating failure rate seen in the first production run: whichever
# consecutive-sample gap happened to be >=3s (writing recorder/events data,
# ASR, etc.) succeeded, whichever was <3s didn't. FD-Bench's own client
# never hits this because it launches a brand-new OS process per sample
# (`start_clients_single.sh`'s `while read audio_file; do client-audio.py
# ...; done` loop) against a long-lived server -- the inter-process
# overhead alone easily exceeds 3s every time, so they never needed to
# pace this explicitly. We reuse one client process for many samples (see
# module docstring), so we enforce the same effective cooldown ourselves
# rather than relying on incidental timing. Padded to 4.0s for scheduling
# jitter margin above the server's exact 3.0s.
SERVER_DISCONNECT_COOLDOWN_S = 4.0

VITA_HOST = "127.0.0.1"
# Per-job port, same rationale as MOSHI_PORT / FLM_AUDIO_PORT -- two
# same-model jobs can land on the same physical node.
VITA_PORT = int(os.environ.get("VITA15_SERVER_PORT", "8081"))

_ROOT = opensource_root_for(VITA15_SLUG)
# The 2-GPU server needs its OWN pinned stack (torch==2.4.0, vllm==0.5.5,
# transformers==4.44.2 -- see web_demo/web_demo_requirements.txt -- plus a
# hand-patched vLLM model registry for VITA's custom `Qwen2ForConditional
# Generation` architecture) that is incompatible with this process's own
# venv (which needs a much newer torch for other models sharing infra).
# It is therefore always launched as a fully separate interpreter/venv --
# see model_docs/opensource/vita-1.5.md "2-GPU server setup" for exactly
# how vita_demo_env/ and demo_VITA_ckpt/ were built.
VITA_DEMO_ENV_PYTHON = str(_ROOT / "vita_demo_env" / "bin" / "python3")
VITA_DEMO_CKPT = _ROOT / "demo_VITA_ckpt"
REPO_DIR = _ROOT / "repo"

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

# One real dataset turn, streamed and thrown away, to warm the audio path (see
# _warm_audio_path). Any sample works; pinned to a sorted glob so the warmup is
# deterministic across chunks and its cost is a known constant.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
WARMUP_CLIP_GLOB = "turn_taking/take_turn_immediately/*/input.wav"
WARMUP_CONTENT_S = 10.0
# Stop pulling once TTS has been quiet this long: the leftover has to be pulled
# out here, not left in the server's queue for sample 1 to inherit.
WARMUP_QUIET_S = 3.0
WARMUP_SESSION_TIMEOUT_S = 90.0

# Per-sample drain of the server's cross-session TTS backlog, spent before the
# recorded window opens. Clearing TTS_QUEUE/TTS_OUTPUT_QUEUE at server.py's
# session hooks is not enough: tts_worker takes no stop event (see its
# signature), so an in-flight synthesis refills the queue right after the
# clear. The 2026-09-20 smoke of that patch still had 42% of samples receiving
# audio 50ms in, after one input chunk. Observed quiet is race-free where a
# point-in-time clear is not. Quiet is 2.0s rather than WARMUP_QUIET_S's 3.0s
# because this is paid 2000 times, not once; pre_stream_drain is logged per
# sample so the threshold can be retuned from data instead of guesswork.
SAMPLE_DRAIN_QUIET_S = 2.0
SAMPLE_DRAIN_MAX_S = 30.0


def _f32_to_i16_bytes(arr_f32: np.ndarray) -> bytes:
    return (np.clip(arr_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class VITA15Client(RealtimeClientBase):
    MODEL_SLUG = VITA15_SLUG
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = INPUT_SR
    # 2-GPU generate can outlast the default 90s API-client overhead.
    TIMEOUT_OVERHEAD_S = 180.0
    # Subsequent only sees speech inside the 15s observation window
    # (input_duration = content + pad). Do not keep writing through
    # socket teardown after recording-stopped.
    HARD_CAP_AT_TARGET = True

    # Class-level (not instance-level) so it's shared across the many
    # short-lived VITA15Client instances `run_inference.py` creates over
    # the job's lifetime -- one per sample -- all talking to the SAME
    # cached server process. See SERVER_DISCONNECT_COOLDOWN_S above.
    _last_disconnect_monotonic: float | None = None

    @staticmethod
    def _server_cmd() -> list[str]:
        return [
            VITA_DEMO_ENV_PYTHON, "-m", "web_demo.server",
            "--ip", VITA_HOST, "--port", str(VITA_PORT),
            "--model_path", str(VITA_DEMO_CKPT),
            "--max_users", "1",
            "--timeout", "600",
        ]

    @staticmethod
    def _visible_gpu_mem_mib() -> list[float]:
        """Memory used (MiB) on GPUs named by CUDA_VISIBLE_DEVICES."""
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        want = {int(x) for x in visible.split(",") if x.strip()} if visible else None
        used: list[float] = []
        for line in out.strip().splitlines():
            idx_s, mem_s = [p.strip() for p in line.split(",")[:2]]
            if want is None or int(idx_s) in want:
                used.append(float(mem_s))
        return used

    @classmethod
    async def _wait_engines_loaded(cls, timeout_s: float = 1500.0) -> None:
        """Port-open is Flask only; TTS + two vLLM copies load in subprocesses.

        First smoke ran the sample in ~82s while GPU memory was still 0.
        Wait until the weights are resident (TTS, then 8B×2).

        Summed rather than per-GPU so one lane per GPU also works: server.py
        requires two CUDA_VISIBLE_DEVICES entries but hands each worker a single
        id, so "0,0" puts both 8B copies on physical GPU 0 (~36 GiB measured,
        fits an 80 GiB card) and lets three lanes run on three GPUs without
        patching the server. _visible_gpu_mem_mib dedupes, so that case reports
        one number; a per-GPU `min` test would wait forever on it. The sum still
        holds for two distinct GPUs, where one loaded copy (~19.6 GiB) stays
        under the threshold and the wait continues.
        """
        deadline = time.monotonic() + timeout_s
        last = []
        while time.monotonic() < deadline:
            try:
                used = cls._visible_gpu_mem_mib()
            except Exception:
                used = []
            last = used
            if used and sum(used) >= 30000.0:
                print(f"[vita-1.5] engines loaded gpu_mem_mib={used}", flush=True)
                return
            await asyncio.sleep(5.0)
        raise TimeoutError(
            f"VITA-1.5 engines did not reach 30000 MiB resident within {timeout_s:.0f}s "
            f"(last nvidia-smi MiB={last}). Check vita15_server.*.log"
        )

    @classmethod
    async def _warm_audio_path(cls, attempts: int = 3) -> None:
        """Stream a throwaway turn until the server actually returns audio.

        _wait_engines_loaded only proves the weights are resident. The
        socket.io -> WakeupAndVAD -> generate -> TTS path needs a real session
        on top of that: across the 33 chunk logs of the 2026-09-19 re-run,
        chunk slot 1 came back with no audio 33/33 times and slot 2 15/18,
        while slot 3 onward was fine. With a fresh server every 50 samples
        (run_vita_ids.sh) that silently costs ~1.8 real samples per chunk.

        It has to be spent here rather than detected later, because the loss is
        invisible per-sample: audio_bytes_received=0 is a legitimate outcome on
        this dataset (freeze-omni 557/2000, grok-voice 169/2000), so neither
        _validate_attempt nor run_metrics can tell a cold audio path apart from
        a model that chose to stay quiet -- and _validate_attempt deliberately
        refuses to reject empty output for exactly that reason. A throwaway
        turn converts it into a pre-flight check with a verifiable signal.

        Pulls the reply out instead of just waiting for its first chunk:
        server.py's TTS_OUTPUT_QUEUE is app-scoped and only drained by
        handle_audio, so anything still queued when we disconnect is handed to
        the next session ~50ms in. Leaving our own warmup reply behind would
        hand sample 1 a spurious onset at t=0 -- the very artifact that made
        the 2026-09-19 pass unscoreable for 52% of samples.
        """
        import socketio

        clips = sorted(DATA_ROOT.glob(WARMUP_CLIP_GLOB))
        if not clips:
            raise FileNotFoundError(
                f"No warmup clip matched {DATA_ROOT / WARMUP_CLIP_GLOB}. Refusing to "
                f"start: without the audio-path warmup the first 1-2 samples of this "
                f"chunk come back empty and nothing downstream can flag them."
            )
        pcm, sr = sf.read(str(clips[0]), dtype="float32")
        if sr != INPUT_SR:
            raise ValueError(f"warmup clip {clips[0]} is {sr}Hz, expected {INPUT_SR}Hz")
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        pad_samples = int(round(DATASET_PAD_S * INPUT_SR))
        content = pcm[:-pad_samples] if len(pcm) > pad_samples else pcm
        content = content[: int(WARMUP_CONTENT_S * INPUT_SR)]
        silence_chunk = np.zeros(SEND_CHUNK_SAMPLES, dtype=np.float32)

        for attempt in range(1, attempts + 1):
            state: dict = {"last_audio": None, "n_bytes": 0}
            sio = socketio.AsyncClient(ssl_verify=False)

            @sio.on("audio")
            async def _on_audio(data) -> None:
                if data:
                    state["last_audio"] = time.monotonic()
                    state["n_bytes"] += len(bytes(data))

            try:
                await sio.connect(f"https://{VITA_HOST}:{VITA_PORT}",
                                  transports=["websocket"], wait_timeout=60)
                await sio.emit("reset_state")
                await sio.emit("recording-started")

                idx = 0
                next_deadline = time.monotonic()
                deadline = next_deadline + WARMUP_SESSION_TIMEOUT_S
                while time.monotonic() < deadline:
                    chunk = content[idx: idx + SEND_CHUNK_SAMPLES]
                    idx += SEND_CHUNK_SAMPLES
                    await sio.emit("audio", json.dumps({
                        "sample_rate": INPUT_SR,
                        "audio": list(_f32_to_i16_bytes(
                            chunk if len(chunk) else silence_chunk)),
                    }))
                    last = state["last_audio"]
                    if last is not None and time.monotonic() - last > WARMUP_QUIET_S:
                        break
                    next_deadline += SEND_CHUNK_SEC
                    sleep_s = next_deadline - time.monotonic()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                await sio.emit("recording-stopped")
            finally:
                try:
                    await sio.emit("reset_state")
                except Exception:
                    pass
                try:
                    await sio.disconnect()
                except Exception:
                    pass
                # Same cooldown bookkeeping a real sample does, so sample 1
                # does not race server.py's 3s post-disconnect teardown.
                cls._last_disconnect_monotonic = time.monotonic()

            if state["n_bytes"] > 0:
                print(f"[vita-1.5] audio path warm after {attempt} throwaway turn(s), "
                      f"drained {state['n_bytes']} bytes", flush=True)
                return
            print(f"[vita-1.5] warmup turn {attempt}/{attempts} returned no audio",
                  flush=True)

        raise TimeoutError(
            f"VITA-1.5 audio path stayed cold across {attempts} throwaway turns. "
            f"Check vita15_server.*.log -- a run started from here would lose its "
            f"first samples silently."
        )

    @classmethod
    async def warmup(cls) -> None:
        # Two vLLM 8B engines take 3-6 minutes; must not run under the
        # per-sample timeout (see run_inference.py WARMUP_TIMEOUT_S).
        await LocalServerManager.ensure(
            key=f"vita15_server:{DEFAULT_MODEL}",
            cmd=cls._server_cmd(),
            cwd=REPO_DIR,
            host=VITA_HOST, port=VITA_PORT,
            log_path=LOG_DIR / f"vita15_server.{os.environ.get('SLURM_JOB_ID', 'local')}.log",
            startup_timeout_s=1800.0,
        )
        await cls._wait_engines_loaded()
        await cls._warm_audio_path()

    @staticmethod
    async def _drain_stale_tts(sio, drain: dict, event_logger: EventLogger) -> None:
        """Empty the server's TTS backlog before the recorded window opens.

        server.py's TTS_OUTPUT_QUEUE is app-scoped, so whatever the previous
        sample left in it is handed to this session. Clearing it at the session
        hooks does not fix that -- `tts_worker(model_path, inputs_queue,
        outputs_queue, worker_ready, wait_workers_ready)` takes no stop event,
        so a synthesis already in flight writes into the queue after the clear
        and wins the race. The 09-20 smoke of the cleared server still saw
        42% (21/50) of samples receive audio ~50ms in, after a single 200ms
        input chunk, 21/21 with a same-instant stop_tts.

        handle_audio is pull-on-write: it releases at most one queue item per
        `audio` event we send. So streaming silence and waiting for the
        deliveries to stop empties the queue by observation rather than by
        timing, which is what makes this race-free. The same mechanism already
        works once per server in _warm_audio_path (it drained 5.97 MB there);
        the leak is per-session, so it has to be paid per sample.

        Received audio is discarded by _on_audio's stream_t0 guard, so nothing
        here can reach the recorder or shift the measured onset.
        """
        silence = list(_f32_to_i16_bytes(np.zeros(SEND_CHUNK_SAMPLES, dtype=np.float32)))
        payload = json.dumps({"sample_rate": INPUT_SR, "audio": silence})

        t0 = time.monotonic()
        next_deadline = t0
        capped = False
        while True:
            await sio.emit("audio", payload)
            now = time.monotonic()
            if now - t0 > SAMPLE_DRAIN_MAX_S:
                capped = True
                break
            last = drain["last_audio"]
            if now - (t0 if last is None else last) > SAMPLE_DRAIN_QUIET_S:
                break
            next_deadline += SEND_CHUNK_SEC
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        payload_log = {
            "seconds": round(time.monotonic() - t0, 2),
            "n_bytes": drain["n_bytes"],
        }
        if capped:
            # Still producing after SAMPLE_DRAIN_MAX_S: the recorded window
            # opens anyway, so this sample may carry the artifact. Flagged
            # rather than raised, and auditable from the event log.
            payload_log["capped"] = True
        event_logger.log("send", "pre_stream_drain", payload_log)

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        import socketio   # python-socketio[asyncio_client] -- this process's own venv

        await LocalServerManager.ensure(
            key=f"vita15_server:{self.model_name}",
            cmd=self._server_cmd(),
            # `python -m web_demo.server` resolves both `web_demo.*` and
            # the top-level `vita.*` package via CWD (module-mode adds CWD
            # to sys.path[0]) -- must be the repo root, same as the
            # official README's own invocation.
            cwd=REPO_DIR,
            host=VITA_HOST, port=VITA_PORT,
            log_path=LOG_DIR / f"vita15_server.{os.environ.get('SLURM_JOB_ID', 'local')}.log",
            # Two 8B-parameter vLLM engines (CUDA graph capture included)
            # plus the TTS decoder loading serially across 3 subprocesses --
            # observed 3-6 minutes on an idle A100 pair; generous margin for
            # a busy/shared node.
            startup_timeout_s=1800.0,
        )

        stream_t0: dict = {"t": None}
        drain: dict = {"last_audio": None, "n_bytes": 0}

        sio = socketio.AsyncClient(ssl_verify=False)

        @sio.on("audio")
        async def _on_audio(data) -> None:
            if not data:
                return
            pcm_bytes = bytes(data)
            if stream_t0["t"] is None:
                # Previous session's leftover, pulled out by _drain_stale_tts.
                # Counted for the log, never handed to the recorder: writing it
                # is what produced the t=0 segment in the 09-19 and 09-20 runs.
                drain["last_audio"] = time.monotonic()
                drain["n_bytes"] += len(pcm_bytes)
                return
            await recorder.add(pcm_bytes)
            event_logger.log("recv", "audio", {"n_bytes": len(pcm_bytes)})

        @sio.on("stop_tts")
        async def _on_stop_tts() -> None:
            # Server-side signal used for live barge-in / stale
            # cross-worker race responses (see server.py's
            # `handle_audio()` "Received output from other process"
            # branch). Not expected in our one-session-one-turn-per-sample
            # protocol, but logged for visibility; no action needed since
            # SynchronizedRecorder has no "erase already-written audio"
            # concept to begin with (nor do any of our other clients).
            event_logger.log("recv", "stop_tts")

        @sio.on("too_many_users")
        async def _on_too_many_users(*_a) -> None:
            event_logger.log("recv", "too_many_users")

        # Respect server.py's own 3s post-disconnect cooldown (see
        # SERVER_DISCONNECT_COOLDOWN_S above) before reconnecting for this
        # sample -- only actually sleeps if the previous sample's own
        # bookkeeping (recorder/events/ASR writes) didn't already cover it.
        last_dc = type(self)._last_disconnect_monotonic
        if last_dc is not None:
            elapsed = time.monotonic() - last_dc
            remaining = SERVER_DISCONNECT_COOLDOWN_S - elapsed
            if remaining > 0:
                event_logger.log("send", "cooldown_wait", {"seconds": round(remaining, 2)})
                await asyncio.sleep(remaining)

        url = f"https://{VITA_HOST}:{VITA_PORT}"
        await sio.connect(url, transports=["websocket"], wait_timeout=60)
        event_logger.log("recv", "connected")

        await sio.emit("reset_state")
        await sio.emit("recording-started")
        event_logger.log("send", "recording-started")

        # Must sit after recording-started: handle_audio drops any sid absent
        # from connected_users straight into disconnect(), and that dict is
        # populated by handle_recording_started.
        await self._drain_stale_tts(sio, drain, event_logger)

        await start_streaming()
        stream_t0["t"] = time.monotonic()

        pcm_f32_all = input_pcm.astype(np.float32) / 32768.0
        pad_samples = int(round(DATASET_PAD_S * INPUT_SR))
        # Do not stream the dataset's 15s trailing pad as still-recording.
        # VAD `end` needs only ~500ms of zeros after real content.
        if len(pcm_f32_all) > pad_samples:
            content = pcm_f32_all[:-pad_samples]
        else:
            content = pcm_f32_all
        silence_chunk = np.zeros(SEND_CHUNK_SAMPLES, dtype=np.float32)

        try:
            idx = 0
            next_deadline = stream_t0["t"]

            async def _emit_pcm(chunk: np.ndarray, kind: str) -> None:
                payload = json.dumps({
                    "sample_rate": INPUT_SR,
                    "audio": list(_f32_to_i16_bytes(chunk)),
                })
                await sio.emit("audio", payload)
                event_logger.log("send", "audio", {
                    "n_samples": int(len(chunk)),
                    "kind": kind,
                })

            while idx < len(content):
                chunk = content[idx: idx + SEND_CHUNK_SAMPLES]
                idx += SEND_CHUNK_SAMPLES
                await _emit_pcm(chunk, "content")
                next_deadline += SEND_CHUNK_SEC
                sleep_s = next_deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

            content_s = len(content) / INPUT_SR
            event_logger.log("send", "content_exhausted", {
                "content_s": round(content_s, 3),
            })

            # Observation window = same 15s as the dataset pad / every other
            # client, clocked from real-content end. Pull-silence, not sleep:
            # a dead sleep would starve handle_audio and drop TTS.
            # recording-stopped still interrupt()s in-flight generate.
            observe_end = stream_t0["t"] + content_s + DATASET_PAD_S
            vad_tail_end = time.monotonic() + VAD_SILENCE_TAIL_S
            deadline = max(observe_end, vad_tail_end)
            while time.monotonic() < deadline:
                await _emit_pcm(silence_chunk, "pull_silence")
                next_deadline += SEND_CHUNK_SEC
                sleep_s = next_deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

            await sio.emit("recording-stopped")
            event_logger.log("send", "recording-stopped")
            # Freeze the wav at observe_end before disconnect() blocks
            # (server handle_disconnect sleeps 3s+). Otherwise the shared
            # recorder pads to teardown wall-clock (~84s) and Subsequent
            # ASR includes that tail.
            recorder.stop()
        finally:
            # Cross-sample independence guard: `global_history` (server.py's
            # multi-turn context, global_history_limit=1) is process-wide,
            # not per-connection -- shared by every sample that reuses this
            # LocalServerManager-cached server (see local_common.py). The
            # `reset_state` emit at connect time (above) clears it before
            # THIS sample starts, but `handle_disconnect()`'s own
            # `GlobalParams.interrupt()` does NOT actually wait for this
            # sample's LLM generation to finish before returning -- checked
            # both conditions it waits on (`is_generate`, `tts_data.
            # is_empty()`) and neither is ever set/populated anywhere in
            # server.py, so the wait is a no-op. If this sample's own
            # generation is still in flight when we disconnect, the LLM
            # worker (a separate process, decoupled from our connection)
            # will still append it to global_history whenever it finishes --
            # possibly after we've already moved on to the next sample. A
            # second `reset_state` right before disconnecting closes most
            # of that window (by now we've already streamed this sample's
            # full input_duration, so generation has almost always long
            # since finished) without requiring us to actually observe
            # generation-complete, which isn't exposed over this protocol
            # at all.
            try:
                await sio.emit("reset_state")
                event_logger.log("send", "reset_state_pre_disconnect")
            except Exception:
                pass
            try:
                await sio.disconnect()
            except Exception:
                pass
            finally:
                # Stamped unconditionally (even if disconnect() itself
                # raised) -- server.py's handle_disconnect() still fires
                # its own 3s cooldown off the underlying socket teardown
                # regardless of whether our disconnect() call cleanly
                # returned, so the next sample must still wait it out.
                type(self)._last_disconnect_monotonic = time.monotonic()


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = VITA15Client(
        api_key="",
        model_name=args.model,
        system_prompt=args.system_prompt,
        verbose=args.verbose,
    )
    result = await client.run_sample(
        input_wav=args.input_wav,
        output_wav=args.output_wav,
        events_path=args.events_path,
        input_duration=input_duration,
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_wav", type=Path, required=True)
    parser.add_argument("--output_wav", type=Path, required=True)
    parser.add_argument("--events_path", type=Path, required=True)
    parser.add_argument("--input_duration", type=float, default=None,
                         help="Defaults to the actual duration of input_wav")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
