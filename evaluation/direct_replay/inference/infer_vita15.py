#!/usr/bin/env python3
"""
VITA-MLLM/VITA-1.5 client. Drives the model's official real-time Socket.IO
server (`web_demo/server.py`), which runs two copies of the 8B LLM on two GPUs
plus a TTS worker, gated by a bundled Silero VAD (threshold 0.7, 500ms min
silence). One copy keeps consuming the live PCM stream while the other
generates, so the model still hears the user while it is speaking.

Protocol (as in FD-Bench's `examples/vita/vita/web_demo/client-audio.py`):
  - HTTPS/WSS with a self-signed cert; connect with `ssl_verify=False`.
  - `reset_state` then `recording-started` open a turn.
  - Input is `emit('audio', json.dumps({'sample_rate': 16000, 'audio':
    list(<int16 PCM bytes>)}))` in 200ms chunks paced in real time. The
    server always assumes 16kHz regardless of the payload's `sample_rate`.
  - Turn end is decided by the server's VAD (500ms silence).
  - Model audio (24kHz int16 PCM) comes back via `emit('audio', <bytes>)`
    only inside the server's `handle_audio()`, i.e. one reply per input
    chunk we send (pull-on-write). After real user content we keep sending
    silence chunks so the VAD can fire and TTS output can be drained.
  - The server treats incoming PCM as still recording, and
    `recording-stopped` interrupts and resets it. So the dataset's 15s
    trailing pad is stripped from the user turn and the same 15s
    observation window is spent pulling silence chunks before
    `recording-stopped`.

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

INPUT_SR  = 16_000   # server-side `target_sample_rate`; the payload's sample_rate is ignored
OUTPUT_SR = 24_000   # TTS output rate

SEND_CHUNK_SAMPLES = int(0.2 * INPUT_SR)   # 200ms chunks, as in FD-Bench's client-audio.py
SEND_CHUNK_SEC = SEND_CHUNK_SAMPLES / INPUT_SR

# Trailing silence at the end of every Direct Replay input.wav. It is stripped
# from the user turn and replaced by an equal wait after real content, spent
# sending zero chunks (kind=pull_silence) because handle_audio is pull-on-write.
DATASET_PAD_S = 15.0
# Covers Silero's 500ms min silence; runs inside the 15s wait, not after it.
VAD_SILENCE_TAIL_S = 1.0

# server.py sleeps 3s after a disconnect before freeing the session's
# connected_users slot. With `--max_users 1`, reconnecting inside that window
# is rejected (`BadNamespaceError: / is not a connected namespace.`), so we
# wait at least this long between samples. 4s leaves margin over the 3s.
SERVER_DISCONNECT_COOLDOWN_S = 4.0

VITA_HOST = "127.0.0.1"
# Per-job port so two jobs on the same node do not collide.
VITA_PORT = int(os.environ.get("VITA15_SERVER_PORT", "8081"))

_ROOT = opensource_root_for(VITA15_SLUG)
# The server needs its own pinned stack (torch==2.4.0, vllm==0.5.5,
# transformers==4.44.2, see web_demo/web_demo_requirements.txt, plus a vLLM
# model registry patched for VITA's `Qwen2ForConditionalGeneration`), so it
# runs from a separate venv.
VITA_DEMO_ENV_PYTHON = str(_ROOT / "vita_demo_env" / "bin" / "python3")
VITA_DEMO_CKPT = _ROOT / "demo_VITA_ckpt"
REPO_DIR = _ROOT / "repo"

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

# One real dataset turn, streamed and discarded, to warm the audio path (see
# _warm_audio_path). The first match of a sorted glob keeps it deterministic.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
WARMUP_CLIP_GLOB = "turn_taking/take_turn_immediately/*/input.wav"
WARMUP_CONTENT_S = 10.0
# Keep pulling until TTS has been quiet this long, so the warmup reply is not
# left in the server's queue for the first sample.
WARMUP_QUIET_S = 3.0
WARMUP_SESSION_TIMEOUT_S = 90.0

# Per-sample drain of the server's cross-session TTS backlog before the
# recorded window opens (see _drain_stale_tts). Shorter than WARMUP_QUIET_S
# because it is paid on every sample; the time spent is logged as
# pre_stream_drain.
SAMPLE_DRAIN_QUIET_S = 2.0
SAMPLE_DRAIN_MAX_S = 30.0


def _f32_to_i16_bytes(arr_f32: np.ndarray) -> bytes:
    return (np.clip(arr_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class VITA15Client(RealtimeClientBase):
    MODEL_SLUG = VITA15_SLUG
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = INPUT_SR
    # 2-GPU generation can outlast the default 90s overhead.
    TIMEOUT_OVERHEAD_S = 180.0
    # Stop recording at input_duration (content + 15s pad) rather than
    # writing through socket teardown after recording-stopped.
    HARD_CAP_AT_TARGET = True

    # Class-level so it is shared by the per-sample client instances that all
    # talk to the same cached server. See SERVER_DISCONNECT_COOLDOWN_S.
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
        """Wait until the TTS model and both 8B copies are resident on the GPUs.

        The port opens as soon as Flask is up, before the subprocesses have
        loaded their weights, so readiness is judged from GPU memory.

        Memory is summed over visible GPUs rather than checked per GPU so that
        CUDA_VISIBLE_DEVICES="0,0" (both copies on one 80 GiB card, ~36 GiB)
        also works; _visible_gpu_mem_mib reports that case as one GPU. With two
        distinct GPUs, one loaded copy (~19.6 GiB) stays under the threshold.
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

        Resident weights are not enough: the first one or two sessions on a
        fresh server return no audio. Empty output is a legitimate model
        outcome on this dataset, so that loss cannot be detected per sample
        afterwards; this turn serves as a pre-flight check instead.

        The reply is pulled out completely rather than just waiting for its
        first chunk: server.py's TTS_OUTPUT_QUEUE is app-scoped and only
        drained by handle_audio, so anything left queued would be delivered to
        the next session as a spurious onset at t=0.
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
                # So the first sample also respects the 3s post-disconnect cooldown.
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
        # Two vLLM 8B engines take minutes to load, so this runs outside the
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
        sample left in it is handed to this session. Clearing the queue is not
        enough because tts_worker has no stop event and an in-flight synthesis
        refills it afterwards.

        handle_audio releases at most one queue item per `audio` event we send,
        so streaming silence until deliveries stop empties the queue by
        observation rather than by timing.

        Audio received here is discarded by _on_audio's stream_t0 guard, so it
        never reaches the recorder or shifts the measured onset.
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
            # opens anyway and the sample is flagged in the event log.
            payload_log["capped"] = True
        event_logger.log("send", "pre_stream_drain", payload_log)

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        import socketio   # python-socketio[asyncio_client]

        await LocalServerManager.ensure(
            key=f"vita15_server:{self.model_name}",
            cmd=self._server_cmd(),
            # `python -m web_demo.server` imports `web_demo.*` and `vita.*`
            # from the CWD, so this must be the repo root.
            cwd=REPO_DIR,
            host=VITA_HOST, port=VITA_PORT,
            log_path=LOG_DIR / f"vita15_server.{os.environ.get('SLURM_JOB_ID', 'local')}.log",
            # Two 8B vLLM engines plus the TTS decoder load serially across
            # three subprocesses; typically 3-6 minutes.
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
                # Counted for the log but never recorded.
                drain["last_audio"] = time.monotonic()
                drain["n_bytes"] += len(pcm_bytes)
                return
            await recorder.add(pcm_bytes)
            event_logger.log("recv", "audio", {"n_bytes": len(pcm_bytes)})

        @sio.on("stop_tts")
        async def _on_stop_tts() -> None:
            # Sent by the server on barge-in or when the other LLM worker's
            # output supersedes this one. Not expected with one turn per
            # sample; logged only, since recorded audio is never erased.
            event_logger.log("recv", "stop_tts")

        @sio.on("too_many_users")
        async def _on_too_many_users(*_a) -> None:
            event_logger.log("recv", "too_many_users")

        # Wait out the rest of SERVER_DISCONNECT_COOLDOWN_S since the previous
        # sample disconnected, if any remains.
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
        # Strip the dataset's 15s trailing pad from the user turn.
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

            # 15s observation window from the end of real content, spent
            # sending silence because handle_audio only returns TTS audio in
            # response to input. recording-stopped interrupts any in-flight
            # generation, so it is sent only after the window.
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
            # Stop the recording at observe_end, before disconnect() blocks on
            # the server's teardown, so the wav does not include that tail.
            recorder.stop()
        finally:
            # server.py's `global_history` (multi-turn context) is shared by
            # every session on the server, and its disconnect handler does
            # not wait for in-flight generation, which may still append to it
            # later. A second `reset_state` before disconnecting keeps this
            # sample's context out of the next one.
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
                # Stamped even if disconnect() raised: the server still
                # starts its 3s cooldown when the socket goes away.
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
