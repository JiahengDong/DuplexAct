#!/usr/bin/env python3
"""
HIT-TMG/Lychee-FD client: in-process incremental streaming (no network server).

Reproduces the core round loop of the official `app.py`
(`_run_realtime_session_worker()`), without its multi-user/SSE scaffolding:

  1. Create one incremental stream session on `app.generator` and one
     persistent-mode `RealtimeTTSPool`. Output audio is drained from the
     tuples yielded by `run_chunk_dialogue_inference()`.
  2. Feed `input_pcm` in `INFER_WINDOW_MS` windows (400 ms, the official
     default), dispatching each round only once that much wall-clock time has
     elapsed since stream start, so latency is measured in real time.
  3. Carry model state across rounds via `round_ctx["next_prefix"]` and
     `round_ctx["next_listening_state"]`.
  4. After the last window (`stream_flush_audio_tail=True`), close the session
     and stop the TTS pool (hard cutoff at `input_duration`, as in every client;
     see `realtime_common.py`).

`model_speech_segments` come from the native `round_trace["events"]`: each
event's `start_rel_ms`/`end_rel_ms` is shifted by its round's dispatch offset
and passed to `recorder.set_speaking()`.

Backends (selected by `LYCHEEFD_USE_VLLM`, default `1`):
  - vLLM (default): `create_incremental_stream_session()` keeps the same vLLM
    request alive across consecutive speaking rounds
    (`keep_alive_for_speaking=True`), so one reply stays one event.
  - HF: the open speaking event is reset at the start of every round, so a
    reply spanning a 400 ms round boundary can be recorded as several events.
    This is the official HF-backend behavior and is not stitched here.

The vLLM backend needs the upstream repo's patched vLLM fork
(`third_party/vllm`, which provides `MultiHeadRequestState`) ahead of the
pip-installed `vllm==0.6.5` on `sys.path`, with the wheel's compiled native
artifacts (`_C.abi3.so`, `_moe_C.abi3.so`, `vllm_flash_attn/`, `_version.py`)
copied into it, as the upstream `scripts/start_backend.sh` does.

Standalone usage (single sample, for debugging):
    python infer_lychee_fd.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from local_common import opensource_root_for
from realtime_common import (
    EventLogger,
    RealtimeClientBase,
    SynchronizedRecorder,
    sleep_until_deadline,
)

LYCHEE_FD_SLUG = "lychee-fd"
DEFAULT_MODEL = "HIT-TMG/Lychee-FD"
DEFAULT_PROMPT_VOICE = "default_male"

INPUT_SR  = 16_000   # app.py's INPUT_SAMPLE_RATE
OUTPUT_SR = 24_000   # app.py's TTS_SAMPLE_RATE (Token2Wav vocoder output rate)

# Per-round window size; matches app.py's LYCHEEFD_REALTIME_INFER_WINDOW_MS default.
INFER_WINDOW_MS = int(os.environ.get("LYCHEEFD_REALTIME_INFER_WINDOW_MS", "400"))

REPO_DIR = opensource_root_for(LYCHEE_FD_SLUG) / "repo"
STEPAUDIO2_SOURCE_DIR = REPO_DIR / "third_party" / "Step-Audio2"
VLLM_SOURCE_DIR = REPO_DIR / "third_party" / "vllm"     # patched vLLM fork
CKPT_ROOT = opensource_root_for(LYCHEE_FD_SLUG) / "checkpoints" / "model_root"
MODEL_PATH = CKPT_ROOT / "lychee_full_duplex"
TOKEN2WAV_PATH = CKPT_ROOT / "token2wav"

os.environ.setdefault("STEPAUDIO2_SOURCE_DIR", str(STEPAUDIO2_SOURCE_DIR))
# vLLM backend by default (see module docstring); set LYCHEEFD_USE_VLLM=0 for
# the HF backend.
os.environ.setdefault("LYCHEEFD_USE_VLLM", "1")
os.environ.setdefault("LYCHEEFD_T2W_REMOTE_ENABLED", "0")  # load Token2Wav in-process
os.environ.setdefault("LYCHEEFD_STARTUP_WARMUP", "0")     # we warm up on first real sample instead
# vLLM engine settings for a single 80GB GPU (one ~24GB checkpoint, no
# tensor/pipeline parallelism). enforce_eager keeps its upstream default (True).
os.environ.setdefault("LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION", "0.85")
os.environ.setdefault("LYCHEEFD_VLLM_MAX_MODEL_LEN", "8192")
# Upstream opt-in flag: call torch.cuda.empty_cache() + ipc_collect() when a
# session closes. One process runs many samples back-to-back, and without this
# per-session tensors are not returned to the driver and GPU memory grows until
# OOM. It does not affect model outputs.
os.environ.setdefault("LYCHEEFD_CUDA_EMPTY_CACHE_ON_SESSION_CLOSE", "1")

USE_VLLM = os.environ.get("LYCHEEFD_USE_VLLM", "1") == "1"

if USE_VLLM:
    # Must precede any (including transitive) `import vllm`, otherwise the
    # site-packages copy is cached and the patched tree is never used.
    if str(VLLM_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(VLLM_SOURCE_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(STEPAUDIO2_SOURCE_DIR) not in sys.path:
    sys.path.append(str(STEPAUDIO2_SOURCE_DIR))


def _extract_events_from_round_trace(round_trace: dict, round_offset_s: float) -> list[dict]:
    """Convert one round's closed events onto the sample timeline.

    round_trace["events"] entries look like {"kind": "response",
    "start_rel_ms": ..., "end_rel_ms": ...}, relative to the start of that
    `run_chunk_dialogue_inference()` call. `round_offset_s` is the round's
    dispatch time relative to stream start (the origin `recorder` and
    `event_logger` use). Events missing a start or end are skipped.
    """
    out: list[dict] = []
    for ev in round_trace.get("events", []) or []:
        if not isinstance(ev, dict):
            continue
        s_ms, e_ms = ev.get("start_rel_ms"), ev.get("end_rel_ms")
        if s_ms is None or e_ms is None:
            continue
        try:
            s = round(round_offset_s + float(s_ms) / 1000.0, 4)
            e = round(round_offset_s + float(e_ms) / 1000.0, 4)
        except (TypeError, ValueError):
            continue
        if e > s:
            out.append({"kind": ev.get("kind"), "start_s": s, "end_s": e})
    return out


class LycheeFDClient(RealtimeClientBase):
    MODEL_SLUG = LYCHEE_FD_SLUG
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = INPUT_SR

    _app = None
    _loaded_model_path: Optional[str] = None
    # `app.load_models()` must run once per process. With `--workers N`,
    # concurrent loads would build the vLLM engine twice and fail with
    # "Duplicate layer name" errors, so loading is serialized.
    _load_lock = threading.Lock()

    @classmethod
    def _load(cls):
        if cls._app is not None and cls._loaded_model_path == str(MODEL_PATH):
            return cls._app

        with cls._load_lock:
            # Double-checked locking: another thread may have finished
            # loading while we were waiting on the lock above.
            if cls._app is not None and cls._loaded_model_path == str(MODEL_PATH):
                return cls._app
            return cls._load_locked()

    @classmethod
    def _load_locked(cls):
        import lychee_fd.app as app  # noqa: F401  (module-level import triggers app.py's own setup)

        # StepAudio2ForCausalLM has no sdpa attention path, so "eager" is the
        # only supported attention implementation.
        status = app.load_models(str(MODEL_PATH), str(TOKEN2WAV_PATH), attn_impl="eager")
        for line in status.splitlines():
            if line.strip():
                print(f"[lychee-fd load_models] {line}", file=sys.stderr)

        cls._app = app
        cls._loaded_model_path = str(MODEL_PATH)
        return app

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder,
                                   event_logger: EventLogger,
                                   start_streaming) -> None:
        app = await asyncio.to_thread(self._load)

        def _setup_session():
            prompt_wav_path = app._resolve_prompt_wav_path(DEFAULT_PROMPT_VOICE)
            tts_bridge = app.RealtimeTTSPool(
                prompt_wav_path=prompt_wav_path,
                vocoder_hop_size=app.TTS_VOCODER_HOP_SIZE,
                pre_lookahead_len=int(app.get_token2wav_pre_lookahead_len()),
                worker_name="lychee_fd_replay_tts",
                persistent_mode=True,   # kept alive across every round of this sample
            )
            # Prefer the native session (supports keep_alive_for_speaking on
            # vLLM) over the HF-compatible one, in the same order as app.py.
            session_kwargs = dict(
                prefix=None,
                initial_listening_state="l",
                start_speak_token_factor=1.2,
                start_listen_token_factor=1.2,
                end_speak_token_factor=1.0,
                audio_incremental_mode=True,
            )
            if hasattr(app.generator, "create_incremental_stream_session"):
                stream_session = app.generator.create_incremental_stream_session(**session_kwargs)
            else:
                stream_session = app.generator.create_hf_incremental_stream_session(**session_kwargs)
            return tts_bridge, stream_session

        tts_bridge, stream_session = await asyncio.to_thread(_setup_session)

        pcm_f32_all = input_pcm.astype(np.float32) / 32768.0
        total_samples = len(pcm_f32_all)
        window_samples = max(1, int(round(INPUT_SR * INFER_WINDOW_MS / 1000.0)))

        t0 = await start_streaming()
        loop = asyncio.get_event_loop()

        # Switch the recorder to manual segments before any audio arrives.
        # Native events can close several rounds after speech onset; without
        # this, automatic tracking would open a segment first and it would be
        # left open until end of recording.
        recorder.set_speaking(False, at_s=0.0)

        round_prefix: Optional[dict] = None
        round_state = "l"
        idx = 0
        round_id = 0

        try:
            while idx < total_samples:
                round_id += 1
                next_idx = min(idx + window_samples, total_samples)
                is_last_round = next_idx >= total_samples

                # Dispatch each window only once its audio has elapsed in
                # wall-clock time.
                deadline = t0 + (next_idx / INPUT_SR)
                await sleep_until_deadline(loop, deadline)

                chunk_f32 = pcm_f32_all[idx:next_idx]
                round_start_s = idx / INPUT_SR
                idx = next_idx

                def _run_round(chunk=chunk_f32, prefix=round_prefix, state=round_state,
                                flush_tail=is_last_round):
                    realtime_ctx: dict = {}
                    out_chunks: list[np.ndarray] = []
                    for _msgs, audio_output, _status, _json, _dbg in app.run_chunk_dialogue_inference(
                        audio_input=(INPUT_SR, chunk),
                        start_speak_factor=1.2,
                        end_speak_factor=1.0,
                        start_listen_factor=1.2,
                        prompt_voice=DEFAULT_PROMPT_VOICE,
                        tts_chunk_size=getattr(app, "REALTIME_TTS_CHUNK_SIZE_DEFAULT", 10),
                        prefix=prefix,
                        initial_listening_state=state,
                        realtime_ctx=realtime_ctx,
                        stream_session=stream_session,
                        stream_audio_incremental=True,
                        stream_flush_audio_tail=flush_tail,
                        tts_bridge=tts_bridge,
                        flush_tts_on_round_end=False,
                    ):
                        if audio_output is not None:
                            _sr, a = audio_output
                            if a is not None and len(a) > 0:
                                out_chunks.append(np.asarray(a, dtype=np.float32).reshape(-1))
                    full = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
                    return full, realtime_ctx

                round_audio, round_ctx = await asyncio.to_thread(_run_round)

                round_prefix = round_ctx.get("next_prefix")
                round_state = round_ctx.get("next_listening_state") or "l"

                round_trace = round_ctx.get("round_trace")
                native_events = (
                    _extract_events_from_round_trace(round_trace, round_start_s)
                    if isinstance(round_trace, dict) else []
                )
                for ev in native_events:
                    recorder.set_speaking(True, at_s=ev["start_s"])
                    recorder.set_speaking(False, at_s=ev["end_s"])

                event_logger.log("recv", "round_done", {
                    "round_id": round_id,
                    "round_start_s": round(round_start_s, 4),
                    "output_samples": int(len(round_audio)),
                    "native_events": native_events,
                    "open_event_unfinished": bool(
                        isinstance(round_trace, dict) and round_trace.get("open_event_unfinished")
                    ),
                })

                if len(round_audio) > 0:
                    pcm_i16 = (np.clip(round_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                    await recorder.add(pcm_i16.tobytes())
        finally:
            def _teardown():
                try:
                    app._close_incremental_stream_session(stream_session, "replay_sample", "sample_end")
                except Exception:
                    pass
                try:
                    tts_bridge.stop(force_flush=False)
                except Exception:
                    pass

            await asyncio.to_thread(_teardown)


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = LycheeFDClient(
        api_key="",
        model_name=args.model,
        system_prompt="",
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
