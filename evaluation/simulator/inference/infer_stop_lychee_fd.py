#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_stop_lychee_fd.py

HIT-TMG/Lychee-FD client for the `stop` Level-2 action, simulator-driven
(live LLM-triggered injection -- see trigger_controller.py's module
docstring for the full architecture rationale).

Reuses direct_replay/inference/infer_lychee_fd.py's exact round-based
incremental-streaming architecture (see that file's own extensive module
docstring for the full writeup: `app.run_chunk_dialogue_inference(...,
stream_session=..., stream_audio_incremental=True)` called once per
400ms round, `round_ctx["next_prefix"]`/`round_ctx["next_listening_state"]`
carried forward across rounds, `round_trace["events"]` as the native
speaking-interval signal fed into `recorder.set_speaking()`) essentially
unchanged. The one adaptation (per MODEL_COMPATIBILITY.md's "MINOR ADJUST"
classification): Lychee-FD's official incremental API has NO token-level
text signal anywhere (confirmed in infer_lychee_fd.py's own docstring --
only coarse `l`/`s`/`b` listening-state + closed speaking-interval events),
so `LiveTriggerController` here runs on the SAME fallback signal
raon-speechchat's OWN client used before its `state.sequences` patch:
speaking-duration/phase only, `transcript_so_far` permanently empty
(`_build_user_message()` already handles this gracefully -- "(not
available)"). This is a REAL model limitation, not something we chose not
to implement: grep-confirmed no per-token text exists anywhere in the HF
backend's round loop (see infer_lychee_fd.py's own docstring for the
cross-round event-fragmentation wart this same limitation already causes
for `direct_replay`'s own metrics).

Because there is no text signal, `compute_model_speech_transcripts()` is
overridden here to return `""` per segment rather than using
`stop_common.py`'s `default_model_speech_transcripts()` default (which
requires a `text_tokens` list this client never populates) -- exactly the
guidance in that function's own docstring.

Native round window is `INFER_WINDOW_MS` (400ms, matches app.py's own
`LYCHEEFD_REALTIME_INFER_WINDOW_MS` default). Poll interval is the unified
1.0s simulated-user cadence, not 10x that window.

IMPORTANT correction vs. an earlier version of this file: `round_trace
["events"]` (the native speaking-interval signal `infer_lychee_fd.py`
feeds into `recorder.set_speaking()`) was found, during this file's own
smoke testing, to report SEVERELY wrong timestamps once round dispatch
falls behind real-time pacing -- confirmed identically on an
already-completed `direct_replay` PRODUCTION sample too (NOT a bug
introduced here). Full root-cause writeup:
`evaluation/simulator/MODEL_COMPATIBILITY.md`'s "SEVERE finding" section.
Fix applied here: `round_trace["events"]` is still logged into
events.jsonl (informational/diagnostic only), but is NO LONGER fed into
`recorder.set_speaking()` -- the recorder is left in its default
AUTOMATIC (audio-presence) segmentation mode during the live run (same as
every closed-source client and minicpm-o-4.5), and once the sample
finishes, `model_speech_segments` is unconditionally OVERWRITTEN by
`recompute_model_speech_segments()` below, which re-derives segments
directly from the finished `output.wav`'s own audio energy via
`realtime_common.recompute_speech_segments_from_audio()` -- sidestepping
the broken native timestamps entirely rather than trying to patch their
math. Live automatic segmentation during the run is only used
approximately, for `LiveTriggerController`'s own `current_speaking_
duration()` polling gate (coarse by nature already, tolerant of this).

This file is deliberately a SEPARATE module from
direct_replay/inference/infer_lychee_fd.py (never touches it, never
imported by it, loads its own independent copy of the model into this
process) -- see trigger_controller.py's own module docstring for the
"always a separate module" rule.

Standalone usage (single sample, for smoke testing):
    python infer_stop_lychee_fd.py --sample_dir <path to Candor_stop_0038 etc> \\
        --output_wav out.wav --events_path events.jsonl
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

DIRECT_REPLAY_INFERENCE = Path(__file__).resolve().parents[2] / "direct_replay" / "inference"
sys.path.insert(0, str(DIRECT_REPLAY_INFERENCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger, DynamicPCMSource,
    recompute_speech_segments_from_audio,
)
from local_common import opensource_root_for  # noqa: E402
from trigger_controller import LiveTriggerController, poll_interval_for_native_chunk  # noqa: E402
from stop_common import stop_cli_main, POST_CUE_OBSERVATION_S  # noqa: E402

LYCHEE_FD_SLUG = "lychee-fd"
DEFAULT_MODEL = "HIT-TMG/Lychee-FD"
DEFAULT_PROMPT_VOICE = "default_male"

INPUT_SR  = 16_000
OUTPUT_SR = 24_000

INFER_WINDOW_MS = int(os.environ.get("LYCHEEFD_REALTIME_INFER_WINDOW_MS", "400"))
INFER_WINDOW_SEC = INFER_WINDOW_MS / 1000.0
WINDOW_SAMPLES = max(1, int(round(INPUT_SR * INFER_WINDOW_MS / 1000.0)))
WINDOW_BYTES = WINDOW_SAMPLES * 2   # int16

LYCHEE_FD_POLL_INTERVAL_S = poll_interval_for_native_chunk(INFER_WINDOW_SEC)  # 1.0s unified user-side poll

REPO_DIR = opensource_root_for(LYCHEE_FD_SLUG) / "repo"
STEPAUDIO2_SOURCE_DIR = REPO_DIR / "third_party" / "Step-Audio2"
VLLM_SOURCE_DIR = REPO_DIR / "third_party" / "vllm"
CKPT_ROOT = opensource_root_for(LYCHEE_FD_SLUG) / "checkpoints" / "model_root"
MODEL_PATH = CKPT_ROOT / "lychee_full_duplex"
TOKEN2WAV_PATH = CKPT_ROOT / "token2wav"

os.environ.setdefault("STEPAUDIO2_SOURCE_DIR", str(STEPAUDIO2_SOURCE_DIR))
os.environ.setdefault("LYCHEEFD_USE_VLLM", "1")
os.environ.setdefault("LYCHEEFD_T2W_REMOTE_ENABLED", "0")
os.environ.setdefault("LYCHEEFD_STARTUP_WARMUP", "0")
os.environ.setdefault("LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION", "0.85")
os.environ.setdefault("LYCHEEFD_VLLM_MAX_MODEL_LEN", "8192")
os.environ.setdefault("LYCHEEFD_CUDA_EMPTY_CACHE_ON_SESSION_CLOSE", "1")

USE_VLLM = os.environ.get("LYCHEEFD_USE_VLLM", "1") == "1"

if USE_VLLM:
    if str(VLLM_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(VLLM_SOURCE_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(STEPAUDIO2_SOURCE_DIR) not in sys.path:
    sys.path.append(str(STEPAUDIO2_SOURCE_DIR))


def _extract_events_from_round_trace(round_trace: dict, round_offset_s: float) -> list[dict]:
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


class LycheeFDStopClient(RealtimeClientBase):
    MODEL_SLUG = LYCHEE_FD_SLUG
    OUTPUT_SR = OUTPUT_SR
    INPUT_SR = INPUT_SR

    _app = None
    _loaded_model_path: Optional[str] = None
    _load_lock = threading.Lock()

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, cue_pcm: bytes, reference_context: dict, llm_client):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.cue_pcm = cue_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result: dict = {}
        self.simulator_speech_segments: list[dict] = []

    def compute_model_speech_transcripts(self, segments: list) -> list[str]:
        # No text signal exists for this model at all -- see module
        # docstring and stop_common.py's default_model_speech_transcripts()
        # own docstring for why this must be an explicit override rather
        # than silently falling back to an empty `text_tokens` list.
        return ["" for _ in segments]

    def recompute_model_speech_segments(self, output_wav: Path, native_segments: list) -> list:
        # native_segments (whatever RealtimeClientBase.run_sample() derived
        # from the recorder's own automatic/live tracking during the run)
        # is intentionally discarded here in favor of a full post-hoc
        # energy-based pass over the FINISHED output.wav -- see module
        # docstring's "IMPORTANT correction" note for why the native signal
        # (and even live automatic tracking, given round-dispatch-induced
        # micro-gaps) isn't trusted for the final authoritative value.
        return recompute_speech_segments_from_audio(output_wav)

    @classmethod
    def _load(cls):
        if cls._app is not None and cls._loaded_model_path == str(MODEL_PATH):
            return cls._app
        with cls._load_lock:
            if cls._app is not None and cls._loaded_model_path == str(MODEL_PATH):
                return cls._app
            return cls._load_locked()

    @classmethod
    def _load_locked(cls):
        import lychee_fd.app as app  # noqa: F401

        status = app.load_models(str(MODEL_PATH), str(TOKEN2WAV_PATH), attn_impl="eager")
        for line in status.splitlines():
            if line.strip():
                print(f"[lychee-fd load_models] {line}", file=sys.stderr)

        cls._app = app
        cls._loaded_model_path = str(MODEL_PATH)
        return app

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        app = await asyncio.to_thread(self._load)

        def _setup_session():
            prompt_wav_path = app._resolve_prompt_wav_path(DEFAULT_PROMPT_VOICE)
            tts_bridge = app.RealtimeTTSPool(
                prompt_wav_path=prompt_wav_path,
                vocoder_hop_size=app.TTS_VOCODER_HOP_SIZE,
                pre_lookahead_len=int(app.get_token2wav_pre_lookahead_len()),
                worker_name="lychee_fd_stop_tts",
                persistent_mode=True,
            )
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

        await start_streaming()
        # Deliberately NOT calling recorder.set_speaking() here -- left in
        # default AUTOMATIC (audio-presence) segmentation mode. See module
        # docstring's "IMPORTANT correction" note: the native
        # round_trace["events"] signal is untrustworthy, so this client no
        # longer uses it for segmentation at all (only logs it for
        # diagnostics); the AUTHORITATIVE model_speech_segments is instead
        # computed post-hoc from output.wav itself once the sample finishes
        # (see recompute_model_speech_segments() below).
        t0 = time.monotonic()

        def now_fn() -> float:
            return time.monotonic() - t0

        source = DynamicPCMSource(initial_pcm=input_pcm.tobytes())

        controller = LiveTriggerController(
            recorder=recorder,
            source=source,
            cue_pcm=self.cue_pcm,
            now_fn=now_fn,
            reference_context=self.reference_context,
            client=self.llm_client,
            lead_in_duration_s=input_duration,
            event_logger=event_logger,
            poll_interval_s=LYCHEE_FD_POLL_INTERVAL_S,
        )
        trigger_task = asyncio.create_task(controller.run())

        def _on_trigger_done(t: asyncio.Task) -> None:
            source.finish()

        trigger_task.add_done_callback(_on_trigger_done)

        round_prefix: Optional[dict] = None
        round_state = "l"
        round_id = 0
        cue_drained_at: Optional[float] = None
        flush_sent = False
        i = 0

        try:
            while True:
                round_id += 1
                round_start_s = now_fn()
                chunk_bytes = source.pull(WINDOW_BYTES)
                chunk_f32 = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # Only send a real flush-tail ONCE (the round where the
                # source first becomes fully drained) -- mirrors
                # infer_lychee_fd.py's own `is_last_round` semantics (true
                # for exactly one round there too, since it iterates a fixed
                # known-length array). Every subsequent post-cue-observation
                # round after that keeps calling the model normally (still
                # feeding real/padding audio so any in-flight recovery
                # response keeps being generated), just without re-signalling
                # a tail flush that has already happened once.
                flush_tail = source.is_done and not flush_sent
                if flush_tail:
                    flush_sent = True

                def _run_round(chunk=chunk_f32, prefix=round_prefix, state=round_state,
                                flush_tail=flush_tail):
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
                # native_events is logged for diagnostics ONLY now -- no
                # longer fed into recorder.set_speaking(). See module
                # docstring's "IMPORTANT correction" note.
                native_events = (
                    _extract_events_from_round_trace(round_trace, round_start_s)
                    if isinstance(round_trace, dict) else []
                )

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

                now = now_fn()
                if source.is_done and cue_drained_at is None:
                    cue_drained_at = now
                    event_logger.log("send", "cue_drained", {"t": cue_drained_at})
                if cue_drained_at is not None and now - cue_drained_at >= POST_CUE_OBSERVATION_S:
                    break

                i += 1
                deadline = t0 + i * INFER_WINDOW_SEC
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        finally:
            def _teardown():
                try:
                    app._close_incremental_stream_session(stream_session, "stop_replay_sample", "sample_end")
                except Exception:
                    pass
                try:
                    tts_bridge.stop(force_flush=False)
                except Exception:
                    pass

            await asyncio.to_thread(_teardown)

            if not trigger_task.done():
                trigger_task.cancel()
                try:
                    await trigger_task
                except (asyncio.CancelledError, Exception):
                    pass
            else:
                try:
                    self.trigger_result = trigger_task.result()
                except Exception as e:
                    self.trigger_result = {"error": f"{type(e).__name__}: {e}"}

            self.simulator_speech_segments = [
                {
                    "start": 0.0,
                    "end": round(input_duration, 4),
                    "text": self.reference_context.get("transcript_user", ""),
                    "source": "lead_in",
                }
            ]
            t_trigger = self.trigger_result.get("t_user_int_start")
            if t_trigger is not None:
                cue_duration = len(self.cue_pcm) / 2 / INPUT_SR
                self.simulator_speech_segments.append({
                    "start": round(t_trigger, 4),
                    "end": round(t_trigger + cue_duration, 4),
                    "text": self.reference_context.get("reference_interruption_text", ""),
                    "source": "cue",
                })


def main():
    stop_cli_main(LycheeFDStopClient, LYCHEE_FD_SLUG, INPUT_SR, DEFAULT_MODEL, output_sr=OUTPUT_SR)


if __name__ == "__main__":
    main()
