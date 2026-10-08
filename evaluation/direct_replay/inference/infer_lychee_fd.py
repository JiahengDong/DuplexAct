#!/usr/bin/env python3
"""
direct_replay/inference/infer_lychee_fd.py
==========================================
HITsz-TMG/Lychee-FD client. Category A/B (genuine incremental streaming,
in-process, no network server) -- upgraded from an earlier Category C
(single batch call) implementation once investigation confirmed the
official release's `app.py` ships a real, working live/incremental driver
(the same one its own browser demo and `scripts/start_realtime_all.sh` use),
not just a Gradio "upload one file" batch path. See §2 of
`model_docs/opensource/lychee-fd.md` for the full history of why the batch
fallback was chosen first and what changed.

How this differs from the old batch-call client: instead of one call to
`app.run_chunk_dialogue_inference(..., stream_session=None)` over the whole
clip at GPU speed, this client reproduces the CORE of `app.py`'s own
`_run_realtime_session_worker()` round loop (the function its live
SSE/browser session handler runs) -- minus the multi-user/SSE/locking
scaffolding we don't need for a single offline sample:

  1. Create ONE incremental stream session up front via the loaded
     generator's own `create_hf_incremental_stream_session()` (HF backend
     only -- we never load the vLLM backend, see infer_lychee_fd.py's
     sibling model_docs entry) and ONE persistent-mode `RealtimeTTSPool`
     (`persistent_mode=True`, no `pcm_emit_callback` needed since we drain
     `audio_out` straight from the generator's own yielded tuples, exactly
     like the old batch client already did -- app.py's own for-loop over
     `inference_gen` does the same thing regardless of tts_bridge mode).
  2. Feed `input_pcm` in sequential, REAL-wall-clock-paced windows
     (`INFER_WINDOW_MS`, matching `app.py`'s own
     `LYCHEEFD_REALTIME_INFER_WINDOW_MS` default of 400ms) -- each round's
     `run_chunk_dialogue_inference(..., stream_session=session,
     stream_audio_incremental=True, tts_bridge=<persistent pool>)` call is
     only dispatched once that much real time has genuinely elapsed since
     stream start, so response latency now measures real wall-clock time
     (unlike Voila-Autonomous, whose only entry point has no such pacing
     option at all -- see voila-autonomous.md).
  3. Carry model state across rounds via `round_ctx["next_prefix"]` /
     `round_ctx["next_listening_state"]` -- the exact mechanism app.py's own
     loop uses (`session.realtime_prefix` / `session.realtime_listening_state`
     in app.py, just held as plain local variables here since we only ever
     run one session per sample, never concurrently).
  4. Close the stream session (`app._close_incremental_stream_session`) and
     stop the TTS pool (`tts_bridge.stop(force_flush=False)`) once all of
     input_pcm (including its own trailing padding) has been fed and the
     final round (`stream_flush_audio_tail=True`) returns -- matching this
     benchmark's hard-cutoff-at-input_duration policy shared by every other
     client (see realtime_common.py's module docstring).

`model_speech_segments` still comes from the native `round_trace["events"]`
signal, exactly as the old batch client used it -- fed into the recorder via
`recorder.set_speaking(True/False, at_s=...)` per event (each event's
`start_rel_ms`/`end_rel_ms` converted to absolute sample-timeline seconds by
adding that round's own real wall-clock dispatch offset). We follow the
official pipeline's own behavior here as-is, including a known wart: traced
through `hf_v9_realtime.py`/app.py and found that the per-round "currently
open speaking event" bookkeeping (`current_event_trace`) is a plain
function-local variable, reset to `None` at the top of EVERY
`run_chunk_dialogue_inference()` call -- it is NOT part of what
`next_prefix`/`next_listening_state` carry forward across rounds (only the
raw LLM token prefix and the coarse l/s/b state are). There IS a
`keep_alive_for_speaking` mechanism that looks purpose-built to solve
exactly this continuity problem, but grep confirms it only exists in
`vllm_generation.py` (the vLLM backend) -- `hf_v9_realtime.py` (the HF
backend this client, and the official release's own default non-vLLM
deployment path, both actually drive) has no equivalent. Concretely: this
means a reply that keeps speaking across one of our 400ms round boundaries
(likely, since replies routinely run several seconds) can be recorded as a
new event starting at ~0ms into the next round rather than a continuation
of the previous one -- i.e. this is a genuine limitation of the model's own
official HF-backend round-based session design, not something introduced
by this client. We deliberately do NOT layer our own cross-round stitching
heuristic on top to "fix" this, since that would deviate from how the
official pipeline itself behaves; any fragmentation this produces is
Lychee-FD's own official behavior under its own official incremental API,
faithfully reproduced.

## 11. vLLM backend (default, upgraded from the HF backend)

The public repo frames the Docker+vLLM pipeline (`lychee_fd.vllm_integration.
engine`, `lychee_fd.runtime.vllm_generation.VLLMGenerationFramework`) as its
own "recommended production path" (`model_docs/opensource/lychee-fd.md` §1);
we now drive that path directly (no Docker/container involved -- same
in-process approach as the HF backend, just a different `app.load_models()`
branch, selected purely via `LYCHEEFD_USE_VLLM=1`, both backends' `load_models`
call signature is identical, see `app.py`'s own `load_models()`).

Two concrete things this gets us over the HF backend (§2/§3's documented
limitation): `VLLMGenerationFramework.create_incremental_stream_session()`
(vllm_generation.py) sets `keep_alive_for_speaking=True` on its session,
reusing the same vLLM *request* across consecutive speaking-decode calls
within one open utterance -- unlike `create_hf_incremental_stream_session()`
(available on BOTH backends, for HF-compatible semantics; that's what this
client used to call unconditionally), which never reuses a request and is
therefore the actual cause of the cross-round event-fragmentation wart
documented in lychee-fd.md §2. `_setup_session()` below now calls whichever
of the two `app.generator` actually exposes preferring the native one --
copying `app.py`'s own dispatch order verbatim (see its
`_run_realtime_session_worker()`) rather than hardcoding one name.

Setup this requires that the HF backend didn't: `VLLMGenerationFramework`
imports `vllm.model_executor.models.registry.ModelRegistry`/`SamplingParams`/
etc. and, critically, `MultiHeadRequestState` -- a class that exists ONLY in
the repo's own patched vLLM fork (`third_party/vllm/vllm/sequence.py`), not
in vanilla pip `vllm==0.6.5` (confirmed by direct import test). Per the
repo's own README: "The online vLLM backend requires two pieces at the same
time: an installed vLLM wheel ... which provides compiled native libraries
... _C.abi3.so, _moe_C.abi3.so, vllm_flash_attn_c.abi3.so; the patched source
tree in third_party/vllm, which implements the Lychee-FD multi-stream
serving path... importing plain site-packages vLLM will not use the
Lychee-FD serving implementation." We therefore (one-time, already done for
this venv): (1) `pip install vllm==0.6.5` into `opensource_models/lychee-fd/
venv` (reuses the venv's existing `torch==2.5.1+cu121`, unchanged) purely as
a source of compiled native artifacts; (2) copied those artifacts
(`_C.abi3.so`, `_moe_C.abi3.so`, the whole `vllm_flash_attn/` dir,
`_version.py`) from the venv's installed `vllm/` package into `third_party/
vllm/vllm/` (the exact copy `scripts/start_backend.sh`'s own
`sync_vllm_native_artifacts()` performs); (3) prepend `VLLM_SOURCE_DIR`
(`third_party/vllm`) to `sys.path` here, before the pip-installed copy can
ever be imported -- confirmed via `import vllm; vllm.MultiHeadRequestState`
resolving correctly with this ordering. All of this -- checkpoints, venv,
and the patched vllm source tree -- lives under the same `punim2758`-backed
`opensource_models/lychee-fd/` mount already used for every other model's
large storage, not `punim2960`.

Not yet GPU-validated end-to-end (written and import-tested only) at the
time of this change -- Spartan's GPU partitions were administratively DOWN
for cluster maintenance. A smoke-test job is queued (`slurm_smoke_lychee_fd_
vllm.sh`) to auto-run once partitions reopen; see that job's log before
trusting any real inference results from this backend. Set
`LYCHEEFD_USE_VLLM=0` to fall back to the previously-smoke-tested HF backend
(§2-§9 above) if the vLLM path turns out to need further fixes.

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

# Matches app.py's own LYCHEEFD_REALTIME_INFER_WINDOW_MS default -- the exact
# per-round window size the authors' own live browser session uses.
INFER_WINDOW_MS = int(os.environ.get("LYCHEEFD_REALTIME_INFER_WINDOW_MS", "400"))

REPO_DIR = opensource_root_for(LYCHEE_FD_SLUG) / "repo"
STEPAUDIO2_SOURCE_DIR = REPO_DIR / "third_party" / "Step-Audio2"
VLLM_SOURCE_DIR = REPO_DIR / "third_party" / "vllm"     # patched vLLM fork -- see §11 below
CKPT_ROOT = opensource_root_for(LYCHEE_FD_SLUG) / "checkpoints" / "model_root"
MODEL_PATH = CKPT_ROOT / "lychee_full_duplex"
TOKEN2WAV_PATH = CKPT_ROOT / "token2wav"

os.environ.setdefault("STEPAUDIO2_SOURCE_DIR", str(STEPAUDIO2_SOURCE_DIR))
# vLLM backend -- the repo's own "recommended production path" (see module
# docstring §11) and now our default: gives native `keep_alive_for_speaking`
# (fixes the HF backend's cross-round event-fragmentation limitation, see
# lychee-fd.md §2) plus a real PagedAttention KV cache instead of HF's
# `transformers`-generic one. Requires the patched `third_party/vllm`
# fork's Python source ahead of the pip-installed vanilla `vllm==0.6.5` on
# sys.path (below) -- importing plain site-packages vllm would silently
# miss `MultiHeadRequestState`/`multihead_sampling` and this whole backend
# would refuse to construct a request. Set LYCHEEFD_USE_VLLM=0 to fall back
# to the HF backend (§2's Category-C-safe path) for debugging.
os.environ.setdefault("LYCHEEFD_USE_VLLM", "1")
os.environ.setdefault("LYCHEEFD_T2W_REMOTE_ENABLED", "0")  # load Token2Wav in-process
os.environ.setdefault("LYCHEEFD_STARTUP_WARMUP", "0")     # we warm up on first real sample instead
# vLLM engine tuning -- conservative defaults for a single A100 80GB, no
# tensor/pipeline parallelism needed for one ~24GB checkpoint. enforce_eager
# stays at the module's own default (True) to skip CUDA-graph capture
# (slower per-step, but avoids graph-capture edge cases with the custom
# multi-branch model architecture -- not something we've stress-tested).
os.environ.setdefault("LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION", "0.85")
os.environ.setdefault("LYCHEEFD_VLLM_MAX_MODEL_LEN", "8192")
# Official, opt-in flag (see vllm_generation.py's `_close_incremental_stream_
# session`-equivalent cleanup path) that calls `torch.cuda.empty_cache()` +
# `ipc_collect()` after each session's own gc.collect(). Defaults to "0"
# upstream because a live multi-user server closing/reopening sessions
# constantly shouldn't pay that sync cost per-request. We're the opposite
# case: one long-lived process runs HUNDREDS of sequential offline samples
# back-to-back (see run_lychee_fd_vllm_validation.sh), and non-vLLM-pool
# Python-side tensors (audio feature/embed caches, HF-path KV cache refs --
# see the attrs this same cleanup routine nulls out just above the gc.collect
# call) only actually return their CUDA blocks to the driver once this flag
# is on; left off, we saw allocated memory climb sample-over-sample (49GB ->
# 79GB over ~30 samples) until a real OutOfMemoryError. Turning this on
# doesn't touch model weights/behavior/outputs at all -- purely a device
# memory bookkeeping call -- so it's a zero-bias, official-mechanism fix.
os.environ.setdefault("LYCHEEFD_CUDA_EMPTY_CACHE_ON_SESSION_CLOSE", "1")

USE_VLLM = os.environ.get("LYCHEEFD_USE_VLLM", "1") == "1"

if USE_VLLM:
    # Must be inserted BEFORE any `import vllm` happens (including
    # transitively, e.g. via `lychee_fd.app` -> `vllm_generation.py`) --
    # Python caches modules by name on first import, so if plain
    # site-packages vllm gets imported first, the patched tree is never
    # consulted for the rest of the process.
    if str(VLLM_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(VLLM_SOURCE_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(STEPAUDIO2_SOURCE_DIR) not in sys.path:
    sys.path.append(str(STEPAUDIO2_SOURCE_DIR))


def _extract_events_from_round_trace(round_trace: dict, round_offset_s: float) -> list[dict]:
    """
    round_trace["events"] entries look like {"kind": "response",
    "start_rel_ms": ..., "end_rel_ms": ...}, relative to THIS round's own
    call start (`t_stream_start_perf`, a real `time.perf_counter()` read
    taken fresh at the top of every `run_chunk_dialogue_inference()` call).
    `round_offset_s` is this round's real wall-clock dispatch time relative
    to the sample's own stream start (the same origin `recorder`/
    `event_logger` use), so adding it converts each event onto the sample's
    single absolute timeline. These fully-closed (start AND end present)
    events become `recorder.set_speaking()` calls -- see module docstring
    for the official-pipeline HF-backend caveat about events that span a
    round boundary.
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
    # The official release only ever calls `app.load_models()` once, single-
    # threaded, before its server starts accepting requests (see app.py's
    # `__main__` block: `load_models(...)` runs to completion, THEN the
    # FastAPI/Gradio app starts serving). We replicate that exact single-
    # load guarantee here for our own concurrent (`--workers N`) driver:
    # without this lock, two worker threads can both observe `cls._app is
    # None` and both call `app.load_models(...)` concurrently, racing to
    # construct the same vLLM engine/register the same custom layers twice
    # in one process -- this is what caused the "Duplicate layer name" /
    # KeyError(".model.layers.N.self_attn.attn") failures in the first
    # validation batch, NOT a bug in the model or the official code itself.
    # Serializing this is pure client-side concurrency control -- it changes
    # nothing about how the model is loaded or how it behaves once loaded.
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

        # StepAudio2ForCausalLM (the checkpoint's own bundled modeling class,
        # loaded via the lychee_fd.models.lychee_duplex shim -- see that
        # file's docstring) doesn't implement an sdpa attention path at all
        # ("does not support an attention implementation through
        # torch.nn.functional.scaled_dot_product_attention yet"), unlike
        # Voila/MiniCPM-o/VITA-1.5 where sdpa was available and chosen only
        # to dodge a from-source flash-attn compile. "eager" is the only
        # option this specific architecture supports in this transformers
        # version.
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
            # Prefer the native/generic session (keep_alive_for_speaking-
            # capable when the backend supports it, see module docstring
            # §11) over the HF-compatibility-shim name -- exact dispatch
            # order app.py's own `_run_realtime_session_worker()` uses.
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

        # Force manual-segment mode from t=0, before any audio is added.
        # Without this, SynchronizedRecorder's own automatic (queue-presence)
        # tracking would run for however many early rounds pass before the
        # FIRST native round_trace event actually closes (see module
        # docstring -- closing can lag several rounds behind a real speech
        # onset), auto-opening a segment on real audio presence; the first
        # set_speaking() call below would then permanently switch to manual
        # mode mid-flight and abandon that auto-opened segment (it only ever
        # gets closed, incorrectly spanning to end-of-recording, by run()'s
        # unconditional final _close_open_segment() call). This no-op call
        # (nothing is open yet, so it appends nothing) just flips
        # `_manual_segments` on early so the two mechanisms never overlap.
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

                # Pace DISPATCH to real wall-clock time: only call the model
                # with this round's slice once that much of input.wav's own
                # timeline has genuinely elapsed -- otherwise we're back to
                # gulping audio faster than it was "spoken" (Voila's problem).
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
