#!/usr/bin/env python3
"""
common/local_common.py
==========================================
Shared infrastructure for locally-hosted open-source duplex models, used
alongside `realtime_common.py` (which every model client -- closed- or
open-source -- still uses for `RealtimeClientBase` / `SynchronizedRecorder`
/ `EventLogger`).

Provides:
  - LocalServerManager   : lazily launches a model's own local server (e.g.
                            `python -m moshi.server`) as a background
                            subprocess ONCE per job (not once per sample --
                            model load can take minutes), waits until it
                            accepts TCP connections, and tears it down at
                            interpreter exit.
  - env_for_model()       : per-model HF cache / checkpoint env-var overrides,
                            enforcing the project's environment-isolation
                            policy (never touches ~/.cache).
  - opensource_root_for   : canonical opensource_models/<slug>/ layout.
  - BatchReplayClientBase : for models with NO incremental streaming API at
                            all (e.g. Voila-Autonomous's public
                            `run_generate()`, a single synchronous call over
                            the whole clip) -- see its own docstring.

Design note: unlike the closed-source clients, a local model server is
expensive to start (loading + warming up multi-GB weights takes anywhere
from 30s to several minutes) and typically supports only one *useful*
concurrent session (most open-source servers serialize model access with a
lock internally, e.g. Moshi's `asyncio.Lock`). So the server is started
once, lazily, on the first sample of a batch job, and reused by every
subsequent sample in that same process -- run_inference.py's existing
per-sample loop and --workers flag are otherwise unchanged (just run with
--workers 1 for local models, so samples are naturally serialized instead
of piling up unused connections).
"""

from __future__ import annotations

import abc
import asyncio
import atexit
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

def _duplexact_root() -> Path:
    import os
    for key in ("DUPLEXACT_ROOT", "BENCH_ROOT"):
        v = os.environ.get(key)
        if v:
            return Path(v).resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "evaluation" / "paths.py").exists():
            return p
    return here.parents[min(3, len(here.parents)-1)]

OPENSOURCE_MODELS_ROOT = Path(os.environ.get("OPENSOURCE_MODELS_ROOT") or (_duplexact_root() / "opensource_models"))


def opensource_root_for(slug: str) -> Path:
    """opensource_models/<slug>/ -- venv/, checkpoints/, repo/ live under this."""
    return OPENSOURCE_MODELS_ROOT / slug


def env_for_model(slug: str, extra: Optional[dict] = None) -> dict:
    """
    Env-var overrides that redirect a model's HF cache to its own
    opensource_models/<slug>/checkpoints/ directory -- enforces the
    environment-isolation policy (weights are never cached under the
    default ~/.cache/huggingface, and never shared between models).
    """
    ckpt_dir = opensource_root_for(slug) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # TRANSFORMERS_CACHE and HUGGINGFACE_HUB_CACHE MUST point at the same
    # physical directory ("hub/"), not siblings: `transformers`' own legacy
    # loaders (e.g. AutoImageProcessor.from_pretrained() inside
    # build_vision_tower(), used by infer_vita15.py) pass
    # cache_dir=TRANSFORMERS_CACHE straight through to hf_hub_download(),
    # while raw huggingface_hub calls elsewhere (snapshot_download() with no
    # explicit cache_dir) default to HUGGINGFACE_HUB_CACHE -- if those two
    # env vars point at different directories, a checkpoint downloaded via
    # one code path silently isn't found by the other (this exact bug
    # broke VITA-1.5's InternViT vision-tower load: it downloaded fine into
    # the old separate "transformers/" dir was empty, "hub/" had the real
    # snapshot). Same root cause as the MiniCPM-o 4.5 HF-cache-isolation fix
    # documented elsewhere in this project.
    env = {
        "HF_HOME":                str(ckpt_dir),
        "HUGGINGFACE_HUB_CACHE":  str(ckpt_dir / "hub"),
        "TRANSFORMERS_CACHE":     str(ckpt_dir / "hub"),
        "TORCH_HOME":             str(ckpt_dir / "torch"),
    }
    if extra:
        env.update(extra)
    return env


# ── Local server lifecycle ───────────────────────────────────────────────────

_SERVERS: dict[str, subprocess.Popen] = {}


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _terminate_all() -> None:
    for key, proc in list(_SERVERS.items()):
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


atexit.register(_terminate_all)


class LocalServerManager:
    """
    Lazily launches (once per process, cached by `key`) a model's own local
    server as a background subprocess, and blocks (off the event loop, via
    `asyncio.to_thread`) until it accepts TCP connections on `port` -- or
    raises if it exits early or never becomes ready within
    `startup_timeout_s`.
    """

    @staticmethod
    async def ensure(
        key: str,
        cmd: list[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        host: str = "localhost",
        port: int = 8998,
        log_path: Optional[Path] = None,
        startup_timeout_s: float = 900.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        existing = _SERVERS.get(key)
        if existing is not None and existing.poll() is None:
            return  # already running and healthy

        def _launch() -> subprocess.Popen:
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_fh = open(log_path, "w")
            else:
                log_fh = subprocess.DEVNULL
            merged_env = {**os.environ, **(env or {})}
            return subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=merged_env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,   # own process group -> clean teardown
            )

        proc = await asyncio.to_thread(_launch)
        _SERVERS[key] = proc

        def _wait_ready() -> None:
            deadline = time.monotonic() + startup_timeout_s
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Local server for {key!r} exited early (code {proc.returncode}). "
                        f"Check log: {log_path}"
                    )
                if _port_open(host, port):
                    return
                time.sleep(poll_interval_s)
            raise TimeoutError(
                f"Local server for {key!r} did not become ready on {host}:{port} "
                f"within {startup_timeout_s}s. Check log: {log_path}"
            )

        await asyncio.to_thread(_wait_ready)

    @staticmethod
    def stop(key: str) -> None:
        proc = _SERVERS.pop(key, None)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                proc.kill()


# ── Batch-replay clients (no incremental streaming API at all) ─────────────

class BatchReplayClientBase(abc.ABC):
    """
    For models with NO incremental/real-time streaming API whatsoever --
    e.g. Voila-Autonomous, whose only public entry point
    (`VoilaAutonomousModel.run_generate`) is a single synchronous call that
    consumes the entire pre-tokenized input in one autoregressive loop with
    no wall-clock pacing, and returns the complete response only once
    finished. Contrast with `RealtimeClientBase` (realtime_common.py),
    which drives genuinely incremental clients (websocket frames,
    streaming_prefill/generate chunks) in real wall-clock time via
    `SynchronizedRecorder`.

    Precedent for not forking the model to add fake real-time pacing: even
    Voila's own authors only ever exercise the autonomous checkpoint via
    one batch call over a full pre-recorded clip -- their official
    `gradio_demo.py` doesn't use the autonomous checkpoint for live
    interaction at all (it loads the turn-based `Voila-chat` model
    instead). So calling the model once per sample here doesn't sacrifice
    any real-time rigor the original implementation had, since it never
    had any to begin with.

    Segmentation caveat: because there's no incremental delivery to watch
    "is there audio in the pipe right now" on (unlike
    SynchronizedRecorder's automatic tracking, or Moshi's/MiniCPM's
    explicit per-chunk speaking signals), `model_speech_segments` here are
    reconstructed post-hoc via simple RMS-energy thresholding over the
    full returned waveform -- a coarser, heuristic substitute for a real
    "is speaking" signal, and the whole reason this is a *last-resort*
    fallback category rather than the default. Tune SILENCE_RMS_DBFS /
    MIN_GAP_S / MIN_DUR_S per model after inspecting real smoke-test
    output (same kind of calibration infer_moshi.py's
    MOSHI_SILENCE_GRACE_S and infer_minicpm_o.py's is_listen handling
    needed).
    """
    MODEL_SLUG: str = "unknown"
    OUTPUT_SR: int = 16_000
    INPUT_SR: int = 16_000

    SILENCE_RMS_DBFS = -40.0    # below this (dBFS rel. to this clip's own peak) = silence
    ENERGY_WINDOW_S  = 0.02     # 20ms analysis windows
    MIN_GAP_S        = 0.3      # merge speech runs separated by less than this
    MIN_DUR_S        = 0.15     # discard speech runs shorter than this (spurious blips)

    def __init__(self, api_key: str, model_name: str, system_prompt: str, verbose: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose

    @classmethod
    async def warmup(cls) -> None:
        """
        Optional one-time, untimed hook run by run_inference.py exactly once
        per process, BEFORE the per-sample SAMPLE_TIMEOUT_OVERHEAD_S-guarded
        loop starts. Default no-op. Override for clients that cache an
        expensive resource at the CLASS level (e.g. a multi-GB local
        checkpoint) whose one-time load cost can alone exceed a single
        sample's timeout budget -- see run_inference.py's WARMUP_TIMEOUT_S
        comment for the exact failure mode this prevents (concurrent
        duplicate loads racing each other into GPU OOM after a timeout+retry).
        """
        return None

    async def run_sample(self, input_wav: Path, output_wav: Path, events_path: Path,
                          input_duration: float) -> dict:
        from realtime_common import load_and_resample, EventLogger

        t0 = time.monotonic()
        event_logger = EventLogger(events_path, t0)
        event_logger.finalize_start_time(t0)   # no handshake phase to buffer -- t=0 now

        input_pcm = load_and_resample(input_wav, self.INPUT_SR)
        event_logger.log("send", "batch_call_start")

        errors: list[str] = []
        target_samples = int(round(input_duration * self.OUTPUT_SR))
        output_f32 = np.zeros(target_samples, dtype=np.float32)
        try:
            output_f32 = await self._generate_batch(input_pcm, input_duration)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            event_logger.log("recv", "client_error", {"message": str(e)})

        event_logger.log("recv", "batch_call_end")

        native_segments = self._native_segments()

        # Pad up to input_duration if the model returned less audio than
        # that (silence-fill, same convention as the real-time clients'
        # output.wav) -- but never truncate: unlike input_duration (a fixed
        # property of the input clip), the model's own reply is allowed to
        # run past it, exactly as the real-time clients let trailing speech
        # extend beyond input_duration for as long as the model keeps
        # talking (see RealtimeClientBase.run_sample -- it only pads up to
        # input_duration if the client finished early, it never cuts a
        # still-speaking model off at that mark).
        if len(output_f32) < target_samples:
            output_f32 = np.concatenate(
                [output_f32, np.zeros(target_samples - len(output_f32), dtype=np.float32)]
            )

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), output_f32, self.OUTPUT_SR, subtype="PCM_16")

        segments = native_segments if native_segments is not None else \
            self._energy_segments(output_f32, self.OUTPUT_SR)
        for s, e in segments:
            event_logger.log("recv", "speech_segment", {"start": s, "end": e})

        event_logger.close()
        wallclock = time.monotonic() - t0
        return {
            "model":                 self.model_name,
            "input_duration_s":      round(input_duration, 4),
            "output_duration_s":     round(len(output_f32) / self.OUTPUT_SR, 4),
            "model_speech_segments": [list(s) for s in segments],
            "interruptions":         [],
            "errors":                errors,
            "inference_wallclock_s": round(wallclock, 4),
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }

    def _energy_segments(self, audio_f32: np.ndarray, sr: int) -> list[tuple[float, float]]:
        win = max(1, int(self.ENERGY_WINDOW_S * sr))
        n_win = len(audio_f32) // win
        if n_win == 0:
            return []
        trimmed = audio_f32[: n_win * win].reshape(n_win, win)
        rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
        peak = float(np.max(np.abs(audio_f32))) + 1e-9
        dbfs = 20.0 * np.log10(rms / peak + 1e-12)
        speaking = dbfs > self.SILENCE_RMS_DBFS

        raw_segments: list[tuple[float, float]] = []
        open_start: Optional[float] = None
        for i, sp in enumerate(speaking):
            t = i * self.ENERGY_WINDOW_S
            if sp and open_start is None:
                open_start = t
            elif not sp and open_start is not None:
                raw_segments.append((open_start, t))
                open_start = None
        if open_start is not None:
            raw_segments.append((open_start, n_win * self.ENERGY_WINDOW_S))

        merged: list[tuple[float, float]] = []
        for s, e in raw_segments:
            if merged and s - merged[-1][1] <= self.MIN_GAP_S:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        merged = [(s, e) for s, e in merged if (e - s) >= self.MIN_DUR_S]
        return [(round(s, 4), round(e, 4)) for s, e in merged]

    def _native_segments(self) -> Optional[list[tuple[float, float]]]:
        """
        Optional override: if the subclass's just-completed
        `_generate_batch()` call captured an EXACT model-native
        speaking-state signal (e.g. Voila's `streamer=` hook into
        `run_generate`, which exposes the same internal `speaking` boolean
        the model itself tracks -- see infer_voila.py), return the
        resulting segments here instead of falling back to the base
        class's coarse energy-threshold heuristic. Called immediately
        after `_generate_batch()` returns. `None` (the default) means "no
        native signal available for this model, use energy thresholding."
        """
        return None

    @abc.abstractmethod
    async def _generate_batch(self, input_pcm: np.ndarray, input_duration_s: float) -> np.ndarray:
        """
        Model-specific: run the model's own one-shot batch generation over
        the WHOLE input_pcm (int16 @ self.INPUT_SR) and return the model's
        full response waveform as float32 in [-1, 1] @ self.OUTPUT_SR,
        assumed frame-aligned 1:1 with input_pcm's own timeline (i.e.
        output sample i corresponds to conversation-time i/OUTPUT_SR) --
        the base class pads/truncates to input_duration_s afterward.
        """
        raise NotImplementedError
