#!/usr/bin/env python3
"""
Shared infrastructure for locally hosted open-source duplex models, used
alongside `realtime_common.py`.

  - LocalServerManager    : launches a model's local server (e.g.
                            `python -m moshi.server`) once per process, waits
                            until it accepts TCP connections, and stops it at
                            interpreter exit.
  - env_for_model()       : per-model HF cache / checkpoint env-var overrides
                            (keeps weights out of ~/.cache).
  - opensource_root_for() : opensource_models/<slug>/ layout.
  - BatchReplayClientBase : for models with no incremental streaming API
                            (one synchronous call over the whole clip).

Local servers are slow to start and usually serialize model access
internally, so the server is reused across all samples in a process; run
local models with `--workers 1`.
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
    opensource_models/<slug>/checkpoints/ directory, so weights are never
    cached under ~/.cache/huggingface or shared between models.
    """
    ckpt_dir = opensource_root_for(slug) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # TRANSFORMERS_CACHE and HUGGINGFACE_HUB_CACHE must be the same directory:
    # some transformers loaders pass cache_dir=TRANSFORMERS_CACHE to
    # hf_hub_download(), while snapshot_download() defaults to
    # HUGGINGFACE_HUB_CACHE, and each would otherwise miss the other's files.
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
    Launches a model's local server as a background subprocess (once per
    process, cached by `key`) and waits off the event loop until it accepts
    TCP connections on `port`. Raises if it exits early or is not ready
    within `startup_timeout_s`.
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
    For models with no incremental streaming API, e.g. Voila-Autonomous,
    whose only entry point (`VoilaAutonomousModel.run_generate`) consumes the
    whole input in one call and returns the complete response. Real-time
    clients use `RealtimeClientBase` (realtime_common.py) instead.

    Unless the subclass provides a native speaking signal via
    `_native_segments()`, `model_speech_segments` are reconstructed from the
    returned waveform by RMS-energy thresholding (SILENCE_RMS_DBFS,
    MIN_GAP_S, MIN_DUR_S; tune per model).
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
        Optional one-time hook run by run_inference.py once per process,
        before the per-sample timeout applies. Default no-op. Override for
        clients that load an expensive class-level resource (e.g. a multi-GB
        checkpoint) whose load time could exceed one sample's timeout.
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

        # Pad with silence up to input_duration, but never truncate: as in the
        # real-time clients, the reply may run past input_duration.
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
        Optional override, called right after `_generate_batch()`: return
        segments from a model-native speaking signal captured during that
        call (e.g. Voila's `streamer=` hook, see infer_voila.py). `None`
        (the default) means use energy thresholding.
        """
        return None

    @abc.abstractmethod
    async def _generate_batch(self, input_pcm: np.ndarray, input_duration_s: float) -> np.ndarray:
        """
        Run one-shot generation over the whole input_pcm (int16 @ INPUT_SR)
        and return the response as float32 in [-1, 1] @ OUTPUT_SR, aligned
        with the input timeline (output sample i is at time i/OUTPUT_SR).
        The base class pads it to input_duration_s.
        """
        raise NotImplementedError
