#!/usr/bin/env python3
"""
Batch driver for the simulator-replay clients (infer_stop_<model>.py) over every
sample in evaluation/data/stop/direct_interruption/.

For each sample it builds the argparse.Namespace that `run_stop_sample_cli()`
(stop_common.py) expects and calls it in-process, so local models start their
server once per process and reuse it across samples. Like
direct_replay/inference/run_inference.py, it skips finished samples, retries
transient errors, and writes a per-model run manifest. Samples contain
lead_in.wav / cue.wav / metadata.json.

Only models with a live text signal are registered: the trigger LLM decides
when to inject the cue from what the model is saying (`.on_text()`). lychee-fd
has no token-level text signal and is excluded.

Usage:
    python run_stop_batch.py --model moshi --workers 1
    python run_stop_batch.py --model gpt-realtime-2.1 --workers 8
    python run_stop_batch.py --model raon-speechchat --limit 5 --verbose
    python run_stop_batch.py --model minicpm-o-4.5 --overwrite --sample_ids AMI_stop_0000,Candor_stop_0038
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))


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

REPO_ROOT = _duplexact_root()
DEFAULT_DATA_ROOT = REPO_ROOT / "evaluation" / "data" / "stop" / "direct_interruption"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation" / "simulator_replay" / "results" / "stop"

# Per-sample timeout headroom on top of the lead-in duration: covers
# connect/handshake, the wait for the trigger (trigger_controller.py), the
# cue, and POST_CUE_OBSERVATION_S (stop_common.py).
SAMPLE_TIMEOUT_OVERHEAD_S = 120.0

_RETRYABLE_ERROR_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"too many concurrent sessions",
        r"thread pool exausted",
        r"thread pool exhausted",
        r"rate limit exceeded",
        r"ConnectionClosedError",
        r"TimeoutError",
        r"connect call failed",
        r"insufficient_quota",
        r"credit_balance",
        r"HTTP 403",
        r"InvalidStatus",
    ]
]
RETRYABLE_ERROR_RETRIES = 3
RETRYABLE_ERROR_BASE_DELAY_S = 5.0


def _is_retryable_error(error_text: str) -> bool:
    return any(p.search(error_text) for p in _RETRYABLE_ERROR_PATTERNS)


# ── Model registry ───────────────────────────────────────────────────────────
# Each entry mirrors the stop_cli_main() call in infer_stop_<model>.py.
# `local=True` forces workers=1 (one local model server per process).

@dataclass
class ModelEntry:
    module: str
    client_cls_attr: str
    api_sr_attr: str          # attribute name on the module holding api_sr
    default_model_attr: str
    local: bool
    output_sr_attr: Optional[str] = None
    api_key_env: Optional[str] = None
    client_kwargs: dict = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "moshi": ModelEntry("infer_stop_moshi", "MoshiStopClient", "API_SR", "DEFAULT_MODEL", local=True),
    "flm-audio": ModelEntry("infer_stop_flm_audio", "FLMAudioStopClient", "API_SR", "DEFAULT_MODEL", local=True),
    "bayling-duplex": ModelEntry("infer_stop_bayling", "BayLingDuplexStopClient", "INPUT_SR", "DEFAULT_MODEL",
                                  local=True, output_sr_attr="API_SR"),
    "raon-speechchat": ModelEntry("infer_stop_raon", "RaonSpeechChatStopClient", "API_SR", "DEFAULT_MODEL",
                                   local=True),
    "minicpm-o-4.5": ModelEntry("infer_stop_minicpm", "MiniCPMStopClient", "INPUT_SR", "DEFAULT_MODEL",
                                 local=True, output_sr_attr="OUTPUT_SR"),
    "freeze-omni": ModelEntry("infer_stop_freeze_omni", "FreezeOmniStopClient", "INPUT_SR", "FREEZE_OMNI_REPO_ID",
                               local=True, output_sr_attr="OUTPUT_SR"),
    "gpt-realtime-2.1": ModelEntry("infer_stop_gpt_realtime", "GPTRealtimeStopClient", "API_SR", "DEFAULT_MODEL",
                                    local=False, api_key_env="OPENAI_API_KEY"),
    "gemini-3.1-flash-live": ModelEntry("infer_stop_gemini_live", "GeminiLiveStopClient", "SEND_SAMPLE_RATE",
                                         "DEFAULT_MODEL", local=False, output_sr_attr="RECEIVE_SAMPLE_RATE",
                                         api_key_env="GEMINI_API_KEY"),
    "grok-voice": ModelEntry(
        "infer_stop_grok_voice", "GrokVoiceStopClient", "API_SR", "DEFAULT_MODEL",
        local=False, api_key_env="XAI_API_KEY",
        client_kwargs={"voice": "eve"},
    ),
    "qwen3.5-omni-flash-realtime": ModelEntry(
        "infer_stop_qwen_realtime", "QwenRealtimeStopClient", "API_SR", "DEFAULT_MODEL",
        local=False, output_sr_attr="OUTPUT_SR", api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"voice": "Ethan"},
    ),
    # lychee-fd is excluded: no live text signal for the trigger LLM.
}


def _load_entry(model_slug: str):
    import importlib
    entry = MODEL_REGISTRY.get(model_slug)
    if entry is None:
        raise SystemExit(
            f"Model '{model_slug}' is not registered for the stop-action batch runner "
            f"(either lacks a live text signal, or is blocked -- see module docstring). "
            f"Registered: {sorted(MODEL_REGISTRY)}"
        )
    mod = importlib.import_module(entry.module)
    client_cls = getattr(mod, entry.client_cls_attr)
    api_sr = getattr(mod, entry.api_sr_attr)
    default_model = getattr(mod, entry.default_model_attr)
    output_sr = getattr(mod, entry.output_sr_attr) if entry.output_sr_attr else None
    return entry, client_cls, api_sr, default_model, output_sr


def _existing_result_ok(out_dir: Path) -> bool:
    output_wav = out_dir / "output.wav"
    info_path = out_dir / "inference_info.json"
    if not (output_wav.exists() and info_path.exists()):
        return False
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return False
    return not info.get("inference", {}).get("errors")


def _gc_orphaned_attempts(model_dir: Path) -> None:
    if not model_dir.is_dir():
        return
    for child in model_dir.iterdir():
        if child.is_dir() and (child.name.startswith(".attempt-") or child.name.startswith(".backup-")):
            shutil.rmtree(child, ignore_errors=True)


def _publish_attempt(attempt_dir: Path, out_dir: Path) -> None:
    backup_dir = None
    if out_dir.exists():
        backup_dir = out_dir.parent / f".backup-{out_dir.name}-{uuid.uuid4().hex[:8]}"
        os.rename(out_dir, backup_dir)
    try:
        os.rename(attempt_dir, out_dir)
    except Exception:
        if backup_dir is not None:
            os.rename(backup_dir, out_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


async def run_one_sample(
    sample_dir: Path, out_dir: Path, model_slug: str, client_cls, api_sr: int,
    default_model: str, output_sr: Optional[int], client_kwargs: dict,
    api_key_env: Optional[str], verbose: bool,
) -> tuple[str, str, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stop_common import run_stop_sample_cli  # noqa: E402

    sample_id = sample_dir.name
    attempt_dir = out_dir.parent / f".attempt-{sample_id}-{uuid.uuid4().hex[:8]}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        lead_in_duration = sf.info(str(sample_dir / "lead_in.wav")).duration
    except Exception:
        lead_in_duration = None
    # Headroom after lead-in: T_SEARCH_S=8s wall from reply onset (or
    # T_ONSET_S=5s if silent) plus POST_CUE_OBSERVATION_S.
    sample_timeout = (lead_in_duration or 10.0) + SAMPLE_TIMEOUT_OVERHEAD_S + 20.0

    args = argparse.Namespace(
        sample_dir=sample_dir,
        output_wav=attempt_dir / "output.wav",
        events_path=attempt_dir / "events.jsonl",
        info_path=attempt_dir / "inference_info.json",
        model=default_model,
        verbose=verbose,
    )

    try:
        try:
            info = await asyncio.wait_for(
                run_stop_sample_cli(client_cls, model_slug, api_sr, args,
                                     client_kwargs=client_kwargs, api_key_env=api_key_env,
                                     output_sr=output_sr),
                timeout=sample_timeout,
            )
        except asyncio.TimeoutError:
            return sample_id, "error", {"error": f"TimeoutError: exceeded hard timeout ({sample_timeout:.0f}s)"}
        except Exception as e:
            tb = traceback.format_exc()
            if verbose:
                print(f"  [ERROR] {sample_id}: {e}\n{tb}", file=sys.stderr)
            return sample_id, "error", {"error": f"{type(e).__name__}: {e}"}

        if not (attempt_dir / "output.wav").exists() or not (attempt_dir / "inference_info.json").exists():
            return sample_id, "error", {"error": "attempt missing output.wav/inference_info.json"}

        _publish_attempt(attempt_dir, out_dir)

        errs = info.get("inference", {}).get("errors") or []
        status = "error" if errs else "ok"
        n_segs = len(info.get("inference", {}).get("model_speech_segments", []))
        forced = info.get("inference", {}).get("trigger", {}).get("forced")
        if verbose or status == "error":
            print(f"  [{status.upper()}] {sample_id}  segments={n_segs}  forced={forced}  errors={errs}")
        detail = {"n_segments": n_segs, "forced": forced}
        if errs:
            detail["error"] = "; ".join(str(e) for e in errs)
        return sample_id, status, detail
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)


async def run_batch(model_slug: str, data_root: Path, results_root: Path, workers: int,
                     overwrite: bool, limit: Optional[int], sample_ids: Optional[list[str]],
                     verbose: bool) -> None:
    entry, client_cls, api_sr, default_model, output_sr = _load_entry(model_slug)
    workers = 1 if entry.local else workers

    if entry.api_key_env and not os.environ.get(entry.api_key_env):
        from dotenv import load_dotenv
        load_dotenv()
    if entry.api_key_env and not os.environ.get(entry.api_key_env):
        raise SystemExit(f"No API key found in ${entry.api_key_env} (env / .env)")

    all_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if sample_ids:
        wanted = set(sample_ids)
        all_dirs = [p for p in all_dirs if p.name in wanted]

    model_dir = results_root / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = model_dir / "_run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text()) if run_manifest_path.exists() else {}
    _gc_orphaned_attempts(model_dir)

    pending = []
    for sd in all_dirs:
        out_dir = model_dir / sd.name
        if _existing_result_ok(out_dir) and not overwrite:
            continue
        pending.append(sd)
    n_skipped = len(all_dirs) - len(pending)
    todo = pending[:limit] if limit is not None else pending

    print(f"[run_stop_batch] model={model_slug} local={entry.local} workers={workers} "
          f"candidates={len(all_dirs)} to_run={len(todo)} already_done={n_skipped}")

    sem = asyncio.Semaphore(max(1, workers))

    async def _guarded(sd: Path):
        async with sem:
            out_dir = model_dir / sd.name
            for attempt in range(1 + RETRYABLE_ERROR_RETRIES):
                sample_id, status, detail = await run_one_sample(
                    sd, out_dir, model_slug, client_cls, api_sr, default_model, output_sr,
                    entry.client_kwargs, entry.api_key_env, verbose,
                )
                if status == "ok" or attempt == RETRYABLE_ERROR_RETRIES:
                    return sample_id, status, detail
                error_text = detail.get("error", "")
                if not _is_retryable_error(error_text):
                    return sample_id, status, detail
                from openai_keys import is_quota_error, note_quota_from_error
                if is_quota_error(error_text):
                    note_quota_from_error(error_text)
                    delay = 0.5
                else:
                    delay = RETRYABLE_ERROR_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1.0)
                if verbose:
                    print(f"  [RETRY {attempt + 1}/{RETRYABLE_ERROR_RETRIES}] {sd.name} "
                          f"after {delay:.1f}s ({error_text})", file=sys.stderr)
                await asyncio.sleep(delay)

    n_ok = n_err = 0
    t_start = time.monotonic()
    for i, coro in enumerate(asyncio.as_completed([_guarded(sd) for sd in todo])):
        try:
            sample_id, status, detail = await coro
        except Exception as e:
            print(f"  [UNHANDLED] {type(e).__name__}: {e}", file=sys.stderr)
            n_err += 1
            continue
        run_manifest[sample_id] = {
            "status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **detail,
        }
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2))
        if status == "ok":
            n_ok += 1
        else:
            n_err += 1
        elapsed = time.monotonic() - t_start
        print(f"  [{i + 1}/{len(todo)}] ok={n_ok} err={n_err} elapsed={elapsed:.0f}s", flush=True)

    print(f"[run_stop_batch] model={model_slug} DONE: {n_ok} ok, {n_err} errors "
          f"(of {len(todo)} attempted, {n_skipped} already done)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Ignored (forced to 1) for local models.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_ids", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sample_ids = args.sample_ids.split(",") if args.sample_ids else None
    asyncio.run(run_batch(args.model, args.data_root, args.results_root, args.workers,
                          args.overwrite, args.limit, sample_ids, args.verbose))


if __name__ == "__main__":
    main()
