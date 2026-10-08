#!/usr/bin/env python3
"""
evaluation/simulator/inference/run_interrupt_batch.py
=========================================================
Batch driver for the `interrupt_user` Level-2 action's simulator clients
(infer_interrupt_<model>.py). Sibling of run_action_batch.py -- deliberately
a NEW file rather than extending run_action_batch / run_stop_batch, because
interrupt_user cannot reuse infer_stop_<model>.py (different live driver:
InterruptMonitorController + user_speech/branch wavs, see
interrupt_user_common.py).

Reuses run_stop_batch.py's retry/dedup/publish helpers BY IMPORT. Own
model registry: only models that already have an infer_interrupt_<model>.py
(currently: moshi, smoke-tested 2026-09-08). Does not import or modify
MODEL_REGISTRY in run_stop_batch.py, and does not touch stop / hold_floor /
respond_and_recover clients.

Usage:
    python run_interrupt_batch.py --model moshi --workers 1
    python run_interrupt_batch.py --model moshi --limit 5 --verbose
    python run_interrupt_batch.py --model moshi --conditions interrupt_for_help
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "direct_replay" / "inference"))

from run_stop_batch import (  # noqa: E402
    _existing_result_ok, _gc_orphaned_attempts, _publish_attempt,
    _is_retryable_error, RETRYABLE_ERROR_RETRIES, RETRYABLE_ERROR_BASE_DELAY_S,
    SAMPLE_TIMEOUT_OVERHEAD_S,
)

# Same budget as direct_replay/inference/run_inference.py. Interrupt
# clients load servers/weights inside the first sample unless we warmup;
# FLM/Raon/Moshi cold-start otherwise races the per-sample hard timeout.
WARMUP_TIMEOUT_S = 900.0
from interrupt_user_common import (  # noqa: E402
    run_interrupt_sample_cli, POST_BRANCH_OBSERVATION_S,
)
from interrupt_controller import (  # noqa: E402
    GRACE_AFTER_TARGET_S, MAX_POST_ONSET_WAIT_S,
)


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
DEFAULT_DATA_ROOT = REPO_ROOT / "evaluation" / "direct_replay" / "data" / "interrupt_user"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation" / "simulator" / "results" / "interrupt_user"


@dataclass
class InterruptModelEntry:
    module: str
    client_cls_attr: str
    api_sr_attr: str
    default_model_attr: str
    local: bool
    output_sr_attr: Optional[str] = None
    api_key_env: Optional[str] = None
    client_kwargs: dict = field(default_factory=dict)


# Currently excluded: lychee-fd / voila / vita / mini-omni2 / nemotron
# (no live interrupt driver).
INTERRUPT_MODEL_REGISTRY: dict[str, InterruptModelEntry] = {
    "moshi": InterruptModelEntry(
        "infer_interrupt_moshi", "InterruptMoshiClient", "API_SR", "DEFAULT_MODEL",
        local=True,
    ),
    "flm-audio": InterruptModelEntry(
        "infer_interrupt_flm_audio", "InterruptFLMAudioClient", "API_SR", "DEFAULT_MODEL",
        local=True,
    ),
    "bayling-duplex": InterruptModelEntry(
        "infer_interrupt_bayling", "InterruptBayLingDuplexClient", "INPUT_SR", "DEFAULT_MODEL",
        local=True, output_sr_attr="API_SR",
    ),
    "raon-speechchat": InterruptModelEntry(
        "infer_interrupt_raon", "InterruptRaonSpeechChatClient", "API_SR", "DEFAULT_MODEL",
        local=True,
    ),
    "minicpm-o-4.5": InterruptModelEntry(
        "infer_interrupt_minicpm", "InterruptMiniCPMClient", "INPUT_SR", "DEFAULT_MODEL",
        local=True, output_sr_attr="OUTPUT_SR",
    ),
    "freeze-omni": InterruptModelEntry(
        "infer_interrupt_freeze_omni", "InterruptFreezeOmniClient", "INPUT_SR", "FREEZE_OMNI_REPO_ID",
        local=True, output_sr_attr="OUTPUT_SR",
    ),
    "gpt-realtime-2.1": InterruptModelEntry(
        "infer_interrupt_gpt_realtime", "InterruptGPTRealtimeClient", "API_SR", "DEFAULT_MODEL",
        local=False, api_key_env="OPENAI_API_KEY",
    ),
    "gemini-3.1-flash-live": InterruptModelEntry(
        "infer_interrupt_gemini_live", "InterruptGeminiLiveClient", "SEND_SAMPLE_RATE", "DEFAULT_MODEL",
        local=False, output_sr_attr="RECEIVE_SAMPLE_RATE", api_key_env="GEMINI_API_KEY",
    ),
    "grok-voice": InterruptModelEntry(
        "infer_interrupt_grok_voice", "InterruptGrokVoiceClient", "API_SR", "DEFAULT_MODEL",
        local=False, api_key_env="XAI_API_KEY",
        client_kwargs={"voice": "eve"},
    ),
    "qwen3.5-omni-flash-realtime": InterruptModelEntry(
        "infer_interrupt_qwen_realtime", "InterruptQwenRealtimeClient", "API_SR", "DEFAULT_MODEL",
        local=False, output_sr_attr="OUTPUT_SR", api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"voice": "Ethan"},
    ),
}


def _load_entry(model_slug: str):
    import importlib
    entry = INTERRUPT_MODEL_REGISTRY.get(model_slug)
    if entry is None:
        raise SystemExit(
            f"Model '{model_slug}' is not registered for interrupt_user "
            f"(no infer_interrupt client yet). "
            f"Registered: {sorted(INTERRUPT_MODEL_REGISTRY)}"
        )
    mod = importlib.import_module(entry.module)
    client_cls = getattr(mod, entry.client_cls_attr)
    api_sr = getattr(mod, entry.api_sr_attr)
    default_model = getattr(mod, entry.default_model_attr)
    output_sr = getattr(mod, entry.output_sr_attr) if entry.output_sr_attr else None
    return entry, client_cls, api_sr, default_model, output_sr


def _discover_sample_dirs(data_root: Path) -> list[Path]:
    """interrupt_user/{interrupt_for_help,interrupt_for_safety}/<sample_id>/."""
    return sorted(
        p for p in data_root.glob("*/*")
        if p.is_dir() and (p / "user_speech.wav").exists()
    )


async def run_one_sample(
    sample_dir: Path, out_dir: Path, model_slug: str, client_cls, api_sr: int,
    default_model: str, output_sr: Optional[int], client_kwargs: dict,
    api_key_env: Optional[str], verbose: bool,
) -> tuple[str, str, dict]:
    sample_id = sample_dir.name
    attempt_dir = out_dir.parent / f".attempt-{sample_id}-{uuid.uuid4().hex[:8]}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        full_wav = sample_dir / "branch_ignored.wav"
        if not full_wav.exists():
            full_wav = sample_dir / "user_speech.wav"
        user_speech_duration = sf.info(str(full_wav)).duration
    except Exception:
        user_speech_duration = None
    # user_speech + grace + post-onset wait cap + post-branch drain +
    # the same connect/handshake overhead as run_stop_batch.
    sample_timeout = (
        (user_speech_duration or 10.0)
        + GRACE_AFTER_TARGET_S
        + MAX_POST_ONSET_WAIT_S
        + POST_BRANCH_OBSERVATION_S
        + SAMPLE_TIMEOUT_OVERHEAD_S
    )

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
                run_interrupt_sample_cli(
                    client_cls, model_slug, api_sr, args,
                    client_kwargs=client_kwargs, api_key_env=api_key_env,
                    output_sr=output_sr,
                ),
                timeout=sample_timeout,
            )
        except asyncio.TimeoutError:
            return sample_id, "error", {
                "error": f"TimeoutError: exceeded hard timeout ({sample_timeout:.0f}s)",
            }
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
        monitor = info.get("inference", {}).get("monitor") or {}
        outcome = monitor.get("outcome")
        forced = monitor.get("forced")
        if verbose or status == "error":
            print(f"  [{status.upper()}] {sample_id}  segments={n_segs}  "
                  f"outcome={outcome}  forced={forced}  errors={errs}")
        detail = {"n_segments": n_segs, "outcome": outcome, "forced": forced}
        if errs:
            detail["error"] = "; ".join(str(e) for e in errs)
        return sample_id, status, detail
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)


async def run_batch(model_slug: str, data_root: Path, results_root: Path, workers: int,
                     overwrite: bool, limit: Optional[int], sample_ids: Optional[list[str]],
                     conditions: Optional[list[str]], verbose: bool) -> None:
    entry, client_cls, api_sr, default_model, output_sr = _load_entry(model_slug)
    workers = 1 if entry.local else workers

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("No OPENAI_API_KEY in env / project .env "
                         "(needed by InterruptMonitorController).")
    if entry.api_key_env and not os.environ.get(entry.api_key_env):
        raise SystemExit(f"No API key found in ${entry.api_key_env} (env / .env)")

    all_dirs = _discover_sample_dirs(data_root)
    if conditions:
        wanted_cond = set(conditions)
        all_dirs = [p for p in all_dirs if p.parent.name in wanted_cond]
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

    print(f"[run_interrupt_batch] model={model_slug} local={entry.local} workers={workers} "
          f"candidates={len(all_dirs)} to_run={len(todo)} already_done={n_skipped}")

    if todo:
        warmup = getattr(client_cls, "warmup", None)
        if warmup is not None:
            t_warm = time.monotonic()
            print(f"[run_interrupt_batch] warming up {model_slug} "
                  f"(one-time model load, untimed up to {WARMUP_TIMEOUT_S:.0f}s)...", flush=True)
            await asyncio.wait_for(warmup(), timeout=WARMUP_TIMEOUT_S)
            print(f"[run_interrupt_batch] warmup done in {time.monotonic() - t_warm:.1f}s", flush=True)

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

    print(f"[run_interrupt_batch] model={model_slug} DONE: {n_ok} ok, {n_err} errors "
          f"(of {len(todo)} attempted, {n_skipped} already done)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(INTERRUPT_MODEL_REGISTRY))
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Ignored (forced to 1) for local models.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_ids", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None,
                         help="Comma-separated l3_condition subdir names "
                              "(interrupt_for_help, interrupt_for_safety).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sample_ids = args.sample_ids.split(",") if args.sample_ids else None
    conditions = args.conditions.split(",") if args.conditions else None
    asyncio.run(run_batch(args.model, args.data_root, args.results_root, args.workers,
                          args.overwrite, args.limit, sample_ids, conditions, args.verbose))


if __name__ == "__main__":
    main()
