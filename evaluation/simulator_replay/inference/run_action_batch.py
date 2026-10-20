#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/run_action_batch.py
======================================================
Action-generalized sibling of run_stop_batch.py -- drives `hold_floor` and
`respond_and_recover` (the two OTHER Level-2 actions confirmed to share
stop's own lead_in.wav/cue.wav/metadata.json live-cue-injection shape, see
cue_action_common.py's own module docstring for why interrupt_user is
excluded) through the exact same per-model infer_stop_<model>.py clients,
completely unchanged, just via cue_action_common.run_cue_action_sample_cli
instead of stop_common.run_stop_sample_cli.

Deliberately reuses run_stop_batch.py's MODEL_REGISTRY, retry/dedup/publish
helpers, and overall shape BY IMPORT (not by duplicating the code) -- the
only real differences are: (1) which driver function gets called per
sample, (2) sample discovery has to glob across each action's multiple
condition subdirectories instead of stop's single `direct_interruption/`
dir, (3) results land under `results/<action>/<model>/` instead of
`results/stop/<model>/`. run_stop_batch.py itself is never imported BY
this file's own callers and never modified -- this is purely additive.

Usage:
    python run_action_batch.py --action respond_and_recover --model moshi --limit 3 --verbose
    python run_action_batch.py --action hold_floor --model gpt-realtime-2.1 --workers 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from run_stop_batch import (  # noqa: E402
    MODEL_REGISTRY, _load_entry, _existing_result_ok, _gc_orphaned_attempts,
    _publish_attempt, _is_retryable_error, RETRYABLE_ERROR_RETRIES,
    RETRYABLE_ERROR_BASE_DELAY_S, SAMPLE_TIMEOUT_OVERHEAD_S,
)
from cue_action_common import ACTION_DATA_ROOTS, run_cue_action_sample_cli  # noqa: E402


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
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation" / "simulator_replay" / "results"


def _discover_sample_dirs(data_root: Path) -> list[Path]:
    """
    Unlike stop's single `direct_interruption/` subdir, hold_floor and
    respond_and_recover each have multiple condition subdirectories (e.g.
    respond_and_recover/{enrich,shorten,repeat,shift_topic,ask_follow_up,
    correct_response_after_interruption}_.../<sample_id>/) -- glob two
    levels down and filter to real sample dirs (must contain lead_in.wav,
    excludes the *.json manifest/report files sitting alongside the
    condition subdirs at the top level).
    """
    return sorted(
        p for p in data_root.glob("*/*")
        if p.is_dir() and (p / "lead_in.wav").exists()
    )


async def run_one_sample(
    sample_dir: Path, out_dir: Path, model_slug: str, action: str, client_cls, api_sr: int,
    default_model: str, output_sr: Optional[int], client_kwargs: dict,
    api_key_env: Optional[str], verbose: bool,
) -> tuple[str, str, dict]:
    sample_id = sample_dir.name
    attempt_dir = out_dir.parent / f".attempt-{sample_id}-{uuid.uuid4().hex[:8]}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        lead_in_duration = sf.info(str(sample_dir / "lead_in.wav")).duration
    except Exception:
        lead_in_duration = None
    # Same slack as run_stop_batch: T_SEARCH/T_ONSET after lead-in, not the
    # retired 20s-from-t=0 MAX_WAIT.
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
                run_cue_action_sample_cli(client_cls, model_slug, api_sr, action, args,
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
        import shutil
        shutil.rmtree(attempt_dir, ignore_errors=True)


async def run_batch(action: str, model_slug: str, data_root: Path, results_root: Path, workers: int,
                     overwrite: bool, limit: Optional[int], sample_ids: Optional[list[str]],
                     conditions: Optional[list[str]], verbose: bool) -> None:
    entry, client_cls, api_sr, default_model, output_sr = _load_entry(model_slug)
    workers = 1 if entry.local else workers

    if entry.api_key_env and not os.environ.get(entry.api_key_env):
        from dotenv import load_dotenv
        load_dotenv()
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

    print(f"[run_action_batch] action={action} model={model_slug} local={entry.local} workers={workers} "
          f"candidates={len(all_dirs)} to_run={len(todo)} already_done={n_skipped}")

    sem = asyncio.Semaphore(max(1, workers))

    async def _guarded(sd: Path):
        async with sem:
            out_dir = model_dir / sd.name
            for attempt in range(1 + RETRYABLE_ERROR_RETRIES):
                sample_id, status, detail = await run_one_sample(
                    sd, out_dir, model_slug, action, client_cls, api_sr, default_model, output_sr,
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
                elif "403" in error_text or "InvalidStatus" in error_text:
                    delay = 20.0 + random.uniform(0, 5.0)
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

    print(f"[run_action_batch] action={action} model={model_slug} DONE: {n_ok} ok, {n_err} errors "
          f"(of {len(todo)} attempted, {n_skipped} already done)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", required=True, choices=sorted(ACTION_DATA_ROOTS))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--results_root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Ignored (forced to 1) for local models.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_ids", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None,
                         help="Comma-separated l3_condition subdir names to restrict to "
                              "(e.g. enrich_response_after_interruption).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root or ACTION_DATA_ROOTS[args.action]
    results_root = args.results_root or (DEFAULT_RESULTS_ROOT / args.action)
    sample_ids = args.sample_ids.split(",") if args.sample_ids else None
    conditions = args.conditions.split(",") if args.conditions else None

    asyncio.run(run_batch(args.action, args.model, data_root, results_root, args.workers,
                          args.overwrite, args.limit, sample_ids, conditions, args.verbose))


if __name__ == "__main__":
    main()
