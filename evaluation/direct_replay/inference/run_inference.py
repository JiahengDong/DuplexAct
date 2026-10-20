#!/usr/bin/env python3
"""
Batch inference driver for direct replay. Given an action, an optional
l3_condition and a model slug, streams every matching sample's input.wav to
that model and writes:

  results/inference_results/<action>/<l3_condition>/<model_slug>/<sample_id>/
    output.wav          # time-aligned model audio
    inference_info.json # {sample_id, action, l3_condition, model, instance, inference}
    events.jsonl         # raw per-event API log
  results/inference_results/<action>/<l3_condition>/<model_slug>/_run_manifest.json

Samples are listed from data/<action>/manifest.json (written by
scripts/build_manifests.py); each sample directory is resolved as
data_root/<action>/<l3_dir>/<sample_id>.

Usage:
    python run_inference.py \\
        --action wait --l3_condition floor_holding \\
        --model gpt-realtime-2.1 --workers 4

    # All l3_conditions for an action:
    python run_inference.py --action backchannel --model gpt-realtime-2.1

    # A couple of samples only:
    python run_inference.py --action wait --l3_condition floor_holding \\
        --model gpt-realtime-2.1 --limit 2 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import fcntl
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv

from model_registry import get_model_spec, get_client_cls, all_registered_models
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import result_dir_for
from validate_data import validate_sample


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

REPO_ROOT     = _duplexact_root()
DEFAULT_DATA_ROOT    = REPO_ROOT / "evaluation" / "data"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation" / "direct_replay" / "results" / "inference_results"

ACTIONS = ["wait", "backchannel", "ignore", "turn_taking"]

# Per-sample hard timeout = input_duration (real-time streaming + drain) plus
# this flat overhead for connect/handshake/close, so a stalled connection
# cannot occupy a worker slot forever.
SAMPLE_TIMEOUT_OVERHEAD_S = 90.0

# Some clients cache a heavy model at the class level. The first load can take
# longer than the per-sample timeout and cannot be cancelled inside
# `asyncio.to_thread`, so a timeout + retry would start a second concurrent
# load on the same GPU. `client_cls.warmup()` (no-op by default) runs once up
# front with this separate timeout.
WARMUP_TIMEOUT_S = 900.0

# A slow local client can override this via a TIMEOUT_OVERHEAD_S class
# attribute instead of raising the shared default for API-based models.
DEFAULT_TIMEOUT_OVERHEAD_S = SAMPLE_TIMEOUT_OVERHEAD_S

# Max allowed difference between metadata `input_duration` and the actual
# input.wav duration; a larger mismatch means stale/corrupt data and would
# shift every downstream timestamp. 20ms covers metadata.json's 2-decimal
# rounding plus resampling/read precision.
DURATION_MISMATCH_HARD_S = 0.02


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_data_manifest(data_root: Path, action: str) -> list[dict]:
    manifest_path = data_root / action / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json for action {action!r} at {manifest_path}")
    return json.loads(manifest_path.read_text())


def filter_unrunnable(
    manifest: list[dict],
    action: str,
    data_root: Path,
    verbose: bool,
) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Drop entries whose metadata.json is missing the timestamp fields the
    metrics need for B_i/T_i (as defined by validate_data.py's REQUIRED_FIELDS).
    Returns (runnable_entries, {sample_id: problems} for everything dropped).
    """
    runnable = []
    dropped: dict[str, list[str]] = {}
    for e in manifest:
        l3_dir = l3_dir_of(e)
        sample_dir = data_root / action / l3_dir / e["sample_id"]
        status, problems = validate_sample(action, l3_dir, sample_dir)
        if status == "ok":
            runnable.append(e)
        else:
            dropped[e["sample_id"]] = problems
    if dropped:
        print(f"[run_inference] skipping {len(dropped)} unrunnable sample(s) "
              f"(missing required metadata -- see validate_data.py):")
        for sid, problems in dropped.items():
            print(f"    - {sid}: {', '.join(problems)}")
    return runnable, dropped


def load_run_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_run_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


# Transient failures worth retrying in place: API-side concurrent-session and
# rate-limit rejections and dropped handshakes (e.g. grok-voice "Too many
# concurrent sessions", Qwen realtime "thread pool exausted"). Content-policy
# rejections are excluded since retrying the same input would fail again.
_RETRYABLE_ERROR_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"too many concurrent sessions",
        r"thread pool exausted",
        r"thread pool exhausted",
        r"rate limit exceeded",
        r"ConnectionClosedError",
        r"TimeoutError",
        r"connect call failed",
    ]
]
_NON_RETRYABLE_ERROR_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"inappropriate content",
    ]
]
RETRYABLE_ERROR_RETRIES = 3
RETRYABLE_ERROR_BASE_DELAY_S = 5.0


def _is_retryable_error(error_text: str) -> bool:
    if any(p.search(error_text) for p in _NON_RETRYABLE_ERROR_PATTERNS):
        return False
    return any(p.search(error_text) for p in _RETRYABLE_ERROR_PATTERNS)


def _existing_result_ok(out_dir: Path) -> bool:
    """
    True only if a prior run's output.wav + inference_info.json both exist
    and that inference recorded no errors. A failed sample still gets an
    inference_info.json (empty model_speech_segments, populated `errors`),
    which for wait/ignore would be indistinguishable from a genuinely silent
    model, so it must be retried rather than treated as done.
    """
    output_wav = out_dir / "output.wav"
    info_path = out_dir / "inference_info.json"
    if not (output_wav.exists() and info_path.exists()):
        return False
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return False
    return not info.get("inference", {}).get("errors")


def l3_dir_of(entry: dict) -> str:
    """
    The actual data/ subdirectory name for this sample. NOT always equal to
    entry["l3_condition"] -- e.g. turn_taking's rubric l3_condition values
    (immediate_take_turn, delayed_take_turn, slight_overlap_take_turn) differ
    from their directory names (take_turn_immediately, take_turn_with_delay,
    take_turn_with_overlap). The directory name is taken from the parent of
    the manifest's sample_dir (only its relative structure is relied on).
    """
    return Path(entry["sample_dir"]).parent.name


# ── Atomic artifact publication ──────────────────────────────────────────────
#
# Each attempt writes to a private temporary sibling directory, is validated
# as a whole, and is then renamed into out_dir's place. A failed or aborted
# attempt never touches out_dir, so a previous good result is never mixed
# with partial new audio.

class _SampleLockBusy(Exception):
    """Another process already holds this sample's lock."""


class _SampleLock:
    """
    Cross-process advisory lock scoped to one (action, l3, model, sample_id)
    output directory, so two run_inference.py processes cannot interleave
    writes and publish steps for the same sample. Non-blocking: raises
    _SampleLockBusy immediately if another process holds the lock.
    """

    def __init__(self, out_dir: Path):
        self.lock_path = out_dir.parent / f".{out_dir.name}.lock"
        self._fh = None

    def __enter__(self) -> "_SampleLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.lock_path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self._fh.close()
            self._fh = None
            if e.errno in (errno.EACCES, errno.EAGAIN):
                raise _SampleLockBusy() from e
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def _validate_attempt(attempt_dir: Path, expected_output_duration_s: float | None) -> str | None:
    """
    Returns None if the attempt directory's artifacts are internally
    consistent and safe to publish, else a human-readable reason they
    aren't. Checked BEFORE the atomic rename into out_dir's place, so a
    validation failure just means the (untouched) previous result stands --
    never a half-published mix.
    """
    output_wav  = attempt_dir / "output.wav"
    events_path = attempt_dir / "events.jsonl"
    info_path   = attempt_dir / "inference_info.json"

    if not output_wav.exists():
        return "output.wav missing from attempt"
    try:
        wav_info = sf.info(str(output_wav))
    except Exception as e:
        return f"output.wav unreadable: {type(e).__name__}: {e}"
    if expected_output_duration_s is not None:
        # Loose tolerance: this only needs to catch gross truncation of the
        # recorder's output (e.g. a crash mid-write).
        if abs(wav_info.duration - expected_output_duration_s) > 0.5:
            return (f"output.wav duration={wav_info.duration:.3f}s does not match "
                    f"reported output_duration_s={expected_output_duration_s:.3f}s")

    # events.jsonl is created up front but only written to when an event
    # happens, so a model that never detects speech or responds legitimately
    # leaves it empty. Only require that it exists.
    if not events_path.exists():
        return "events.jsonl missing from attempt"
    try:
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)
    except Exception as e:
        return f"events.jsonl not valid JSONL: {type(e).__name__}: {e}"

    if not info_path.exists():
        return "inference_info.json missing from attempt"
    try:
        json.loads(info_path.read_text())
    except Exception as e:
        return f"inference_info.json not valid JSON: {type(e).__name__}: {e}"

    return None


def _gc_orphaned_attempts(model_dir: Path) -> None:
    """
    Best-effort cleanup of `.attempt-*`/`.backup-*` directories left behind
    by a previously killed process (a hard kill skips `finally` cleanup).
    Must run at the start of a batch, before any worker creates its own
    attempt directory. Real <sample_id>/ results never start with a dot, so
    they are never touched.
    """
    if not model_dir.is_dir():
        return
    for child in model_dir.iterdir():
        if child.is_dir() and (child.name.startswith(".attempt-") or child.name.startswith(".backup-")):
            shutil.rmtree(child, ignore_errors=True)


def _publish_attempt(attempt_dir: Path, out_dir: Path) -> None:
    """
    Atomically swap a validated attempt directory into out_dir's place.
    os.rename() can't replace a non-empty existing directory in one step on
    Linux (ENOTEMPTY), so a prior out_dir is first renamed aside to a backup
    path -- each individual rename is atomic, and if the process dies
    between them, out_dir is simply absent (not corrupt/mixed) and the next
    run retries the sample from scratch; the backup itself is only removed
    after the new directory is safely in place.
    """
    backup_dir = None
    if out_dir.exists():
        backup_dir = out_dir.parent / f".backup-{out_dir.name}-{uuid.uuid4().hex[:8]}"
        os.rename(out_dir, backup_dir)
    try:
        os.rename(attempt_dir, out_dir)
    except Exception:
        if backup_dir is not None:
            os.rename(backup_dir, out_dir)   # best-effort rollback
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


# ── Per-sample task ───────────────────────────────────────────────────────────

async def run_one_sample(
    entry: dict,
    action: str,
    data_root: Path,
    results_root: Path,
    model_slug: str,
    model_name: str,
    system_prompt: str,
    api_key: str,
    client_cls,
    verbose: bool,
) -> tuple[str, str, dict]:
    """
    Returns (sample_id, status, detail_dict).

    Metadata loading, client construction and inference share one failure
    boundary, so a corrupt metadata.json or a constructor error fails only
    this sample rather than the whole batch. `client.run_sample(...)` runs
    under a hard wall-clock timeout.
    """
    sample_id    = entry["sample_id"]
    l3_condition = entry["l3_condition"]     # rubric ID, stored verbatim in results
    l3_dir       = l3_dir_of(entry)          # actual data/ subdirectory name
    sample_timeout = None

    out_dir = result_dir_for(results_root, action, l3_dir, model_slug, sample_id)
    try:
        lock = _SampleLock(out_dir)
        lock.__enter__()
    except _SampleLockBusy:
        # Another process is working on this sample; report it separately
        # so it is not counted as an error.
        return sample_id, "skipped_locked", {"note": f"lock held by another process: {out_dir}"}

    # Every artifact for this attempt is written here, never directly into
    # out_dir (see the note above _SampleLock).
    attempt_dir = out_dir.parent / f".attempt-{sample_id}-{uuid.uuid4().hex[:8]}"

    try:
        try:
            sample_dir = data_root / action / l3_dir / sample_id
            input_wav  = sample_dir / "input.wav"
            meta_path  = sample_dir / "metadata.json"

            if not input_wav.exists() or not meta_path.exists():
                return sample_id, "error", {"error": f"missing input.wav/metadata.json in {sample_dir}"}

            instance_metadata = json.loads(meta_path.read_text())
            input_duration = instance_metadata.get("input_duration")
            if input_duration is None:
                return sample_id, "error", {"error": "metadata.json missing 'input_duration'"}
            input_duration = float(input_duration)

            # The sender streams the actual WAV while the recorder's target
            # length and padding use metadata's input_duration; a mismatch
            # would misalign timing, so reject the sample.
            try:
                actual_duration = sf.info(str(input_wav)).duration
            except Exception:
                actual_duration = None
            duration_mismatch_s = None
            if actual_duration is not None:
                duration_mismatch_s = abs(actual_duration - input_duration)
                if duration_mismatch_s > DURATION_MISMATCH_HARD_S:
                    return sample_id, "error", {
                        "error": f"metadata input_duration={input_duration:.3f}s but input.wav "
                                 f"duration={actual_duration:.3f}s (mismatch={duration_mismatch_s:.3f}s)"
                    }

            attempt_dir.mkdir(parents=True, exist_ok=True)
            output_wav  = attempt_dir / "output.wav"
            events_path = attempt_dir / "events.jsonl"
            info_path   = attempt_dir / "inference_info.json"

            client = client_cls(
                api_key=api_key,
                model_name=model_name,
                system_prompt=system_prompt,
                verbose=verbose,
            )

            timeout_overhead = getattr(client_cls, "TIMEOUT_OVERHEAD_S", DEFAULT_TIMEOUT_OVERHEAD_S)
            sample_timeout = input_duration + timeout_overhead
            inference_dict = await asyncio.wait_for(
                client.run_sample(
                    input_wav=input_wav,
                    output_wav=output_wav,
                    events_path=events_path,
                    input_duration=input_duration,
                ),
                timeout=sample_timeout,
            )
        except asyncio.TimeoutError:
            detail = {"error": f"TimeoutError: sample exceeded hard timeout "
                                f"({sample_timeout:.0f}s)" if sample_timeout else
                                "TimeoutError: sample exceeded hard timeout"}
            if verbose:
                print(f"  [ERROR] {sample_id}: {detail['error']}", file=sys.stderr)
            return sample_id, "error", detail
        except Exception as e:
            tb = traceback.format_exc()
            if verbose:
                print(f"  [ERROR] {sample_id}: {e}\n{tb}", file=sys.stderr)
            return sample_id, "error", {"error": f"{type(e).__name__}: {e}"}

        try:
            result = {
                "sample_id":    sample_id,
                "action":       action,
                "l3_condition": l3_condition,
                "model":        model_slug,
                "instance":     instance_metadata,
                "inference":    inference_dict,
            }
            if duration_mismatch_s is not None:
                result["duration_mismatch_s"] = round(duration_mismatch_s, 4)
            info_path.write_text(json.dumps(result, indent=2))

            # Validate the whole attempt before publishing; on failure out_dir
            # is left exactly as it was.
            reason = _validate_attempt(attempt_dir, inference_dict.get("output_duration_s"))
            if reason is not None:
                if verbose:
                    print(f"  [ERROR] {sample_id}: attempt failed validation: {reason}", file=sys.stderr)
                return sample_id, "error", {"error": f"attempt failed validation: {reason}"}

            _publish_attempt(attempt_dir, out_dir)
        except Exception as e:
            tb = traceback.format_exc()
            if verbose:
                print(f"  [ERROR] {sample_id}: failed publishing result: {e}\n{tb}", file=sys.stderr)
            return sample_id, "error", {"error": f"publish failed: {type(e).__name__}: {e}"}

        errs = inference_dict.get("errors") or []
        status = "error" if errs else "ok"
        n_segs = len(inference_dict.get("model_speech_segments", []))
        if verbose or status == "error":
            print(f"  [{status.upper()}] {sample_id}  segments={n_segs}  "
                  f"errors={inference_dict.get('errors')}")
        detail = {"n_segments": n_segs}
        if errs:
            detail["error"] = "; ".join(str(e) for e in errs)
        return sample_id, status, detail
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)   # no-op once published (already moved)
        lock.__exit__()


# ── Batch driver ──────────────────────────────────────────────────────────────

async def run_batch(
    action: str,
    l3_condition: str | None,
    model_slug: str,
    data_root: Path,
    results_root: Path,
    workers: int,
    overwrite: bool,
    limit: int | None,
    sample_ids: list[str] | None,
    system_prompt: str | None,
    api_key_override: str | None,
    verbose: bool,
    skip_unrunnable: bool = True,
) -> None:
    spec = get_model_spec(model_slug)
    client_cls = get_client_cls(model_slug)
    api_key = api_key_override or os.getenv(spec.api_key_env, "")
    # AWS_PROFILE: Bedrock uses the ambient credential chain, no single key.
    # LOCAL_NO_KEY: locally-hosted open-source models need no API key at all.
    if not api_key and spec.api_key_env not in ("AWS_PROFILE", "LOCAL_NO_KEY"):
        raise RuntimeError(f"No API key found in ${spec.api_key_env} (env / .env / --api_key)")

    manifest = load_data_manifest(data_root, action)
    if l3_condition:
        # Accept either the rubric l3_condition value (e.g. immediate_take_turn)
        # or the actual data/ directory name (e.g. take_turn_immediately) --
        # they differ for turn_taking but are identical for every other action.
        manifest = [e for e in manifest
                    if e["l3_condition"] == l3_condition or l3_dir_of(e) == l3_condition]
    if sample_ids:
        wanted = set(sample_ids)
        manifest = [e for e in manifest if e["sample_id"] in wanted]

    if skip_unrunnable:
        manifest, _dropped = filter_unrunnable(manifest, action, data_root, verbose)

    print(f"[run_inference] action={action} l3_condition={l3_condition or 'ALL'} "
          f"model={model_slug} candidates={len(manifest)}")

    if manifest:
        warmup = getattr(client_cls, "warmup", None)
        if warmup is not None:
            t_warm = time.monotonic()
            print(f"[run_inference] warming up {model_slug} "
                  f"(one-time model load, untimed up to {WARMUP_TIMEOUT_S:.0f}s)...", flush=True)
            await asyncio.wait_for(warmup(), timeout=WARMUP_TIMEOUT_S)
            print(f"[run_inference] warmup done in {time.monotonic() - t_warm:.1f}s", flush=True)

    # Group by the actual directory name so each group gets its own
    # _run_manifest.json and correctly matching output path.
    by_l3: dict[str, list[dict]] = {}
    for e in manifest:
        by_l3.setdefault(l3_dir_of(e), []).append(e)

    system_prompt = system_prompt or spec.default_system_prompt
    sem = asyncio.Semaphore(max(1, workers))

    for l3, entries in by_l3.items():
        model_dir = result_dir_for(results_root, action, l3, model_slug, "")
        run_manifest_path = model_dir / "_run_manifest.json"
        run_manifest = load_run_manifest(run_manifest_path)
        _gc_orphaned_attempts(model_dir)

        pending = []
        for e in entries:
            sid = e["sample_id"]
            out_dir = result_dir_for(results_root, action, l3, model_slug, sid)
            if _existing_result_ok(out_dir) and not overwrite:
                continue
            pending.append(e)
        n_skipped = len(entries) - len(pending)
        todo = pending[:limit] if limit is not None else pending

        print(f"  [{l3}] {len(todo)} to run "
              f"({n_skipped} already done, skipped"
              f"{f'; {len(pending) - len(todo)} more deferred by --limit' if limit is not None and len(pending) > len(todo) else ''})")

        async def _guarded(entry):
            async with sem:
                for attempt in range(1 + RETRYABLE_ERROR_RETRIES):
                    sample_id, status, detail = await run_one_sample(
                        entry, action, data_root, results_root, model_slug,
                        spec.model_name, system_prompt, api_key, client_cls, verbose,
                    )
                    if status == "ok" or attempt == RETRYABLE_ERROR_RETRIES:
                        return sample_id, status, detail
                    error_text = detail.get("error", "")
                    if not _is_retryable_error(error_text):
                        return sample_id, status, detail
                    # Transient API-side rejection: back off with jitter and
                    # retry in place.
                    delay = RETRYABLE_ERROR_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1.0)
                    if verbose:
                        print(f"  [RETRY {attempt + 1}/{RETRYABLE_ERROR_RETRIES}] "
                              f"{entry['sample_id']} after {delay:.1f}s ({error_text})",
                              file=sys.stderr)
                    await asyncio.sleep(delay)

        n_ok = n_err = 0
        for coro in asyncio.as_completed([_guarded(e) for e in todo]):
            try:
                sample_id, status, detail = await coro
            except Exception as e:
                # Last-resort guard so one sample can never abort the batch.
                # The sample_id is unknown here, so it is not recorded in
                # the run manifest.
                print(f"  [UNHANDLED] {type(e).__name__}: {e}", file=sys.stderr)
                n_err += 1
                continue
            run_manifest[sample_id] = {
                "status":     status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **detail,
            }
            save_run_manifest(run_manifest_path, run_manifest)
            if status == "ok":
                n_ok += 1
            else:
                n_err += 1

        print(f"  [{l3}] done: {n_ok} ok, {n_err} errors")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--l3_condition", default=None,
                         help="Restrict to one l3_condition subfolder. Default: all for the action.")
    parser.add_argument("--model", required=True,
                         help=f"Model slug. Registered: {all_registered_models()}")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--workers", type=int, default=4,
                         help="Max concurrent realtime sessions.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-run samples that already have output.wav + inference_info.json")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap number of samples per l3_condition (for smoke tests).")
    parser.add_argument("--sample_ids", type=str, default=None,
                         help="Comma-separated list of specific sample_ids to run.")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--include_unrunnable", action="store_true",
                         help="Do NOT skip samples missing required metadata timestamps "
                              "(see validate_data.py). Off by default -- unrunnable samples "
                              "can never get a real B_i/T_i, so running inference on them "
                              "just burns API/GPU cost for nothing.")
    args = parser.parse_args()

    sample_ids = args.sample_ids.split(",") if args.sample_ids else None

    asyncio.run(run_batch(
        action           = args.action,
        l3_condition     = args.l3_condition,
        model_slug       = args.model,
        data_root        = args.data_root,
        results_root     = args.results_root,
        workers          = args.workers,
        overwrite        = args.overwrite,
        limit            = args.limit,
        sample_ids       = sample_ids,
        system_prompt    = args.system_prompt,
        api_key_override = args.api_key,
        verbose          = args.verbose,
        skip_unrunnable  = not args.include_unrunnable,
    ))


if __name__ == "__main__":
    main()
