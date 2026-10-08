#!/usr/bin/env python3
"""
evaluation/direct_replay/inference/recompute_lychee_fd_segments.py
======================================================================
One-off (re-runnable, idempotent) fix-up script for a bug found while
implementing lychee-fd's `stop`-action simulator client
(evaluation/simulator/inference/infer_stop_lychee_fd.py). Full root-cause
writeup: evaluation/simulator/MODEL_COMPATIBILITY.md's "SEVERE finding"
section.

Summary of the bug: `infer_lychee_fd.py`'s `model_speech_segments` are
derived from the model's own native `round_trace["events"]`, whose
timestamps are computed relative to an INPUT-TIMELINE offset
(`idx / INPUT_SR`) rather than real wall-clock arrival time. Once any
single round's own generation call overruns its 400ms dispatch budget
(routine for any response long/slow enough to matter), real-time pacing
breaks down and this offset diverges severely from when the audio was
actually produced/played -- confirmed on a live sample: an event that
closed at real wall-clock t=33.55s was labelled with a ~22.8s timestamp,
an 11-second error. Net effect: `model_speech_segments` can report as
little as ~2% of the model's real total speaking duration for a given
sample, while `output.wav` itself is completely intact (the actual PCM
samples are unaffected -- only the derived event LABELS are wrong).

Fix: re-derive `model_speech_segments` directly from each sample's own
`output.wav` audio energy (see `realtime_common.
recompute_speech_segments_from_audio()` for the method/tuning), the same
fix now built into `infer_stop_lychee_fd.py` for new simulator runs. This
script applies that identical fix retroactively to the EXISTING
`inference_results/**/lychee-fd/*/inference_info.json` production files
(the "2000/2000 clean" dataset referenced in prior status updates) so
already-collected data isn't left silently wrong.

Safety / idempotency:
  - The ORIGINAL `model_speech_segments` value is preserved under a new
    `model_speech_segments_original_round_trace` key (never overwritten
    again once present, unless --force) -- so this is always reversible
    and safe to re-run.
  - A `model_speech_segments_recomputed_at` ISO timestamp marks which
    files have already been fixed; re-running without --force skips them.
  - `--dry_run` computes and reports the recomputation without writing
    anything, for a first pass to sanity-check the scale of the change
    before committing.

Usage:
    python recompute_lychee_fd_segments.py --dry_run
    python recompute_lychee_fd_segments.py                 # applies the fix
    python recompute_lychee_fd_segments.py --results_root <path> --force
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_common import recompute_speech_segments_from_audio  # noqa: E402

DEFAULT_RESULTS_ROOT = Path(
    "/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/results/inference_results"
)


def _segment_total(segments) -> float:
    return sum(max(0.0, e - s) for s, e in segments) if segments else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT,
                         help="Root dir to search for lychee-fd/*/inference_info.json "
                              f"(default: {DEFAULT_RESULTS_ROOT})")
    parser.add_argument("--force", action="store_true",
                         help="Recompute even for files already marked as fixed")
    parser.add_argument("--dry_run", action="store_true",
                         help="Compute and report only -- write nothing")
    args = parser.parse_args()

    info_paths = sorted(args.results_root.glob("**/lychee-fd/*/inference_info.json"))
    print(f"Found {len(info_paths)} lychee-fd inference_info.json files under {args.results_root}")

    n_processed = 0
    n_skipped_already_done = 0
    n_missing_wav = 0
    n_errors = 0
    total_before = 0.0
    total_after = 0.0
    big_changes: list[tuple[str, float, float]] = []   # (sample_id, before_s, after_s)

    for info_path in info_paths:
        sample_dir = info_path.parent
        sample_id = sample_dir.name
        try:
            data = json.loads(info_path.read_text())
        except Exception as e:
            print(f"[ERROR] {info_path}: failed to parse JSON: {e}")
            n_errors += 1
            continue

        inf = data.get("inference", data)   # some files may be flat, some nested
        if not args.force and "model_speech_segments_recomputed_at" in inf:
            n_skipped_already_done += 1
            continue

        output_wav = sample_dir / "output.wav"
        if not output_wav.exists():
            print(f"[WARN] {sample_id}: no output.wav found, skipping")
            n_missing_wav += 1
            continue

        original_segments = inf.get("model_speech_segments", [])
        try:
            new_segments = recompute_speech_segments_from_audio(output_wav)
        except Exception as e:
            print(f"[ERROR] {sample_id}: recompute failed: {e}")
            n_errors += 1
            continue

        before_s = _segment_total(original_segments)
        after_s = _segment_total(new_segments)
        total_before += before_s
        total_after += after_s
        if after_s - before_s > 1.0:   # more than 1s of previously-invisible speech recovered
            big_changes.append((sample_id, before_s, after_s))

        if not args.dry_run:
            if "model_speech_segments_original_round_trace" not in inf or args.force:
                inf["model_speech_segments_original_round_trace"] = original_segments
            inf["model_speech_segments"] = new_segments
            inf["model_speech_segments_recomputed_at"] = datetime.now(timezone.utc).isoformat()
            info_path.write_text(json.dumps(data, indent=2))

        n_processed += 1

    print()
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processed: {n_processed}")
    print(f"Already fixed (skipped): {n_skipped_already_done}")
    print(f"Missing output.wav: {n_missing_wav}")
    print(f"Errors: {n_errors}")
    print(f"Total reported speaking time -- before: {total_before:.1f}s, after: {total_after:.1f}s "
          f"({'+' if total_after >= total_before else ''}{total_after - total_before:.1f}s)")
    print(f"Samples with >1s of newly-recovered speaking time: {len(big_changes)}")
    if big_changes:
        big_changes.sort(key=lambda x: x[2] - x[1], reverse=True)
        print("Top 10 largest corrections (sample_id, before_s, after_s):")
        for sid, b, a in big_changes[:10]:
            print(f"  {sid}: {b:.2f}s -> {a:.2f}s")


if __name__ == "__main__":
    main()
