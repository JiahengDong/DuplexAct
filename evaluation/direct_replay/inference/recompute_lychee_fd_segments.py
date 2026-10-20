#!/usr/bin/env python3
"""
Idempotent post-inference step for lychee-fd. Run it after direct-replay
inference for lychee-fd and before scoring.

`infer_lychee_fd.py` derives `model_speech_segments` from the model's native
`round_trace["events"]`, whose timestamps are input-timeline offsets
(`idx / INPUT_SR`), not wall-clock arrival times. When a round's generation
overruns its 400ms dispatch budget those offsets drift from when the audio
was actually produced, so the segments can badly under-report speech.
`output.wav` itself is correct.

This script re-derives `model_speech_segments` from each sample's
`output.wav` energy (`realtime_common.recompute_speech_segments_from_audio()`)
for every `inference_results/**/lychee-fd/*/inference_info.json`:
  - The original value is kept under
    `model_speech_segments_original_round_trace` (not overwritten unless
    --force).
  - A `model_speech_segments_recomputed_at` timestamp marks processed files;
    they are skipped unless --force.
  - `--dry_run` reports the change without writing anything.

Usage:
    python evaluation/direct_replay/inference/recompute_lychee_fd_segments.py --dry_run
    python evaluation/direct_replay/inference/recompute_lychee_fd_segments.py
    python evaluation/direct_replay/inference/recompute_lychee_fd_segments.py --results_root <path> --force
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from realtime_common import recompute_speech_segments_from_audio  # noqa: E402

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "inference_results"


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
        if after_s - before_s > 1.0:   # more than 1s of extra speech recovered
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
