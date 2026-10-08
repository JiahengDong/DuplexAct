#!/usr/bin/env python3
"""
Batch processor for CANDOR interruption extraction.

Usage:
  python batch_interruptions.py                      # process all parts
  python batch_interruptions.py --parts 1            # only part 1
  python batch_interruptions.py --workers 8          # more parallel workers
  python batch_interruptions.py --force              # reprocess already-done conversations
  python batch_interruptions.py --dry_run            # show what would be processed
  python batch_interruptions.py --results_dir /path  # global results dir (default: dataset/../results/interruptions)
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CANDOR_ROOT  = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
SCRIPT       = Path("/data/gpfs/projects/punim2960/jiahengd/scripts/extract_interruptions.py")
PYTHON       = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/interruptions")

DONE_MARKER = "interruption_candidates/filter2_gpt/all_with_gpt_decisions.json"
QUOTA_EXIT_CODE = 2


def find_conversations(parts: list[int] | None = None) -> list[Path]:
    convs = []
    for part_dir in sorted(CANDOR_ROOT.glob("part_*")):
        if parts:
            if int(part_dir.name.split("_")[1]) not in parts:
                continue
        for conv_dir in sorted(part_dir.iterdir()):
            if conv_dir.is_dir():
                convs.append(conv_dir)
    return convs


def is_done(conv_dir: Path) -> bool:
    return (conv_dir / DONE_MARKER).exists()


def process_one(conv_dir: Path, results_dir: Path | None = None) -> dict:
    start = time.time()
    cmd = [PYTHON, str(SCRIPT), "--conv_dir", str(conv_dir)]
    if results_dir:
        cmd += ["--results_dir", str(results_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - start, 1)

    quota_hit = result.returncode == QUOTA_EXIT_CODE
    success   = result.returncode == 0

    # Parse summary counts from stdout
    timing_found = 0
    valid_found  = 0
    for line in result.stdout.splitlines():
        if "Candidates found:" in line:
            try: timing_found = int(line.split(":")[1].strip())
            except: pass
        elif "Valid after GPT:" in line:
            try: valid_found = int(line.split(":")[1].split("/")[0].strip())
            except: pass

    return {
        "conv_dir":     str(conv_dir),
        "success":      success,
        "quota_hit":    quota_hit,
        "elapsed":      elapsed,
        "timing_found": timing_found,
        "valid_found":  valid_found,
        "stderr":       result.stderr[-500:] if not success else "",
    }


def aggregate_results(results_dir: Path, all_convs: list[Path]) -> None:
    """Collect every per-conversation valid.json into the global results directory."""
    all_valid: list[dict] = []
    for conv_dir in all_convs:
        vpath = conv_dir / "interruption_candidates" / "filter2_gpt" / "valid.json"
        if vpath.exists():
            try:
                entries = json.load(open(vpath))
                all_valid.extend(entries)
            except Exception:
                pass

    if not all_valid:
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # Master list
    with open(results_dir / "valid.json", "w") as f:
        json.dump(all_valid, f, indent=2)

    # Per-style lists
    for style in ("hard", "soft"):
        subset = [c for c in all_valid if c.get("interruption_style") == style]
        if subset:
            with open(results_dir / f"{style}.json", "w") as f:
                json.dump(subset, f, indent=2)

    # Per-style/type lists
    from collections import defaultdict
    by_style_type: dict[str, list] = defaultdict(list)
    for c in all_valid:
        style = c.get("interruption_style", "unknown")
        itype = c.get("interruption_type",  "unknown")
        by_style_type[f"{style}/{itype}"].append(c)

    for key, entries in by_style_type.items():
        style, itype = key.split("/", 1)
        subdir = results_dir / style / itype
        subdir.mkdir(parents=True, exist_ok=True)
        with open(subdir / "candidates.json", "w") as f:
            json.dump(entries, f, indent=2)

    print(f"\n── Global results aggregated → {results_dir} ──")
    print(f"  Total valid: {len(all_valid)}")
    for key, entries in sorted(by_style_type.items()):
        print(f"    {key}: {len(entries)}")


def main():
    parser = argparse.ArgumentParser(description="Batch process interruption extraction.")
    parser.add_argument("--workers",     type=int, default=4,
                        help="Parallel workers (default 4)")
    parser.add_argument("--force",       action="store_true",
                        help="Reprocess already-done conversations")
    parser.add_argument("--parts",       type=int, nargs="+",
                        help="Only process specific part numbers")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Show what would be processed, don't run")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f"Global results directory (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args()

    all_convs  = find_conversations(args.parts)
    todo_convs = all_convs if args.force else [c for c in all_convs if not is_done(c)]
    done_convs = [c for c in all_convs if is_done(c)]

    print(f"Found {len(all_convs)} conversations total")
    print(f"  Already processed : {len(done_convs)}")
    print(f"  To process        : {len(todo_convs)}")
    print(f"  Workers           : {args.workers}")
    print(f"  Results dir       : {args.results_dir}")

    if args.dry_run:
        print("\nDry run — conversations to process:")
        for c in todo_convs:
            print(f"  {c}")
        return

    if not todo_convs:
        print("\nNothing to do. Use --force to reprocess.")
        aggregate_results(args.results_dir, all_convs)
        return

    print(f"\nStarting batch...\n{'─'*60}")
    batch_start   = time.time()
    results       = []
    n_done        = 0
    n_failed      = 0
    total_valid   = 0
    total_timing  = 0
    quota_abort   = False

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, c, args.results_dir): c
                   for c in todo_convs}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            n_done += 1

            if res.get("quota_hit"):
                quota_abort = True
                for f in futures:
                    f.cancel()
                print(
                    f"\n{'!'*60}\n"
                    f"  QUOTA EXCEEDED — aborting batch.\n"
                    f"  Failed at: {Path(res['conv_dir']).name}\n"
                    f"  Completed so far: {n_done}/{len(todo_convs)}\n"
                    f"{'!'*60}"
                )
                break

            status = "OK " if res["success"] else "ERR"
            print(
                f"[{n_done:3d}/{len(todo_convs)}] {status}  "
                f"{Path(res['conv_dir']).name}  "
                f"timing={res['timing_found']}  valid={res['valid_found']}  "
                f"{res['elapsed']}s"
            )
            if not res["success"]:
                n_failed += 1
                print(f"         STDERR: {res['stderr']}")
            else:
                total_timing += res["timing_found"]
                total_valid  += res["valid_found"]

    elapsed_total = round(time.time() - batch_start, 1)
    print(f"\n{'─'*60}")
    if quota_abort:
        print(f"Batch ABORTED (quota exceeded) after {elapsed_total}s")
    else:
        print(f"Batch complete in {elapsed_total}s")
        print(f"  Succeeded        : {n_done - n_failed}/{len(todo_convs)}")
        print(f"  Failed           : {n_failed}")
        print(f"  Total timing hits: {total_timing}")
        print(f"  Total valid (GPT): {total_valid}")

    if n_failed:
        print("\nFailed conversations:")
        for r in results:
            if not r["success"]:
                print(f"  {r['conv_dir']}")
                print(f"    {r['stderr']}")

    # Aggregate all valid candidates into the global results directory
    if not quota_abort:
        aggregate_results(args.results_dir, all_convs)
        # Generate statistical report
        report_script = Path(__file__).parent / "generate_interruption_report.py"
        if report_script.exists():
            print("\nGenerating statistical report...")
            subprocess.run(
                [PYTHON, str(report_script),
                 "--results_dir", str(args.results_dir),
                 "--candor_root", str(CANDOR_ROOT)],
                check=False
            )


if __name__ == "__main__":
    main()
