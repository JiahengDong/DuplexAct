#!/usr/bin/env python3
"""
Batch processor for CANDOR interruption extraction (Backbiter edition).

Usage:
  python batch_interruptions_backbiter.py                      # process all parts
  python batch_interruptions_backbiter.py --parts 1            # only part 1
  python batch_interruptions_backbiter.py --workers 8          # more parallel workers
  python batch_interruptions_backbiter.py --force              # reprocess already-done
  python batch_interruptions_backbiter.py --dry_run            # show what would run
  python batch_interruptions_backbiter.py --results_dir /path  # global results dir
  python batch_interruptions_backbiter.py --no_gpt             # timing filter only
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CANDOR_ROOT         = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
SCRIPT              = Path("/data/gpfs/projects/punim2960/jiahengd/scripts/Candor/extract_interruptions_backbiter.py")
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/interruptions_backbiter")

DONE_MARKER_GPT    = "interruption_candidates_backbiter/filter2_gpt/all_with_gpt_decisions.json"
DONE_MARKER_NO_GPT = "interruption_candidates_backbiter/filter1_timing/all.json"
QUOTA_EXIT_CODE    = 2
TARGET_PER_TYPE    = 500  # cap per recovery type at aggregation

TIMING_LABELS: dict[str, str] = {
    "windows_checked":    "Windows checked",
    "not_abab":           "Not A-B-A-B alternation",
    "t2_overlaps_t1":     "T2 starts before T1 ends",
    "t3_overlap_small":   "T3 overlap too small",
    "t4_overlaps_t3":     "T4 overlaps T3",
    "t1_no_sent_final":   "T1 no sentence-final punct",
    "t1_few_words":       "T1 too few words",
    "t1_has_pauses":      "T1 has internal pauses",
    "t2_is_question":     "T2 is a question",
    "t2_few_words":       "T2 too few words",
    "t2_pre_words_short": "T2 pre-overlap words too short",
    "t3_has_pauses":      "T3 has internal pauses",
    "t3_too_long":        "T3 too long",
    "t3_empty":           "T3 empty",
    "t4_few_words":       "T4 too few words",
    "duplicate":          "Duplicate",
    "passed":             "PASSED timing",
}


def find_conversations(parts: list[int] | None = None) -> list[Path]:
    convs = []
    for part_dir in sorted(CANDOR_ROOT.glob("part_*")):
        if parts:
            if int(part_dir.name.split("_")[1]) not in parts:
                continue
        for conv_dir in sorted(part_dir.iterdir()):
            if conv_dir.is_dir() and (conv_dir / "transcription" / "transcript_backbiter.csv").exists():
                convs.append(conv_dir)
    return convs


def is_done(conv_dir: Path, no_gpt: bool = False) -> bool:
    marker = DONE_MARKER_NO_GPT if no_gpt else DONE_MARKER_GPT
    return (conv_dir / marker).exists()


def process_one(conv_dir: Path, results_dir: Path | None = None,
                no_gpt: bool = False) -> dict:
    start = time.time()
    cmd = [PYTHON, str(SCRIPT), "--conv_dir", str(conv_dir)]
    if results_dir:
        cmd += ["--results_dir", str(results_dir)]
    if no_gpt:
        cmd.append("--no_gpt")
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - start, 1)

    quota_hit = result.returncode == QUOTA_EXIT_CODE
    success   = result.returncode == 0

    timing_found  = 0
    valid_found   = 0
    timing_stats: dict = {}
    gpt_stats:    dict = {}
    for line in result.stdout.splitlines():
        if "Candidates found:" in line:
            try: timing_found = int(line.split(":")[1].strip())
            except: pass
        elif "Valid after GPT:" in line:
            try: valid_found = int(line.split(":")[1].split("/")[0].strip())
            except: pass
        elif line.startswith("TIMING_STATS:"):
            try: timing_stats = json.loads(line[len("TIMING_STATS:"):])
            except: pass
        elif line.startswith("GPT_STATS:"):
            try: gpt_stats = json.loads(line[len("GPT_STATS:"):])
            except: pass

    return {
        "conv_dir":     str(conv_dir),
        "success":      success,
        "quota_hit":    quota_hit,
        "elapsed":      elapsed,
        "timing_found": timing_found,
        "valid_found":  valid_found,
        "timing_stats": timing_stats,
        "gpt_stats":    gpt_stats,
        "stderr":       result.stderr[-500:] if not success else "",
    }


def aggregate_results(results_dir: Path, all_convs: list[Path],
                      no_gpt: bool = False,
                      target_per_type: int = TARGET_PER_TYPE) -> None:
    import shutil
    from collections import defaultdict

    if no_gpt:
        all_candidates: list[dict] = []
        for conv_dir in all_convs:
            cpath = (conv_dir / "interruption_candidates_backbiter"
                     / "filter1_timing" / "all.json")
            if cpath.exists():
                try:
                    all_candidates.extend(json.load(open(cpath)))
                except Exception:
                    pass
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "timing_candidates.json", "w") as f:
            json.dump(all_candidates, f, indent=2)
        print(f"\n── Timing-only results aggregated → {results_dir} ──")
        print(f"  Total timing candidates: {len(all_candidates)}")
        return

    all_valid: list[dict] = []
    for conv_dir in all_convs:
        vpath = conv_dir / "interruption_candidates_backbiter" / "filter2_gpt" / "valid.json"
        if vpath.exists():
            try:
                all_valid.extend(json.load(open(vpath)))
            except Exception:
                pass

    # ── Cap at target_per_type per recovery type ───────────────────────────────
    counts_by_type: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        itype = c.get("interruption_type") or "unknown"
        if counts_by_type[itype] < target_per_type:
            capped.append(c)
            counts_by_type[itype] += 1
    if len(capped) < len(all_valid):
        print(f"  Capped: {len(all_valid)} → {len(capped)} "
              f"(target={target_per_type}/type)")
    all_valid = capped

    # Wipe the global results dir before rebuilding so stale audio from
    # previous runs (candidates no longer valid) don't linger.
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return

    with open(results_dir / "valid.json", "w") as f:
        json.dump(all_valid, f, indent=2)

    for style in ("hard",):
        subset = [c for c in all_valid if c.get("interruption_style") == style]
        if subset:
            with open(results_dir / f"{style}.json", "w") as f:
                json.dump(subset, f, indent=2)

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
        # Copy audio clips from each candidate's per-conversation audio dir.
        for cand in entries:
            conv_id   = cand["conv_id"]
            base      = f"{conv_id}_t{cand['t1_turn_id']}"
            local_audio = cand.get("audio_dir")
            if local_audio:
                local_dir = Path(local_audio)
            else:
                conv_dirs = list(CANDOR_ROOT.glob(f"*/{conv_id}"))
                if not conv_dirs:
                    continue
                local_dir = conv_dirs[0] / "interruption_candidates_backbiter" / "audio" / style / itype
            for suffix in ("_t1.wav", "_t2.wav", "_t3.wav", "_t4.wav", "_full_stereo.wav"):
                src = local_dir / f"{base}{suffix}"
                if src.exists():
                    shutil.copy2(src, subdir / f"{base}{suffix}")

    print(f"\n── Global results aggregated → {results_dir} ──")
    print(f"  Total valid (capped at {target_per_type}/type): {len(all_valid)}")
    for key, entries in sorted(by_style_type.items()):
        print(f"    {key}: {len(entries)}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process interruption extraction (Backbiter edition).")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--parts",       type=int, nargs="+")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--no_gpt",      action="store_true",
                        help="Run timing filter only; skip GPT classification and audio export.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--target",      type=int, default=TARGET_PER_TYPE,
                        help=f"Max valid candidates per recovery type (default: {TARGET_PER_TYPE})")
    args = parser.parse_args()

    if args.no_gpt and args.results_dir == DEFAULT_RESULTS_DIR:
        args.results_dir = Path(str(DEFAULT_RESULTS_DIR) + "_no_gpt")

    all_convs  = find_conversations(args.parts)
    todo_convs = all_convs if args.force else [
        c for c in all_convs if not is_done(c, args.no_gpt)]
    done_convs = [c for c in all_convs if is_done(c, args.no_gpt)]

    print(f"Found {len(all_convs)} conversations total")
    print(f"  Already processed : {len(done_convs)}")
    print(f"  To process        : {len(todo_convs)}")
    print(f"  Workers           : {args.workers}")
    print(f"  Mode              : {'timing only (--no_gpt)' if args.no_gpt else 'full (with GPT)'}")
    print(f"  Target/type       : {args.target}")
    print(f"  Results dir       : {args.results_dir}")

    if args.dry_run:
        for c in todo_convs:
            print(f"  {c}")
        return

    if not todo_convs:
        print("\nNothing to do. Use --force to reprocess.")
        aggregate_results(args.results_dir, all_convs, args.no_gpt, args.target)
        return

    print(f"\nStarting batch...\n{'─'*60}")
    batch_start   = time.time()
    results       = []
    n_done        = 0
    n_failed      = 0
    total_valid   = 0
    total_timing  = 0
    quota_abort   = False
    global_timing: dict[str, int] = {}
    global_gpt:   dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, c, args.results_dir, args.no_gpt): c
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
                for k, v in res.get("timing_stats", {}).items():
                    global_timing[k] = global_timing.get(k, 0) + v
                for k, v in res.get("gpt_stats", {}).items():
                    if isinstance(v, int):
                        global_gpt[k] = global_gpt.get(k, 0) + v

    elapsed_total = round(time.time() - batch_start, 1)
    print(f"\n{'─'*60}")
    if quota_abort:
        print(f"Batch ABORTED (quota exceeded) after {elapsed_total}s")
    else:
        print(f"Batch complete in {elapsed_total}s")
        print(f"  Succeeded        : {n_done - n_failed}/{len(todo_convs)}")
        print(f"  Failed           : {n_failed}")
        print(f"  Total timing hits: {total_timing}")
        if not args.no_gpt:
            print(f"  Total valid (GPT): {total_valid}")

    if global_timing:
        total_checked = global_timing.get("windows_checked", 1) or 1
        print(f"\n── Timing filter breakdown (all {n_done - n_failed} conversations) ──")
        for key, label in TIMING_LABELS.items():
            val = global_timing.get(key, 0)
            if key == "windows_checked":
                print(f"  {label}: {val}")
            else:
                pct = 100.0 * val / total_checked
                print(f"  {label}: {val} ({pct:.1f}%)")

    if not args.no_gpt and global_gpt:
        judged = global_gpt.get("total_judged", 1) or 1
        sent   = global_gpt.get("total_sent", 0)
        print(f"\n── GPT filter breakdown (all {n_done - n_failed} conversations) ──")
        print(f"  Sent to GPT          : {sent}")
        print(f"  Judged               : {judged}")
        print(f"  Valid after GPT      : {total_valid}")
        print(f"  Acceptance rate      : {100.0 * total_valid / judged:.1f}%")
        print(f"\n  Rejection criteria (% of judged):")
        gpt_criteria = [
            ("turn1_complete = false",       "turn1_complete_false"),
            ("turn1_self_contained = false", "turn1_self_contained_false"),
            ("turn2_responds = false",       "turn2_responds_false"),
            ("turn2_interrupted = false",    "turn2_interrupted_false"),
            ("turn4_meaningful = false",     "turn4_meaningful_false"),
        ]
        for label, key in gpt_criteria:
            val = global_gpt.get(key, 0)
            pct = 100.0 * val / judged
            print(f"    {label}: {val} ({pct:.1f}%)")

    if n_failed:
        for r in results:
            if not r["success"]:
                print(f"  {r['conv_dir']}: {r['stderr']}")

    if not quota_abort:
        aggregate_results(args.results_dir, all_convs, args.no_gpt, args.target)
        report_script = Path(__file__).parent / "generate_interruption_report.py"
        if report_script.exists():
            print("\nGenerating report...")
            report_cmd = [
                PYTHON, str(report_script),
                "--results_dir", str(args.results_dir),
                "--candor_root", str(CANDOR_ROOT),
                "--out", str(args.results_dir / "report.md"),
            ]
            if args.no_gpt:
                report_cmd.append("--no_gpt")
            subprocess.run(report_cmd, check=False)


if __name__ == "__main__":
    main()
