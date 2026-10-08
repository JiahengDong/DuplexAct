#!/usr/bin/env python3
"""
Batch processor for CANDOR pause candidate extraction.

Finds all conversation directories across all parts and runs
extract_pause_candidates.py on each, with parallel workers.

Usage:
  python batch_process.py                        # process all, 4 workers
  python batch_process.py --workers 8            # more parallel workers
  python batch_process.py --force                # reprocess already-done conversations
  python batch_process.py --parts 1 2 3          # only specific parts
  python batch_process.py --dry_run              # show what would be processed
  python batch_process.py --retry_quota          # re-run GPT+audio only on quota-failed convs
  python batch_process.py --results_dir /path    # global results directory
"""

# Exit code 2 from extract_pause_candidates.py means quota exceeded — abort everything.
QUOTA_EXIT_CODE = 2

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CANDOR_ROOT         = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
SCRIPT              = Path("/data/gpfs/projects/punim2960/jiahengd/scripts/Candor/extract_pause_candidates.py")
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/pauses")

DONE_MARKER_GPT    = "filter2_gpt/all_with_gpt_decisions.json"
DONE_MARKER_NO_GPT = "filter1_timing/all.json"

PAUSE_TYPES        = ('inter_sentential', 'intra_sentential', 'floor_holding')
GLOBAL_CAP_DEFAULT = 500   # max valid samples per sub-action type


def find_conversations(parts: list[int] | None = None) -> list[Path]:
    convs = []
    part_dirs = sorted(CANDOR_ROOT.glob("part_*"))
    for part_dir in part_dirs:
        if parts:
            part_num = int(part_dir.name.split("_")[1])
            if part_num not in parts:
                continue
        for conv_dir in sorted(part_dir.iterdir()):
            if conv_dir.is_dir():
                convs.append(conv_dir)
    return convs


def is_done(conv_dir: Path, skip_gpt: bool = False) -> bool:
    marker = DONE_MARKER_NO_GPT if skip_gpt else DONE_MARKER_GPT
    return (conv_dir / "pause_candidates" / marker).exists()


def has_quota_errors(conv_dir: Path) -> bool:
    f = conv_dir / "pause_candidates" / DONE_MARKER_GPT
    if not f.exists():
        return False
    import json
    data = json.load(open(f))
    return any(c.get('gpt_valid') is None for c in data)


TIMING_LABELS = {
    'rej_no_words':           'No transcribed words',
    'rej_trim_failed':        'Trim failed',
    'rej_overlap':            'Overlap with other speaker',
    'rej_mixed_pauses':       'Mixed pause types',
    'rej_pre_gap_too_large':  'Pre-gap too large',
    'rej_no_pauses':          'No valid pauses',
    'rej_before_short':       'Pre-pause segment too short',
    'rej_after_short':        'Post-pause segment too short',
    'rej_too_short':          'Utterance too short',
    'rej_duration_short':     'Duration too short',
    'rej_trim_short':         'After trim too short',
    'rej_trim_dur_short':     'After trim duration too short',
    'rej_other_speech':       'Other speaker speech present',
    'rej_has_pauses':         'Floor-holding has pauses',
    'rej_sentence_short':     'Sentence too short',
    'rej_sentence_large_gap': 'Sentence gap too large',
    'total_sentences':        'Total sentences checked (intra)',
    'passed':                 'Passed timing filter',
}

GPT_CRITERIA_LABELS = {
    'complete_sentences': 'Complete sentences (inter)',
    'complete_sentence':  'Complete sentence (intra)',
    'natural_pause':      'Natural pause position',
    'self_contained':     'Self-contained (no prior context)',
    'is_complete':        'Utterance complete (floor)',
    'is_coherent':        'Utterance coherent (floor)',
}


def count_valid_in_results(results_dir: Path) -> dict[str, int]:
    """
    Return the number of valid samples currently in results_dir per type.
    Counts .json sidecar files (only present with the updated extraction script),
    so old WAV-only results from previous runs don't falsely trigger the soft-stop.
    """
    counts: dict[str, int] = {}
    for ptype in PAUSE_TYPES:
        d = results_dir / ptype
        counts[ptype] = len(list(d.glob('*.json'))) if d.exists() else 0
    return counts


def process_one(conv_dir: Path, gpt_only: bool = False,
                results_dir: Path | None = None,
                skip_gpt: bool = False,
                skip_types: list[str] | None = None) -> dict:
    start = time.time()
    cmd = [PYTHON, str(SCRIPT), "--conv_dir", str(conv_dir)]
    if gpt_only:
        cmd.append("--gpt_only")
    if skip_gpt:
        cmd.append("--skip_gpt")
    if results_dir:
        cmd += ["--results_dir", str(results_dir)]
    if skip_types:
        cmd += ["--skip_types"] + skip_types
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - start, 1)
    quota_hit = result.returncode == QUOTA_EXIT_CODE
    success   = result.returncode == 0

    # Parse counts and structured stats from stdout
    counts       = {"inter": 0, "floor": 0, "intra": 0, "clips": 0}
    timing_stats = {}
    gpt_stats    = {}
    for line in result.stdout.splitlines():
        if "inter_sentential :" in line and "/" in line:
            try: counts["inter"] = int(line.split(":")[1].split("/")[0].strip())
            except: pass
        elif "floor_holding    :" in line and "/" in line:
            try: counts["floor"] = int(line.split(":")[1].split("/")[0].strip())
            except: pass
        elif "intra_sentential :" in line and "/" in line:
            try: counts["intra"] = int(line.split(":")[1].split("/")[0].strip())
            except: pass
        elif "Exported" in line and "audio clips" in line:
            try: counts["clips"] = int(line.strip().split()[1])
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
        "counts":       counts,
        "timing_stats": timing_stats,
        "gpt_stats":    gpt_stats,
        "stderr":       result.stderr[-500:] if not success else "",
    }


def aggregate_results(results_dir: Path, all_convs: list[Path],
                      skip_gpt: bool = False,
                      cap_per_type: int = GLOBAL_CAP_DEFAULT) -> None:
    """Collect per-conversation candidates into global results dir.
    Caps the output to cap_per_type valid samples per sub-action type.
    """

    if skip_gpt:
        all_candidates: list[dict] = []
        for conv_dir in all_convs:
            p = conv_dir / "pause_candidates" / "filter1_timing" / "all.json"
            if p.exists():
                try:
                    all_candidates.extend(json.load(open(p)))
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
        f2 = conv_dir / "pause_candidates" / "filter2_gpt"
        for ptype in PAUSE_TYPES:
            p = f2 / f"{ptype}.json"
            if p.exists():
                try:
                    entries = [c for c in json.load(open(p)) if c.get('gpt_valid')]
                    all_valid.extend(entries)
                except Exception:
                    pass

    if not all_valid:
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-type lists with hard cap applied
    by_type: dict[str, list] = defaultdict(list)
    for c in all_valid:
        by_type[c.get('type', 'unknown')].append(c)

    capped: list[dict] = []
    for ptype in PAUSE_TYPES:
        entries = by_type.get(ptype, [])
        if len(entries) > cap_per_type:
            print(f"  [{ptype}] capping {len(entries)} → {cap_per_type}")
            entries = entries[:cap_per_type]
        by_type[ptype] = entries
        capped.extend(entries)

    # Master list (capped)
    with open(results_dir / "valid.json", "w") as f:
        json.dump(capped, f, indent=2)

    for ptype, entries in by_type.items():
        with open(results_dir / f"{ptype}.json", "w") as f:
            json.dump(entries, f, indent=2)

        if ptype != 'floor_holding':
            by_n: dict[int, list] = defaultdict(list)
            for c in entries:
                by_n[c.get('n_pauses', 1)].append(c)
            for n, subset in by_n.items():
                subdir = results_dir / ptype / f"n{n}"
                subdir.mkdir(parents=True, exist_ok=True)
                with open(subdir / "candidates.json", "w") as f:
                    json.dump(subset, f, indent=2)
        else:
            subdir = results_dir / ptype
            subdir.mkdir(parents=True, exist_ok=True)
            with open(subdir / "candidates.json", "w") as f:
                json.dump(entries, f, indent=2)

    print(f"\n── Global results aggregated → {results_dir} ──")
    print(f"  Total valid (after cap={cap_per_type}/type): {len(capped)}")
    for ptype in PAUSE_TYPES:
        print(f"    {ptype}: {len(by_type.get(ptype, []))}")


def main():
    parser = argparse.ArgumentParser(description="Batch process all CANDOR conversations.")
    parser.add_argument("--workers",      type=int, default=4,  help="Parallel workers (default 4)")
    parser.add_argument("--force",        action="store_true",  help="Reprocess already-done conversations")
    parser.add_argument("--parts",        type=int, nargs="+",  help="Only process specific part numbers")
    parser.add_argument("--dry_run",      action="store_true",  help="Show what would be processed, don't run")
    parser.add_argument("--retry_quota",  action="store_true",  help="Re-run GPT+audio only on quota-failed conversations")
    parser.add_argument("--skip_gpt",     action="store_true",  help="Run timing filter only; skip GPT and audio export")
    parser.add_argument("--results_dir",  type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f"Global results directory (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--cap_per_type", type=int, default=GLOBAL_CAP_DEFAULT,
                        help=f"Max valid samples per sub-action type (default: {GLOBAL_CAP_DEFAULT}). "
                             "Once reached, GPT judging is skipped for that type in subsequent conversations.")
    args = parser.parse_args()

    if args.skip_gpt and args.results_dir == DEFAULT_RESULTS_DIR:
        args.results_dir = Path(str(DEFAULT_RESULTS_DIR) + "_no_gpt")

    all_convs = find_conversations(args.parts)

    if args.retry_quota:
        todo_convs = [c for c in all_convs if has_quota_errors(c)]
        gpt_only   = True
        print(f"Retry-quota mode: {len(todo_convs)} conversations have null GPT results")
    elif args.force:
        todo_convs = all_convs
        gpt_only   = False
    else:
        todo_convs = [c for c in all_convs if not is_done(c, args.skip_gpt)]
        gpt_only   = False

    done_convs = [c for c in all_convs if is_done(c, args.skip_gpt) and not has_quota_errors(c)]

    print(f"Found {len(all_convs)} conversations total")
    print(f"  Cleanly processed : {len(done_convs)}")
    print(f"  To process        : {len(todo_convs)}")
    print(f"  Workers           : {args.workers}")
    print(f"  Mode              : {'timing only (--skip_gpt)' if args.skip_gpt else 'full (with GPT)'}")
    print(f"  Results dir       : {args.results_dir}")
    if gpt_only:
        print(f"  Mode              : GPT+audio only (skipping timing filter)")

    if args.dry_run:
        print("\nDry run — conversations to process:")
        for c in todo_convs:
            print(f"  {c}")
        return

    if not todo_convs:
        print("\nNothing to do. Use --force to reprocess.")
        aggregate_results(args.results_dir, all_convs, args.skip_gpt)
        return

    print(f"\nStarting batch processing...\n{'─'*60}")
    batch_start = time.time()

    results        = []
    n_done         = 0
    n_failed       = 0
    total_clips    = 0
    quota_abort    = False
    global_timing: dict[str, dict[str, int]] = {}
    global_gpt:    dict[str, dict]           = {}

    # Track running global valid counts for soft-stop
    global_valid_counts: dict[str, int] = {t: 0 for t in PAUSE_TYPES}
    if args.results_dir.exists():
        global_valid_counts = count_valid_in_results(args.results_dir)
        capped_types = {t for t, n in global_valid_counts.items() if n >= args.cap_per_type}
        if capped_types:
            print(f"  Already at cap for: {sorted(capped_types)} — will skip GPT for these types")

    def _get_skip_types() -> list[str]:
        return [t for t, n in global_valid_counts.items() if n >= args.cap_per_type]

    # Submit futures incrementally so skip_types reflects current counts
    # before each new conversation is dispatched.
    pending = list(todo_convs)
    futures: dict = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Seed the pool up to max_workers
        while pending and len(futures) < args.workers:
            c = pending.pop(0)
            fut = executor.submit(
                process_one, c, gpt_only, args.results_dir, args.skip_gpt,
                _get_skip_types()
            )
            futures[fut] = c

        while futures:
            future = next(as_completed(futures))
            c_conv  = futures.pop(future)
            res     = future.result()
            results.append(res)
            n_done += 1

            if res.get("quota_hit"):
                quota_abort = True
                for f in futures:
                    f.cancel()
                print(
                    f"\n{'!'*60}\n"
                    f"  QUOTA EXCEEDED — aborting batch.\n"
                    f"  Failed at conversation: {Path(res['conv_dir']).name}\n"
                    f"  Completed so far: {n_done}/{len(todo_convs)}\n"
                    f"{'!'*60}"
                )
                break

            status = "OK " if res["success"] else "ERR"
            c = res["counts"]
            print(
                f"[{n_done:3d}/{len(todo_convs)}] {status}  "
                f"{Path(c_conv).name}  "
                f"inter={c['inter']} floor={c['floor']} intra={c['intra']}  "
                f"clips={c['clips']}  {res['elapsed']}s"
            )
            if not res["success"]:
                n_failed += 1
                print(f"         STDERR: {res['stderr']}")
            else:
                total_clips += c["clips"]
                # Update counts so the next submission uses current skip_types
                if args.results_dir.exists():
                    global_valid_counts = count_valid_in_results(args.results_dir)

            # Dispatch next conversation with updated skip_types
            if pending and not quota_abort:
                c_next = pending.pop(0)
                fut = executor.submit(
                    process_one, c_next, gpt_only, args.results_dir, args.skip_gpt,
                    _get_skip_types()
                )
                futures[fut] = c_next

            # Accumulate timing stats
            for cat, cat_ts in (res.get("timing_stats") or {}).items():
                gs = global_timing.setdefault(cat, {})
                for k, v in cat_ts.items():
                    gs[k] = gs.get(k, 0) + v

            # Accumulate GPT stats
            for cat, cat_gs in (res.get("gpt_stats") or {}).items():
                entry = global_gpt.setdefault(
                    cat, {"total_sent": 0, "valid": 0, "rejected": 0, "criteria": {}})
                entry["total_sent"] += cat_gs.get("total_sent", 0)
                entry["valid"]      += cat_gs.get("valid", 0)
                entry["rejected"]   += cat_gs.get("rejected", 0)
                for k, v in (cat_gs.get("criteria") or {}).items():
                    entry["criteria"][k] = entry["criteria"].get(k, 0) + v

    elapsed_total = round(time.time() - batch_start, 1)
    print(f"\n{'─'*60}")
    if quota_abort:
        print(f"Batch ABORTED (quota exceeded) after {elapsed_total}s")
        print(f"  Completed : {n_done}/{len(todo_convs)}")
        print(f"  Remaining : {len(todo_convs) - n_done} conversations still need processing")
        print(f"  Top up your OpenAI quota then rerun with --retry_quota")
    else:
        print(f"Batch complete in {elapsed_total}s")
        print(f"  Succeeded : {n_done - n_failed}/{len(todo_convs)}")
        print(f"  Failed    : {n_failed}")
    print(f"  Total new audio clips : {total_clips}")

    if n_failed:
        print("\nFailed conversations:")
        for r in results:
            if not r["success"]:
                print(f"  {r['conv_dir']}")
                print(f"    {r['stderr']}")

    # Print timing filter breakdown
    if global_timing:
        print(f"\n{'─'*60}")
        print("Timing filter breakdown:")
        for cat in ('inter_sentential', 'intra_sentential', 'floor_holding'):
            ts = global_timing.get(cat)
            if not ts:
                continue
            print(f"\n  {cat}:")
            for key in sorted(ts, key=lambda k: -ts[k]):
                label = TIMING_LABELS.get(key, key)
                print(f"    {label:<42s}: {ts[key]:6d}")

    # Print GPT filter breakdown
    if global_gpt and not args.skip_gpt:
        print(f"\n{'─'*60}")
        print("GPT filter breakdown:")
        for cat in ('inter_sentential', 'intra_sentential', 'floor_holding'):
            gs = global_gpt.get(cat)
            if not gs:
                continue
            total = gs['total_sent']
            acc   = f"{100 * gs['valid'] / total:.1f}%" if total else "—"
            print(f"\n  {cat}:")
            print(f"    Sent={total}  Valid={gs['valid']}  Rejected={gs['rejected']}  "
                  f"Acceptance={acc}")
            for k, v in sorted((gs.get("criteria") or {}).items(), key=lambda x: -x[1]):
                label = GPT_CRITERIA_LABELS.get(k, k)
                print(f"    Failed '{label}': {v}")

    # Aggregate and report
    if not quota_abort:
        aggregate_results(args.results_dir, all_convs, args.skip_gpt, args.cap_per_type)

        # Save accumulated stats for report generator
        args.results_dir.mkdir(parents=True, exist_ok=True)
        if global_timing:
            with open(args.results_dir / "timing_filter_stats.json", "w") as f:
                json.dump(global_timing, f, indent=2)
        if global_gpt and not args.skip_gpt:
            with open(args.results_dir / "gpt_filter_stats.json", "w") as f:
                json.dump(global_gpt, f, indent=2)

        report_script = Path(__file__).parent / "generate_pause_report.py"
        if report_script.exists():
            print("\nGenerating statistical report...")
            cmd = [PYTHON, str(report_script),
                   "--results_dir", str(args.results_dir),
                   "--candor_root", str(CANDOR_ROOT)]
            if args.skip_gpt:
                cmd.append("--no_gpt")
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
