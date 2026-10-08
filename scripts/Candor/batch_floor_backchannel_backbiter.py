#!/usr/bin/env python3
"""
Batch processor for CANDOR floor-holding + backchannel extraction (Backbiter edition).

Runs all conversations in parallel (per-conv GPT calls), then aggregates results
and caps at TARGET_PER_LABEL valid candidates per floor_holding_label.

Usage:
  python batch_floor_backchannel_backbiter.py                      # all parts
  python batch_floor_backchannel_backbiter.py --parts 1            # only part_001
  python batch_floor_backchannel_backbiter.py --workers 8          # more workers
  python batch_floor_backchannel_backbiter.py --force              # reprocess done
  python batch_floor_backchannel_backbiter.py --dry_run            # show what would run
  python batch_floor_backchannel_backbiter.py --target 5           # 5 per label
  python batch_floor_backchannel_backbiter.py --no_gpt             # timing filter only

NOTE: This batch script drives extract_floor_backchannel_backbiter.py (the original
extractor).  For the relaxed/configurable version use
batch_floor_backchannel_backbiter_relaxed.py instead.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

CANDOR_ROOT         = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
SCRIPT              = Path("/data/gpfs/projects/punim2960/jiahengd/scripts/Candor/extract_floor_backchannel_backbiter.py")
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/floor_backchannel_backbiter")
DEFAULT_TARGET      = 500  # cap per floor_holding_label

FLOOR_LABELS        = ["continue_current", "continue_shorten", "continue_simplify", "none"]
DONE_MARKER_GPT     = "floor_backchannel_backbiter/filter2_gpt/all_with_gpt_decisions.json"
DONE_MARKER_NO_GPT  = "floor_backchannel_backbiter/filter1_timing/all.json"
QUOTA_EXIT_CODE     = 2

# Mirror thresholds from extract_floor_backchannel_backbiter.py — used in report labels.
MIN_T0_WORDS    = 4
MIN_T1_WORDS    = 4
MIN_AFTER_WORDS = 4

# ── Timing rejection labels (mirrors extract script's stats keys) ──────────────

TIMING_LABELS: dict[str, str] = {
    "windows_checked":  "Windows checked",
    # floor_holding/both only — skipped entirely for backchannel
    "same_speaker":     "Same speaker / not B-A pattern (floor_holding/both only)",
    "t0_few_words":     f"T0 < {MIN_T0_WORDS} words (floor_holding/both only)",
    "t0_has_pause":     "T0 has internal pause (floor_holding/both only)",
    # always checked
    "t1_few_words":     f"T1 < {MIN_T1_WORDS} words",
    "t1_has_pause":     "T1 has internal pause (≥ PAUSE_MIN)",
    "t1_overlap":       "T1 overlaps T0 or cut short by external speaker",
    "no_bc_annotation": "No valid BC annotation",
    "before_no_words":  "No T1 words before backchannel (BC starts at T1 onset)",
    "after_few_words":  f"T_after < {MIN_AFTER_WORDS} words",
    "duplicate":        "Duplicate",
    "passed":           "PASSED timing",
}


def find_conversations(parts: list[int] | None = None) -> list[Path]:
    convs = []
    for part_dir in sorted(CANDOR_ROOT.glob("part_*")):
        if parts and int(part_dir.name.split("_")[1]) not in parts:
            continue
        for conv_dir in sorted(part_dir.iterdir()):
            if conv_dir.is_dir() and (conv_dir / "transcription" / "transcript_backbiter.csv").exists():
                convs.append(conv_dir)
    return convs


def is_done(conv_dir: Path, no_gpt: bool = False) -> bool:
    marker = DONE_MARKER_NO_GPT if no_gpt else DONE_MARKER_GPT
    return (conv_dir / marker).exists()


def process_one(conv_dir: Path, results_dir: Path | None = None,
                task: str = "backchannel", no_gpt: bool = False,
                skip_flags: list | None = None) -> dict:
    start = time.time()
    cmd = [PYTHON, str(SCRIPT), "--conv_dir", str(conv_dir), "--task", task]
    if results_dir:
        cmd += ["--results_dir", str(results_dir)]
    if no_gpt:
        cmd.append("--no_gpt")
    if skip_flags:
        cmd.extend(skip_flags)
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - start, 1)

    quota_hit = result.returncode == QUOTA_EXIT_CODE
    success   = result.returncode == 0

    timing_found = 0
    valid_found  = 0
    timing_stats: dict = {}
    gpt_stats:    dict = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "Candidates found:" in stripped:
            try: timing_found = int(stripped.split(":")[-1].strip())
            except: pass
        elif "Valid after GPT:" in stripped:
            try: valid_found = int(stripped.split(":")[1].split("/")[0].strip())
            except: pass
        if line.startswith("TIMING_STATS:"):
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
                      target_per_label: int, no_gpt: bool = False) -> None:
    if no_gpt:
        # Timing-only run: collect all candidates from filter1_timing/all.json
        all_candidates: list[dict] = []
        for conv_dir in all_convs:
            cpath = conv_dir / "floor_backchannel_backbiter" / "filter1_timing" / "all.json"
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
        vpath = conv_dir / "floor_backchannel_backbiter" / "filter2_gpt" / "valid.json"
        if vpath.exists():
            try:
                all_valid.extend(json.load(open(vpath)))
            except Exception:
                pass

    # Cap at target per floor_holding_label
    counts: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        label = c.get("floor_holding_label", "unknown")
        if counts[label] < target_per_label:
            capped.append(c)
            counts[label] += 1
    all_valid = capped

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return

    with open(results_dir / "valid.json", "w") as f:
        json.dump(all_valid, f, indent=2)

    # Nest: floor_holding_label → backchannel_type → [candidates]
    nested: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in all_valid:
        label   = c.get("floor_holding_label", "unknown")
        bc_type = c.get("backchannel_type", "unknown")
        nested[label][bc_type].append(c)

    for label, bc_map in nested.items():
        label_dir = results_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        # Write a flat candidates.json at the label level for convenience
        label_entries = [c for entries in bc_map.values() for c in entries]
        with open(label_dir / "candidates.json", "w") as f:
            json.dump(label_entries, f, indent=2)

        for bc_type, entries in bc_map.items():
            subdir = label_dir / bc_type
            subdir.mkdir(parents=True, exist_ok=True)
            with open(subdir / "candidates.json", "w") as f:
                json.dump(entries, f, indent=2)
            # Copy audio from each candidate's local audio dir
            for cand in entries:
                conv_id   = cand["conv_id"]
                base      = f"{conv_id}_t{cand['t1_turn_id']}"
                local_dir = Path(cand["audio_dir"]) if cand.get("audio_dir") else None
                if local_dir is None:
                    conv_dirs = list(CANDOR_ROOT.glob(f"*/{conv_id}"))
                    if not conv_dirs:
                        continue
                    local_dir = (conv_dirs[0] / "floor_backchannel_backbiter"
                                 / "audio" / bc_type)
                for suffix in ("_t0.wav", "_t1_full.wav", "_backchannel.wav",
                               "_full_stereo.wav"):
                    src = local_dir / f"{base}{suffix}"
                    if src.exists():
                        shutil.copy2(src, subdir / f"{base}{suffix}")

    print(f"\n── Global results aggregated → {results_dir} ──")
    print(f"  Total valid (capped at {target_per_label}/label): {len(all_valid)}")
    for label in FLOOR_LABELS:
        bc_map = nested.get(label, {})
        total  = sum(len(v) for v in bc_map.values())
        print(f"  {label}: {total}")
        for bc_type, entries in sorted(bc_map.items()):
            print(f"    ├─ {bc_type}: {len(entries)}")


BC_TYPES     = ["acknowledgement", "understanding", "agreement",
                "uncertainty", "disagreement", "none"]
GPT_CRITERIA_BY_TASK: dict[str, dict[str, str]] = {
    "backchannel": {
        "turn1_complete_false":        "turn1_complete = false",
        "turn1_self_contained_false":  "turn1_self_contained = false",
    },
    "floor_holding": {
        "turn1_complete_false":        "turn1_complete = false",
        "turn0_complete_false":        "turn0_complete = false",
        "turn0_self_contained_false":  "turn0_self_contained = false",
    },
    "both": {
        "turn1_complete_false":        "turn1_complete = false",
        "turn1_self_contained_false":  "turn1_self_contained = false",
        "turn0_complete_false":        "turn0_complete = false",
        "turn0_self_contained_false":  "turn0_self_contained = false",
    },
}


def _stats(values: list[float]) -> dict:
    import statistics as _stat
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None}
    return {
        "n":      len(values),
        "mean":   round(_stat.mean(values), 3),
        "median": round(_stat.median(values), 3),
        "std":    round(_stat.stdev(values), 3) if len(values) > 1 else 0.0,
        "min":    round(min(values), 3),
        "max":    round(max(values), 3),
    }


def write_report(results_dir: Path, all_convs: list[Path],
                 global_timing: dict[str, int], global_gpt: dict[str, int],
                 no_gpt: bool, task: str, n_convs_ok: int) -> None:
    from collections import Counter
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Load candidates: timing candidates in no_gpt mode, valid candidates otherwise
    candidates: list[dict] = []
    if no_gpt:
        tpath = results_dir / "timing_candidates.json"
        if tpath.exists():
            try:
                candidates = json.load(open(tpath))
            except Exception:
                pass
    else:
        vpath = results_dir / "valid.json"
        if vpath.exists():
            try:
                candidates = json.load(open(vpath))
            except Exception:
                pass

    lines: list[str] = []
    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")
    def row(*cells):
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    def sep(n: int):
        lines.append("| " + " | ".join(["---"] * n) + " |")
    def stat_row(label, values, unit=""):
        s = _stats(values)
        if s["n"] == 0:
            row(label, *["—"] * 6)
            return
        u = f" {unit}" if unit else ""
        row(label,
            f"{s['mean']}{u}", f"{s['median']}{u}", f"{s['std']}{u}",
            f"{s['min']}{u}", f"{s['max']}{u}", s["n"])

    mode = "Timing-only (--no_gpt)" if no_gpt else f"Full (timing + GPT, task={task})"
    h(1, "CANDOR Floor-Holding + Backchannel Report")
    lines.append(f"> Generated: {now}  |  Conversations: {n_convs_ok}  |  Mode: {mode}\n")

    # 1. Timing filter
    sec = 1
    total_checked = global_timing.get("windows_checked", 1) or 1
    h(2, f"{sec}. Timing Filter — Rejection Criteria")
    row("Criterion", "Count", "% of checked")
    sep(3)
    for key, label in TIMING_LABELS.items():
        val = global_timing.get(key, 0)
        if key == "windows_checked":
            row(label, val, "—")
        else:
            row(label, val, f"{100.0 * val / total_checked:.1f}%")
    lines.append("")
    sec += 1

    if no_gpt:
        lines.append(f"*Timing-only run — GPT classification skipped. "
                     f"Statistics below are computed over all {len(candidates)} "
                     f"timing-filter candidates.*\n")
    else:
        # 2. GPT filter
        total_sent   = global_gpt.get("total_sent", 0)
        total_judged = global_gpt.get("total_judged", 0)
        total_valid  = global_gpt.get("total_valid", 0)
        h(2, f"{sec}. GPT Filter — Rejection Criteria")
        row("Metric", "Count")
        sep(2)
        row("Sent to GPT", total_sent)
        row("Judged", total_judged)
        row("Valid", total_valid)
        if total_sent:
            row("Acceptance rate", f"{100.0 * total_valid / total_sent:.1f}%")
        lines.append("")
        if total_judged:
            row("Rejection criterion", "Count", "% of judged")
            sep(3)
            task_criteria = GPT_CRITERIA_BY_TASK.get(task, GPT_CRITERIA_BY_TASK["both"])
            for key, label in task_criteria.items():
                n = global_gpt.get(key, 0)
                lines.append(f"| {label} | {n} | {100.0 * n / total_judged:.1f}% |")
            lines.append("")
        sec += 1

    # Breakdown by floor-holding label
    h(2, f"{sec}. Breakdown by Floor-Holding Label")
    by_floor = Counter(c.get("floor_holding_label") for c in candidates)
    row("Floor-holding label", "Count", "% of total")
    sep(3)
    for lbl in FLOOR_LABELS:
        n = by_floor.get(lbl, 0)
        pct = f"{100.0 * n / len(candidates):.1f}%" if candidates else "—"
        row(lbl, n, pct)
    unclass = sum(v for k, v in by_floor.items() if k not in FLOOR_LABELS)
    if unclass:
        row("_(unclassified/None)_", unclass,
            f"{100.0 * unclass / len(candidates):.1f}%" if candidates else "—")
    row("**Total**", f"**{len(candidates)}**", "")
    lines.append("")
    sec += 1

    # Breakdown by backchannel type
    h(2, f"{sec}. Breakdown by Backchannel Type")
    by_bc = Counter(c.get("backchannel_type") for c in candidates)
    row("Backchannel type", "Count", "% of total")
    sep(3)
    for bt in BC_TYPES:
        n = by_bc.get(bt, 0)
        pct = f"{100.0 * n / len(candidates):.1f}%" if candidates else "—"
        row(bt, n, pct)
    unclass_bc = sum(v for k, v in by_bc.items() if k not in BC_TYPES)
    if unclass_bc:
        row("_(unclassified/None)_", unclass_bc,
            f"{100.0 * unclass_bc / len(candidates):.1f}%" if candidates else "—")
    lines.append("")
    sec += 1

    # Timing statistics
    h(2, f"{sec}. Timing Statistics")
    t0_dur = [float(c["t0_end"]) - float(c["t0_start"]) for c in candidates
              if c.get("t0_start") and c.get("t0_end")]
    t1_dur = [float(c["t1_end"]) - float(c["t1_start"]) for c in candidates
              if c.get("t1_start") and c.get("t1_end")]
    bc_dur = [float(c["bc_end"]) - float(c["bc_start"]) for c in candidates
              if c.get("bc_start") and c.get("bc_end")]
    t0_words = [len((c.get("t0_text") or "").split()) for c in candidates]
    t1_words = [len((c.get("t1_text") or "").split()) for c in candidates]
    bc_words = [int(c["n_backchannel_words"]) for c in candidates
                if c.get("n_backchannel_words") is not None]
    row("Metric", "Mean", "Median", "Std", "Min", "Max", "N")
    sep(7)
    stat_row("T0 (floor-holder) duration", t0_dur, "s")
    stat_row("T1 (backchannel) duration",  t1_dur, "s")
    stat_row("BC segment duration",        bc_dur, "s")
    stat_row("T0 word count",              t0_words, "w")
    stat_row("T1 word count",              t1_words, "w")
    stat_row("BC word count",              bc_words, "w")
    lines.append("")
    sec += 1

    # Per-part summary
    h(2, f"{sec}. Per-Part Summary")
    by_part: dict[str, list] = defaultdict(list)
    for c in candidates:
        conv_id = c.get("conv_id", "")
        for part_dir in sorted(CANDOR_ROOT.glob("part_*")):
            if (part_dir / conv_id).is_dir():
                by_part[part_dir.name].append(c)
                break

    col_label = "Timing candidates" if no_gpt else "Valid candidates"
    row("Part", "Conversations", col_label)
    sep(3)
    for part_dir in sorted(CANDOR_ROOT.glob("part_*")):
        n_convs = sum(1 for d in part_dir.iterdir() if d.is_dir())
        row(part_dir.name, n_convs, len(by_part.get(part_dir.name, [])))
    lines.append("")
    sec += 1

    # Example / all candidates
    n_show = min(50, len(candidates)) if no_gpt else min(20, len(candidates))
    if candidates:
        title = f"Sample Candidates (first {n_show})" if no_gpt \
                else f"Example Candidates (first {n_show})"
        h(2, f"{sec}. {title}")
        row("Conv ID", "Floor label", "BC type", "T0 dur (s)", "BC words",
            "T0 text (truncated)", "BC text")
        sep(7)
        for c in candidates[:n_show]:
            t0_d = round(float(c["t0_end"]) - float(c["t0_start"]), 2) \
                   if c.get("t0_start") and c.get("t0_end") else "—"
            row(
                (c.get("conv_id", ""))[:8] + "…",
                c.get("floor_holding_label") or "—",
                c.get("backchannel_type") or "—",
                t0_d,
                c.get("n_backchannel_words", "—"),
                (c.get("t0_text", "") or "")[:55] + "…",
                c.get("bc_text", ""),
            )
        lines.append("")

    report_path = results_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\nReport written → {report_path}")
    print(f"  {'Timing' if no_gpt else 'Valid'} candidates: {len(candidates)}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process floor-holding + backchannel extraction.")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--parts",       type=int, nargs="+")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--target",      type=int, default=DEFAULT_TARGET,
                        help="Max valid candidates per floor_holding_label")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Results directory (default: auto-suffixed by --task).")
    parser.add_argument("--task", choices=["backchannel", "floor_holding", "both"],
                        default="backchannel",
                        help="GPT evaluation task passed through to extract script.")
    parser.add_argument("--no_gpt",      action="store_true",
                        help="Run timing filter only; skip GPT classification and audio export.")

    # ── Timing skip flags (forwarded to extractor) ───────────────────────────
    parser.add_argument("--skip_t0",             action="store_true")
    parser.add_argument("--skip_t0_words",       action="store_true")
    parser.add_argument("--skip_t0_pauses",      action="store_true")
    parser.add_argument("--skip_t0_timing",      action="store_true")
    parser.add_argument("--skip_t1_words",       action="store_true")
    parser.add_argument("--skip_t1_pauses",      action="store_true")
    parser.add_argument("--skip_t1_interrupted", action="store_true")
    parser.add_argument("--skip_after_words",    action="store_true")

    # ── Threshold overrides ──────────────────────────────────────────────────
    parser.add_argument("--min_t0_words",    type=int,   default=None)
    parser.add_argument("--min_t1_words",    type=int,   default=None)
    parser.add_argument("--min_after_words", type=int,   default=None)
    parser.add_argument("--max_t0_t1_overlap",        type=float, default=None,
                        help="Timing tolerance (s) for T1 not-interrupted check (default 0.05s)")

    # ── GPT skip flags (forwarded to extractor) ──────────────────────────────
    parser.add_argument("--skip_gpt_t0_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t0_self_contained", action="store_true")
    parser.add_argument("--skip_gpt_t1_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t1_self_contained", action="store_true")
    args = parser.parse_args()

    # Auto-suffix results dir based on task
    if args.results_dir is None:
        suffix = "" if args.task == "backchannel" else f"_{args.task}"
        if args.no_gpt:
            suffix += "_no_gpt"
        args.results_dir = Path(str(DEFAULT_RESULTS_DIR) + suffix)

    all_convs  = find_conversations(args.parts)
    todo_convs = all_convs if args.force else [
        c for c in all_convs if not is_done(c, args.no_gpt)]
    done_convs = [c for c in all_convs if is_done(c, args.no_gpt)]

    print(f"Found {len(all_convs)} conversations total")
    print(f"  Already processed : {len(done_convs)}")
    print(f"  To process        : {len(todo_convs)}")
    print(f"  Workers           : {args.workers}")
    print(f"  Mode              : {'timing only (--no_gpt)' if args.no_gpt else 'full (with GPT)'}")
    print(f"  Target per label  : {args.target}")
    print(f"  Results dir       : {args.results_dir}")

    if args.dry_run:
        for c in todo_convs:
            print(f"  {c}")
        return

    if not todo_convs:
        print("\nNothing to do. Use --force to reprocess.")
        aggregate_results(args.results_dir, all_convs, args.target, args.no_gpt)
        return

    print(f"\nStarting batch...\n{'─'*60}")
    batch_start    = time.time()
    results        = []
    n_done         = 0
    n_failed       = 0
    total_valid    = 0
    total_timing   = 0
    quota_abort    = False
    global_timing: dict[str, int] = {}
    global_gpt:    dict[str, int] = {}

    # Collect skip/threshold flags to forward to each extractor subprocess
    skip_flags: list[str] = []
    for flag in ('skip_t0', 'skip_t0_words', 'skip_t0_pauses', 'skip_t0_timing',
                 'skip_t1_words', 'skip_t1_pauses', 'skip_t1_interrupted', 'skip_after_words',
                 'skip_gpt_t0_complete', 'skip_gpt_t0_self_contained',
                 'skip_gpt_t1_complete', 'skip_gpt_t1_self_contained'):
        if getattr(args, flag, False):
            skip_flags.append(f'--{flag}')
    for flag, val in (('min_t0_words', args.min_t0_words),
                      ('min_t1_words', args.min_t1_words),
                      ('min_after_words', args.min_after_words),
                      ('max_t0_t1_overlap', args.max_t0_t1_overlap)):
        if val is not None:
            skip_flags += [f'--{flag}', str(val)]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, c, args.results_dir, args.task,
                                   args.no_gpt, skip_flags): c
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
                    if isinstance(v, (int, float)):
                        global_gpt[k] = global_gpt.get(k, 0) + int(v)

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
        print(f"\n── Timing filter breakdown (all {n_done - n_failed} files) ──")
        for key, label in TIMING_LABELS.items():
            val = global_timing.get(key, 0)
            if key == "windows_checked":
                print(f"  {label}: {val}")
            else:
                pct = 100.0 * val / total_checked
                print(f"  {label}: {val} ({pct:.1f}%)")

    if n_failed:
        for r in results:
            if not r["success"]:
                print(f"  {r['conv_dir']}: {r['stderr']}")

    if not quota_abort:
        aggregate_results(args.results_dir, all_convs, args.target, args.no_gpt)
        write_report(args.results_dir, all_convs, global_timing, global_gpt,
                     args.no_gpt, args.task, n_done - n_failed)


if __name__ == "__main__":
    main()
