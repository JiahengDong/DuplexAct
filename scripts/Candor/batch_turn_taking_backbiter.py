#!/usr/bin/env python3
"""
Batch processor for CANDOR turn-taking extraction (Backbiter edition).

Runs all conversations in parallel, then aggregates and caps at
TARGET_PER_CAT valid candidates per timing category.

Usage:
  python batch_turn_taking_backbiter.py                      # all parts
  python batch_turn_taking_backbiter.py --parts 1            # only part_001
  python batch_turn_taking_backbiter.py --workers 8          # more workers
  python batch_turn_taking_backbiter.py --force              # reprocess done
  python batch_turn_taking_backbiter.py --dry_run            # show what would run
  python batch_turn_taking_backbiter.py --target 5           # 5 per category
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CANDOR_ROOT         = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
SCRIPT              = Path("/data/gpfs/projects/punim2960/jiahengd/scripts/Candor/extract_turn_taking_backbiter.py")
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/turn_taking_backbiter")
DEFAULT_TARGET      = 500

CATEGORIES    = ["take_turn_immediately", "take_turn_with_latency", "take_turn_with_overlap"]
DONE_MARKER_GPT    = "turn_taking_backbiter/filter2_gpt/all_with_gpt_decisions.json"
DONE_MARKER_NO_GPT = "turn_taking_backbiter/filter1_timing/all.json"
QUOTA_EXIT_CODE    = 2

TIMING_CRITERION_LABELS = {
    'pairs_checked':         'Consecutive pairs checked',
    'rej_same_speaker':      'Rejected: same speaker',
    'rej_t2_too_short':      'Rejected: T2 too few words (< 4, cheap gate)',
    'rej_t1_too_short':      'Rejected: T1 too few words (< 4)',
    'rej_t1_incomplete':     'Rejected: T1 no sentence-final punctuation',
    'rej_t1_has_pause':      'Rejected: T1 has internal pause',
    'rej_t2_words_short':    'Rejected: T2 word-level recheck too few words (< 4)',
    'rej_gap_too_large':     'Rejected: gap > 3.0 s',
    'rej_overlap_too_large': 'Rejected: overlap > 1.5 s',
    'rej_duplicate':         'Rejected: duplicate candidate',
    'passed':                'PASSED timing filter',
}

GPT_CRITERIA_LABELS = {
    'turn1_complete':        'turn1_complete = false',
    'turn1_self_contained':  'turn1_self_contained = false',
    'turn2_responds':        'turn2_responds = false',
    'category_appropriate':  'category_appropriate = false',
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
                no_gpt: bool = False,
                max_valid_per_category: int = DEFAULT_TARGET) -> dict:
    start = time.time()
    cmd = [PYTHON, str(SCRIPT), "--conv_dir", str(conv_dir),
           "--max_valid_per_category", str(max_valid_per_category)]
    if results_dir:
        cmd += ["--results_dir", str(results_dir)]
    if no_gpt:
        cmd.append("--no_gpt")
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - start, 1)

    quota_hit = result.returncode == QUOTA_EXIT_CODE
    success   = result.returncode == 0

    timing_stats: dict        = {}
    gpt_stats: dict           = {}
    valid_candidates: list    = []
    send_by_cat: dict         = {c: 0 for c in CATEGORIES}
    valid_by_cat: dict        = {c: 0 for c in CATEGORIES}

    for ln in result.stdout.splitlines():
        stripped = ln.strip()
        if ln.startswith("TIMING_STATS:"):
            try:
                timing_stats = json.loads(ln[len("TIMING_STATS:"):])
            except Exception:
                pass
        elif ln.startswith("GPT_STATS:"):
            try:
                gpt_stats = json.loads(ln[len("GPT_STATS:"):])
            except Exception:
                pass
        elif ln.startswith("VALID_JSON:"):
            try:
                valid_candidates = json.loads(ln[len("VALID_JSON:"):])
            except Exception:
                pass
        else:
            for cat in CATEGORIES:
                if stripped.startswith(f"SEND_TO_LLM {cat}:"):
                    try:
                        send_by_cat[cat] = int(stripped.split(":")[-1].strip())
                    except Exception:
                        pass
                    break
                elif stripped.startswith(f"VALID_AFTER_LLM {cat}:"):
                    try:
                        valid_by_cat[cat] = int(stripped.split(":")[-1].strip())
                    except Exception:
                        pass
                    break

    timing_n = timing_stats.get("passed", 0)
    valid_n  = sum(valid_by_cat.values()) or gpt_stats.get("valid", 0)

    return {
        "conv_dir":         str(conv_dir),
        "success":          success,
        "quota_hit":        quota_hit,
        "elapsed":          elapsed,
        "timing_n":         timing_n,
        "send_by_cat":      send_by_cat,
        "valid_by_cat":     valid_by_cat,
        "timing_stats":     timing_stats,
        "gpt_stats":        gpt_stats,
        "valid_candidates": valid_candidates,
        "stderr":           result.stderr[-500:] if not success else "",
    }


def aggregate_results(results_dir: Path, completed: list[dict],
                      target_per_cat: int) -> list[dict]:
    all_valid: list[dict] = []
    for res in completed:
        all_valid.extend(res.get("valid_candidates", []))

    # Fallback: load from per-conversation valid.json files if in-memory is empty
    if not all_valid:
        for res in completed:
            vpath = Path(res["conv_dir"]) / "turn_taking_backbiter" / "filter2_gpt" / "valid.json"
            if vpath.exists():
                try:
                    all_valid.extend(json.load(open(vpath)))
                except Exception:
                    pass

    # Cap at target per category
    counts: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        cat = c.get("category", "unknown")
        if counts[cat] < target_per_cat:
            capped.append(c)
            counts[cat] += 1
    all_valid = capped

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return []

    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "valid.json", "w") as f:
        json.dump(all_valid, f, indent=2)

    by_cat: dict[str, list] = defaultdict(list)
    for c in all_valid:
        by_cat[c.get("category", "unknown")].append(c)

    for cat in CATEGORIES:
        subdir = results_dir / cat
        subdir.mkdir(parents=True, exist_ok=True)
        with open(subdir / "candidates.json", "w") as f:
            json.dump(entries, f, indent=2)
        for cand in entries:
            conv_id   = cand["conv_id"]
            base      = f"{conv_id}_t{cand['t1_turn_id']}"
            local_dir = Path(cand["audio_dir"]) if cand.get("audio_dir") else None
            if local_dir is None:
                conv_dirs = list(CANDOR_ROOT.glob(f"*/{conv_id}"))
                if not conv_dirs:
                    continue
                local_dir = conv_dirs[0] / "turn_taking_backbiter" / "audio" / cat
            for suffix in ("_t1.wav", "_t2.wav", "_full_stereo.wav"):
                src = local_dir / f"{base}{suffix}"
                if src.exists():
                    shutil.copy2(src, subdir / f"{base}{suffix}")

    print(f"\n── Global results aggregated → {results_dir} ──")
    print(f"  Total valid (capped at {target_per_cat}/category): {len(all_valid)}")
    for cat in CATEGORIES:
        print(f"    {cat}: {len(by_cat.get(cat, []))}")
    return all_valid


def _print_filter_report(results: list[dict],
                         results_dir: Path | None = None,
                         no_gpt: bool = False) -> None:
    from datetime import datetime
    agg_timing: dict[str, int] = {k: 0 for k in TIMING_CRITERION_LABELS}
    by_cat_timing: dict[str, int] = {c: 0 for c in CATEGORIES}

    agg_sent = agg_judged = agg_valid = 0
    agg_criteria: dict[str, int] = {k: 0 for k in GPT_CRITERIA_LABELS}
    by_cat_valid: dict[str, int] = {c: 0 for c in CATEGORIES}

    for r in results:
        if not r["success"]:
            continue
        ts = r.get("timing_stats", {})
        for k in TIMING_CRITERION_LABELS:
            agg_timing[k] += ts.get(k, 0)
        for c in CATEGORIES:
            by_cat_timing[c] += ts.get("by_category", {}).get(c, 0)
        gs = r.get("gpt_stats", {})
        agg_sent   += gs.get("total_sent",   0)
        agg_judged += gs.get("total_judged", 0)
        agg_valid  += gs.get("valid",        0)
        for k in GPT_CRITERIA_LABELS:
            agg_criteria[k] += gs.get("criteria", {}).get(k, 0)
        for c in CATEGORIES:
            by_cat_valid[c] += gs.get("by_category", {}).get(c, 0)

    base = agg_timing.get("pairs_checked", 1) or 1
    print(f"\n{'═'*65}")
    print("TIMING FILTER STATISTICS  (Candor)")
    print(f"{'═'*65}")
    for k, label in TIMING_CRITERION_LABELS.items():
        v = agg_timing[k]
        pct = 100.0 * v / base
        print(f"  {label:<55s}: {v:>6d}  ({pct:5.1f}%)")
    print(f"\n  -- Passed timing filter by category --")
    passed = agg_timing.get("passed", 1) or 1
    for c in CATEGORIES:
        v = by_cat_timing[c]
        pct = 100.0 * v / passed if passed else 0.0
        print(f"  {c:<55s}: {v:>6d}  ({pct:5.1f}%)")

    if agg_sent:
        judged_base = agg_judged or 1
        print(f"\n{'═'*65}")
        print("GPT FILTER STATISTICS  (Candor)")
        print(f"{'═'*65}")
        print(f"  Sent={agg_sent}  Judged={agg_judged}  Valid={agg_valid}")
        print(f"\n  -- Rejection criteria (of {agg_judged} judged) --")
        for k, label in GPT_CRITERIA_LABELS.items():
            v = agg_criteria[k]
            pct = 100.0 * v / judged_base
            print(f"  {label:<55s}: {v:>6d}  ({pct:5.1f}%)")
        print(f"\n  -- Valid by category --")
        for c in CATEGORIES:
            print(f"    {c}: {by_cat_valid.get(c, 0)}")
    print(f"{'═'*65}")

    if results_dir is None:
        return

    # ── Write markdown report ─────────────────────────────────────────────────
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "Timing-only (--no_gpt)" if no_gpt else "Full (timing + GPT)"
    lines: list[str] = []
    def row(*cells):
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    def sep(n: int):
        lines.append("| " + " | ".join(["---"] * n) + " |")

    lines.append("# CANDOR Turn-Taking Report")
    lines.append(f"\nGenerated: {now}  |  Conversations: {sum(1 for r in results if r['success'])}  |  Mode: {mode}")
    lines.append("\n---")

    # Timing filter
    lines.append("\n## 1. Timing Filter\n")
    row("Criterion", "Count", "% of checked")
    sep(3)
    for k, label in TIMING_CRITERION_LABELS.items():
        v = agg_timing[k]
        row(label, v, f"{100.0 * v / base:.1f}%")
    lines.append("")
    lines.append("**Passed by category:**\n")
    row("Category", "Count", "% of passed")
    sep(3)
    for c in CATEGORIES:
        v = by_cat_timing[c]
        row(c, v, f"{100.0 * v / passed:.1f}%" if passed else "—")

    if no_gpt:
        lines.append(f"\n*Timing-only run — GPT step skipped.*\n")
    elif agg_sent:
        judged_base = agg_judged or 1
        lines.append("\n## 2. GPT Filter\n")
        lines.append(f"- Sent to GPT: {agg_sent}")
        lines.append(f"- Judged: {agg_judged}")
        lines.append(f"- Valid: {agg_valid}")
        lines.append("")
        row("Rejection criterion", "Count", "% of judged")
        sep(3)
        for k, label in GPT_CRITERIA_LABELS.items():
            v = agg_criteria[k]
            row(label, v, f"{100.0 * v / judged_base:.1f}%")
        lines.append("\n## 3. Valid by Category\n")
        row("Category", "Count")
        sep(2)
        for c in CATEGORIES:
            row(c, by_cat_valid.get(c, 0))
        row("**Total**", f"**{agg_valid}**")

        # Example candidates
        valid: list[dict] = []
        vpath = results_dir / "valid.json"
        if vpath.exists():
            try:
                valid = json.load(open(vpath))
            except Exception:
                pass
        if valid:
            lines.append(f"\n## 4. Example Candidates (first {min(20, len(valid))})\n")
            row("Conv ID", "Category", "T1 text (truncated)", "T2 text (truncated)")
            sep(4)
            for c in valid[:20]:
                row(
                    (c.get("conv_id", ""))[:8] + "…",
                    c.get("category", ""),
                    (c.get("t1_text", "") or "")[:50] + "…",
                    (c.get("t2_text", "") or "")[:50] + "…",
                )

    report_path = results_dir / "report.md"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\nReport written → {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process turn-taking extraction (Backbiter edition).")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--parts",       type=int, nargs="+")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--no_gpt",      action="store_true",
                        help="Timing-filter only, skip GPT and audio.")
    parser.add_argument("--target",      type=int, default=DEFAULT_TARGET,
                        help="Max valid candidates per timing category")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    all_convs  = find_conversations(args.parts)
    todo_convs = all_convs if args.force else [
        c for c in all_convs if not is_done(c, args.no_gpt)]
    done_convs = [c for c in all_convs if is_done(c, args.no_gpt)]

    print(f"\n{'═'*70}")
    print("  CANDOR TURN-TAKING BATCH PROCESSOR")
    print(f"  Found       : {len(all_convs)} conversations")
    print(f"  To process  : {len(todo_convs)}  (already done: {len(done_convs)})")
    print(f"  Workers     : {args.workers}")
    print(f"  Target/cat  : {args.target}")
    print(f"  Mode        : {'TIMING ONLY' if args.no_gpt else 'FULL (timing + GPT)'}")
    print(f"  Results dir : {args.results_dir}")
    print(f"{'═'*70}\n")

    if args.dry_run:
        for c in todo_convs:
            print(f"  {c}")
        return

    if not todo_convs:
        print("Nothing to do. Use --force to reprocess.")
        aggregate_results(args.results_dir, [], args.target)
        return

    print(f"Starting batch...\n{'─'*60}")
    batch_start = time.time()
    results: list[dict] = []
    n_done = n_failed = 0
    quota_abort = False

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, c, args.results_dir,
                            args.no_gpt, args.target): c
            for c in todo_convs
        }
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

            timing_n = res.get("timing_n", 0)
            vbc      = res.get("valid_by_cat", {})
            valid_n  = sum(vbc.values())
            status   = "OK " if res["success"] else "ERR"
            cat_str  = "  ".join(f"{c[:4]}={vbc.get(c,0)}" for c in CATEGORIES)
            print(
                f"[{n_done:3d}/{len(todo_convs)}] {status}  "
                f"{Path(res['conv_dir']).name}  "
                f"timing={timing_n}  {cat_str}  ({res['elapsed']}s)"
            )
            if not res["success"]:
                n_failed += 1
                print(f"         STDERR: {res['stderr']}")

    elapsed_total = round(time.time() - batch_start, 1)
    print(f"\n{'─'*60}")
    if quota_abort:
        print(f"Batch ABORTED (quota exceeded) after {elapsed_total}s")
    else:
        print(f"Batch complete in {elapsed_total}s")
        print(f"  Succeeded : {n_done - n_failed}/{len(todo_convs)}")
        print(f"  Failed    : {n_failed}")

    _print_filter_report(results, args.results_dir, args.no_gpt)

    if not quota_abort:
        aggregate_results(args.results_dir, results, args.target)


if __name__ == "__main__":
    main()
