#!/usr/bin/env python3
"""
Batch processor for Loqui turn-taking extraction.

Results layout:
  <results_dir>/
    valid.json
    take_turn_immediately/
      sequential.json
      *.wav
    take_turn_with_latency/
      sequential.json
      *.wav
    take_turn_with_overlap/
      sequential.json   ← Pattern A overlap (annotation noise gap < 0, if any)
      structural.json   ← Pattern B overlap (via structural overlap turn)
      *.wav

Usage:
  python batch_turn_taking_loqui.py
  python batch_turn_taking_loqui.py --workers 8 --force
  python batch_turn_taking_loqui.py --no_gpt --no_audio
"""

import argparse
import os
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

LOQUI_ROOT          = Path(os.environ.get(
    "LOQUI_ROOT", "/data/gpfs/projects/punim2960/jiahengd/dataset/Loqui"))

def resolve_trans_dir(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "Transcriptions").is_dir():
        return root / "Transcriptions"
    nested = root / "LoquiPackage" / "Transcriptions"
    if nested.is_dir():
        return nested
    if any(root.glob("*.final.trs")):
        return root
    return nested

LOQUI_TRANS_DIR     = resolve_trans_dir(LOQUI_ROOT)
SCRIPT              = Path(__file__).resolve().parent / "extract_turn_taking_loqui.py"
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/turn_taking_loqui")
DEFAULT_REPORT      = DEFAULT_RESULTS_DIR / "report.md"

CATEGORIES       = ("take_turn_immediately", "take_turn_with_latency", "take_turn_with_overlap")
TARGET_PER_CAT   = 500
QUOTA_EXIT_CODE  = 2

# ── Timing rejection labels ────────────────────────────────────────────────────

TIMING_LABELS: dict[str, str] = {
    "pairs_checked":           "Pairs checked",
    "rej_same_speaker":        "Rejected: same speaker",
    "rej_t2_too_short":        "Rejected: T2 too few words (< 4, cheap gate)",
    "rej_t1_too_short":        "Rejected: T1 too few words (< 4)",
    "rej_t1_has_pause":        "Rejected: T1 contains <P> pause marker",
    "rej_t1_anon":             "Rejected: T1 non-standard / PII marker",
    "rej_t2_words_short":      "Rejected: T2 word-level recheck too few words (< 4)",
    "rej_gap_too_large":       "Rejected: gap > 3.0 s",
    "rej_overlap_too_large":   "Rejected: overlap > 1.5 s",
    "rej_ovl_b_too_short":     "Rejected: [ovl] B's Who-block < 4 words",
    "rej_duplicate":           "Rejected: duplicate candidate",
    "passed":                  "PASSED timing filter",
}

GPT_CRITERIA_LABELS: dict[str, str] = {
    "turn1_complete":       "turn1_complete=false",
    "turn1_self_contained": "turn1_self_contained=false",
    "turn2_responds":       "turn2_responds=false",
    "category_appropriate": "category_appropriate=false",
}


# ── File discovery ─────────────────────────────────────────────────────────────

def find_files() -> list[dict]:
    entries = []
    for trs in sorted(LOQUI_TRANS_DIR.glob("*.final.trs")):
        file_id = trs.stem.replace(".final", "")
        wav = trs.with_name(f"{file_id}.san.wav")
        if not wav.exists():
            print(f"  [warn] no .san.wav for {trs.name}")
            continue
        entries.append({"file_id": file_id, "trs_path": trs, "wav_path": wav})
    return entries


def is_done(entry: dict, results_dir: Path) -> bool:
    return (results_dir / entry["file_id"] / "valid.json").exists()


# ── Per-file runner ────────────────────────────────────────────────────────────

def process_one(entry: dict, results_dir: Path,
                no_gpt: bool, no_audio: bool, force: bool) -> dict:
    cmd = [
        PYTHON, str(SCRIPT),
        "--file_id",              entry["file_id"],
        "--trs_path",             str(entry["trs_path"]),
        "--wav_path",             str(entry["wav_path"]),
        "--results_dir",          str(results_dir),
        "--max_valid_per_category", str(TARGET_PER_CAT),
    ]
    if no_gpt:   cmd.append("--no_gpt")
    if no_audio: cmd.append("--no_audio")
    if force:    cmd.append("--force")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    timing_stats:     dict       = {}
    gpt_stats:        dict       = {}
    valid_candidates: list[dict] = []

    for ln in result.stdout.splitlines():
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

    return {
        "file_id":          entry["file_id"],
        "success":          result.returncode == 0,
        "elapsed":          elapsed,
        "timing_stats":     timing_stats,
        "gpt_stats":        gpt_stats,
        "valid_candidates": valid_candidates,
        "stderr":           result.stderr[-500:] if result.returncode != 0 else "",
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_results(results_dir: Path, completed: list[dict]) -> list[dict]:
    all_valid: list[dict] = []
    for res in completed:
        all_valid.extend(res.get("valid_candidates", []))

    # Cap per category
    counts: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        cat = c.get("category", "unknown") or "unknown"
        if counts[cat] < TARGET_PER_CAT:
            capped.append(c)
            counts[cat] += 1
    all_valid = capped

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return []

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "valid.json").write_text(json.dumps(all_valid, indent=2))

    # Per-category → per-pattern JSON files
    # Structure: results_dir/<category>/<pattern>.json
    by_cat_pat: dict[str, dict[str, list]] = {
        cat: {"sequential": [], "overlap": []} for cat in CATEGORIES
    }
    for c in all_valid:
        cat = c.get("category", "unknown") or "unknown"
        pat = c.get("pattern",  "sequential") or "sequential"
        if cat in by_cat_pat:
            by_cat_pat[cat][pat].append(c)

    for cat in CATEGORIES:
        cat_dir = results_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for pat in ("sequential", "overlap"):
            cands = by_cat_pat.get(cat, {}).get(pat, [])
            (cat_dir / f"{pat}.json").write_text(json.dumps(cands, indent=2))

    print(f"\n── Results aggregated → {results_dir} ──")
    print(f"  Total valid: {len(all_valid)}")
    for cat in CATEGORIES:
        sub = by_cat_pat.get(cat, {})
        n_s = len(sub.get("sequential", []))
        n_o = len(sub.get("overlap",    []))
        if n_s + n_o:
            print(f"    {cat}: sequential={n_s}, overlap={n_o}, total={n_s+n_o}")

    return all_valid


# ── Console report ─────────────────────────────────────────────────────────────

def _print_filter_report(results: list[dict]) -> None:
    t_agg: dict = {}
    g_sent = g_valid = g_judged = 0
    cat_valid: dict[str, int] = defaultdict(int)

    for r in results:
        if not r["success"]:
            continue
        for k, v in r["timing_stats"].items():
            if k != "by_category" and isinstance(v, (int, float)):
                t_agg[k] = t_agg.get(k, 0) + v
        gs = r["gpt_stats"]
        g_sent   += gs.get("total_sent",   0)
        g_valid  += gs.get("valid",        0)
        g_judged += gs.get("total_judged", 0)
        for cat, n in (gs.get("by_category") or {}).items():
            cat_valid[cat] += n
        # merge by_category timing
        for cat, n in (r["timing_stats"].get("by_category") or {}).items():
            t_agg[f"by_cat_{cat}"] = t_agg.get(f"by_cat_{cat}", 0) + n

    total  = t_agg.get("pairs_checked", 0)
    passed = t_agg.get("passed",        0)

    print(f"\n{'═'*70}")
    print("  TURN-TAKING EXTRACTION — FILTER REPORT  (Loqui)")
    print(f"{'═'*70}")
    print(f"\n── Timing filter ──")
    print(f"  Pairs checked: {total:>6}")
    for key, label in TIMING_LABELS.items():
        if key == "pairs_checked":
            continue
        n = t_agg.get(key, 0)
        if n:
            pct = f"{100*n/total:.1f}%" if total else "—"
            print(f"  Rejected: {label:<55} {n:>5}  ({pct})")
    if total:
        print(f"  PASSED timing : {passed:>6}  ({100*passed/total:.1f}%)")
        for cat in CATEGORIES:
            n = t_agg.get(f"by_cat_{cat}", 0)
            print(f"    {cat}: {n}")

    if g_sent:
        print(f"\n── GPT filter ──")
        print(f"  Sent={g_sent}, Judged={g_judged}, Valid={g_valid} "
              f"({100*g_valid/g_sent:.1f}%)")
        for cat in CATEGORIES:
            n = cat_valid.get(cat, 0)
            if n:
                print(f"    {cat}: {n}")
    print(f"\n{'═'*70}\n")


# ── Markdown report ────────────────────────────────────────────────────────────

def write_md_report(results: list[dict], all_valid: list[dict],
                    report_path: Path, no_gpt: bool) -> None:
    t_agg: dict = {}
    g_sent = g_valid = g_judged = 0
    g_criteria: dict[str, int] = defaultdict(int)
    n_files_ok = 0

    for r in results:
        if not r["success"]:
            continue
        n_files_ok += 1
        for k, v in r["timing_stats"].items():
            if k != "by_category" and isinstance(v, (int, float)):
                t_agg[k] = t_agg.get(k, 0) + v
        for cat, n in (r["timing_stats"].get("by_category") or {}).items():
            t_agg[f"by_cat_{cat}"] = t_agg.get(f"by_cat_{cat}", 0) + n
        gs = r["gpt_stats"]
        g_sent   += gs.get("total_sent",   0)
        g_valid  += gs.get("valid",        0)
        g_judged += gs.get("total_judged", 0)
        for k, v in (gs.get("criteria") or {}).items():
            g_criteria[k] += v

    total  = t_agg.get("pairs_checked", 0)
    passed = t_agg.get("passed",        0)

    # Fallback: reload from per-category JSON files
    if not all_valid:
        all_valid = []
        for cat in CATEGORIES:
            for pat in ("sequential", "overlap"):
                p = report_path.parent / cat / f"{pat}.json"
                if p.exists():
                    all_valid.extend(json.loads(p.read_text()))
        if all_valid:
            print(f"  (reloaded {len(all_valid)} valid candidates from disk)")

    total_valid = len(all_valid)

    # Build groupings
    by_cat:     dict[str, list] = defaultdict(list)
    by_pat:     dict[str, list] = defaultdict(list)
    by_cat_pat: dict[str, dict[str, list]] = {
        cat: {"sequential": [], "overlap": []} for cat in CATEGORIES}

    for c in all_valid:
        cat = c.get("category", "unknown") or "unknown"
        pat = c.get("pattern",  "sequential") or "sequential"
        by_cat[cat].append(c)
        by_pat[pat].append(c)
        if cat in by_cat_pat:
            by_cat_pat[cat][pat].append(c)

    lines = [
        "# Loqui Turn-Taking Extraction Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Files processed: {n_files_ok} / {len(results)}  ",
        f"GPT filter: {'**disabled** (--no_gpt)' if no_gpt else 'enabled'}  ",
        "",
        "---",
        "",
        "## 1. Overall Summary",
        "",
        "| Stage | Count |",
        "| --- | --- |",
        f"| Pairs examined (T1 → T2) | {total} |",
        f"| Timing-filter passed | {passed} |",
        f"| Sent to GPT | {g_sent} |",
        f"| GPT-validated (final) | {total_valid} |",
    ]
    if g_sent:
        lines.append(f"| GPT acceptance rate | {100*total_valid/g_sent:.1f}% |")

    # ── Section 2: Category breakdown ──────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 2. Category Breakdown",
        "",
        "| Category | Timing passed | GPT valid | % of valid |",
        "| --- | --- | --- | --- |",
    ]
    for cat in CATEGORIES:
        t_n = t_agg.get(f"by_cat_{cat}", 0)
        g_n = len(by_cat.get(cat, []))
        pct = f"{100*g_n/total_valid:.1f}%" if total_valid else "—"
        lines.append(f"| {cat} | {t_n} | {g_n} | {pct} |")
    lines.append(f"| **Total** | **{passed}** | **{total_valid}** | |")

    # ── Section 3: Pattern breakdown ───────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 3. Structural Pattern Breakdown",
        "",
        "Sequential — consecutive single-speaker turns T1→T2.  ",
        "Overlap — T1(A) → overlap(A+B) → T2(B) structural overlap turn.  ",
        "",
        "| Category | Sequential | Overlap | Total |",
        "| --- | --- | --- | --- |",
    ]
    for cat in CATEGORIES:
        sub = by_cat_pat.get(cat, {})
        n_s = len(sub.get("sequential", []))
        n_o = len(sub.get("overlap",    []))
        lines.append(f"| {cat} | {n_s} | {n_o} | {n_s+n_o} |")
    tot_s = len(by_pat.get("sequential", []))
    tot_o = len(by_pat.get("overlap",    []))
    lines.append(f"| **Total** | **{tot_s}** | **{tot_o}** | **{total_valid}** |")

    # ── Section 4: Gap distribution ────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 4. Gap Distribution",
        "",
        "| Category | Min gap (s) | Median gap (s) | Max gap (s) |",
        "| --- | --- | --- | --- |",
    ]
    for cat in CATEGORIES:
        cands = by_cat.get(cat, [])
        if cands:
            gaps = sorted(c.get("gap", 0) for c in cands)
            mid  = gaps[len(gaps) // 2]
            lines.append(f"| {cat} | {min(gaps):.3f} | {mid:.3f} | {max(gaps):.3f} |")
        else:
            lines.append(f"| {cat} | — | — | — |")

    # ── Section 5: Timing Filter Details ───────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 5. Timing Filter Details",
        "",
        f"Pairs examined: **{total}** → passed: **{passed}**"
        + (f" ({100*passed/total:.1f}%)" if total else ""),
        "",
        "| Rejection reason | Count | % of pairs |",
        "| --- | --- | --- |",
    ]
    for key, label in TIMING_LABELS.items():
        if key == "pairs_checked":
            continue
        n = t_agg.get(key, 0)
        if n:
            pct = f"{100*n/total:.1f}%" if total else "—"
            lines.append(f"| {label} | {n} | {pct} |")

    if not no_gpt and g_sent:
        lines += [
            "",
            "---",
            "",
            "## 6. GPT Filter Details",
            "",
            f"Sent: **{g_sent}** | Judged: **{g_judged}** | "
            f"Valid: **{g_valid}** | Rejected: **{g_judged - g_valid}**"
            + (f" ({100*g_valid/g_sent:.1f}% pass rate)" if g_sent else ""),
            "",
            "| Criterion failed | Count | % of sent |",
            "| --- | --- | --- |",
        ]
        for key, label in GPT_CRITERIA_LABELS.items():
            n = g_criteria.get(key, 0)
            if n:
                pct = f"{100*n/g_sent:.1f}%" if g_sent else "—"
                lines.append(f"| {label} | {n} | {pct} |")

    # ── Section 7: Per-File Summary ─────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 7. Per-File Summary",
        "",
        "| File ID | Timing imm | Timing lat | Timing ovl | GPT sent | GPT valid |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda x: x["file_id"]):
        if not r["success"]:
            lines.append(f"| {r['file_id']} | ERROR | — | — | — | — |")
            continue
        ts     = r["timing_stats"]
        gs     = r["gpt_stats"]
        bc     = ts.get("by_category", {})
        t_imm  = bc.get("take_turn_immediately",    0)
        t_lat  = bc.get("take_turn_with_latency",   0)
        t_ovl  = bc.get("take_turn_with_overlap",   0)
        g_s    = gs.get("total_sent", 0)
        g_v    = gs.get("valid",      0)
        lines.append(
            f"| {r['file_id']} | {t_imm} | {t_lat} | {t_ovl} | {g_s} | {g_v} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n── Report written → {report_path} ──")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global LOQUI_ROOT, LOQUI_TRANS_DIR
    parser = argparse.ArgumentParser(
        description="Batch Loqui turn-taking extraction.")
    parser.add_argument("--workers",              type=int,  default=4)
    parser.add_argument("--results_dir",          type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--report_path",          type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no_gpt",               action="store_true")
    parser.add_argument("--no_audio",             action="store_true")
    parser.add_argument("--force",                action="store_true")
    parser.add_argument("--loqui_root", type=Path, default=LOQUI_ROOT,
                        help="Loqui root (LoquiPackage, or the folder that contains it). Also LOQUI_ROOT.")
    args = parser.parse_args()
    LOQUI_ROOT = args.loqui_root.expanduser().resolve()
    LOQUI_TRANS_DIR = resolve_trans_dir(LOQUI_ROOT)

    entries = find_files()
    if not entries:
        print("No .trs/.san.wav pairs found.")
        return

    if not args.force:
        todo    = [e for e in entries if not is_done(e, args.results_dir)]
        skipped = len(entries) - len(todo)
        if skipped:
            print(f"Skipping {skipped} already-done files (use --force to redo).")
        entries = todo

    print(f"Processing {len(entries)} files with {args.workers} workers …")
    completed: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, e,
                        args.results_dir, args.no_gpt, args.no_audio, args.force): e
            for e in entries
        }
        for fut in as_completed(futures):
            res = fut.result()
            completed.append(res)
            if res["success"]:
                gs     = res["gpt_stats"]
                g_sent = gs.get("total_sent", 0)
                g_val  = gs.get("valid",      0)
                print(f"  [OK]  {res['file_id']:>3} ({res['elapsed']:.0f}s) "
                      f"sent={g_sent} valid={g_val}")
            else:
                print(f"  [ERR] {res['file_id']:>3} ({res['elapsed']:.0f}s) "
                      f"{res['stderr'][:120]}")

    _print_filter_report(completed)
    all_valid = aggregate_results(args.results_dir, completed)

    if not all_valid:
        for cat in CATEGORIES:
            for pat in ("sequential", "overlap"):
                p = args.results_dir / cat / f"{pat}.json"
                if p.exists():
                    all_valid.extend(json.loads(p.read_text()))
        if all_valid:
            print(f"  (reloaded {len(all_valid)} valid candidates from disk)")

    write_md_report(completed, all_valid, args.report_path, args.no_gpt)


if __name__ == "__main__":
    main()
