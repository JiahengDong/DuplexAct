#!/usr/bin/env python3
"""
Batch processor for Loqui pause candidate extraction.

Discovers all .final.trs / .san.wav pairs and runs extract_pause_loqui.py
on each file in parallel, then aggregates results and writes a markdown report.

Usage:
  python batch_pause_loqui.py
  python batch_pause_loqui.py --workers 8 --force
  python batch_pause_loqui.py --no_gpt
  python batch_pause_loqui.py --results_dir /path --report_path /path/report.md
"""

import argparse
import os
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

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
SCRIPT              = Path(__file__).resolve().parent / "extract_pause_loqui.py"
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/pauses_loqui")
DEFAULT_REPORT      = DEFAULT_RESULTS_DIR / "report.md"

PAUSE_TYPES     = ("inter_sentential", "intra_sentential", "floor_holding")
TARGET_PER_TYPE = 500

# ── Labels for console / markdown report ─────────────────────────────────────

# Timing stats from extract_pause_loqui.py use "pause" and "floor_holding" keys.
# GPT splits "pause" into inter_sentential / intra_sentential post-hoc.
# TIMING_LABELS maps the timing key → rejection label dict.
TIMING_LABELS: dict[str, dict[str, str]] = {
    "pause": {
        "rej_non_standard": "Non-standard / PII speech in turn",
        "rej_no_pauses":    "No <P> marker found",
        "rej_too_short":    "Too short (< 8 words total)",
        "rej_before_short": "Words before first <P> < 4",
        "rej_after_short":  "Words after last  <P> < 4",
    },
    "floor_holding": {
        "rej_non_standard":   "Non-standard / PII speech in turn",
        "rej_has_pauses":     "Contains <P> marker",
        "rej_too_short":      "Too short (< 8 words)",
        "rej_duration_short": "Duration < 3.0 s",
    },
}
# Keys used for timing aggregation (extract script output)
TIMING_KEYS = ("pause", "floor_holding")

GPT_CRITERIA_LABELS: dict[str, dict[str, str]] = {
    "inter_sentential": {
        "complete_utterance":  "complete_utterance=false",
        "natural_pause":       "natural_pause=false",
        "self_contained":      "self_contained=false",
        "pause_type_mixed":    "pause_type=mixed (inconsistent inter/intra)",
    },
    "intra_sentential": {},   # same GPT call; results split by pause_type
    "floor_holding": {
        "is_complete":    "is_complete=false",
        "is_coherent":    "is_coherent=false",
        "self_contained": "self_contained=false",
    },
}


# ── Discovery ─────────────────────────────────────────────────────────────────

def find_files() -> list[dict]:
    entries = []
    for trs in sorted(LOQUI_TRANS_DIR.glob("*.final.trs")):
        file_id = trs.stem.replace(".final", "")
        wav = trs.with_suffix("").with_suffix(".san.wav")
        if not wav.exists():
            wav = trs.with_name(f"{file_id}.san.wav")
        if not wav.exists():
            print(f"  [warn] no .san.wav for {trs.name}")
            continue
        entries.append({"file_id": file_id, "trs_path": trs, "wav_path": wav})
    return entries


def is_done(entry: dict, results_dir: Path) -> bool:
    """A file is done if its valid.json exists (written at the end of processing)."""
    return (results_dir / entry["file_id"] / "valid.json").exists()


# ── Per-file runner ───────────────────────────────────────────────────────────

def process_one(entry: dict, results_dir: Path,
                no_gpt: bool, no_audio: bool, force: bool) -> dict:
    cmd = [
        PYTHON, str(SCRIPT),
        "--file_id",            entry["file_id"],
        "--trs_path",           str(entry["trs_path"]),
        "--wav_path",           str(entry["wav_path"]),
        "--results_dir",        str(results_dir),
        "--max_valid_per_type", str(TARGET_PER_TYPE),
    ]
    if no_gpt:
        cmd.append("--no_gpt")
    if no_audio:
        cmd.append("--no_audio")
    if force:
        cmd.append("--force")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    success = result.returncode == 0

    # timing only produces 'pause' and 'floor_holding' keys; GPT splits pause → inter/intra
    send         = {"pause": 0, "floor_holding": 0}
    valid_counts = {pt: 0 for pt in PAUSE_TYPES}
    timing_stats: dict = {}
    gpt_stats:    dict = {}
    valid_candidates: list[dict] = []
    display_lines: list[str] = []

    for ln in result.stdout.splitlines():
        stripped = ln.strip()
        if stripped.startswith("SEND_TO_LLM "):
            parts = stripped.split()
            try:
                pt = parts[1].rstrip(":")
                n  = int(parts[2])
                if pt in send:
                    send[pt] = n
            except Exception:
                pass
        elif stripped.startswith("VALID_AFTER_LLM "):
            parts = stripped.split()
            try:
                pt = parts[1].rstrip(":")
                n  = int(parts[2])
                if pt in valid_counts:
                    valid_counts[pt] = n
            except Exception:
                pass
        elif ln.startswith("TIMING_STATS:"):
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
            display_lines.append(ln)

    return {
        "file_id":          entry["file_id"],
        "success":          success,
        "elapsed":          elapsed,
        "send":             send,
        "valid_counts":     valid_counts,
        "timing_stats":     timing_stats,
        "gpt_stats":        gpt_stats,
        "valid_candidates": valid_candidates,
        "stdout_lines":     display_lines,
        "stderr":           result.stderr[-500:] if not success else "",
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(results_dir: Path, completed: list[dict]) -> list[dict]:
    all_valid: list[dict] = []
    for res in completed:
        all_valid.extend(res.get("valid_candidates", []))

    counts: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        pt = c.get("type", "unknown")
        if counts[pt] < TARGET_PER_TYPE:
            capped.append(c)
            counts[pt] += 1
    all_valid = capped

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return []

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "valid.json", "w") as f:
        json.dump(all_valid, f, indent=2)

    # Group by effective type: pause candidates use pause_type (inter/intra),
    # floor_holding candidates stay as floor_holding.
    by_type: dict[str, list] = defaultdict(list)
    for c in all_valid:
        if c.get("type") == "floor_holding":
            by_type["floor_holding"].append(c)
        else:
            pt = c.get("pause_type", "unknown")
            by_type[pt].append(c)

    for pt in PAUSE_TYPES:
        with open(results_dir / f"{pt}.json", "w") as f:
            json.dump(by_type.get(pt, []), f, indent=2)

    print(f"\n── Results aggregated → {results_dir} ──")
    print(f"  Total valid (cap={TARGET_PER_TYPE}/type): {len(all_valid)}")
    for pt in PAUSE_TYPES:
        print(f"    {pt}: {len(by_type.get(pt, []))}")

    return all_valid


# ── Console report ────────────────────────────────────────────────────────────

def _print_filter_report(results: list[dict]) -> None:
    # Timing stats keyed by "pause" and "floor_holding"
    t_agg: dict[str, dict] = {k: {} for k in TIMING_KEYS}
    # GPT stats keyed by "pause" and "floor_holding"
    g_agg: dict[str, dict] = {k: {"total_sent": 0, "valid": 0,
                                   "rejected": 0, "criteria": {}}
                               for k in TIMING_KEYS}

    for r in results:
        if not r["success"]:
            continue
        for k in TIMING_KEYS:
            ts = r["timing_stats"].get(k, {})
            for key, v in ts.items():
                t_agg[k][key] = t_agg[k].get(key, 0) + v
            gs = r["gpt_stats"].get(k, {})
            for stat in ("total_sent", "valid", "rejected"):
                g_agg[k][stat] += gs.get(stat, 0)
            for crit, cnt in (gs.get("criteria") or {}).items():
                g_agg[k]["criteria"][crit] = g_agg[k]["criteria"].get(crit, 0) + cnt

    print(f"\n{'═'*70}")
    print("  PAUSE EXTRACTION — FILTER REPORT  (Loqui)")
    print(f"{'═'*70}")

    for k in TIMING_KEYS:
        ta     = t_agg[k]
        ga     = g_agg[k]
        total  = ta.get("total_turns", 0)
        passed = ta.get("passed", 0)
        labels = TIMING_LABELS.get(k, {})

        print(f"\n── {k} ──")
        print(f"  Total turns examined       : {total:>6}")
        for key, label in labels.items():
            n = ta.get(key, 0)
            if n:
                pct = f"{100*n/total:.1f}%" if total else "—"
                print(f"  Rejected: {label:<50} {n:>5}  ({pct})")
        if total:
            print(f"  PASSED timing filter       : {passed:>6}  ({100*passed/total:.1f}%)")

        sent  = ga["total_sent"]
        valid = ga["valid"]
        if sent:
            print(f"  GPT: sent={sent}, valid={valid} ({100*valid/sent:.1f}%)")
            for crit, cnt in sorted(ga["criteria"].items(), key=lambda x: -x[1]):
                lbl = GPT_CRITERIA_LABELS.get(k, {}).get(crit, crit)
                print(f"    ✗ {lbl:<45} {cnt:>5}  ({100*cnt/sent:.1f}%)")

    # Post-GPT breakdown: pause → inter / intra
    inter = sum(r.get("valid_counts", {}).get("inter_sentential", 0) for r in results if r["success"])
    intra = sum(r.get("valid_counts", {}).get("intra_sentential", 0) for r in results if r["success"])
    floor = sum(r.get("valid_counts", {}).get("floor_holding",    0) for r in results if r["success"])
    if inter + intra + floor:
        print(f"\n── Post-GPT type breakdown ──")
        print(f"  inter_sentential : {inter}")
        print(f"  intra_sentential : {intra}")
        print(f"  floor_holding    : {floor}")

    print(f"\n{'═'*70}\n")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_md_report(results: list[dict], all_valid: list[dict],
                    report_path: Path, no_gpt: bool) -> None:
    t_agg: dict[str, dict] = {k: {} for k in TIMING_KEYS}
    g_agg: dict[str, dict] = {k: {"total_sent": 0, "valid": 0,
                                   "rejected": 0, "criteria": {}}
                               for k in TIMING_KEYS}

    n_files_ok = 0
    for r in results:
        if not r["success"]:
            continue
        n_files_ok += 1
        for k in TIMING_KEYS:
            ts = r["timing_stats"].get(k, {})
            for key, v in ts.items():
                t_agg[k][key] = t_agg[k].get(key, 0) + v
            gs = r["gpt_stats"].get(k, {})
            for stat in ("total_sent", "valid", "rejected"):
                g_agg[k][stat] += gs.get(stat, 0)
            for crit, cnt in (gs.get("criteria") or {}).items():
                g_agg[k]["criteria"][crit] = g_agg[k]["criteria"].get(crit, 0) + cnt

    by_type: dict[str, list] = defaultdict(list)
    for c in all_valid:
        # pause_type is the GPT-assigned final category (inter/intra/floor_holding);
        # fall back to type (timing category) when pause_type is absent.
        final_type = c.get("pause_type") or c.get("type", "unknown")
        by_type[final_type].append(c)

    # Timing filter uses "pause" (→ inter/intra split by GPT) + "floor_holding".
    # Build combined "pause" stats for the summary table rows.
    pause_ta = t_agg.get("pause", {})
    pause_ga = g_agg.get("pause", {"total_sent": 0, "valid": 0, "rejected": 0})
    floor_ta = t_agg.get("floor_holding", {})
    floor_ga = g_agg.get("floor_holding", {"total_sent": 0, "valid": 0, "rejected": 0})

    pause_passed = pause_ta.get("passed", 0)
    pause_sent   = pause_ga["total_sent"]
    pause_valid  = pause_ga["valid"]   # combined inter+intra GPT valid

    floor_passed = floor_ta.get("passed", 0)
    floor_sent   = floor_ga["total_sent"]
    floor_valid  = floor_ga["valid"]

    inter_final = len(by_type.get("inter_sentential", []))
    intra_final = len(by_type.get("intra_sentential", []))
    floor_final = len(by_type.get("floor_holding", []))

    lines = [
        f"# Loqui Pause Extraction Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Files processed: {n_files_ok} / {len(results)}  ",
        f"GPT filter: {'**disabled** (--no_gpt)' if no_gpt else 'enabled'}  ",
        f"",
        f"---",
        f"",
        f"## 1. Overall Summary",
        f"",
        f"> Note: timing filter produces **pause** (has `<P>`) and **floor_holding** candidates."
        f" GPT then splits *pause* into *inter_sentential* / *intra_sentential*.",
        f"",
        f"| Pause type | Timing passed | GPT sent | GPT valid | Final |",
        f"|------------|:-------------:|:--------:|:---------:|:-----:|",
        f"| inter_sentential *(from pause)* | {pause_passed} | {pause_sent} | {inter_final} | **{inter_final}** |",
        f"| intra_sentential *(from pause)* | {pause_passed} | {pause_sent} | {intra_final} | **{intra_final}** |",
        f"| floor_holding | {floor_passed} | {floor_sent} | {floor_valid} | **{floor_final}** |",
    ]

    lines += [
        f"",
        f"---",
        f"",
        f"## 2. Timing Filter Details",
        f"",
    ]

    for idx, k in enumerate(TIMING_KEYS):
        ta     = t_agg[k]
        total  = ta.get("total_turns", 0)
        passed = ta.get("passed", 0)
        labels = TIMING_LABELS.get(k, {})

        lines += [
            f"### 2.{idx+1} {k}",
            f"",
            f"Total turns: **{total}** → passed: **{passed}**"
            + (f" ({100*passed/total:.1f}%)" if total else ""),
            f"",
            f"| Rejection reason | Count | % of total |",
            f"|-----------------|------:|----------:|",
        ]
        for key, label in labels.items():
            n = ta.get(key, 0)
            if n:
                pct = f"{100*n/total:.1f}%" if total else "—"
                lines.append(f"| {label} | {n} | {pct} |")
        lines.append("")

    if not no_gpt:
        lines += [
            f"---",
            f"",
            f"## 3. GPT Filter Details",
            f"",
        ]
        for idx, k in enumerate(TIMING_KEYS):
            ga   = g_agg[k]
            sent = ga["total_sent"]
            valid = ga["valid"]
            rej  = ga["rejected"]
            lines += [
                f"### 3.{idx+1} {k}",
                f"",
                f"Sent: **{sent}** | Valid: **{valid}** | Rejected: **{rej}**"
                + (f" (pass rate {100*valid/sent:.1f}%)" if sent else ""),
                f"",
            ]
            if ga["criteria"]:
                lines += [
                    f"| Criterion failed | Count | % of sent |",
                    f"|-----------------|------:|----------:|",
                ]
                for crit, cnt in sorted(ga["criteria"].items(), key=lambda x: -x[1]):
                    lbl = GPT_CRITERIA_LABELS.get(k, {}).get(crit, crit)
                    pct = f"{100*cnt/sent:.1f}%" if sent else "—"
                    lines.append(f"| {lbl} | {cnt} | {pct} |")
            lines.append("")

    lines += [
        f"---",
        f"",
        f"## 4. Final Valid Candidates",
        f"",
        f"| Pause type | Count |",
        f"|------------|------:|",
    ]
    total_valid = 0
    for pt in PAUSE_TYPES:
        n = len(by_type.get(pt, []))
        total_valid += n
        lines.append(f"| {pt} | {n} |")
    lines.append(f"| **Total** | **{total_valid}** |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n── Report written → {report_path} ──")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global LOQUI_ROOT, LOQUI_TRANS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--no_gpt",      action="store_true")
    parser.add_argument("--no_audio",    action="store_true")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--loqui_root", type=Path, default=LOQUI_ROOT,
                        help="Loqui root (LoquiPackage, or the folder that contains it). Also LOQUI_ROOT.")
    args = parser.parse_args()
    LOQUI_ROOT = args.loqui_root.expanduser().resolve()
    LOQUI_TRANS_DIR = resolve_trans_dir(LOQUI_ROOT)

    entries = find_files()
    print(f"Found {len(entries)} Loqui file pairs.")

    if not args.force:
        pending = [e for e in entries if not is_done(e, args.results_dir)]
        skipped = len(entries) - len(pending)
        if skipped:
            print(f"  Skipping {skipped} already-done files (--force to redo).")
    else:
        pending = entries

    if args.dry_run:
        for e in pending:
            print(f"  [dry_run] {e['file_id']}")
        return

    if not pending:
        print("All files already processed.")
        # Still produce the report from existing data
        all_valid = _load_existing_valid(args.results_dir)
        write_md_report([], all_valid, args.report_path, args.no_gpt)
        return

    print(f"Processing {len(pending)} files with {args.workers} workers...\n")

    completed: list[dict] = []
    kwargs = dict(results_dir=args.results_dir,
                  no_gpt=args.no_gpt, no_audio=args.no_audio, force=args.force)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, e, **kwargs): e for e in pending}
        for fut in as_completed(futures):
            res = fut.result()
            completed.append(res)
            status = "OK" if res["success"] else "FAIL"
            send_str = " | ".join(f"{pt[:5]}={res['send'].get(pt, 0)}" for pt in ("pause", "floor_holding"))
            print(f"  [{status}] {res['file_id']:>4} ({res['elapsed']:.0f}s) "
                  f"send=[{send_str}]")
            if not res["success"] and res["stderr"]:
                print(f"         STDERR: {res['stderr'][:200]}")
            for ln in res["stdout_lines"]:
                if any(k in ln for k in ("WARN", "Error", "error")):
                    print(f"         {ln}")

    n_ok   = sum(1 for r in completed if r["success"])
    n_fail = len(completed) - n_ok
    print(f"\nCompleted: {n_ok} OK, {n_fail} failed")

    _print_filter_report(completed)
    all_valid = aggregate_results(args.results_dir, completed)

    # Fallback: if in-memory aggregation is empty (VALID_JSON too large to parse
    # from subprocess stdout), reload from the per-type JSON files written to disk.
    if not all_valid:
        all_valid = []
        for pt in PAUSE_TYPES:
            pt_path = args.results_dir / f"{pt}.json"
            if pt_path.exists():
                all_valid.extend(json.loads(pt_path.read_text()))
        if all_valid:
            print(f"  (reloaded {len(all_valid)} valid candidates from disk for report)")

    write_md_report(completed, all_valid, args.report_path, args.no_gpt)


def _load_existing_valid(results_dir: Path) -> list[dict]:
    vp = results_dir / "valid.json"
    if vp.exists():
        return json.loads(vp.read_text())
    return []


if __name__ == "__main__":
    main()
