#!/usr/bin/env python3
"""
Batch processor for Loqui floor-holding + backchannel extraction.

Uses extract_floor_backchannel_loqui.py which supports per-criterion
--skip_* flags and threshold overrides.

Discovers all .final.trs / .san.wav pairs in the Loqui Transcriptions dir and
runs extract_floor_backchannel_loqui.py on each file in parallel, then
aggregates results and writes a markdown report.

Results layout:
  <results_dir>/
    valid.json                              ← all GPT-validated candidates
    <backchannel_type>/*.wav                           (--task backchannel)
    <floor_holding_label>/
      sequential.json                       ← sequential BC candidates for this label
      overlap.json                          ← overlap BC candidates for this label
      sequential/<backchannel_type>/*.wav   ← audio clips
      overlap/<backchannel_type>/*.wav

Usage:
  python batch_floor_backchannel_loqui.py
  python batch_floor_backchannel_loqui.py --workers 8 --force
  python batch_floor_backchannel_loqui.py --no_gpt
  python batch_floor_backchannel_loqui.py --task floor_holding --skip_t1_pauses
  python batch_floor_backchannel_loqui.py --results_dir /path --report_path /path/report.md
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
SCRIPT              = Path(__file__).resolve().parent / "extract_floor_backchannel_loqui.py"
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/Loqui/floor_backchannel")
DEFAULT_REPORT      = DEFAULT_RESULTS_DIR / "report.md"

FLOOR_LABELS     = ("continue_current", "continue_shorten", "continue_simplify", "none")
BC_SUBTYPES      = ("sequential", "overlap", "whoblock")
BC_TYPES         = ("acknowledgement", "understanding", "agreement",
                    "uncertainty", "disagreement", "none")
TARGET_PER_LABEL = 500

QUOTA_EXIT_CODE = 2

# ── Timing rejection labels (mirrors extract script's stats keys) ──────────────

TIMING_LABELS: dict[str, str] = {
    "windows_checked":   "Windows checked",
    "not_baba":            "Core T_before/T_bc/T_after pattern mismatch (or T0 speaker for floor_holding/both)",
    "before_ovl_b_short":  "[ovl T_before] B's portion ≤ MAX_BC_WORDS (T_before itself is a BC, ambiguous)",
    "t0_non_std":          "T0 non-standard / PII (floor_holding/both only)",
    "t0_few_words":      "T0 < min words (floor_holding/both only)",
    "t0_has_pause":      "T0 contains <P> (floor_holding/both only)",
    "before_non_std":    "T_before non-standard / PII",
    # Sequential BC
    "seq_bc_non_std":    "[seq] T_bc non-standard / PII",
    "seq_bc_empty":      "[seq] T_bc empty",
    "seq_bc_too_many":   "[seq] T_bc > 3 words",
    "seq_bc_has_pause":  "[seq] T_bc contains <P>",
    # Overlap BC
    "ovl_bc_not_b":      "[ovl] overlap has no B <Who> block",
    "ovl_bc_non_std":    "[ovl] B <Who> non-standard / PII",
    "ovl_bc_too_many":   "[ovl] B <Who> > 3 words",
    "ovl_bc_empty":      "[ovl] B <Who> empty",
    "ovl_bc_has_pause":  "[ovl] B <Who> contains <P>",
    "ovl_during_short":  "[ovl] A <Who> < 2 words",
    "ovl_during_pause":  "[ovl] A <Who> contains <P>",
    # Shared post-BC
    "t1_has_pause":      "T_before or T_after has internal <P>",
    "after_non_std":     "T_after non-standard / PII",
    "after_few_words":   "T_after < min words",
    "duplicate":         "Duplicate",
    "passed_sequential": "PASSED (sequential)",
    "passed_overlap":    "PASSED (overlap)",
    # Pattern C — Who-block split
    "who_bc_empty":      "[who] B Who block empty",
    "who_bc_too_many":   "[who] B Who block > 3 words",
    "who_bc_has_pause":  "[who] B Who block contains <P>",
    "who_bc_non_std":    "[who] B Who block non-standard",
    "who_before_short":  "[who] A's text before [os < min words",
    "who_t0_few_words":  "[who] T0 too few words (floor_holding)",
    "who_t0_has_pause":  "[who] T0 has <P> (floor_holding)",
    "who_t0_non_std":    "[who] T0 non-standard (floor_holding)",
    "who_t1_anon":       "[who] T_before or T_after non-standard",
    "who_t1_pause":      "[who] T_before or T_after internal <P>",
    "who_after_missing": "[who] no T_after within 2s",
    "who_after_non_std": "[who] T_after non-standard",
    "who_after_short":   "[who] T_after < min words",
    "passed_whoblock":   "PASSED (whoblock)",
    "passed":            "PASSED (total)",
}

_ALL_GPT_CRITERIA: dict[str, str] = {
    "turn0_complete_false":       "turn0_complete=false",
    "turn0_self_contained_false": "turn0_self_contained=false",
    "turn1_complete_false":       "turn1_complete=false",
    "turn1_self_contained_false": "turn1_self_contained=false",
}

def _gpt_criteria_labels(task: str) -> dict[str, str]:
    if task == "backchannel":
        keys = ["turn1_complete_false", "turn1_self_contained_false"]
    elif task == "floor_holding":
        keys = ["turn0_complete_false", "turn0_self_contained_false", "turn1_complete_false"]
    else:  # both
        keys = list(_ALL_GPT_CRITERIA)
    return {k: _ALL_GPT_CRITERIA[k] for k in keys}


# ── File discovery ─────────────────────────────────────────────────────────────

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
    return (results_dir / entry["file_id"] / "valid.json").exists()


# ── Per-file runner ────────────────────────────────────────────────────────────

def process_one(entry: dict, results_dir: Path,
                no_gpt: bool, no_audio: bool, force: bool,
                task: str = "backchannel",
                skip_flags: list | None = None) -> dict:
    cmd = [
        PYTHON, str(SCRIPT),
        "--file_id",             entry["file_id"],
        "--trs_path",            str(entry["trs_path"]),
        "--wav_path",            str(entry["wav_path"]),
        "--results_dir",         str(results_dir),
        "--max_valid_per_label", str(TARGET_PER_LABEL),
        "--task",                task,
    ]
    if no_gpt:
        cmd.append("--no_gpt")
    if no_audio:
        cmd.append("--no_audio")
    if force:
        cmd.append("--force")
    if skip_flags:
        cmd.extend(skip_flags)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    success = result.returncode == 0
    timing_stats:    dict       = {}
    gpt_stats:       dict       = {}
    valid_candidates: list[dict] = []
    send_count       = 0
    valid_count      = 0
    display_lines:   list[str]  = []

    for ln in result.stdout.splitlines():
        stripped = ln.strip()
        if stripped.startswith("SEND_TO_LLM floor_bc"):
            try:
                send_count = int(stripped.split(":")[-1].strip())
            except Exception:
                pass
        elif stripped.startswith("VALID_AFTER_LLM:"):
            try:
                valid_count = int(stripped.split(":")[-1].strip())
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
        "send_count":       send_count,
        "valid_count":      valid_count,
        "timing_stats":     timing_stats,
        "gpt_stats":        gpt_stats,
        "valid_candidates": valid_candidates,
        "stdout_lines":     display_lines,
        "stderr":           result.stderr[-500:] if not success else "",
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_results(results_dir: Path, completed: list[dict],
                      task: str = 'both') -> list[dict]:
    """Aggregate results with per-category cap.

    For task='backchannel': cap per backchannel_type; flat directory structure.
    For task='floor_holding'/'both': cap per floor_holding_label; nested structure.
    """
    all_valid: list[dict] = []
    for res in completed:
        all_valid.extend(res.get("valid_candidates", []))

    def _cap_key(c: dict) -> str:
        if task == 'backchannel':
            return c.get("backchannel_type", "none") or "none"
        return c.get("floor_holding_label", "none") or "none"

    counts: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        key = _cap_key(c)
        if counts[key] < TARGET_PER_LABEL:
            capped.append(c)
            counts[key] += 1
    all_valid = capped

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return []

    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "valid.json", "w") as f:
        json.dump(all_valid, f, indent=2)

    print(f"\n── Results aggregated → {results_dir} ──")

    if task == 'backchannel':
        # Flat: results_dir/<backchannel_type>/
        by_bc: dict[str, list] = defaultdict(list)
        for c in all_valid:
            by_bc[c.get("backchannel_type", "none") or "none"].append(c)
        print(f"  Total valid (cap={TARGET_PER_LABEL}/backchannel_type): {len(all_valid)}")
        for bt in BC_TYPES:
            if by_bc.get(bt):
                print(f"    {bt}: {len(by_bc[bt])}")
    else:
        # Nested: results_dir/<floor_holding_label>/<subtype>.json
        by_label_subtype: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for c in all_valid:
            lbl     = c.get("floor_holding_label", "none") or "none"
            subtype = c.get("bc_subtype",          "unknown") or "unknown"
            by_label_subtype[lbl][subtype].append(c)

        for lbl in FLOOR_LABELS:
            lbl_dir = results_dir / lbl
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for subtype in BC_SUBTYPES:
                cands = by_label_subtype.get(lbl, {}).get(subtype, [])
                with open(lbl_dir / f"{subtype}.json", "w") as f:
                    json.dump(cands, f, indent=2)

        print(f"  Total valid (cap={TARGET_PER_LABEL}/floor_holding_label): {len(all_valid)}")
        for lbl in FLOOR_LABELS:
            sub = by_label_subtype.get(lbl, {})
            n_seq = len(sub.get("sequential", []))
            n_ovl = len(sub.get("overlap",    []))
            if n_seq + n_ovl:
                print(f"    {lbl}: sequential={n_seq}, overlap={n_ovl}, total={n_seq+n_ovl}")

    return all_valid


# ── Console report ─────────────────────────────────────────────────────────────

def _print_filter_report(results: list[dict], task: str = "backchannel") -> None:
    t_agg: dict = {}
    g_total_sent  = 0
    g_valid       = 0
    g_judged      = 0
    g_criteria:   dict[str, int] = {}
    floor_counts: dict[str, int] = defaultdict(int)
    bc_counts:    dict[str, int] = defaultdict(int)

    for r in results:
        if not r["success"]:
            continue
        for key, v in r["timing_stats"].items():
            if isinstance(v, (int, float)):
                t_agg[key] = t_agg.get(key, 0) + v
        gs = r["gpt_stats"]
        g_total_sent += gs.get("total_sent",    0)
        g_valid      += gs.get("total_valid",   0)
        g_judged     += gs.get("total_judged",  0)
        for crit_key in _ALL_GPT_CRITERIA:
            if crit_key in gs:
                g_criteria[crit_key] = g_criteria.get(crit_key, 0) + gs[crit_key]
        for lbl, cnt in (gs.get("valid_by_label") or {}).items():
            floor_counts[lbl] += cnt
        for bt, cnt in (gs.get("backchannel_type") or {}).items():
            bc_counts[bt] += cnt

    total  = t_agg.get("windows_checked",   0)
    passed = t_agg.get("passed",            0)
    p_seq  = t_agg.get("passed_sequential", 0)
    p_ovl  = t_agg.get("passed_overlap",    0)
    p_who  = t_agg.get("passed_whoblock",   0)

    print(f"\n{'═'*65}")
    print("  FLOOR-BACKCHANNEL EXTRACTION — FILTER REPORT  (Loqui)")
    print(f"{'═'*65}")
    print(f"\n── Timing filter ──")
    print(f"  Windows checked : {total:>6}")
    for key, label in TIMING_LABELS.items():
        if key in ("windows_checked", "passed", "passed_sequential", "passed_overlap", "passed_whoblock"):
            continue
        n = t_agg.get(key, 0)
        if n:
            pct = f"{100*n/total:.1f}%" if total else "—"
            print(f"  Rejected: {label:<50} {n:>5}  ({pct})")
    p_who = t_agg.get("passed_whoblock", 0)
    if total:
        print(f"  PASSED timing (sequential) : {p_seq:>6}  ({100*p_seq/total:.1f}%)")
        print(f"  PASSED timing (overlap)    : {p_ovl:>6}  ({100*p_ovl/total:.1f}%)")
        print(f"  PASSED timing (whoblock)   : {p_who:>6}  ({100*p_who/total:.1f}%)")
        print(f"  PASSED timing (total)      : {passed:>6}  ({100*passed/total:.1f}%)")

    if g_total_sent:
        print(f"\n── GPT filter ──")
        print(f"  Sent={g_total_sent}, Judged={g_judged}, Valid={g_valid} "
              f"({100*g_valid/g_total_sent:.1f}%)")
        for crit_key, crit_label in _gpt_criteria_labels(task).items():
            n = g_criteria.get(crit_key, 0)
            if n:
                print(f"    ✗ {crit_label:<40} {n:>5}  "
                      f"({100*n/g_total_sent:.1f}%)")
        if task in ("floor_holding", "both") and floor_counts:
            print(f"\n── floor_holding_label breakdown (valid) ──")
            for lbl in FLOOR_LABELS:
                print(f"    {lbl}: {floor_counts.get(lbl, 0)}")
        if task in ("backchannel", "both") and bc_counts:
            print(f"\n── backchannel_type breakdown (valid) ──")
            for bt in BC_TYPES:
                print(f"    {bt}: {bc_counts.get(bt, 0)}")

    print(f"\n{'═'*65}\n")


# ── Markdown report ────────────────────────────────────────────────────────────

def write_md_report(results: list[dict], all_valid: list[dict],
                    report_path: Path, no_gpt: bool,
                    task: str = "backchannel") -> None:
    t_agg: dict = {}
    g_total_sent = 0
    g_valid      = 0
    g_judged     = 0
    g_criteria:  dict[str, int] = {}
    n_files_ok   = 0

    for r in results:
        if not r["success"]:
            continue
        n_files_ok += 1
        for key, v in r["timing_stats"].items():
            if isinstance(v, (int, float)):
                t_agg[key] = t_agg.get(key, 0) + v
        gs = r["gpt_stats"]
        g_total_sent += gs.get("total_sent",   0)
        g_valid      += gs.get("total_valid",  0)
        g_judged     += gs.get("total_judged", 0)
        for crit_key in _ALL_GPT_CRITERIA:
            if crit_key in gs:
                g_criteria[crit_key] = g_criteria.get(crit_key, 0) + gs[crit_key]

    total  = t_agg.get("windows_checked", 0)
    passed = t_agg.get("passed", 0)
    p_seq  = t_agg.get("passed_sequential", 0)
    p_ovl  = t_agg.get("passed_overlap",    0)
    p_who  = t_agg.get("passed_whoblock",   0)

    # Fallback: reload from per-label/subtype JSON files if in-memory is empty
    if not all_valid:
        all_valid = []
        for lbl in FLOOR_LABELS:
            for subtype in BC_SUBTYPES:
                subtype_path = report_path.parent / lbl / f"{subtype}.json"
                if subtype_path.exists():
                    all_valid.extend(json.loads(subtype_path.read_text()))
        if all_valid:
            print(f"  (reloaded {len(all_valid)} valid candidates from disk for report)")

    total_valid = len(all_valid)

    by_label:          dict[str, list]  = defaultdict(list)
    by_subtype:        dict[str, list]  = defaultdict(list)
    by_label_sub_bc:   dict[str, dict[str, dict[str, int]]] = {
        lbl: {st: defaultdict(int) for st in BC_SUBTYPES} for lbl in FLOOR_LABELS
    }

    for c in all_valid:
        lbl     = c.get("floor_holding_label", "none")    or "none"
        subtype = c.get("bc_subtype",          "unknown") or "unknown"
        bt      = c.get("backchannel_type",    "none")    or "none"
        by_label[lbl].append(c)
        by_subtype[subtype].append(c)
        if lbl in by_label_sub_bc and subtype in by_label_sub_bc[lbl]:
            by_label_sub_bc[lbl][subtype][bt] += 1

    lines = [
        "# Loqui Floor-Holding + Backchannel Extraction Report",
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
        f"| Windows examined (T0 → T_before → BC → T_after) | {total} |",
        f"| Timing-filter passed — sequential | {p_seq} |",
        f"| Timing-filter passed — overlap | {p_ovl} |",
        f"| Timing-filter passed — whoblock | {p_who} |",
        f"| Timing-filter passed — **total** | {passed} |",
        f"| Sent to GPT | {g_total_sent} |",
        f"| GPT-validated (final) | {total_valid} |",
    ]
    if g_total_sent:
        rate = f"{100*total_valid/g_total_sent:.1f}%"
        lines.append(f"| GPT acceptance rate | {rate} |")

    lines += [
        "",
        "---",
        "",
        "## 2. BC Encoding Subtype Breakdown",
        "",
        "Sequential — B speaks in a short turn after A briefly pauses.  ",
        "Overlap — B speaks simultaneously with A in a multi-speaker overlap turn.  ",
        "",
        "| BC subtype | Count | % of valid |",
        "| --- | --- | --- |",
    ]
    for st in BC_SUBTYPES:
        n   = len(by_subtype.get(st, []))
        pct = f"{100*n/total_valid:.1f}%" if total_valid else "—"
        lines.append(f"| {st} | {n} | {pct} |")
    lines.append(f"| **Total** | **{total_valid}** | |")

    by_bc_seq: dict[str, int] = Counter(
        c.get("backchannel_type", "none") or "none"
        for c in by_subtype.get("sequential", []))
    by_bc_ovl: dict[str, int] = Counter(
        c.get("backchannel_type", "none") or "none"
        for c in by_subtype.get("overlap", []))
    by_bc_all: dict[str, int] = Counter(
        c.get("backchannel_type", "none") or "none" for c in all_valid)

    if task in ("floor_holding", "both"):
        lines += [
            "",
            "---",
            "",
            "## 3. Floor-Holding Label Breakdown",
            "",
            "| Label | Sequential | Overlap | Total | % of valid |",
            "| --- | --- | --- | --- | --- |",
        ]
        for lbl in FLOOR_LABELS:
            cands = by_label.get(lbl, [])
            n_seq = sum(1 for c in cands if (c.get("bc_subtype") or "") == "sequential")
            n_ovl = sum(1 for c in cands if (c.get("bc_subtype") or "") == "overlap")
            n_tot = len(cands)
            pct   = f"{100*n_tot/total_valid:.1f}%" if total_valid else "—"
            lines.append(f"| {lbl} | {n_seq} | {n_ovl} | {n_tot} | {pct} |")
        lines.append(f"| **Total** | **{len(by_subtype.get('sequential',[]))}** "
                     f"| **{len(by_subtype.get('overlap',[]))}** "
                     f"| **{total_valid}** | |")

    if task in ("backchannel", "both"):
        sec_num = 4 if task == "both" else 3
        lines += [
            "",
            "---",
            "",
            f"## {sec_num}. Backchannel Type Breakdown",
            "",
            "| Backchannel type | Sequential | Overlap | Total | % of valid |",
            "| --- | --- | --- | --- | --- |",
        ]
        for bt in BC_TYPES:
            n_s = by_bc_seq.get(bt, 0)
            n_o = by_bc_ovl.get(bt, 0)
            n_t = by_bc_all.get(bt, 0)
            pct = f"{100*n_t/total_valid:.1f}%" if total_valid else "—"
            lines.append(f"| {bt} | {n_s} | {n_o} | {n_t} | {pct} |")

    _s = 1 if task == "both" else 0

    lines += [
        "",
        "---",
        "",
        f"## {5 + _s}. Timing Filter Details",
        "",
        f"Windows examined: **{total}** → sequential: **{p_seq}**, "
        f"overlap: **{p_ovl}**, whoblock: **{p_who}**, total: **{passed}**"
        + (f" ({100*passed/total:.1f}%)" if total else ""),
        "",
        "| Rejection reason | Count | % of windows |",
        "| --- | --- | --- |",
    ]
    for key, label in TIMING_LABELS.items():
        if key in ("windows_checked", "passed", "passed_sequential", "passed_overlap", "passed_whoblock"):
            continue
        n = t_agg.get(key, 0)
        if n:
            pct = f"{100*n/total:.1f}%" if total else "—"
            lines.append(f"| {label} | {n} | {pct} |")

    if not no_gpt and g_total_sent:
        lines += [
            "",
            "---",
            "",
            f"## {6 + _s}. GPT Filter Details",
            "",
            f"Sent: **{g_total_sent}** | Judged: **{g_judged}** | "
            f"Valid: **{g_valid}** | Rejected: **{g_judged - g_valid}**"
            + (f" (pass rate {100*g_valid/g_total_sent:.1f}%)" if g_total_sent else ""),
            "",
            "| Criterion failed | Count | % of sent |",
            "| --- | --- | --- |",
        ]
        for key, label in _gpt_criteria_labels(task).items():
            n = g_criteria.get(key, 0)
            if n:
                pct = f"{100*n/g_total_sent:.1f}%" if g_total_sent else "—"
                lines.append(f"| {label} | {n} | {pct} |")

    lines += [
        "",
        "---",
        "",
        f"## {7 + _s}. BC Words Distribution",
        "",
        "| BC word count | Sequential | Overlap | Total |",
        "| --- | --- | --- | --- |",
    ]
    wdist_seq: dict[int,int] = defaultdict(int)
    wdist_ovl: dict[int,int] = defaultdict(int)
    for c in all_valid:
        w = c.get("bc_words", 0)
        if (c.get("bc_subtype") or "") == "overlap":
            wdist_ovl[w] += 1
        else:
            wdist_seq[w] += 1
    for w in sorted(set(wdist_seq) | set(wdist_ovl)):
        s, o = wdist_seq.get(w,0), wdist_ovl.get(w,0)
        lines.append(f"| {w} | {s} | {o} | {s+o} |")

    lines += [
        "",
        "---",
        "",
        f"## {8 + _s}. Per-File Summary",
        "",
        "| File ID | Timing seq | Timing ovl | GPT sent | GPT valid |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda x: x["file_id"]):
        if not r["success"]:
            lines.append(f"| {r['file_id']} | ERROR | — | — | — |")
            continue
        ts     = r["timing_stats"]
        gs     = r["gpt_stats"]
        t_seq  = ts.get("passed_sequential", 0)
        t_ovl  = ts.get("passed_overlap",    0)
        g_sent = gs.get("total_sent",  0)
        g_val  = gs.get("total_valid", 0)
        lines.append(f"| {r['file_id']} | {t_seq} | {t_ovl} | {g_sent} | {g_val} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n── Report written → {report_path} ──")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global LOQUI_ROOT, LOQUI_TRANS_DIR
    parser = argparse.ArgumentParser(
        description="Batch Loqui floor-holding + backchannel extraction.")
    parser.add_argument("--workers",     type=int,  default=4)
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Results directory (default: auto-suffixed by --task).")
    parser.add_argument("--report_path", type=Path, default=None,
                        help="Report path (default: results_dir/report.md).")
    parser.add_argument("--no_gpt",      action="store_true")
    parser.add_argument("--no_audio",    action="store_true")
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--task", choices=["backchannel", "floor_holding", "both"],
                        default="backchannel",
                        help="GPT evaluation task passed through to extract script.")

    # ── Timing skip flags ──────────────────────────────────────────────────────
    parser.add_argument("--skip_t0",          action="store_true")
    parser.add_argument("--skip_t0_words",    action="store_true")
    parser.add_argument("--skip_t0_pauses",           action="store_true")
    parser.add_argument("--t0_pause_uppercase_only",  action="store_true",
                        help="Tolerate <p> in T0; only <P> (uppercase) rejects.")
    parser.add_argument("--skip_t1_pauses",         action="store_true")
    parser.add_argument("--t1_pause_uppercase_only", action="store_true",
                        help="Tolerate <p> in T1; only <P> (uppercase) rejects.")
    parser.add_argument("--skip_t1_anon",            action="store_true")
    parser.add_argument("--skip_after_words",       action="store_true")
    parser.add_argument("--allow_t_before_overlap", action="store_true")

    # ── Threshold overrides ────────────────────────────────────────────────────
    parser.add_argument("--min_t0_words",    type=int, default=None)
    parser.add_argument("--min_after_words", type=int, default=None)

    # ── GPT skip flags ─────────────────────────────────────────────────────────
    parser.add_argument("--skip_gpt_t0_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t0_self_contained", action="store_true")
    parser.add_argument("--skip_gpt_t1_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t1_self_contained", action="store_true")

    parser.add_argument("--loqui_root", type=Path, default=LOQUI_ROOT,
                        help="Loqui root (LoquiPackage, or the folder that contains it). Also LOQUI_ROOT.")
    args = parser.parse_args()
    LOQUI_ROOT = args.loqui_root.expanduser().resolve()
    LOQUI_TRANS_DIR = resolve_trans_dir(LOQUI_ROOT)

    # Auto-suffix results dir and report path based on task
    if args.results_dir is None:
        suffix = "" if args.task == "backchannel" else f"_{args.task}"
        args.results_dir = Path(str(DEFAULT_RESULTS_DIR) + suffix)
    if args.report_path is None:
        args.report_path = args.results_dir / "report.md"

    # Build skip_flags list to forward to each extractor subprocess
    skip_flags: list[str] = []
    if args.skip_t0:          skip_flags.append("--skip_t0")
    if args.skip_t0_words:    skip_flags.append("--skip_t0_words")
    if args.skip_t0_pauses:           skip_flags.append("--skip_t0_pauses")
    if args.t0_pause_uppercase_only:  skip_flags.append("--t0_pause_uppercase_only")
    if args.skip_t1_pauses:          skip_flags.append("--skip_t1_pauses")
    if args.t1_pause_uppercase_only: skip_flags.append("--t1_pause_uppercase_only")
    if args.skip_t1_anon:            skip_flags.append("--skip_t1_anon")
    if args.skip_after_words:      skip_flags.append("--skip_after_words")
    if args.allow_t_before_overlap: skip_flags.append("--allow_t_before_overlap")
    if args.min_t0_words    is not None: skip_flags += ["--min_t0_words",    str(args.min_t0_words)]
    if args.min_after_words is not None: skip_flags += ["--min_after_words", str(args.min_after_words)]
    if args.skip_gpt_t0_complete:       skip_flags.append("--skip_gpt_t0_complete")
    if args.skip_gpt_t0_self_contained: skip_flags.append("--skip_gpt_t0_self_contained")
    if args.skip_gpt_t1_complete:       skip_flags.append("--skip_gpt_t1_complete")
    if args.skip_gpt_t1_self_contained: skip_flags.append("--skip_gpt_t1_self_contained")

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
    n_done  = 0
    n_total = len(entries)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, e,
                        args.results_dir, args.no_gpt, args.no_audio,
                        args.force, args.task, skip_flags): e
            for e in entries
        }
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if res["success"]:
                send  = res["send_count"]
                valid = res["valid_count"]
                print(f"  [OK]  {res['file_id']:>3} ({res['elapsed']:.0f}s) "
                      f"send={send} valid={valid}")
            else:
                print(f"  [ERR] {res['file_id']:>3} ({res['elapsed']:.0f}s) "
                      f"rc≠0  {res['stderr'][:120]}")
            completed.append(res)

    _print_filter_report(completed, task=args.task)
    all_valid = aggregate_results(args.results_dir, completed, task=args.task)

    if not all_valid:
        all_valid = []
        for lbl in FLOOR_LABELS:
            lbl_path = args.results_dir / f"{lbl}.json"
            if lbl_path.exists():
                all_valid.extend(json.loads(lbl_path.read_text()))
        if all_valid:
            print(f"  (reloaded {len(all_valid)} valid candidates from disk for report)")

    write_md_report(completed, all_valid, args.report_path, args.no_gpt, task=args.task)


if __name__ == "__main__":
    main()
