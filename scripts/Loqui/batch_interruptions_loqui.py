#!/usr/bin/env python3
"""
Batch processor for Loqui interruption extraction.

Discovers all .final.trs / .san.wav pairs in the Loqui Transcriptions folder
and runs extract_interruptions_loqui.py on each file using parallel
workers.

Every timing / GPT criterion can be individually toggled off via --skip_* flags,
mirroring the extractor's FilterConfig.

Results are aggregated into:
  <results_dir>/
    valid.json
    hard/<recovery_type>/<file_id>_ov*_{t1,t2,t3,t4,full_stereo}.wav
    report.md

Usage:
  python batch_interruptions_loqui.py --workers 8 --no_gpt
  python batch_interruptions_loqui.py --skip_gpt_t1_self_contained \\
      --results_dir .../interruptions_loqui_no_t1sc
"""

QUOTA_EXIT_CODE = 2

import argparse
import os
import json
import math
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
SCRIPT              = Path(__file__).resolve().parent / "extract_interruptions_loqui.py"
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results"
                           "/interruptions_loqui")

RECOVERY_TYPES = ["correct", "enrich", "shorten", "repeat",
                  "shift_topic", "ask_followup", "none"]
TARGET_PER_TYPE = 500

# ── Timing skip flags forwarded to the extractor ──────────────────────────────
TIMING_BOOL_FLAGS = [
    "skip_t2_overlap_t1", "skip_t3_overlap", "skip_t4_overlap_t3",
    "skip_t1_words", "skip_t1_pauses", "t1_allow_lowercase_pause",
    "skip_t2_question", "skip_t2_words", "skip_t2_pre_words",
    "skip_t3_pauses", "t3_allow_lowercase_pause",
    "skip_t3_duration", "skip_t4_words", "skip_non_standard",
]
TIMING_LABELS = {
    "skip_t2_overlap_t1":        "Skip T2/T1 no-overlap check",
    "skip_t3_overlap":           "Skip T3 overlap check",
    "skip_t4_overlap_t3":        "Skip T4/T3 no-overlap check",
    "skip_t1_words":             "Skip T1 min-words check",
    "skip_t1_pauses":            "Skip T1 internal-pauses check",
    "t1_allow_lowercase_pause":  "T1 pause check: uppercase <P> only (allow <p>)",
    "skip_t2_question":          "Skip T2 not-question check",
    "skip_t2_words":             "Skip T2 min-words check",
    "skip_t2_pre_words":         "Skip T2 pre-overlap word check",
    "skip_t3_pauses":            "Skip T3 internal-pauses check",
    "t3_allow_lowercase_pause":  "T3 pause check: uppercase <P> only (allow <p>)",
    "skip_t3_duration":          "Skip T3 max-duration check",
    "skip_t4_words":             "Skip T4 min-words check",
    "skip_non_standard":         "Skip non-standard speech check",
}

# ── GPT skip flags forwarded to the extractor ─────────────────────────────────
GPT_BOOL_FLAGS = [
    "skip_gpt_t1_complete", "skip_gpt_t1_self_contained",
    "skip_gpt_t2_responds",
    "skip_gpt_t3_floor_intent", "skip_gpt_t3_pragmatic_complete",
    "skip_gpt_t3_mid_flow",
    "skip_gpt_t4_meaningful",
]
GPT_LABELS = {
    "skip_gpt_t1_complete":           "Skip GPT: turn1_complete",
    "skip_gpt_t1_self_contained":     "Skip GPT: turn1_self_contained",
    "skip_gpt_t2_responds":           "Skip GPT: turn2_responds",
    "skip_gpt_t3_floor_intent":       "Skip GPT: t3_floor_intent",
    "skip_gpt_t3_pragmatic_complete": "Skip GPT: t3_pragmatic_complete",
    "skip_gpt_t3_mid_flow":           "Skip GPT: t3_mid_flow_onset",
    "skip_gpt_t4_meaningful":         "Skip GPT: turn4_meaningful",
}


# ── File discovery ─────────────────────────────────────────────────────────────

def find_files() -> list[dict]:
    entries = []
    for trs in sorted(LOQUI_TRANS_DIR.glob("*.final.trs")):
        file_id = trs.stem.replace(".final", "")   # e.g. "1"
        wav = trs.parent / f"{file_id}.san.wav"
        if not wav.exists():
            print(f"  [warn] no .san.wav for {trs.name}")
            continue
        entries.append({"file_id": file_id, "trs_path": trs, "wav_path": wav})
    return entries


def is_done(entry: dict, results_dir: Path) -> bool:
    if results_dir.exists():
        return any(results_dir.rglob(f"{entry['file_id']}_ov*"))
    return False


# ── Per-file runner ────────────────────────────────────────────────────────────

def process_one(entry: dict, results_dir: Path,
                no_gpt: bool,
                no_bc_merge: bool,
                timing_skips: list[str],
                min_t3_overlap: float,
                max_t2_t1_overlap: float,
                max_t4_t3_overlap: float,
                gpt_skips: list[str]) -> dict:
    cmd = [
        PYTHON, str(SCRIPT),
        "--file_id",           entry["file_id"],
        "--trs_path",          str(entry["trs_path"]),
        "--wav_path",          str(entry["wav_path"]),
        "--results_dir",       str(results_dir),
        "--min_t3_overlap",    str(min_t3_overlap),
        "--max_t2_t1_overlap", str(max_t2_t1_overlap),
        "--max_t4_t3_overlap", str(max_t4_t3_overlap),
    ]
    if no_gpt:
        cmd.append("--no_gpt")
    if no_bc_merge:
        cmd.append("--no_bc_merge")
    for flag in timing_skips:
        cmd.append(f"--{flag}")
    for flag in gpt_skips:
        cmd.append(f"--{flag}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    quota_hit = (result.returncode == QUOTA_EXIT_CODE
                 and ("QUOTA" in result.stdout or "quota" in result.stdout.lower()))
    success   = result.returncode == 0

    timing_n          = 0
    valid_by_type:    dict[str, int] = {}
    timing_stats:     dict = {}
    gpt_stats:        dict = {}
    display_lines:    list[str] = []

    for ln in result.stdout.splitlines():
        stripped = ln.strip()
        if "Candidates found:" in stripped:
            try:
                timing_n = int(stripped.split(":")[-1].strip())
            except Exception:
                pass
        elif stripped.startswith("VALID_AFTER_LLM "):
            try:
                rest = stripped[len("VALID_AFTER_LLM "):]
                rtype, n = rest.rsplit(":", 1)
                valid_by_type[rtype.strip()] = int(n.strip())
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
        else:
            display_lines.append(ln)

    total_valid = sum(valid_by_type.values())
    return {
        "file_id":       entry["file_id"],
        "success":       success,
        "quota_hit":     quota_hit,
        "elapsed":       elapsed,
        "timing_n":      timing_n,
        "total_valid":   total_valid,
        "valid_by_type": valid_by_type,
        "timing_stats":  timing_stats,
        "gpt_stats":     gpt_stats,
        "stdout_lines":  display_lines,
        "stderr":        result.stderr[-500:] if not success else "",
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_results(results_dir: Path, completed_results: list[dict]) -> list[dict]:
    all_valid: list[dict] = []
    for cjson in results_dir.rglob("candidates.json"):
        try:
            with open(cjson) as f:
                data = json.load(f)
            all_valid.extend([c for c in data if c.get("gpt_valid")])
        except Exception:
            pass

    # Fall back: scan audio files if no candidates.json
    if not all_valid:
        for r in completed_results:
            for ln in r.get("stdout_lines", []):
                if ln.startswith("VALID_JSON:"):
                    try:
                        batch = json.loads(ln[len("VALID_JSON:"):])
                        all_valid.extend([c for c in batch if c.get("gpt_valid")])
                    except Exception:
                        pass

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return []

    counts: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        rtype = c.get("interruption_type") or "none"
        if counts[rtype] < TARGET_PER_TYPE:
            capped.append(c)
            counts[rtype] += 1

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "valid.json", "w") as f:
        json.dump(capped, f, indent=2)

    by_type: dict[str, list] = defaultdict(list)
    for c in capped:
        by_type[c.get("interruption_type") or "none"].append(c)

    print(f"\n── Global results aggregated → {results_dir} ──")
    print(f"  Total valid (cap={TARGET_PER_TYPE}/type): {len(capped)}")
    for rtype in RECOVERY_TYPES:
        n = len(by_type.get(rtype, []))
        if n:
            print(f"    {rtype}: {n}")
    return capped


# ── Filter statistics ──────────────────────────────────────────────────────────

def _print_filter_report(results: list[dict], no_gpt: bool,
                         active_timing_skips: list[str],
                         active_gpt_skips: list[str]) -> None:
    timing_keys = [
        "windows_checked", "overlap_not_two", "not_abab",
        "t2_overlaps_t1", "t3_overlap_short", "t4_overlaps_t3",
        "t1_few_words", "t1_has_pauses",
        "t2_is_question", "t2_few_words", "t2_pre_words_short",
        "t3_has_pauses", "t3_too_long", "t3_empty",
        "t4_few_words", "non_standard", "duplicate", "passed",
    ]
    timing_labels_map = {
        "windows_checked":    "Windows checked (overlap turns)",
        "overlap_not_two":    "Rejected: overlap != 2 speakers",
        "not_abab":           "Rejected: no A-only turns or no T4",
        "t2_overlaps_t1":     "Rejected: T2 overlaps T1",
        "t3_overlap_short":   "Rejected: T3–T2 overlap too small",
        "t4_overlaps_t3":     "Rejected: T4 overlaps T3",
        "t1_few_words":       "Rejected: T1 too few words",
        "t1_has_pauses":      "Rejected: T1 has internal pauses",
        "t2_is_question":     "Rejected: T2 ends with question",
        "t2_few_words":       "Rejected: T2 too few words",
        "t2_pre_words_short": "Rejected: T2 pre-overlap too few words",
        "t3_has_pauses":      "Rejected: T3 has internal pauses",
        "t3_too_long":        "Rejected: T3 duration too long",
        "t3_empty":           "Rejected: T3 empty after cleaning",
        "t4_few_words":       "Rejected: T4 too few words",
        "non_standard":       "Rejected: non-standard speech",
        "duplicate":          "Rejected: duplicate candidate",
        "passed":             "PASSED timing filter",
    }
    skipped_stat_map = {
        "t2_overlaps_t1":     "skip_t2_overlap_t1",
        "t3_overlap_short":   "skip_t3_overlap",
        "t4_overlaps_t3":     "skip_t4_overlap_t3",
        "t1_few_words":       "skip_t1_words",
        "t1_has_pauses":      "skip_t1_pauses",
        "t2_is_question":     "skip_t2_question",
        "t2_few_words":       "skip_t2_words",
        "t2_pre_words_short": "skip_t2_pre_words",
        "t3_has_pauses":      "skip_t3_pauses",
        "t3_too_long":        "skip_t3_duration",
        "t4_few_words":       "skip_t4_words",
        "non_standard":       "skip_non_standard",
    }
    skipped_timing_stat_keys = {
        stat for stat, flag in skipped_stat_map.items()
        if flag in active_timing_skips
    }

    agg_timing: dict = {k: 0 for k in timing_keys}
    for r in results:
        for k in timing_keys:
            agg_timing[k] += r.get("timing_stats", {}).get(k, 0)

    print(f"\n{'═'*65}")
    print("TIMING FILTER STATISTICS")
    print(f"{'═'*65}")
    w = agg_timing.get("windows_checked", 1) or 1
    for k in timing_keys:
        v = agg_timing[k]
        pct = 100.0 * v / w
        label = timing_labels_map.get(k, k)
        suffix = "  [SKIPPED]" if k in skipped_timing_stat_keys else ""
        print(f"  {label:<52s}: {v:>6d}  ({pct:5.1f}%){suffix}")

    if no_gpt:
        return

    gpt_bool_keys = [
        "turn1_complete_false", "turn1_self_contained_false",
        "turn2_responds_false", "turn2_interrupted_false",
        "t3_floor_intent_false", "t3_pragmatic_complete_false",
        "t3_mid_flow_onset_false", "turn4_meaningful_false",
    ]
    gpt_bool_labels_map = {
        "turn1_complete_false":        "turn1_complete = false",
        "turn1_self_contained_false":  "turn1_self_contained = false",
        "turn2_responds_false":        "turn2_responds = false",
        "turn2_interrupted_false":     "turn2_interrupted = false",
        "t3_floor_intent_false":       "t3_floor_intent = false",
        "t3_pragmatic_complete_false": "t3_pragmatic_complete = false",
        "t3_mid_flow_onset_false":     "t3_mid_flow_onset = false",
        "turn4_meaningful_false":      "turn4_meaningful = false",
    }
    gpt_skip_stat_map = {
        "turn1_complete_false":        "skip_gpt_t1_complete",
        "turn1_self_contained_false":  "skip_gpt_t1_self_contained",
        "turn2_responds_false":        "skip_gpt_t2_responds",
        "t3_floor_intent_false":       "skip_gpt_t3_floor_intent",
        "t3_pragmatic_complete_false": "skip_gpt_t3_pragmatic_complete",
        "t3_mid_flow_onset_false":     "skip_gpt_t3_mid_flow",
        "turn4_meaningful_false":      "skip_gpt_t4_meaningful",
    }
    gpt_skipped_stat_keys = {
        stat for stat, flag in gpt_skip_stat_map.items()
        if flag in active_gpt_skips
    }

    agg_sent   = sum(r.get("gpt_stats", {}).get("total_sent",   0) for r in results)
    agg_judged = sum(r.get("gpt_stats", {}).get("total_judged", 0) for r in results)
    agg_bool: dict = {k: 0 for k in gpt_bool_keys}
    for r in results:
        for k in gpt_bool_keys:
            agg_bool[k] += r.get("gpt_stats", {}).get(k, 0)

    recovery_ctr: Counter = Counter()
    valid_by_type_agg: Counter = Counter()
    for r in results:
        gs = r.get("gpt_stats", {})
        for rt, cnt in gs.get("recovery_type", {}).items():
            recovery_ctr[rt] += cnt
        for rt, cnt in gs.get("valid_by_type", {}).items():
            valid_by_type_agg[rt] += cnt

    judged_base = agg_judged or 1
    total_valid = sum(valid_by_type_agg.values())

    print(f"\n{'═'*65}")
    print("GPT FILTER STATISTICS")
    print(f"{'═'*65}")
    print(f"  Sent to LLM              : {agg_sent:>6d}")
    print(f"  Judged (non-error)       : {agg_judged:>6d}")
    print(f"\n  -- Boolean criteria (count judged = {agg_judged}) --")
    for k in gpt_bool_keys:
        v = agg_bool[k]
        pct = 100.0 * v / judged_base
        label = gpt_bool_labels_map.get(k, k)
        suffix = "  [SKIPPED]" if k in gpt_skipped_stat_keys else ""
        print(f"  {label:<52s}: {v:>6d}  ({pct:5.1f}%){suffix}")

    print(f"\n  -- recovery_type distribution (all judged) --")
    for rt in RECOVERY_TYPES:
        v = recovery_ctr.get(rt, 0)
        pct = 100.0 * v / judged_base
        print(f"  {rt:<52s}: {v:>6d}  ({pct:5.1f}%)")

    print(f"\n  -- Valid by recovery type --")
    for rt in RECOVERY_TYPES:
        n = valid_by_type_agg.get(rt, 0)
        if n:
            print(f"    {rt}: {n}")
    print(f"  Total valid              : {total_valid:>6d}  "
          f"({100.0*total_valid/judged_base:5.1f}%)")
    print(f"{'═'*65}")


# ── Markdown report ────────────────────────────────────────────────────────────

def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    n = len(vals); mean = sum(vals) / n; s = sorted(vals)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    var = sum((x - mean) ** 2 for x in vals) / n
    import math as _math
    return {"mean": round(mean, 3), "median": round(med, 3),
            "std": round(_math.sqrt(var), 3),
            "min": round(s[0], 3), "max": round(s[-1], 3), "n": n}


def _fmt(v) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "—"
    return str(v)


def write_md_report(report_path: Path, results: list[dict],
                    all_valid: list[dict], results_dir: Path,
                    active_timing_skips: list[str],
                    active_gpt_skips: list[str]) -> None:
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_files   = len(results)
    n_success = sum(1 for r in results if r["success"])
    total_timing = sum(r.get("timing_n", 0) for r in results if r["success"])
    total_sent   = sum(r.get("gpt_stats", {}).get("total_sent",   0) for r in results)
    total_judged = sum(r.get("gpt_stats", {}).get("total_judged", 0) for r in results)
    judged_base  = total_judged or 1

    by_style: dict[str, list] = defaultdict(list)
    by_type:  dict[str, list] = defaultdict(list)
    for c in all_valid:
        by_style[c.get("interruption_style", "unknown")].append(c)
        by_type[c.get("interruption_type") or "none"].append(c)
    n_hard  = len(by_style.get("hard", []))
    n_total = len(all_valid)
    accept_pct = f"{100.0 * n_total / judged_base:.1f}%" if total_judged else "—"

    gpt_bool_keys = [
        "turn1_complete_false", "turn1_self_contained_false",
        "turn2_responds_false", "turn2_interrupted_false",
        "t3_floor_intent_false", "t3_pragmatic_complete_false",
        "t3_mid_flow_onset_false", "turn4_meaningful_false",
    ]
    gpt_bool_labels_map = {
        "turn1_complete_false":        "turn1_complete = false",
        "turn1_self_contained_false":  "turn1_self_contained = false",
        "turn2_responds_false":        "turn2_responds = false",
        "turn2_interrupted_false":     "turn2_interrupted = false",
        "t3_floor_intent_false":       "t3_floor_intent = false",
        "t3_pragmatic_complete_false": "t3_pragmatic_complete = false",
        "t3_mid_flow_onset_false":     "t3_mid_flow_onset = false",
        "turn4_meaningful_false":      "turn4_meaningful = false",
    }
    gpt_skip_stat_map = {
        "turn1_complete_false":        "skip_gpt_t1_complete",
        "turn1_self_contained_false":  "skip_gpt_t1_self_contained",
        "turn2_responds_false":        "skip_gpt_t2_responds",
        "t3_floor_intent_false":       "skip_gpt_t3_floor_intent",
        "t3_pragmatic_complete_false": "skip_gpt_t3_pragmatic_complete",
        "t3_mid_flow_onset_false":     "skip_gpt_t3_mid_flow",
        "turn4_meaningful_false":      "skip_gpt_t4_meaningful",
    }
    agg_bool: dict = {k: 0 for k in gpt_bool_keys}
    recovery_ctr: Counter = Counter()
    for r in results:
        gs = r.get("gpt_stats", {})
        for k in gpt_bool_keys:
            agg_bool[k] += gs.get(k, 0)
        for rt, cnt in gs.get("recovery_type", {}).items():
            recovery_ctr[rt] += cnt

    timing_keys = [
        "windows_checked", "overlap_not_two", "not_abab",
        "t2_overlaps_t1", "t3_overlap_short", "t4_overlaps_t3",
        "t1_few_words", "t1_has_pauses",
        "t2_is_question", "t2_few_words", "t2_pre_words_short",
        "t3_has_pauses", "t3_too_long", "t3_empty",
        "t4_few_words", "non_standard", "duplicate", "passed",
    ]
    timing_labels_map = {
        "windows_checked":    "Overlap turns examined",
        "overlap_not_two":    "Rejected: overlap != 2 speakers",
        "not_abab":           "Rejected: no A-only turns or no T4",
        "t2_overlaps_t1":     "Rejected: T2 overlaps T1",
        "t3_overlap_short":   "Rejected: T3–T2 overlap too small",
        "t4_overlaps_t3":     "Rejected: T4 overlaps T3",
        "t1_few_words":       "Rejected: T1 too few words",
        "t1_has_pauses":      "Rejected: T1 has internal pauses",
        "t2_is_question":     "Rejected: T2 ends with question",
        "t2_few_words":       "Rejected: T2 too few words",
        "t2_pre_words_short": "Rejected: T2 pre-overlap too few words",
        "t3_has_pauses":      "Rejected: T3 has internal pauses",
        "t3_too_long":        "Rejected: T3 duration too long",
        "t3_empty":           "Rejected: T3 empty after cleaning",
        "t4_few_words":       "Rejected: T4 too few words",
        "non_standard":       "Rejected: non-standard speech",
        "duplicate":          "Rejected: duplicate candidate",
        "passed":             "**Passed timing filter**",
    }
    skipped_stat_map = {
        "t2_overlaps_t1":     "skip_t2_overlap_t1",
        "t3_overlap_short":   "skip_t3_overlap",
        "t4_overlaps_t3":     "skip_t4_overlap_t3",
        "t1_few_words":       "skip_t1_words",
        "t1_has_pauses":      "skip_t1_pauses",
        "t2_is_question":     "skip_t2_question",
        "t2_few_words":       "skip_t2_words",
        "t2_pre_words_short": "skip_t2_pre_words",
        "t3_has_pauses":      "skip_t3_pauses",
        "t3_too_long":        "skip_t3_duration",
        "t4_few_words":       "skip_t4_words",
        "non_standard":       "skip_non_standard",
    }
    agg_timing: dict = {k: 0 for k in timing_keys}
    for r in results:
        for k in timing_keys:
            agg_timing[k] += r.get("timing_stats", {}).get(k, 0)
    w = agg_timing.get("windows_checked", 1) or 1

    lines: list[str] = []
    lines += [
        "# Loqui Interruption Extraction — Statistical Report",
        "",
        f"> Dataset: Loqui  ",
        f"> Results: `{results_dir}`  ",
        f"> Generated: {now}  ",
        "",
    ]
    if active_timing_skips or active_gpt_skips:
        lines.append("**Active skip flags:**\n")
        if active_timing_skips:
            lines.append("Timing skips: " +
                         ", ".join(f"`--{f}`" for f in active_timing_skips))
        if active_gpt_skips:
            lines.append("GPT skips: " +
                         ", ".join(f"`--{f}`" for f in active_gpt_skips))
        lines.append("")

    lines += [
        "## 1. Pipeline Funnel", "",
        "| Stage | Count |", "| --- | --- |",
        f"| Total files discovered | {n_files} |",
        f"| Successfully processed | {n_success} |",
        f"| Timing-filter candidates | {total_timing} |",
        f"| Sent to GPT | {total_sent} |",
        f"| GPT judged (non-error) | {total_judged} |",
        f"| GPT-validated candidates (final) | {n_total} |",
        f"| GPT acceptance rate | {accept_pct} |", "",
    ]

    lines += [
        "## 2. Breakdown by Interruption Style", "",
        "| Style | Count | % of valid |", "| --- | --- | --- |",
    ]
    for style in ("hard", "none", "unknown"):
        n = len(by_style.get(style, []))
        if n:
            pct = f"{100.0 * n / n_total:.1f}%" if n_total else "—"
            lines.append(f"| {style} | {n} | {pct} |")
    lines += [f"| **Total** | **{n_total}** |  |", ""]

    lines += [
        "## 3. Breakdown by Recovery Type", "",
        "| Type | Count | % of valid |", "| --- | --- | --- |",
    ]
    for rt in RECOVERY_TYPES:
        n = len(by_type.get(rt, []))
        pct = f"{100.0 * n / n_total:.1f}%" if n_total else "—"
        lines.append(f"| {rt} | {n} | {pct} |")
    lines.append("")

    trows = {
        "overlap_duration": [c.get("overlap_duration", 0) for c in all_valid],
        "t2_pre_duration":  [c.get("t2_pre_duration",  0) for c in all_valid],
        "t3_duration":      [c["t3_end"] - c["t3_start"] for c in all_valid],
        "t4_duration":      [c["t4_end"] - c["t4_start"] for c in all_valid],
    }
    TIMING_METRIC_LABELS = {
        "overlap_duration": "T2–T3 overlap duration (s)",
        "t2_pre_duration":  "T2 pre-overlap duration (s)",
        "t3_duration":      "T3 (interruption) duration (s)",
        "t4_duration":      "T4 (recovery) duration (s)",
    }
    lines += [
        "## 4. Timing Statistics", "",
        "| Metric | Mean | Median | Std | Min | Max | N |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, label in TIMING_METRIC_LABELS.items():
        s = _stats(trows[key])
        lines.append(
            f"| {label} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
            f"{_fmt(s['std'])} | {_fmt(s['min'])} | {_fmt(s['max'])} | {s['n']} |"
        )
    lines.append("")

    lines += [
        "## 5. Timing Filter Statistics", "",
        "| Criterion | Count | % of windows | Active |",
        "| --- | --- | --- | --- |",
    ]
    for k in timing_keys:
        v = agg_timing[k]
        pct = f"{100.0 * v / w:.1f}%"
        label = timing_labels_map.get(k, k)
        skip_flag = skipped_stat_map.get(k)
        active = "No" if (skip_flag and skip_flag in active_timing_skips) else "Yes"
        lines.append(f"| {label} | {v} | {pct} | {active} |")
    lines.append("")

    lines += [
        "## 6. GPT Filter Statistics", "",
        f"Total sent: {total_sent}  |  Total judged: {total_judged}", "",
        "### 6.1 Boolean Rejection Criteria", "",
        "| Criterion | Rejected count | % of judged | Active |",
        "| --- | --- | --- | --- |",
    ]
    for k in gpt_bool_keys:
        v = agg_bool[k]
        pct = f"{100.0 * v / judged_base:.1f}%"
        skip_flag = gpt_skip_stat_map.get(k)
        active = "No" if (skip_flag and skip_flag in active_gpt_skips) else "Yes"
        lines.append(f"| {gpt_bool_labels_map[k]} | {v} | {pct} | {active} |")
    lines.append("")

    lines += [
        "### 6.2 recovery_type Distribution (all judged)", "",
        "| Recovery type | Count | % of judged |",
        "| --- | --- | --- |",
    ]
    for rt in RECOVERY_TYPES:
        v = recovery_ctr.get(rt, 0)
        pct = f"{100.0 * v / judged_base:.1f}%"
        lines.append(f"| {rt} | {v} | {pct} |")
    lines += ["",
              f"**GPT acceptance rate: {accept_pct}** (hard={n_hard}, total={n_total})",
              ""]

    lines += [
        "## 7. All Valid Candidates", "",
        "| File ID | Overlap idx | Style | Recovery type | "
        "Overlap (s) | T2 pre (s) | T3 dur (s) | T4 dur (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in sorted(all_valid, key=lambda x: (x.get("file_id", ""),
                                               x.get("ov_idx", 0))):
        t3d = round(c["t3_end"] - c["t3_start"], 2)
        t4d = round(c["t4_end"] - c["t4_start"], 2)
        lines.append(
            f"| {c['file_id']} | ov{c.get('ov_idx','?')} "
            f"| {c.get('interruption_style','?')} "
            f"| {c.get('interruption_type','?')} "
            f"| {c.get('overlap_duration','?')} | {c.get('t2_pre_duration','?')} "
            f"| {t3d} | {t4d} |"
        )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Markdown report → {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global LOQUI_ROOT, LOQUI_TRANS_DIR
    parser = argparse.ArgumentParser(
        description="Batch Loqui interruption extraction.")

    parser.add_argument("--workers",     type=int,  default=4)
    parser.add_argument("--force",       action="store_true",
                        help="Reprocess already-done files")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--no_gpt",      action="store_true")
    parser.add_argument("--no_bc_merge", action="store_true",
                        help="Disable backchannel overlap merge pre-processing")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--report_path", type=Path, default=None)

    for flag in TIMING_BOOL_FLAGS:
        parser.add_argument(f"--{flag}", action="store_true",
                            help=TIMING_LABELS.get(flag, ""))
    parser.add_argument("--min_t3_overlap",    type=float, default=0.3,
                        help="Minimum T3–T2 overlap in seconds (default 0.3)")
    parser.add_argument("--max_t2_t1_overlap", type=float, default=1.5,
                        help="Max allowed T2-before-T1 overlap (default 1.5s)")
    parser.add_argument("--max_t4_t3_overlap", type=float, default=1.5,
                        help="Max allowed T4-before-T3 overlap (default 1.5s)")

    for flag in GPT_BOOL_FLAGS:
        parser.add_argument(f"--{flag}", action="store_true",
                            help=GPT_LABELS.get(flag, ""))

    parser.add_argument("--loqui_root", type=Path, default=LOQUI_ROOT,
                        help="Loqui root (LoquiPackage, or the folder that contains it). Also LOQUI_ROOT.")
    args = parser.parse_args()
    LOQUI_ROOT = args.loqui_root.expanduser().resolve()
    LOQUI_TRANS_DIR = resolve_trans_dir(LOQUI_ROOT)
    if args.report_path is None:
        args.report_path = args.results_dir / "report.md"

    active_timing_skips = [f for f in TIMING_BOOL_FLAGS if getattr(args, f)]
    active_gpt_skips    = [f for f in GPT_BOOL_FLAGS    if getattr(args, f)]

    print("Discovering Loqui files...")
    all_entries = find_files()

    if args.force or args.no_gpt:
        todo = all_entries
    else:
        todo = [e for e in all_entries if not is_done(e, args.results_dir)]

    done_count = len(all_entries) - len(todo)
    print(f"Found {len(all_entries)} file(s) total")
    print(f"  Already done      : {done_count}")
    print(f"  To process        : {len(todo)}")
    print(f"  Workers           : {args.workers}")
    print(f"  Results dir       : {args.results_dir}")
    print(f"  BC-merge          : {'disabled' if args.no_bc_merge else 'enabled'}")
    print(f"  min_t3_overlap    : {args.min_t3_overlap}s")
    print(f"  max_t2_t1_overlap : {args.max_t2_t1_overlap}s")
    print(f"  max_t4_t3_overlap : {args.max_t4_t3_overlap}s")
    if active_timing_skips:
        print(f"  Timing skips      : {', '.join(active_timing_skips)}")
    if active_gpt_skips:
        print(f"  GPT skips         : {', '.join(active_gpt_skips)}")

    if args.dry_run:
        print("\nDry run — files to process:")
        for e in todo:
            print(f"  {e['file_id']}")
        return

    if not todo:
        print("\nNothing to do. Use --force to reprocess.")
        return

    print(f"\nStarting batch processing...\n{'─'*60}")
    batch_start = time.time()

    results      = []
    n_done       = 0
    n_failed     = 0
    total_timing = 0
    total_valid  = 0
    quota_abort  = False

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one, e, args.results_dir,
                args.no_gpt, args.no_bc_merge,
                active_timing_skips,
                args.min_t3_overlap,
                args.max_t2_t1_overlap,
                args.max_t4_t3_overlap,
                active_gpt_skips,
            ): e
            for e in todo
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
                    f"  Failed at: {res['file_id']}\n"
                    f"  Completed: {n_done}/{len(todo)}\n"
                    f"{'!'*60}"
                )
                break

            status = "OK " if res["success"] else "ERR"
            if args.no_gpt:
                print(f"[{n_done:3d}/{len(todo)}] {status}  {res['file_id']}  "
                      f"timing={res['timing_n']}  {res['elapsed']}s")
            else:
                by_type_str = "  ".join(
                    f"{t}={n}" for t, n in sorted(res["valid_by_type"].items()) if n)
                print(f"[{n_done:3d}/{len(todo)}] {status}  {res['file_id']}  "
                      f"timing={res['timing_n']}  valid={res['total_valid']}"
                      + (f"  [{by_type_str}]" if by_type_str else "")
                      + f"  {res['elapsed']}s")

            if not res["success"]:
                n_failed += 1
                print(f"    STDERR: {res['stderr']}")
            else:
                total_timing += res["timing_n"]
                total_valid  += res["total_valid"]

    elapsed_total = round(time.time() - batch_start, 1)
    print(f"\n{'─'*60}")
    if quota_abort:
        print(f"Batch ABORTED (quota exceeded) after {elapsed_total}s")
    else:
        print(f"Batch complete in {elapsed_total}s")
        print(f"  Succeeded        : {n_done - n_failed}/{len(todo)}")
        print(f"  Failed           : {n_failed}")
        print(f"  Total timing hits: {total_timing}")
        if not args.no_gpt:
            print(f"  Total valid      : {total_valid}")

    if n_failed:
        print("\nFailed files:")
        for r in results:
            if not r["success"]:
                print(f"  {r['file_id']}: {r['stderr']}")

    _print_filter_report(results, args.no_gpt, active_timing_skips, active_gpt_skips)

    if not quota_abort and not args.no_gpt:
        all_valid = aggregate_results(args.results_dir, results)
        write_md_report(args.report_path, results, all_valid, args.results_dir,
                        active_timing_skips, active_gpt_skips)


if __name__ == "__main__":
    main()
