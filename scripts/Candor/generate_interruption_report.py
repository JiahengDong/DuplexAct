#!/usr/bin/env python3
"""
Generate a Markdown statistical report from all interruption extraction results.

Works with both the Backbiter pipeline (interruption_candidates_backbiter/) and
the legacy pipeline (interruption_candidates/).

Usage:
  python generate_interruption_report.py
  python generate_interruption_report.py --results_dir /path/to/results/interruptions_backbiter
  python generate_interruption_report.py --out report.md
  python generate_interruption_report.py --no_gpt   # timing-only run (no valid.json)
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CANDOR_ROOT  = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
DEFAULT_RESULTS = Path("/data/gpfs/projects/punim2960/jiahengd/results/interruptions_backbiter")
DEFAULT_OUT     = Path("/data/gpfs/projects/punim2960/jiahengd/results/interruptions_backbiter/report.md")

# Per-conversation subdirectory name (backbiter vs legacy)
CONV_SUBDIR = "interruption_candidates_backbiter"


def load_all_valid(results_dir: Path) -> list[dict]:
    vpath = results_dir / "valid.json"
    if vpath.exists():
        return json.load(open(vpath))
    return []


def load_timing_candidates(results_dir: Path) -> list[dict]:
    p = results_dir / "timing_candidates.json"
    if p.exists():
        return json.load(open(p))
    return []


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None}
    return {
        "n":      len(values),
        "mean":   round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "std":    round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min":    round(min(values), 3),
        "max":    round(max(values), 3),
    }


def fmt_stat(s: dict, unit: str = "") -> str:
    if s["n"] == 0:
        return "—"
    u = f" {unit}" if unit else ""
    return (f"{s['mean']}{u} "
            f"(median {s['median']}{u}, "
            f"std {s['std']}{u}, "
            f"range [{s['min']}–{s['max']}{u}])")


def count_conversations(candor_root: Path, no_gpt: bool = False,
                        conv_subdir: str = CONV_SUBDIR) -> dict:
    """Count how many conversations were processed."""
    total, processed, with_valid = 0, 0, 0
    for part_dir in sorted(candor_root.glob("part_*")):
        for conv_dir in part_dir.iterdir():
            if not conv_dir.is_dir():
                continue
            total += 1
            if no_gpt:
                f1 = conv_dir / conv_subdir / "filter1_timing" / "all.json"
                if f1.exists():
                    processed += 1
            else:
                f2 = conv_dir / conv_subdir / "filter2_gpt" / "all_with_gpt_decisions.json"
                if f2.exists():
                    processed += 1
                    valid = conv_dir / conv_subdir / "filter2_gpt" / "valid.json"
                    if valid.exists():
                        data = json.load(open(valid))
                        if data:
                            with_valid += 1
    return {"total": total, "processed": processed, "with_valid": with_valid}


def timing_filter_totals(candor_root: Path, conv_subdir: str = CONV_SUBDIR) -> int:
    """Count total timing-filter candidates across all processed conversations."""
    total = 0
    for part_dir in sorted(candor_root.glob("part_*")):
        for conv_dir in part_dir.iterdir():
            f1 = conv_dir / conv_subdir / "filter1_timing" / "all.json"
            if f1.exists():
                try:
                    total += len(json.load(open(f1)))
                except Exception:
                    pass
    return total


def gpt_rejection_totals(candor_root: Path, conv_subdir: str = CONV_SUBDIR) -> dict:
    """Aggregate GPT rejection counts across all processed conversations."""
    totals: dict[str, int] = {
        "total_sent": 0, "total_judged": 0, "valid_hard": 0,
        "turn1_complete_false": 0, "turn1_self_contained_false": 0,
        "turn2_responds_false": 0, "turn2_interrupted_false": 0,
        "t3_floor_intent_false": 0, "t3_pragmatic_complete_false": 0,
        "t3_mid_flow_onset_false": 0, "turn4_meaningful_false": 0,
    }
    for part_dir in sorted(candor_root.glob("part_*")):
        for conv_dir in part_dir.iterdir():
            f2 = conv_dir / conv_subdir / "filter2_gpt" / "all_with_gpt_decisions.json"
            if not f2.exists():
                continue
            try:
                candidates = json.load(open(f2))
            except Exception:
                continue
            judged = [c for c in candidates if c.get("gpt_valid") is not None]
            valid  = [c for c in judged  if c.get("gpt_valid")]
            totals["total_sent"]   += len(candidates)
            totals["total_judged"] += len(judged)
            totals["valid_hard"]   += sum(1 for c in valid
                                          if c.get("interruption_style") == "hard")
            for c in judged:
                if c.get("gpt_valid"):
                    continue
                if c.get("turn1_complete") is False:
                    totals["turn1_complete_false"] += 1
                if c.get("turn1_self_contained") is False:
                    totals["turn1_self_contained_false"] += 1
                if c.get("turn2_responds") is False:
                    totals["turn2_responds_false"] += 1
                if c.get("turn2_interrupted") is False:
                    totals["turn2_interrupted_false"] += 1
                if c.get("t3_floor_intent") is False:
                    totals["t3_floor_intent_false"] += 1
                if c.get("t3_pragmatic_complete") is False:
                    totals["t3_pragmatic_complete_false"] += 1
                if c.get("t3_mid_flow_onset") is False:
                    totals["t3_mid_flow_onset_false"] += 1
                if c.get("turn4_meaningful") is False:
                    totals["turn4_meaningful_false"] += 1
    return totals


def generate_report(results_dir: Path, candor_root: Path, out_path: Path,
                    no_gpt: bool = False,
                    skipped_gpt: set[str] | None = None,
                    conv_subdir: str = CONV_SUBDIR) -> None:
    """
    skipped_gpt: set of criterion names that were skipped in this run.
    Accepted values: 't1_complete', 't1_self_contained', 't2_responds',
                     't3_floor_intent', 't3_pragmatic_complete', 't3_mid_flow_onset',
                     't4_meaningful'
    """
    if skipped_gpt is None:
        skipped_gpt = set()

    if no_gpt:
        valid = load_timing_candidates(results_dir)
    else:
        valid = load_all_valid(results_dir)
    conv_counts  = count_conversations(candor_root, no_gpt, conv_subdir)
    timing_total = timing_filter_totals(candor_root, conv_subdir)
    gpt_totals   = {} if no_gpt else gpt_rejection_totals(candor_root, conv_subdir)
    now          = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")
    def row(*cells):
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    def sep(n: int):
        lines.append("| " + " | ".join(["---"] * n) + " |")

    # ── Header ────────────────────────────────────────────────────────────────
    h(1, "Interruption Extraction — Statistical Report")
    mode_note = "Timing-only run (--no_gpt)" if no_gpt else "Full run (timing + GPT)"
    lines.append(f"> Dataset: CANDOR  \n"
                 f"> Results: `{results_dir}`  \n"
                 f"> Generated: {now}  |  Mode: {mode_note}  \n")

    def extract(key):
        return [float(c[key]) for c in valid if c.get(key) is not None]

    def stat_row(label, values, unit=""):
        s = stats(values)
        if s["n"] == 0:
            row(label, *["—"] * 6)
        else:
            u = f" {unit}" if unit else ""
            row(label,
                f"{s['mean']}{u}", f"{s['median']}{u}", f"{s['std']}{u}",
                f"{s['min']}{u}", f"{s['max']}{u}", s["n"])

    def srow(label, values, unit=""):
        s = stats(values)
        if s["n"] == 0:
            row(label, *["—"] * 5)
            return
        u = f" {unit}" if unit else ""
        row(label,
            f"{s['mean']}{u}", f"{s['median']}{u}", f"{s['std']}{u}",
            f"{s['min']}{u}", f"{s['max']}{u}")

    # ── Load filter_stats.json if present ─────────────────────────────────────
    filter_stats_path = results_dir / "filter_stats.json"
    filter_stats = json.load(open(filter_stats_path)) if filter_stats_path.exists() else {}
    fs_timing = filter_stats.get("timing", {})
    fs_gpt    = filter_stats.get("gpt", {})
    fs_convs  = filter_stats.get("n_convs", conv_counts.get("processed", 0))

    # ── 1. Pipeline funnel ─────────────────────────────────────────────────────
    h(2, "1. Pipeline Funnel")
    row("Stage", "Count")
    sep(2)
    row("Total conversations", conv_counts["total"])
    if no_gpt:
        row("Processed (timing filter)", conv_counts["processed"])
        row("Timing-filter candidates (pre-GPT)", timing_total)
        row("Candidates loaded for analysis", len(valid))
        lines.append(f"\n*Timing-only run — GPT classification skipped. "
                     f"Statistics below are computed over all timing-filter candidates.*\n")
    else:
        row("Processed (filter1 + GPT)", conv_counts["processed"])
        row("Conversations with ≥1 valid candidate", conv_counts["with_valid"])
        row("Timing-filter candidates (pre-GPT)", timing_total)
        gpt_sent = gpt_totals.get("total_sent", 0)
        row("Total sent to GPT", gpt_sent)
        row("GPT-validated candidates (final)", len(valid))
        if gpt_sent:
            row("GPT acceptance rate", f"{100 * len(valid) / gpt_sent:.1f}%")
    lines.append("")

    # ── 1b. Timing filter breakdown ────────────────────────────────────────────
    if fs_timing:
        h(2, "1b. Timing Filter Breakdown")
        windows = fs_timing.get("windows_checked", 1) or 1
        row("Criterion", "Count", "% of windows checked")
        sep(3)
        TIMING_LABELS = {
            "windows_checked":         "Windows checked",
            "not_alternating":         "Not A-B-A-B alternation",
            "t2_before_t1_ends":       "T2 starts before T1 ends",
            "t3_overlap_too_small":    "T3 overlap too small",
            "t4_overlaps_t3":          "T4 overlaps T3",
            "t1_no_punct":             "T1 no sentence-final punct",
            "t1_few_words":            "T1 too few words",
            "t1_has_pauses":           "T1 has internal pauses",
            "t1_has_backchannel":      "T1 has backchannel from B",
            "t2_is_question":          "T2 is a question",
            "t2_few_words":            "T2 too few words",
            "t2_pre_too_short":        "T2 pre-overlap words too short",
            "t2_has_backchannel":      "T2 has backchannel from A",
            "t3_has_backchannel":      "T3 has backchannel from B",
            "t3_has_pauses":           "T3 has internal pauses",
            "t3_too_long":             "T3 too long",
            "t3_empty":                "T3 empty",
            "t4_few_words":            "T4 too few words",
            "duplicate":               "Duplicate",
            "passed":                  "PASSED timing",
        }
        for key, label in TIMING_LABELS.items():
            val = fs_timing.get(key, 0)
            if key == "windows_checked":
                row(label, val, "—")
            else:
                pct = f"{100.0 * val / windows:.1f}%"
                row(label, val, pct)
        lines.append(f"\n*Aggregated over {fs_convs} conversations.*\n")

    # ── 2. GPT rejection criteria (full run only) ─────────────────────────────
    sec = 2
    if not no_gpt and gpt_totals.get("total_judged", 0):
        h(2, "2. GPT Filter — Rejection Criteria")
        judged = gpt_totals["total_judged"]
        row("Metric", "Count")
        sep(2)
        row("Sent to GPT", gpt_totals["total_sent"])
        row("Judged", judged)
        row("Valid (GPT-accepted)", len(valid))
        row("Valid — hard interruption", gpt_totals.get("valid_hard", 0))
        if skipped_gpt:
            lines.append(f"\n> **Relaxed run** — the following validity criteria were "
                         f"skipped: `{'`, `'.join(sorted(skipped_gpt))}`\n")
        lines.append("")
        row("Rejection criterion", "Count", "% of judged", "Status")
        sep(4)
        # (label, totals-key, skip-key)
        criteria = [
            ("turn1_complete = false",         "turn1_complete_false",        "t1_complete"),
            ("turn1_self_contained = false",   "turn1_self_contained_false",  "t1_self_contained"),
            ("turn2_responds = false",         "turn2_responds_false",        "t2_responds"),
            ("turn2_interrupted = false",      "turn2_interrupted_false",     None),
            ("t3_floor_intent = false",        "t3_floor_intent_false",       "t3_floor_intent"),
            ("t3_pragmatic_complete = false",  "t3_pragmatic_complete_false", "t3_pragmatic_complete"),
            ("t3_mid_flow_onset = false",      "t3_mid_flow_onset_false",     "t3_mid_flow_onset"),
            ("turn4_meaningful = false",       "turn4_meaningful_false",      "t4_meaningful"),
        ]
        for label, key, skip_key in criteria:
            n   = gpt_totals.get(key, 0)
            pct = f"{100 * n / judged:.1f}%" if judged else "—"
            if skip_key and skip_key in skipped_gpt:
                status = "*(skipped — not required)*"
            else:
                status = "active"
            row(label, n, pct, status)
        lines.append("")
        sec = 3

    # ── Breakdown by style ─────────────────────────────────────────────────────
    hard = [c for c in valid if c.get("interruption_style") == "hard"]
    soft = [c for c in valid if c.get("interruption_style") == "soft"]
    unclassified = [c for c in valid
                    if c.get("interruption_style") not in ("hard", "soft")]

    if not no_gpt:
        h(2, f"{sec}. Breakdown by Interruption Style")
        row("Style", "Count", "% of valid")
        sep(3)
        for label, subset in [("Hard", hard), ("Soft", soft)]:
            pct = f"{100 * len(subset) / len(valid):.1f}%" if valid else "—"
            row(label, len(subset), pct)
        row("**Total**", f"**{len(valid)}**", "")
        lines.append("")
        sec += 1

        # ── Breakdown by recovery type ─────────────────────────────────────────
        h(2, f"{sec}. Breakdown by Recovery / Interruption Type")
        by_type: dict[str, int] = defaultdict(int)
        for c in valid:
            by_type[c.get("interruption_type", "unknown")] += 1
        row("Type", "Count", "% of valid")
        sep(3)
        for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
            pct = f"{100 * n / len(valid):.1f}%" if valid else "—"
            row(t, n, pct)
        lines.append("")
        sec += 1

        # ── Style × Type cross-table ───────────────────────────────────────────
        h(2, f"{sec}. Style × Type Cross-Table")
        all_types = sorted(by_type.keys())
        row("Style \\ Type", *all_types, "**Total**")
        sep(len(all_types) + 2)
        for style_label, subset in [("Hard", hard), ("Soft", soft)]:
            counts = [sum(1 for c in subset if c.get("interruption_type") == t)
                      for t in all_types]
            row(style_label, *counts, sum(counts))
        totals_row = [by_type[t] for t in all_types]
        row("**Total**", *totals_row, sum(totals_row))
        lines.append("")
        sec += 1

    # ── Timing statistics ──────────────────────────────────────────────────────
    h(2, f"{sec}. Timing Statistics")

    overlap_dur   = extract("overlap_duration")
    yield_lat     = extract("yield_latency")
    t2_pre_words  = [len(c.get("t2_pre_overlap", "").split()) for c in valid]
    t2_post_words = [len(c.get("t2_post_overlap", "").split()) for c in valid]
    t3_words      = [len(c.get("t3_text", "").split()) for c in valid]
    t3_dur        = [float(c["t3_end"]) - float(c["t3_start"]) for c in valid
                     if c.get("t3_start") and c.get("t3_end")]
    t4_words      = [len(c.get("t4_text", "").split()) for c in valid]
    t2_dur        = [float(c["t2_end"]) - float(c["t2_start"]) for c in valid
                     if c.get("t2_start") and c.get("t2_end")]

    row("Metric", "Mean", "Median", "Std", "Min", "Max", "N")
    sep(7)
    stat_row("T2–T3 overlap duration", overlap_dur, "s")
    stat_row("Yield latency (T3 onset → T2 yield)", yield_lat, "s")
    stat_row("T2 duration", t2_dur, "s")
    stat_row("T2 pre-overlap words", t2_pre_words, "w")
    stat_row("T2 post-overlap words (capped)", t2_post_words, "w")
    stat_row("T3 word count", t3_words, "w")
    stat_row("T3 duration", t3_dur, "s")
    stat_row("T4 word count", t4_words, "w")
    lines.append("")
    sec += 1

    # ── Per-style timing breakdown (full run only) ─────────────────────────────
    if not no_gpt:
        h(2, f"{sec}. Timing by Style")
        for style_label, subset in [("Hard", hard), ("Soft", soft)]:
            h(3, f"{sec}.{1 if style_label=='Hard' else 2}. {style_label} Interruptions "
                 f"(n={len(subset)})")
            if not subset:
                lines.append("_No candidates._\n")
                continue
            row("Metric", "Mean", "Median", "Std", "Min", "Max")
            sep(6)
            srow("Overlap duration",
                 [float(c["overlap_duration"]) for c in subset
                  if c.get("overlap_duration") is not None], "s")
            srow("Yield latency",
                 [float(c["yield_latency"]) for c in subset
                  if c.get("yield_latency") is not None], "s")
            srow("T2 duration",
                 [float(c["t2_end"]) - float(c["t2_start"]) for c in subset
                  if c.get("t2_start") and c.get("t2_end")], "s")
            srow("T2 pre words",
                 [len(c.get("t2_pre_overlap", "").split()) for c in subset], "w")
            srow("T2 post words",
                 [len(c.get("t2_post_overlap", "").split()) for c in subset], "w")
            srow("T3 duration",
                 [float(c["t3_end"]) - float(c["t3_start"]) for c in subset
                  if c.get("t3_start") and c.get("t3_end")], "s")
            lines.append("")
        sec += 1

    # ── Per-part summary ──────────────────────────────────────────────────────
    h(2, f"{sec}. Per-Part Summary")
    by_part: dict[str, list] = defaultdict(list)
    for c in valid:
        conv_id = c.get("conv_id", "")
        for part_dir in sorted(candor_root.glob("part_*")):
            if (part_dir / conv_id).is_dir():
                by_part[part_dir.name].append(c)
                break

    if no_gpt:
        row("Part", "Conversations", "Timing candidates")
        sep(3)
        for part_dir in sorted(candor_root.glob("part_*")):
            n_convs = sum(1 for d in part_dir.iterdir() if d.is_dir())
            subset  = by_part.get(part_dir.name, [])
            row(part_dir.name, n_convs, len(subset))
    else:
        row("Part", "Conversations", "Valid candidates", "Hard", "Soft")
        sep(5)
        for part_dir in sorted(candor_root.glob("part_*")):
            n_convs = sum(1 for d in part_dir.iterdir() if d.is_dir())
            subset  = by_part.get(part_dir.name, [])
            h_n = sum(1 for c in subset if c.get("interruption_style") == "hard")
            s_n = sum(1 for c in subset if c.get("interruption_style") == "soft")
            row(part_dir.name, n_convs, len(subset), h_n, s_n)
    lines.append("")
    sec += 1

    # ── Candidate listing ──────────────────────────────────────────────────────
    if no_gpt:
        h(2, f"{sec}. Sample Timing Candidates (first 50)")
        row("Conv ID", "Turns", "Overlap (s)", "Yield latency (s)",
            "T2 pre (w)", "T3 (w)", "T2 pre text (truncated)")
        sep(7)
        for c in valid[:50]:
            row(
                c.get("conv_id", "")[:8] + "…",
                f"t{c.get('t2_turn_id')}→t{c.get('t3_turn_id')}",
                round(float(c["overlap_duration"]), 2) if c.get("overlap_duration") else "—",
                round(float(c["yield_latency"]), 2) if c.get("yield_latency") is not None else "—",
                len(c.get("t2_pre_overlap", "").split()),
                len(c.get("t3_text", "").split()),
                (c.get("t2_pre_overlap", "") or "")[:60] + "…",
            )
    else:
        h(2, f"{sec}. All Valid Candidates")
        row("Conv ID", "Turns", "Style", "Type",
            "T2 pre (w)", "T3 (w)", "Overlap (s)", "Yield latency (s)")
        sep(8)
        for c in valid:
            row(
                c.get("conv_id", "")[:8] + "…",
                f"t{c.get('t1_turn_id')}–t{c.get('t4_turn_id')}",
                c.get("interruption_style", ""),
                c.get("interruption_type", ""),
                len(c.get("t2_pre_overlap", "").split()),
                len(c.get("t3_text", "").split()),
                round(float(c["overlap_duration"]), 2) if c.get("overlap_duration") else "—",
                round(float(c["yield_latency"]), 2) if c.get("yield_latency") is not None else "—",
            )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written → {out_path}")
    print(f"  Total valid candidates : {len(valid)}")
    print(f"  Hard: {len(hard)}  Soft: {len(soft)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--candor_root", type=Path, default=CANDOR_ROOT)
    parser.add_argument("--out",         type=Path, default=None)
    parser.add_argument("--no_gpt",      action="store_true",
                        help="Timing-only run — skip GPT sections")
    parser.add_argument("--conv_subdir", type=str, default=CONV_SUBDIR,
                        help="Per-conversation result subdirectory name")
    # GPT criteria that were skipped in this run (used to label report)
    parser.add_argument("--skip_gpt_t1_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t1_self_contained", action="store_true")
    parser.add_argument("--skip_gpt_t2_responds",           action="store_true")
    parser.add_argument("--skip_gpt_t3_floor_intent",       action="store_true")
    parser.add_argument("--skip_gpt_t3_pragmatic_complete", action="store_true")
    parser.add_argument("--skip_gpt_t3_mid_flow",           action="store_true")
    parser.add_argument("--skip_gpt_t4_meaningful",         action="store_true")
    args = parser.parse_args()

    skipped: set[str] = set()
    if args.skip_gpt_t1_complete:           skipped.add("t1_complete")
    if args.skip_gpt_t1_self_contained:     skipped.add("t1_self_contained")
    if args.skip_gpt_t2_responds:           skipped.add("t2_responds")
    if args.skip_gpt_t3_floor_intent:       skipped.add("t3_floor_intent")
    if args.skip_gpt_t3_pragmatic_complete: skipped.add("t3_pragmatic_complete")
    if args.skip_gpt_t3_mid_flow:           skipped.add("t3_mid_flow_onset")
    if args.skip_gpt_t4_meaningful:     skipped.add("t4_meaningful")

    out = args.out or (args.results_dir / "report.md")
    generate_report(args.results_dir, args.candor_root, out,
                    no_gpt=args.no_gpt, skipped_gpt=skipped,
                    conv_subdir=args.conv_subdir)
