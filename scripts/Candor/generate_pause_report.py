#!/usr/bin/env python3
"""
Generate a Markdown statistical report from all pause extraction results.

Usage:
  python generate_pause_report.py
  python generate_pause_report.py --results_dir /path/to/results/pauses
  python generate_pause_report.py --out report.md
  python generate_pause_report.py --no_gpt          # timing-only run
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

CANDOR_ROOT     = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
DEFAULT_RESULTS = Path("/data/gpfs/projects/punim2960/jiahengd/results/pauses")

PAUSE_TYPES = ('inter_sentential', 'intra_sentential', 'floor_holding')

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


def load_all_valid(results_dir: Path) -> list[dict]:
    p = results_dir / "valid.json"
    return json.load(open(p)) if p.exists() else []


def load_timing_candidates(results_dir: Path) -> list[dict]:
    p = results_dir / "timing_candidates.json"
    return json.load(open(p)) if p.exists() else []


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


def count_conversations(candor_root: Path, no_gpt: bool = False) -> dict:
    total = processed = with_valid = 0
    for part_dir in sorted(candor_root.glob("part_*")):
        for conv_dir in part_dir.iterdir():
            if not conv_dir.is_dir():
                continue
            total += 1
            if no_gpt:
                f = conv_dir / "pause_candidates" / "filter1_timing" / "all.json"
                if f.exists():
                    processed += 1
                    try:
                        data = json.load(open(f))
                        if data:
                            with_valid += 1
                    except Exception:
                        pass
            else:
                f2 = conv_dir / "pause_candidates" / "filter2_gpt" / "all_with_gpt_decisions.json"
                if f2.exists():
                    processed += 1
                    try:
                        data = json.load(open(f2))
                        if any(c.get('gpt_valid') for c in data):
                            with_valid += 1
                    except Exception:
                        pass
    return {"total": total, "processed": processed, "with_valid": with_valid}


def timing_filter_totals(candor_root: Path) -> dict[str, int]:
    """Count per-type timing-filter candidates (passed) across all conversations."""
    totals = {t: 0 for t in PAUSE_TYPES}
    for part_dir in sorted(candor_root.glob("part_*")):
        for conv_dir in part_dir.iterdir():
            for ptype in PAUSE_TYPES:
                p = conv_dir / "pause_candidates" / "filter1_timing" / f"{ptype}.json"
                if p.exists():
                    try:
                        totals[ptype] += len(json.load(open(p)))
                    except Exception:
                        pass
    return totals


def gpt_rejection_totals(candor_root: Path) -> dict[str, dict]:
    """Aggregate GPT rejection criterion counts from per-conversation files."""
    totals: dict[str, dict] = {
        t: {"total_sent": 0, "valid": 0, "rejected": 0, "criteria": defaultdict(int)}
        for t in PAUSE_TYPES
    }
    for part_dir in sorted(candor_root.glob("part_*")):
        for conv_dir in part_dir.iterdir():
            p = conv_dir / "pause_candidates" / "filter2_gpt" / "all_with_gpt_decisions.json"
            if not p.exists():
                continue
            try:
                data = json.load(open(p))
            except Exception:
                continue
            for c in data:
                ptype = c.get("type")
                if ptype not in totals:
                    continue
                entry = totals[ptype]
                if c.get("gpt_valid") is None:
                    continue
                entry["total_sent"] += 1
                if c.get("gpt_valid"):
                    entry["valid"] += 1
                else:
                    entry["rejected"] += 1
                    for k, v in (c.get("gpt_details") or {}).items():
                        if v is False:
                            entry["criteria"][k] += 1
    return totals


def generate_report(results_dir: Path, candor_root: Path, out_path: Path,
                    no_gpt: bool = False) -> None:
    if no_gpt:
        candidates = load_timing_candidates(results_dir)
    else:
        candidates = load_all_valid(results_dir)

    conv_counts = count_conversations(candor_root, no_gpt=no_gpt)
    timing_tots = timing_filter_totals(candor_root)

    # Load pre-computed filter stats saved by the batch script (if present)
    timing_filter_stats: dict = {}
    gpt_filter_stats:    dict = {}
    tf_path = results_dir / "timing_filter_stats.json"
    gf_path = results_dir / "gpt_filter_stats.json"
    if tf_path.exists():
        try:
            timing_filter_stats = json.load(open(tf_path))
        except Exception:
            pass
    if gf_path.exists() and not no_gpt:
        try:
            gpt_filter_stats = json.load(open(gf_path))
        except Exception:
            pass

    # Fall back to scanning filesystem for GPT stats when batch file not present
    if not no_gpt and not gpt_filter_stats:
        gpt_filter_stats = gpt_rejection_totals(candor_root)

    by_type: dict[str, list] = defaultdict(list)
    for c in candidates:
        by_type[c.get('type', 'unknown')].append(c)

    lines = []

    def h(level, text):
        lines.append(f"\n{'#' * level} {text}\n")

    def row(*cells):
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    def sep(n):
        lines.append("| " + " | ".join(["---"] * n) + " |")

    mode_label = "Timing Only (no GPT)" if no_gpt else "Full Pipeline (Timing + GPT)"

    # ── Header ────────────────────────────────────────────────────────────────
    h(1, "Pause Extraction — Statistical Report")
    lines.append(f"> Dataset: CANDOR  \n"
                 f"> Mode: {mode_label}  \n"
                 f"> Results: `{results_dir}`  \n")

    # ── 1. Pipeline funnel ─────────────────────────────────────────────────────
    h(2, "1. Pipeline Funnel")
    row("Stage", "Count")
    sep(2)
    row("Total conversations", conv_counts["total"])
    if no_gpt:
        row("Processed (timing filter)", conv_counts["processed"])
        row("Conversations with ≥1 timing candidate", conv_counts["with_valid"])
    else:
        row("Processed (filter1 + GPT)", conv_counts["processed"])
        row("Conversations with ≥1 valid candidate", conv_counts["with_valid"])

    timing_total = sum(timing_tots.values())
    row("Timing-filter candidates (pre-GPT)", timing_total)
    for ptype in PAUSE_TYPES:
        row(f"  └ {ptype}", timing_tots[ptype])

    if not no_gpt:
        gpt_total = sum(
            len(json.load(open(p)))
            for part in sorted(candor_root.glob("part_*"))
            for conv in part.iterdir()
            for p in [conv / "pause_candidates" / "filter2_gpt" / "all_with_gpt_decisions.json"]
            if p.exists()
        )
        row("Total sent to GPT", gpt_total)
        row("GPT-validated candidates (final)", len(candidates))
        if gpt_total:
            row("GPT acceptance rate", f"{100 * len(candidates) / gpt_total:.1f}%")
    else:
        row("Total timing candidates (all types)", len(candidates))
    lines.append("")

    # ── 2. Timing filter rejection criteria ───────────────────────────────────
    if timing_filter_stats:
        h(2, "2. Timing Filter — Rejection Criteria")
        for cat in PAUSE_TYPES:
            ts = timing_filter_stats.get(cat)
            if not ts:
                continue
            passed = ts.get("passed", 0)
            total_checked = passed + sum(
                v for k, v in ts.items()
                if k.startswith("rej_")
            )
            h(3, f"{cat.replace('_', ' ').title()} "
                 f"(checked={total_checked}, passed={passed})")
            row("Rejection reason", "Count", "% of checked")
            sep(3)
            for key in sorted(ts, key=lambda k: -ts[k]):
                if not key.startswith("rej_"):
                    continue
                label = TIMING_LABELS.get(key, key)
                pct = f"{100 * ts[key] / total_checked:.1f}%" if total_checked else "—"
                row(label, ts[key], pct)
            lines.append("")
        sect_offset = 1
    else:
        sect_offset = 0

    # ── 3. GPT filter rejection criteria (full-pipeline only) ─────────────────
    if not no_gpt and gpt_filter_stats:
        h(2, f"{2 + sect_offset}. GPT Filter — Rejection Criteria")
        row("Category", "Sent", "Valid", "Rejected", "Acceptance %")
        sep(5)
        for cat in PAUSE_TYPES:
            gs = gpt_filter_stats.get(cat)
            if not gs:
                continue
            total_s = gs.get("total_sent", 0) if isinstance(gs, dict) else 0
            valid   = gs.get("valid", 0)      if isinstance(gs, dict) else 0
            rej     = gs.get("rejected", 0)   if isinstance(gs, dict) else 0
            acc     = f"{100 * valid / total_s:.1f}%" if total_s else "—"
            row(cat.replace("_", " ").title(), total_s, valid, rej, acc)
        lines.append("")

        for cat in PAUSE_TYPES:
            gs = gpt_filter_stats.get(cat)
            if not gs:
                continue
            criteria = gs.get("criteria") or {}
            if not criteria:
                continue
            rejected = gs.get("rejected", 0)
            h(3, f"{cat.replace('_', ' ').title()} — criteria breakdown")
            row("Criterion", "Failed count", "% of rejected")
            sep(3)
            for k, v in sorted(criteria.items(), key=lambda x: -x[1]):
                label = GPT_CRITERIA_LABELS.get(k, k)
                pct = f"{100 * v / rejected:.1f}%" if rejected else "—"
                row(label, v, pct)
            lines.append("")
        sect_offset += 1

    # ── Breakdown by type ──────────────────────────────────────────────────────
    h(2, f"{2 + sect_offset}. Breakdown by Pause Type")
    row("Type", "Count", "% of total")
    sep(3)
    for ptype in PAUSE_TYPES:
        n = len(by_type.get(ptype, []))
        pct = f"{100 * n / len(candidates):.1f}%" if candidates else "—"
        row(ptype.replace("_", " ").title(), n, pct)
    row("**Total**", f"**{len(candidates)}**", "")
    lines.append("")

    sect_offset += 1

    # ── Inter/Intra breakdown by pause count ──────────────────────────────────
    h(2, f"{2 + sect_offset}. Pause-Count Distribution (Inter & Intra Sentential)")
    row("Type", "n_pauses", "Count")
    sep(3)
    for ptype in ("inter_sentential", "intra_sentential"):
        by_n: dict[int, int] = defaultdict(int)
        for c in by_type.get(ptype, []):
            by_n[c.get("n_pauses", 1)] += 1
        for n in sorted(by_n):
            row(ptype.replace("_", " ").title(), n, by_n[n])
    lines.append("")

    sect_offset += 1

    # ── Timing statistics ──────────────────────────────────────────────────────
    h(2, f"{2 + sect_offset}. Timing Statistics")

    def stat_row(label, values, unit=""):
        s = stats(values)
        if s["n"] == 0:
            row(label, *["—"] * 6)
            return
        u = f" {unit}" if unit else ""
        row(label,
            f"{s['mean']}{u}", f"{s['median']}{u}", f"{s['std']}{u}",
            f"{s['min']}{u}", f"{s['max']}{u}", s["n"])

    row("Metric", "Mean", "Median", "Std", "Min", "Max", "N")
    sep(7)

    pause_durs_all = [p['duration'] for c in candidates for p in c.get('pauses', [])]
    clip_durs_all  = [float(c['clip_end']) - float(c['clip_start']) for c in candidates
                      if c.get('clip_start') and c.get('clip_end')]
    word_counts    = [int(c['n_words']) if 'n_words' in c
                      else len(c.get('full_transcript', '').split())
                      for c in candidates]
    n_pauses_all   = [c.get('n_pauses', 0) for c in candidates if c.get('type') != 'floor_holding']

    stat_row("Clip duration (all types)", clip_durs_all, "s")
    stat_row("Pause duration (inter + intra)", pause_durs_all, "s")
    stat_row("Pauses per clip (inter + intra)", n_pauses_all)
    lines.append("")

    sect_offset += 1

    # ── Per-type timing breakdown ──────────────────────────────────────────────
    h(2, f"{2 + sect_offset}. Timing by Pause Type")
    for ptype in PAUSE_TYPES:
        subset = by_type.get(ptype, [])
        h(3, f"{ptype.replace('_', ' ').title()} (n={len(subset)})")
        if not subset:
            lines.append("_No candidates._\n")
            continue
        row("Metric", "Mean", "Median", "Std", "Min", "Max")
        sep(6)

        def srow(label, values, unit=""):
            s = stats(values)
            if s["n"] == 0:
                row(label, *["—"] * 5)
                return
            u = f" {unit}" if unit else ""
            row(label,
                f"{s['mean']}{u}", f"{s['median']}{u}", f"{s['std']}{u}",
                f"{s['min']}{u}", f"{s['max']}{u}")

        srow("Clip duration",
             [float(c['clip_end']) - float(c['clip_start']) for c in subset], "s")
        if ptype != 'floor_holding':
            pause_durs = [p['duration'] for c in subset for p in c.get('pauses', [])]
            n_pauses   = [c.get('n_pauses', 1) for c in subset]
            srow("Pause duration", pause_durs, "s")
            srow("Pauses per clip", n_pauses)
        else:
            srow("Word count", [c.get('n_words', 0) for c in subset], "w")
        lines.append("")

    sect_offset += 1

    # ── Per-part summary ────────────────────────────────────────────────────────
    h(2, f"{2 + sect_offset}. Per-Part Summary")
    by_part: dict[str, list] = defaultdict(list)
    for c in candidates:
        conv_id = c.get("conv_id", "")
        for part_dir in sorted(candor_root.glob("part_*")):
            if (part_dir / conv_id).is_dir():
                by_part[part_dir.name].append(c)
                break

    row("Part", "Conversations", "Inter", "Intra", "Floor", "Total")
    sep(6)
    for part_dir in sorted(candor_root.glob("part_*")):
        n_convs = sum(1 for d in part_dir.iterdir() if d.is_dir())
        subset  = by_part.get(part_dir.name, [])
        counts  = {t: sum(1 for c in subset if c.get('type') == t) for t in PAUSE_TYPES}
        row(part_dir.name, n_convs,
            counts['inter_sentential'], counts['intra_sentential'],
            counts['floor_holding'], len(subset))
    lines.append("")

    sect_offset += 1

    # ── Example candidates (or all valid for full run) ─────────────────────────
    if no_gpt:
        h(2, f"{2 + sect_offset}. Sample Timing Candidates (first 50)")
        row("Conv ID", "Type", "n_pauses", "Clip dur (s)", "Pause dur (s)")
        sep(5)
        for c in candidates[:50]:
            pause_durs_c = [p['duration'] for p in c.get('pauses', [])]
            avg_pause    = round(statistics.mean(pause_durs_c), 3) if pause_durs_c else "—"
            clip_dur     = round(float(c['clip_end']) - float(c['clip_start']), 2) \
                           if c.get('clip_start') and c.get('clip_end') else "—"
            row(
                c.get('conv_id', '')[:8] + "…",
                c.get('type', '').replace('_', ' '),
                c.get('n_pauses', '—'),
                clip_dur,
                avg_pause,
            )
    else:
        h(2, f"{2 + sect_offset}. All Valid Candidates")
        row("Conv ID", "Type", "n_pauses", "Clip dur (s)", "Pause dur (s)", "GPT reason")
        sep(6)
        for c in candidates:
            pause_durs_c = [p['duration'] for p in c.get('pauses', [])]
            avg_pause    = round(statistics.mean(pause_durs_c), 3) if pause_durs_c else "—"
            clip_dur     = round(float(c['clip_end']) - float(c['clip_start']), 2) \
                           if c.get('clip_start') and c.get('clip_end') else "—"
            reason = c.get('gpt_reason', '')
            if len(reason) > 80:
                reason = reason[:77] + "..."
            row(
                c.get('conv_id', '')[:8] + "…",
                c.get('type', '').replace('_', ' '),
                c.get('n_pauses', '—'),
                clip_dur,
                avg_pause,
                reason,
            )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written → {out_path}")
    print(f"  Total {'timing' if no_gpt else 'valid'} candidates: {len(candidates)}")
    for ptype in PAUSE_TYPES:
        print(f"    {ptype}: {len(by_type.get(ptype, []))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--candor_root", type=Path, default=CANDOR_ROOT)
    parser.add_argument("--out",         type=Path, default=None,
                        help="Output path (default: <results_dir>/report.md)")
    parser.add_argument("--no_gpt",      action="store_true",
                        help="Generate a timing-only report (no GPT section)")
    args = parser.parse_args()

    if args.out is None:
        args.out = args.results_dir / "report.md"

    generate_report(args.results_dir, args.candor_root, args.out, no_gpt=args.no_gpt)
