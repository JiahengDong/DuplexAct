#!/usr/bin/env python3
"""
Statistical report generator for CANDOR pause candidate results.

Reads all filter2_gpt/*.json files across all processed conversations
and outputs a detailed statistical summary (console + JSON).

Usage:
  python generate_stats.py
  python generate_stats.py --output stats_report.json
  python generate_stats.py --parts 1 2 3
"""

import argparse
import json
import math
from pathlib import Path

CANDOR_ROOT = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")


def find_result_files(parts: list[int] | None = None) -> list[Path]:
    files = []
    for part_dir in sorted(CANDOR_ROOT.glob("part_*")):
        if parts:
            part_num = int(part_dir.name.split("_")[1])
            if part_num not in parts:
                continue
        for conv_dir in sorted(part_dir.iterdir()):
            f = conv_dir / "pause_candidates" / "filter2_gpt" / "all_with_gpt_decisions.json"
            if f.exists():
                files.append(f)
    return files


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None,
                "min": None, "max": None, "std": None}
    n    = len(values)
    mean = sum(values) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    sv   = sorted(values)
    med  = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return {
        "count":  n,
        "mean":   round(mean, 3),
        "median": round(med, 3),
        "min":    round(min(values), 3),
        "max":    round(max(values), 3),
        "std":    round(std, 3),
    }


def analyze(candidates: list[dict]) -> dict:
    durations     = [c["clip_end"] - c["clip_start"] for c in candidates if "clip_start" in c and "clip_end" in c]
    word_counts   = [c["n_words"] for c in candidates if "n_words" in c]
    n_pauses_list = [c["n_pauses"] for c in candidates if "n_pauses" in c]

    pause_durations = []
    for c in candidates:
        for p in c.get("pauses", []):
            pause_durations.append(p["duration"])

    pause_count_dist: dict[int, int] = {}
    for n in n_pauses_list:
        pause_count_dist[n] = pause_count_dist.get(n, 0) + 1

    return {
        "total_valid": len(candidates),
        "clip_duration_s":    stats(durations),
        "word_count":         stats(word_counts),
        "n_pauses_per_clip":  stats(n_pauses_list),
        "pause_duration_s":   stats(pause_durations),
        "pause_count_distribution": dict(sorted(pause_count_dist.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate statistics for pause candidate results.")
    parser.add_argument("--output", type=Path, default=None, help="Save JSON report to file")
    parser.add_argument("--parts",  type=int, nargs="+",    help="Only include specific part numbers")
    args = parser.parse_args()

    files = find_result_files(args.parts)
    print(f"Found {len(files)} processed conversations\n")

    if not files:
        print("No processed conversations found. Run batch_process.py first.")
        return

    all_candidates: list[dict] = []
    conv_summaries: list[dict] = []
    parts_seen: dict[str, list] = {}

    for f in files:
        with open(f) as fp:
            data = json.load(fp)

        valid = [c for c in data if c.get("gpt_valid")]
        all_candidates.extend(valid)

        part = f.parts[f.parts.index("Candor") + 1]  # e.g. "part_001"
        parts_seen.setdefault(part, []).extend(valid)

        by_type: dict[str, int] = {}
        for c in valid:
            t = c.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        conv_summaries.append({
            "conv_id":    f.parent.parent.name,
            "part":       part,
            "total_valid": len(valid),
            "by_type":    by_type,
        })

    # ── Per-type breakdown ─────────────────────────────────────────────────────
    by_type_all: dict[str, list] = {
        "inter_sentential": [],
        "intra_sentential": [],
        "floor_holding":    [],
    }
    for c in all_candidates:
        t = c.get("type")
        if t in by_type_all:
            by_type_all[t].append(c)

    # ── Build report ───────────────────────────────────────────────────────────
    report = {
        "overview": {
            "conversations_processed": len(files),
            "total_valid_clips":       len(all_candidates),
            "by_type": {t: len(v) for t, v in by_type_all.items()},
        },
        "per_type_stats": {t: analyze(v) for t, v in by_type_all.items()},
        "per_part_summary": {},
        "top_conversations_by_yield": [],
    }

    # Per-part
    for part, cands in sorted(parts_seen.items()):
        part_by_type = {}
        for c in cands:
            t = c.get("type", "unknown")
            part_by_type[t] = part_by_type.get(t, 0) + 1
        report["per_part_summary"][part] = {
            "total_valid": len(cands),
            "by_type": part_by_type,
        }

    # Top 10 conversations by yield
    top = sorted(conv_summaries, key=lambda x: x["total_valid"], reverse=True)[:10]
    report["top_conversations_by_yield"] = top

    # ── Print report ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("CANDOR PAUSE CANDIDATE — STATISTICAL REPORT")
    print("=" * 60)

    ov = report["overview"]
    print(f"\n── Overview ──")
    print(f"  Conversations processed : {ov['conversations_processed']}")
    print(f"  Total valid clips       : {ov['total_valid_clips']}")
    print(f"  By type:")
    for t, n in ov["by_type"].items():
        print(f"    {t:25s}: {n}")

    for t, s in report["per_type_stats"].items():
        if s["total_valid"] == 0:
            continue
        print(f"\n── {t} (n={s['total_valid']}) ──")

        def fmt(d):
            if d["count"] == 0:
                return "  (no data)"
            return (f"  mean={d['mean']}  median={d['median']}  "
                    f"min={d['min']}  max={d['max']}  std={d['std']}")

        print(f"  Clip duration (s)   :{fmt(s['clip_duration_s'])}")
        if s["word_count"]["count"] > 0:
            print(f"  Word count          :{fmt(s['word_count'])}")
        if s["n_pauses_per_clip"]["count"] > 0:
            print(f"  Pauses per clip     :{fmt(s['n_pauses_per_clip'])}")
            dist = s["pause_count_distribution"]
            print(f"  Pause count dist    :  " + "  ".join(f"n{k}={v}" for k, v in dist.items()))
        if s["pause_duration_s"]["count"] > 0:
            print(f"  Pause duration (s)  :{fmt(s['pause_duration_s'])}")

    print(f"\n── Per-part summary ──")
    print(f"  {'Part':<12} {'Total':>7}  {'inter':>7}  {'floor':>7}  {'intra':>7}")
    for part, ps in sorted(report["per_part_summary"].items()):
        bt = ps["by_type"]
        print(f"  {part:<12} {ps['total_valid']:>7}  "
              f"{bt.get('inter_sentential',0):>7}  "
              f"{bt.get('floor_holding',0):>7}  "
              f"{bt.get('intra_sentential',0):>7}")

    print(f"\n── Top 10 conversations by yield ──")
    for r in report["top_conversations_by_yield"]:
        bt = r["by_type"]
        print(f"  {r['conv_id']}  [{r['part']}]  "
              f"total={r['total_valid']}  "
              f"inter={bt.get('inter_sentential',0)}  "
              f"floor={bt.get('floor_holding',0)}  "
              f"intra={bt.get('intra_sentential',0)}")

    print()

    if args.output:
        with open(args.output, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"Report saved → {args.output}")


if __name__ == "__main__":
    main()
