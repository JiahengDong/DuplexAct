"""
Orchestrator for the Full-Duplex Level-2 Rubric evaluation.

Evaluation chain (Section 2.1 of rubric):
    1. Score each instance independently → InstanceRecord
    2. Aggregate InstanceRecords per Level-3 condition → L3Result
    3. Macro-average Level-3 results per Level-2 action → L2Result

Usage
-----
    python evaluate.py --instances instances.jsonl --output results/

Instance JSON format (one per line):
    {
        "instance_id": "...",
        "l2_action":   "smooth_turn_taking",
        "l3_condition": "take_turn_immediately",
        "rubric_version": "1.0",
        "timestamps": {"t_user_end": 4.2, ...},
        "model_speech_segments": [[3.9, 5.1], ...],
        "transcript_user": "...",
        "transcript_model": "...",
        "transcript_context": "...",
        "audio_paths": {},
        "metadata": {}
    }

Judge integration
-----------------
By default, `--judge none` leaves all reliability items as 0 and marks
judge_source="pending".  Supply --judge llm or --judge human to route
instances to the configured LLM or human annotation pipeline.
(LLM and human judge wiring is a stub; implement in `_call_judge`.)
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .models import EvalInstance, InstanceRecord, L2Result, L3Result
from .actions import (
    backchannel,
    hold_floor,
    ignore,
    interrupt_user,
    respond_and_recover,
    smooth_turn_taking,
    stop,
    wait,
)

# ─── Action module registry ──────────────────────────────────────────────────

ACTION_MODULES = {
    "smooth_turn_taking":  smooth_turn_taking,
    "interrupt_user":      interrupt_user,
    "respond_and_recover": respond_and_recover,
    "hold_floor":          hold_floor,
    "ignore":              ignore,
    "wait":                wait,
    "stop":                stop,
    "backchannel":         backchannel,
}


# ─── Judge stub ──────────────────────────────────────────────────────────────

def _call_judge(
    instance: EvalInstance,
    prompts: list[dict],
    judge_mode: str,
) -> tuple[list[int], str]:
    """
    Route instance + prompts to the appropriate judge.

    Returns:
        (judge_outputs, judge_source_label)
        judge_outputs: list of 0/1 aligned to prompts
    """
    if judge_mode == "none":
        return [0] * len(prompts), "pending"

    if judge_mode == "llm":
        # TODO: implement LLM judge call (e.g. OpenAI, Gemini, or local model)
        # Inputs per criterion: prompts[k]["question"] + evidence from instance
        # Parse binary output per criterion
        raise NotImplementedError("LLM judge not yet wired up. Use --judge none.")

    if judge_mode == "human":
        # TODO: implement human annotation interface or batch job
        raise NotImplementedError("Human judge not yet wired up. Use --judge none.")

    raise ValueError(f"Unknown judge mode: {judge_mode!r}")


# ─── Instance loader ─────────────────────────────────────────────────────────

def load_instances(path: Path) -> list[EvalInstance]:
    """Load EvalInstances from a JSONL file."""
    instances: list[EvalInstance] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                inst = EvalInstance(
                    instance_id=d["instance_id"],
                    l2_action=d["l2_action"],
                    l3_condition=d["l3_condition"],
                    rubric_version=d.get("rubric_version", "1.0"),
                    timestamps=d.get("timestamps", {}),
                    model_speech_segments=[tuple(s) for s in d.get("model_speech_segments", [])],
                    transcript_user=d.get("transcript_user"),
                    transcript_model=d.get("transcript_model"),
                    transcript_context=d.get("transcript_context"),
                    transcript_original_plan=d.get("transcript_original_plan"),
                    audio_paths=d.get("audio_paths", {}),
                    metadata=d.get("metadata", {}),
                )
                instances.append(inst)
            except (KeyError, TypeError) as e:
                print(f"WARNING: line {lineno} skipped: {e}")
    return instances


# ─── Core evaluation logic ───────────────────────────────────────────────────

def score_instance(instance: EvalInstance, judge_mode: str = "none") -> InstanceRecord:
    """Score one instance via the appropriate action module."""
    module = ACTION_MODULES.get(instance.l2_action)
    if module is None:
        raise ValueError(f"No action module for l2_action={instance.l2_action!r}")

    prompts = module.build_judge_prompts(instance)
    judge_outputs, judge_source = _call_judge(instance, prompts, judge_mode)

    record = module.score_instance(instance, judge_outputs)
    record.judge_source = judge_source
    return record


def aggregate_l3(
    records: list[InstanceRecord],
    l2_action: str,
    l3_condition: str,
) -> L3Result:
    """Aggregate all InstanceRecords for one Level-3 condition."""
    if not records:
        raise ValueError(f"No records for {l2_action}/{l3_condition}")

    B_vals = [r.B_i for r in records]
    R_vals = [r.R_i for r in records]

    # Timing: collect T_i_raw where available
    t_vals = [r.T_i_raw for r in records if r.T_i_raw is not None]
    timing_median: Optional[float] = statistics.median(t_vals) if t_vals else None
    timing_iqr: Optional[tuple[float, float]] = None
    if len(t_vals) >= 4:
        t_sorted = sorted(t_vals)
        n = len(t_sorted)
        q1 = statistics.median(t_sorted[: n // 2])
        q3 = statistics.median(t_sorted[(n + 1) // 2 :])
        timing_iqr = (q1, q3)

    # Criterion means
    n_criteria = len(records[0].criterion_labels)
    criterion_means: list[float] = []
    for k in range(n_criteria):
        applicable = [
            r.e_i[k]
            for r in records
            if k < len(r.e_i) and k < len(r.applicability_mask) and r.applicability_mask[k]
        ]
        criterion_means.append(sum(applicable) / len(applicable) if applicable else float("nan"))

    coverage = sum(1 for r in records if r.event_time is not None) / len(records)

    return L3Result(
        l2_action=l2_action,
        l3_condition=l3_condition,
        n_instances=len(records),
        B_c=sum(B_vals) / len(B_vals),
        R_c=sum(R_vals) / len(R_vals),
        timing_median=timing_median,
        timing_iqr=timing_iqr,
        criterion_means=criterion_means,
        criterion_labels=records[0].criterion_labels,
        coverage=coverage,
    )


def aggregate_l2(l3_results: list[L3Result], l2_action: str) -> L2Result:
    """
    Macro-average Level-3 results for one Level-2 action (Section 4.2).
    Each condition contributes equally by default.
    """
    B_l2 = sum(r.B_c for r in l3_results) / len(l3_results)
    R_l2 = sum(r.R_c for r in l3_results) / len(l3_results)

    # Combined timing summary from all L3 children
    all_timing = [
        r.timing_median for r in l3_results if r.timing_median is not None
    ]
    timing_summary: Optional[dict] = None
    if all_timing:
        timing_summary = {
            "overall_median_s": statistics.median(all_timing),
            "by_condition":     {r.l3_condition: r.timing_median for r in l3_results},
        }

    return L2Result(
        l2_action=l2_action,
        B_l2=B_l2,
        R_l2=R_l2,
        timing_summary=timing_summary,
        children=l3_results,
    )


# ─── Top-level evaluation pipeline ──────────────────────────────────────────

def evaluate(
    instances: list[EvalInstance],
    judge_mode: str = "none",
) -> tuple[list[InstanceRecord], list[L3Result], list[L2Result]]:
    """
    Run the full three-stage evaluation pipeline.

    Returns:
        (instance_records, l3_results, l2_results)
    """
    # Stage 1: score all instances
    instance_records: list[InstanceRecord] = []
    for inst in instances:
        try:
            rec = score_instance(inst, judge_mode=judge_mode)
            instance_records.append(rec)
        except Exception as exc:
            print(f"ERROR scoring {inst.instance_id}: {exc}")

    # Stage 2: group by (l2_action, l3_condition) and aggregate to L3
    from collections import defaultdict
    groups: dict[tuple[str, str], list[InstanceRecord]] = defaultdict(list)
    for rec in instance_records:
        groups[(rec.l2_action, rec.l3_condition)].append(rec)

    l3_results: list[L3Result] = []
    for (l2, l3), recs in sorted(groups.items()):
        try:
            l3_results.append(aggregate_l3(recs, l2, l3))
        except Exception as exc:
            print(f"ERROR aggregating L3 {l2}/{l3}: {exc}")

    # Stage 3: group L3 results by l2_action and aggregate to L2
    l3_by_action: dict[str, list[L3Result]] = defaultdict(list)
    for r in l3_results:
        l3_by_action[r.l2_action].append(r)

    l2_results: list[L2Result] = []
    for l2_action, children in sorted(l3_by_action.items()):
        try:
            l2_results.append(aggregate_l2(children, l2_action))
        except Exception as exc:
            print(f"ERROR aggregating L2 {l2_action}: {exc}")

    return instance_records, l3_results, l2_results


# ─── Output helpers ──────────────────────────────────────────────────────────

def save_results(
    instance_records: list[InstanceRecord],
    l3_results: list[L3Result],
    l2_results: list[L2Result],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "instance_records.jsonl", "w") as f:
        for r in instance_records:
            f.write(json.dumps(asdict(r)) + "\n")

    with open(output_dir / "l3_results.jsonl", "w") as f:
        for r in l3_results:
            f.write(json.dumps(asdict(r)) + "\n")

    with open(output_dir / "l2_results.jsonl", "w") as f:
        for r in l2_results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Human-readable summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FULL-DUPLEX LEVEL-2 RUBRIC EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        for l2 in l2_results:
            f.write(f"[{l2.l2_action.upper()}]\n")
            f.write(f"  B_l2 (macro): {l2.B_l2:.3f}\n")
            f.write(f"  R_l2 (macro): {l2.R_l2:.3f}\n")
            if l2.timing_summary:
                f.write(f"  Timing (median): {l2.timing_summary['overall_median_s']*1000:.0f}ms\n")
            f.write("  Children:\n")
            for c in l2.children:
                t_str = f"{c.timing_median*1000:.0f}ms" if c.timing_median else "N/A"
                f.write(
                    f"    {c.l3_condition:<35}  n={c.n_instances:4d}  "
                    f"B={c.B_c:.3f}  R={c.R_c:.3f}  cov={c.coverage:.2f}  T={t_str}\n"
                )
            f.write("\n")

    print(f"Results written to {output_dir}/")


def print_summary(l2_results: list[L2Result]) -> None:
    print()
    print("=" * 70)
    print("LEVEL-2 RESULTS")
    print("-" * 70)
    print(f"{'Action':<30}  {'B_l2':>6}  {'R_l2':>6}")
    print("-" * 70)
    for r in l2_results:
        print(f"{r.l2_action:<30}  {r.B_l2:>6.3f}  {r.R_l2:>6.3f}")
    print("=" * 70)


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Full-Duplex Level-2 Rubric evaluation pipeline."
    )
    parser.add_argument(
        "--instances", required=True, type=Path,
        help="Path to JSONL file of EvalInstance records.",
    )
    parser.add_argument(
        "--output", default="results", type=Path,
        help="Directory for output files (default: results/).",
    )
    parser.add_argument(
        "--judge", default="none", choices=["none", "llm", "human"],
        help="Judge mode (default: none; reliability items remain 0 / pending).",
    )
    args = parser.parse_args()

    print(f"Loading instances from {args.instances} ...")
    instances = load_instances(args.instances)
    print(f"  Loaded {len(instances)} instances.")

    print(f"Running evaluation (judge={args.judge}) ...")
    instance_records, l3_results, l2_results = evaluate(instances, judge_mode=args.judge)

    print_summary(l2_results)
    save_results(instance_records, l3_results, l2_results, args.output)


if __name__ == "__main__":
    main()
