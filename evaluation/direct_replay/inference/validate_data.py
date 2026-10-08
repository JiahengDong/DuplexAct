#!/usr/bin/env python3
"""
direct_replay/inference/validate_data.py
=========================================
Pre-flight check: scans every sample under data/<action>/<l3_dir>/ and
flags which ones are missing the specific metadata.json timestamp fields
that evaluation/actions/*.py actually reads for B_i/T_i (see each
module's own "Timestamp keys read from EvalInstance.timestamps"
docstring section -- this script mirrors those exactly, not a guess).

Only checks metadata/audio-file completeness (i.e. "is this sample even
runnable/scoreable at all") -- does NOT touch inference_info.json/asr.json,
since those don't exist until after inference/ASR actually run.

Usage:
    python validate_data.py                      # full report, all actions
    python validate_data.py --action wait         # one action only
    python validate_data.py --json report.json    # also write machine-readable report
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def _duplexact_root() -> Path:
    import os
    for key in ("DUPLEXACT_ROOT", "BENCH_ROOT"):
        v = os.environ.get(key)
        if v:
            return Path(v).resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "evaluation" / "paths.py").exists():
            return p
    return here.parents[min(3, len(here.parents)-1)]

DATA_ROOT = _duplexact_root() / "evaluation" / "direct_replay" / "data"

# Required (non-None) metadata.json fields per (action, l3_dir), mirroring
# build_eval_instance.py's _build_* functions exactly.
REQUIRED_FIELDS: dict[tuple[str, str], list[str]] = {
    ("wait", "inter_sentential"):  ["pause_start", "pause_end", "clip_end"],
    ("wait", "intra_sentential"):  ["pause_start", "pause_end", "clip_end"],
    ("wait", "floor_holding"):     ["clip_end"],  # clip_start defaults to 0.0 if absent
    ("backchannel", "backchannel_affirmative"): ["bc_start"],
    ("backchannel", "backchannel_uncertainty"): ["bc_start"],
    ("ignore", "ignore_background_speech_during_user_speech"):
        ["bg_speech_start_s", "bg_speech_end_s", "speaker_a_duration"],
    ("ignore", "ignore_other_directed_speech_during_user_speech"):
        ["side_speech_start_s", "side_speech_end_s", "speaker_a_duration"],
    ("turn_taking", "take_turn_immediately"):  ["t1_end"],
    ("turn_taking", "take_turn_with_delay"):   ["t1_end"],
    ("turn_taking", "take_turn_with_overlap"): ["t1_end"],
}

ACTIONS = ["wait", "backchannel", "ignore", "turn_taking"]


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def validate_sample(action: str, l3_dir: str, sample_dir: Path) -> tuple[str, list[str]]:
    """Returns (status, problems) where status in {"ok", "bad"}."""
    problems: list[str] = []

    wav_path = sample_dir / "input.wav"
    meta_path = sample_dir / "metadata.json"

    if not wav_path.exists():
        problems.append("missing_input_wav")
    else:
        dur = _wav_duration(wav_path)
        if dur is None:
            problems.append("unreadable_input_wav")
        elif dur <= 0.05:
            problems.append(f"degenerate_input_wav:{dur:.3f}s")

    if not meta_path.exists():
        problems.append("missing_metadata_json")
        return "bad", problems

    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        problems.append(f"unparseable_metadata_json:{type(e).__name__}")
        return "bad", problems

    if "input_duration" not in meta or meta.get("input_duration") is None:
        problems.append("missing_field:input_duration")

    required = REQUIRED_FIELDS.get((action, l3_dir), [])
    for field in required:
        if meta.get(field) is None:
            problems.append(f"missing_field:{field}")

    # Sanity checks on ordering/validity, only if the fields are present.
    if action == "wait" and l3_dir in ("inter_sentential", "intra_sentential"):
        ps, pe = meta.get("pause_start"), meta.get("pause_end")
        if ps is not None and pe is not None and pe <= ps:
            problems.append(f"invalid_range:pause_end({pe})<=pause_start({ps})")
    if action == "backchannel":
        bs, be = meta.get("bc_start"), meta.get("bc_end")
        if bs is not None and be is not None and be <= bs:
            problems.append(f"invalid_range:bc_end({be})<=bc_start({bs})")
    if action == "ignore":
        if l3_dir == "ignore_background_speech_during_user_speech":
            s, e = meta.get("bg_speech_start_s"), meta.get("bg_speech_end_s")
        else:
            s, e = meta.get("side_speech_start_s"), meta.get("side_speech_end_s")
        if s is not None and e is not None and e <= s:
            problems.append(f"invalid_range:other_end({e})<=other_start({s})")

    return ("bad" if problems else "ok"), problems


def run(action_filter: str | None, verbose: bool) -> dict:
    report: dict = {"actions": {}, "summary": {}}
    total_ok = total_bad = 0

    actions = [action_filter] if action_filter else ACTIONS
    for action in actions:
        action_dir = DATA_ROOT / action
        if not action_dir.is_dir():
            continue
        report["actions"][action] = {}
        for l3_dir_path in sorted(action_dir.iterdir()):
            if not l3_dir_path.is_dir():
                continue
            l3 = l3_dir_path.name
            samples = sorted(p for p in l3_dir_path.iterdir() if p.is_dir())
            ok_ids: list[str] = []
            bad: dict[str, list[str]] = {}
            for sample_dir in samples:
                status, problems = validate_sample(action, l3, sample_dir)
                if status == "ok":
                    ok_ids.append(sample_dir.name)
                else:
                    bad[sample_dir.name] = problems

            total_ok += len(ok_ids)
            total_bad += len(bad)
            report["actions"][action][l3] = {
                "n_total": len(samples),
                "n_ok": len(ok_ids),
                "n_bad": len(bad),
                "ok_sample_ids": ok_ids,
                "bad_samples": bad,
            }
            print(f"[{action}/{l3}] {len(ok_ids)}/{len(samples)} runnable"
                  + (f"  ({len(bad)} bad)" if bad else ""))
            if verbose and bad:
                for sid, probs in bad.items():
                    print(f"    - {sid}: {', '.join(probs)}")

    report["summary"] = {"total_ok": total_ok, "total_bad": total_bad}
    print(f"\n[validate_data] TOTAL: {total_ok} runnable, {total_bad} bad "
          f"(out of {total_ok + total_bad})")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", default=None, choices=ACTIONS)
    parser.add_argument("--json", type=Path, default=None, help="Write full machine-readable report here.")
    parser.add_argument("--verbose", action="store_true", help="List every bad sample's specific problems.")
    args = parser.parse_args()

    report = run(args.action, args.verbose)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"[validate_data] full report -> {args.json}")


if __name__ == "__main__":
    main()
