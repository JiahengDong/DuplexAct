#!/usr/bin/env python3
"""Backchannel Immediate under the two-stage definition.

The automatic rule flags a backchannel whenever a segment of at most three
words starts inside the user's turn. That also accepts the truncated opening of
a full turn, segments with no intelligible speech, and short strings with no
listener-feedback function, and how often it does so varies strongly by model.

Immediate therefore requires both halves of the action: the model produced a
backchannel at all (stage 1, screen), and it carried the function the
condition asked for (stage 2, function match).
`evaluation/direct_replay/metrics/bc_two_stage_judge.py` writes those verdicts;
this module joins them onto the cache so Immediate, Subsequent and Timing all
read one population. `bc_rule_immediate` keeps the rule-only value for the
decomposition table.

Verdicts only cover the rule's hits, so Subsequent needs no re-judging.
"""
from __future__ import annotations

import json
from pathlib import Path

VERDICTS = Path(__file__).resolve().parent / "BC_TWO_STAGE.json"

_BY_KEY: dict[tuple[str, str], dict] | None = None
_MISSING: set[tuple[str, str]] = set()
_UNJUDGED: set[tuple[str, str]] = set()


def _by_key() -> dict[tuple[str, str], dict]:
    global _BY_KEY
    if _BY_KEY is None:
        rows = json.loads(VERDICTS.read_text())
        _BY_KEY = {(r["model"], r["sample_id"]): r for r in rows}
    return _BY_KEY


def apply(rec: dict) -> dict:
    """Rewrite a backchannel record's Immediate in place. Other actions pass."""
    if rec.get("action") != "backchannel":
        return rec

    rule = rec.get("immediate")
    rec["bc_rule_immediate"] = rule
    rec["bc_screened"] = None
    rec["bc_function_label"] = None

    if rule is None:
        return rec
    if rule != 1:
        # Never a candidate, so nothing to screen and no function to match.
        rec["bc_screened"] = 0
        return rec

    v = _by_key().get((rec.get("model"), rec.get("sample_id")))
    if v is None:
        # Verdicts do not cover this record. Drop it from the denominator
        # rather than score it zero, and let the writer refuse to publish.
        _MISSING.add((rec.get("model"), rec.get("sample_id")))
        rec["immediate"] = None
        rec["immediate_tag"] = "bc_verdict_missing"
        return rec

    rec["bc_screened"] = int(bool(v.get("screen")))
    rec["bc_function_label"] = v.get("function_label")

    if v.get("screen") and v.get("function_match") is None:
        # Screened in but stage 2 returned no label (usually because output.wav
        # was unavailable). Scoring it 0 would count an unjudgeable sample as a
        # function mismatch, so withhold it like a missing verdict.
        _UNJUDGED.add((rec.get("model"), rec.get("sample_id")))
        rec["immediate"] = None
        rec["immediate_tag"] = "bc_function_unjudged"
        return rec

    rec["immediate"] = int(bool(v.get("screen")) and v.get("function_match") == 1)
    if not rec["immediate"]:
        rec["immediate_tag"] = (
            "bc_not_a_backchannel" if not v.get("screen") else "bc_function_mismatch"
        )
    return rec


def assert_complete() -> None:
    """Refuse to write a report off verdicts that do not cover the cache."""
    if _MISSING:
        sample = sorted(_MISSING)[:5]
        raise SystemExit(
            f"{len(_MISSING)} backchannel records have no two-stage verdict "
            f"(e.g. {sample}). Re-run evaluation/direct_replay/metrics/bc_two_stage_judge.py."
        )
    if _UNJUDGED:
        sample = sorted(_UNJUDGED)[:5]
        raise SystemExit(
            f"{len(_UNJUDGED)} backchannel records passed stage 1 but carry no "
            f"stage 2 label (e.g. {sample}), usually an unreachable output.wav. "
            "Re-run bc_two_stage_judge.py where the audio lives, then --merge "
            "the shard; do not publish these as zeros."
        )
