#!/usr/bin/env python3
"""Backchannel Immediate under the two-stage definition.

The automatic rule confirms a backchannel whenever a segment of at most three
words starts inside the user's turn. Screening those candidates showed the rule
is wrong about 44% of the time: it keeps the truncated opening of a full turn
(`Yeah` followed 0.04 s later by `Yeah that's` / `that's totally
understandable`), segments with no intelligible speech, and short strings that
carry no listener-feedback function. Worse for ranking, the error rate is not a
constant offset -- survival runs from 4% to 97% across models -- so the rule
reorders the table rather than shifting it.

Immediate therefore now means both halves of the action: the model produced a
backchannel at all, and it carried the function the condition asked for.
`evaluation/direct_replay/metrics/bc_two_stage_judge.py` writes those two verdicts; this
module joins them onto the cache so Immediate, Subsequent and Timing all read
one population. `bc_rule_immediate` keeps the old rule-only value for the
decomposition table.

Verdicts are a strict subset of the rule's hits, so Subsequent needs no
re-judging: every record in the new population was already scored.
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
        # Stale verdicts against a re-run cache. Drop it from the denominator
        # rather than score it zero, and let the writer refuse to publish.
        _MISSING.add((rec.get("model"), rec.get("sample_id")))
        rec["immediate"] = None
        rec["immediate_tag"] = "bc_verdict_missing"
        return rec

    rec["bc_screened"] = int(bool(v.get("screen")))
    rec["bc_function_label"] = v.get("function_label")

    if v.get("screen") and v.get("function_match") is None:
        # Screened in but stage 2 never returned a label -- the usual cause is
        # output.wav being unreachable (vita-1.5's audio only ever existed on
        # the pod). Scoring it 0 would file an unjudgeable sample as a function
        # mismatch, so withhold it the same way a missing verdict is withheld.
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
