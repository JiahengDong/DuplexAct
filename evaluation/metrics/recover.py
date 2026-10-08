"""Recover: re-enter after interruption and adapt as required."""
from __future__ import annotations

from typing import Optional

from .framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)
from .take_turn import score_immediate_takeover
from .wait import BACKCHANNEL_MAX_WORDS

ACTION_ID = "recover"
DATA_ACTION = "respond_and_recover"

# Anchor: user-interruption offset (t_interruption_end). Immediate is TOR
# in this observation window, shared by every Recover condition.
# Recover is not a take-turn test, so the window is the Take turn
# observation span [-2, +5], not the tight take-turn success windows.
WINDOW_ID = "minus2_plus5"
RECOVERY_LO_S = -2.0
RECOVERY_HI_S = 5.0

DEFINITION = (
    "After an interruption, the model should re-enter the interaction "
    "within the recovery window and adapt its next contribution to what "
    "the interruption required."
)
IMMEDIATE_DEFINITION = (
    "Start a substantive recovery turn inside the observation window "
    "around interruption end."
)
IMMEDIATE_CRITERION = (
    "A valid takeover onset (backchannel excluded) falls in "
    f"[{RECOVERY_LO_S:.1f}, {RECOVERY_HI_S:.1f}] s relative to "
    "t_interruption_end. Any in-window onset counts; it need not be the "
    "first takeover. No in-window takeover is a fail."
)
IMMEDIATE_NOTES = (
    "t = 0 is interruption offset (t_interruption_end). Observation "
    f"window [{RECOVERY_LO_S:.1f}, {RECOVERY_HI_S:.1f}] s is also the "
    "Immediate success window (TOR). All six conditions share it. "
    "Immediate = at least one valid takeover onset in that window, not "
    "necessarily the first. Headline reporting is among speaking-at-cue "
    "(forced allowed); cue in silence or after the model finished is "
    "Take turn, not Recover. Takeover = substantive floor-claiming speech; "
    f"backchannel (≤ {BACKCHANNEL_MAX_WORDS} words on the whole segment) "
    "is skipped. Window bounds are inclusive. Subsequent is gated on "
    "Immediate: only the in-window response is judged for adaptation."
)

IMMEDIATE_CHECKS = (
    Check(
        id="timing_conditioned_takeover",
        name="Take-over rate",
        pass_when=(
            "A valid takeover onset falls inside the observation window."
        ),
        fail_when=(
            "No valid takeover onset falls inside the observation window."
        ),
    ),
)

RECOVERY_WINDOW = ImmediateInterval(
    anchor="t_interruption_end",
    lo_s=RECOVERY_LO_S,
    hi_s=RECOVERY_HI_S,
    notes=(
        "Observation and Immediate success are the same window. "
        "Matches take-turn observation [-2, +5]; Recover does not use "
        "the tight take-turn success windows."
    ),
)

SUBSEQUENT = SubsequentSpec(
    definition=(
        "The in-window recovery makes the adjustment the interruption "
        "required, instead of resuming obsolete content."
    ),
    notes=(
        "Gated on Immediate: only scored when an in-window takeover "
        "occurred; otherwise Subsequent is N/A. Judge the response whose "
        "onset is in the observation window. One shared question for "
        "every Recover condition. The judge sees the interrupted "
        "contribution and the user interruption only — not the "
        "condition label, a condition prompt, or "
        "reference_recovery_text. Conditions are for aggregation. "
        "`required_adjustment` documents what that slice asked for; it "
        "is not shown to the judge."
    ),
    checks=(
        Check(
            id="recovery_adaptation",
            name="Recovery adaptation",
            pass_when="Makes the adjustment required by the intervention.",
            fail_when=(
                "Ignores the requirement, adjusts incorrectly, or resumes "
                "obsolete content."
            ),
        ),
    ),
)

CONDITION_ROWS = [
    (
        "enrich_response_after_interruption",
        "enrich_response_after_interruption",
        "Re-enter and enrich / expand the interrupted response.",
        "Enrich or add the detail the interruption asked for.",
    ),
    (
        "shorten_response_after_interruption",
        "shorten_response_after_interruption",
        "Re-enter and shorten / condense the interrupted response.",
        "Shorten the remaining contribution as the interruption required.",
    ),
    (
        "repeat_response_after_interruption",
        "repeat_response_after_interruption",
        "Re-enter and repeat the requested part of the prior response.",
        "Repeat the requested content rather than continuing past it.",
    ),
    (
        "shift_topic_after_interruption",
        "shift_topic_after_interruption",
        "Re-enter by shifting to the topic the interruption introduced.",
        "Shift to the new topic; do not resume the obsolete thread as if unchanged.",
    ),
    (
        "ask_follow_up_after_interruption",
        "ask_follow_up_after_interruption",
        "Re-enter by asking a follow-up prompted by the interruption.",
        "Ask the follow-up the interruption called for.",
    ),
    (
        "correct_response_after_interruption",
        "correct_response_after_interruption",
        "Re-enter and correct the prior response as the interruption required.",
        "Correct the obsolete content; do not repeat the uncorrected version.",
    ),
]

CONDITIONS = make_conditions(
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    subsequent=SUBSEQUENT,
    immediate_checks=IMMEDIATE_CHECKS,
    immediate_notes=IMMEDIATE_NOTES,
    intervals={row[0]: RECOVERY_WINDOW for row in CONDITION_ROWS},
    rows=CONDITION_ROWS,
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Recover",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "Helisten (2017): returning to a suspended telling. Oloff (2013): "
        "stopping can precede resumption or withdrawal. Roever & Kasper "
        "(2018, p. 334): repair and context-sensitive adaptation."
    ),
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown the interrupted contribution, the user's "
    "interruption, and the model's recovery in the observation window, and "
    "asked a single yes/no question. Answer using ONLY the evidence given. "
    "If the evidence is ambiguous or insufficient, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

JUDGE_QUESTION = (
    "Does the model's response in the observation window make the "
    "adjustment this interruption required? "
    "Answer YES if it adapts as the user's interruption called for, "
    "given what the model was saying. "
    "Answer NO if it ignores the interruption, adjusts incorrectly, or "
    "resumes obsolete content as if the interruption had not happened."
)


def build_judge_prompt(
    *,
    interruption: str,
    model_recovery: str,
    interrupted_contribution: str = "",
) -> str:
    """Subsequent judge user message. History + interruption only; no condition label."""
    prior = interrupted_contribution.strip() or "(not shown)"
    return (
        f"Action: recover\n\n"
        f"Interrupted contribution (what the model was saying):\n{prior}\n\n"
        f"User interruption:\n{interruption.strip()}\n\n"
        f"Model recovery (speech with onset in the observation window):\n"
        f"{model_recovery.strip() or '(model produced no speech)'}\n\n"
        f"Question: {JUDGE_QUESTION}\n\n"
        'Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
    )


def score_immediate_recover(
    model_speech_segments: list[tuple[float, float]],
    t_interruption_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[int, Optional[str], Optional[float]]:
    """Immediate TOR: any valid onset in [-2.0, 5.0] s vs interruption end."""
    return score_immediate_takeover(
        model_speech_segments,
        t_interruption_end,
        RECOVERY_LO_S,
        RECOVERY_HI_S,
        segment_transcripts=segment_transcripts,
    )
