"""Take turn: timely entry and a relevant continuation."""
from __future__ import annotations

from typing import Optional

from .framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)
from .wait import BACKCHANNEL_MAX_WORDS, is_backchannel

ACTION_ID = "take_turn"
DATA_ACTION = "turn_taking"

# Anchor for all Take turn windows: user-speech end (metadata t1_end / t_user_end).
# Observation window is where we look for takeovers. It is not a success
# criterion. Success windows are per condition, inside it.
OBSERVATION_LO_S = -2.0
OBSERVATION_HI_S = 5.0
# Last 1.0 s of user speech may count as take-turn-with-overlap, not as
# Interrupt success or as a Stop Subsequent reclaim.
OVERLAP_LOOKBACK_S = 1.0

DEFINITION = (
    "When the model should take a turn, it should begin a substantive "
    "response within the appropriate timing window and continue by addressing "
    "the user's latest communicative intent, while respecting the information "
    "and constraints currently in effect."
)
IMMEDIATE_DEFINITION = (
    "Start a substantive response within the annotated acceptable window."
)
IMMEDIATE_CRITERION = (
    "A valid takeover onset (backchannel excluded) falls in the condition "
    "success window relative to user-speech end. Any in-window onset "
    "counts; it need not be the first takeover. No in-window takeover is "
    "a fail."
)
IMMEDIATE_NOTES = (
    f"t = 0 is user-speech end. Observation window "
    f"[{OBSERVATION_LO_S}, {OBSERVATION_HI_S}] s covers overlap, immediate, "
    "and delayed entry; it is not itself a success window. Immediate = at "
    "least one valid takeover onset in the condition window, not "
    "necessarily the first. Takeover = substantive floor-claiming speech; "
    f"backchannel (≤ {BACKCHANNEL_MAX_WORDS} words on the whole segment) "
    "is skipped. Window bounds are inclusive. Subsequent is gated on "
    "Immediate: only in-window takeovers are judged."
)

IMMEDIATE_CHECKS = (
    Check(
        id="timing_conditioned_takeover",
        name="Timing-conditioned takeover",
        pass_when=(
            "A valid takeover onset falls inside the condition success "
            "window."
        ),
        fail_when=(
            "No valid takeover onset falls inside the condition success "
            "window."
        ),
    ),
)

OBSERVATION_WINDOW = ImmediateInterval(
    anchor="t_user_end",
    lo_s=OBSERVATION_LO_S,
    hi_s=OBSERVATION_HI_S,
    notes="Look for takeovers here. Not a success criterion.",
)

SUCCESS_WINDOWS = {
    "take_turn_immediately": ImmediateInterval(
        anchor="t_user_end",
        lo_s=0.0,
        hi_s=0.5,
        notes="After user end, promptly.",
    ),
    "take_turn_with_delay": ImmediateInterval(
        anchor="t_user_end",
        lo_s=0.0,
        hi_s=3.0,
        notes="After user end, longer preparation still succeeds.",
    ),
    "take_turn_with_overlap": ImmediateInterval(
        anchor="t_user_end",
        lo_s=-OVERLAP_LOOKBACK_S,
        hi_s=0.5,
        notes=f"May enter in the last {OVERLAP_LOOKBACK_S:.1f} s of user speech; overlap not required.",
    ),
}

SUBSEQUENT = SubsequentSpec(
    definition=(
        "The response addresses the user's intent by respecting the "
        "information and constraints in the user's speech."
    ),
    notes=(
        "One check for every Take turn condition. Conditional on "
        "Immediate: Immediate fail → N/A. An in-window response that "
        "fails this check scores 0."
    ),
    checks=(
        Check(
            id="intent_with_constraints",
            name="Intent under user constraints",
            pass_when=(
                "The response addresses the user's intent while respecting "
                "the information and constraints in the user's speech."
            ),
            fail_when=(
                "The response misses the intent, or it ignores or "
                "contradicts information or constraints from the user's "
                "speech."
            ),
        ),
    ),
)

CONDITIONS = make_conditions(
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    subsequent=SUBSEQUENT,
    immediate_checks=IMMEDIATE_CHECKS,
    immediate_notes=IMMEDIATE_NOTES,
    intervals=SUCCESS_WINDOWS,
    rows=[
        (
            "take_turn_immediately",
            "take_turn_immediately",
            "Take the floor promptly after the user's turn ends.",
            "Address user intent by respecting information and constraints "
            "in the user's speech.",
        ),
        (
            "take_turn_with_delay",
            "take_turn_with_delay",
            "Take the floor after a delayed but still acceptable gap.",
            "Address user intent by respecting information and constraints "
            "in the user's speech.",
        ),
        (
            "take_turn_with_overlap",
            "take_turn_with_overlap",
            "Take the floor with slight overlap as the user is finishing.",
            "Address user intent by respecting information and constraints "
            "in the user's speech.",
        ),
    ],
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Take turn",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "May et al. (2020, pp. 174–176) support relevant contributions. "
        "Lam (2018) supports contingent responding."
    ),
)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown one user utterance and the model's spoken "
    "response, and asked a single yes/no question. "
    "Answer using ONLY the evidence given. "
    "If the evidence is ambiguous, insufficient, or the model produced no "
    "relevant speech, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

JUDGE_QUESTION = (
    "Does the model's response address the user's intent by respecting the "
    "information and constraints in the user's speech? "
    "Answer YES if the response takes up what the user was asking or doing, "
    "and it uses (does not ignore or contradict) the facts, limits, and "
    "revisions in that speech. "
    "Answer NO if it misses the intent, or if it ignores or contradicts "
    "information or constraints from the user's speech."
)


def build_judge_prompt(
    *,
    condition_id: str,
    user_speech: str,
    model_response: str,
) -> str:
    """Subsequent judge user message. Same question for every Take turn condition."""
    return (
        f"Action: take_turn\n"
        f"Condition: {condition_id}\n\n"
        f"User speech:\n{user_speech.strip()}\n\n"
        f"Model response:\n{model_response.strip() or '(model produced no speech)'}\n\n"
        f"Question: {JUDGE_QUESTION}\n\n"
        'Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
    )


def takeover_onsets(
    model_speech_segments: list[tuple[float, float]],
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[float]:
    """Substantive onsets, earliest first. Backchannel segments are skipped."""
    transcripts = segment_transcripts
    if transcripts is not None and len(transcripts) != len(model_speech_segments):
        transcripts = None
    paired = sorted(
        (
            (on, off, (transcripts[i] if transcripts is not None else None))
            for i, (on, off) in enumerate(model_speech_segments)
        ),
        key=lambda x: x[0],
    )
    onsets: list[float] = []
    for on, _off, transcript in paired:
        bc = is_backchannel(transcript)
        if bc is True:
            continue
        onsets.append(on)
    return onsets


def score_immediate_takeover(
    model_speech_segments: list[tuple[float, float]],
    t_user_end: float,
    success_lo_s: float,
    success_hi_s: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[int, Optional[str], Optional[float]]:
    """Immediate 0/1: any valid onset in the success window.

    Returns (score, failure_tag, delta_s). delta_s is the first in-window
    onset relative to t_user_end, or None if there is no takeover at all.
    """
    onsets = takeover_onsets(model_speech_segments, segment_transcripts)
    if not onsets:
        return 0, "no_takeover", None

    in_window = [
        on - t_user_end
        for on in onsets
        if success_lo_s <= (on - t_user_end) <= success_hi_s
    ]
    if in_window:
        return 1, None, in_window[0]
    return 0, "no_takeover_in_success_window", onsets[0] - t_user_end
