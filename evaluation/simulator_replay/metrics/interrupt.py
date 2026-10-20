"""Interrupt: barge in before the user finishes, and do the required work."""
from __future__ import annotations

from typing import Optional

from evaluation.scoring.framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)
from evaluation.direct_replay.metrics.take_turn import OVERLAP_LOOKBACK_S, takeover_onsets
from evaluation.direct_replay.metrics.wait import BACKCHANNEL_MAX_WORDS

ACTION_ID = "interrupt"
DATA_ACTION = "interrupt_user"

# Anchor: user onset (t_user_start). Immediate deadline is the played
# user utterance end (one-clip: branch_ignored.wav / simulator user_speech
# segment), minus OVERLAP_LOOKBACK_S. Authored t_target is the designed
# interrupt point and is Timing only (Δ = onset − t_target).
# Both conditions share this window.
WINDOW_ID = "input_duration_minus_1"

DEFINITION = (
    "When the user is still speaking and an intervention is required, "
    "the model should enter before that contribution ends and do the "
    "work the condition asked for, not merely announce an interruption."
)
IMMEDIATE_DEFINITION = (
    "Start a substantive interruption before the take-turn-with-overlap "
    "tail of the user's speech."
)
IMMEDIATE_CRITERION = (
    "A valid takeover onset (backchannel excluded) falls in "
    f"[t_user_start, t_user_end − {OVERLAP_LOOKBACK_S:.1f} s]. Any "
    "in-window onset counts; it need not be the first takeover. Onset "
    "in the last 1.0 s of the played user utterance is "
    "take-turn-with-overlap, not an interrupt."
)
IMMEDIATE_NOTES = (
    "t = 0 is user onset. t_user_end is the streamed user wav duration "
    "(inference input_duration_s), not live truncation and not t_target. "
    "t_target is Timing only: Δ = first in-window onset − t_target. Both "
    "conditions use "
    "the same success window. Immediate = at least one valid takeover "
    "onset in that window, not necessarily the first. "
    f"Backchannel (≤ {BACKCHANNEL_MAX_WORDS} words on the whole "
    "segment) is skipped. Window bounds are inclusive. Subsequent is "
    "gated on Immediate: only scored when an in-window interrupt "
    "actually occurred."
)

IMMEDIATE_CHECKS = (
    Check(
        id="timing_conditioned_interrupt",
        name="Timing-conditioned interrupt",
        pass_when=(
            "A valid takeover onset falls inside "
            f"[t_user_start, t_user_end − {OVERLAP_LOOKBACK_S:.1f} s]."
        ),
        fail_when=(
            "No valid takeover onset falls inside that window "
            "(including when the only onset is in the last 1.0 s of "
            "the played user utterance, or after the user finished)."
        ),
    ),
)

INTERRUPT_WINDOW = ImmediateInterval(
    start_key="t_user_start",
    end_key="t_user_end",
    notes=(
        f"Success onset in [t_user_start, t_user_end − {OVERLAP_LOOKBACK_S:.1f} s]. "
        "The last 1.0 s is take-turn-with-overlap, not Interrupt. "
        "t_target is Timing only."
    ),
)

SUBSEQUENT = SubsequentSpec(
    definition=(
        "The interruption itself does the work the condition asked for, "
        "not merely announcing that an interruption is happening."
    ),
    notes=(
        "Gated on Immediate: only scored when an in-window interrupt "
        "occurred; otherwise Subsequent is N/A. One general fulfilment "
        "question for every Interrupt condition. A per-condition prompt "
        "states why this user must be interrupted and what the "
        "interruption needs to do; the judge uses both. Do not show "
        "reference_interruption_text. Same check text; "
        "`required_adjustment` names the obstacle or action."
    ),
    checks=(
        Check(
            id="intervention_fulfilment",
            name="Intervention fulfilment",
            pass_when=(
                "Addresses the identified obstacle or proposed action "
                "with the required contribution."
            ),
            fail_when=(
                "Only announces the interruption, misses the issue, or "
                "gives unrelated content."
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
    intervals={
        "interrupt_for_safety": INTERRUPT_WINDOW,
        "interrupt_for_help": INTERRUPT_WINDOW,
    },
    rows=[
        (
            "interrupt_for_safety",
            "interrupt_for_safety",
            "Interrupt before the user's contribution ends to flag a safety risk.",
            "Address the identified safety obstacle with the required contribution.",
        ),
        (
            "interrupt_for_help",
            "interrupt_for_help",
            "Interrupt before the user's contribution ends to offer needed help or a constraint.",
            "Address the identified help request or time-sensitive action with the required contribution.",
        ),
    ],
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Interrupt",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "CA: Schegloff (2000, pp. 5–6) describes access to an unfinished "
        "turn to further the ongoing undertaking. IC: May et al. (2020, "
        "Tables 1–2) include taking the floor when necessary and bringing "
        "interaction back on track."
    ),
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown why an interruption is needed, the user's "
    "speech, and the model's interruption, and asked a single yes/no "
    "question. Answer using ONLY the evidence given. "
    "If the evidence is ambiguous or insufficient, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

JUDGE_QUESTION = (
    "Did this interruption do the work this condition required, rather "
    "than merely announcing that an interruption is happening? "
    "Answer YES if the model's speech addresses the obstacle or action "
    "described in the condition. "
    "Answer NO if it only announces the interruption, is generic, "
    "off-topic, or misses the issue."
)

CONDITION_PROMPTS: dict[str, str] = {
    "interrupt_for_safety": (
        "The user is describing a dangerous or hazardous behaviour. "
        "An interruption is needed to flag the specific safety risk "
        "and to say why it is dangerous or what to do instead."
    ),
    "interrupt_for_help": (
        "The user is proceeding in a direction where timely help would "
        "matter. An interruption is needed to offer the specific "
        "assistance or constraint the current situation requires."
    ),
}


def build_judge_prompt(
    *,
    condition_id: str,
    user_speech: str,
    model_interruption: str,
) -> str:
    """Subsequent judge user message. Shared question plus condition prompt."""
    condition_prompt = CONDITION_PROMPTS.get(condition_id, "").strip()
    return (
        f"Action: interrupt\n"
        f"Condition: {condition_id}\n\n"
        f"Condition (why an interruption is needed):\n"
        f"{condition_prompt or '(not specified)'}\n\n"
        f"User speech (what was underway):\n{user_speech.strip()}\n\n"
        f"Model interruption:\n"
        f"{model_interruption.strip() or '(model produced no speech)'}\n\n"
        f"Question: {JUDGE_QUESTION}\n\n"
        'Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
    )


def resolve_user_end(
    instance: Optional[dict] = None,
    inference: Optional[dict] = None,
) -> Optional[float]:
    """Full user-utterance duration (the clip that was streamed).

    Do not use simulator user_speech.end: that is live truncation at
    barge-in (`t_model_onset_detected`), not the whole utterance.
    """
    inf = inference or {}
    inst = instance or {}
    if inf.get("input_duration_s") is not None:
        return float(inf["input_duration_s"])
    for seg in inf.get("simulator_speech_segments") or []:
        if seg.get("source") == "user_speech" and seg.get("end") is not None:
            return float(seg["end"])
    if inst.get("t_target") is not None:
        return float(inst["t_target"])
    return None


def score_immediate_interrupt(
    model_speech_segments: list[tuple[float, float]],
    t_user_start: float,
    t_user_end: float,
    t_target: Optional[float] = None,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[int, Optional[str], Optional[float]]:
    """Immediate 0/1: any valid onset in [t_user_start, t_user_end − 1.0 s].

    Returns (score, failure_tag, delta_s). delta_s is first in-window
    onset minus t_target (negative = before the designed point). If
    t_target is missing, delta uses t_user_end. None if no takeover.
    """
    hi = t_user_end - OVERLAP_LOOKBACK_S
    anchor = t_user_end if t_target is None else t_target
    onsets = takeover_onsets(model_speech_segments, segment_transcripts)
    if not onsets:
        return 0, "no_takeover", None

    in_window = [on for on in onsets if t_user_start <= on <= hi]
    if in_window:
        return 1, None, in_window[0] - anchor
    return 0, "no_takeover_in_success_window", onsets[0] - anchor
