"""Ignore: stay silent on the distractor, then exclude it from the later reply."""
from __future__ import annotations

from typing import Optional

from evaluation.scoring.framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)
from evaluation.direct_replay.metrics.wait import BACKCHANNEL_MAX_WORDS, score_immediate_silence

ACTION_ID = "ignore"
DATA_ACTION = "ignore"

# Distractor span. Metadata aliases: background uses bg_speech_start_s /
# bg_speech_end_s; other-directed uses side_speech_start_s /
# side_speech_end_s. Unified scoring keys: t_other_start / t_other_end.
# No extra grace after t_other_end (unlike the old 0.5 s SR window).
# Trailing user speech after the distractor is outside Immediate.

DEFINITION = (
    "While listening, the model should not respond to input that is "
    "irrelevant to the ongoing interaction, and it should not let that "
    "input alter its subsequent response to the addressed user."
)
IMMEDIATE_DEFINITION = (
    "Withhold a substantive response throughout the annotated distractor "
    "interval."
)
IMMEDIATE_CRITERION = (
    "Silence rate = 1 − takeover. Takeover = a substantive model turn "
    "that overlaps [t_other_start, t_other_end]. Backchannel "
    f"(≤ {BACKCHANNEL_MAX_WORDS} words on the whole overlapping segment) "
    "is not a takeover."
)
IMMEDIATE_NOTES = (
    "Same construct as Wait Immediate, on the distractor span rather "
    "than a pause. Headline is overlap, not onset: speech that started "
    "before the distractor and continued into it counts. Both conditions "
    "share this interval shape; only the timestamp fields differ. No "
    "grace after t_other_end. Take-turn-with-overlap at user end is "
    "outside Immediate unless that onset also overlaps the distractor. "
    "Subsequent is gated on Immediate. Subsequent uses all model "
    "speech from t_other_end onward (overlap, not a new onset). N/A "
    "if Immediate failed, or if nothing is spoken after t_other_end."
)

IMMEDIATE_CHECKS = (
    Check(
        id="silence",
        name="Silence rate (1 − takeover)",
        pass_when=(
            "No substantive model speech overlaps the distractor "
            "interval. A backchannel (≤ 3 words) is not a takeover."
        ),
        fail_when=(
            "A model speech segment overlaps the distractor and the "
            "whole segment has more than 3 words, or overlapping VAD "
            "has no usable transcript."
        ),
        notes="Failure tag: takeover_during_wait (same helper as Wait).",
    ),
)

IGNORE_INTERVAL = ImmediateInterval(
    start_key="t_other_start",
    end_key="t_other_end",
    notes=(
        "Distractor [t_other_start, t_other_end]. Background: "
        "bg_speech_*; other-directed: side_speech_*. No grace after end."
    ),
)

SUBSEQUENT = SubsequentSpec(
    definition=(
        "Later output does not treat the ignored input as a task "
        "requirement or governing fact."
    ),
    notes=(
        "Judge all model speech that overlaps after t_other_end, "
        "including a turn that started during the distractor and "
        "continued past the end. Not an onset check. Conditional on "
        "Immediate: no speech after t_other_end scores 0, not N/A. "
        "The judge sees the addressed user speech and the ignored "
        "speech — not the condition label, a condition prompt, or "
        "speaker_b_text. Conditions are for aggregation. Same check "
        "for every Ignore condition."
    ),
    checks=(
        Check(
            id="irrelevant_input_exclusion",
            name="Irrelevant-input exclusion",
            pass_when=(
                "Later output excludes the irrelevant input as a task "
                "requirement or governing fact."
            ),
            fail_when=(
                "Later output contains, echoes, or is shaped by the "
                "irrelevant input."
            ),
            notes=(
                "Do not score whether the reply addresses the user. Only "
                "whether it includes the ignored speech. Example: user "
                "asks for a table for two; to Alex, “Bring six plates.” "
                "Pass: later speech has no six-plate / Alex content. "
                "Fail: it takes up the six plates."
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
        "ignore_background_speech_during_user_speech": IGNORE_INTERVAL,
        "ignore_other_directed_speech_during_user_speech": IGNORE_INTERVAL,
    },
    rows=[
        (
            "ignore_background_speech_during_user_speech",
            "ignore_background_speech_during_user_speech",
            "Background speech during the user's addressed turn; do not respond to it.",
            "Do not adopt the background speech as a task fact in the later response.",
        ),
        (
            "ignore_other_directed_speech_during_user_speech",
            "ignore_other_directed_speech_during_user_speech",
            "Other-directed / side speech during the user's addressed turn; do not respond to it.",
            "Do not adopt the other-directed speech as a task fact in the later response.",
        ),
    ],
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Ignore",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "Lerner (2003, pp. 182–183): non-selected recipients withhold a "
        "response. Clark & Carlson (1982): hearers have different "
        "participation roles. The Subsequent criterion extends these "
        "distinctions to benchmark evaluation."
    ),
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown the user's addressed speech, other speech "
    "that should be ignored, and the model's speech (already restricted to "
    "after that other speech ended). Answer a single yes/no question using "
    "ONLY the evidence given. "
    "If the evidence is ambiguous or insufficient, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

JUDGE_QUESTION = (
    "Does the model's reply stay free of the other speech? "
    "Answer YES if it neither contains that other speech nor is "
    "influenced by it: no quoting, echoing, answering, or treating it "
    "as a fact or task. "
    "Answer NO if the reply includes the other speech or is shaped by "
    "it."
)


def build_judge_prompt(
    *,
    addressed_user_speech: str,
    ignored_speech: str,
    model_response: str,
) -> str:
    """Subsequent judge user message. No condition label or gold reply."""
    return (
        f"Action: ignore\n\n"
        f"User speech addressed to the model:\n"
        f"{addressed_user_speech.strip()}\n\n"
        f"Other speech (should be ignored):\n"
        f"{ignored_speech.strip()}\n\n"
        f"Model speech (after the other speech ended):\n"
        f"{model_response.strip() or '(none)'}\n\n"
        f"Question: {JUDGE_QUESTION}\n\n"
        'Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
    )


def resolve_ignore_interval(
    *,
    t_other_start: Optional[float] = None,
    t_other_end: Optional[float] = None,
    bg_speech_start_s: Optional[float] = None,
    bg_speech_end_s: Optional[float] = None,
    side_speech_start_s: Optional[float] = None,
    side_speech_end_s: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    """Map metadata aliases onto [t_other_start, t_other_end]."""
    start = t_other_start
    end = t_other_end
    if start is None:
        start = bg_speech_start_s if bg_speech_start_s is not None else side_speech_start_s
    if end is None:
        end = bg_speech_end_s if bg_speech_end_s is not None else side_speech_end_s
    if start is None or end is None:
        return None
    return start, end


def score_immediate_ignore(
    model_speech_segments: list[tuple[float, float]],
    t_other_start: float,
    t_other_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[int, Optional[str]]:
    """Immediate silence rate on the distractor span."""
    return score_immediate_silence(
        model_speech_segments,
        t_other_start,
        t_other_end,
        segment_transcripts=segment_transcripts,
    )


def speech_after_other_end(
    model_speech_segments: list[tuple[float, float]],
    t_other_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """Transcripts of every segment that still has speech after t_other_end."""
    transcripts = segment_transcripts
    if transcripts is not None and len(transcripts) != len(model_speech_segments):
        transcripts = None
    texts: list[str] = []
    for i, (_on, off) in enumerate(model_speech_segments):
        if off <= t_other_end:
            continue
        piece = ""
        text = transcripts[i] if transcripts is not None else None
        if text:
            piece = text.strip()
        texts.append(piece)
    return texts


def score_subsequent_applicable(
    model_speech_segments: list[tuple[float, float]],
    t_other_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> bool:
    """True if any model speech continues or starts after t_other_end."""
    del segment_transcripts
    return any(off > t_other_end for _on, off in model_speech_segments)
