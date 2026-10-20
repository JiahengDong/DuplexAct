"""Backchannel: brief feedback during user speech, then later consistency.

This module does not score Immediate by itself. The automatic part here is
only a candidate detector (the first <= 3-word segment starting inside the
user's turn), which often keeps the truncated opening of a full turn. The
reported Immediate is the two-stage judgement in `bc_two_stage_judge.py`:
stage 1 (`gpt-5.4-2026-03-05`, text only) rules on whether the candidate
is a genuine backchannel, stage 2 (`gpt-audio`, listening) gives it a
function, and `reports/_bc_verdicts.py` joins both onto the cache when a
report is written. See APPENDIX.md I.9.
"""
from __future__ import annotations

from typing import Optional

from evaluation.scoring.framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)
from evaluation.direct_replay.metrics.wait import BACKCHANNEL_MAX_WORDS, is_backchannel

ACTION_ID = "backchannel"
DATA_ACTION = "backchannel"

# Direct-replay listen protocol (not simulator cue injection).
# No success window around the human reference. Immediate is three steps:
# (1) detect a candidate — any BC onset during user speech, type ignored,
# whole segment ≤ BACKCHANNEL_MAX_WORDS (same rule as Wait / Take turn);
# (2) stage 1 screens that candidate on text, rejecting turn openings,
# unintelligible audio, and segments with no feedback function;
# (3) stage 2 listens to the candidate span and forces a binary
# affirmative / uncertain label, which is then matched to the condition.
# Immediate = 1 only when (2) confirms and (3) matches. Steps 2-3 live in
# bc_two_stage_judge.py; this module implements step 1 only.
# Time difference vs human bc_start / t_target is a diagnostic only.
# Trailing padding after clip_end is outside Immediate.
# Subsequent is a text LLM: user before+overlap, user between BC and
# later model speech, model BC, model later.

DEFINITION = (
    "When the user still holds the floor, the model should give brief "
    "feedback of the required function without taking over, and any "
    "later response should fit that feedback."
)
IMMEDIATE_DEFINITION = (
    "Produce a genuine brief backchannel during the user-speech interval, "
    "and have it carry the function the condition asked for."
)
IMMEDIATE_CRITERION = (
    "Immediate = 1 only if all three hold. (1) A candidate onset (≤ "
    f"{BACKCHANNEL_MAX_WORDS} words on the whole segment) falls in "
    "[clip_start, clip_end], type ignored. (2) Stage 1 confirms that "
    "candidate is a self-contained backchannel rather than the truncated "
    "opening of a full turn, unintelligible speech, or a segment with no "
    "listener-feedback function. (3) Stage 2, listening to the candidate "
    "span, labels it affirmative or uncertain and that label is the "
    "condition function. Timing vs the human reference is reported, not "
    "scored."
)
IMMEDIATE_NOTES = (
    "No window around bc_start. User speech is [clip_start, clip_end]; "
    "trailing padding is outside Immediate. Check 1 is a detector, not a "
    f"score: ≤ {BACKCHANNEL_MAX_WORDS} words, type ignored, and it "
    "over-fires on turn openings, so it is published only as the `rule` "
    "column of the decomposition. Checks 2 and 3 are the two-stage judge "
    "in bc_two_stage_judge.py; neither stage sees the condition label, "
    "the gold human bc_text, or bc_type_original. Stage 2 also has the "
    "candidate's own transcript withheld, so the function is read off "
    "the audio rather than the words. Missing / unusable transcript "
    "cannot be proven a backchannel. First in-interval candidate is the "
    "event. delta_s = t_bc_onset − t_target is a diagnostic. Subsequent "
    "is a separate text LLM, gated on check 2 (a real backchannel "
    "occurred), not on check 3."
)

IMMEDIATE_CHECKS = (
    Check(
        id="overall_backchannel_rate",
        name="Candidate detection (rule only, not a score)",
        pass_when=(
            "A candidate backchannel onset falls inside the user-speech "
            "interval, regardless of function."
        ),
        fail_when=(
            "No candidate onset in the user-speech interval "
            "(silence, padding-only speech, or only a takeover / "
            "unclassifiable segment)."
        ),
        notes=(
            "Detector that gates the judges. Published as the `rule` "
            "column only; it keeps turn openings and so over-counts."
        ),
    ),
    Check(
        id="backchannel_screen",
        name="Backchannel existence (stage 1, text; GPT-5.4)",
        pass_when=(
            "The candidate is self-contained brief listener feedback "
            "that leaves the floor with the user."
        ),
        fail_when=(
            "It runs on into the next segment as one continuous "
            "sentence, carries no feedback function, or has no "
            "intelligible speech."
        ),
        notes=(
            "Text judge (GPT-5.4), blind to the condition. Rejected "
            "candidates are not backchannels, so Subsequent is N/A."
        ),
    ),
    Check(
        id="function_match",
        name="Function match (stage 2, audio; gpt-audio, only listening judge)",
        pass_when=(
            "The listening judge labels the confirmed backchannel as "
            "the condition function (affirmative or uncertain)."
        ),
        fail_when="It labels the other function.",
        notes=(
            "Audio judge, forced binary with no unknown. Only scored "
            "on candidates stage 1 confirmed. The backchannel's own "
            "transcript is withheld; classification then match, and "
            "the condition is never leaked."
        ),
    ),
)

USER_SPEECH_INTERVAL = ImmediateInterval(
    start_key="clip_start",
    end_key="clip_end",
    notes=(
        "User speech [clip_start, clip_end]. Any backchannel onset in "
        "this interval. Not a window around bc_start."
    ),
)

SUBSEQUENT = SubsequentSpec(
    definition=(
        "After the backchannel, later model speech is consistent with "
        "the understanding or stance that backchannel displayed."
    ),
    notes=(
        "Gated on a screened backchannel (Immediate check 2), not on "
        "function match, so the denominator is the set stage 1 "
        "confirmed -- smaller than the rule's hits, and small enough "
        "on most models that a rate off it should not be read alone. "
        "Text LLM, separate from the stage 2 listening judge. The "
        "judge sees (1) user speech before and overlapping the model "
        "backchannel, (2) user speech after that backchannel and "
        "before the model's later response, (3) the model backchannel, "
        "and (4) later model speech. Pre-sliced text; no timestamps; "
        "no gold bc_text; no condition prompt. No later speech "
        "scores 0, not N/A."
    ),
    checks=(
        Check(
            id="feedback_response_consistency",
            name="Feedback–response consistency",
            pass_when=(
                "Later speech fits the stance displayed in the "
                "backchannel, or is justified by later user input."
            ),
            fail_when=(
                "Later speech contradicts the backchannel without "
                "justification from subsequent input."
            ),
        ),
    ),
)

CONDITION_ROWS = [
    (
        "backchannel_affirmative",
        "backchannel_affirmative",
        "Brief acknowledgement / understanding / agreement; do not take the floor.",
        "The backchannel should signal acknowledgement, understanding, or agreement.",
    ),
    (
        "backchannel_uncertainty",
        "backchannel_uncertainty",
        "Brief uncertainty / repair-relevant feedback; do not take the floor.",
        "The backchannel should signal uncertainty, hesitation, or a need for clarification.",
    ),
]

CONDITIONS = make_conditions(
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    subsequent=SUBSEQUENT,
    immediate_checks=IMMEDIATE_CHECKS,
    immediate_notes=IMMEDIATE_NOTES,
    intervals={row[0]: USER_SPEECH_INTERVAL for row in CONDITION_ROWS},
    rows=CONDITION_ROWS,
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Backchannel",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "CA: Schegloff (1982, p. 81) analyses continuers as passing a "
        "full-turn opportunity. IC: May et al. (2020, pp. 177–178) use "
        "backchannels and next-turn contributions as listening evidence. "
        "The consistency metric is our operationalisation."
    ),
)

# The Immediate prompts are NOT here. Stage 1 (SCREEN_SYSTEM /
# SCREEN_QUESTION) and stage 2 (FUNCTION_SYSTEM / FUNCTION_QUESTION),
# together with the shared context block that lists every model segment
# and the silence before it, live in bc_two_stage_judge.py. Editing a
# prompt in this file will not change Immediate.

SUBSEQUENT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown the user's speech before and overlapping "
    "the model's backchannel, the user's speech after that backchannel and "
    "before the model's later response, the backchannel itself, and the "
    "later model speech. Answer a single yes/no question using ONLY the "
    "evidence given. "
    "If the evidence is ambiguous or insufficient, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

SUBSEQUENT_JUDGE_QUESTION = (
    "Is the model's later speech consistent with the understanding or "
    "stance displayed in its backchannel? "
    "Answer YES if later speech fits that backchannel, or is justified "
    "by what the user said afterwards. "
    "Answer NO if later speech contradicts the backchannel without "
    "such justification."
)

# Aliases: JUDGE_* refer to the Subsequent prompts.
JUDGE_SYSTEM_PROMPT = SUBSEQUENT_JUDGE_SYSTEM_PROMPT
JUDGE_QUESTION = SUBSEQUENT_JUDGE_QUESTION


FUNCTION_BY_CONDITION = {
    "backchannel_affirmative": "affirmative",
    "backchannel_uncertainty": "uncertain",
}


def score_function_match(condition_id: str, predicted: str) -> tuple[int, Optional[str]]:
    """Compare a stage 2 function label to the condition function."""
    wanted = FUNCTION_BY_CONDITION.get(condition_id)
    label = (predicted or "").strip().lower()
    if wanted is None:
        return 0, f"unknown_condition:{condition_id}"
    if label == wanted:
        return 1, None
    return 0, f"function_mismatch:{label or 'empty'}"


def build_judge_prompt(
    *,
    user_before_and_overlap: str,
    user_between: str,
    model_backchannel: str,
    model_later: str,
) -> str:
    """Subsequent text-LLM prompt. No condition prompt or gold bc_text."""
    return (
        f"Action: backchannel\n\n"
        f"User speech before and overlapping the model backchannel:\n"
        f"{user_before_and_overlap.strip() or '(none)'}\n\n"
        f"User speech after the backchannel and before the model's later response:\n"
        f"{user_between.strip() or '(none)'}\n\n"
        f"Model backchannel:\n"
        f"{model_backchannel.strip() or '(none)'}\n\n"
        f"Model later speech:\n"
        f"{model_later.strip() or '(none)'}\n\n"
        f"Question: {SUBSEQUENT_JUDGE_QUESTION}\n\n"
        'Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
    )


def _paired_segments(
    model_speech_segments: list[tuple[float, float]],
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[tuple[float, float, Optional[str]]]:
    transcripts = segment_transcripts
    if transcripts is not None and len(transcripts) != len(model_speech_segments):
        transcripts = None
    paired = [
        (on, off, (transcripts[i] if transcripts is not None else None))
        for i, (on, off) in enumerate(model_speech_segments)
    ]
    paired.sort(key=lambda x: x[0])
    return paired


def first_user_speech_backchannel(
    model_speech_segments: list[tuple[float, float]],
    t_user_start: float,
    t_user_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> Optional[tuple[float, float, str]]:
    """First valid backchannel whose onset is in [t_user_start, t_user_end]."""
    for on, off, transcript in _paired_segments(
        model_speech_segments, segment_transcripts
    ):
        if on < t_user_start or on > t_user_end:
            continue
        if is_backchannel(transcript) is True:
            return on, off, (transcript or "").strip()
    return None


def score_immediate_backchannel(
    model_speech_segments: list[tuple[float, float]],
    t_user_start: float,
    t_user_end: float,
    t_target: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[int, Optional[str], Optional[float]]:
    """Automatic Immediate check: a backchannel during user speech.

    Returns (score, failure_tag, delta_s). delta_s is t_bc_onset −
    t_target (human reference) when a backchannel was found, else None.
    Function match is a separate judge check.
    """
    event = first_user_speech_backchannel(
        model_speech_segments,
        t_user_start,
        t_user_end,
        segment_transcripts,
    )
    if event is None:
        return 0, "no_backchannel", None
    on, _off, _text = event
    return 1, None, on - t_target


def speech_overlapping(
    speech_segments: list[tuple[float, float]],
    win_start: float,
    win_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """Transcripts of segments that overlap [win_start, win_end]."""
    if win_end <= win_start:
        return []
    texts: list[str] = []
    for on, off, transcript in _paired_segments(speech_segments, segment_transcripts):
        if on < win_end and off > win_start:
            texts.append((transcript or "").strip())
    return texts


def speech_after_backchannel(
    model_speech_segments: list[tuple[float, float]],
    t_bc_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """Transcripts of model segments that start after the backchannel ends."""
    texts: list[str] = []
    for on, _off, transcript in _paired_segments(
        model_speech_segments, segment_transcripts
    ):
        if on < t_bc_end:
            continue
        texts.append((transcript or "").strip())
    return texts


def first_later_onset(
    model_speech_segments: list[tuple[float, float]],
    t_bc_end: float,
) -> Optional[float]:
    """Onset of the first model segment that starts after the backchannel."""
    later = [on for on, _off in model_speech_segments if on >= t_bc_end]
    return min(later) if later else None


def user_before_and_overlap(
    user_speech_segments: list[tuple[float, float]],
    t_bc_end: float,
    t_user_start: float = 0.0,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """User speech before and overlapping the model backchannel."""
    return speech_overlapping(
        user_speech_segments,
        t_user_start,
        t_bc_end,
        segment_transcripts,
    )


def user_between_bc_and_later(
    user_speech_segments: list[tuple[float, float]],
    t_bc_end: float,
    t_later_start: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """User speech after the backchannel and before later model speech."""
    return speech_overlapping(
        user_speech_segments,
        t_bc_end,
        t_later_start,
        segment_transcripts,
    )
