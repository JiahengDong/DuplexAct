"""Wait: defer responding and retain new information."""
from __future__ import annotations

from typing import Optional

from .framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)

ACTION_ID = "wait"
DATA_ACTION = "wait"

# Substantive vs backchannel. A model segment that overlaps the Immediate
# interval is a takeover iff word count > BACKCHANNEL_MAX_WORDS. ≤ 3 words
# is a backchannel and does not count. Empty ASR on a VAD segment is 0
# words → backchannel. Transcript list missing / length-mismatched means
# we cannot prove a backchannel, so overlapping VAD counts as takeover.
BACKCHANNEL_MAX_WORDS: int = 3

DEFINITION = (
    "When the model should wait while listening, it should withhold a "
    "substantive response during the designated waiting interval and, when "
    "a subsequent response is called for, respond in a way that appropriately "
    "reflects the user's continuing contribution."
)
IMMEDIATE_DEFINITION = (
    "Withhold a substantive response throughout the annotated waiting interval."
)
IMMEDIATE_CRITERION = (
    "Silence rate = 1 − takeover. Takeover = a substantive model turn that "
    "overlaps the condition interval. Backchannel (≤ 3 words on the whole "
    "overlapping segment) is not a takeover. Word count is never taken on "
    "the slice that happens to fall inside a short pause."
)
IMMEDIATE_NOTES = (
    "Immediate score is silence rate = 1 − takeover. Takeover = any "
    "substantive model speech that overlaps the condition interval, "
    "including speech that started before the interval and continued into "
    "it. Backchannel is a property of the whole overlapping segment "
    f"(word count ≤ {BACKCHANNEL_MAX_WORDS}), not of how many words land "
    "inside the wait window. A short pause that only catches the first or "
    "last few words of a longer turn is still a takeover. Headline is "
    "overlap, not onset: Immediate asks whether the model was quiet "
    "throughout the wait, which is the same construct as no_pause. "
    "Pause-onset (did the pause itself trigger a new start) is a "
    "diagnostic, not the Immediate score. No pause does not grant "
    "take-turn-with-overlap; the interval is the whole user speech. "
    "Trailing padding after speech end is outside Immediate."
)

IMMEDIATE_CHECKS = (
    Check(
        id="silence",
        name="Silence rate (1 − takeover)",
        pass_when=(
            "No substantive model speech overlaps the annotated waiting "
            "interval. A backchannel (≤ 3 words) is not a takeover."
        ),
        fail_when=(
            "A model speech segment overlaps the interval and the whole "
            "segment has more than 3 words (including when a short pause "
            "only contains a few of those words), or overlapping VAD has "
            "no usable transcript (cannot confirm a backchannel)."
        ),
        notes=(
            "Aggregate silence rate = mean of this 0/1. "
            "Failure tag: takeover_during_wait."
        ),
    ),
)

PAUSE_INTERVAL = ImmediateInterval(
    start_key="pause_start",
    end_key="pause_end",
    notes="Annotated pause [pause_start, pause_end]. Inclusive.",
)

NO_PAUSE_INTERVAL = ImmediateInterval(
    start_key="clip_start",
    end_key="clip_end",
    notes=(
        "No pause: entire user speech [clip_start, clip_end]. No overlap "
        "grace at the end: take-turn-with-overlap is out of scope here. "
        "Trailing 15s padding is after clip_end and is not part of Immediate."
    ),
)

SUBSEQUENT = SubsequentSpec(
    definition=(
        "The model's response is relevant and includes some information "
        "(not all) from the user content during or after the wait "
        "interval, reflecting the user's continuing contribution."
    ),
    notes=(
        "Same judge question for every Wait condition. Only the "
        "condition prompt changes (inter-sentential pause, "
        "intra-sentential pause, or no pause). Do not require complete "
        "uptake of everything the user said. Conditional on Immediate: "
        "denominator is Immediate = 1. A later response that meets this "
        "check scores 1; no later response or an inappropriate later "
        "response scores 0 (not N/A). Immediate fail is N/A."
    ),
    checks=(
        Check(
            id="continuation_uptake",
            name="Continuation uptake",
            pass_when=(
                "The response is relevant and includes some information "
                "from the user content during or after the wait "
                "interval."
            ),
            fail_when=(
                "The model never produces a later response, the "
                "response is irrelevant, or it ignores the user's "
                "continuing contribution from during or after the wait."
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
        "inter_sentential_pause": PAUSE_INTERVAL,
        "intra_sentential_pause": PAUSE_INTERVAL,
        "no_pause": NO_PAUSE_INTERVAL,
    },
    rows=[
        (
            "inter_sentential_pause",
            "inter_sentential",
            "Pause between sentences. Immediate interval is the pause.",
            "User content during or after this between-sentence pause.",
        ),
        (
            "intra_sentential_pause",
            "intra_sentential",
            "Pause within a sentence. Immediate interval is the pause.",
            "User content during or after this mid-sentence pause.",
        ),
        (
            "no_pause",
            "floor_holding",
            "User speaks continuously with no annotated pause. Immediate "
            "interval is the whole user speech.",
            "User content during this continuous turn (and after it, if any).",
        ),
    ],
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Wait",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "May et al. (2020, pp. 177–178): subsequent development and "
        "follow-up questions provide listening evidence. Lam (2018): "
        "contingent responding. Adaptation targets information received "
        "during or after the wait."
    ),
)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown the wait condition, the user's speech "
    "during or after that wait interval, and the model's later response. "
    "Answer a single yes/no question using ONLY the evidence given. "
    "If the evidence is ambiguous or insufficient, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

JUDGE_QUESTION = (
    "Is the model's response relevant, and does it include some "
    "information from the user's continuing contribution during or after "
    "the wait interval? "
    "Answer YES if the response is on-topic and uses at least some of "
    "that later or overlapping user content. Complete uptake of every "
    "detail is not required. "
    "Answer NO if the response is irrelevant, or if it ignores that "
    "continuing contribution."
)

CONDITION_PROMPTS: dict[str, str] = {
    "inter_sentential_pause": (
        "The wait interval is a pause between sentences. The user's "
        "continuing contribution is what they say after that "
        "pause (the next sentence and anything that follows until the "
        "model responds)."
    ),
    "intra_sentential_pause": (
        "The wait interval is a pause inside a sentence. The user's "
        "continuing contribution is what they say after that "
        "pause (the rest of the sentence and anything that follows until "
        "the model responds)."
    ),
    "no_pause": (
        "There is no annotated pause. The wait interval is the user's "
        "continuous turn. The user's continuing contribution is that "
        "turn (and anything said after it, if any) until the model "
        "responds."
    ),
}


def build_judge_prompt(
    *,
    condition_id: str,
    user_during_or_after_wait: str,
    model_response: str,
) -> str:
    """Subsequent judge user message. Shared question; condition prompt differs."""
    condition_prompt = CONDITION_PROMPTS.get(condition_id, "").strip()
    return (
        f"Action: wait\n"
        f"Condition: {condition_id}\n\n"
        f"Condition:\n{condition_prompt or '(not specified)'}\n\n"
        f"User speech during or after the wait interval:\n"
        f"{user_during_or_after_wait.strip() or '(none)'}\n\n"
        f"Model response:\n"
        f"{model_response.strip() or '(model produced no speech)'}\n\n"
        f"Question: {JUDGE_QUESTION}\n\n"
        'Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
    )


def is_backchannel(transcript: Optional[str]) -> Optional[bool]:
    """True if ≤ BACKCHANNEL_MAX_WORDS. None if transcript is missing."""
    if transcript is None:
        return None
    return len(transcript.split()) <= BACKCHANNEL_MAX_WORDS


def _overlaps(on: float, off: float, start: float, end: float) -> bool:
    return on < end and off > start


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


def _is_substantive_overlap(transcript: Optional[str]) -> bool:
    """True if this overlapping segment counts as takeover.

    Word count is on the full segment transcript, not the words whose
    timestamps fall inside the wait interval. A short pause that only
    contains the head or tail of a longer turn is still takeover.
    """
    bc = is_backchannel(transcript)
    return bc is None or not bc


def score_immediate_silence(
    model_speech_segments: list[tuple[float, float]],
    interval_start: float,
    interval_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[int, Optional[str]]:
    """Immediate 0/1: 1 = no substantive speech overlapping the interval.

    Overlap, not onset: a turn that started before the wait and continued
    into it is a takeover. Backchannel (≤ 3 words) is skipped.
    """
    for on, off, transcript in _paired_segments(
        model_speech_segments, segment_transcripts
    ):
        if not _overlaps(on, off, interval_start, interval_end):
            continue
        if _is_substantive_overlap(transcript):
            return 0, f"takeover_during_wait:onset={on:.3f}s"
    return 1, None


def score_pause_onset_takeover(
    model_speech_segments: list[tuple[float, float]],
    pause_start: float,
    pause_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[int, Optional[str]]:
    """Diagnostic only: 1 = a new substantive onset started inside the pause.

    This is the old Wait B_i inverted. It asks whether the pause itself
    triggered a start, not whether the model was quiet throughout. Do not
    mix this with floor_holding Immediate when aggregating Wait.
    """
    for on, _off, transcript in _paired_segments(
        model_speech_segments, segment_transcripts
    ):
        if not (pause_start <= on <= pause_end):
            continue
        if _is_substantive_overlap(transcript):
            return 1, f"pause_onset_takeover:onset={on:.3f}s"
    return 0, None
