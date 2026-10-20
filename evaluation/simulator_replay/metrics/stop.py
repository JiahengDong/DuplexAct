"""Stop: speech cessation and maintained yielding."""
from __future__ import annotations

from typing import Optional

from evaluation.scoring.framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)
from evaluation.direct_replay.metrics.take_turn import OVERLAP_LOOKBACK_S
from evaluation.direct_replay.metrics.wait import BACKCHANNEL_MAX_WORDS, is_backchannel

ACTION_ID = "stop"
DATA_ACTION = "stop"

# Anchor: cue / interruption onset (t_cue_start / t_user_int_start).
# Immediate stop rate has no success window: after the interrupt starts,
# did the ongoing turn end. How fast is yield latency, not stop rate.
# Subsequent interval is the cue [user onset, user offset]. Gated on a
# first stop during that interval; then no renewed competitive speech
# before the last OVERLAP_LOOKBACK_S of the cue. Onset in that last
# 1.0 s is take-turn-with-overlap, not a reclaim.

DEFINITION = (
    "When an intervention requires the model to yield, it should stop its "
    "ongoing speech and refrain from continuing or renewing its claim to the "
    "current turn while the intervention is still underway."
)
IMMEDIATE_DEFINITION = (
    "Speech cessation: stop the ongoing output after the interruption begins."
)
IMMEDIATE_CRITERION = (
    "Stop rate = 1 if the model was speaking when the interrupt started "
    "and that turn ends afterwards (they actually go silent). No time "
    "cutoff. Yield latency = t_yield_end − t_cue_start, recorded separately. "
    "Still talking at the end of the recording = did not stop."
)
IMMEDIATE_NOTES = (
    "t = 0 is cue onset. Headline Immediate is stop rate on speaking-at-cue "
    "(forced allowed). Cue in silence or after the model finished is out of "
    "the denominator. Slow stops still count as stops; latency captures delay. "
    "Subsequent is gated on Immediate: only scored after a first stop "
    "during the cue. Take-turn-with-overlap in the last "
    f"{OVERLAP_LOOKBACK_S:.1f} s of the cue is not a Subsequent fail."
)

IMMEDIATE_CHECKS = (
    Check(
        id="stop_rate",
        name="Stop rate",
        pass_when=(
            "The turn that was active at interrupt onset ends after that "
            "onset (the model goes silent)."
        ),
        fail_when=(
            "The model was not speaking at the interrupt, or it never "
            "goes silent before the recording ends."
        ),
    ),
)

STOP_INTERVAL = ImmediateInterval(
    start_key="t_cue_start",
    notes=(
        "After interrupt onset. No upper bound. Stop rate is whether they "
        "stopped, not whether they stopped inside N seconds."
    ),
)

SUBSEQUENT = SubsequentSpec(
    definition=(
        "After the model first stops, it stays yielded through the "
        "interruption except for the take-turn-with-overlap tail. A "
        "brief listener response is not a floor reclaim. Resumption "
        "after user offset is allowed."
    ),
    notes=(
        "Cue = [user onset, user offset] = [t_cue_start, t_cue_end]. "
        "Gated on Immediate: only scored when stop_rate=1 and the first "
        "yield is at or before user offset; otherwise Subsequent is N/A. "
        "After that first yield, a new substantive onset (not a "
        f"backchannel, ≤ {BACKCHANNEL_MAX_WORDS} words) before "
        f"user offset − {OVERLAP_LOOKBACK_S:.1f} s fails. Onset in the "
        f"last {OVERLAP_LOOKBACK_S:.1f} s of the cue is "
        "take-turn-with-overlap and does not count as a reclaim. Speech "
        "after user offset is allowed. Same check for every Stop "
        "condition."
    ),
    checks=(
        Check(
            id="yielding_maintained",
            name="Yielding maintained",
            pass_when=(
                "After the first stop, no renewed competitive speech "
                f"before user offset − {OVERLAP_LOOKBACK_S:.1f} s. "
                "Backchannel is not a reclaim. Take-turn-with-overlap "
                "in the last 1.0 s of the cue, and speech after user "
                "offset, are allowed."
            ),
            fail_when=(
                "After the first stop, the model starts another "
                "substantive turn before the take-turn-with-overlap "
                "tail of the cue."
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
    intervals={"direct_interruption": STOP_INTERVAL},
    rows=[
        (
            "direct_interruption",
            "direct_interruption",
            "User addresses the model and claims the floor; stop and yield.",
            "After first stop, stay yielded until the last 1.0 s of the cue; "
            "take-turn-with-overlap there is not a reclaim.",
        ),
    ],
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Stop",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "Schegloff (2000): stopping or dropping out resolves overlap. "
        "May et al. (2020): assess effective turn-sharing. "
        "Oloff (2013): stopping does not require abandoning later resumption. "
        "The Subsequent yielding-maintained criterion is our adaptation, "
        "not a published IC scale."
    ),
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown the interruption (cue) and the model's "
    "speech around it, and asked a single yes/no question. "
    "Answer using ONLY the evidence given. "
    "If the evidence is ambiguous or insufficient, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

JUDGE_QUESTION = (
    "Once the interruption had begun and the model had first gone silent, "
    "did it stay yielded until the user was finishing? "
    "Answer YES if, after that first stop, it started no new substantive "
    f"turn before the last {OVERLAP_LOOKBACK_S:.1f} s of the "
    "interruption. A brief listener response / backchannel is not a "
    "reclaim. Starting to speak in that last 1.0 s is take-turn-with-"
    "overlap and does not count as reclaiming the floor. Resuming after "
    "the interruption has ended is allowed. "
    "Answer NO if, after that first stop, it resumed the old answer or "
    "started another competing contribution before that last 1.0 s."
)


def build_judge_prompt(
    *,
    condition_id: str,
    interruption: str,
    model_speech_during_cue: str,
    model_speech_after_cue: str = "",
) -> str:
    after = model_speech_after_cue.strip() or "(none / not shown)"
    return (
        f"Action: stop\n"
        f"Condition: {condition_id}\n\n"
        f"Interruption (cue):\n{interruption.strip()}\n\n"
        f"Model speech during the interruption:\n"
        f"{model_speech_during_cue.strip() or '(none)'}\n\n"
        f"Model speech after the interruption ended:\n{after}\n\n"
        f"Question: {JUDGE_QUESTION}\n\n"
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


def yield_end_at_cue(
    model_speech_segments: list[tuple[float, float]],
    t_cue_start: float,
) -> Optional[float]:
    """Offset of the segment that contains cue onset. None if not speaking."""
    containing = [
        off for on, off in model_speech_segments if on <= t_cue_start <= off
    ]
    if not containing:
        return None
    return min(containing)


def score_stop_rate(
    model_speech_segments: list[tuple[float, float]],
    t_cue_start: float,
    t_recording_end: Optional[float] = None,
) -> tuple[int, Optional[str], Optional[float]]:
    """Immediate 0/1: did the active turn end after the interrupt started.

    Yield latency is t_yield_end - t_cue_start when a yield end exists.
    If t_recording_end is set and the active segment runs to that end,
    they never stopped.
    """
    t_end = yield_end_at_cue(model_speech_segments, t_cue_start)
    if t_end is None:
        return 0, "not_speaking_at_cue", None
    latency = t_end - t_cue_start
    if t_recording_end is not None and t_end >= t_recording_end - 0.05:
        return 0, "never_stopped", latency
    return 1, None, latency


def score_yielding_maintained(
    model_speech_segments: list[tuple[float, float]],
    t_cue_start: float,
    t_cue_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> tuple[Optional[int], Optional[str]]:
    """Subsequent 0/1 after a first stop during the cue; else N/A.

    N/A if there was no first stop, or the first stop is after user offset.
    Onset in the last OVERLAP_LOOKBACK_S of the cue is take-turn-with-
    overlap and does not fail Subsequent.
    """
    t_yield = yield_end_at_cue(model_speech_segments, t_cue_start)
    if t_yield is None:
        return None, "no_first_stop"
    if t_yield > t_cue_end:
        return None, "first_stop_after_cue"

    t_overlap_from = t_cue_end - OVERLAP_LOOKBACK_S
    for on, _off, transcript in _paired_segments(
        model_speech_segments, segment_transcripts
    ):
        if on <= t_yield:
            continue
        if on > t_overlap_from:
            continue
        bc = is_backchannel(transcript)
        if bc is True:
            continue
        return 0, f"renewed_speech_during_cue:onset={on:.3f}s"
    return 1, None
