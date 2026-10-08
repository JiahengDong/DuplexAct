"""Hold floor: keep talking through the cue and adapt (or preserve) the turn."""
from __future__ import annotations

from typing import Optional

from .framework import (
    ActionEval,
    Check,
    ImmediateInterval,
    SubsequentSpec,
    make_conditions,
)

ACTION_ID = "hold_floor"
DATA_ACTION = "hold_floor"

# Immediate window is the cue plus a 1.0 s yielding tail:
# [t_cue_start, t_cue_end + YIELD_WINDOW_S].
# Short continuer cues ("Yes." / "Hmm.") are otherwise shorter than
# MAX_GAP_S, so a full yield inside the raw cue would still pass.
# Same 1.0 s as Stop's last-second yielding / take-turn-with-overlap
# lookback — try this tail first. Pass if max silence in that window
# ≤ MAX_GAP_S. Diagnostic: the observed max gap.
YIELD_WINDOW_S = 1.0
MAX_GAP_S = 0.50

DEFINITION = (
    "When the model already holds the floor and a non-takeover cue "
    "arrives, it should keep speaking through the cue and adapt (or "
    "preserve) the ongoing turn as that cue called for."
)
IMMEDIATE_DEFINITION = (
    "Keep the floor through the hold window: no silence gap longer "
    f"than {MAX_GAP_S:.2f} s in "
    f"[t_cue_start, t_cue_end + {YIELD_WINDOW_S:.1f}]."
)
IMMEDIATE_CRITERION = (
    "Floor-holding rate = 1 if the longest silence in the hold window "
    f"is ≤ {MAX_GAP_S:.2f} s, including a leading gap from cue onset "
    "and a trailing gap to the window end. Window end is "
    f"t_cue_end + {YIELD_WINDOW_S:.1f} s. No speech in the window is "
    "a fail."
)
IMMEDIATE_NOTES = (
    "t = 0 is cue onset. Inverse of Stop Immediate: Stop asks whether "
    "the active turn ended; Hold floor asks whether speech continues "
    f"through the cue and {YIELD_WINDOW_S:.1f} s after cue end with "
    "only short gaps. Headline reporting is among speaking-at-cue "
    "(forced allowed). All five conditions share "
    f"[{MAX_GAP_S:.2f} s] as the max gap and "
    f"[{YIELD_WINDOW_S:.1f} s] as the post-cue yielding tail. Max gap "
    "is also a diagnostic. Subsequent is gated on Immediate: only "
    "scored when they held the floor. Subsequent splits at t_cue_end: "
    "speech before and during the cue vs speech after cue end. N/A if "
    "Immediate failed or there is no speech after cue end."
)

IMMEDIATE_CHECKS = (
    Check(
        id="floor_holding_rate",
        name="Floor-holding rate",
        pass_when=(
            "Max silence in "
            f"[t_cue_start, t_cue_end + {YIELD_WINDOW_S:.1f}] "
            f"is ≤ {MAX_GAP_S:.2f} s."
        ),
        fail_when=(
            "No model speech in the hold window, or a silence gap "
            f"longer than {MAX_GAP_S:.2f} s."
        ),
    ),
)

CUE_INTERVAL = ImmediateInterval(
    start_key="t_cue_start",
    end_key="t_cue_end",
    notes=(
        f"Scored window [t_cue_start, t_cue_end + {YIELD_WINDOW_S:.1f}]. "
        f"Hold if max gap ≤ {MAX_GAP_S:.2f} s. "
        f"Yielding tail is {YIELD_WINDOW_S:.1f} s after cue end."
    ),
)

SUBSEQUENT = SubsequentSpec(
    definition=(
        "After the cue ends, the continuation adapts (or preserves) "
        "the ongoing turn as this condition required."
    ),
    notes=(
        "Gated on Immediate. Split at t_cue_end, not t_cue_start. "
        "The judge sees (1) model speech before the cue and during "
        "overlap, (2) the cue, (3) model speech after cue end, and "
        "(4) this condition's required adjustment. Pre-sliced text; "
        "no timestamps. A turn that crosses cue end may appear in "
        "both speech blocks. Do not show reference_after_text. Same "
        "check text; `required_adjustment` is shown because real cues "
        "are often 'Yes' / 'Hmm' / inaudible. N/A if Immediate failed "
        "or there is no speech after cue end."
    ),
    checks=(
        Check(
            id="continuation_adaptation",
            name="Continuation adaptation",
            pass_when=(
                "After cue end, continues, shortens, simplifies, or "
                "preserves as this condition required, and does not "
                "adopt a distractor cue."
            ),
            fail_when=(
                "After cue end, abandons the thread, takes up a cue "
                "that should be ignored, or fails to make the change "
                "this condition required."
            ),
        ),
    ),
)

CONDITION_ROWS = [
    (
        "continue_with_background_speech",
        "continue_with_background_speech",
        "Background speech while the model is talking; keep the floor and ignore it.",
        "Preserve the current content; do not treat background speech as addressed input.",
    ),
    (
        "continue_with_other_directed_speech",
        "continue_with_other_directed_speech",
        "User briefly addresses someone else; keep the floor and ignore the side talk.",
        "Preserve the current content; do not treat other-directed speech as addressed input.",
    ),
    (
        "continue_current_response",
        "continue_current_response",
        "Neutral continuer; keep going without restructuring.",
        "Continue the current response without unnecessary change.",
    ),
    (
        "continue_but_shorten",
        "continue_but_shorten",
        "Cue implies wrap up; keep the floor but shorten.",
        "Shorten or condense the ongoing response.",
    ),
    (
        "continue_with_simplification",
        "continue_with_simplification",
        "Cue implies simplify; keep the floor but simplify.",
        "Simplify the language or content of the ongoing response.",
    ),
]

CONDITIONS = make_conditions(
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    subsequent=SUBSEQUENT,
    immediate_checks=IMMEDIATE_CHECKS,
    immediate_notes=IMMEDIATE_NOTES,
    intervals={row[0]: CUE_INTERVAL for row in CONDITION_ROWS},
    rows=CONDITION_ROWS,
)

ACTION = ActionEval(
    id=ACTION_ID,
    title="Hold floor",
    definition=DEFINITION,
    data_action=DATA_ACTION,
    immediate_definition=IMMEDIATE_DEFINITION,
    immediate_criterion=IMMEDIATE_CRITERION,
    immediate_checks=IMMEDIATE_CHECKS,
    subsequent=SUBSEQUENT,
    conditions=CONDITIONS,
    human_support=(
        "May et al. (2020, pp. 176–177): holding a turn through prosody "
        "and pause fillers. Schegloff (1982, p. 81): continuers support "
        "continuation of an extended turn. Roever & Kasper (2018, p. 334): "
        "adapt interactional methods to the local context."
    ),
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. You will be shown what the model was saying before and during "
    "a cue, what it said after the cue ended, and what this condition "
    "required. Answer a single yes/no question using ONLY the evidence given. "
    "If the evidence is ambiguous or insufficient, answer NO. "
    'Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
)

JUDGE_QUESTION = (
    "Relative to what the model was saying before and during the cue, "
    "does its speech after the cue ended meet this condition's "
    "requirement? "
    "Answer YES if it keeps the same thread and does what the condition "
    "asked (keep going, shorten, simplify, or ignore a distractor). "
    "Answer NO if it abandons the thread, takes up a cue that should be "
    "ignored, or fails to make the change the condition required."
)


def build_judge_prompt(
    *,
    condition_id: str,
    required_adjustment: str,
    cue: str,
    model_through_cue: str,
    model_after_cue: str,
) -> str:
    """Subsequent judge user message. Condition requirement; no gold after-text."""
    return (
        f"Action: hold_floor\n"
        f"Condition: {condition_id}\n\n"
        f"What this condition requires:\n"
        f"{required_adjustment.strip() or '(not specified)'}\n\n"
        f"Cue:\n{cue.strip() or '(none)'}\n\n"
        f"Model speech before and during the cue:\n"
        f"{model_through_cue.strip() or '(none)'}\n\n"
        f"Model speech after the cue ended:\n"
        f"{model_after_cue.strip() or '(none)'}\n\n"
        f"Question: {JUDGE_QUESTION}\n\n"
        'Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.'
    )


def hold_window_end(t_cue_end: float) -> float:
    """End of the Immediate hold window (cue end plus yielding tail)."""
    return t_cue_end + YIELD_WINDOW_S


def max_gap_in_window(
    model_speech_segments: list[tuple[float, float]],
    win_start: float,
    win_end: float,
) -> float:
    """Longest silence in [win_start, win_end], including lead and trail."""
    clipped: list[tuple[float, float]] = []
    for on, off in sorted(model_speech_segments):
        start = max(on, win_start)
        end = min(off, win_end)
        if start < end:
            clipped.append((start, end))
    if not clipped:
        return max(0.0, win_end - win_start)
    max_gap = 0.0
    if clipped[0][0] > win_start:
        max_gap = clipped[0][0] - win_start
    for i in range(1, len(clipped)):
        gap = clipped[i][0] - clipped[i - 1][1]
        if gap > max_gap:
            max_gap = gap
    if clipped[-1][1] < win_end:
        max_gap = max(max_gap, win_end - clipped[-1][1])
    return max_gap


def score_immediate_hold_floor(
    model_speech_segments: list[tuple[float, float]],
    t_cue_start: float,
    t_cue_end: float,
) -> tuple[int, Optional[str], float]:
    """Immediate 0/1 and the observed max gap in the hold window."""
    win_end = hold_window_end(t_cue_end)
    gap = max_gap_in_window(model_speech_segments, t_cue_start, win_end)
    if gap > MAX_GAP_S:
        return 0, f"floor_released:gap={gap:.3f}s", gap
    return 1, None, gap


def _segment_texts(
    model_speech_segments: list[tuple[float, float]],
    segment_transcripts: Optional[list[Optional[str]]],
    keep,
) -> list[str]:
    transcripts = segment_transcripts
    if transcripts is not None and len(transcripts) != len(model_speech_segments):
        transcripts = None
    texts: list[str] = []
    for i, seg in enumerate(model_speech_segments):
        if not keep(seg):
            continue
        piece = ""
        text = transcripts[i] if transcripts is not None else None
        if text:
            piece = text.strip()
        texts.append(piece)
    return texts


def speech_through_cue_end(
    model_speech_segments: list[tuple[float, float]],
    t_cue_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """Transcripts of segments that start before cue end (before + overlap)."""
    return _segment_texts(
        model_speech_segments,
        segment_transcripts,
        lambda seg: seg[0] < t_cue_end,
    )


def speech_after_cue_end(
    model_speech_segments: list[tuple[float, float]],
    t_cue_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """Transcripts of every segment that still has speech after cue end."""
    return _segment_texts(
        model_speech_segments,
        segment_transcripts,
        lambda seg: seg[1] > t_cue_end,
    )


def score_subsequent_applicable(
    model_speech_segments: list[tuple[float, float]],
    t_cue_end: float,
    segment_transcripts: Optional[list[Optional[str]]] = None,
) -> bool:
    """True if any model speech continues or starts after t_cue_end."""
    del segment_transcripts
    return any(off > t_cue_end for _on, off in model_speech_segments)
