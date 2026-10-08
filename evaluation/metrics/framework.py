"""Shared Immediate + Subsequent evaluation schema.

Every Level-2 action is scored as two halves:

- Immediate: did the model do the right thing in the annotated timing
  interval? The *interval* is condition-specific and is left unspecified
  until we go through each action.
- Subsequent: did later behaviour correctly use / respect what happened
  during or after that Immediate window? The Subsequent checks are the
  same for every condition of an action unless a later pass adds a tweak.

Scores are 0/1 per check. Immediate windows are not inherited from the
old B/T/R scorers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Paper evaluation judges. GPT-5.4 does all the text judging: every
# action's Subsequent (and old R_i), and stage 1 of backchannel Immediate.
# Backchannel stage 2 is the only audio-in judge. Simulator cue/branch
# decisions also use GPT-5.4, but that is apparatus, not this scorer.
TEXT_JUDGE_MODEL = "gpt-5.4"
# Not read by the scorer, and not the pin. Both backchannel judges are
# pinned in bc_two_stage_judge.py, where stage 1 is TEXT_MODEL
# ("gpt-5.4-2026-03-05", a dated snapshot of TEXT_JUDGE_MODEL above) and
# stage 2 is AUDIO_MODEL. This name survives only as a re-export.
AUDIO_JUDGE_MODEL = "gpt-audio"


@dataclass(frozen=True)
class ImmediateInterval:
    """Decision interval for Immediate scoring, filled per condition later.

    Use one of:
    - named span: ``start_key`` / ``end_key`` (metadata or timestamp fields)
    - relative window: ``anchor`` + ``lo_s`` / ``hi_s``
    - absolute window: ``abs_start_s`` / ``abs_end_s``
    """

    start_key: Optional[str] = None
    end_key: Optional[str] = None
    anchor: Optional[str] = None
    lo_s: Optional[float] = None
    hi_s: Optional[float] = None
    abs_start_s: Optional[float] = None
    abs_end_s: Optional[float] = None
    notes: str = ""

    def is_specified(self) -> bool:
        if self.start_key and self.end_key:
            return True
        if self.lo_s is not None or self.hi_s is not None:
            return True
        return self.abs_start_s is not None and self.abs_end_s is not None


UNSPECIFIED_INTERVAL = ImmediateInterval()


@dataclass(frozen=True)
class Check:
    """One 0/1 evaluation item."""

    id: str
    name: str
    pass_when: str
    fail_when: str
    notes: str = ""


@dataclass(frozen=True)
class ImmediateSpec:
    definition: str
    criterion: str
    interval: ImmediateInterval = UNSPECIFIED_INTERVAL
    checks: tuple[Check, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class SubsequentSpec:
    definition: str
    checks: tuple[Check, ...]
    notes: str = ""


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    data_dir: str
    description: str
    immediate: ImmediateSpec
    subsequent: SubsequentSpec
    # What this condition requires when Subsequent says "as the condition
    # requires". Not a separate Subsequent metric.
    required_adjustment: str = ""


@dataclass(frozen=True)
class ActionEval:
    id: str
    title: str
    definition: str
    data_action: str
    immediate_definition: str
    immediate_criterion: str
    subsequent: SubsequentSpec
    immediate_checks: tuple[Check, ...] = ()
    conditions: dict[str, ConditionSpec] = field(default_factory=dict)
    human_support: str = ""


def make_conditions(
    *,
    immediate_definition: str,
    immediate_criterion: str,
    subsequent: SubsequentSpec,
    rows: list[tuple[str, str, str, str]],
    intervals: Optional[dict[str, ImmediateInterval]] = None,
    immediate_checks: tuple[Check, ...] = (),
    immediate_notes: str = "",
) -> dict[str, ConditionSpec]:
    """Build per-condition specs.

    Each row is (condition_id, data_dir, description, required_adjustment).
    Subsequent and Immediate checks are shared by reference. Immediate
    interval defaults to empty unless ``intervals[condition_id]`` is set.
    """
    intervals = intervals or {}
    conditions: dict[str, ConditionSpec] = {}
    for condition_id, data_dir, description, required_adjustment in rows:
        conditions[condition_id] = ConditionSpec(
            id=condition_id,
            data_dir=data_dir,
            description=description,
            required_adjustment=required_adjustment,
            immediate=ImmediateSpec(
                definition=immediate_definition,
                criterion=immediate_criterion,
                interval=intervals.get(condition_id, ImmediateInterval()),
                checks=immediate_checks,
                notes=immediate_notes,
            ),
            subsequent=subsequent,
        )
    return conditions
