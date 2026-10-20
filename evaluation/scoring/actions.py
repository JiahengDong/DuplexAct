"""Registry of the eight actions.

Subsequent checks are shared across conditions of the same action.
"""
from evaluation.direct_replay.metrics import backchannel, ignore, take_turn, wait
from evaluation.simulator_replay.metrics import hold_floor, interrupt, recover, stop
from evaluation.scoring.framework import (
    AUDIO_JUDGE_MODEL,
    TEXT_JUDGE_MODEL,
    ActionEval,
    Check,
    ConditionSpec,
    ImmediateInterval,
    ImmediateSpec,
    SubsequentSpec,
    UNSPECIFIED_INTERVAL,
)

ACTIONS: dict[str, ActionEval] = {
    wait.ACTION.id: wait.ACTION,
    take_turn.ACTION.id: take_turn.ACTION,
    stop.ACTION.id: stop.ACTION,
    hold_floor.ACTION.id: hold_floor.ACTION,
    interrupt.ACTION.id: interrupt.ACTION,
    backchannel.ACTION.id: backchannel.ACTION,
    recover.ACTION.id: recover.ACTION,
    ignore.ACTION.id: ignore.ACTION,
}

__all__ = [
    "ACTIONS",
    "AUDIO_JUDGE_MODEL",
    "TEXT_JUDGE_MODEL",
    "ActionEval",
    "Check",
    "ConditionSpec",
    "ImmediateInterval",
    "ImmediateSpec",
    "SubsequentSpec",
    "UNSPECIFIED_INTERVAL",
    "backchannel",
    "hold_floor",
    "ignore",
    "interrupt",
    "recover",
    "stop",
    "take_turn",
    "wait",
]
