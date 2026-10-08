#!/usr/bin/env python3
"""Write SUBSEQUENT.md from the metrics cache (same layout as IMMEDIATE.md)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evaluation.metrics.reports import _bc_verdicts
from evaluation.metrics.cue_landing import (
    CUE_ACTIONS,
    cache_complete,
    on_speech,
    report_out,
    report_scope,
)

from evaluation.paths import repo_root

_REPO = repo_root()
CACHE = _REPO / "evaluation" / "metrics" / "cache"
OUT = report_out(_REPO / "evaluation" / "metrics" / "reports" / "SUBSEQUENT.md")

EXP = {
    "wait": 600,
    "backchannel": 400,
    "ignore": 400,
    "turn_taking": 600,
    "stop": 200,
    "hold_floor": 891,
    "respond_and_recover": 1200,
    "interrupt_user": 400,
}
ALL = list(EXP)
SIM = {"stop", "hold_floor", "respond_and_recover", "interrupt_user"}
DR = {"wait", "backchannel", "ignore", "turn_taking"}
ALF = {
    "wait": "Wait",
    "backchannel": "Backchannel",
    "ignore": "Ignore",
    "turn_taking": "Take turn",
    "stop": "Stop",
    "hold_floor": "Hold floor",
    "respond_and_recover": "Recover",
    "interrupt_user": "Interrupt",
}
MODELS = [
    "bayling-duplex",
    "flm-audio",
    "freeze-omni",
    "gemini-3.1-flash-live",
    "gpt-realtime-2.1",
    "grok-voice",
    "lychee-fd",
    "mini-omni2",
    "minicpm-o-4.5",
    "moshi",
    "qwen3.5-omni-flash-realtime",
    "raon-speechchat",
    "vita-1.5",
    "voila-autonomous",
]
COND_LAB = {
    "inter_sentential_pause": "inter-sentential",
    "intra_sentential_pause": "intra-sentential",
    "no_pause": "no pause",
    "backchannel_affirmative": "affirmative",
    "backchannel_uncertainty": "uncertain",
    "ignore_background_speech_during_user_speech": "background",
    "ignore_other_directed_speech_during_user_speech": "other-directed",
    "take_turn_immediately": "immediate",
    "take_turn_with_delay": "delay",
    "take_turn_with_overlap": "overlap",
    "direct_interruption": "direct",
    "continue_current_response": "continue",
    "continue_but_shorten": "shorten",
    "continue_with_simplification": "simplify",
    "continue_with_background_speech": "background speech",
    "continue_with_other_directed_speech": "other-directed",
    "enrich_response_after_interruption": "enrich",
    "shift_topic_after_interruption": "shift topic",
    "correct_response_after_interruption": "correct",
    "repeat_response_after_interruption": "repeat",
    "ask_follow_up_after_interruption": "follow-up",
    "shorten_response_after_interruption": "shorten",
    "interrupt_for_help": "help",
    "interrupt_for_safety": "safety",
}
COND_ORDER = {
    "wait": ["inter-sentential", "intra-sentential", "no pause"],
    "backchannel": ["affirmative", "uncertain"],
    "ignore": ["background", "other-directed"],
    "turn_taking": ["immediate", "delay", "overlap"],
    "stop": ["direct"],
    "hold_floor": [
        "continue",
        "shorten",
        "simplify",
        "background speech",
        "other-directed",
    ],
    "respond_and_recover": [
        "enrich",
        "shift topic",
        "correct",
        "repeat",
        "follow-up",
        "shorten",
    ],
    "interrupt_user": ["help", "safety"],
}
EXP_COND = {
    "wait": {
        "inter-sentential": 200,
        "intra-sentential": 200,
        "no pause": 200,
    },
    "backchannel": {"affirmative": 200, "uncertain": 200},
    "ignore": {"background": 200, "other-directed": 200},
    "turn_taking": {"immediate": 200, "delay": 200, "overlap": 200},
    "stop": {"direct": 200},
    "hold_floor": {
        "continue": 200,
        "shorten": 131,
        "simplify": 200,
        "background speech": 200,
        "other-directed": 160,
    },
    "respond_and_recover": {
        "enrich": 200,
        "shift topic": 200,
        "correct": 200,
        "repeat": 200,
        "follow-up": 200,
        "shorten": 200,
    },
    "interrupt_user": {"help": 200, "safety": 200},
}
SC_LAB = {
    "casual_conversation": "Casual",
    "meeting": "Meeting",
    "interview": "Interview",
    "customer_service": "Customer service",
    "unknown": "Unknown (synth)",
    "others": "Others",
}
SC_ORDER = {
    "wait": ["Casual", "Meeting", "Interview", "Customer service"],
    "backchannel": ["Casual", "Meeting", "Interview", "Customer service"],
    "ignore": ["Casual", "Meeting", "Interview", "Customer service"],
    "turn_taking": ["Casual", "Meeting", "Interview", "Customer service"],
    "stop": ["Casual", "Meeting", "Interview"],
    "hold_floor": [
        "Casual",
        "Meeting",
        "Interview",
        "Customer service",
    ],
    "respond_and_recover": [
        "Casual",
        "Meeting",
        "Interview",
        "Customer service",
    ],
    "interrupt_user": [
        "Casual",
        "Meeting",
        "Interview",
        "Customer service",
        "Others",
    ],
}
EXP_SCEN = {
    "wait": {"Casual": 230, "Meeting": 72, "Interview": 235, "Customer service": 63},
    "backchannel": {
        "Casual": 106,
        "Meeting": 68,
        "Interview": 171,
        "Customer service": 55,
    },
    "ignore": {
        "Casual": 100,
        "Meeting": 100,
        "Interview": 100,
        "Customer service": 100,
    },
    "turn_taking": {
        "Casual": 255,
        "Meeting": 123,
        "Interview": 122,
        "Customer service": 100,
    },
    "stop": {"Casual": 158, "Meeting": 37, "Interview": 5},
    "hold_floor": {
        "Casual": 222,
        "Meeting": 252,
        "Interview": 309,
        "Customer service": 108,
    },
    "respond_and_recover": {
        "Casual": 574,
        "Meeting": 310,
        "Interview": 166,
        "Customer service": 150,
    },
    "interrupt_user": {
        "Casual": 50,
        "Meeting": 50,
        "Interview": 50,
        "Customer service": 50,
        "Others": 200,
    },
}


# Was: BayLing's listen Subsequent withheld because it was scored off ASR of
# chopped vocoder audio (0.54 words/s). Resolved 2026-09-19 -- the slug now
# scores off its own native text, reconstructed offline from events.jsonl, hit
# on 2000/2000 samples, and run_metrics no longer falls back to ASR for
# LIVE_TRANSCRIPT_MODELS at all. Scope alone decides now.
WITHHOLD_PAIRS: set[tuple[str, str]] = set()
_SCOPE = report_scope()
if _SCOPE is not None:
    WITHHOLD_PAIRS |= {
        (m, a) for m in MODELS for a in ALL if (m, a) not in _SCOPE
    }


def withheld(model: str, action: str) -> bool:
    return (model, action) in WITHHOLD_PAIRS


def fmt(rows: list[dict], action: str | None = None) -> str:
    if action in CUE_ACTIONS:
        rows = [r for r in rows if on_speech(r)]
    if action in DR:
        xs = [
            1 if r.get("subsequent") == 1 else 0
            for r in rows
            if r.get("immediate") == 1
        ]
    else:
        xs = [
            int(r["subsequent"])
            for r in rows
            if r.get("subsequent") is not None and r.get("immediate") == 1
        ]
    if not xs:
        return "n/a"
    return f"{100 * sum(xs) / len(xs):.1f}% ({len(xs)})"


def main() -> None:
    by_ma: dict[tuple[str, str], list] = defaultdict(list)
    by_mc: dict[tuple[str, str, str], list] = defaultdict(list)
    by_ms: dict[tuple[str, str, str], list] = defaultdict(list)
    for p in CACHE.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        r = _bc_verdicts.apply(json.loads(p.read_text()))
        m, a = r.get("model"), r.get("action")
        if m not in MODELS or a not in EXP:
            continue
        by_ma[(m, a)].append(r)
        by_mc[(m, a, COND_LAB.get(r.get("condition") or "", r.get("condition") or ""))].append(r)
        by_ms[(m, a, SC_LAB.get(r.get("scenario") or "unknown", "Unknown (synth)"))].append(r)

    _bc_verdicts.assert_complete()
    complete = {(m, a) for (m, a), rs in by_ma.items() if cache_complete(m, a, len(rs))}

    def cell(m, a, subset=None) -> str:
        if withheld(m, a):
            return "—"
        if (m, a) not in complete:
            return "—"
        return fmt(subset if subset is not None else by_ma[(m, a)], action=a)

    lines = [
        "# Subsequent results",
        "",
        "GPT-5.4 text Subsequent (`temperature=0`, JSON YES/NO). Stop Subsequent is automatic from segments (no LLM). Backchannel function-match now sits in **Immediate**, not here: Immediate = 1 requires both that the model backchannelled and that the backchannel carried the condition's function, so this column asks only whether the later response follows on. See `IMMEDIATE_V2.md` and `BC_TWO_STAGE.md`.",
        "",
        "**Inclusion.** Same as Immediate: a cell is shown only if that model finished the **full** action (inference *N* = dataset *N* and metrics cache *N* = dataset *N*). Smoke-only or partial runs are omitted (`—`).",
        "",
        "**Denominator.** Subsequent is **conditional** on Immediate = 1. Direct Replay (Wait, Ignore, Take turn, Backchannel): denominator is all Immediate = 1; no later response is 0, not N/A; spoke but failed the later check is 0; S = 1 only if the later response meets the action requirement. Immediate fail is excluded. Stop / Hold floor / Recover: among Immediate = 1 **and** speaking-at-cue (forced allowed). Interrupt: among Immediate = 1 (full sample).",
        "",
        "**N/A.** Direct Replay `n/a` = fully run but no Immediate = 1 sample. Simulator `n/a` = no Immediate = 1 sample was Subsequent-applicable (no speech after the cue, no first yield, etc.).",
        "",
        "**Gating.** Every action: Subsequent only if Immediate = 1. Direct Replay: Immediate-success trials with no later response score 0. Simulator extra N/A gates still apply (Stop with no first yield during the cue, no speech after the Hold-floor cue, etc.).",
        "",
        "**Backchannel *n* collapsed and the rates are not comparable to the previous revision.** The Immediate gate went from the ≤3-word rule to screened-and-function-matched, which cut the conditional population from 685 to 200 — raon holds 124 of those, and most other models are single digits. No sample was re-judged: the new population is a strict subset of the old one, so every Subsequent verdict here was already in the cache. What changed is which verdicts are counted.",
        "",
        "**Coverage.** Same model roster as Immediate. Closed-source Wait Subsequent was rescored after whisper-1 ASR (jobs 30543451–30543455, 2026-09-14). Closed-source Backchannel Subsequent is often `n/a` because Immediate found no function-matched backchannel, not because ASR is missing. OSS simulator Stop / Hold / Recover / Interrupt is from `runpod_results/` (live transcripts). Grok Interrupt is the 2026-09-18 one-clip re-run.",
        "",
        "**Exclusions.** Simulator Stop Subsequent is segment-based. Hold / Recover Subsequent uses live `model_speech_transcripts` (no Whisper). Bayling-duplex Wait / Backchannel / Ignore / Take turn Subsequent is omitted (`—`) until the contiguous-speech re-run + ASR.",
        "",
        "Companion Immediate tables: `evaluation/metrics/reports/IMMEDIATE.md`.",
        "",
        "## Expected *N* per model (when complete)",
        "",
        "| Action | Dataset *N* | Subsequent notes |",
        "|---|---:|---|",
        "| Wait | 600 | Conditional on Immediate; no later response = 0. Bayling withheld. |",
        "| Backchannel | 400 | Conditional on Immediate (BC produced **and** function matched); no later speech = 0. Bayling withheld. |",
        "| Ignore | 400 | Conditional on Immediate; no later speech = 0. Bayling withheld. |",
        "| Take turn | 600 | Conditional on Immediate. Bayling withheld. |",
        "| Stop | 200 | Conditional on Immediate among speaking-at-cue. Live transcript / segments. |",
        "| Hold floor | 891 | Conditional on Immediate among speaking-at-cue. Live transcript. |",
        "| Recover | 1200 | Conditional on Immediate among speaking-at-cue. Live transcript. |",
        "| Interrupt | 400 | Gated on Immediate. Grok `—`. |",
        "",
        "## 1. Model × action",
        "",
        "| Model | Wait | Backchannel | Ignore | Take turn | Stop | Hold floor | Recover | Interrupt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        cells = [cell(m, a) for a in ALL]
        if _SCOPE is None and all(c == "—" for c in cells):
            continue
        lines.append("| " + " | ".join([m] + cells) + " |")

    lines += ["", "## 2. Model × condition", ""]
    for a in ALL:
        parts = [f"{lab}={EXP_COND[a][lab]}" for lab in COND_ORDER[a]]
        lines += [
            f"### {ALF[a]}",
            "",
            f"Expected dataset *n* per condition: {', '.join(parts)}. "
            + (
                "Rates below are conditional on Immediate = 1 (parentheses); no later response is 0."
                if a in DR
                else "Rates below use Subsequent-applicable *n* (in parentheses)."
            ),
            "",
            "| Model | " + " | ".join(COND_ORDER[a]) + " |",
            "|---|" + "|".join(["---:"] * len(COND_ORDER[a])) + "|",
        ]
        for m in MODELS:
            if (m, a) not in complete:
                if a not in SIM:
                    continue
                if not any((m, x) in complete for x in SIM):
                    continue
                continue
            cells = [cell(m, a, by_mc[(m, a, lab)]) for lab in COND_ORDER[a]]
            lines.append("| " + " | ".join([m] + cells) + " |")
        lines.append("")

    lines += [
        "## 3. Model × scenario",
        "",
        "Synthesized Hold floor (360) and Recover (600) use the same four conversation scenarios they were generated under; they are not a fifth scene. Hold floor synth is uneven (Casual 60, Meeting / Interview / Customer service 100 each) because 40 other-directed samples had no `other` turn. Others = Interrupt safety (home + driving, 200).",
        "",
    ]
    for a in ALL:
        parts = [f"{lab}={EXP_SCEN[a][lab]}" for lab in SC_ORDER[a]]
        lines += [
            f"### {ALF[a]}",
            "",
            f"Expected dataset *n* per scenario: {', '.join(parts)}. "
            + (
                "Rates below are conditional on Immediate = 1 (parentheses); no later response is 0."
                if a in DR
                else "Rates below use Subsequent-applicable *n* (in parentheses)."
            ),
            "",
            "| Model | " + " | ".join(SC_ORDER[a]) + " |",
            "|---|" + "|".join(["---:"] * len(SC_ORDER[a])) + "|",
        ]
        for m in MODELS:
            if (m, a) not in complete:
                continue
            cells = [cell(m, a, by_ms[(m, a, lab)]) for lab in SC_ORDER[a]]
            lines.append("| " + " | ".join([m] + cells) + " |")
        lines.append("")

    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUT}")
    for m in (
        "gpt-realtime-2.1",
        "gemini-3.1-flash-live",
        "grok-voice",
        "qwen3.5-omni-flash-realtime",
    ):
        print("wait", m, cell(m, "wait"), "bc", cell(m, "backchannel"))


if __name__ == "__main__":
    main()
