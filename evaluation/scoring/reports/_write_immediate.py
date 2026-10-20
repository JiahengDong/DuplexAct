#!/usr/bin/env python3
"""Write IMMEDIATE.md from the metrics cache."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evaluation.scoring.reports import _bc_verdicts
from evaluation.scoring.cue_landing import (
    CUE_ACTIONS,
    OSS_MODELS,
    cache_complete,
    cue_class,
    on_speech,
    report_out,
    report_scope,
)

from evaluation.paths import repo_root

_REPO = repo_root()
CACHE = _REPO / "evaluation" / "scoring" / "cache"
OUT = report_out(_REPO / "evaluation" / "scoring" / "reports" / "IMMEDIATE.md")

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
CLOSED = {
    "gpt-realtime-2.1",
    "gemini-3.1-flash-live",
    "grok-voice",
    "qwen3.5-omni-flash-realtime",
}
WITHHOLD_PAIRS: set[tuple[str, str]] = set()
_SCOPE = report_scope()
if _SCOPE is not None:
    WITHHOLD_PAIRS |= {
        (m, a) for m in MODELS for a in ALL if (m, a) not in _SCOPE
    }
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


def landing_counts(rows: list[dict]) -> tuple[int, int, int, int]:
    """Return (n_known, speaking, not_speaking, unknown)."""
    sp = ns = unk = 0
    for r in rows:
        c = cue_class(r)
        if c == "speaking":
            sp += 1
        elif c in ("no_reply", "finished", "missing_cue"):
            ns += 1
        else:
            unk += 1
    return sp + ns, sp, ns, unk


def drop_never_stopped(model: str) -> bool:
    """Stop's never_stopped is a real zero under the pad protocol, so it is never dropped."""
    return False


def pop(rows: list[dict], action: str, model: str) -> list[dict]:
    xs = list(rows)
    if action in CUE_ACTIONS:
        xs = [r for r in xs if on_speech(r)]
    if action == "stop" and drop_never_stopped(model):
        xs = [r for r in xs if r.get("immediate_tag") != "never_stopped"]
    return xs


def fmt(rows: list[dict], action: str, model: str) -> str:
    xs = pop(rows, action, model)
    vals = [int(r["immediate"]) for r in xs if r.get("immediate") is not None]
    if not vals:
        return "n/a"
    return f"{100 * sum(vals) / len(vals):.1f}%"


def main() -> None:
    by_ma: dict[tuple[str, str], list] = defaultdict(list)
    by_mc: dict[tuple[str, str, str], list] = defaultdict(list)
    by_ms: dict[tuple[str, str, str], list] = defaultdict(list)
    never_stopped: dict[str, int] = defaultdict(int)
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
        if a == "stop" and r.get("immediate_tag") == "never_stopped":
            never_stopped[m] += 1

    _bc_verdicts.assert_complete()
    complete = {(m, a) for (m, a), rs in by_ma.items() if cache_complete(m, a, len(rs))}

    def cell(m, a, subset=None) -> str:
        if (m, a) in WITHHOLD_PAIRS:
            return "—"
        if (m, a) not in complete:
            return "—"
        return fmt(subset if subset is not None else by_ma[(m, a)], a, m)

    ns_txt = ", ".join(
        f"{m} {never_stopped.get(m, 0)}"
        for m in MODELS
        if (m, "stop") in complete
        and (m, "stop") not in WITHHOLD_PAIRS
        and never_stopped.get(m, 0)
    ) or "none"

    lines = [
        "# Immediate results",
        "",
        "Automatic Immediate scores from `evaluation/scoring/`. Generated from the metrics cache after a full Immediate pass.",
        "",
        "**Inclusion.** A cell is shown only if that model finished the **full** action: inference count = dataset *N* and Immediate cache count = dataset *N*. Partial runs are omitted (`—`).",
        "",
        "**Population.** Listen actions (Wait, Backchannel, Ignore, Take turn) and Interrupt: all scored samples. Stop / Hold floor / Recover: cue landed while the model was **speaking** (post-lead-in turn overlaps cue onset). Forced is allowed. Cue in silence (`no_reply`) or after the model already finished is coverage, not in the denominator. See `CUE_LANDING.md`. `n/a` means the action/condition/scenario was fully run but the scored set is empty.",
        "",
        "**Hold floor Immediate:** max silence gap ≤ 0.50 s in `[t_cue_start, t_cue_end + 1.0]`.",
        "",
        "**Timing (latency / Δ / slack).** 0/1 rates only in this file. Take turn / Backchannel / Stop / Recover / Interrupt diagnostics are in `evaluation/scoring/reports/TIMING.md`.",
        "**Interrupt.** Immediate requires an onset at least 1.0 s before the user finishes, which a VAD-gated realtime API cannot produce. Read this column together with Subsequent and Ignore.",
        "",
        "**Stop Immediate exclusion.** `never_stopped` is kept as a real zero. `not_speaking_at_cue` / finished / no_reply are out of the headline denominator for every model.",
        "",
        "**Coverage.** A `—` means that model did not finish the full action.",
        "",
        "**Backchannel Immediate is two-stage: the model backchannelled, and the backchannel carried the function the condition asked for.** A ≤3-word segment starting inside the user's turn is only a candidate. Stage 1 (text) confirms it is a backchannel rather than the opening of a full turn; stage 2 (audio) checks that it carries the condition's function. Both stages are blind to the condition label, the human `bc_text`, and `bc_type_original`. The rule-only and screened columns are broken out below. Method and per-sample verdicts: `BC_TWO_STAGE.md`.",
        "",
        "## Expected *N* per model (when complete)",
        "",
        "| Action | *N* | Notes |",
        "|---|---:|---|",
        "| Wait | 600 | 200 × 3 conditions |",
        "| Backchannel | 400 | 200 × 2 |",
        "| Ignore | 400 | 200 × 2 |",
        "| Take turn | 600 | 200 × 3 |",
        "| Stop | 200 | speaking-at-cue *n* varies |",
        "| Hold floor | 891 | speaking-at-cue *n* varies |",
        "| Recover | 1200 | 200 × 6; speaking-at-cue *n* varies |",
        "| Interrupt | 400 | 200 × 2 |",
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

    lines += [
        "",
        "## Backchannel: where the rule-only number goes",
        "",
        "**rule** is a ≤3-word segment starting inside the user's turn. **screened** is what survives stage 1, i.e. the model really did backchannel. **function** adds stage 2, the backchannel carrying the condition's function, and is the Backchannel column in section 1. Counts are out of the same denominator as the rate.",
        "",
        "Collapse between *rule* and *screened* is the rule's false positives; collapse between *screened* and *function* is a model that backchannels but cannot make the affirmative/uncertain distinction audible.",
        "",
        "| Model | rule | screened | function | survives stage 1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        if (m, "backchannel") in WITHHOLD_PAIRS or (m, "backchannel") not in complete:
            continue
        xs = [r for r in pop(by_ma[(m, "backchannel")], "backchannel", m)
              if r.get("bc_rule_immediate") is not None]
        if not xs:
            lines.append(f"| {m} | n/a | n/a | n/a | n/a |")
            continue
        n = len(xs)
        n_rule = sum(int(r["bc_rule_immediate"]) for r in xs)
        n_scr = sum(int(r.get("bc_screened") or 0) for r in xs)
        n_fn = sum(int(r["immediate"]) for r in xs if r.get("immediate") is not None)
        surv = f"{100 * n_scr / n_rule:.0f}%" if n_rule else "n/a"
        lines.append(
            f"| {m} | {100 * n_rule / n:.1f}% ({n_rule}) | {100 * n_scr / n:.1f}% ({n_scr}) "
            f"| {100 * n_fn / n:.1f}% ({n_fn}) | {surv} |"
        )

    lines += [
        "",
        "## Cue landing (Stop / Hold floor / Recover)",
        "",
        "Where the injected cue sat, **before** Immediate scoring. **speaking** = cue overlapped post-lead-in model speech (this is the Immediate denominator; forced allowed). **not speaking** = model silent at cue (`no_reply` or already finished). Interrupt is not a cue-into-speech action and is omitted here. Full split (`no_reply` / `finished`) is in `CUE_LANDING.md`.",
        "",
        "| Model | Stop speaking | Stop not | Hold speaking | Hold not | Recover speaking | Recover not |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cue_acts = ("stop", "hold_floor", "respond_and_recover")
    for m in MODELS:
        if not any((m, a) in complete for a in cue_acts):
            continue
        nums = []
        for a in cue_acts:
            if (m, a) in WITHHOLD_PAIRS or (m, a) not in complete:
                nums.extend(["—", "—"])
            else:
                _n, sp, ns, _unk = landing_counts(by_ma[(m, a)])
                nums.extend([str(sp), str(ns)])
        lines.append("| " + " | ".join([m] + nums) + " |")

    lines += [
        "",
        "Per-action counts (dataset *N*; speaking share = speaking / *N*):",
        "",
    ]
    for a, title, exp_n in (
        ("stop", "Stop", 200),
        ("hold_floor", "Hold floor", 891),
        ("respond_and_recover", "Recover", 1200),
    ):
        lines += [
            f"### {title}  (expected *N*={exp_n})",
            "",
            "| Model | N | speaking | not speaking | speaking share |",
            "|---|---:|---:|---:|---:|",
        ]
        for m in MODELS:
            if (m, a) in WITHHOLD_PAIRS:
                lines.append(f"| {m} | — | — | — | — |")
                continue
            if (m, a) not in complete:
                continue
            rows = by_ma[(m, a)]
            n, sp, ns, unk = landing_counts(rows)
            share = f"{100.0 * sp / n:.1f}% ({sp}/{n})" if n else "n/a"
            n_cell = str(n) if unk == 0 else f"{n} (+{unk} missing)"
            lines.append(f"| {m} | {n_cell} | {sp} | {ns} | {share} |")
        lines.append("")

    lines += [
        f"`never_stopped` kept as real zeros (pad protocol). Counts: {ns_txt}.",
        "",
        "## 2. Model × condition",
        "",
    ]
    for a in ALL:
        extra = " / `never_stopped`" if a == "stop" else ""
        filt = (
            f" (before speaking-at-cue{extra} filter)"
            if a in CUE_ACTIONS
            else (f" (before{extra} filter)" if extra else "")
        )
        parts = [f"{lab}={EXP_COND[a][lab]}" for lab in COND_ORDER[a]]
        lines += [
            f"### {ALF[a]}",
            "",
            f"Expected *n* per condition{filt}: {', '.join(parts)}.",
            "",
            "| Model | " + " | ".join(COND_ORDER[a]) + " |",
            "|---|" + "|".join(["---:"] * len(COND_ORDER[a])) + "|",
        ]
        for m in MODELS:
            if (m, a) not in complete:
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
        extra = " / `never_stopped`" if a == "stop" else ""
        filt = (
            f" (before speaking-at-cue{extra} filter)"
            if a in CUE_ACTIONS
            else (f" (before{extra} filter)" if extra else "")
        )
        parts = [f"{lab}={EXP_SCEN[a][lab]}" for lab in SC_ORDER[a]]
        lines += [
            f"### {ALF[a]}",
            "",
            f"Expected *n* per scenario{filt}: {', '.join(parts)}.",
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
    for m in sorted(CLOSED):
        print(m, "stop", cell(m, "stop"), "hf", cell(m, "hold_floor"), "rec", cell(m, "respond_and_recover"))


if __name__ == "__main__":
    main()
