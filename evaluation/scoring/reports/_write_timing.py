#!/usr/bin/env python3
"""Write TIMING.md: Immediate timing diagnostics among Immediate = 1."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.scoring.run_metrics import (
    CACHE,
    COND_CANON,
    DR_ROOT,
    REPORTS,
    TT_WINDOW,
    drop_moshi,
    segment_texts,
    to_instance_clock,
)
from evaluation.direct_replay.metrics import backchannel, take_turn
from evaluation.simulator_replay.metrics import interrupt, recover
from evaluation.scoring.reports import _bc_verdicts
from evaluation.scoring.cue_landing import (
    CUE_ACTIONS,
    cache_complete,
    dr_root,
    inference_info_path,
    on_speech,
    report_out,
    report_scope,
)

OUT = report_out(REPORTS / "TIMING.md")

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
SIM = {"stop", "hold_floor", "respond_and_recover", "interrupt_user"}
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
TT_LAB = {
    "take_turn_immediately": "immediate",
    "take_turn_with_delay": "delay",
    "take_turn_with_overlap": "overlap",
}
BC_LAB = {
    "affirmative": "affirmative",
    "uncertain": "uncertain",
    "backchannel_affirmative": "affirmative",
    "backchannel_uncertainty": "uncertain",
}
REC_LAB = {
    "enrich_response_after_interruption": "enrich",
    "shift_topic_after_interruption": "shift topic",
    "correct_response_after_interruption": "correct",
    "repeat_response_after_interruption": "repeat",
    "ask_follow_up_after_interruption": "follow-up",
    "shorten_response_after_interruption": "shorten",
}
INT_LAB = {
    "interrupt_for_help": "help",
    "interrupt_for_safety": "safety",
}
WITHHOLD_PAIRS: set[tuple[str, str]] = set()
_SCOPE = report_scope()
if _SCOPE is not None:
    WITHHOLD_PAIRS |= {
        (m, a) for m in MODELS for a in EXP if (m, a) not in _SCOPE
    }


def info_path(r: dict) -> Path | None:
    a, m, sid = r["action"], r["model"], r["sample_id"]
    l3 = r.get("l3_raw") or r.get("condition") or ""
    if a in ("wait", "backchannel", "ignore", "turn_taking"):
        base = dr_root(m)
        p = base / a / l3 / m / sid / "inference_info.json"
        if p.exists():
            return p
        hits = list((base / a).glob(f"*/{m}/{sid}/inference_info.json"))
        return hits[0] if hits else None
    p = inference_info_path(a, m, sid)
    return p if p.exists() else None


def segs_texts(r: dict):
    p = info_path(r)
    if p is None:
        return None, None, None
    info = json.loads(p.read_text())
    inf = info.get("inference") or {}
    segs = [tuple(s) for s in inf.get("model_speech_segments") or []]
    segs = drop_moshi(segs, r["model"])
    segs = to_instance_clock(segs, r["model"], r["action"], inf)
    # instance comes back with the rest: timing_of used to re-read and re-parse
    # the same file for it, doubling the cost of the only slow path here.
    return segs, segment_texts(p.parent, info, segs), (info.get("instance") or {})


# Actions with a section in this report. wait / ignore / hold_floor are scored
# as "did the model stay quiet", a verdict with no onset to report, and have no
# section below -- but they are 23899 of the 56892 cached rows, and without this
# guard each one paid a full inference_info.json + asr.json + glob just to reach
# timing_of's final `return None` and be discarded.
TIMED_ACTIONS = {"stop", "interrupt_user", "turn_taking", "backchannel",
                 "respond_and_recover"}


def timing_of(r: dict) -> float | None:
    a = r["action"]
    if a not in TIMED_ACTIONS:
        return None
    if a == "stop":
        v = r.get("yield_latency")
        return float(v) if v is not None else None
    if a == "interrupt_user":
        v = r.get("slack")
        return float(v) if v is not None else None
    if r.get("delta_s") is not None:
        return float(r["delta_s"])

    segs, texts, inst = segs_texts(r)
    if segs is None:
        return None

    if a == "turn_taking":
        t_end = inst.get("t1_end")
        if t_end is None:
            return None
        cond = r.get("condition") or ""
        if cond not in TT_WINDOW:
            cond = COND_CANON.get(r.get("l3_raw") or "", cond)
        if cond not in TT_WINDOW:
            return None
        lo, hi = TT_WINDOW[cond]
        _s, _t, d = take_turn.score_immediate_takeover(segs, float(t_end), lo, hi, texts)
        return d
    if a == "backchannel":
        t0 = float(inst.get("clip_start") or 0.0)
        t1 = inst.get("clip_end")
        if t1 is None:
            t1 = inst.get("t_user_end")
        t_ref = inst.get("bc_start")
        if t1 is None or t_ref is None:
            return None
        _s, _t, d = backchannel.score_immediate_backchannel(
            segs, t0, float(t1), float(t_ref), texts
        )
        return d
    if a == "respond_and_recover":
        t_end = r.get("t_interruption_end")
        if t_end is None:
            t_end = inst.get("t_interruption_end")
        if t_end is None:
            return None
        _s, _t, d = recover.score_immediate_recover(segs, float(t_end), texts)
        return d
    return None


def include_row(r: dict) -> bool:
    if r.get("immediate") != 1:
        return False
    if r.get("action") in CUE_ACTIONS and not on_speech(r):
        return False
    return True


def stats(xs: list[float]) -> str:
    """median (IQR, n). The spread is the interquartile range, not the mean:
    a single pathological onset used to drag the mean far enough to flip its
    sign against the median (grok backchannel/uncertain was median 1.28 vs
    mean -2.51), which read as "earlier than the human" for a model that was
    in fact later."""
    if not xs:
        return "n/a"
    med = statistics.median(xs)
    if len(xs) < 2:
        return f"{med:.2f} (IQR n/a, n={len(xs)})"
    q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return f"{med:.2f} (IQR {q3 - q1:.2f}, n={len(xs)})"


def main() -> None:
    rows = []
    n_by_ma: dict[tuple[str, str], int] = defaultdict(int)
    for p in CACHE.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        rec = _bc_verdicts.apply(json.loads(p.read_text()))
        m, a = rec.get("model"), rec.get("action")
        if m not in MODELS or a not in EXP:
            continue
        n_by_ma[(m, a)] += 1
        rows.append(rec)
    _bc_verdicts.assert_complete()
    complete = {
        (m, a)
        for (m, a), n in n_by_ma.items()
        if cache_complete(m, a, n) and (m, a) not in WITHHOLD_PAIRS
    }

    need = [
        r
        for r in rows
        if include_row(r)
        and r["action"]
        in ("turn_taking", "backchannel", "stop", "respond_and_recover", "interrupt_user")
        and (r["model"], r["action"]) in complete
    ]
    vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for i, r in enumerate(need, 1):
        d = timing_of(r)
        if d is None:
            continue
        a = r["action"]
        cond = r.get("condition") or ""
        if a == "turn_taking":
            lab = TT_LAB.get(cond, cond)
        elif a == "backchannel":
            lab = BC_LAB.get(cond, cond)
        elif a == "respond_and_recover":
            lab = REC_LAB.get(cond, cond)
        elif a == "interrupt_user":
            lab = INT_LAB.get(cond, cond)
        else:
            lab = "direct"
        vals[(r["model"], a, lab)].append(d)
        vals[(r["model"], a, "_all")].append(d)
        if i % 2000 == 0:
            print(f"timed {i}/{len(need)}", flush=True)

    def cell(m, a, lab="_all") -> str:
        if (m, a) not in complete:
            return "—"
        return stats(vals.get((m, a, lab), []))

    def models_for(a: str) -> list[str]:
        return [m for m in MODELS if (m, a) in complete or (m, a) in WITHHOLD_PAIRS]

    lines = [
        "# Timing diagnostics",
        "",
        "Immediate timing fields from `evaluation/scoring/` (APPENDIX H). **Not** the 0/1 headline; that stays in `IMMEDIATE.md`.",
        "",
        "**Population.** Same inclusion as Immediate: full model × action only. Among **Immediate = 1**. Stop / Hold floor / Recover: speaking-at-cue (forced allowed). Interrupt: all Immediate = 1.",
        "",
        "Each cell is **median (IQR, n)** in seconds -- the median is the headline and the interquartile range is the spread. Both are outlier-robust; an earlier revision printed the mean here, but a single pathological onset was enough to flip its sign relative to the median. `n/a` = full run but no Immediate pass with a defined diagnostic; `IQR n/a` = only one value.",
        "",
        "| Action | Diagnostic | Definition | Who has a value |",
        "|---|---|---|---|",
        "| Take turn | $\\Delta$ | first in-window takeover onset $-$ $t_{\\mathrm{user\\ end}}$ | Immediate = 1 |",
        "| Backchannel | $\\Delta$ | $t_{\\mathrm{bc\\ onset}} - t_{\\mathrm{target}}$ (human `bc_start`) | Immediate = 1 (screened + function-matched) |",
        "| Stop | yield latency $L$ | $t_{\\mathrm{yield}} - t_{\\mathrm{cue\\ start}}$ | Immediate = 1 |",
        "| Recover | $\\Delta$ | first in-window recovery onset $-$ $t_{\\mathrm{int\\ end}}$ | Immediate = 1 |",
        "| Interrupt | $\\Delta$ | $t_{\\mathrm{onset}} - t_{\\mathrm{target}}$ | Immediate = 1 |",
        "",
        "Take turn windows: immediate $[0, 0.5]$, delay $[0, 3.0]$, overlap $[-1.0, 0.5]$. Recover window $[-2.0, 5.0]$. Interrupt success is $[t_{\\mathrm{user\\ start}}, t_{\\mathrm{user\\ end}}-1.0]$; $\\Delta = t_{\\mathrm{onset}} - t_{\\mathrm{target}}$ (negative = before the designed point).",
        "",
        "## Take turn ($\\Delta$ vs user end)",
        "",
        "| Model | all | immediate | delay | overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in models_for("turn_taking"):
        lines.append(
            "| "
            + " | ".join(
                [
                    m,
                    cell(m, "turn_taking"),
                    cell(m, "turn_taking", "immediate"),
                    cell(m, "turn_taking", "delay"),
                    cell(m, "turn_taking", "overlap"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Backchannel ($\\Delta$ vs human `bc_start`)",
        "",
        "Negative = earlier than the annotated human backchannel.",
        "",
        "Immediate = 1 here means screened **and** function-matched, the same gate as `IMMEDIATE_V2.md`, so these are onsets of backchannels that were confirmed to be backchannels rather than every ≤3-word segment the rule caught. That drops the population from 685 to 200 and removes the truncated turn-openings, whose onsets were never backchannel onsets in the first place. Most models are now single-digit *n*: read the medians for raon, and treat the rest as anecdote.",
        "",
        "| Model | all | affirmative | uncertain |",
        "|---|---:|---:|---:|",
    ]
    for m in models_for("backchannel"):
        lines.append(
            "| "
            + " | ".join(
                [
                    m,
                    cell(m, "backchannel"),
                    cell(m, "backchannel", "affirmative"),
                    cell(m, "backchannel", "uncertain"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Stop (yield latency)",
        "",
        "| Model | $L$ |",
        "|---|---:|",
    ]
    for m in models_for("stop"):
        lines.append(f"| {m} | {cell(m, 'stop')} |")

    lines += [
        "",
        "## Recover ($\\Delta$ vs interruption end)",
        "",
        "| Model | all | enrich | shift topic | correct | repeat | follow-up | shorten |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in models_for("respond_and_recover"):
        lines.append(
            "| "
            + " | ".join(
                [
                    m,
                    cell(m, "respond_and_recover"),
                    cell(m, "respond_and_recover", "enrich"),
                    cell(m, "respond_and_recover", "shift topic"),
                    cell(m, "respond_and_recover", "correct"),
                    cell(m, "respond_and_recover", "repeat"),
                    cell(m, "respond_and_recover", "follow-up"),
                    cell(m, "respond_and_recover", "shorten"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Interrupt ($\\Delta$ vs $t_{\\mathrm{target}}$)",
        "",
        "| Model | all | help | safety |",
        "|---|---:|---:|---:|",
    ]
    for m in models_for("interrupt_user"):
        lines.append(
            "| "
            + " | ".join(
                [
                    m,
                    cell(m, "interrupt_user"),
                    cell(m, "interrupt_user", "help"),
                    cell(m, "interrupt_user", "safety"),
                ]
            )
            + " |"
        )

    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
