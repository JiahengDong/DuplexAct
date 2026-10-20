#!/usr/bin/env python3
"""Classify where the simulator cue landed relative to model speech."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evaluation.scoring.cue_landing import SIM_ROOT, report_out, report_scope

from evaluation.paths import repo_root

REPO = repo_root()
OUT_MD = report_out(REPO / "evaluation" / "scoring" / "reports" / "CUE_LANDING.md")
OUT_JSON = report_out(REPO / "evaluation" / "scoring" / "reports" / "CUE_LANDING.json")

OSS = [
    "moshi",
    "flm-audio",
    "bayling-duplex",
    "raon-speechchat",
    "minicpm-o-4.5",
    "freeze-omni",
]
CLOSED = [
    "gpt-realtime-2.1",
    "gemini-3.1-flash-live",
    "qwen3.5-omni-flash-realtime",
    "grok-voice",
]
SKIP: set[tuple[str, str]] = set()
_SCOPE = report_scope()


def in_scope(action: str, model: str) -> bool:
    return _SCOPE is None or (model, action) in _SCOPE

CUE_ACTIONS = ["stop", "hold_floor", "respond_and_recover"]
ALL_ACTIONS = CUE_ACTIONS + ["interrupt_user"]
LABEL = {
    "stop": "Stop",
    "hold_floor": "Hold floor",
    "respond_and_recover": "Recover",
    "interrupt_user": "Interrupt",
}
EXPECTED = {
    "stop": 200,
    "hold_floor": 891,
    "respond_and_recover": 1200,
    "interrupt_user": 400,
}

EPS = 0.05
# One poll tick (1.0 s) + MIN_SPEAKING_S (1.0 s). Shorter than this, gpt-5.4
# typically cannot return a trigger while the turn is still open.
SHORT_TURN_S = 2.0
OVERLAP_LOOKBACK_S = 1.0


def pct(n, d):
    return None if not d else 100.0 * n / d


def fmt_pct(n, d):
    if not d:
        return "n/a"
    return f"{100.0 * n / d:5.1f}% ({n}/{d})"


def segs_of(inf):
    raw = inf.get("model_speech_segments") or []
    out = []
    for s in raw:
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            out.append((float(s[0]), float(s[1])))
        elif isinstance(s, dict):
            out.append((float(s["start"]), float(s["end"])))
    return out


def load_info(path: Path):
    return json.loads(path.read_text())


def classify_cue(rec: dict):
    inst = rec.get("instance") or {}
    inf = rec.get("inference") or {}
    trig = inf.get("trigger") or {}
    segs = segs_of(inf)

    t_cue = inst.get("t_user_int_start")
    if t_cue is None:
        t_cue = trig.get("t_user_int_start")
    if t_cue is None:
        cue = [s for s in (inf.get("simulator_speech_segments") or []) if s.get("source") == "cue"]
        t_cue = cue[0]["start"] if cue else None
    lead_end = inst.get("lead_in_duration_s")
    if lead_end is None:
        lead_end = trig.get("lead_in_duration_s")
    if lead_end is None:
        lead = [s for s in (inf.get("simulator_speech_segments") or []) if s.get("source") == "lead_in"]
        lead_end = lead[0]["end"] if lead else 0.0
    t_cue = None if t_cue is None else float(t_cue)
    lead_end = float(lead_end or 0.0)
    forced = bool(inst.get("forced") if inst.get("forced") is not None else trig.get("forced"))
    num_polls = inst.get("num_polls")
    if num_polls is None:
        num_polls = trig.get("num_polls")
    num_polls = int(num_polls or 0)
    last_reason = inst.get("last_reason") or trig.get("last_reason") or ""
    if last_reason not in ("onset_timeout", "search_timeout"):
        last_reason = "llm" if not forced else "other_forced"

    row = {
        "t_cue": t_cue,
        "lead_end": lead_end,
        "forced": forced,
        "num_polls": num_polls,
        "last_reason": last_reason,
        "class": "missing_cue",
        "finished_why": None,
        "max_turn_s": None,
        "n_turns_before_cue": 0,
    }
    if t_cue is None:
        return row

    post = []
    for s, e in segs:
        if e <= lead_end + EPS:
            continue
        s2 = max(s, lead_end)
        if e - s2 > EPS:
            post.append((s2, e))

    before = [(s, e) for s, e in post if s < t_cue - EPS]
    overlapping = [(s, e) for s, e in post if s < t_cue + EPS and e > t_cue - EPS]

    if overlapping:
        row["class"] = "speaking"
    elif not before:
        row["class"] = "no_reply"
    else:
        row["class"] = "finished"
        turns = [(s, e) for s, e in before if e <= t_cue + EPS]
        if not turns:
            turns = before
        durs = [e - s for s, e in turns]
        row["n_turns_before_cue"] = len(turns)
        row["max_turn_s"] = max(durs) if durs else 0.0
        if row["max_turn_s"] < SHORT_TURN_S:
            row["finished_why"] = "too_short"
        elif num_polls == 0:
            row["finished_why"] = "no_poll_gap"
        else:
            row["finished_why"] = "no_insert_point"
    return row


def classify_interrupt(rec: dict):
    inst = rec.get("instance") or {}
    inf = rec.get("inference") or {}
    segs = segs_of(inf)
    t0 = float(inst.get("t_user_start") or 0.0)
    t_target = inst.get("t_target")
    t_target = None if t_target is None else float(t_target)
    deadline = None if t_target is None else t_target - OVERLAP_LOOKBACK_S
    row = {
        "t_target": t_target,
        "class": "missing_t_target",
        "finished_why": None,
        "max_turn_s": None,
        "forced": False,
        "num_polls": inst.get("num_polls") or (inf.get("monitor") or {}).get("num_polls") or 0,
    }
    if t_target is None:
        return row
    if not segs:
        row["class"] = "no_reply"
        return row
    during = [(s, e) for s, e in segs if e > t0 + EPS and s < t_target + EPS]
    in_success = [(s, e) for s, e in segs if s >= t0 - EPS and s <= deadline + EPS]
    only_tail = [(s, e) for s, e in segs if deadline < s <= t_target + EPS]
    after = [(s, e) for s, e in segs if s > t_target + EPS]
    if in_success:
        row["class"] = "speaking"  # barged in before last 1s
    elif only_tail and not in_success:
        row["class"] = "finished"  # spoke in overlap tail / too late in the clip
        row["finished_why"] = "overlap_tail"
        row["max_turn_s"] = max(e - s for s, e in only_tail)
    elif during:
        row["class"] = "speaking"
    elif after:
        row["class"] = "finished"
        row["finished_why"] = "after_user"
        row["max_turn_s"] = max(e - s for s, e in after)
    else:
        row["class"] = "no_reply"
    return row


def root_for(action, model):
    return SIM_ROOT


def iter_infos(action, model):
    d = root_for(action, model) / action / model
    if not d.is_dir():
        return []
    return sorted(d.glob("*/inference_info.json"))


def analyze():
    tables = {}
    for action in ALL_ACTIONS:
        tables[action] = {}
        for model in CLOSED + OSS:
            if (action, model) in SKIP or not in_scope(action, model):
                continue
            paths = iter_infos(action, model)
            buckets = defaultdict(int)
            fin = defaultdict(int)
            forced_x = defaultdict(int)
            polls_fin = defaultdict(int)
            max_turns = []
            n_err = 0
            for p in paths:
                rec = load_info(p)
                inf = rec.get("inference") or {}
                if inf.get("errors"):
                    n_err += 1
                    continue
                if action == "interrupt_user":
                    row = classify_interrupt(rec)
                else:
                    row = classify_cue(rec)
                buckets[row["class"]] += 1
                if row["class"] == "finished" and row.get("finished_why"):
                    fin[row["finished_why"]] += 1
                    polls_fin["polls0" if row["num_polls"] == 0 else "polls1"] += 1
                    if row.get("max_turn_s") is not None:
                        max_turns.append(row["max_turn_s"])
                if action != "interrupt_user":
                    forced_x[f"{'forced' if row['forced'] else 'unforced'}_{row['class']}"] += 1
            n = sum(buckets.values())
            tables[action][model] = {
                "n": n,
                "expected": EXPECTED[action],
                "n_err_skipped": n_err,
                "n_files": len(paths),
                "speaking": buckets["speaking"],
                "no_reply": buckets["no_reply"],
                "finished": buckets["finished"],
                "missing": buckets["missing_cue"] + buckets["missing_t_target"],
                "too_short": fin["too_short"],
                "no_poll_gap": fin["no_poll_gap"],
                "no_insert_point": fin["no_insert_point"],
                "overlap_tail": fin["overlap_tail"],
                "after_user": fin["after_user"],
                "fin_polls0": polls_fin["polls0"],
                "fin_polls1": polls_fin["polls1"],
                "forced": forced_x,
                "max_turn_mean": (sum(max_turns) / len(max_turns)) if max_turns else None,
            }
    return tables


def cell(n, d):
    if not d:
        return "—"
    return f"{100.0 * n / d:4.1f}% ({n})"


def write_md(tables):
    lines = []
    a = lines.append
    a("# Simulator cue landing")
    a("")
    a("Cue in speaking = cue overlapped model speech. No response = model never spoke after lead-in. Cue after finished = model spoke, then went silent before the cue.")
    a("")
    for action in CUE_ACTIONS:
        a(f"## {LABEL[action]}")
        a("")
        a("| Model | N | cue in speaking | no response | cue after finished |")
        a("|---|---:|---:|---:|---:|")
        for model in CLOSED + OSS:
            if not in_scope(action, model):
                a(f"| {model} | — | — | — | — |")
                continue
            t = tables[action].get(model)
            if not t:
                continue
            sp = t["speaking"]
            nr = t["no_reply"]
            fi = t["finished"]
            n = sp + nr + fi
            a(f"| {model} | {n} | {sp} | {nr} | {fi} |")
        a("")
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    tables = analyze()
    OUT_JSON.write_text(json.dumps(tables, indent=2))
    write_md(tables)
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    for action in ALL_ACTIONS:
        ns = [tables[action][m]["n"] for m in tables[action]]
        rng = f"N range {min(ns)} {max(ns)}" if ns else "out of scope"
        print(action, "models", len(tables[action]), rng)
