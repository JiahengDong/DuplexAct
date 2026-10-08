"""Where the simulator cue sat relative to post-lead-in model speech.

Headline Immediate for Stop / Hold floor / Recover uses cue_class ==
\"speaking\" (forced allowed). Silence and already-finished are coverage,
not Recover/Stop/Hold trials.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from evaluation.paths import repo_root

# $DUPLEXACT_ROOT / $BENCH_ROOT override the repo root (cluster or pod layouts).
REPO = repo_root()
OSS_ROOT = REPO / "runpod_results" / "evaluation" / "simulator" / "results"
CL_ROOT = REPO / "evaluation" / "simulator" / "results"

OSS_MODELS = {
    "moshi",
    "flm-audio",
    "bayling-duplex",
    "raon-speechchat",
    "minicpm-o-4.5",
    "freeze-omni",
}
EXP_N = {
    "wait": 600,
    "backchannel": 400,
    "ignore": 400,
    "turn_taking": 600,
    "stop": 200,
    "hold_floor": 891,
    "respond_and_recover": 1200,
    "interrupt_user": 400,
}


def sim_root(model: str) -> Path:
    return OSS_ROOT if model in OSS_MODELS else CL_ROOT


def report_out(path: Path) -> Path:
    """Insert $REPORT_SUFFIX before the extension, so a re-scored run can be
    written alongside a committed report instead of replacing it."""
    sfx = os.environ.get("REPORT_SUFFIX", "")
    return path.with_name(f"{path.stem}{sfx}{path.suffix}") if sfx else path


def report_scope() -> Optional[set[tuple[str, str]]]:
    """(model, action) pairs the report may show, from the $REPORT_SCOPE manifest,
    or None when unset (show everything). Pairs outside the scope are withheld
    rather than dropped, so a table never mixes re-run and pre-fix numbers and
    still names what is outstanding."""
    src = os.environ.get("REPORT_SCOPE", "")
    if not src:
        return None
    waves = json.loads(Path(src).read_text()).get("waves") or []
    return {(m, w["action"]) for w in waves for m in w["models"]}


DR_ROOT = REPO / "evaluation" / "direct_replay" / "results" / "inference_results"
RUNPOD_DR_ROOT = (
    REPO / "runpod_results" / "evaluation" / "direct_replay" / "results" / "inference_results"
)
DR_ACTIONS = {"wait", "backchannel", "ignore", "turn_taking"}

# Direct-replay slugs whose authoritative inference was produced on the pod, so
# the Spartan copy under DR_ROOT is a superseded earlier run and must not be
# read. Same split OSS_MODELS already draws for the simulator tree.
#
# The split is an artifact of Spartan filing pulled-back output separately. On
# the pod there is one tree, so scoring there sets $BENCH_POD_DR_MODELS empty.
POD_DR_MODELS = {
    m for m in (os.environ.get("BENCH_POD_DR_MODELS", "vita-1.5")).split(",") if m.strip()
}


def dr_root(model: str) -> Path:
    return RUNPOD_DR_ROOT if model in POD_DR_MODELS else DR_ROOT


@lru_cache(maxsize=None)
def n_error_samples(model: str, action: str) -> int:
    """Samples whose inference_info.json carries a non-empty `errors` list.

    run_metrics.score_one drops these instead of scoring a transport failure
    as a model that stayed silent, so they are legitimately missing from the
    cache and must not be charged against completeness. Transient failures
    (403, 1011, keepalive timeout) should be re-run to zero; what is expected
    to survive here is the permanent kind, e.g. a provider refusing the turn
    outright on content policy."""
    if action in DR_ACTIONS:
        paths = dr_root(model).glob(f"{action}/*/{model}/*/inference_info.json")
    else:
        paths = (sim_root(model) / action / model).glob("*/inference_info.json")
    n = 0
    for p in paths:
        try:
            info = json.loads(p.read_text())
        except Exception:
            continue
        if (info.get("inference") or {}).get("errors"):
            n += 1
    return n


# Errored samples we will forgive when deciding completeness. Bounded on
# purpose: subtracting an unbounded error count would invert the gate, letting
# a model that failed almost every sample pass more easily than one that missed
# a handful, and publishing a cell computed over the few samples that survived.
MAX_FORGIVEN_ERRORS = 5


def cache_complete(model: str, action: str, n: int) -> bool:
    exp = EXP_N.get(action)
    if exp is None:
        return False
    n_err = n_error_samples(model, action)
    # Upper bound covers a cache written before score_one started dropping
    # errored samples (it still holds an unscorable entry per failure); lower
    # bound covers one written after.
    if n_err <= MAX_FORGIVEN_ERRORS and exp - n_err <= n <= exp:
        return True
    # RunPod raon hold_floor is 887/891; show the cell.
    if model in OSS_MODELS and n >= exp - 5:
        return True
    return False
CUE_ACTIONS = {"stop", "hold_floor", "respond_and_recover"}
EPS = 0.05


def segs_of(inf: dict) -> list[tuple[float, float]]:
    raw = inf.get("model_speech_segments") or []
    out = []
    for s in raw:
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            out.append((float(s[0]), float(s[1])))
        elif isinstance(s, dict):
            out.append((float(s["start"]), float(s["end"])))
    return out


def classify_cue(rec: dict) -> str:
    """speaking | no_reply | finished | missing_cue | error."""
    inf = rec.get("inference") or {}
    if inf.get("errors"):
        return "error"
    inst = rec.get("instance") or {}
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
    if t_cue is None:
        return "missing_cue"
    t_cue = float(t_cue)
    lead_end = float(lead_end or 0.0)

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
        return "speaking"
    if not before:
        return "no_reply"
    return "finished"


def inference_info_path(action: str, model: str, sample_id: str) -> Path:
    return sim_root(model) / action / model / sample_id / "inference_info.json"


@lru_cache(maxsize=None)
def _class_from_disk(action: str, model: str, sample_id: str) -> Optional[str]:
    p = inference_info_path(action, model, sample_id)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return classify_cue(rec)


def cue_class(row: dict) -> Optional[str]:
    """Metrics-cache row → landing class. Interrupt/DR → None (not a cue-into-speech action)."""
    action = row.get("action")
    if action not in CUE_ACTIONS:
        return None
    cached = row.get("cue_class")
    if cached in ("speaking", "no_reply", "finished", "missing_cue", "error"):
        return cached
    sid = row.get("sample_id")
    model = row.get("model")
    if not sid or not model:
        return None
    return _class_from_disk(action, model, sid)


def on_speech(row: dict) -> bool:
    """True if this row belongs in the Stop/Hold/Recover Immediate headline.

    Forced is allowed. Cue must overlap post-lead-in speech. Interrupt and
    listen actions are not filtered here (caller should not use this).
    """
    action = row.get("action")
    if action not in CUE_ACTIONS:
        return True
    return cue_class(row) == "speaking"
