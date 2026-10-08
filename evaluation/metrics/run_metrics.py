#!/usr/bin/env python3
"""Immediate + Subsequent scorer for all 8 actions.

Immediate is automatic. Subsequent is GPT-5.4 (text) except Stop
(automatic yielding). Backchannel is the exception: what this file
writes is only the <=3-word candidate detector, and the reported
Immediate is the two-stage judgement in bc_two_stage_judge.py, joined
on at report time. Hold floor max gap is 0.50 s (locked).

Usage:
    python run_metrics.py --immediate_only
    python run_metrics.py --model gpt-realtime-2.1 --action wait
    python run_metrics.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from evaluation.paths import repo_root

REPO = repo_root()
sys.path.insert(0, str(REPO))

from evaluation.metrics import (  # noqa: E402
    TEXT_JUDGE_MODEL,
    backchannel,
    hold_floor,
    ignore,
    interrupt,
    recover,
    stop,
    take_turn,
    wait,
)
from evaluation.metrics.cue_landing import (  # noqa: E402
    CUE_ACTIONS,
    OSS_MODELS,
    OSS_ROOT,
    POD_DR_MODELS,
    RUNPOD_DR_ROOT,
    classify_cue,
    on_speech,
    sim_root,
)

DATA = REPO / "evaluation" / "direct_replay" / "data"
DR_ROOT = REPO / "evaluation" / "direct_replay" / "results" / "inference_results"
SIM_ROOT = REPO / "evaluation" / "simulator" / "results"
CACHE = REPO / "evaluation" / "metrics" / "cache"
REPORTS = REPO / "evaluation" / "metrics" / "reports"

DR_ACTIONS = ["wait", "backchannel", "ignore", "turn_taking"]
SIM_ACTIONS = ["stop", "hold_floor", "respond_and_recover", "interrupt_user"]
ALL_ACTIONS = DR_ACTIONS + SIM_ACTIONS

SKIP_MODEL_SUBSTR = ("backup",)
LIVE_TRANSCRIPT_MODELS = {"bayling-duplex", "raon-speechchat"}

DS2SC = {
    "AMI": "meeting",
    "Candor": "casual_conversation",
    "CORAAL": "interview",
    "Loqui": "customer_service",
}
SC_LABEL = {
    "casual_conversation": "Casual conversation",
    "meeting": "Meeting",
    "interview": "Interview",
    "customer_service": "Customer service",
    "others": "Others",
}
COND_CANON = {
    "floor_holding": "no_pause",
    "inter_sentential": "inter_sentential_pause",
    "intra_sentential": "intra_sentential_pause",
    "immediate_take_turn": "take_turn_immediately",
    "delayed_take_turn": "take_turn_with_delay",
    "slight_overlap_take_turn": "take_turn_with_overlap",
}
TT_WINDOW = {
    "take_turn_immediately": (0.0, 0.5),
    "take_turn_with_delay": (0.0, 3.0),
    "take_turn_with_overlap": (-1.0, 0.5),
}
ACTION_LABEL = {
    "wait": "Wait",
    "backchannel": "Backchannel",
    "ignore": "Ignore",
    "turn_taking": "Take turn",
    "stop": "Stop",
    "hold_floor": "Hold floor",
    "respond_and_recover": "Recover",
    "interrupt_user": "Interrupt",
}


def _skip_model(name: str) -> bool:
    return any(s in name for s in SKIP_MODEL_SUBSTR)


_SYNTH_SCENARIO_BY_UUID: dict[str, str] | None = None


def _synth_scenario_by_uuid() -> dict[str, str]:
    """Hold floor / Recover prepare scripts originally omitted `scenario`.
    Recover it from data_synthesize/<task>/<subcat>/<scenario>/<uuid>/."""
    global _SYNTH_SCENARIO_BY_UUID
    if _SYNTH_SCENARIO_BY_UUID is not None:
        return _SYNTH_SCENARIO_BY_UUID
    names = set(SC_LABEL) - {"others"}
    out: dict[str, str] = {}
    for task in ("hold_floor", "respond_and_recover"):
        root = REPO / "data_synthesize" / task
        if not root.exists():
            continue
        for p in root.rglob("transcript.json"):
            sc = p.parent.parent.name
            if sc not in names:
                try:
                    sc = json.loads(p.read_text()).get("scenario")
                except Exception:
                    continue
            if sc in names:
                out[p.parent.name] = sc
    _SYNTH_SCENARIO_BY_UUID = out
    return out


def scenario_of(meta: dict) -> str:
    ds = str(meta.get("dataset") or "")
    if str(ds).lower() == "synthesized":
        src = "Synthesized"
    else:
        src = "Real"
    sc = meta.get("scenario") or meta.get("setting")
    if not sc and src == "Real":
        sc = DS2SC.get(ds)
    if not sc and src == "Synthesized":
        sid = meta.get("source_id")
        if sid:
            sc = _synth_scenario_by_uuid().get(sid)
    if not sc and meta.get("hazard_type"):
        ht = str(meta["hazard_type"])
        if ht.startswith("driving") or ht.startswith("home"):
            sc = "others"
    if sc in ("task_oriented", "driving", "home"):
        sc = "customer_service" if sc == "task_oriented" else "others"
    if sc == "Loqui":
        sc = "customer_service"
    return sc if sc in SC_LABEL else (sc or "unknown")


def load_asr_transcripts(out_dir: Path, n_seg: int) -> Optional[list[Optional[str]]]:
    p = out_dir / "asr.json"
    if not p.exists():
        return None
    try:
        asr = json.loads(p.read_text())
    except Exception:
        return None
    chunks = asr.get("chunks") or []
    return None if not chunks else None  # filled by caller with segments


def words_in(chunks: list, on: float, off: float) -> str:
    out = []
    for c in chunks:
        ts = c.get("timestamp") or [None, None]
        s, e = ts[0], ts[1] if len(ts) > 1 else ts[0]
        if s is None:
            continue
        if s < off and (e if e is not None else s) > on:
            w = (c.get("text") or "").strip()
            if w:
                out.append(w)
    return " ".join(out)


def _native_segment_texts(info: dict, segs: list) -> Optional[list[Optional[str]]]:
    native = (info.get("inference") or {}).get("model_speech_transcripts")
    if isinstance(native, list) and len(native) == len(segs):
        return [t if t else None for t in native]
    return None


def segment_texts(out_dir: Path, info: dict, segs: list, model: Optional[str] = None) -> list[Optional[str]]:
    model = model or info.get("model")
    if model in LIVE_TRANSCRIPT_MODELS:
        # Native text or nothing -- deliberately NOT falling through to
        # asr.json. For these slugs ASR is not a degraded option, it is a wrong
        # one: over half the frames inside their speaking segments are all-zero
        # padding, and Whisper returns 0.5-1.2 words/s against a 2.5-3.5
        # conversational band. Falling back would quietly score a sample on
        # text known to be unusable, and it would be invisible in the output.
        # A stale asr.json makes that worse than merely wrong -- raon's direct
        # replay is being re-run, so its leftover asr.json describe an older
        # timeline than the segments they would be matched against.
        # Returning None per segment lets the caller treat the text as absent,
        # which is the honest answer.
        nat = _native_segment_texts(info, segs)
        return nat if nat is not None else [None] * len(segs)
    asr_p = out_dir / "asr.json"
    if asr_p.exists():
        try:
            asr = json.loads(asr_p.read_text())
            chunks = asr.get("chunks") or []
            if chunks:
                return [words_in(chunks, on, off) for on, off in segs]
            text = (asr.get("text") or "").strip()
            if text and len(segs) == 1:
                return [text]
        except Exception:
            pass
    nat = _native_segment_texts(info, segs)
    if nat is not None:
        return nat
    return [None] * len(segs)


def drop_moshi(segs: list, model: str) -> list:
    if model != "moshi" or not segs:
        return segs
    return sorted(segs, key=lambda s: s[0])[1:]


# Slugs too slow to stream in real time, whose recorder therefore timestamps on
# a dilated wall clock. See to_instance_clock.
DILATED_CLOCK_MODELS = {"raon-speechchat"}


def to_instance_clock(segs: list, model: str, action: str, inf: dict) -> list:
    """Map segment times onto the clock the instance metadata is written on.

    Direct replay only. Raon needs ~270 ms of wall clock per 80 ms frame, and
    SynchronizedRecorder fills the gaps by wall clock, so output.wav runs ~2.5x
    long and every segment time is stretched by that factor. The cue times it is
    compared against (t1_end, pause_start, bg_speech_start_s) live on the input
    clip's fixed timeline, so the two decouple and every timing metric reads the
    wrong instant -- infer_raon.py's docstring flags this, nothing acted on it.
    Uncorrected, take_turn scored 1.5% against 65.2% here, and wait/ignore were
    inflated because the model's speech was pushed past the interval it was
    supposed to stay quiet through.

    The simulator is deliberately excluded. There the cue is played by the
    simulator into the same recording, so cue and model speech share the dilated
    clock and stay mutually consistent; input_duration_s there is only the user
    audio, not the session, so this ratio would be meaningless.

    Validated on take_turn_immediately, where the design puts the human takeover
    0-0.5 s after t1_end: corrected onsets land at median +0.23 s (IQR 0.90),
    61% inside the window. Stripping the recorder's padding out of the audio
    instead lands at +3.84 s, because ~half the padding sits inside the speaking
    segments rather than between them.
    """
    if model not in DILATED_CLOCK_MODELS or action not in DR_ACTIONS or not segs:
        return segs
    t_in, t_out = inf.get("input_duration_s"), inf.get("output_duration_s")
    if not t_in or not t_out or t_out <= t_in:
        return segs
    k = float(t_in) / float(t_out)
    return [tuple(v * k for v in s) for s in segs]


def discover(action: Optional[str], model: Optional[str]) -> list[Path]:
    paths: list[Path] = []
    actions = [action] if action else ALL_ACTIONS
    for a in actions:
        if a in DR_ACTIONS:
            # Two trees, same split the simulator branch below draws: a pod-run
            # slug is read only from runpod_results, everyone else only from
            # Spartan, so a superseded local copy can never be picked up.
            for base, is_pod in ((DR_ROOT, False), (RUNPOD_DR_ROOT, True)):
                root = base / a
                if not root.exists():
                    continue
                for l3 in sorted(root.iterdir()):
                    if not l3.is_dir() or l3.name.startswith("_"):
                        continue
                    for mdir in sorted(l3.iterdir()):
                        if not mdir.is_dir() or _skip_model(mdir.name):
                            continue
                        if (mdir.name in POD_DR_MODELS) != is_pod:
                            continue
                        if model and mdir.name != model:
                            continue
                        paths.extend(sorted(mdir.glob("*/inference_info.json")))
        else:
            # OSS simulator: RunPod tree. Closed-source: Spartan tree.
            if model:
                root = sim_root(model) / a / model
                if root.is_dir():
                    paths.extend(sorted(root.glob("*/inference_info.json")))
            else:
                oss_action = OSS_ROOT / a
                if oss_action.is_dir():
                    for mdir in sorted(oss_action.iterdir()):
                        if (
                            mdir.is_dir()
                            and not _skip_model(mdir.name)
                            and mdir.name in OSS_MODELS
                        ):
                            paths.extend(sorted(mdir.glob("*/inference_info.json")))
                cl_action = SIM_ROOT / a
                if cl_action.is_dir():
                    for mdir in sorted(cl_action.iterdir()):
                        if not mdir.is_dir() or _skip_model(mdir.name):
                            continue
                        if mdir.name in OSS_MODELS:
                            continue
                        paths.extend(sorted(mdir.glob("*/inference_info.json")))
    return paths


def cache_path(action: str, model: str, sample_id: str) -> Path:
    return CACHE / action / model / f"{sample_id}.json"


_JUDGE_LOCK = threading.Lock()
_JUDGE_CLIENTS: Optional[list] = None
_JUDGE_IDX = 0


def _openai_keys() -> list[str]:
    primary = (os.environ.get("OPENAI_API_KEY") or "").strip()
    fallback = (os.environ.get("OPENAI_API_KEY_FALLBACK") or "").strip()
    keys = []
    for k in (primary, fallback):
        if k and k not in keys:
            keys.append(k)
    return keys


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return True
    if "credit_balance" in text or "no credits remaining" in text:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        code = err.get("code") if isinstance(err, dict) else None
        if code == "insufficient_quota":
            return True
    return getattr(exc, "code", None) == "insufficient_quota"


def _judge_clients():
    global _JUDGE_CLIENTS
    if _JUDGE_CLIENTS is None:
        from openai import OpenAI
        keys = _openai_keys()
        if not keys:
            raise RuntimeError("no OPENAI_API_KEY / OPENAI_API_KEY_FALLBACK")
        _JUDGE_CLIENTS = [OpenAI(api_key=k) for k in keys]
        print(f"[metrics] OpenAI judge ready ({len(keys)} key(s))", flush=True)
    return _JUDGE_CLIENTS


def _judge_client():
    clients = _judge_clients()
    with _JUDGE_LOCK:
        return clients[_JUDGE_IDX], _JUDGE_IDX


def _advance_key(from_idx: int) -> bool:
    global _JUDGE_IDX
    clients = _judge_clients()
    with _JUDGE_LOCK:
        if from_idx != _JUDGE_IDX:
            return True
        if _JUDGE_IDX + 1 >= len(clients):
            return False
        _JUDGE_IDX += 1
        print(
            f"[metrics] OpenAI key {from_idx + 1} hit quota; "
            f"switched to key {_JUDGE_IDX + 1}/{len(clients)}",
            flush=True,
        )
        return True


def call_text_judge(system: str, user: str) -> tuple[Optional[int], str]:
    last = ""
    for attempt in range(4):
        client, idx = _judge_client()
        try:
            resp = client.chat.completions.create(
                model=TEXT_JUDGE_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            ans = str(parsed.get("answer", "NO")).strip().upper()
            reason = str(parsed.get("reason", ""))
            return (1 if ans.startswith("Y") else 0), reason
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if _is_quota_error(exc) and _advance_key(idx):
                continue
            time.sleep(2 * (attempt + 1))
    return None, f"judge_error:{last}"


def _join(texts: list[str]) -> str:
    return " ".join(t for t in texts if t).strip()


def score_one(
    info_path: Path,
    *,
    do_subsequent: bool,
    overwrite: bool,
) -> dict | None:
    info = json.loads(info_path.read_text())
    action = info.get("action") or info_path.parts[-5] if "inference_results" in str(info_path) else info.get("action")
    # DR: .../inference_results/<action>/<l3>/<model>/<sample>/inference_info.json
    # SIM: .../results/<action>/<model>/<sample>/inference_info.json
    parts = info_path.parts
    if "inference_results" in parts:
        i = parts.index("inference_results")
        action = parts[i + 1]
        model = parts[i + 3]
    else:
        i = parts.index("results")
        action = info.get("action") or parts[i + 1]
        model = parts[i + 2]
    sample_id = info.get("sample_id") or info_path.parent.name
    l3_raw = info.get("l3_condition") or ""
    cond = COND_CANON.get(l3_raw, l3_raw)
    if action == "turn_taking" and cond not in TT_WINDOW and "inference_results" in parts:
        folder_l3 = parts[parts.index("inference_results") + 2]
        cond = COND_CANON.get(folder_l3, folder_l3)
    inst = info.get("instance") or {}
    inf = info.get("inference") or {}
    if inf.get("errors"):
        # The provider never delivered a usable turn (403, 1011, keepalive
        # timeout, ...), so there is no behaviour to score. Matching
        # reports/_cue_landing.py, drop the sample instead of letting a
        # transport failure masquerade as a model that stayed silent. Any
        # entry an earlier pass wrote for it is stale: the report writers
        # read the cache directory, not our return value.
        stale = cache_path(action, model, sample_id)
        stale.unlink(missing_ok=True)
        return None
    meta = dict(inst)
    # data metadata may have extra fields
    if action in DR_ACTIONS:
        data_dir = DATA / action / l3_raw / sample_id
    else:
        data_dir = DATA / action / (l3_raw or cond) / sample_id
        if not data_dir.exists():
            # interrupt / recover folders match l3
            hits = list((DATA / action).glob(f"*/{sample_id}/metadata.json"))
            if hits:
                data_dir = hits[0].parent
    if (data_dir / "metadata.json").exists():
        meta = {**json.loads((data_dir / "metadata.json").read_text()), **inst}
    ts = {**(meta.get("timestamps") or {}), **(inst.get("timestamps") or {})}
    for k, v in ts.items():
        meta.setdefault(k, v)

    cached = cache_path(action, model, sample_id)
    src_path = str(info_path.resolve())
    src_ts = inf.get("timestamp")
    prev = json.loads(cached.read_text()) if cached.exists() else None
    if prev is not None and not overwrite:
        same_src = (
            prev.get("source_path") == src_path
            and prev.get("inference_timestamp") == src_ts
        )
        if action == "interrupt_user":
            schema_ok = prev.get("interrupt_window") == interrupt.WINDOW_ID
        elif action == "respond_and_recover":
            schema_ok = prev.get("recover_window") == recover.WINDOW_ID
        else:
            schema_ok = True
        if same_src and schema_ok and prev.get("immediate") is not None and (
            not do_subsequent or prev.get("subsequent_done")
        ):
            return prev

    segs = [tuple(s) for s in inf.get("model_speech_segments") or []]
    segs = drop_moshi(segs, model)
    segs = to_instance_clock(segs, model, action, inf)
    texts = segment_texts(info_path.parent, info, segs)
    sc = scenario_of(meta)
    src = "Synthesized" if str(meta.get("dataset") or "").lower() == "synthesized" else "Real"
    forced = bool(inst.get("forced") or (inf.get("trigger") or {}).get("forced"))
    spoke = bool(segs)
    rec: dict = {
        "sample_id": sample_id,
        "action": action,
        "condition": cond,
        "l3_raw": l3_raw,
        "model": model,
        "scenario": sc,
        "source": src,
        "forced": forced,
        "spoke": spoke,
        "cue_class": (
            classify_cue({"instance": inst, "inference": inf})
            if action in CUE_ACTIONS
            else None
        ),
        "immediate": None,
        "immediate_tag": None,
        "subsequent": None,
        "subsequent_na": None,
        "subsequent_reason": None,
        "subsequent_done": False,
        "function_match": None,
        "function_label": None,
        "source_path": src_path,
        "inference_timestamp": src_ts,
    }

    # ── Immediate ────────────────────────────────────────────────────────
    try:
        if action == "wait":
            if cond == "no_pause" or l3_raw == "floor_holding":
                lo, hi = float(meta.get("clip_start") or 0.0), float(meta["clip_end"])
            else:
                lo, hi = float(meta["pause_start"]), float(meta["pause_end"])
            rec["immediate"], rec["immediate_tag"] = wait.score_immediate_silence(
                segs, lo, hi, texts
            )
        elif action == "ignore":
            iv = ignore.resolve_ignore_interval(
                bg_speech_start_s=meta.get("bg_speech_start_s"),
                bg_speech_end_s=meta.get("bg_speech_end_s"),
                side_speech_start_s=meta.get("side_speech_start_s"),
                side_speech_end_s=meta.get("side_speech_end_s"),
            )
            if iv is None:
                rec["immediate_tag"] = "missing_distractor"
            else:
                rec["immediate"], rec["immediate_tag"] = ignore.score_immediate_ignore(
                    segs, iv[0], iv[1], texts
                )
        elif action == "turn_taking":
            t_end = float(meta.get("t1_end") if meta.get("t1_end") is not None else meta["t_user_end"])
            lo, hi = TT_WINDOW[cond]
            rec["immediate"], rec["immediate_tag"], rec["delta_s"] = take_turn.score_immediate_takeover(
                segs, t_end, lo, hi, texts
            )
        elif action == "backchannel":
            t0 = float(meta.get("clip_start") or 0.0)
            t1 = float(meta.get("clip_end") if meta.get("clip_end") is not None else meta["t_user_end"])
            t_ref = float(meta.get("bc_start") or t0)
            rec["immediate"], rec["immediate_tag"], rec["delta_s"] = backchannel.score_immediate_backchannel(
                segs, t0, t1, t_ref, texts
            )
        elif action == "stop":
            t_cue = inst.get("t_user_int_start")
            if t_cue is None:
                cue = [s for s in (inf.get("simulator_speech_segments") or []) if s.get("source") == "cue"]
                t_cue = cue[0]["start"] if cue else None
            t_rec = inf.get("output_duration_s")
            if t_cue is None:
                rec["immediate_tag"] = "missing_cue"
            else:
                rec["immediate"], rec["immediate_tag"], rec["yield_latency"] = stop.score_stop_rate(
                    segs, float(t_cue), t_rec
                )
                rec["t_cue_start"] = float(t_cue)
                cue = [s for s in (inf.get("simulator_speech_segments") or []) if s.get("source") == "cue"]
                t_end = float(cue[0]["end"]) if cue else None
                rec["t_cue_end"] = t_end
                if rec["immediate"] == 1 and t_end is not None:
                    rec["subsequent"], rec["subsequent_na"] = stop.score_yielding_maintained(
                        segs, float(t_cue), t_end, texts
                    )
                    rec["subsequent_done"] = True
                else:
                    rec["subsequent"] = None
                    rec["subsequent_na"] = rec["immediate_tag"] or "no_first_stop"
                    rec["subsequent_done"] = True
        elif action == "hold_floor":
            t_cue = meta.get("t_cue_start") or inst.get("t_user_int_start")
            t_end = meta.get("t_cue_end")
            if t_cue is None or t_end is None:
                cue = [s for s in (inf.get("simulator_speech_segments") or []) if s.get("source") == "cue"]
                if cue:
                    t_cue = cue[0]["start"]
                    t_end = cue[0]["end"]
            if t_cue is None or t_end is None:
                rec["immediate_tag"] = "missing_cue"
            else:
                rec["immediate"], rec["immediate_tag"], rec["max_gap"] = (
                    hold_floor.score_immediate_hold_floor(segs, float(t_cue), float(t_end))
                )
                rec["t_cue_start"] = float(t_cue)
                rec["t_cue_end"] = float(t_end)
        elif action == "respond_and_recover":
            t_end = meta.get("t_interruption_end")
            if t_end is None:
                cue = [s for s in (inf.get("simulator_speech_segments") or []) if s.get("source") == "cue"]
                t_end = cue[0]["end"] if cue else None
            if t_end is None:
                rec["immediate_tag"] = "missing_interruption_end"
            else:
                rec["immediate"], rec["immediate_tag"], rec["delta_s"] = recover.score_immediate_recover(
                    segs, float(t_end), texts
                )
                rec["t_interruption_end"] = float(t_end)
                rec["recover_window"] = recover.WINDOW_ID
        elif action == "interrupt_user":
            t0 = float(meta.get("t_user_start") or 0.0)
            t_user_end = interrupt.resolve_user_end(inst, inf)
            t_target = meta.get("t_target")
            rec["interrupt_window"] = interrupt.WINDOW_ID
            if t_user_end is None:
                rec["immediate_tag"] = "missing_user_end"
            else:
                rec["immediate"], rec["immediate_tag"], rec["slack"] = (
                    interrupt.score_immediate_interrupt(
                        segs,
                        t0,
                        float(t_user_end),
                        None if t_target is None else float(t_target),
                        texts,
                    )
                )
                rec["t_user_end"] = float(t_user_end)
                if t_target is not None:
                    rec["t_target"] = float(t_target)
        else:
            rec["immediate_tag"] = f"unknown_action:{action}"
    except Exception as exc:
        rec["immediate_tag"] = f"score_error:{type(exc).__name__}:{exc}"

    # ── Subsequent (text) ────────────────────────────────────────────────
    if do_subsequent and not rec.get("subsequent_done"):
        reused = False
        if (
            action == "interrupt_user"
            and rec.get("immediate") == 1
            and prev is not None
            and prev.get("immediate") == 1
            and prev.get("subsequent_done")
        ):
            rec["subsequent"] = prev.get("subsequent")
            rec["subsequent_na"] = prev.get("subsequent_na")
            rec["subsequent_reason"] = prev.get("subsequent_reason")
            rec["subsequent_done"] = True
            reused = True
        if not reused:
            try:
                rec.update(_score_subsequent(action, cond, meta, inst, inf, segs, texts, rec))
            except Exception as exc:
                rec["subsequent_na"] = f"sub_error:{type(exc).__name__}:{exc}"
                rec["subsequent_done"] = True

    # Backchannel function match is not scored here. It is stage 2 of
    # bc_two_stage_judge.py, which needs output.wav and only runs on the
    # candidates stage 1 confirmed; reports/_bc_verdicts.py joins both
    # verdicts onto this record when a report is written.

    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def _score_subsequent(action, cond, meta, inst, inf, segs, texts, rec) -> dict:
    out = {"subsequent_done": True, "subsequent": None, "subsequent_na": None, "subsequent_reason": None}
    model_all = _join([(t or "") for t in texts])

    if rec.get("immediate") != 1:
        out["subsequent_na"] = rec.get("immediate_tag") or "immediate_fail"
        return out

    if action == "wait":
        if cond == "no_pause" or meta.get("task_type") == "floor_holding":
            user = meta.get("full_transcript") or ""
            cid = "no_pause"
        elif "intra" in cond:
            user = meta.get("post_pause_text") or ""
            cid = "intra_sentential_pause"
        else:
            user = meta.get("post_pause_text") or ""
            cid = "inter_sentential_pause"
        if not model_all:
            out["subsequent"] = 0
            out["subsequent_reason"] = "no later response"
            return out
        prompt = wait.build_judge_prompt(
            condition_id=cid,
            user_during_or_after_wait=user,
            model_response=model_all,
        )
        out["subsequent"], out["subsequent_reason"] = call_text_judge(
            wait.JUDGE_SYSTEM_PROMPT, prompt
        )
        return out

    if action == "turn_taking":
        onsets = take_turn.takeover_onsets(segs, texts)
        t_end = float(meta.get("t1_end") if meta.get("t1_end") is not None else 0.0)
        in_obs = [on for on in onsets if (take_turn.OBSERVATION_LO_S <= on - t_end <= take_turn.OBSERVATION_HI_S)]
        if not in_obs:
            out["subsequent"] = 0
            out["subsequent_reason"] = "no_takeover_in_observation"
            return out
        user = meta.get("t1_text") or meta.get("full_transcript") or ""
        prompt = take_turn.build_judge_prompt(
            condition_id=cond, user_speech=user, model_response=model_all
        )
        out["subsequent"], out["subsequent_reason"] = call_text_judge(
            take_turn.JUDGE_SYSTEM_PROMPT, prompt
        )
        return out

    if action == "ignore":
        iv = ignore.resolve_ignore_interval(
            bg_speech_start_s=meta.get("bg_speech_start_s"),
            bg_speech_end_s=meta.get("bg_speech_end_s"),
            side_speech_start_s=meta.get("side_speech_start_s"),
            side_speech_end_s=meta.get("side_speech_end_s"),
        )
        if iv is None or not ignore.score_subsequent_applicable(segs, iv[1]):
            out["subsequent"] = 0
            out["subsequent_reason"] = "no later response"
            return out
        after = ignore.speech_after_other_end(segs, iv[1], texts)
        prompt = ignore.build_judge_prompt(
            addressed_user_speech=meta.get("speaker_a_text") or "",
            ignored_speech=meta.get("other_text") or "",
            model_response=_join(after),
        )
        out["subsequent"], out["subsequent_reason"] = call_text_judge(
            ignore.JUDGE_SYSTEM_PROMPT, prompt
        )
        return out

    if action == "backchannel":
        event = backchannel.first_user_speech_backchannel(
            segs,
            float(meta.get("clip_start") or 0.0),
            float(meta.get("clip_end") or 0.0),
            texts,
        )
        if event is None:
            out["subsequent"] = 0
            out["subsequent_reason"] = "no_backchannel"
            return out
        _on, t_bc_end, bc_text = event
        later = backchannel.speech_after_backchannel(segs, t_bc_end, texts)
        if not any(later):
            out["subsequent"] = 0
            out["subsequent_reason"] = "no later response"
            return out
        user_before = (meta.get("before_text") or "") + " " + (meta.get("bc_text") or "")
        user_between = meta.get("after_text") or ""
        prompt = backchannel.build_judge_prompt(
            user_before_and_overlap=user_before,
            user_between=user_between,
            model_backchannel=bc_text,
            model_later=_join(later),
        )
        out["subsequent"], out["subsequent_reason"] = call_text_judge(
            backchannel.SUBSEQUENT_JUDGE_SYSTEM_PROMPT, prompt
        )
        return out

    if action == "hold_floor":
        if rec.get("immediate") != 1:
            out["subsequent_na"] = "immediate_fail"
            return out
        t_end = rec.get("t_cue_end")
        if t_end is None or not hold_floor.score_subsequent_applicable(segs, t_end):
            out["subsequent_na"] = "no_speech_after_cue"
            return out
        req = hold_floor.ACTION.conditions.get(cond)
        req_txt = req.required_adjustment if req else ""
        cue_txt = meta.get("cue_text") or inst.get("reference_interruption_text") or ""
        prompt = hold_floor.build_judge_prompt(
            condition_id=cond,
            required_adjustment=req_txt,
            cue=cue_txt,
            model_through_cue=_join(hold_floor.speech_through_cue_end(segs, t_end, texts)),
            model_after_cue=_join(hold_floor.speech_after_cue_end(segs, t_end, texts)),
        )
        out["subsequent"], out["subsequent_reason"] = call_text_judge(
            hold_floor.JUDGE_SYSTEM_PROMPT, prompt
        )
        return out

    if action == "respond_and_recover":
        if rec.get("immediate") != 1:
            out["subsequent_na"] = "immediate_fail"
            return out
        t_end = rec.get("t_interruption_end")
        if t_end is None:
            out["subsequent_na"] = "missing_t_end"
            return out
        in_win = []
        for (on, off), t in zip(segs, texts):
            if recover.RECOVERY_LO_S <= (on - t_end) <= recover.RECOVERY_HI_S:
                in_win.append(t or "")
        prior = []
        for (on, off), t in zip(segs, texts):
            if off <= t_end + 1e-6 or on < t_end:
                if on < t_end:
                    prior.append(t or "")
        cue = [s for s in (inf.get("simulator_speech_segments") or []) if s.get("source") == "cue"]
        inter = (cue[0].get("text") if cue else None) or inst.get("reference_interruption_text") or ""
        prompt = recover.build_judge_prompt(
            interruption=inter,
            model_recovery=_join(in_win),
            interrupted_contribution=_join(prior),
        )
        out["subsequent"], out["subsequent_reason"] = call_text_judge(
            recover.JUDGE_SYSTEM_PROMPT, prompt
        )
        return out

    if action == "interrupt_user":
        if rec.get("immediate") != 1:
            out["subsequent_na"] = "immediate_fail"
            return out
        user = meta.get("transcript_user") or inst.get("transcript_user") or ""
        # first in-window takeover text
        t0 = float(meta.get("t_user_start") or 0.0)
        t_end = rec.get("t_user_end")
        if t_end is None:
            t_end = interrupt.resolve_user_end(inst, inf)
        hi = float(t_end or rec.get("t_target") or 0.0) - interrupt.OVERLAP_LOOKBACK_S
        chosen = []
        for (on, off), t in zip(segs, texts):
            if t0 <= on <= hi:
                chosen.append(t or "")
                break
        prompt = interrupt.build_judge_prompt(
            condition_id=cond,
            user_speech=user,
            model_interruption=_join(chosen) or model_all,
        )
        out["subsequent"], out["subsequent_reason"] = call_text_judge(
            interrupt.JUDGE_SYSTEM_PROMPT, prompt
        )
        return out

    out["subsequent_na"] = "no_subsequent"
    return out


def _rate(vals: list) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def aggregate(records: list[dict]) -> dict:
    def pack(rows):
        imm = [r["immediate"] for r in rows if r.get("immediate") is not None]
        # Headline: cue on speech (forced allowed). Unforced kept as a diagnostic.
        imm_h = [
            r["immediate"] for r in rows
            if r.get("immediate") is not None and on_speech(r)
        ]
        imm_u = [
            r["immediate"] for r in rows
            if r.get("immediate") is not None and not r.get("forced")
        ]
        sub = []
        for r in rows:
            if r.get("immediate") != 1:
                continue
            if r.get("action") in CUE_ACTIONS and not on_speech(r):
                continue
            if r.get("action") in DR_ACTIONS:
                sub.append(1 if r.get("subsequent") == 1 else 0)
            elif r.get("subsequent") is not None:
                sub.append(r["subsequent"])
        fn = [r["function_match"] for r in rows if r.get("function_match") is not None]
        return {
            "n": len(rows),
            "n_forced": sum(1 for r in rows if r.get("forced")),
            "immediate": _rate(imm),
            "immediate_n": len(imm),
            "immediate_speaking": _rate(imm_h),
            "immediate_speaking_n": len(imm_h),
            "immediate_unforced": _rate(imm_u),
            "immediate_unforced_n": len(imm_u),
            "subsequent": _rate(sub),
            "subsequent_n": len(sub),
            "function_match": _rate(fn),
            "function_n": len(fn),
        }

    by_action = defaultdict(list)
    by_cond = defaultdict(list)
    by_scen = defaultdict(list)
    by_model_action = defaultdict(list)
    by_scen_only = defaultdict(list)
    for r in records:
        by_action[r["action"]].append(r)
        by_cond[(r["action"], r["condition"])].append(r)
        by_scen[(r["action"], r.get("scenario") or "unknown")].append(r)
        by_scen_only[r.get("scenario") or "unknown"].append(r)
        by_model_action[(r["model"], r["action"])].append(r)

    return {
        "n": len(records),
        "action": {k: pack(v) for k, v in sorted(by_action.items())},
        "condition": {f"{a}/{c}": pack(v) for (a, c), v in sorted(by_cond.items())},
        "scenario": {f"{a}/{s}": pack(v) for (a, s), v in sorted(by_scen.items())},
        "scenario_pooled": {k: pack(v) for k, v in sorted(by_scen_only.items())},
        "model_action": {f"{m}/{a}": pack(v) for (m, a), v in sorted(by_model_action.items())},
    }


def load_all_cache(action=None, model=None) -> list[dict]:
    rows = []
    root = CACHE
    if not root.exists():
        return rows
    for p in root.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if action and rec.get("action") != action:
            continue
        if model and rec.get("model") != model:
            continue
        rows.append(rec)
    return rows


def fmt_pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def print_report(agg: dict) -> str:
    lines = [f"# Metrics report  N={agg['n']}", "", "## Action", ""]
    lines.append("| Action | N | Imm | Imm (on speech) | Sub | Fun |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for a, s in agg["action"].items():
        lines.append(
            f"| {ACTION_LABEL.get(a, a)} | {s['n']} | {fmt_pct(s['immediate'])} "
            f"({s['immediate_n']}) | {fmt_pct(s['immediate_speaking'])} ({s['immediate_speaking_n']}) "
            f"| {fmt_pct(s['subsequent'])} ({s['subsequent_n']}) | {fmt_pct(s['function_match'])} |"
        )
    lines += ["", "## Condition", "", "| Action / condition | N | Imm | Imm on speech | Sub |",
              "|---|---:|---:|---:|---:|"]
    for k, s in agg["condition"].items():
        lines.append(
            f"| {k} | {s['n']} | {fmt_pct(s['immediate'])} | {fmt_pct(s['immediate_speaking'])} "
            f"| {fmt_pct(s['subsequent'])} ({s['subsequent_n']}) |"
        )
    lines += ["", "## Scenario (pooled)", "", "| Scenario | N | Imm | Imm on speech | Sub |",
              "|---|---:|---:|---:|---:|"]
    for k, s in (agg.get("scenario_pooled") or {}).items():
        lines.append(
            f"| {SC_LABEL.get(k, k)} | {s['n']} | {fmt_pct(s['immediate'])} | {fmt_pct(s['immediate_speaking'])} "
            f"| {fmt_pct(s['subsequent'])} ({s['subsequent_n']}) |"
        )
    lines += ["", "## Scenario × action", "", "| Action / scenario | N | Imm | Imm on speech | Sub |",
              "|---|---:|---:|---:|---:|"]
    for k, s in agg["scenario"].items():
        lines.append(
            f"| {k} | {s['n']} | {fmt_pct(s['immediate'])} | {fmt_pct(s['immediate_speaking'])} "
            f"| {fmt_pct(s['subsequent'])} ({s['subsequent_n']}) |"
        )
    lines += ["", "## Model × action", "", "| Model / action | N | Imm | Imm on speech | Sub |",
              "|---|---:|---:|---:|---:|"]
    for k, s in agg["model_action"].items():
        lines.append(
            f"| {k} | {s['n']} | {fmt_pct(s['immediate'])} | {fmt_pct(s['immediate_speaking'])} "
            f"| {fmt_pct(s['subsequent'])} ({s['subsequent_n']}) |"
        )
    return "\n".join(lines) + "\n"


def main():
    load_dotenv(REPO / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=ALL_ACTIONS)
    ap.add_argument("--model")
    ap.add_argument("--immediate_only", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.aggregate:
        rows = load_all_cache(args.action, args.model)
        agg = aggregate(rows)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "summary.json").write_text(json.dumps(agg, indent=2))
        md = print_report(agg)
        (REPORTS / "summary.md").write_text(md)
        print(md)
        return

    paths = discover(args.action, args.model)
    if args.limit:
        paths = paths[: args.limit]
    print(f"scoring {len(paths)} samples  subsequent={not args.immediate_only}", flush=True)

    rows = []
    n_err = 0
    n_skipped = 0

    def _job(p):
        return score_one(
            p,
            do_subsequent=not args.immediate_only,
            overwrite=args.overwrite,
        )

    if args.workers <= 1:
        for i, p in enumerate(paths, 1):
            try:
                row = _job(p)
                if row is None:
                    n_skipped += 1
                else:
                    rows.append(row)
            except Exception as exc:
                n_err += 1
                print(f"FAIL {p}: {exc}", file=sys.stderr)
            if i % 500 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, p): p for p in paths}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    row = fut.result()
                    if row is None:
                        n_skipped += 1
                    else:
                        rows.append(row)
                except Exception as exc:
                    n_err += 1
                    print(f"FAIL {futs[fut]}: {exc}", file=sys.stderr)
                if done % 500 == 0:
                    print(f"  {done}/{len(paths)}", flush=True)

    agg = aggregate(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tag = args.action or "all"
    if args.model:
        tag += f"_{args.model}"
    (REPORTS / f"summary_{tag}.json").write_text(json.dumps(agg, indent=2))
    md = print_report(agg)
    (REPORTS / f"summary_{tag}.md").write_text(md)
    print(md)
    print(f"errors={n_err}  cache={CACHE}", flush=True)
    if n_skipped:
        # Loud on purpose: a skipped sample shrinks the cache below dataset N,
        # which makes reports/ withhold the whole cell. Re-run the inference
        # for these rather than leaving them out.
        print(f"SKIPPED {n_skipped} sample(s) carrying inference errors -- "
              f"cell will render as '--' until they are re-run", flush=True)


if __name__ == "__main__":
    main()
