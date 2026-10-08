#!/usr/bin/env python3
"""Backchannel function match: transcript judge vs audio judge, side by side.

Both judges receive the SAME text context (user transcript, model segment map,
the backchannel segment marked with its time range). The audio judge also
receives the model output audio cropped around that backchannel. The only
difference between the two is whether the judge can hear it, so a disagreement
isolates what prosody contributes.

Never sent to either judge: the L3 condition, the gold human `bc_text`, and
`bc_type_original`. Those are recorded alongside the predictions for scoring.

    python evaluation/metrics/compare_bc_function_judges.py --per_cell 8
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import soundfile as sf
from dotenv import load_dotenv

from evaluation.paths import repo_root

REPO = repo_root()
sys.path.insert(0, str(REPO))

from evaluation.metrics import backchannel  # noqa: E402
from evaluation.metrics.run_metrics import (  # noqa: E402
    drop_moshi,
    segment_texts,
    to_instance_clock,
)

CACHE = REPO / "evaluation" / "metrics" / "cache" / "backchannel"
OUT = REPO / "evaluation" / "metrics" / "reports" / "BC_FUNCTION_JUDGE_COMPARE.json"

TEXT_MODEL = "gpt-4o"
AUDIO_MODEL = "gpt-audio"
AUDIO_PAD_S = 2.0

SYSTEM = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. The user holds the floor and the model produces one short "
    "backchannel inside the user's turn. Decide whether that backchannel "
    "signals an affirmative function (acknowledgement, understanding, or "
    "agreement) or an uncertain function (hesitation, confusion, or a need "
    "for clarification). Judge only the marked backchannel, not the model's "
    "later speech. If the evidence is ambiguous or insufficient, use unknown. "
    'Return strict JSON only: {"function": "affirmative"|"uncertain"|"unknown", '
    '"reason": "<one short sentence>"}.'
)

QUESTION = (
    "Does the marked backchannel signal affirmative or uncertain? "
    "Answer affirmative if it acknowledges, shows understanding, or agrees. "
    "Answer uncertain if it shows hesitation, confusion, or a need for "
    "clarification. Answer unknown if it cannot be typed."
)

AUDIO_NOTE = (
    "Attached audio is the model output track cropped around the marked "
    "backchannel. Base your decision on how the backchannel sounds "
    "(intonation, pitch contour, hesitation), using the transcript above "
    "only as context."
)


def build_context(
    *,
    user_text: str,
    clip_start: float,
    clip_end: float,
    segments: list[tuple[float, float]],
    texts: list[Optional[str]],
    bc_index: int,
    hide_bc_text: bool,
) -> str:
    lines = [
        "Action: backchannel",
        "",
        f"The user speaks continuously from {clip_start:.2f}s to {clip_end:.2f}s. "
        "The model should give brief feedback inside that interval without taking the turn.",
        "",
        "User transcript:",
        user_text.strip() or "(none)",
        "",
        "Model output segments, in order:",
    ]
    for i, (on, off) in enumerate(segments):
        raw = (texts[i] if i < len(texts) else None) or ""
        raw = raw.strip()
        if i == bc_index and hide_bc_text:
            shown = "<<< THE BACKCHANNEL BEING JUDGED — listen to the audio for this segment"
        elif i == bc_index:
            shown = f"{raw or '(no transcript)'}   <<< THE BACKCHANNEL BEING JUDGED"
        else:
            shown = raw or "(no transcript)"
        lines.append(f"  [{on:.2f}s - {off:.2f}s] {shown}")

    on, off = segments[bc_index]
    lines += [
        "",
        f"The backchannel under judgement starts at {on:.2f}s and ends at {off:.2f}s, "
        f"i.e. {on - clip_start:.2f}s into the user's turn.",
        "",
        f"Question: {QUESTION}",
        "",
        'Reply with JSON: {"function": "affirmative"|"uncertain"|"unknown", '
        '"reason": "<one short sentence>"}.',
    ]
    return "\n".join(lines)


def crop_wav_b64(wav_path: Path, start_s: float, end_s: float) -> Optional[str]:
    try:
        info = sf.info(str(wav_path))
        sr = info.samplerate
        lo = max(0, int((start_s - AUDIO_PAD_S) * sr))
        hi = min(info.frames, int((end_s + AUDIO_PAD_S) * sr))
        if hi <= lo:
            return None
        audio, _ = sf.read(str(wav_path), start=lo, stop=hi, dtype="int16", always_2d=False)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"function": "unknown", "reason": f"unparsed:{(raw or '')[:120]}"}


def call_text(client, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return parse_json(resp.choices[0].message.content or "")


def call_audio(client, prompt: str, audio_b64: str) -> dict:
    resp = client.chat.completions.create(
        model=AUDIO_MODEL,
        temperature=0,
        modalities=["text"],
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n\n{AUDIO_NOTE}"},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            },
        ],
    )
    return parse_json(resp.choices[0].message.content or "")


def load_candidates(per_cell: int, models: list[str]) -> list[dict]:
    picked: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for model in models:
        d = CACHE / model
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text())
            if rec.get("immediate") != 1:
                continue
            cond = rec.get("l3_raw") or rec.get("condition") or ""
            key = (model, cond)
            if len(picked[key]) >= per_cell:
                continue
            picked[key].append(rec)
    out: list[dict] = []
    for v in picked.values():
        out.extend(v)
    return out


def prepare(rec: dict) -> Optional[dict]:
    info_path = Path(rec["source_path"])
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    inst = info.get("instance") or {}
    inf = info.get("inference") or {}
    model = rec["model"]

    raw_segs = [tuple(s) for s in inf.get("model_speech_segments") or []]
    segs = to_instance_clock(drop_moshi(raw_segs, model), model, "backchannel", inf)
    texts = segment_texts(info_path.parent, info, segs)

    t0 = float(inst.get("clip_start") or 0.0)
    t1 = float(inst.get("clip_end") or 0.0)
    event = backchannel.first_user_speech_backchannel(segs, t0, t1, texts)
    if event is None:
        return None
    bc_on, bc_off, bc_text = event
    bc_index = next(
        (i for i, (on, off) in enumerate(segs) if on == bc_on and off == bc_off), None
    )
    if bc_index is None:
        return None

    user_text = " ".join(
        t for t in ((inst.get("before_text") or ""), (inst.get("after_text") or "")) if t
    ).strip()

    # output.wav lives on the output clock; crop with the pre-rescale times.
    raw_on, raw_off = raw_segs[bc_index] if bc_index < len(raw_segs) else (bc_on, bc_off)

    return {
        "sample_id": rec["sample_id"],
        "model": model,
        "condition": rec.get("l3_raw") or rec.get("condition"),
        "wanted": backchannel.FUNCTION_BY_CONDITION.get(
            rec.get("l3_raw") or rec.get("condition") or ""
        ),
        "bc_type_original": inst.get("bc_type_original"),
        "gold_human_bc_text": inst.get("bc_text"),
        "model_bc_text": bc_text,
        "bc_span": [bc_on, bc_off],
        "wav": info_path.parent / "output.wav",
        "raw_span": [raw_on, raw_off],
        "prompt_text": build_context(
            user_text=user_text,
            clip_start=t0,
            clip_end=t1,
            segments=segs,
            texts=texts,
            bc_index=bc_index,
            hide_bc_text=False,
        ),
        "prompt_audio": build_context(
            user_text=user_text,
            clip_start=t0,
            clip_end=t1,
            segments=segs,
            texts=texts,
            bc_index=bc_index,
            hide_bc_text=True,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cell", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--models",
        nargs="*",
        default=["minicpm-o-4.5", "moshi", "raon-speechchat", "gemini-3.1-flash-live"],
    )
    ap.add_argument("--show", type=int, default=3, help="print N full prompts")
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    from openai import OpenAI

    client = OpenAI(api_key=(os.environ.get("OPENAI_API_KEY") or "").strip())

    cands = [c for c in (prepare(r) for r in load_candidates(args.per_cell, args.models)) if c]
    print(f"prepared {len(cands)} samples", flush=True)

    for c in cands[: args.show]:
        print("=" * 78)
        print(f"{c['model']}  {c['sample_id']}  cond={c['condition']}  (gold hidden from judges)")
        print("-" * 78)
        print(c["prompt_text"])
        print("-" * 78)
        print("[audio judge sees the same, with the BC line replaced by a listen instruction]")
    print("=" * 78, flush=True)

    def job(c: dict) -> dict:
        row = {k: v for k, v in c.items() if k not in ("wav", "prompt_text", "prompt_audio")}
        row["bc_span"] = [round(x, 2) for x in c["bc_span"]]
        try:
            t = call_text(client, c["prompt_text"])
        except Exception as exc:
            t = {"function": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]}
        b64 = crop_wav_b64(c["wav"], c["raw_span"][0], c["raw_span"][1])
        if b64 is None:
            a = {"function": "error", "reason": "crop_failed"}
        else:
            try:
                a = call_audio(client, c["prompt_audio"], b64)
            except Exception as exc:
                a = {"function": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]}
        row["text_label"] = str(t.get("function", "unknown")).strip().lower()
        row["text_reason"] = t.get("reason")
        row["audio_label"] = str(a.get("function", "unknown")).strip().lower()
        row["audio_reason"] = a.get("reason")
        row["text_match"] = int(row["text_label"] == row["wanted"])
        row["audio_match"] = int(row["audio_label"] == row["wanted"])
        return row

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, c): c for c in cands}
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 10 == 0:
                print(f"  {n}/{len(cands)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    def rate(sub: list[dict], key: str) -> str:
        vals = [r[key] for r in sub]
        return f"{100 * sum(vals) / len(vals):.1f}% ({sum(vals)}/{len(vals)})" if vals else "n/a"

    print("\n## Function match: transcript (gpt-4o) vs audio (gpt-audio)\n")
    print(f"| {'Model / condition':<44} | n | transcript | audio | agree |")
    print("|---|---:|---:|---:|---:|")
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)
    for (m, c), sub in sorted(by.items()):
        agree = sum(1 for r in sub if r["text_label"] == r["audio_label"])
        print(
            f"| {m + ' / ' + str(c):<44} | {len(sub)} | {rate(sub, 'text_match')} "
            f"| {rate(sub, 'audio_match')} | {100 * agree / len(sub):.0f}% |"
        )
    agree_all = sum(1 for r in rows if r["text_label"] == r["audio_label"])
    print(
        f"| {'ALL':<44} | {len(rows)} | {rate(rows, 'text_match')} "
        f"| {rate(rows, 'audio_match')} | {100 * agree_all / len(rows):.0f}% |"
    )

    print("\nLabel distribution")
    print("  transcript:", dict(Counter(r["text_label"] for r in rows)))
    print("  audio     :", dict(Counter(r["audio_label"] for r in rows)))

    print("\nDisagreements (transcript != audio)")
    for r in rows:
        if r["text_label"] != r["audio_label"]:
            print(
                f"  {r['model']:<22} {r['sample_id']:<28} want={r['wanted']:<11} "
                f"text={r['text_label']:<11} audio={r['audio_label']:<11} "
                f"bc={r['model_bc_text']!r}"
            )
            print(f"      audio reason: {r['audio_reason']}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
