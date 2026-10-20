#!/usr/bin/env python3
"""Backchannel two-stage scoring: LLM screen, then LALM function label.

Stage 1 (text, gpt-5.4, json_object). Every segment the automatic rule flagged as
a backchannel is shown in context -- user transcript, the full model segment map,
and the candidate marked. The judge rules on whether it is a genuine backchannel
(brief listener feedback inside the user's turn) or something the ≤3-word rule
mis-flagged: a truncated start of a full turn, a filler with no feedback
function, or a segment with no speech at all.

Stage 2 (audio, gpt-audio). Only the segments stage 1 confirmed. The model output
track, cropped around the backchannel, decides affirmative vs uncertain.

Neither stage sees the L3 condition, the gold human `bc_text`, or
`bc_type_original`.

Three rates come out, all over the same dataset N so they stack:
  automatic  -- the ≤3-word rule alone, i.e. the current published number
  screened   -- automatic AND stage 1 confirmed it is a backchannel
  function   -- screened AND stage 2 gave the condition's function

    python evaluation/scoring/bc_two_stage_judge.py
"""
from __future__ import annotations

import argparse
import base64
import functools
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

from evaluation.direct_replay.metrics import backchannel  # noqa: E402
from evaluation.scoring.run_metrics import (  # noqa: E402
    drop_moshi,
    segment_texts,
    to_instance_clock,
)

CACHE = REPO / "evaluation" / "scoring" / "cache" / "backchannel"
OUT_JSON = REPO / "evaluation" / "scoring" / "reports" / "BC_TWO_STAGE.json"
OUT_MD = REPO / "evaluation" / "scoring" / "reports" / "BC_TWO_STAGE.md"

TEXT_MODEL = "gpt-5.4-2026-03-05"
AUDIO_MODEL = "gpt-audio"
AUDIO_PAD_S = 2.0
DATASET_N = 400

MODELS = [
    "bayling-duplex", "flm-audio", "freeze-omni", "gemini-3.1-flash-live",
    "gpt-realtime-2.1", "grok-voice", "lychee-fd", "mini-omni2",
    "minicpm-o-4.5", "moshi", "qwen3.5-omni-flash-realtime",
    "raon-speechchat", "vita-1.5", "voila-autonomous",
]

SCREEN_SYSTEM = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. The user holds the floor for the whole interval. A detector has "
    "flagged one short model segment as a possible backchannel, and you must "
    "rule on whether it really is one. A backchannel is brief listener feedback "
    "produced while the other speaker keeps the floor -- an acknowledgement, a "
    "continuer, or a signal of trouble.\n"
    "The model is expected to backchannel during the user's turn AND to answer "
    "properly later, so the mere existence of later model speech does NOT make "
    "the marked segment a turn attempt. What matters is whether the marked "
    "segment is self-contained. Treat it as the truncated opening of a full turn "
    "only when the following segment resumes the same utterance with no real "
    "pause, so that reading them together gives one continuous sentence.\n"
    "Answer NO when the segment runs on into the next one as a single sentence, "
    "when it carries no listener-feedback function, or when it has no "
    "intelligible speech. Use ONLY the evidence given. "
    'Return strict JSON only: {"is_backchannel": "YES"|"NO", "reason": "<one short sentence>"}.'
)

SCREEN_QUESTION = (
    "Is the marked segment a genuine backchannel? "
    "Answer YES if it is self-contained brief listener feedback that leaves the "
    "floor with the user, even when the model speaks again later. "
    "Answer NO if it runs straight on into the next segment as one continuous "
    "sentence, carries no feedback function, or contains no intelligible speech."
)

FUNCTION_SYSTEM = (
    "You are a strict, consistent evaluator for a full-duplex spoken dialogue "
    "benchmark. The user holds the floor and the model produces one short "
    "backchannel inside the user's turn. Decide whether that backchannel signals "
    "an affirmative function (acknowledgement, understanding, or agreement) or an "
    "uncertain function (hesitation, confusion, or a need for clarification). "
    "Judge only the marked backchannel, not the model's later speech. Base the "
    "decision on how it sounds -- intonation, pitch contour, hesitation. "
    "This is a forced binary choice: you must answer affirmative or uncertain. "
    "There is no unknown option, so if the audio is ambiguous commit to whichever "
    "of the two the sound leans towards. "
    'Return strict JSON only: {"function": "affirmative"|"uncertain", '
    '"reason": "<one short sentence>"}.'
)

FUNCTION_QUESTION = (
    "Does the marked backchannel signal affirmative or uncertain? "
    "Answer affirmative if it acknowledges, shows understanding, or agrees. "
    "Answer uncertain if it shows hesitation, confusion, or a need for "
    "clarification. You must pick one of the two."
)

FUNCTION_LABELS = ("affirmative", "uncertain")

CONTROL_TOKEN = re.compile(r"<\|[^|]*\|>")


def clean_text(t: Optional[str]) -> str:
    """Model control tokens are apparatus, not speech (raon emits them inline)."""
    return CONTROL_TOKEN.sub("", t or "").strip()


def build_context(
    *,
    user_text: str,
    clip_start: float,
    clip_end: float,
    segments: list[tuple[float, float]],
    texts: list[Optional[str]],
    bc_index: int,
    hide_bc_text: bool,
    question: str,
    reply_shape: str,
) -> str:
    lines = [
        "Action: backchannel",
        "",
        f"The user speaks continuously from {clip_start:.2f}s to {clip_end:.2f}s.",
        "",
        "User transcript:",
        user_text.strip() or "(none)",
        "",
        "Model output segments, in order, with the silence before each one:",
    ]
    prev_off: Optional[float] = None
    for i, (on, off) in enumerate(segments):
        raw = clean_text(texts[i] if i < len(texts) else None)
        if i == bc_index and hide_bc_text:
            shown = "<<< THE MARKED SEGMENT - listen to the attached audio for it"
        elif i == bc_index:
            shown = f"{raw or '(no transcript)'}   <<< THE MARKED SEGMENT"
        else:
            shown = raw or "(no transcript)"
        gap = "start" if prev_off is None else f"{on - prev_off:.2f}s silence before"
        lines.append(f"  [{on:.2f}s - {off:.2f}s] ({gap}) {shown}")
        prev_off = off

    on, off = segments[bc_index]
    after = [(o, f) for o, f in segments if o > off]
    if after:
        gap_after = after[0][0] - off
        tail = (
            f"The next model segment begins {gap_after:.2f}s later."
            if gap_after >= 0.5
            else f"The next model segment begins only {gap_after:.2f}s later, "
            "so check whether the two are one continuous utterance."
        )
    else:
        tail = "No further model speech follows."
    lines += [
        "",
        f"The marked segment runs {on:.2f}s to {off:.2f}s, i.e. {on - clip_start:.2f}s "
        f"into the user's turn. {tail}",
        "",
        f"Question: {question}",
        "",
        f"Reply with JSON: {reply_shape}",
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
    return {}


def prepare(rec: dict) -> Optional[dict]:
    info_path = Path(rec["source_path"])
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    inst = info.get("instance") or {}
    inf = info.get("inference") or {}
    model = rec["model"]

    # drop_moshi sorts and drops a segment, so only its output is index-aligned
    # with `segs`; indexing the unsorted raw list would crop the wrong audio.
    raw_segs = drop_moshi([tuple(s) for s in inf.get("model_speech_segments") or []], model)
    segs = to_instance_clock(raw_segs, model, "backchannel", inf)
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
    cond = rec.get("l3_raw") or rec.get("condition") or ""
    raw_on, raw_off = raw_segs[bc_index] if bc_index < len(raw_segs) else (bc_on, bc_off)

    common = functools.partial(
        build_context,
        user_text=user_text,
        clip_start=t0,
        clip_end=t1,
        segments=segs,
        texts=texts,
        bc_index=bc_index,
    )
    return {
        "sample_id": rec["sample_id"],
        "model": model,
        "condition": cond,
        "wanted": backchannel.FUNCTION_BY_CONDITION.get(cond),
        "model_bc_text": clean_text(bc_text),
        "bc_span": [round(bc_on, 2), round(bc_off, 2)],
        "wav": info_path.parent / "output.wav",
        "raw_span": [raw_on, raw_off],
        "prompt_screen": common(
            hide_bc_text=False,
            question=SCREEN_QUESTION,
            reply_shape='{"is_backchannel": "YES"|"NO", "reason": "<one short sentence>"}.',
        ),
        "prompt_function": common(
            hide_bc_text=True,
            question=FUNCTION_QUESTION,
            reply_shape='{"function": "affirmative"|"uncertain", "reason": "<one short sentence>"}.',
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit_per_model", type=int)
    ap.add_argument(
        "--models",
        help="Comma-separated slugs to judge. Default is every model. Stage 2 "
        "needs output.wav, so a model whose audio only exists on the pod has "
        "to be judged there and merged back with --merge.",
    )
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        help="Fold verdict shards into --out and rewrite the report, judging "
        "nothing. Rows for a model named in a shard replace that model's rows.",
    )
    ap.add_argument(
        "--reuse_screen",
        type=Path,
        help="Take stage-1 verdicts from a previous run and redo stage 2 only.",
    )
    args = ap.parse_args()

    scope = [m.strip() for m in args.models.split(",")] if args.models else list(MODELS)
    unknown = [m for m in scope if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}\nknown: {MODELS}")

    reuse: dict[tuple[str, str], dict] = {}
    if args.reuse_screen:
        reuse = {
            (r["model"], r["sample_id"]): r
            for r in json.loads(args.reuse_screen.read_text())
        }
        print(f"reusing {len(reuse)} stage-1 verdicts from {args.reuse_screen}", flush=True)

    auto_counts = rule_hits()

    if args.merge:
        rows = load_rows(args.out)
        for shard in args.merge:
            incoming = json.loads(shard.read_text())
            replaced = {r["model"] for r in incoming}
            rows = [r for r in rows if r["model"] not in replaced] + incoming
            print(f"merged {len(incoming)} rows for {sorted(replaced)} from {shard}", flush=True)
        write_out(rows, auto_counts, args.out)
        return

    load_dotenv(REPO / ".env")
    from openai import OpenAI

    client = OpenAI(api_key=(os.environ.get("OPENAI_API_KEY") or "").strip())

    cands: list[dict] = []
    for model in scope:
        d = CACHE / model
        if not d.is_dir():
            continue
        hits = []
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text())
            if rec.get("immediate") == 1:
                hits.append(rec)
        if args.limit_per_model:
            hits = hits[: args.limit_per_model]
        cands.extend(x for x in (prepare(r) for r in hits) if x)

    in_scope = sum(auto_counts.get(m, 0) for m in scope)
    print(f"scope {scope}: {in_scope} rule hits, prepared {len(cands)}", flush=True)

    def job(c: dict) -> dict:
        row = {k: v for k, v in c.items() if k not in ("wav", "prompt_screen", "prompt_function")}
        prior = reuse.get((c["model"], c["sample_id"]))
        if prior is not None:
            row["screen"] = prior["screen"]
            row["screen_reason"] = prior["screen_reason"]
        else:
            try:
                r = client.chat.completions.create(
                    model=TEXT_MODEL, temperature=0,
                    messages=[{"role": "system", "content": SCREEN_SYSTEM},
                              {"role": "user", "content": c["prompt_screen"]}],
                    response_format={"type": "json_object"},
                )
                p = parse_json(r.choices[0].message.content or "")
                row["screen"] = str(p.get("is_backchannel", "NO")).strip().upper().startswith("Y")
                row["screen_reason"] = p.get("reason")
            except Exception as exc:
                row["screen"] = None
                row["screen_reason"] = f"{type(exc).__name__}: {exc}"[:200]

        row["function_label"] = None
        row["function_reason"] = None
        row["function_match"] = None
        if row["screen"]:
            b64 = crop_wav_b64(c["wav"], c["raw_span"][0], c["raw_span"][1])
            if b64 is None:
                row["function_reason"] = "crop_failed"
            else:
                msgs = [{"role": "system", "content": FUNCTION_SYSTEM},
                        {"role": "user", "content": [
                            {"type": "text", "text": c["prompt_function"]},
                            {"type": "input_audio",
                             "input_audio": {"data": b64, "format": "wav"}}]}]
                # gpt-audio takes no response_format, so the binary is only as
                # firm as the prompt. It sometimes stalls ("let me listen...")
                # instead of answering, and at temperature 0 a blind retry just
                # reproduces that, so push the stall back as context and re-ask.
                for attempt in range(3):
                    try:
                        r = client.chat.completions.create(
                            model=AUDIO_MODEL, temperature=0, modalities=["text"], messages=msgs)
                        said = r.choices[0].message.content or ""
                        p = parse_json(said)
                        label = str(p.get("function", "")).strip().lower()
                        if label in FUNCTION_LABELS:
                            row["function_label"] = label
                            row["function_reason"] = p.get("reason")
                            row["function_match"] = int(label == c["wanted"])
                            break
                        row["function_reason"] = f"off_menu:{label or 'empty'}"
                        msgs = msgs + [
                            {"role": "assistant", "content": said},
                            {"role": "user", "content":
                                "You already have the audio. Give the verdict now, "
                                "affirmative or uncertain, as JSON only: "
                                '{"function": "affirmative"|"uncertain", "reason": "<one short sentence>"}.'},
                        ]
                    except Exception as exc:
                        row["function_reason"] = f"{type(exc).__name__}: {exc}"[:200]
        return row

    fresh: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, c) for c in cands]
        for n, fut in enumerate(as_completed(futs), 1):
            fresh.append(fut.result())
            if n % 50 == 0:
                print(f"  {n}/{len(cands)}", flush=True)

    # Judging one model must not discard the others already in --out.
    rows = [r for r in load_rows(args.out) if r["model"] not in scope] + fresh
    write_out(rows, auto_counts, args.out)


def rule_hits() -> dict[str, int]:
    """Rule-only backchannel hits per model, the denominator stage 1 shrinks."""
    out: dict[str, int] = {}
    for model in MODELS:
        d = CACHE / model
        if not d.is_dir():
            continue
        out[model] = sum(
            1 for p in d.glob("*.json") if json.loads(p.read_text()).get("immediate") == 1
        )
    return out


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def write_out(rows: list[dict], auto_counts: dict[str, int], out_json: Path) -> None:
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    md = [
        "# Backchannel: two-stage scoring",
        "",
        f"Stage 1 `{TEXT_MODEL}` rules on whether the segment the ≤3-word rule flagged is a",
        "genuine backchannel. Stage 2 `gpt-audio` labels the confirmed ones affirmative",
        "or uncertain from the cropped output audio. Neither stage sees the condition,",
        f"the gold human `bc_text`, or `bc_type_original`. Rates are over dataset *N* = {DATASET_N}.",
        "",
        "- **Automatic** -- the ≤3-word rule alone; the currently published Backchannel rate.",
        "- **Screened** -- automatic and stage 1 confirmed a backchannel.",
        "- **Function** -- screened and stage 2 returned the condition's function.",
        "",
        "| Model | Automatic | Screened | Function | survived stage 1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        sub = by_model.get(model, [])
        auto = auto_counts.get(model, 0)
        if not auto:
            md.append(f"| {model} | 0.0% (0) | — | — | — |")
            continue
        scr = sum(1 for r in sub if r["screen"])
        fn = sum(1 for r in sub if r["function_match"] == 1)
        md.append(
            f"| {model} | {100*auto/DATASET_N:.1f}% ({auto}) | {100*scr/DATASET_N:.1f}% ({scr}) "
            f"| {100*fn/DATASET_N:.1f}% ({fn}) | {100*scr/auto:.0f}% |"
        )

    md += ["", "## Stage 2 detail, among stage-1 confirmed", "",
           "| Model | n | says affirmative | says uncertain | match |",
           "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        sub = [r for r in by_model.get(model, []) if r["screen"] and r["function_label"]]
        if not sub:
            continue
        c = Counter(r["function_label"] for r in sub)
        fn = sum(1 for r in sub if r["function_match"] == 1)
        md.append(
            f"| {model} | {len(sub)} | {c.get('affirmative',0)} | {c.get('uncertain',0)} "
            f"| {100*fn/len(sub):.1f}% |"
        )

    md += ["", "## Stage 2 by condition, among stage-1 confirmed", "",
           "| Condition | n | affirmative | uncertain | match |",
           "|---|---:|---:|---:|---:|"]
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["screen"] and r["function_label"]:
            by_cond[r["condition"]].append(r)
    for cond, sub in sorted(by_cond.items()):
        c = Counter(r["function_label"] for r in sub)
        fn = sum(1 for r in sub if r["function_match"] == 1)
        md.append(
            f"| {cond} | {len(sub)} | {c.get('affirmative',0)} | {c.get('uncertain',0)} "
            f"| {100*fn/len(sub):.1f}% |"
        )

    rejected = [r for r in rows if r["screen"] is False]
    md += ["", f"## What stage 1 rejected ({len(rejected)} of {len(rows)})", "",
           "| model | sample | flagged text | reason |", "|---|---|---|---|"]
    for r in rejected[:40]:
        md.append(
            f"| {r['model']} | {r['sample_id']} | `{r['model_bc_text'] or '(empty)'}` "
            f"| {(r['screen_reason'] or '').replace('|','/')} |"
        )

    out_md = out_json.with_suffix(".md")
    out_md.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_md}\nwrote {out_json}")


if __name__ == "__main__":
    main()
