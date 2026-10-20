#!/usr/bin/env python3
"""
backfill_bayling_transcripts.py -- add `model_speech_transcripts` to BayLing's
existing direct-replay results, offline, with no GPU and no re-inference.

Why BayLing needs live transcript instead of ASR: 55% of the 20 ms frames
inside its speaking segments are literally all-zero samples (the vocoder runs
slower than realtime and the recorder writes that GPU stall as silence inside
an open turn), 96% of that zero time sitting in runs longer than 0.3 s. Whisper
returns 0.54 words/s on it against a 2.5-3.5 words/s conversational band, and
comes back completely empty on a quarter of samples. `run_metrics.
LIVE_TRANSCRIPT_MODELS` therefore prefers the model's own text for this slug --
but direct replay never recorded that field, so Subsequent was withheld.

Why no re-run is needed: `infer_bayling.py` already logs BayLing's native
`DuplexStreamEvent.text` as a timestamped `text_token` event, so every existing
sample's events.jsonl carries the text and its timing. 99.5% of 2000 samples
have it. This script buckets those tokens into the sample's own
`model_speech_segments` and writes the result back into inference_info.json.

Bucketing is imported from the simulator's `default_model_speech_transcripts()`
rather than reimplemented, so direct replay and simulator produce the same text
for the same audio -- in particular the ""-join, which is deliberate: BayLing's
tokens already carry their own leading space at word boundaries.

Writes are additive (only the one key), atomic (temp file + rename), and
idempotent: re-running overwrites the same key with the same value. Samples
with no text_token get a list of "" of the right length -- NOT a short list or
None, because `run_metrics.segment_texts()` decides whether native text is
usable by checking `len(native) == len(segs)`.

Usage:
  python3 backfill_bayling_transcripts.py --dry_run     # report, write nothing
  python3 backfill_bayling_transcripts.py               # write
  python3 backfill_bayling_transcripts.py --action wait
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(os.environ.get("DUPLEXACT_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(REPO / "evaluation" / "common"))
sys.path.insert(0, str(REPO / "evaluation" / "simulator_replay" / "inference"))

from stop_common import default_model_speech_transcripts  # noqa: E402

DR_RESULTS = REPO / "evaluation" / "direct_replay" / "results" / "inference_results"
ACTIONS = ["wait", "backchannel", "ignore", "turn_taking"]
MODEL = "bayling-duplex"


def read_text_tokens(events_path: Path) -> list[tuple[float, str]]:
    """(t, text) for every text_token event, in file order."""
    out: list[tuple[float, str]] = []
    with events_path.open() as fh:
        for line in fh:
            if '"text_token"' not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") != "text_token":
                continue
            txt = (ev.get("data") or {}).get("text")
            t = ev.get("t")
            if txt and t is not None:
                out.append((float(t), txt))
    return out


def write_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def sample_dirs(action: str):
    root = DR_RESULTS / action
    if not root.exists():
        return
    for l3 in sorted(root.iterdir()):
        if not l3.is_dir() or l3.name.startswith("_"):
            continue
        md = l3 / MODEL
        if not md.is_dir():
            continue
        for s in sorted(md.iterdir()):
            if s.is_dir():
                yield l3.name, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=ACTIONS, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    actions = [args.action] if args.action else ACTIONS
    grand = {"n": 0, "written": 0, "no_events": 0, "no_info": 0,
             "no_tokens": 0, "no_segs": 0, "empty_all": 0}
    wps_all: list[float] = []

    for action in actions:
        st = {"n": 0, "written": 0, "no_tokens": 0, "empty_all": 0}
        wps: list[float] = []
        for l3, s in sample_dirs(action):
            grand["n"] += 1
            st["n"] += 1
            ip, ep = s / "inference_info.json", s / "events.jsonl"
            if not ip.exists():
                grand["no_info"] += 1
                continue
            if not ep.exists():
                grand["no_events"] += 1
                continue
            try:
                info = json.loads(ip.read_text())
            except Exception:
                grand["no_info"] += 1
                continue
            inf = info.get("inference") or {}
            segs = [tuple(x) for x in (inf.get("model_speech_segments") or [])]
            if not segs:
                grand["no_segs"] += 1
                continue

            toks = read_text_tokens(ep)
            if not toks:
                grand["no_tokens"] += 1
                st["no_tokens"] += 1

            texts = default_model_speech_transcripts(toks, segs)
            if not any(texts):
                st["empty_all"] += 1
                grand["empty_all"] += 1
            else:
                dur = sum(off - on for on, off in segs)
                if dur > 0:
                    wps.append(len(" ".join(texts).split()) / dur)

            if not args.dry_run:
                inf["model_speech_transcripts"] = texts
                info["inference"] = inf
                write_atomic(ip, info)
            st["written"] += 1
            grand["written"] += 1

        wps_all += wps
        med = f"{statistics.median(wps):.2f}" if wps else "--"
        print(f"  {action:<12} 样本={st['n']:<5} 写入={st['written']:<5} "
              f"无token={st['no_tokens']:<4} 全空={st['empty_all']:<4} "
              f"live词/秒中位={med}")

    print(f"\n  合计 {grand['n']} 条，"
          f"{'将写入' if args.dry_run else '已写入'} {grand['written']}，"
          f"无 events {grand['no_events']}，无 info {grand['no_info']}，"
          f"无 segment {grand['no_segs']}，无 token {grand['no_tokens']}，"
          f"分桶后全空 {grand['empty_all']}")
    if wps_all:
        print(f"  live 词/秒 中位 {statistics.median(wps_all):.2f} "
              f"（对话语速正常区间 2.5-3.5；BayLing 的 segment 内含零填充，偏低属预期）")
    if args.dry_run:
        print("  --dry_run：没有写任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
