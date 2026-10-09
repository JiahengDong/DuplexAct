#!/usr/bin/env python3
"""
evaluation/direct_replay/data/prepare_stop_samples.py
========================================================
Collect and write evaluation samples for the Stop action (YR / Yield Rate:
model must end its current response and relinquish the floor inside the
valid yield interval once the user interrupts).

Conversation pattern (source recording, 4-turn interruption structure):
    T1: A speaks  ->  T2: B starts responding  ->  T3: A interrupts back
                                                 ->  T4: B recovers (NOT used here)
Our model plays the B role. T1 is fed as the initial "user" turn (prompts
the model to start speaking); T3 is the pre-recorded interruption clip that
a future SIMULATOR driver injects into the model's live output stream at
the appropriate live moment (NOT at a fixed offset -- see plan notes: "no
need for drain wait anymore", fixed-window designs like the old
demo_initial_eval/prepare_stop_samples.py's WAIT_TIME_S=7s hack are
explicitly superseded). This script only packages the ASSETS; the
simulator (not yet implemented) decides WHEN to play cue.wav during
inference.

L3 condition coverage -- IMPORTANT GAP (still open, flagged for joint
review): evaluation/actions/stop.py defines THREE L3 conditions with
distinct yield windows: direct_interruption, urgent_stop_cue,
correction_redirection. None of our real-corpus interruption pools
(Candor/AMI/CORAAL/Loqui) carry an "urgency" or "is this a correction"
annotation -- interruption_type there labels the ORIGINAL turn-4 RECOVERY
behaviour (correct/enrich/shorten/repeat/shift_topic/ask_followup/none), a
different axis entirely. So this script only produces "direct_interruption"
samples (pooling ALL interruption_type/style="hard" candidates, since
recovery type is irrelevant to whether/how fast the model yields).
urgent_stop_cue and correction_redirection have NO sourced data yet -- they
would need new synthesized TTS clips (analogous to interrupt_user's
hazard-cue design).

Dataset status (2026-09-01, after fixing the CORAAL missing-metadata bug
and re-downloading AMI/CORAAL raw audio):
    Candor  : READY.  969 hard candidates (any recovery type), audio ready.
    AMI     : READY.  337 hard candidates, audio (re-)exported via
              scripts/AMI/slurm_export_interruptions_ami_audio.sh.
    CORAAL  : READY.  89 hard candidates, audio + metadata now both
              persisted correctly (was previously blocked -- see
              interruption_pool_common.py's module docstring).
    Loqui   : READY but negligible -- only 1 hard candidate total.

Per user request, all four are now POOLED TOGETHER (not "Candor first,
others as overflow") and randomly shuffled with a fixed seed before
drawing `--target` samples, so the resulting set is genuinely
dataset-diverse in proportion to each dataset's availability rather than
100% Candor (which alone has enough candidates to fill the whole target).

OUTPUT layout (per sample):
    evaluation/direct_replay/data/stop/direct_interruption/<sample_id>/
        lead_in.wav   <- T1 audio (fed first, prompts the model to speak)
        cue.wav       <- T3 audio (the interruption; simulator-injected)
        metadata.json
    evaluation/direct_replay/data/stop/manifest.json
    evaluation/direct_replay/data/stop/wer_report.json   (WER validation of
                                                            lead_in.wav / cue.wav
                                                            against Whisper)

metadata.json fields:
    l3_condition="direct_interruption", dataset, source_id,
    interruption_type (diagnostic only, not used for scoring),
    transcript_user (T1 text, what lead_in.wav says),
    reference_interruption_text (T3 text, what cue.wav says),
    reference_original_response_text (T2 text -- what B originally said
    before being interrupted; judge-context reference only, never fed),
    gpt_reason.

Usage:
    python prepare_stop_samples.py [--target 200] [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

def _duplexact_root() -> Path:
    import os
    for key in ("DUPLEXACT_ROOT", "BENCH_ROOT"):
        v = os.environ.get(key)
        if v:
            return Path(v).resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "evaluation" / "paths.py").exists():
            return p
    return here.parents[min(3, len(here.parents)-1)]


sys.path.insert(0, str(Path(__file__).parent))
from interruption_pool_common import collect_pool, resolve_audio, write_clip, source_id  # noqa: E402

BASE = _duplexact_root()
OUT_ROOT = BASE / "evaluation" / "direct_replay" / "data" / "stop"

L3_CONDITION = "direct_interruption"
DATASETS = ["Candor", "AMI", "CORAAL", "Loqui"]
TARGET = 200
RANDOM_SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    pool = collect_pool(DATASETS, itypes=None)  # any recovery type
    print(f"[direct_interruption] combined pool: {len(pool)} candidates "
          f"({dict(Counter(item['dataset'] for item in pool))})")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pool)

    out_dir = OUT_ROOT / L3_CONDITION
    manifest: list[dict] = []
    n_ok = 0
    n_by_dataset: Counter = Counter()
    for item in pool:
        if n_ok >= args.target:
            break
        audio_pair = resolve_audio(item)
        if audio_pair is None:
            continue
        t1_wav, t3_wav = audio_pair
        ds, entry = item["dataset"], item["entry"]

        sample_id = f"{ds}_stop_{n_ok:04d}"
        record = {
            "sample_id": sample_id,
            "l3_condition": L3_CONDITION,
            "dataset": ds,
            "source_id": source_id(item),
            "interruption_type": entry.get("interruption_type"),
            "transcript_user": entry.get("t1_text", ""),
            "reference_original_response_text": entry.get("t2_text", ""),
            "reference_interruption_text": entry.get("t3_text", ""),
            "gpt_reason": entry.get("gpt_reason", ""),
        }

        if not args.dry_run:
            sd = out_dir / sample_id
            write_clip(ds, entry, t1_wav, "t1", sd / "lead_in.wav")
            write_clip(ds, entry, t3_wav, "t3", sd / "cue.wav")
            (sd / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

        manifest.append(record)
        n_by_dataset[ds] += 1
        n_ok += 1

    print(f"[direct_interruption] prepared {n_ok}/{args.target}  by dataset: {dict(n_by_dataset)}")
    print("\n[GAP] urgent_stop_cue / correction_redirection: 0 samples -- no data source "
          "identified yet (see module docstring). Flagged for joint review.")

    if not args.dry_run:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nManifest -> {OUT_ROOT / 'manifest.json'} ({len(manifest)} samples)")


if __name__ == "__main__":
    main()
