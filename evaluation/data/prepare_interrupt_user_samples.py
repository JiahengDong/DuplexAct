#!/usr/bin/env python3
"""
evaluation/data/prepare_interrupt_user_samples.py
=================================================================
Collect and write evaluation samples for the Interrupt-User action
(TOR / Take-Over Rate: model interrupts a continuously-speaking user at a
condition-defined target point).

Source: fully synthesized (data_synthesize/interrupt_user/), no real-corpus
audio involved -- so NO WER validation here (see _wer_common.py docstring
for why: clean single-take TTS renders of a known exact script).

L3 conditions (dir names == l3_condition values written to metadata.json;
both already match evaluation/actions/interrupt_user.py's own
CONDITION_ALIASES, so no scorer-code change is needed for this action):
    interrupt_for_help    (200 available)
    interrupt_for_safety  (201 available, capped to 200)

Per-sample synthesized assets (data_synthesize/interrupt_user/<subcat>/
<scenario or ''>/<uuid>/):
    speaker_a_interrupted.wav   -- A's turn, TRAILS OFF at the ideal
                                    interrupt point (this is the continuous
                                    audio fed to the model as "the user")
    speaker_a_full_ignored.wav  -- same utterance, but complete/uninterrupted
                                    (used only to derive t_target = duration
                                    of speaker_a_interrupted.wav, and as the
                                    "what A would have said" reference text)
    speaker_b.wav               -- scripted reference interruption (NOT fed
                                    to the model; judge-context reference
                                    only, analogous to transcript_original_plan)
    speaker_a_acknowledged.wav  -- A's branch-A reaction (acknowledges B's
                                    interruption)
    (speaker_a_full_ignored.wav doubles as the branch-B "ignored" reaction
     text is the tail of the full utterance beyond t_target)

NOTE (simulator design, deferred): once the model interrupts, a live driver
must pick ONE of the two branch continuations to play for A's next turn
(acknowledged vs ignored) per this sample's PRE-ASSIGNED branch label --
this script does not decide that; it just packages both branch assets +
texts so the simulator driver (not yet implemented) can consume them.
This script computes and stores `branch_label` per the plan's earlier
random 50/50 assignment (seeded, reproducible) so the branch is fixed at
data-prep time, not decided by the simulator at run time.

OUTPUT layout (per sample):
    evaluation/data/interrupt_user/<l3_condition>/<sample_id>/
        user_speech.wav          <- speaker_a_interrupted.wav (copy)
        branch_acknowledged.wav  <- speaker_a_acknowledged.wav (copy)
        branch_ignored.wav       <- speaker_a_full_ignored.wav (copy; tail
                                     beyond t_target is the "ignored"-branch
                                     continuation content)
        metadata.json
    evaluation/data/interrupt_user/manifest.json

metadata.json fields:
    l3_condition, dataset="synthesized", source_id (uuid), scenario, topic
    or hazard_type, branch_label ("acknowledged"|"ignored"),
    t_user_start=0.0, t_target (seconds; = duration of user_speech.wav),
    transcript_user (text of user_speech.wav),
    transcript_user_full (text of the complete/uninterrupted utterance),
    reference_interruption_text (speaker_b's scripted line -- judge
    reference only), branch_acknowledged_text, branch_ignored_text.

Usage:
    python prepare_interrupt_user_samples.py [--target 200] [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import wave
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

from typing import Optional

BASE = _duplexact_root()
SYNTH_ROOT = BASE / "data_synthesize" / "interrupt_user"
OUT_ROOT = BASE / "evaluation" / "data" / "interrupt_user"

SUBCATS = {
    "interrupt_for_help": "interrupt_for_help",
    "interrupt_for_safety": "interrupt_for_safety",
}
TARGET_PER_CONDITION = 200
RANDOM_SEED = 42


def _wav_duration(path: Path) -> float:
    with wave.open(str(path)) as wf:
        return wf.getnframes() / wf.getframerate()


def _find_samples(subcat_dir: Path) -> list[Path]:
    """Each sample is a directory containing tts_metadata.json, at any depth
    (interrupt_for_help has a scenario subdir; interrupt_for_safety is flat)."""
    return sorted(p.parent for p in subcat_dir.rglob("tts_metadata.json"))


def _files_by_label(meta: dict) -> dict[str, dict]:
    return {f["label"]: f for f in meta.get("files", [])}


def prepare_one(sample_dir: Path, l3_condition: str, sample_id: str,
                 out_dir: Path, rng: random.Random, dry_run: bool) -> Optional[dict]:
    meta = json.loads((sample_dir / "tts_metadata.json").read_text())
    files = _files_by_label(meta)
    required = ["speaker_a_interrupted", "speaker_a_full_ignored",
                "speaker_b", "speaker_a_acknowledged"]
    if any(r not in files for r in required):
        return None

    user_speech_src = sample_dir / files["speaker_a_interrupted"]["file"]
    full_ignored_src = sample_dir / files["speaker_a_full_ignored"]["file"]
    ack_src = sample_dir / files["speaker_a_acknowledged"]["file"]
    for p in (user_speech_src, full_ignored_src, ack_src):
        if not p.exists():
            return None

    t_target = round(_wav_duration(user_speech_src), 4)
    branch_label = rng.choice(["acknowledged", "ignored"])

    record = {
        "l3_condition": l3_condition,
        "dataset": "synthesized",
        "source_id": meta["sample_id"],
        "scenario": None,
        "topic": None,
        "hazard_type": None,
        "branch_label": branch_label,
        "t_user_start": 0.0,
        "t_target": t_target,
        "transcript_user": files["speaker_a_interrupted"]["tts_text"],
        "transcript_user_full": files["speaker_a_full_ignored"]["tts_text"],
        "reference_interruption_text": files["speaker_b"]["tts_text"],
        "branch_acknowledged_text": files["speaker_a_acknowledged"]["tts_text"],
        "branch_ignored_text": files["speaker_a_full_ignored"]["tts_text"],
    }

    # scenario / topic / hazard_type live in the sibling transcript.json (not
    # tts_metadata.json) -- best-effort enrichment, never required.
    transcript_path = sample_dir / "transcript.json"
    if transcript_path.exists():
        try:
            t = json.loads(transcript_path.read_text())
            record["scenario"] = t.get("scenario")
            record["topic"] = t.get("topic")
            record["hazard_type"] = t.get("hazard_type")
        except Exception:
            pass

    if dry_run:
        return record

    sd = out_dir / sample_id
    sd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(user_speech_src, sd / "user_speech.wav")
    shutil.copyfile(ack_src, sd / "branch_acknowledged.wav")
    shutil.copyfile(full_ignored_src, sd / "branch_ignored.wav")
    (sd / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_PER_CONDITION)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(RANDOM_SEED)
    manifest: list[dict] = []

    for l3, subdir_name in SUBCATS.items():
        subcat_dir = SYNTH_ROOT / subdir_name
        sample_dirs = _find_samples(subcat_dir)
        rng2 = random.Random(RANDOM_SEED)
        rng2.shuffle(sample_dirs)
        sample_dirs = sample_dirs[: args.target]

        out_dir = OUT_ROOT / l3
        print(f"[{l3}] {len(sample_dirs)} candidates -> {out_dir}")

        n_ok = 0
        for i, sd in enumerate(sample_dirs):
            sample_id = f"synth_{l3}_{i:04d}"
            rec = prepare_one(sd, l3, sample_id, out_dir, rng, args.dry_run)
            if rec is None:
                print(f"  [skip] {sd.name}: missing required assets")
                continue
            rec["sample_id"] = sample_id
            manifest.append(rec)
            n_ok += 1
        print(f"[{l3}] prepared {n_ok}/{len(sample_dirs)}")

    if not args.dry_run:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nManifest -> {OUT_ROOT / 'manifest.json'} ({len(manifest)} samples)")

    by_l3: dict[str, int] = {}
    for r in manifest:
        by_l3[r["l3_condition"]] = by_l3.get(r["l3_condition"], 0) + 1
    print("\nSummary:", by_l3)


if __name__ == "__main__":
    main()
