#!/usr/bin/env python3
"""
evaluation/direct_replay/data/prepare_respond_recover_samples.py
====================================================================
Collect and write evaluation samples for the Respond & Recover action
(TOR: model starts the required response/repair inside the recovery region
after the user's interruption ends).

Same 4-turn interruption structure as `stop` (T1 lead-in -> model's own T2
-> T3 interruption injected by a future simulator -> model's T4-like
recovery), but here we ALSO care about *what kind* of recovery the model
should produce, so the interruption's semantic content (T3 text) and a
recovery reference (T4 text) are packaged as well.

L3 conditions (6 total; dir names chosen to match the *_after_interruption
convention already used as CONDITION_ALIASES keys in
evaluation/actions/respond_and_recover.py for the 3 synthesized ones):
    enrich_response_after_interruption    real, pooled across all 4 datasets
    shift_topic_after_interruption        real, pooled across all 4 datasets
    correct_response_after_interruption   real, pooled across all 4 datasets
                                           (added 2026-09-02 -- GPT
                                           extraction classifies a "correct"
                                           recovery_type -- A corrects an
                                           error in B's turn -- across all
                                           three real corpora, but no L3
                                           condition ever drew from it: 292
                                           valid hard candidates (Candor 162
                                           + AMI 101 + CORAAL 29 + Loqui 0)
                                           sat completely unused. Added per
                                           user request as a 6th condition,
                                           data-prep only -- rubric left
                                           untouched; scoring-side wiring is
                                           part of the same still-open GAP #1
                                           below as enrich/shift_topic.)
    shorten_response_after_interruption   synthesized, 200 avail
    repeat_response_after_interruption    synthesized, 200 avail
    ask_follow_up_after_interruption      synthesized, 200 avail

GAP #1 (still open, flagged for joint review): evaluation/actions/
respond_and_recover.py's CONDITION_ALIASES dict has entries for
shorten_response_after_interruption / repeat_response_after_interruption /
ask_follow_up_after_interruption only. It has NO entries for
enrich_response_after_interruption, shift_topic_after_interruption, or
(as of 2026-09-02) correct_response_after_interruption -- our three
REAL-data conditions -- and its L3_WINDOWS dict has no "enrich"/
"shift_topic"/"correct" key either -- so as written, scoring a real-data
instance today would hit `unknown_condition:...`. Three L3_WINDOWS entries
+ three CONDITION_ALIASES entries (and ideally three extra per-condition
branches in the natural_recovery_pos judge prompt, mirroring the existing
3-way branching) need to be added before these can be scored. This is a
scoring-logic decision, left for joint review -- not touched here.

Dataset status (2026-09-01, after fixing the CORAAL missing-metadata bug
and re-downloading AMI/CORAAL raw audio) -- see interruption_pool_common.py
for the full per-dataset convention writeup. Per user request, all four
real-corpus datasets (Candor/AMI/CORAAL/Loqui) are now POOLED TOGETHER
(shuffled, fixed seed) for both real conditions, instead of Candor-only:
    enrich_response_after_interruption : Candor 267 + AMI 78 + CORAAL 38
                                          + Loqui 1 = 384 avail (>= 200)
    shift_topic_after_interruption     : Candor 477 + AMI 134 + CORAAL 14
                                          + Loqui 0 = 625 avail (>= 200)
    correct_response_after_interruption: Candor 162 + AMI 101 + CORAAL 29
                                          + Loqui 0 = 292 avail (>= 200)

Dataset for the 3 synthesized conditions: data_synthesize/respond_and_recover/
<subcategory>/<scenario>/<uuid>/ (ElevenLabs TTS, 4 turns: turn_00_speaker_a,
turn_01_speaker_b [reference only], turn_02_speaker_a [=cue/interruption],
turn_03_speaker_b [=recovery reference]).

OUTPUT layout (per sample):
    evaluation/direct_replay/data/respond_and_recover/<l3_condition>/<sample_id>/
        lead_in.wav   <- T1 / turn_00 audio
        cue.wav       <- T3 / turn_02 audio (interruption; simulator-injected)
        metadata.json
    evaluation/direct_replay/data/respond_and_recover/manifest.json

metadata.json fields:
    l3_condition, dataset ("Candor"|"AMI"|"CORAAL"|"Loqui"|"synthesized"),
    source_id, scenario (four conversation scenes; synthesized only),
    transcript_user (T1/turn_00 text),
    reference_original_response_text (T2/turn_01 text -- NOT fed, judge-context
    only), reference_interruption_text (T3/turn_02 text),
    reference_recovery_text (T4/turn_03 text -- judge-context reference for
    "what a good recovery looks like"), interruption_type (real datasets
    only, diagnostic), gpt_reason (real datasets only).

Usage:
    python prepare_respond_recover_samples.py [--target 200] [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
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
SYNTH_ROOT = BASE / "data_synthesize" / "respond_and_recover"
OUT_ROOT = BASE / "evaluation" / "direct_replay" / "data" / "respond_and_recover"

TARGET = 200
RANDOM_SEED = 42
REAL_DATASETS = ["Candor", "AMI", "CORAAL", "Loqui"]

REAL_CONDITIONS = {
    # dir/l3_condition names match evaluation/actions/respond_and_recover.py's
    # CONDITION_ALIASES keys exactly (which in turn mirror the rubric's own
    # "Data Aliases" table) -- "enrich_response_after_interruption" (NOT
    # "enrich_after_interruption"; note the "_response").
    "enrich_response_after_interruption": {"enrich"},
    "shift_topic_after_interruption": {"shift_topic"},
    "correct_response_after_interruption": {"correct"},
}
SYNTH_CONDITIONS = {
    "shorten_response_after_interruption": "shorten_response_after_interruption",
    "repeat_response_after_interruption": "repeat_response_after_interruption",
    "ask_follow_up_after_interruption": "ask_follow_up_after_interruption",
}


def prepare_real(l3_condition: str, itypes: set[str], target: int, dry_run: bool) -> list[dict]:
    pool = collect_pool(REAL_DATASETS, itypes=itypes)
    print(f"[{l3_condition}] combined pool: {len(pool)} candidates "
          f"({dict(Counter(item['dataset'] for item in pool))})")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pool)

    out_dir = OUT_ROOT / l3_condition
    records: list[dict] = []
    n_ok = 0
    n_by_dataset: Counter = Counter()
    for item in pool:
        if n_ok >= target:
            break
        audio_pair = resolve_audio(item)
        if audio_pair is None:
            continue
        t1_wav, t3_wav = audio_pair
        ds, entry = item["dataset"], item["entry"]

        sample_id = f"{ds}_{l3_condition}_{n_ok:04d}"
        record = {
            "sample_id": sample_id,
            "l3_condition": l3_condition,
            "dataset": ds,
            "source_id": source_id(item),
            "interruption_type": entry.get("interruption_type"),
            "transcript_user": entry.get("t1_text", ""),
            "reference_original_response_text": entry.get("t2_text", ""),
            "reference_interruption_text": entry.get("t3_text", ""),
            "reference_recovery_text": entry.get("t4_text", ""),
            "gpt_reason": entry.get("gpt_reason", ""),
        }
        if not dry_run:
            sd = out_dir / sample_id
            write_clip(ds, entry, t1_wav, "t1", sd / "lead_in.wav")
            write_clip(ds, entry, t3_wav, "t3", sd / "cue.wav")
            (sd / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        records.append(record)
        n_by_dataset[ds] += 1
        n_ok += 1

    print(f"[{l3_condition}] prepared {n_ok}/{target}  by dataset: {dict(n_by_dataset)}")
    return records


def _find_synth_samples(subcat_dir: Path) -> list[Path]:
    return sorted(p.parent for p in subcat_dir.rglob("tts_metadata.json"))


def _files_by_label(meta: dict) -> list[dict]:
    return meta.get("files", [])


def prepare_synth(l3_condition: str, subcat_name: str, target: int, dry_run: bool) -> list[dict]:
    subcat_dir = SYNTH_ROOT / subcat_name
    sample_dirs = _find_synth_samples(subcat_dir)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(sample_dirs)
    sample_dirs = sample_dirs[:target]
    print(f"[{l3_condition}] synthesized pool: {len(sample_dirs)} candidates")

    out_dir = OUT_ROOT / l3_condition
    records: list[dict] = []
    n_ok = 0
    for i, sd in enumerate(sample_dirs):
        meta = json.loads((sd / "tts_metadata.json").read_text())
        files = _files_by_label(meta)
        by_file = {f["file"]: f for f in files}
        turn_files = ["turn_00_speaker_a.wav", "turn_01_speaker_b.wav",
                      "turn_02_speaker_a.wav", "turn_03_speaker_b.wav"]
        if any(tf not in by_file for tf in turn_files):
            continue
        if any(not (sd / tf).exists() for tf in turn_files):
            continue

        sample_id = f"synth_{l3_condition}_{i:04d}"
        scenario = sd.parent.name
        if (sd / "transcript.json").exists():
            scenario = json.loads((sd / "transcript.json").read_text()).get("scenario") or scenario
        record = {
            "sample_id": sample_id,
            "l3_condition": l3_condition,
            "dataset": "synthesized",
            "source_id": meta["sample_id"],
            "scenario": scenario,
            "interruption_type": None,
            "transcript_user": by_file["turn_00_speaker_a.wav"]["tts_text"],
            "reference_original_response_text": by_file["turn_01_speaker_b.wav"]["tts_text"],
            "reference_interruption_text": by_file["turn_02_speaker_a.wav"]["tts_text"],
            "reference_recovery_text": by_file["turn_03_speaker_b.wav"]["tts_text"],
            "gpt_reason": "",
        }
        if not dry_run:
            out_sd = out_dir / sample_id
            out_sd.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sd / "turn_00_speaker_a.wav", out_sd / "lead_in.wav")
            shutil.copyfile(sd / "turn_02_speaker_a.wav", out_sd / "cue.wav")
            (out_sd / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        records.append(record)
        n_ok += 1

    print(f"[{l3_condition}] prepared {n_ok}/{target}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    manifest: list[dict] = []
    for l3, itypes in REAL_CONDITIONS.items():
        manifest.extend(prepare_real(l3, itypes, args.target, args.dry_run))
    for l3, subcat in SYNTH_CONDITIONS.items():
        manifest.extend(prepare_synth(l3, subcat, args.target, args.dry_run))

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
