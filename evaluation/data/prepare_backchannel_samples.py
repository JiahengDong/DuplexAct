"""
evaluation/prepare_backchannel_samples.py
==========================================
Collect and write evaluation samples for the Backchannel action.

L3 conditions
─────────────
  backchannel_affirmative  – acknowledgement + understanding + agreement
  backchannel_uncertainty  – uncertainty

Sources (all four datasets)
────────────────────────────
  CORAAL  · AMI  · Candor  · Loqui

Sample targets
──────────────
  backchannel_affirmative : 200  (equal-balance across 3 source types;
                                  macro-balanced across 4 datasets)
  backchannel_uncertainty : all available (~236)

Audio file lookup (dataset-specific)
─────────────────────────────────────
  CORAAL  →  {file_id}_bc{before_idx:04d}_full.wav
  Candor  →  {conv_id}_t{t1_turn_id}_t1_full.wav
  Loqui   →  {file_id}_bc{bc_idx:04d}_full.wav
  AMI     →  {meeting}_{spkA}x{spkB}_fb{seq:04d}_t1_full.wav
              (seq = position of entry when sorted by bc_start per spk pair)

Searched in: {dataset}/Relaxed/backchannel_refined/{type}/
             {dataset}/Relaxed/backchannel_relaxed/{type}/

Clip-relative timestamps
────────────────────────
  All timing normalised so clip_start = 0 (= before_start in original).
  bc_start_clip = bc_start_orig - clip_audio_start
  bc_end_clip   = bc_end_orig   - clip_audio_start

  clip_audio_start:
    CORAAL / Loqui  →  entry["before_start"]
    Candor / AMI    →  entry["t1_start"]

OUTPUT metadata.json
─────────────────────
{
  "l3_condition":      str,
  "dataset":           str,
  "source_id":         str,
  "bc_type_original":  str,

  "clip_start":        0.0,
  "clip_end":          float,   # actual WAV duration
  "clip_duration":     float,
  "padding_start":     float,
  "padding_end":       float,
  "padding_time":      15.0,
  "input_duration":    float,

  "bc_start":          float,   # clip-relative
  "bc_end":            float,   # clip-relative
  "ab_gap":            float,
  "bc_text":           str,
  "bc_duration":       float,

  "before_text":       str,
  "after_text":        str
}

Usage
─────
    python prepare_backchannel_samples.py \\
        --out_dir evaluation/data/backchannel \\
        [--target_affirmative 200]
        [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import random
import wave
from collections import defaultdict
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

import numpy as np
import scipy.signal as _sig
import soundfile as sf

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE = _duplexact_root()

DATASET_REFINED = {
    "CORAAL": BASE / "results/CORAAL/Relaxed/backchannel_refined",
    "Candor": BASE / "results/Candor/Relaxed/backchannel_refined",
    "AMI":    BASE / "results/AMI/Relaxed/backchannel_refined",
    "Loqui":  BASE / "results/Loqui/Relaxed/backchannel_refined",
}
DATASET_ALT = {
    "CORAAL": BASE / "results/CORAAL/Relaxed/backchannel_relaxed",
    "Candor": BASE / "results/Candor/Relaxed/backchannel_relaxed",
    "AMI":    BASE / "results/AMI/Relaxed/backchannel_relaxed",
    "Loqui":  BASE / "results/Loqui/Relaxed/backchannel_relaxed",
}
ALL_TYPES = ["acknowledgement", "understanding", "agreement",
             "uncertainty", "disagreement", "none"]

# ── Constants ──────────────────────────────────────────────────────────────────

PADDING_TIME_S      = 15.0
OUTPUT_SR           = 16_000
TARGET_AFF          = 200
TARGET_UNC          = 200
RANDOM_SEED         = 42

AFFIRMATIVE_TYPES   = {"acknowledgement", "understanding", "agreement"}
UNCERTAINTY_TYPES   = {"uncertainty"}

# ── Helpers: per-dataset audio resolution ─────────────────────────────────────

def _build_audio_index(dataset: str) -> dict[str, Path]:
    """
    Index all *full*.wav files for a dataset (all types).
    Used only for CORAAL and Candor (which lack a direct path field).
    """
    index: dict[str, Path] = {}
    for src_dir in [DATASET_REFINED[dataset], DATASET_ALT[dataset]]:
        for t in ALL_TYPES:
            d = src_dir / t
            if not d.exists():
                continue
            for f in d.iterdir():
                if "full" in f.name and f.suffix == ".wav":
                    index[f.name] = f
    return index


def _resolve_audio(dataset: str, entry: dict,
                   audio_index: dict[str, Path]) -> Optional[Path]:
    """
    Return the Path to the full.wav for this entry, or None if missing.

    AMI / Loqui   → direct path stored in recategorize_audio_path
    CORAAL        → {file_id}_bc{before_idx:04d}_full.wav  (in audio_index)
    Candor        → {conv_id}_t{t1_turn_id}_t1_full.wav    (in audio_index)
    """
    if dataset in ("AMI", "Loqui"):
        p = entry.get("recategorize_audio_path")
        return Path(p) if p else None
    if dataset == "CORAAL":
        key = f"{entry['file_id']}_bc{entry['before_idx']:04d}_full.wav"
        return audio_index.get(key)
    if dataset == "Candor":
        key = f"{entry['conv_id']}_t{entry['t1_turn_id']}_t1_full.wav"
        return audio_index.get(key)
    raise ValueError(f"Unknown dataset: {dataset}")


def _clip_audio_start(dataset: str, entry: dict) -> float:
    """Absolute timestamp that corresponds to clip t=0."""
    if dataset in ("CORAAL", "Loqui"):
        return entry["before_start"]
    return entry["t1_start"]   # Candor, AMI


def _source_id(dataset: str, entry: dict) -> str:
    if dataset == "CORAAL":
        return f"{entry['file_id']}_bc{entry['before_idx']:04d}"
    if dataset == "Loqui":
        return f"{entry['file_id']}_bc{entry['bc_idx']:04d}"
    if dataset == "Candor":
        return f"{entry['conv_id']}_t{entry['t1_turn_id']}"
    if dataset == "AMI":
        return f"{entry['meeting']}_{entry['spk_a']}x{entry['spk_b']}_bc{entry['bc_start']:.3f}"
    raise ValueError(dataset)

# ── Audio I/O ──────────────────────────────────────────────────────────────────

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _trim_bounds(dataset: str, entry: dict) -> Optional[tuple[float, float]]:
    """
    (trim_start_s, trim_end_s) relative to the RESOLVED audio file's own
    t=0, or None if the resolved file needs no trimming.

    CORAAL/Loqui only: their extraction scripts
    always export "_full.wav" spanning [t0_start, after_end] -- t0 being the
    BACKCHANNELING speaker's own earlier, unrelated turn (the extraction
    script's own comment labels it "T0: B trigger") -- not [before_start,
    after_end] as this module originally (incorrectly) assumed when
    computing clip-relative bc_start/bc_end. Rather than accept t0 into the
    model's input (it isn't part of what the simulated "user"/speaker-A
    says at all) we trim it off: keep only [before_start, after_end],
    computed here relative to the file's own start (t0_start). See
    data/fix_backchannel_coraal_loqui.py (the one-off repair script that
    applied this same fix retroactively to already-written samples) for
    the full incident writeup and empirical verification (Whisper
    transcript of the trimmed clip vs. the original t0-inclusive one).

    AMI/Candor: their own "_t1_full.wav" is already scoped to
    [t1_start, t1_end] with no such prefix -- no trim needed.
    """
    if dataset not in ("CORAAL", "Loqui"):
        return None
    t0_start = entry.get("t0_start")
    before_start = entry.get("before_start")
    after_end = entry.get("after_end")
    if t0_start is None or before_start is None or after_end is None:
        return None
    return (max(0.0, before_start - t0_start), after_end - t0_start)


def _build_padded_wav(src: Path, trim: Optional[tuple[float, float]] = None) -> tuple[np.ndarray, int, float]:
    audio, orig_sr = sf.read(str(src), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if trim is not None:
        s_idx = int(round(trim[0] * orig_sr))
        e_idx = min(len(audio), int(round(trim[1] * orig_sr)))
        audio = audio[s_idx:e_idx]
    if orig_sr != OUTPUT_SR:
        g = _gcd(orig_sr, OUTPUT_SR)
        audio = _sig.resample_poly(audio, OUTPUT_SR // g, orig_sr // g).astype(np.float32)
    clip_dur = len(audio) / OUTPUT_SR
    pad = np.zeros(int(PADDING_TIME_S * OUTPUT_SR), dtype=np.float32)
    return np.concatenate([audio, pad]), OUTPUT_SR, clip_dur

# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(dataset: str) -> tuple[list[dict], dict[str, Path]]:
    """Returns (entries, audio_index)."""
    vj = DATASET_REFINED[dataset] / "valid.json"
    entries = json.loads(vj.read_text())
    audio_index = _build_audio_index(dataset)
    return entries, audio_index

# ── Sample selection ───────────────────────────────────────────────────────────

def _has_audio(dataset: str, entry: dict, audio_index: dict[str, Path]) -> bool:
    return _resolve_audio(dataset, entry, audio_index) is not None


def select_samples(
    datasets: list[str],
    target_aff: int,
    seed: int,
    **kwargs,
) -> dict[str, list[tuple[str, dict]]]:
    """
    Returns { l3_condition: [(dataset, entry), ...] }

    Affirmative: equal balance across 3 source types; macro-balance across
    datasets (each dataset contributes at most ceil(target / n_datasets) samples).

    Uncertainty: all available samples across all datasets.
    """
    rng = random.Random(seed)

    all_aff: dict[str, list[tuple[str, dict]]] = defaultdict(list)  # type → [(ds,e)]
    all_unc: list[tuple[str, dict]] = []

    for ds in datasets:
        entries, audio_index = load_dataset(ds)
        for e in entries:
            if not _has_audio(ds, e, audio_index):
                continue
            bt = e["backchannel_type"]
            if bt in AFFIRMATIVE_TYPES:
                all_aff[bt].append((ds, e))
            elif bt in UNCERTAINTY_TYPES:
                all_unc.append((ds, e))

    # ── Affirmative: dataset-first balance, then type-balance within each quota ─
    per_ds  = -(-target_aff // len(datasets))   # ceil(200/4) = 50
    per_type_per_ds = -(-per_ds // len(all_aff))  # ceil(50/3) ≈ 17

    selected_aff: list[tuple[str, dict]] = []

    for ds in datasets:
        # Collect this dataset's entries per type, shuffled
        by_type: dict[str, list] = defaultdict(list)
        for bt, pool in all_aff.items():
            for d, e in pool:
                if d == ds:
                    by_type[bt].append((d, e))
        for bt in by_type:
            rng.shuffle(by_type[bt])

        # First pass: up to per_type_per_ds from each type
        ds_selected: list[tuple[str, dict]] = []
        taken = {}
        for bt in sorted(all_aff.keys()):
            take = min(per_type_per_ds, len(by_type[bt]))
            ds_selected.extend(by_type[bt][:take])
            taken[bt] = take

        # Backfill within dataset: if under per_ds quota, draw more from any type
        # prioritising ack (most abundant), then agr, then und
        if len(ds_selected) < per_ds:
            for bt in ['acknowledgement', 'agreement', 'understanding']:
                if len(ds_selected) >= per_ds:
                    break
                already = taken.get(bt, 0)
                extra = by_type[bt][already:]
                need  = per_ds - len(ds_selected)
                ds_selected.extend(extra[:need])
                taken[bt] = already + min(len(extra), need)

        rng.shuffle(ds_selected)
        selected_aff.extend(ds_selected[:per_ds])

    rng.shuffle(selected_aff)
    selected_aff = selected_aff[:target_aff]

    # ── Uncertainty: dataset-first balance, cap at TARGET_UNC ─────────────────
    target_unc  = kwargs.get("target_unc", TARGET_UNC)
    per_ds_unc  = -(-target_unc // len(datasets))   # ceil(200/4) = 50

    # Group by dataset, shuffle within
    unc_by_ds: dict[str, list] = defaultdict(list)
    for ds, e in all_unc:
        unc_by_ds[ds].append((ds, e))
    for ds in unc_by_ds:
        rng.shuffle(unc_by_ds[ds])

    # First pass: up to per_ds_unc from each dataset
    selected_unc: list[tuple[str, dict]] = []
    taken_unc: dict[str, int] = {}
    for ds in datasets:
        take = min(per_ds_unc, len(unc_by_ds[ds]))
        selected_unc.extend(unc_by_ds[ds][:take])
        taken_unc[ds] = take

    # Backfill from datasets with surplus (sorted by availability descending)
    if len(selected_unc) < target_unc:
        surplus_ds = sorted(datasets,
                            key=lambda d: len(unc_by_ds[d]) - taken_unc.get(d, 0),
                            reverse=True)
        for ds in surplus_ds:
            if len(selected_unc) >= target_unc:
                break
            already = taken_unc.get(ds, 0)
            extra   = unc_by_ds[ds][already:]
            need    = target_unc - len(selected_unc)
            selected_unc.extend(extra[:need])

    rng.shuffle(selected_unc)
    selected_unc = selected_unc[:target_unc]

    return {
        "backchannel_affirmative": selected_aff,
        "backchannel_uncertainty":  selected_unc,
    }

# ── Write one sample ───────────────────────────────────────────────────────────

def write_sample(
    dataset: str,
    entry: dict,
    audio_index: dict[str, Path],
    l3_condition: str,
    sample_dir: Path,
    dry_run: bool,
) -> dict:
    src = _resolve_audio(dataset, entry, audio_index)
    t0  = _clip_audio_start(dataset, entry)
    trim = _trim_bounds(dataset, entry)

    bc_start_clip = round(entry["bc_start"] - t0, 4)
    bc_end_clip   = round(entry["bc_end"]   - t0, 4)

    if not dry_run:
        sample_dir.mkdir(parents=True, exist_ok=True)
        padded, sr, clip_dur = _build_padded_wav(src, trim=trim)
        sf.write(str(sample_dir / "input.wav"), padded, sr, subtype="PCM_16")
    else:
        if trim is not None:
            clip_dur = round(trim[1] - trim[0], 4)
        else:
            with wave.open(str(src)) as wf:
                clip_dur = round(wf.getnframes() / wf.getframerate(), 4)

    meta = {
        "l3_condition":     l3_condition,
        "dataset":          dataset,
        "source_id":        _source_id(dataset, entry),
        "bc_type_original": entry["backchannel_type"],

        "clip_start":       0.0,
        "clip_end":         round(clip_dur, 4),
        "clip_duration":    round(clip_dur, 4),
        "padding_start":    round(clip_dur, 4),
        "padding_end":      round(clip_dur + PADDING_TIME_S, 4),
        "padding_time":     PADDING_TIME_S,
        "input_duration":   round(clip_dur + PADDING_TIME_S, 4),

        "bc_start":         bc_start_clip,
        "bc_end":           bc_end_clip,
        "ab_gap":           entry.get("ab_gap", entry.get("bc_gap")),
        "bc_text":          entry["bc_text"],
        "bc_duration":      entry.get("bc_duration",
                                round(entry["bc_end"] - entry["bc_start"], 4)),

        "before_text":      entry.get("before_text", ""),
        "after_text":       entry.get("after_text",  ""),
    }

    if not dry_run:
        (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return meta

# ── Manifest ──────────────────────────────────────────────────────────────────

def write_manifest(out_dir: Path, records: list[dict]) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2))
    print(f"  manifest.json  ({len(records)} entries)")

# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = ["CORAAL", "Candor", "AMI", "Loqui"]

L3_SUBDIR = {
    "backchannel_affirmative": "backchannel_affirmative",
    "backchannel_uncertainty":  "backchannel_uncertainty",
}
PREFIX = {
    "backchannel_affirmative": "bc_aff",
    "backchannel_uncertainty":  "bc_unc",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path,
                        default=BASE / "evaluation/data/backchannel")
    parser.add_argument("--target_affirmative", type=int, default=TARGET_AFF)
    parser.add_argument("--target_uncertainty",  type=int, default=TARGET_UNC)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("Selecting samples …")
    plan = select_samples(DATASETS, args.target_affirmative, RANDOM_SEED,
                          target_unc=args.target_uncertainty)

    # Print plan summary
    for l3, pool in plan.items():
        by_ds  = defaultdict(int)
        by_src = defaultdict(int)
        for ds, e in pool:
            by_ds[ds]  += 1
            by_src[e["backchannel_type"]] += 1
        ds_str  = ", ".join(f"{k}={v}" for k, v in sorted(by_ds.items()))
        src_str = ", ".join(f"{k}={v}" for k, v in sorted(by_src.items()))
        print(f"  {l3}: {len(pool)} samples")
        print(f"    by dataset:     {ds_str}")
        print(f"    by source type: {src_str}")

    if args.dry_run:
        print("\n[dry_run] No files written.")
        return

    # Load audio indexes once per dataset
    ds_audio: dict[str, dict[str, Path]] = {}
    for ds in DATASETS:
        _, audio_index = load_dataset(ds)
        ds_audio[ds] = audio_index

    manifest_records: list[dict] = []

    for l3, pool in plan.items():
        subdir = args.out_dir / L3_SUBDIR[l3]
        pfx    = PREFIX[l3]
        # Safety: wipe any stale sample dirs from a previous run first --
        # sample_id embeds a plain enumerate() index (not stable per dataset
        # across reruns with a changed candidate pool/filter), so without
        # this a rerun can leave old + new sample dirs mixed together.
        if subdir.exists():
            import shutil
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting {len(pool)} samples → {subdir}")

        for i, (ds, entry) in enumerate(pool):
            audio_index = ds_audio[ds]
            ds_prefix   = ds[:3].upper()
            sample_id   = f"{ds_prefix}_{pfx}_{i:04d}"
            sample_dir  = subdir / sample_id
            meta = write_sample(ds, entry, audio_index,
                                l3, sample_dir, dry_run=False)
            manifest_records.append({
                "sample_id":    sample_id,
                "l3_condition": l3,
                "dataset":      ds,
                "sample_dir":   str(sample_dir),
            })
            if (i + 1) % 50 == 0 or (i + 1) == len(pool):
                print(f"  [{i+1}/{len(pool)}]")

    print("\nWriting manifest …")
    write_manifest(args.out_dir, manifest_records)
    print("Done.")


if __name__ == "__main__":
    main()
