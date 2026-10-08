"""
evaluation/prepare_turn_samples.py
===================================
Prepare SMOOTH TURN-TAKING evaluation samples for all three L3 conditions:

    immediate_take_turn       – model responds with near-zero latency (synthesized data)
    delayed_take_turn         – model may pause before responding    (synthesized data)
    slight_overlap_take_turn  – model begins just before T1 ends     (real recordings)

SYNTHESIZED CONDITIONS (immediate / delayed)
────────────────────────────────────────────
SOURCE: data_synthesize/take_turn/
  take_turn_immediately/{scenario}/{uuid}/
    transcript.json           – dialogue text + metadata
    turn_00_speaker_a.wav     – T1 audio
    turn_01_speaker_b.wav     – T2 reference

  take_turn_with_delay/{scenario}/{uuid}/
    (same structure)

4 scenarios × 50 samples = 200 per condition.

OVERLAP CONDITION (slight_overlap_take_turn)
────────────────────────────────────────────
SOURCE: pre-extracted T1 WAV files from real recordings in results/:
  results/AMI/turn_taking_ami/take_turn_with_overlap/
  results/CORAAL/turn_taking_coraal/take_turn_with_overlap/
  results/Candor/turn_taking_backbiter/take_turn_with_overlap/

Selection: AMI(all 35) + CORAAL(all 68) + Candor(97) = 200 samples.
Audio filename conventions:
  AMI    : {meeting}_{spk_a}x{spk_b}_tt{N:04d}_t1.wav  (N = sequential per meeting+spk pair)
  CORAAL : {file_id}_tt{t1_idx:04d}_t1.wav
  Candor : {conv_id}_t{t1_turn_id}_t1.wav

INPUT.WAV CONSTRUCTION (all conditions)
────────────────────────────────────────
  input.wav = T1_audio  ++  silence(PADDING_TIME_S)

METADATA SCHEMA (clip-relative, all times in seconds from start of input.wav)
──────────────────────────────────────────────────────────────────────────────
Synthesized:
{
  "l3_condition":   "immediate_take_turn" | "delayed_take_turn",
  "dataset":        "synthesized",
  "source_id":      str,       # UUID
  "scenario":       str,
  "topic":          str,
  "t1_start":       0.0,
  "t1_end":         float,
  "t1_duration":    float,
  "padding_start":  float,
  "padding_end":    float,
  "padding_time":   float,
  "input_duration": float,
  "original_gap":   null,
  "t1_text":        str,
  "t2_text":        str
}

Real (overlap):
{
  "l3_condition":   "slight_overlap_take_turn",
  "dataset":        "AMI" | "CORAAL" | "Candor",
  "source_id":      str,
  "t1_start":       0.0,
  "t1_end":         float,
  "t1_duration":    float,
  "padding_start":  float,
  "padding_end":    float,
  "padding_time":   float,
  "input_duration": float,
  "original_gap":   float,     # negative = T2 started before T1 ended
  "t1_text":        str,
  "t2_text":        str
}

OUTPUT LAYOUT
─────────────
  <out_dir>/
    synth_imm_{NNNN}/
    synth_del_{NNNN}/
    AMI_tt_ovl_{NNNN}/
    CORAAL_tt_ovl_{NNNN}/
    Candor_tt_ovl_{NNNN}/

Usage
─────
  python prepare_turn_samples.py                          # all conditions
  python prepare_turn_samples.py --conditions overlap     # only overlap
  python prepare_turn_samples.py --dry_run
  python prepare_turn_samples.py --clean
"""

from __future__ import annotations

import argparse
import json
import shutil
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
import soundfile as sf
import scipy.signal as _sig

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = _duplexact_root()
SYNTH_ROOT   = BASE / "data_synthesize" / "take_turn"
OUT_ROOT     = BASE / "evaluation" / "data" / "turn_taking"
RESULTS_ROOT = BASE / "results"

RESULTS_AMI    = RESULTS_ROOT / "AMI"    / "turn_taking_ami"
RESULTS_CORAAL = RESULTS_ROOT / "CORAAL" / "turn_taking_coraal"
RESULTS_CANDOR = RESULTS_ROOT / "Candor" / "turn_taking_backbiter"

# Condition subdirectory names under OUT_ROOT
L3_SUBDIR: dict[str, str] = {
    "immediate_take_turn":      "take_turn_immediately",
    "delayed_take_turn":        "take_turn_with_delay",
    "slight_overlap_take_turn": "take_turn_with_overlap",
}

# ── Constants ──────────────────────────────────────────────────────────────────
PADDING_TIME_S  = 15.0     # silence appended after T1 for the model to respond into
OUTPUT_SR       = 16000    # target sample rate for input.wav

# Overlap sample counts per dataset
# AMI and CORAAL: take all available (small pools)
# Candor: fill the remainder to reach OVERLAP_TARGET
OVERLAP_TARGET   = 200
OVERLAP_N_AMI    = None   # None = take all available
OVERLAP_N_CORAAL = None   # None = take all available
# OVERLAP_N_CANDOR is computed dynamically in collect_overlap_samples()

SCENARIOS = ("casual_conversation", "customer_service", "interview", "meeting")

# Map rubric L3 condition name → source subcategory folder name  (synthesized)
L3_TO_SUBCAT: dict[str, str] = {
    "immediate_take_turn": "take_turn_immediately",
    "delayed_take_turn":   "take_turn_with_delay",
}

# Short codes used in output folder names
L3_SHORT: dict[str, str] = {
    "immediate_take_turn":      "imm",
    "delayed_take_turn":        "del",
    "slight_overlap_take_turn": "ovl",
}

ALL_CONDITIONS = list(L3_SHORT.keys())


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_wav_mono_resampled(path: Path, target_sr: int) -> tuple[np.ndarray, float]:
    """Load a WAV as mono float32, resampled to target_sr. Returns (samples, duration_s)."""
    audio, src_sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if src_sr != target_sr:
        audio = _sig.resample_poly(audio, target_sr, src_sr).astype(np.float32)
    return audio, len(audio) / target_sr


def build_input_wav(t1_path: Path, padding_s: float, target_sr: int) -> tuple[np.ndarray, float, float]:
    """
    Concatenate T1 audio with silence padding.
    Returns (audio_array, t1_duration_s, total_duration_s).
    """
    t1_audio, t1_dur = load_wav_mono_resampled(t1_path, target_sr)
    silence = np.zeros(int(padding_s * target_sr), dtype=np.float32)
    combined = np.concatenate([t1_audio, silence])
    return combined, t1_dur, len(combined) / target_sr


# ── Sample collection ──────────────────────────────────────────────────────────

def collect_samples(l3_condition: str) -> list[dict]:
    """
    Collect all synthesized samples for a given L3 condition.
    Returns list of dicts with keys: source_id, scenario, topic, t1_path,
    t1_text, t2_text.
    """
    subcat = L3_TO_SUBCAT[l3_condition]
    samples: list[dict] = []

    for scenario in SCENARIOS:
        scen_dir = SYNTH_ROOT / subcat / scenario
        if not scen_dir.exists():
            print(f"  [WARN] not found: {scen_dir}")
            continue

        uuids = [d for d in sorted(scen_dir.iterdir())
                 if d.is_dir() and not d.name.startswith("_")]

        for uuid_dir in uuids:
            t1_wav   = uuid_dir / "turn_00_speaker_a.wav"
            tx_file  = uuid_dir / "transcript.json"
            if not t1_wav.exists() or not tx_file.exists():
                continue
            try:
                tx = json.loads(tx_file.read_text())
                dialogue = tx.get("dialogue", [])
                spk_a = next((d["text"] for d in dialogue if d["speaker"] == "speaker_a"), "")
                spk_b = next((d["text"] for d in dialogue if d["speaker"] == "speaker_b"), "")
                samples.append({
                    "source_id": uuid_dir.name,
                    "scenario":  scenario,
                    "topic":     tx.get("topic", ""),
                    "t1_path":   t1_wav,
                    "t1_text":   spk_a,
                    "t2_text":   spk_b,
                })
            except Exception as e:
                print(f"  [WARN] {uuid_dir.name}: {e}")

    print(f"[{l3_condition}] collected {len(samples)} samples "
          f"({len(SCENARIOS)} scenarios × {len(samples)//len(SCENARIOS)} each)")
    return samples


# ── Writing ────────────────────────────────────────────────────────────────────

def write_sample(
    sample_dir: Path,
    cand: dict,
    l3_condition: str,
    dry_run: bool,
) -> bool:
    if dry_run:
        print(f"  [dry] {sample_dir.name}  {cand['scenario']}  {cand['source_id'][:8]}")
        return True

    sample_dir.mkdir(parents=True, exist_ok=True)
    dest_wav = sample_dir / "input.wav"

    if not dest_wav.exists():
        combined, t1_dur, total_dur = build_input_wav(
            cand["t1_path"], PADDING_TIME_S, OUTPUT_SR
        )
        sf.write(str(dest_wav), combined, OUTPUT_SR)
    else:
        # Re-measure from existing file
        info = sf.info(str(dest_wav))
        total_dur = info.duration
        # Estimate t1_dur from source
        src_info = sf.info(str(cand["t1_path"]))
        t1_dur = src_info.duration

    padding_start = round(t1_dur, 4)
    padding_end   = round(total_dur, 4)

    meta = {
        "l3_condition":  l3_condition,
        "dataset":       "synthesized",
        "source_id":     cand["source_id"],
        "scenario":      cand["scenario"],
        "topic":         cand["topic"],

        "t1_start":      0.0,
        "t1_end":        padding_start,
        "t1_duration":   padding_start,
        "padding_start": padding_start,
        "padding_end":   padding_end,
        "padding_time":  PADDING_TIME_S,
        "input_duration": padding_end,

        "original_gap":  None,

        "t1_text":       cand["t1_text"],
        "t2_text":       cand["t2_text"],
    }
    (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return True


# ── Overlap: collection from real recordings ───────────────────────────────────

def _collect_ami_overlap(n: Optional[int]) -> list[dict]:
    """
    Load AMI overlap entries from valid.json.
    Audio filename: {meeting}_{spk_a}x{spk_b}_tt{N:04d}_t1.wav
    where N is the sequential index within each (meeting, spk_a, spk_b) group,
    sorted by t1_idx.
    """
    valid = json.loads((RESULTS_AMI / "valid.json").read_text())
    entries = [e for e in valid
               if e.get("category") == "take_turn_with_overlap" and e.get("gpt_valid")]

    # Build sequential index per (meeting, spk_a, spk_b) group
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for e in entries:
        key = (e["meeting"], e["spk_a"], e["spk_b"])
        groups[key].append(e)
    for key in groups:
        groups[key].sort(key=lambda x: x["t1_idx"])

    audio_dir = RESULTS_AMI / "take_turn_with_overlap"
    candidates: list[dict] = []
    for key, group in sorted(groups.items()):
        meeting, spk_a, spk_b = key
        for seq_i, e in enumerate(group):
            stem = f"{meeting}_{spk_a}x{spk_b}_tt{seq_i:04d}"
            t1_path = audio_dir / f"{stem}_t1.wav"
            if not t1_path.exists():
                print(f"  [WARN AMI] missing: {t1_path.name}")
                continue
            candidates.append({
                "dataset":    "AMI",
                "source_id":  f"{meeting}_{spk_a}x{spk_b}_t{e['t1_idx']}",
                "t1_path":    t1_path,
                "original_gap": e["gap"],
                "t1_text":    e["t1_text"],
                "t2_text":    e["t2_text"],
            })

    if n is not None:
        candidates = candidates[:n]
    print(f"  [AMI overlap] {len(candidates)} candidates")
    return candidates


def _collect_coraal_overlap(n: Optional[int]) -> list[dict]:
    """
    Load CORAAL overlap entries from valid.json.
    Audio filename: {file_id}_tt{t1_idx:04d}_t1.wav
    """
    valid = json.loads((RESULTS_CORAAL / "valid.json").read_text())
    entries = [e for e in valid
               if e.get("category") == "take_turn_with_overlap" and e.get("gpt_valid")]

    audio_dir = RESULTS_CORAAL / "take_turn_with_overlap"
    candidates: list[dict] = []
    for e in entries:
        stem = f"{e['file_id']}_tt{e['t1_idx']:04d}"
        t1_path = audio_dir / f"{stem}_t1.wav"
        if not t1_path.exists():
            print(f"  [WARN CORAAL] missing: {t1_path.name}")
            continue
        candidates.append({
            "dataset":    "CORAAL",
            "source_id":  f"{e['file_id']}_t{e['t1_idx']}",
            "t1_path":    t1_path,
            "original_gap": e["gap"],
            "t1_text":    e["t1_text"],
            "t2_text":    e["t2_text"],
        })

    if n is not None:
        candidates = candidates[:n]
    print(f"  [CORAAL overlap] {len(candidates)} candidates")
    return candidates


def _collect_candor_overlap(n: Optional[int]) -> list[dict]:
    """
    Load Candor overlap entries from valid.json.
    Audio filename: {conv_id}_t{t1_turn_id}_t1.wav
    """
    valid = json.loads((RESULTS_CANDOR / "valid.json").read_text())
    entries = [e for e in valid
               if e.get("category") == "take_turn_with_overlap" and e.get("gpt_valid")]

    audio_dir = RESULTS_CANDOR / "take_turn_with_overlap"
    candidates: list[dict] = []
    for e in entries:
        stem = f"{e['conv_id']}_t{e['t1_turn_id']}"
        t1_path = audio_dir / f"{stem}_t1.wav"
        if not t1_path.exists():
            print(f"  [WARN Candor] missing: {t1_path.name}")
            continue
        candidates.append({
            "dataset":    "Candor",
            "source_id":  stem,
            "t1_path":    t1_path,
            "original_gap": e["gap"],
            "t1_text":    e["t1_text"],
            "t2_text":    e["t2_text"],
        })

    if n is not None:
        candidates = candidates[:n]
    print(f"  [Candor overlap] {len(candidates)} candidates")
    return candidates


def collect_overlap_samples() -> list[dict]:
    """
    Collect overlap candidates: all AMI + all CORAAL + enough Candor to reach
    OVERLAP_TARGET total.
    """
    ami    = _collect_ami_overlap(OVERLAP_N_AMI)
    coraal = _collect_coraal_overlap(OVERLAP_N_CORAAL)
    n_candor = max(0, OVERLAP_TARGET - len(ami) - len(coraal))
    candor = _collect_candor_overlap(n_candor)
    total = len(ami) + len(coraal) + len(candor)
    print(f"  [overlap total] {total} samples "
          f"(AMI:{len(ami)} CORAAL:{len(coraal)} Candor:{len(candor)})")
    return ami + coraal + candor


def write_overlap_sample(
    sample_dir: Path,
    cand: dict,
    dry_run: bool,
) -> bool:
    if dry_run:
        print(f"  [dry] {sample_dir.name}  {cand['dataset']}  {cand['source_id']}")
        return True

    sample_dir.mkdir(parents=True, exist_ok=True)
    dest_wav = sample_dir / "input.wav"

    if not dest_wav.exists():
        combined, t1_dur, total_dur = build_input_wav(
            cand["t1_path"], PADDING_TIME_S, OUTPUT_SR
        )
        sf.write(str(dest_wav), combined, OUTPUT_SR)
    else:
        info    = sf.info(str(dest_wav))
        total_dur = info.duration
        src_info  = sf.info(str(cand["t1_path"]))
        t1_dur    = src_info.duration

    padding_start = round(t1_dur, 4)
    padding_end   = round(total_dur, 4)

    meta = {
        "l3_condition":  "slight_overlap_take_turn",
        "dataset":       cand["dataset"],
        "source_id":     cand["source_id"],

        "t1_start":      0.0,
        "t1_end":        padding_start,
        "t1_duration":   padding_start,
        "padding_start": padding_start,
        "padding_end":   padding_end,
        "padding_time":  PADDING_TIME_S,
        "input_duration": padding_end,

        "original_gap":  round(cand["original_gap"], 4),

        "t1_text":       cand["t1_text"],
        "t2_text":       cand["t2_text"],
    }
    (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare smooth turn-taking evaluation samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--conditions", nargs="+",
        choices=ALL_CONDITIONS + ["overlap", "synthesized"],
        default=ALL_CONDITIONS,
        help=(
            "Which L3 conditions to prepare. "
            "Use 'overlap' as shorthand for slight_overlap_take_turn, "
            "'synthesized' for immediate+delayed. Default: all three."
        ),
    )
    parser.add_argument(
        "--out_dir", type=Path, default=OUT_ROOT,
        help=f"Output directory (default: {OUT_ROOT}).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be written without creating files.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing output folders before writing.",
    )
    args = parser.parse_args()

    # Expand shorthand aliases
    expanded: list[str] = []
    for c in args.conditions:
        if c == "synthesized":
            expanded += ["immediate_take_turn", "delayed_take_turn"]
        elif c == "overlap":
            expanded.append("slight_overlap_take_turn")
        else:
            expanded.append(c)
    conditions = list(dict.fromkeys(expanded))  # deduplicate, preserve order

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for l3 in conditions:
        print(f"\n{'='*60}\n  {l3}\n{'='*60}")

        cond_dir = args.out_dir / L3_SUBDIR[l3]
        if not args.dry_run:
            cond_dir.mkdir(parents=True, exist_ok=True)

        if l3 == "slight_overlap_take_turn":
            # ── real-data overlap ──
            prefixes = ("AMI_tt_ovl_", "CORAAL_tt_ovl_", "Candor_tt_ovl_")
            if args.clean and not args.dry_run:
                removed = 0
                for p in prefixes:
                    for d in cond_dir.glob(f"{p}*"):
                        shutil.rmtree(d)
                        removed += 1
                if removed:
                    print(f"  Removed {removed} existing folders.")

            samples = collect_overlap_samples()
            counters: dict[str, int] = {}
            n_ok = 0
            for cand in samples:
                ds = cand["dataset"]
                idx = counters.get(ds, 0)
                counters[ds] = idx + 1
                sample_id = f"{ds}_tt_ovl_{idx:04d}"
                ok = write_overlap_sample(
                    cond_dir / sample_id, cand, dry_run=args.dry_run
                )
                if ok:
                    n_ok += 1

            total = sum(counters.values())
            print(f"[overlap] wrote {n_ok}/{total} samples  "
                  + "  ".join(f"{ds}:{c}" for ds, c in counters.items())
                  + f"  → {cond_dir}")

        else:
            # ── synthesized conditions ──
            short = L3_SHORT[l3]
            if args.clean and not args.dry_run:
                removed = 0
                for d in cond_dir.glob(f"synth_{short}_*"):
                    shutil.rmtree(d)
                    removed += 1
                if removed:
                    print(f"  Removed {removed} existing folders.")

            samples = collect_samples(l3)
            n_ok = 0
            for i, cand in enumerate(samples):
                sample_id = f"synth_{short}_{i:04d}"
                ok = write_sample(
                    cond_dir / sample_id, cand, l3, dry_run=args.dry_run
                )
                if ok:
                    n_ok += 1

            print(f"[{l3}] wrote {n_ok} samples → {cond_dir}")

    # Regenerate manifest.json for the whole turn_taking data root
    if not args.dry_run:
        _write_manifest(args.out_dir)


def _write_manifest(root: Path) -> None:
    entries = []
    for cond_dir in sorted(root.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name.startswith("."):
            continue
        for sample_dir in sorted(cond_dir.iterdir()):
            if not sample_dir.is_dir() or not (sample_dir / "metadata.json").exists():
                continue
            meta = json.loads((sample_dir / "metadata.json").read_text())
            entries.append({
                "sample_id":    sample_dir.name,
                "l3_condition": meta.get("l3_condition", cond_dir.name),
                "dataset":      meta.get("dataset", ""),
                "sample_dir":   str(sample_dir),
            })
    (root / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"\n[manifest] {len(entries)} samples → {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
