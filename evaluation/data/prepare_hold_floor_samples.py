#!/usr/bin/env python3
"""
evaluation/data/prepare_hold_floor_samples.py
===============================================================
Collect and write evaluation samples for the Hold-Floor action (FHR /
Floor-Holding Rate: model must continue its current turn through a
concurrent, non-takeover cue instead of yielding).

Interaction pattern: t0 (context, prompts the model to start a turn as "B")
-> model generates its own turn -> cue (bc/other, injected by a future
SIMULATOR driver at the live-appropriate moment, NOT a fixed offset) ->
model should keep talking through it. This script only packages assets;
the simulator (not yet implemented) decides WHEN to inject cue.wav.

L3 conditions (5 total; canonical names per evaluation/APPENDIX.md):
    continue_with_background_speech       synthesized, 200 avail
    continue_with_other_directed_speech    synthesized, 160 avail (40 have no
                                            `other` turn at all -- shortfall)
    continue_current_response              real, pooled across AMI/Candor/
                                            CORAAL/Loqui, 200 avail (AMI alone
                                            already covers the target)
    continue_but_shorten                   real, pooled across all 4 corpora,
                                            <=132 avail (AMI 40 + Candor 53 +
                                            CORAAL 38 + Loqui 1 -- still short
                                            of 200; this is now the FULL
                                            available real-data pool, not an
                                            AMI-only oversight)
    continue_with_simplification           real, pooled across all 4 corpora,
                                            211 avail (AMI 54 + Candor 42 +
                                            CORAAL 113 + Loqui 2 -- clears 200)

Real-data floor-holding pools (this project has FOUR, not just AMI -- an
earlier pass here only sourced AMI and left continue_but_shorten /
continue_with_simplification at 40/54 out of a 132/211-strong combined pool;
this version pools all four using each corpus's own on-disk naming
convention):
    AMI     results/AMI/Relaxed/floor_holding_relaxed/<label>/<bct>/
                <meeting>_<A>x<B>_fb<seq>_{t0,bc}.wav  (fb<seq> has no direct
                relation to valid.json fields -- resolved by nearest-duration
                match within the (meeting,spk_a,spk_b,label,bct) group, exact
                to 0.05s tolerance)
    CORAAL  results/CORAAL/Relaxed/floor_holding_relaxed/<label>/<bct>/
                <file_id>_bc<idx>_{t0,bc}.wav  (idx is off-by-one from
                valid.json's bc_idx -- resolved the same way, by duration)
    Loqui   results/Loqui/Relaxed/floor_holding_relaxed/<label>/<subtype>/<bct>/
                <file_id>_bc<idx>_{t0,bc}.wav  (idx matches valid.json's
                bc_idx directly, but duration-matched anyway for uniformity/
                safety)
    Candor  results/Candor/Relaxed/floor_holding_relaxed/<label>/<bct>/
                <conv_id>_t<t1_turn_id>_t0.wav   -- ALREADY tightly cropped
                    to [t0_start, t0_end] (direct match via conv_id+
                    t1_turn_id, no ambiguity/duration-matching needed)
                <conv_id>_t<t1_turn_id>_backchannel.wav -- NOT cropped; spans
                    the full [t1_start, t1_end] window with silence outside
                    [bc_start, bc_end] (same padded-window pattern as the
                    interruption pool used by stop/respond_and_recover --
                    must slice [bc_start-t1_start, bc_end-t1_start]
                    before use).

Across all four corpora, `t0.wav` is the lead-in (context prompting the
model to start speaking) and the true backchannel/cue segment is exposed
either as `bc.wav` (AMI/CORAAL/Loqui, already tightly cropped) or extracted
from `backchannel.wav` (Candor, needs cropping -- see _extract_candor_bc).

The two synthesized dir names are written EXACTLY as the canonical rubric
names (continue_with_background_speech / continue_with_other_directed_speech,
lowercase) -- evaluation/actions/hold_floor.py's CONDITION_ALIASES maps the
data_synthesize/ subcategory names (Continue_with_..., capitalized) to these.

OUTPUT layout (per sample):
    evaluation/data/hold_floor/<l3_condition>/<sample_id>/
        lead_in.wav   <- t0.wav / speaker_a.wav (prompts the model to speak)
        cue.wav       <- bc.wav (or cropped backchannel.wav) / other_processed.wav
        metadata.json
    evaluation/data/hold_floor/manifest.json

metadata.json fields:
    l3_condition, dataset ("AMI"|"Candor"|"CORAAL"|"Loqui"|"synthesized"),
    source_id, scenario (four conversation scenes; synthesized only),
    transcript_user (lead_in.wav text), reference_before_text,
    reference_after_text, cue_text, bc_type_original.

Usage:
    python prepare_hold_floor_samples.py [--target 200] [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
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

import soundfile as sf

BASE = _duplexact_root()
RESULTS_ROOT = BASE / "results"
SYNTH_ROOT = BASE / "data_synthesize" / "hold_floor"
OUT_ROOT = BASE / "evaluation" / "data" / "hold_floor"

TARGET = 200
RANDOM_SEED = 42

# canonical L3 -> per-dataset floor_holding_label value (identical across
# all four corpora's valid.json)
REAL_LABEL_TO_L3 = {
    "continue_current": "continue_current_response",
    "continue_shorten": "continue_but_shorten",
    "continue_simplify": "continue_with_simplification",
}
SYNTH_CONDITIONS = [
    "continue_with_background_speech",
    "continue_with_other_directed_speech",
]
SYNTH_DIR_NAME = {
    "continue_with_background_speech":     "Continue_with_background_speech",
    "continue_with_other_directed_speech": "Continue_with_other_directed_speech",
}

# Dataset priority order per corpus's own pool size for the two shortfall
# conditions (largest first) -- AMI first for continue_current_response only
# since it alone already meets target there.
DATASET_ORDER = ["AMI", "Candor", "CORAAL", "Loqui"]


def _wav_duration(path: Path) -> float:
    return sf.info(str(path)).duration


# ─── AMI ──────────────────────────────────────────────────────────────────────

_AMI_PAT = re.compile(r"^(.+?)_(\w)x(\w)_fb(\d+)_t0\.wav$")


def _ami_disk_index(label: str) -> dict[tuple, list[int]]:
    idx: dict[tuple, list[int]] = defaultdict(list)
    label_dir = RESULTS_ROOT / "AMI" / "Relaxed" / "floor_holding_relaxed" / label
    if not label_dir.exists():
        return idx
    for bct_dir in label_dir.iterdir():
        if not bct_dir.is_dir():
            continue
        for f in bct_dir.glob("*_t0.wav"):
            m = _AMI_PAT.match(f.name)
            if not m:
                continue
            meeting, sa, sb, seq = m.group(1), m.group(2), m.group(3), int(m.group(4))
            idx[(meeting, sa, sb, bct_dir.name)].append(seq)
    return idx


def _resolve_ami(entry: dict, label: str, disk_index: dict, used: dict) -> Optional[tuple[Path, Path]]:
    bct = entry.get("backchannel_type") or "none"
    key = (entry["meeting"], entry["spk_a"], entry["spk_b"], bct)
    seqs = disk_index.get(key, [])
    target_dur = entry["t0_end"] - entry["t0_start"]
    used_seqs = used.setdefault(key, set())

    best_seq, best_diff = None, 1e9
    base_dir = RESULTS_ROOT / "AMI" / "Relaxed" / "floor_holding_relaxed" / label / bct
    for seq in seqs:
        if seq in used_seqs:
            continue
        t0_path = base_dir / f"{key[0]}_{key[1]}x{key[2]}_fb{seq:04d}_t0.wav"
        if not t0_path.exists():
            continue
        diff = abs(_wav_duration(t0_path) - target_dur)
        if diff < best_diff:
            best_diff, best_seq = diff, seq
    if best_seq is None or best_diff > 0.05:
        return None
    used_seqs.add(best_seq)

    base = base_dir / f"{key[0]}_{key[1]}x{key[2]}_fb{best_seq:04d}"
    t0_wav, bc_wav = Path(f"{base}_t0.wav"), Path(f"{base}_bc.wav")
    if t0_wav.exists() and bc_wav.exists():
        return t0_wav, bc_wav
    return None


# ─── CORAAL / Loqui (same bc<idx> naming, different nesting depth) ──────────

def _bcidx_disk_index(label_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """key = file_id -> [(bc_idx, dir_containing_the_files), ...] found anywhere
    under label_dir (handles CORAAL's <bct>/ and Loqui's <subtype>/<bct>/ nesting)."""
    idx: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    if not label_dir.exists():
        return idx
    for f in label_dir.rglob("*_t0.wav"):
        m = re.match(r"^(.+)_bc(\d+)_t0\.wav$", f.name)
        if not m:
            continue
        file_id, idx_num = m.group(1), int(m.group(2))
        idx[file_id].append((idx_num, f.parent))
    return idx


def _resolve_bcidx(entry: dict, disk_index: dict, used: dict, file_id_key: str = "file_id") -> Optional[tuple[Path, Path]]:
    file_id = entry[file_id_key]
    candidates = disk_index.get(file_id, [])
    target_dur = entry["t0_end"] - entry["t0_start"]
    used_set = used.setdefault(file_id, set())

    best, best_diff = None, 1e9
    for idx_num, parent in candidates:
        if idx_num in used_set:
            continue
        t0_path = parent / f"{file_id}_bc{idx_num:04d}_t0.wav"
        if not t0_path.exists():
            continue
        diff = abs(_wav_duration(t0_path) - target_dur)
        if diff < best_diff:
            best_diff, best = diff, (idx_num, parent)
    if best is None or best_diff > 0.05:
        return None
    idx_num, parent = best
    used_set.add(idx_num)

    base = parent / f"{file_id}_bc{idx_num:04d}"
    t0_wav, bc_wav = Path(f"{base}_t0.wav"), Path(f"{base}_bc.wav")
    if t0_wav.exists() and bc_wav.exists():
        return t0_wav, bc_wav
    return None


# ─── Candor (direct conv_id+t1_turn_id match; backchannel.wav needs cropping) ─

def _extract_candor_bc(backchannel_wav: Path, t1_start: float, bc_start: float, bc_end: float, dst: Path) -> None:
    audio, sr = sf.read(str(backchannel_wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    s = max(0, int(round((bc_start - t1_start) * sr)))
    e = min(len(audio), int(round((bc_end - t1_start) * sr)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio[s:e], sr, subtype="PCM_16")


def _resolve_candor(entry: dict, label: str) -> Optional[tuple[Path, Path, float]]:
    """Returns (t0_wav, backchannel_wav_full_window, t1_start) -- caller must
    crop backchannel_wav via _extract_candor_bc."""
    bct = entry.get("backchannel_type") or "none"
    base = (RESULTS_ROOT / "Candor" / "Relaxed" / "floor_holding_relaxed" / label / bct
            / f"{entry['conv_id']}_t{entry['t1_turn_id']}")
    t0_wav = Path(f"{base}_t0.wav")
    bc_full_wav = Path(f"{base}_backchannel.wav")
    if t0_wav.exists() and bc_full_wav.exists():
        return t0_wav, bc_full_wav, entry["t1_start"]
    return None


# ─── Per-dataset pool loaders ────────────────────────────────────────────────

def _load_pool(dataset: str, source_label: str) -> list[dict]:
    valid_json = RESULTS_ROOT / dataset / "Relaxed" / "floor_holding_relaxed" / "valid.json"
    if not valid_json.exists():
        return []
    all_entries = json.loads(valid_json.read_text())
    pool = [e for e in all_entries if e.get("floor_holding_label") == source_label]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pool)
    return pool


def _try_resolve(dataset: str, entry: dict, label: str, state: dict, dry_run: bool,
                  out_sd: Optional[Path]) -> Optional[dict]:
    """Resolve + (if not dry_run) write lead_in.wav/cue.wav for one entry.
    Returns a partial metadata dict (without sample_id/l3_condition) or None."""
    if dataset == "AMI":
        idx = state.setdefault("ami_idx", {}).setdefault(label, _ami_disk_index(label))
        used = state.setdefault("ami_used", {}).setdefault(label, {})
        pair = _resolve_ami(entry, label, idx, used)
        if pair is None:
            return None
        t0_wav, bc_wav = pair
        if not dry_run and out_sd is not None:
            out_sd.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(t0_wav, out_sd / "lead_in.wav")
            shutil.copyfile(bc_wav, out_sd / "cue.wav")
        return {
            "source_id": f"{entry['meeting']}_{entry['spk_a']}x{entry['spk_b']}_bc{entry['bc_start']:.3f}",
        }

    if dataset in ("CORAAL", "Loqui"):
        cache_key = f"{dataset}_idx_{label}"
        if cache_key not in state:
            label_dir = RESULTS_ROOT / dataset / "Relaxed" / "floor_holding_relaxed" / label
            state[cache_key] = _bcidx_disk_index(label_dir)
        idx = state[cache_key]
        used = state.setdefault(f"{dataset}_used_{label}", {})
        pair = _resolve_bcidx(entry, idx, used)
        if pair is None:
            return None
        t0_wav, bc_wav = pair
        if not dry_run and out_sd is not None:
            out_sd.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(t0_wav, out_sd / "lead_in.wav")
            shutil.copyfile(bc_wav, out_sd / "cue.wav")
        return {"source_id": f"{entry['file_id']}_bc{entry['bc_start']:.3f}"}

    if dataset == "Candor":
        resolved = _resolve_candor(entry, label)
        if resolved is None:
            return None
        t0_wav, bc_full_wav, t1_start = resolved
        if not dry_run and out_sd is not None:
            out_sd.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(t0_wav, out_sd / "lead_in.wav")
            _extract_candor_bc(bc_full_wav, t1_start, entry["bc_start"], entry["bc_end"], out_sd / "cue.wav")
        return {"source_id": f"{entry['conv_id']}_t{entry['t1_turn_id']}"}

    raise ValueError(f"unknown dataset {dataset}")


def prepare_real(l3_condition: str, source_label: str, target: int, dry_run: bool) -> list[dict]:
    pools = {ds: _load_pool(ds, source_label) for ds in DATASET_ORDER}
    for ds in DATASET_ORDER:
        print(f"[{l3_condition}] {ds} pool: {len(pools[ds])} candidates")

    # Pool ALL datasets together and shuffle before drawing, instead of
    # filling greedily in a fixed DATASET_ORDER priority (AMI first, then
    # Candor, CORAAL, Loqui last). The old fixed-order approach silently
    # starved whichever dataset came later whenever an earlier dataset's
    # pool alone already met `target`: continue_current_response's AMI/
    # Candor/CORAAL pools are each capped at 500 (way over target=200), so
    # AMI alone filled every slot and Candor/CORAAL/Loqui (including all 29
    # of Loqui's valid candidates) got ZERO representation; similarly
    # continue_with_simplification's 211-candidate pool (AMI 54 + Candor 42
    # + CORAAL 113 + Loqui 2) hit target after only 104/113 of CORAAL, so
    # Loqui's 2 candidates were never even tried. Combining + shuffling
    # (same fix already applied to stop/respond_and_recover via
    # interruption_pool_common.py) gives every dataset a fair, random shot
    # up to `target`, proportional to its own pool size -- see chat
    # discussion 2026-09-02.
    combined: list[tuple[str, dict]] = [(ds, e) for ds in DATASET_ORDER for e in pools[ds]]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(combined)

    out_dir = OUT_ROOT / l3_condition
    records: list[dict] = []
    state: dict = {}
    n_ok = 0

    for ds, entry in combined:
        if n_ok >= target:
            break
        sample_id = f"{ds}_{l3_condition}_{n_ok:04d}"
        out_sd = (out_dir / sample_id) if not dry_run else None
        resolved = _try_resolve(ds, entry, source_label, state, dry_run, out_sd)
        if resolved is None:
            continue

        record = {
            "sample_id": sample_id,
            "l3_condition": l3_condition,
            "dataset": ds,
            "source_id": resolved["source_id"],
            "bc_type_original": entry.get("backchannel_type"),
            "transcript_user": entry.get("t0_text", ""),
            "reference_before_text": entry.get("before_text", ""),
            "reference_after_text": entry.get("after_text", ""),
            "cue_text": entry.get("bc_text", ""),
            "gpt_reason": entry.get("gpt_reason", ""),
        }
        if not dry_run:
            (out_dir / sample_id / "metadata.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False))
        records.append(record)
        n_ok += 1

    print(f"[{l3_condition}] prepared {n_ok}/{target}  "
          f"(by dataset: {dict((ds, sum(1 for r in records if r['dataset'] == ds)) for ds in DATASET_ORDER)})")
    return records


def _find_synth_samples(subcat_dir: Path) -> list[Path]:
    return sorted(p.parent for p in subcat_dir.rglob("tts_metadata.json"))


def prepare_synth(l3_condition: str, target: int, dry_run: bool) -> list[dict]:
    subcat_dir = SYNTH_ROOT / SYNTH_DIR_NAME[l3_condition]
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
        by_file = {f["file"]: f for f in meta.get("files", [])}
        if "speaker_a.wav" not in by_file or "speaker_b.wav" not in by_file:
            continue
        cue_label = "other_processed.wav" if "other_processed.wav" in by_file else (
            "other_raw.wav" if "other_raw.wav" in by_file else None
        )
        if cue_label is None:
            # 40 background_speech samples have no `other` turn at all -- skip.
            continue
        if not (sd / "speaker_a.wav").exists() or not (sd / cue_label).exists():
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
            "bc_type_original": None,
            "transcript_user": by_file["speaker_a.wav"]["tts_text"],
            "reference_before_text": None,
            "reference_after_text": None,
            "cue_text": by_file[cue_label]["tts_text"],
            "reference_response_text": by_file["speaker_b.wav"]["tts_text"],
        }
        if not dry_run:
            out_sd = out_dir / sample_id
            out_sd.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sd / "speaker_a.wav", out_sd / "lead_in.wav")
            shutil.copyfile(sd / cue_label, out_sd / "cue.wav")
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
    for real_label, l3 in REAL_LABEL_TO_L3.items():
        manifest.extend(prepare_real(l3, real_label, args.target, args.dry_run))
    for l3 in SYNTH_CONDITIONS:
        manifest.extend(prepare_synth(l3, args.target, args.dry_run))

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
