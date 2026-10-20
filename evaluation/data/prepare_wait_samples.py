"""
evaluation/prepare_wait_samples.py  (revised)
==============================================
Collect, normalize, and write 200 evaluation samples per Wait sub-action:

    inter_sentential  – between-sentence pause
    intra_sentential  – within-sentence pause
    floor_holding     – no pause; user verbally holds the floor

KEY DESIGN DECISIONS
────────────────────
1.  **Clip-relative timestamps**: all timing values (pause_start, pause_end,
    clip_start, clip_end) are stored relative to the start of the replayed
    audio clip (clip_start → 0).  The model will hear audio starting at t=0.

2.  **Transcripts**: every sample carries
        full_transcript   – full text of the spoken clip (with [PAUSE] marker
                            for pause conditions)
        pre_pause_text    – text spoken before the pause  (pause conditions)
        post_pause_text   – text spoken after the pause   (pause conditions)
    When a dataset has no transcript the field is omitted / null.

3.  **200 per sub-action** with priority ordering by transcript quality:
        inter_sentential  → AMI (13) → CORAAL (166) → Loqui (5) → Candor
        intra_sentential  → AMI (14) → CORAAL (186) → [done]
        floor_holding     → AMI (200)                [done]

OUTPUT metadata.json schema
────────────────────────────
{
  "task_type":        "inter_sentential" | "intra_sentential" | "floor_holding",
  "dataset":          "AMI" | "CORAAL" | "Loqui" | "Candor",
  "source_id":        str,
  "input_duration":   float,        # seconds; length of input.wav
  "clip_start":       0.0,          # always 0 in replayed context
  "clip_end":         float,        # == input_duration
  "pause_start":      float | null, # seconds from clip start (pause conds only)
  "pause_end":        float | null, # seconds from clip start (pause conds only)
  "full_transcript":  str | null,
  "pre_pause_text":   str | null,   # text before pause  (pause conds only)
  "post_pause_text":  str | null    # text after pause   (pause conds only)
}

Usage
─────
    python prepare_wait_samples.py \\
        --out_dir  evaluation/data/wait \\
        [--target   200]               # samples per sub-action (default 200)
        [--symlink]                    # symlink instead of copy audio
        [--dry_run]                    # print plan without writing files
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
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

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE    = _duplexact_root()
RESULTS = BASE / "results"

DATASET_DIRS: dict[str, Path] = {
    "AMI":    RESULTS / "AMI"    / "pauses_ami",
    "CORAAL": RESULTS / "CORAAL" / "pauses_coraal",
    "Loqui":  RESULTS / "Loqui"  / "pauses_loqui",
    "Candor": RESULTS / "Candor" / "pause",
}

TARGET_PER_DATASET: int = 50   # take up to this many from each dataset;
                               # if a dataset has fewer, take all it has

PADDING_TIME_S: float = 15.0   # silence appended after the clip for model response
OUTPUT_SR: int = 16_000        # target sample rate for input.wav

# ── Candidate record ───────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """One sample candidate before final selection."""
    dataset:         str
    source_id:       str
    audio_path:      Path

    # All times are CLIP-RELATIVE (seconds from start of audio clip)
    clip_start:      float = 0.0   # always 0
    clip_end:        Optional[float] = None  # filled from audio

    pause_start:     Optional[float] = None
    pause_end:       Optional[float] = None

    full_transcript: Optional[str] = None
    pre_pause_text:  Optional[str] = None
    post_pause_text: Optional[str] = None

    has_transcript:  bool = False
    has_timing:      bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def audio_duration(path: Path) -> Optional[float]:
    try:
        return sf.info(str(path)).duration
    except Exception:
        return None


def _split_at_pause(full: str) -> tuple[str, str]:
    """
    Split `full` at the first [PAUSE ...] marker.
    Returns (pre_text, post_text) stripped; both empty string on failure.
    """
    m = re.split(r"\[PAUSE[^\]]*\]", full, maxsplit=1)
    if len(m) == 2:
        return m[0].strip(), m[1].strip()
    return full.strip(), ""


def _normalize(t: float, origin: float) -> float:
    """Subtract the clip origin and round to 4 decimal places."""
    return round(t - origin, 4)


# ── Dataset collectors ─────────────────────────────────────────────────────────

def collect_ami(task: str) -> list[Candidate]:
    """
    Collect AMI candidates for the given task type.

    floor_holding  → from valid.json; audio in floor_holding/*.wav
    inter/intra    → per-sample *.json files in {task}/n1/ carry t_pre_text,
                     t_post_text, pauses, and the matching *_full.wav.
    """
    results_dir = DATASET_DIRS["AMI"]
    valid_json  = results_dir / "valid.json"
    if not valid_json.exists():
        print(f"[AMI] valid.json not found at {valid_json}", file=sys.stderr)
        return []

    all_cands = json.loads(valid_json.read_text())
    cands     = [c for c in all_cands if c.get("type") == task]
    out: list[Candidate] = []

    if task == "floor_holding":
        audio_dir = results_dir / "floor_holding"
        for cand in cands:
            seg_id = cand.get("seg_id", "")
            # audio filename convention: {meeting}_{speaker}_pNNNN_full.wav
            meeting  = cand.get("meeting", "")
            speaker  = cand.get("speaker", "")
            # find matching wav by seg_id via per-sample JSON lookup
            wav = _ami_floor_wav(results_dir, cand)
            if wav is None or not wav.exists():
                continue
            origin = cand.get("clip_start", 0.0)
            clip_dur = round((cand.get("clip_end", origin) - origin), 4)
            tx = cand.get("utterance") or cand.get("full_transcript")
            out.append(Candidate(
                dataset="AMI",
                source_id=seg_id or meeting,
                audio_path=wav,
                clip_start=0.0,
                clip_end=clip_dur or None,
                full_transcript=tx,
                has_transcript=bool(tx),
                has_timing=False,
            ))

    else:
        # inter/intra: per-sample JSONs carry all metadata
        audio_dir = results_dir / task / "n1"
        if not audio_dir.exists():
            print(f"[AMI] audio dir not found: {audio_dir}", file=sys.stderr)
            return []
        for jf in sorted(audio_dir.glob("*.json")):
            try:
                d    = json.loads(jf.read_text())
                wav  = jf.with_name(jf.stem + "_full.wav")
                if not wav.exists():
                    continue
                origin = d.get("clip_start", 0.0)
                pauses = d.get("pauses", [])
                ps = _normalize(pauses[0]["pause_start"], origin) if pauses else None
                pe = _normalize(pauses[0]["pause_end"],   origin) if pauses else None
                pre  = d.get("t_pre_text") or ""
                post = d.get("t_post_text") or ""
                full = d.get("full_transcript") or ""
                # Rebuild full_transcript with [PAUSE] marker if possible
                if pre and post and ps is not None:
                    full_norm = f"{pre} [PAUSE {pauses[0]['duration']:.2f}s] {post}"
                else:
                    full_norm = full
                out.append(Candidate(
                    dataset="AMI",
                    source_id=d.get("seg_id", jf.stem),
                    audio_path=wav,
                    clip_start=0.0,
                    pause_start=ps,
                    pause_end=pe,
                    full_transcript=full_norm or None,
                    pre_pause_text=pre or None,
                    post_pause_text=post or None,
                    has_transcript=bool(pre and post),
                    has_timing=ps is not None,
                ))
            except Exception as e:
                print(f"[AMI] skipping {jf.name}: {e}", file=sys.stderr)

    print(f"[AMI] {task}: {len(out)} candidates")
    return out


def _ami_floor_wav(results_dir: Path, cand: dict) -> Optional[Path]:
    """Return the _full.wav path for an AMI floor_holding candidate."""
    audio_dir = results_dir / "floor_holding"
    seg_id    = cand.get("seg_id", "")
    # Scan per-sample JSONs to find matching wav via seg_id
    if audio_dir.exists():
        for jf in audio_dir.glob("*.json"):
            try:
                d = json.loads(jf.read_text())
                if d.get("seg_id") == seg_id:
                    wav = jf.with_name(jf.stem + "_full.wav")
                    if wav.exists():
                        return wav
            except Exception:
                pass
    return None


def collect_coraal(task: str) -> list[Candidate]:
    """
    Collect CORAAL candidates.
    full_transcript carries [PAUSE Xs] marker; split to get pre/post text.
    """
    results_dir = DATASET_DIRS["CORAAL"]
    valid_json  = results_dir / "valid.json"
    if not valid_json.exists():
        print(f"[CORAAL] valid.json not found", file=sys.stderr)
        return []

    all_cands = json.loads(valid_json.read_text())
    cands     = [c for c in all_cands if c.get("type") == task]
    out: list[Candidate] = []

    for cand in cands:
        origin = cand.get("clip_start", 0.0)
        # Locate audio
        wav = _coraal_wav(results_dir, cand, task)
        if wav is None or not wav.exists():
            continue

        full = cand.get("full_transcript") or ""
        pre, post = _split_at_pause(full) if task != "floor_holding" else (None, None)

        pauses = cand.get("pauses", [])
        ps = _normalize(pauses[0]["pause_start"], origin) if pauses else None
        pe = _normalize(pauses[0]["pause_end"],   origin) if pauses else None

        out.append(Candidate(
            dataset="CORAAL",
            source_id=cand.get("file_id", ""),
            audio_path=wav,
            clip_start=0.0,
            pause_start=ps,
            pause_end=pe,
            full_transcript=full or None,
            pre_pause_text=pre or None,
            post_pause_text=post or None,
            has_transcript=bool(full),
            has_timing=ps is not None,
        ))

    print(f"[CORAAL] {task}: {len(out)} candidates")
    return out


def _coraal_wav(results_dir: Path, cand: dict, task: str) -> Optional[Path]:
    # Try audio_full first
    if "audio_full" in cand:
        p = Path(cand["audio_full"])
        if p.exists():
            return p
    # Reconstruct
    fid  = cand.get("file_id", "")
    cs   = cand.get("clip_start", 0.0)
    abbr = {"floor_holding": "floor", "inter_sentential": "inter",
            "intra_sentential": "intra"}.get(task, task)
    p = results_dir / task / "n1" / f"{fid}_{abbr}_{cs:.2f}_full.wav"
    return p if p.exists() else None


def collect_loqui(task: str) -> list[Candidate]:
    """
    Collect Loqui candidates.
    full_transcript carries [PAUSE] (no duration); no pause timing available.
    """
    results_dir = DATASET_DIRS["Loqui"]
    valid_json  = results_dir / "valid.json"
    if not valid_json.exists():
        print(f"[Loqui] valid.json not found", file=sys.stderr)
        return []

    all_cands = json.loads(valid_json.read_text())

    if task == "floor_holding":
        cands = [c for c in all_cands if c.get("type") == "floor_holding"]
    else:
        cands = [
            c for c in all_cands
            if c.get("type") == "pause"
            and (
                c.get("gpt_details", {}).get("pause_type") == task
                or c.get("pause_type") == task
            )
        ]

    out: list[Candidate] = []
    for cand in cands:
        wav = _loqui_wav(results_dir, cand, task)
        if wav is None or not wav.exists():
            continue

        full = cand.get("full_transcript") or cand.get("utterance") or ""
        pre, post = _split_at_pause(full) if task != "floor_holding" else (None, None)

        out.append(Candidate(
            dataset="Loqui",
            source_id=cand.get("file_id", ""),
            audio_path=wav,
            clip_start=0.0,
            pause_start=None,   # Loqui has no pause timing
            pause_end=None,
            full_transcript=full or None,
            pre_pause_text=pre or None,
            post_pause_text=post or None,
            has_transcript=bool(full),
            has_timing=False,
        ))

    print(f"[Loqui] {task}: {len(out)} candidates")
    return out


def _loqui_wav(results_dir: Path, cand: dict, task: str) -> Optional[Path]:
    if "audio_full" in cand:
        p = Path(cand["audio_full"])
        if p.exists():
            return p
        # Stale prefix: reconstruct under results_dir
        parts = p.parts
        for i, part in enumerate(parts):
            if part in ("inter_sentential", "intra_sentential", "floor_holding"):
                sub = Path(*parts[i:])
                c = results_dir / sub
                if c.exists():
                    return c
        matches = list(results_dir.glob(f"**/{p.name}"))
        if matches:
            return matches[0]
    return None


def collect_candor(task: str) -> list[Candidate]:
    """
    Collect Candor candidates from per-sample JSON files.
    inter/intra: files are in results/Candor/pause/{task}/n{k}/ subdirectories.
    floor_holding: files are flat in results/Candor/pause/floor_holding/.
    Each JSON has full transcripts, clip-relative pause timing, and pre/post text.
    """
    results_dir = DATASET_DIRS["Candor"]

    if task == "floor_holding":
        json_files = sorted((results_dir / "floor_holding").glob("*.json"))
    else:
        # Collect from all n1, n2, n3... subdirs
        type_dir   = results_dir / task
        json_files = sorted(jf for nd in sorted(type_dir.glob("n*")) for jf in nd.glob("*.json"))

    out: list[Candidate] = []
    for jf in json_files:
        try:
            d   = json.loads(jf.read_text())
            wav = jf.with_suffix(".wav")
            if not wav.exists():
                continue
            origin = d.get("clip_start", 0.0)
            pauses = d.get("pauses", [])
            ps = _normalize(pauses[0]["pause_start"], origin) if pauses else None
            pe = _normalize(pauses[0]["pause_end"],   origin) if pauses else None
            full = d.get("full_transcript") or ""
            pre  = d.get("pre_pause_text")  or (None if task == "floor_holding" else None)
            post = d.get("post_pause_text") or (None if task == "floor_holding" else None)
            out.append(Candidate(
                dataset="Candor",
                source_id=jf.stem,
                audio_path=wav,
                clip_start=0.0,
                pause_start=ps,
                pause_end=pe,
                full_transcript=full or None,
                pre_pause_text=pre or None,
                post_pause_text=post or None,
                has_transcript=bool(full),
                has_timing=(ps is not None),
            ))
        except Exception as e:
            print(f"[Candor] skipping {jf.name}: {e}", file=sys.stderr)

    print(f"[Candor] {task}: {len(out)} candidates")
    return out


# ── Selection ──────────────────────────────────────────────────────────────────

# Priority order per task: best transcript/timing quality first.
# Backfill to reach TARGET is taken from datasets in the same order.
TASK_COLLECTORS: dict[str, list] = {
    "inter_sentential": [collect_ami, collect_coraal, collect_loqui, collect_candor],
    "intra_sentential": [collect_ami, collect_coraal, collect_loqui, collect_candor],
    "floor_holding":    [collect_ami, collect_coraal, collect_loqui, collect_candor],
}

TARGET = 200


def select_candidates(task: str, per_dataset: int = TARGET_PER_DATASET,
                      target: int = TARGET) -> list[Candidate]:
    """
    Selection strategy:
      1. Take min(pool_size, per_dataset) from each dataset.
      2. If total < target, backfill by taking more from datasets in priority
         order (starting where we left off) until target is reached.
    """
    # Collect all pools up front
    pools: list[tuple[str, list[Candidate]]] = []
    for collector in TASK_COLLECTORS[task]:
        pool = collector(task)
        ds   = pool[0].dataset if pool else collector.__name__
        pools.append((ds, pool))

    # Pass 1: initial quota (up to per_dataset each)
    taken:    dict[str, int] = {}
    selected: list[Candidate] = []
    for ds, pool in pools:
        quota = min(len(pool), per_dataset)
        selected.extend(pool[:quota])
        taken[ds] = quota
        print(f"  → taking {quota}/{len(pool)} from {ds} (initial)")

    # Pass 2: backfill to reach target, split evenly across all datasets
    # that still have remaining samples, round-robin until target is met.
    while len(selected) < target:
        progress = False
        for ds, pool in pools:
            if len(selected) >= target:
                break
            already   = taken[ds]
            remaining = pool[already:]
            if not remaining:
                continue
            selected.append(remaining[0])
            taken[ds] += 1
            progress = True
        if not progress:
            break  # all pools exhausted

    # Print backfill summary
    for ds, pool in pools:
        extra = taken[ds] - min(len(pool), per_dataset)
        if extra > 0:
            print(f"  → backfill {extra} more from {ds} (total from {ds}: {taken[ds]})")

    selected = selected[:target]

    n_tx     = sum(1 for c in selected if c.has_transcript)
    n_timing = sum(1 for c in selected if c.has_timing)
    print(
        f"[{task}] total selected: {len(selected)}/{target}  "
        f"(with transcript: {n_tx}, with timing: {n_timing})"
    )
    return selected


# ── Writing ────────────────────────────────────────────────────────────────────

def _build_padded_wav(src: Path) -> tuple[np.ndarray, int, float]:
    """
    Load src WAV as mono float32 at OUTPUT_SR, append PADDING_TIME_S of silence.
    Returns (audio_array, sample_rate, clip_duration_s).
    """
    audio, src_sr = sf.read(str(src), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if src_sr != OUTPUT_SR:
        audio = _sig.resample_poly(audio, OUTPUT_SR, src_sr).astype(np.float32)
    clip_dur = len(audio) / OUTPUT_SR
    silence  = np.zeros(int(PADDING_TIME_S * OUTPUT_SR), dtype=np.float32)
    return np.concatenate([audio, silence]), OUTPUT_SR, clip_dur


def write_sample(
    sample_dir: Path,
    cand: Candidate,
    task: str,
    symlink: bool,
    dry_run: bool,
) -> bool:
    if dry_run:
        print(f"  [dry] {sample_dir.name}  {cand.audio_path.name}")
        return True

    sample_dir.mkdir(parents=True, exist_ok=True)
    dest = sample_dir / "input.wav"

    if not dest.exists():
        combined, sr, clip_dur = _build_padded_wav(cand.audio_path)
        sf.write(str(dest), combined, sr)
    else:
        info = sf.info(str(dest))
        total_dur = info.duration
        src_info  = sf.info(str(cand.audio_path))
        clip_dur  = src_info.duration

    if not dest.exists():
        print(f"  [WARN] failed to create {dest}", file=sys.stderr)
        return False

    clip_dur      = round(clip_dur, 4)
    padding_start = clip_dur
    padding_end   = round(clip_dur + PADDING_TIME_S, 4)

    meta: dict = {
        "task_type":      task,
        "dataset":        cand.dataset,
        "source_id":      cand.source_id,

        "clip_start":     0.0,
        "clip_end":       clip_dur,
        "clip_duration":  clip_dur,
        "padding_start":  padding_start,
        "padding_end":    padding_end,
        "padding_time":   PADDING_TIME_S,
        "input_duration": padding_end,
    }

    if task != "floor_holding":
        meta["pause_start"] = cand.pause_start
        meta["pause_end"]   = cand.pause_end

    meta["full_transcript"] = cand.full_transcript
    if task != "floor_holding":
        meta["pre_pause_text"]  = cand.pre_pause_text
        meta["post_pause_text"] = cand.post_pause_text

    (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare WAIT-action sample folders (200 per sub-action).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=BASE / "evaluation/data/wait",
        help="Root output directory (default: evaluation/data/wait).",
    )
    parser.add_argument(
        "--per_dataset", type=int, default=TARGET_PER_DATASET,
        help=f"Max samples taken from each dataset (default: {TARGET_PER_DATASET}). "
             "If a dataset has fewer, all its samples are used.",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        choices=list(TASK_COLLECTORS),
        default=list(TASK_COLLECTORS),
        help="Which sub-actions to prepare (default: all three).",
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="(Ignored — audio is always rebuilt with silence padding.)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be written without creating files.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing sample folders in out_dir/<task>/ before writing.",
    )
    args = parser.parse_args()

    for task in args.tasks:
        print(f"\n{'='*60}\n  {task}\n{'='*60}")
        out_dir = args.out_dir / task
        if not args.dry_run:
            if args.clean and out_dir.exists():
                shutil.rmtree(out_dir)
                print(f"[{task}] cleaned {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)

        candidates = select_candidates(task, args.per_dataset, target=TARGET)
        n_ok = 0
        for i, cand in enumerate(candidates):
            # sample ID: {DATASET_short}_{task_short}_{NNNN}
            ds_short   = cand.dataset[:6]
            task_short = task[:5]
            sample_id  = f"{ds_short}_{task_short}_{i:04d}"
            ok = write_sample(
                out_dir / sample_id, cand, task,
                symlink=args.symlink, dry_run=args.dry_run,
            )
            if ok:
                n_ok += 1

        print(f"[{task}] wrote {n_ok} samples → {out_dir}")

    # Regenerate manifest.json for the whole wait data root
    if not args.dry_run:
        _write_manifest(args.out_dir)


def _write_manifest(root: Path) -> None:
    entries = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for sample_dir in sorted(task_dir.iterdir()):
            if not sample_dir.is_dir() or not (sample_dir / "metadata.json").exists():
                continue
            meta = json.loads((sample_dir / "metadata.json").read_text())
            entries.append({
                "sample_id":    sample_dir.name,
                "l3_condition": meta.get("task_type", task_dir.name),
                "dataset":      meta.get("dataset", ""),
                "sample_dir":   str(sample_dir),
            })
    (root / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"\n[manifest] {len(entries)} samples → {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
