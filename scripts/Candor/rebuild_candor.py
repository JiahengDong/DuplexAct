#!/usr/bin/env python3
"""Rebuild the 1,235 Candor test instances from an official Candor download.

Candor audio cannot be redistributed, so the instances are shipped as a fixed
recipe (`candor_instances.jsonl`): for every wav, the conversation, the stereo
channel, the exact sample range, the samples that were silenced, and the
trailing silence. This script cuts those ranges out of your local copy and
writes them, with the original `metadata.json`, into the evaluation layout,
then regenerates every `<action>/manifest.json` from the folders on disk.

    python scripts/unpack_hf_dataset.py --repo_id LincolnD/DuplexAct --out_dir evaluation/data
    python scripts/Candor/rebuild_candor.py --candor_root /path/to/Candor --out_dir evaluation/data

Audio is decoded exactly as when the benchmark was built:
`librosa.load(<conversation_id>.mp3, sr=16000, mono=False)`, then written as
16 kHz PCM_16. Each wav is checked against the SHA-256 of its samples; any
mismatch is reported at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_manifests import write_manifests  # noqa: E402

SR = 16000
RECIPE = Path(__file__).resolve().parent / "candor_instances.jsonl"
CHANNELS = {"L": 0, "R": 1}


def load_recipe(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def index_conversations(root: Path) -> dict[str, Path]:
    """Map conversation id -> conversation folder (any depth up to two levels)."""
    found = {}
    for base in [root, *[p for p in root.iterdir() if p.is_dir()]]:
        for d in base.iterdir():
            if (d / "processed" / f"{d.name}.mp3").is_file():
                found[d.name] = d
    return found


def pcm_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<i2").tobytes()).hexdigest()


def to_pcm16(x: np.ndarray) -> np.ndarray:
    """Float -> int16 through libsndfile, the same rounding the benchmark build used."""
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return sf.read(buf, dtype="int16")[0]


def cut(audio: np.ndarray, spec: dict) -> np.ndarray:
    if spec["channel"] is None:
        return np.zeros(spec["pad"], dtype=np.int16)
    seg = audio[CHANNELS[spec["channel"]], spec["start"]:spec["start"] + spec["length"]]
    if len(seg) != spec["length"]:
        raise ValueError(f"conversation audio too short for range {spec['start']}+{spec['length']}")
    pcm = to_pcm16(seg)
    for a, b in spec.get("zero", []):
        pcm[a:b] = 0
    return np.concatenate([pcm, np.zeros(spec.get("pad", 0), dtype=np.int16)])


def build_conversation(conv_dir: Path, items: list[dict], out_dir: Path) -> list[tuple[str, str, bool]]:
    audio, _ = librosa.load(str(conv_dir / "processed" / f"{conv_dir.name}.mp3"), sr=SR, mono=False)
    results = []
    for it in items:
        dst = out_dir / it["sample_dir"]
        dst.mkdir(parents=True, exist_ok=True)
        for name, spec in it["files"].items():
            pcm = cut(audio, spec)
            sf.write(str(dst / name), pcm, SR, subtype="PCM_16")
            results.append((it["sample_dir"], name, pcm_sha256(pcm) == spec["sha256"]))
        (dst / "metadata.json").write_text(json.dumps(it["metadata"], indent=2, ensure_ascii=False))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candor_root", type=Path, required=True,
                    help="Official Candor download (folders <conversation_id>/processed/*.mp3, "
                         "optionally grouped under part_*/).")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--recipe", type=Path, default=RECIPE)
    ap.add_argument("--action", default=None, help="Rebuild only this action.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    recipe = load_recipe(args.recipe)
    if args.action:
        recipe = [it for it in recipe if it["action"] == args.action]
    convs = index_conversations(args.candor_root)
    by_conv = defaultdict(list)
    for it in recipe:
        by_conv[it["conversation_id"]].append(it)
    missing = sorted(c for c in by_conv if c not in convs)
    if missing:
        raise SystemExit(f"{len(missing)} conversations not found under {args.candor_root}, e.g. {missing[:3]}")
    print(f"{len(recipe)} instances from {len(by_conv)} conversations")

    mismatched, done = [], 0
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(build_conversation, convs[c], items, args.out_dir) for c, items in by_conv.items()]
        for fut in as_completed(futs):
            for sample_dir, name, ok in fut.result():
                if not ok:
                    mismatched.append(f"{sample_dir}/{name}")
            done += 1
            if done % 50 == 0 or done == len(futs):
                print(f"  conversations {done}/{len(futs)}", flush=True)

    for action, n in write_manifests(args.out_dir).items():
        print(f"  manifest {action}: {n}")
    n_files = sum(len(it["files"]) for it in recipe)
    if mismatched:
        print(f"\n{len(mismatched)}/{n_files} wavs differ from the reference checksum "
              f"(usually a different mp3 decoder or librosa/soxr version):")
        for m in mismatched[:20]:
            print("  ", m)
    else:
        print(f"\nAll {n_files} wavs match the reference checksums.")


if __name__ == "__main__":
    main()
