#!/usr/bin/env python3
"""Restore DuplexAct parquet (Hub layout) to evaluation/data/.

    python scripts/unpack_hf_dataset.py --local_dir /path/to/hf_export \
      --out_dir evaluation/data
    python scripts/unpack_hf_dataset.py --repo_id LincolnD/DuplexAct \
      --out_dir evaluation/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ACTIONS = [
    "wait", "backchannel", "ignore", "turn_taking",
    "stop", "hold_floor", "respond_and_recover", "interrupt_user",
]
TURN_DIR = {
    "immediate_take_turn": "take_turn_immediately",
    "delayed_take_turn": "take_turn_with_delay",
    "slight_overlap_take_turn": "take_turn_with_overlap",
}
AUDIO_COLS = [
    "input", "lead_in", "cue",
    "user_speech", "branch_acknowledged", "branch_ignored",
]


def _struct_to_dict(val):
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    try:
        return dict(val)
    except Exception:
        return None


def unpack_table(table, action: str, out_dir: Path) -> int:
    rows = table.to_pylist()
    manifest = []
    n = 0
    for row in rows:
        sid = row["sample_id"]
        cond = row["condition"]
        disk = TURN_DIR.get(cond, cond)
        dest = out_dir / action / disk / sid
        dest.mkdir(parents=True, exist_ok=True)
        for col in AUDIO_COLS:
            audio = _struct_to_dict(row.get(col))
            if not audio or not audio.get("bytes"):
                continue
            name = audio.get("path") or f"{col}.wav"
            (dest / Path(name).name).write_bytes(audio["bytes"])
        meta_raw = row.get("metadata") or "{}"
        (dest / "metadata.json").write_text(
            json.dumps(json.loads(meta_raw), indent=2, ensure_ascii=False)
        )
        entry = json.loads(row["manifest"]) if row.get("manifest") else {
            "sample_id": sid,
            "l3_condition": cond,
            "dataset": row.get("dataset") or "",
        }
        entry["sample_dir"] = f"{action}/{disk}/{sid}"
        manifest.append(entry)
        n += 1
    man_path = out_dir / action / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[{action}] unpacked {n} -> {man_path}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", type=Path, default=None)
    ap.add_argument("--repo_id", default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    if not args.local_dir and not args.repo_id:
        raise SystemExit("pass --local_dir or --repo_id")

    total = 0
    if args.local_dir:
        for action in ACTIONS:
            files = sorted((args.local_dir / action).glob("test-*.parquet"))
            if not files:
                print(f"[{action}] no parquet")
                continue
            tables = [pq.read_table(str(p)) for p in files]
            table = tables[0]
            for t in tables[1:]:
                table = pa.concat_tables([table, t])
            total += unpack_table(table, action, args.out_dir)
    else:
        from huggingface_hub import snapshot_download
        cache = snapshot_download(
            args.repo_id, repo_type="dataset", token=args.token
        )
        for action in ACTIONS:
            files = sorted(Path(cache).joinpath(action).glob("test-*.parquet"))
            if not files:
                print(f"[{action}] no parquet")
                continue
            tables = [pq.read_table(str(p)) for p in files]
            table = tables[0]
            for t in tables[1:]:
                table = pa.concat_tables([table, t])
            total += unpack_table(table, action, args.out_dir)
    print(f"total unpacked: {total}")


if __name__ == "__main__":
    main()
