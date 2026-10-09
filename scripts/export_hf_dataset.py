#!/usr/bin/env python3
"""Build and push the DuplexAct evaluation set to Hugging Face.

Candor clips are excluded (license). AMI / CORAAL / Loqui / synthesized stay.
Each action is a dataset config with split ``test`` and Audio columns.

    python scripts/export_hf_dataset.py
    python scripts/export_hf_dataset.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Audio, Dataset, Features, Value
from huggingface_hub import HfApi

REPO_ID = "LincolnD/DuplexAct"

ROOT = Path(os.environ.get("DUPLEXACT_DATA_ROOT")
            or "/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/data")

DR = {"wait", "backchannel", "ignore", "turn_taking"}
SIM_CUE = {"stop", "hold_floor", "respond_and_recover"}
ACTIONS = [
    "wait", "backchannel", "ignore", "turn_taking",
    "stop", "hold_floor", "respond_and_recover", "interrupt_user",
]
TURN_DIR = {
    "immediate_take_turn": "take_turn_immediately",
    "delayed_take_turn": "take_turn_with_delay",
    "slight_overlap_take_turn": "take_turn_with_overlap",
}


def resolve_dir(action: str, entry: dict) -> Path | None:
    sid = entry["sample_id"]
    cond = entry["l3_condition"]
    candidates = [
        ROOT / action / cond / sid,
        ROOT / action / TURN_DIR.get(cond, cond) / sid,
    ]
    sd = entry.get("sample_dir")
    if sd:
        p = Path(sd)
        candidates.append(p if p.is_absolute() else ROOT / p)
        # older layout: evaluation/data/<action>/...
        if not p.is_absolute():
            candidates.append(ROOT.parent.parent / "data" / p)
    for c in candidates:
        if c.is_dir():
            return c
    return None


def protocol(action: str) -> str:
    if action in DR:
        return "direct_replay"
    if action == "interrupt_user":
        return "simulator_barge_in"
    return "simulator_cue"


def audio_or_none(path: Path) -> dict | None:
    """Embed WAV bytes so parquet is self-contained (no torchcodec needed)."""
    if not path.is_file():
        return None
    return {"bytes": path.read_bytes(), "path": path.name}


def collect(action: str) -> list[dict]:
    entries = json.loads((ROOT / action / "manifest.json").read_text())
    rows = []
    skipped_candor = missing = 0
    for e in entries:
        if (e.get("dataset") or "").lower() == "candor":
            skipped_candor += 1
            continue
        d = resolve_dir(action, e)
        if d is None:
            missing += 1
            continue
        meta = {}
        mp = d / "metadata.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        row = {
            "sample_id": e["sample_id"],
            "action": action,
            "condition": e["l3_condition"],
            "dataset": e.get("dataset") or meta.get("dataset") or "",
            "protocol": protocol(action),
            "source_id": str(e.get("source_id") or meta.get("source_id") or ""),
            "manifest": json.dumps(e, ensure_ascii=False),
            "metadata": json.dumps(meta, ensure_ascii=False),
            "input": None,
            "lead_in": None,
            "cue": None,
            "user_speech": None,
            "branch_acknowledged": None,
            "branch_ignored": None,
        }
        if action in DR:
            row["input"] = audio_or_none(d / "input.wav")
        elif action in SIM_CUE:
            row["lead_in"] = audio_or_none(d / "lead_in.wav")
            row["cue"] = audio_or_none(d / "cue.wav")
        else:
            row["user_speech"] = audio_or_none(d / "user_speech.wav")
            row["branch_acknowledged"] = audio_or_none(d / "branch_acknowledged.wav")
            row["branch_ignored"] = audio_or_none(d / "branch_ignored.wav")
        rows.append(row)
    print(f"[{action}] keep={len(rows)}  candor_excluded={skipped_candor}  missing_dir={missing}")
    return rows


# datasets>=4 encode_example() imports torchcodec even for path/bytes.
# Build Arrow with the Audio struct, then tag columns as Audio for the Hub.
AUDIO_STRUCT = {
    "bytes": Value("binary"),
    "path": Value("string"),
}
AUDIO = Audio(sampling_rate=16_000)
META = {
    "sample_id": Value("string"),
    "action": Value("string"),
    "condition": Value("string"),
    "dataset": Value("string"),
    "protocol": Value("string"),
    "source_id": Value("string"),
    "manifest": Value("string"),
    "metadata": Value("string"),
}
AUDIO_COLS = [
    "input", "lead_in", "cue",
    "user_speech", "branch_acknowledged", "branch_ignored",
]
BUILD_FEATURES = Features({**META, **{c: AUDIO_STRUCT for c in AUDIO_COLS}})
FEATURES = Features({**META, **{c: AUDIO for c in AUDIO_COLS}})


CARD = """---
language:
- en
pretty_name: DuplexAct
task_categories:
- audio-classification
- reinforcement-learning
tags:
- spoken-dialogue
- full-duplex
- turn-taking
license: other
size_categories:
- 1K<n<10K
configs:
- config_name: wait
  data_files:
  - split: test
    path: wait/test-*.parquet
- config_name: backchannel
  data_files:
  - split: test
    path: backchannel/test-*.parquet
- config_name: ignore
  data_files:
  - split: test
    path: ignore/test-*.parquet
- config_name: turn_taking
  data_files:
  - split: test
    path: turn_taking/test-*.parquet
- config_name: stop
  data_files:
  - split: test
    path: stop/test-*.parquet
- config_name: hold_floor
  data_files:
  - split: test
    path: hold_floor/test-*.parquet
- config_name: respond_and_recover
  data_files:
  - split: test
    path: respond_and_recover/test-*.parquet
- config_name: interrupt_user
  data_files:
  - split: test
    path: interrupt_user/test-*.parquet
---

# DuplexAct

Evaluation instances for [DuplexAct](https://github.com/JiahengDong/DuplexAct):
interaction-action taking in end-to-end full-duplex spoken language models.

**Candor-derived clips are not included.** Download Candor yourself and run
`scripts/Candor/` in the GitHub repo if you need those 1,152 items.

This Hub dump is AMI, CORAAL, Loqui, and the synthesized conditions (3,539 items).

## Load

```python
from datasets import load_dataset

ds = load_dataset("LincolnD/DuplexAct", "wait", split="test")
# Audio columns are {path, bytes} (self-contained parquet).
# ds[0]["metadata"] is the original metadata.json as a JSON string.
# ds[0]["manifest"] is the original manifest.json entry.
```

To restore the on-disk layout used by inference:

```bash
python scripts/unpack_hf_dataset.py --repo_id LincolnD/DuplexAct \
  --out_dir evaluation/direct_replay/data
```

Configs match Level-2 actions. Audio columns that do not apply to that
protocol are empty:

| Config | Protocol | Audio columns |
|---|---|---|
| `wait`, `backchannel`, `ignore`, `turn_taking` | Direct replay | `input` |
| `stop`, `hold_floor`, `respond_and_recover` | Simulator cue | `lead_in`, `cue` |
| `interrupt_user` | Simulator barge-in | `user_speech`, `branch_acknowledged`, `branch_ignored` |

`condition` is the Level-3 id. `dataset` is `AMI` / `CORAAL` / `Loqui` / `synthesized`.
"""


def _write_shards(ds: Dataset, dest: Path, stem: str = "test") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    max_bytes = 400 * 1024 * 1024
    n_shards = max(1, (ds.data.nbytes + max_bytes - 1) // max_bytes)
    for i in range(n_shards):
        shard = ds.shard(num_shards=n_shards, index=i, contiguous=True)
        name = f"{stem}-{i:05d}-of-{n_shards:05d}.parquet"
        shard.to_parquet(str(dest / name))
        print(f"    wrote {dest / name}  rows={len(shard)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--repo_id", default=REPO_ID)
    ap.add_argument("--local_dir", type=Path, default=None,
                    help="Write Hub-layout parquet here (and upload if not --local_only)")
    ap.add_argument("--local_only", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--actions", nargs="*", default=None,
                    help="Subset of actions to export (default: all)")
    args = ap.parse_args()
    actions = args.actions or ACTIONS
    if any(a not in ACTIONS for a in actions):
        raise SystemExit(f"unknown action in {actions}")

    all_n = 0
    api = None if args.dry_run or args.local_only else HfApi(token=args.token)
    for action in actions:
        rows = collect(action)
        all_n += len(rows)
        if args.dry_run or not rows:
            continue
        ds = Dataset.from_list(rows, features=BUILD_FEATURES)
        ds.info.features = FEATURES
        if args.local_dir:
            print(f"  writing {action} ({len(ds)}) -> {args.local_dir / action}")
            _write_shards(ds, args.local_dir / action)
        if api is not None and not args.local_dir:
            print(f"  pushing {action} ({len(ds)}) -> {args.repo_id}")
            ds.push_to_hub(
                args.repo_id,
                config_name=action,
                split="test",
                private=True,
                max_shard_size="400MB",
                token=args.token,
            )
        del ds, rows

    if args.local_dir and not args.dry_run:
        (args.local_dir / "README.md").write_text(CARD)

    if args.dry_run or args.local_only:
        print(f"total non-Candor samples: {all_n} (expect 3539)")
        return

    assert api is not None
    if args.local_dir:
        print(f"  uploading folder {args.local_dir} -> {args.repo_id}")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(args.local_dir),
            repo_type="dataset",
        )
    else:
        api.upload_file(
            path_or_fileobj=str(Path("/tmp/DuplexAct_HF_README.md")),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print(f"total non-Candor samples: {all_n} (expect 3539)")


if __name__ == "__main__":
    main()
