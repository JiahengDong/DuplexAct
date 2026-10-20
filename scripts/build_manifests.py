#!/usr/bin/env python3
"""Write <action>/manifest.json from the sample folders on disk.

The manifest only lists samples for Direct Replay inference; everything else
is read from each sample's own metadata.json, so the manifest is derived from
it and never shipped.

    python scripts/build_manifests.py --data_dir evaluation/data
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ACTIONS = [
    "wait", "backchannel", "ignore", "turn_taking",
    "stop", "hold_floor", "respond_and_recover", "interrupt_user",
]
CANDOR_SOURCE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def source_dataset(meta: dict) -> str:
    """Corpus of a sample (AMI / CORAAL / Candor / Loqui / synthesized)."""
    ds = meta.get("dataset") or ""
    if ds != "synthesized" and CANDOR_SOURCE.match(str(meta.get("source_id", ""))):
        return "Candor"
    return ds


def l3_condition(meta: dict, folder: str) -> str:
    return meta.get("l3_condition") or folder


def manifest_entry(action: str, sample_dir: Path, meta: dict) -> dict:
    folder = sample_dir.parent.name
    return {
        "sample_id": sample_dir.name,
        "l3_condition": l3_condition(meta, folder),
        "dataset": source_dataset(meta),
        "sample_dir": f"{action}/{folder}/{sample_dir.name}",
    }


def write_manifests(data_dir: Path, actions: list[str] | None = None) -> dict[str, int]:
    counts = {}
    for action in actions or ACTIONS:
        root = data_dir / action
        if not root.is_dir():
            continue
        entries = [
            manifest_entry(action, p.parent, json.loads(p.read_text()))
            for p in sorted(root.glob("*/*/metadata.json"))
        ]
        (root / "manifest.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        counts[action] = len(entries)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, required=True)
    args = ap.parse_args()
    for action, n in write_manifests(args.data_dir).items():
        print(f"  {action}: {n}")


if __name__ == "__main__":
    main()
