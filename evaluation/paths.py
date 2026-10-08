"""Resolve the DuplexAct repo root.

Set ``DUPLEXACT_ROOT`` or ``BENCH_ROOT`` to override. Otherwise this file's
location (``evaluation/paths.py``) is used.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    for key in ("DUPLEXACT_ROOT", "BENCH_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value).resolve()
    return Path(__file__).resolve().parents[1]
