#!/usr/bin/env python3
"""Process-wide OpenAI key pool: primary OPENAI_API_KEY + OPENAI_API_KEY_FALLBACK.

Sticky failover on insufficient_quota / credit_balance_exhausted. Used by
GPT realtime WebSocket clients (the model API) and can be queried by batch
retry loops. Never prints key values.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_LOCK = threading.Lock()
_KEYS: list[str] | None = None
_IDX = 0
_REPO_ENV = Path(__file__).resolve().parents[3] / ".env"


def _is_quota_text(text: str) -> bool:
    t = text.lower()
    return any(
        s in t
        for s in (
            "insufficient_quota",
            "exceeded your current quota",
            "credit_balance",
            "no credits remaining",
        )
    )


def is_quota_error(exc_or_text: object) -> bool:
    if _is_quota_text(str(exc_or_text)):
        return True
    body = getattr(exc_or_text, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        code = err.get("code") if isinstance(err, dict) else None
        if code == "insufficient_quota":
            return True
    return getattr(exc_or_text, "code", None) == "insufficient_quota"


def load_keys() -> list[str]:
    global _KEYS
    if _KEYS is not None:
        return _KEYS
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ENV)
    except Exception:
        pass
    keys: list[str] = []
    for name in ("OPENAI_API_KEY", "OPENAI_API_KEY_FALLBACK"):
        k = (os.environ.get(name) or "").strip()
        if k and k not in keys:
            keys.append(k)
    _KEYS = keys
    if keys:
        print(
            f"[openai_keys] {len(keys)} key(s) ready "
            f"({'fallback armed' if len(keys) > 1 else 'no fallback'})",
            file=sys.stderr,
        )
    return _KEYS


def current_idx() -> int:
    with _LOCK:
        return _IDX


def current_key() -> str:
    keys = load_keys()
    if not keys:
        return ""
    with _LOCK:
        return keys[_IDX]


def note_quota(failed_idx: int, reason: str = "") -> bool:
    """Switch to the next key if failed_idx is still current. True = retry."""
    global _IDX
    keys = load_keys()
    with _LOCK:
        if failed_idx != _IDX:
            return True
        if _IDX + 1 >= len(keys):
            print(
                f"[openai_keys] key {failed_idx + 1}/{len(keys)} hit quota; no further key",
                file=sys.stderr,
            )
            return False
        _IDX += 1
        extra = f" ({reason[:80]})" if reason else ""
        print(
            f"[openai_keys] key {failed_idx + 1} hit quota{extra}; "
            f"switched to key {_IDX + 1}/{len(keys)}",
            file=sys.stderr,
        )
        return True


def note_quota_from_error(error_text: str) -> bool:
    if not is_quota_error(error_text):
        return False
    return note_quota(current_idx(), error_text)


def bearer_key(fallback: str = "") -> str:
    """Key to put on the realtime WebSocket Authorization header."""
    return current_key() or fallback
