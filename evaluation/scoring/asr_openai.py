#!/usr/bin/env python3
"""
evaluation/scoring/asr_openai.py
================================
Batch-transcribes every `output.wav` under an inference_results tree via the
OpenAI Whisper API (`whisper-1`, verbose_json, word + segment timestamps),
writing a sibling `asr.json`. Swap in a local model by replacing
`_transcribe_one`.

Output schema (asr.json) (`chunks: [{text, timestamp:[start,end]}]` = word-level):

    {
      "model": "whisper-1",
      "duration_s": <float>,
      "text": "<full transcript>",
      "chunks":   [{"text": "<word>",  "timestamp": [start, end]}, ...],   # word-level
      "segments": [{"text": "<clause>", "start": s, "end": e}, ...]        # segment-level
      "filled_segments": [{"start": s, "end": e, "text": "..."}, ...]      # only if any
    }

`chunks` is authoritative for per-segment scoring. Any model_speech_segment the
whole-file pass left wordless is re-transcribed on its own and merged into
`chunks` in the same run -- see _fill_unheard_segments for why that is needed
and what it costs to skip. One invocation is enough; there is no second pass.

A silent/near-empty output.wav (< MIN_AUDIO_S) is not sent to the API; an
empty asr.json ({"text": "", "chunks": [], "segments": [], "skipped": true})
is written instead.

KNOWN AND ACCEPTED -- Whisper hallucinates on silence. output.wav is a single
speaker track with long quiet stretches, which is exactly the condition that
makes whisper-1 emit high-frequency training phrases: this corpus contains
"Thanks for watching!", "don't forget to subscribe!", "Transcribed by
https://otter.ai", "Silence.", and bare "you" / "Thank you." as whole segments.
Measured 8731 bare-"you" segment texts across 38764 asr.json.

It is left uncorrected because scoring is structurally insulated from it.
Segments come from the recorder's VAD, never from ASR, and run_metrics.
segment_texts buckets word-level `chunks` into those segments by timestamp, so
a hallucination in silence has no segment to land in and is dropped: the 8731
collapse to 271 bare-token segments on the scoring path (0.27%), and flipping
those to "no transcript" moves 13 of 10992 Immediate scores (0.118%, worst cell
0.51%). Filtering would cost more than it saves, since wait.is_backchannel is a
word count (<= 3) and a real one-word backchannel is indistinguishable from a
hallucinated one by text alone.

Where it does bite hardest is _fill_unheard_segments: an isolated low-energy
slice with no surrounding context is the worst case for hallucination, and 95
of its 1111 outputs (8.6%) carry a suspicious signature, including cross-
language garbage. The absolute count is small enough not to matter, but do not
widen that pass without re-measuring.

Note when reading asr.json by hand: non-Latin text is usually NOT hallucination.
Several models genuinely answer in Chinese, so entire coherent Mandarin
sentences appear; and `chunks` is word-level, so a high count of the token
"you" is ordinary English word frequency rather than a hallucination signal.

Usage:
    python asr_openai.py --action wait --model gpt-realtime-2.1
    python asr_openai.py --action backchannel --l3_condition backchannel_affirmative \\
        --model gemini-3.1-flash-live --workers 8
    python asr_openai.py --action turn_taking --overwrite --limit 5   # smoke test, all models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from dotenv import load_dotenv


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
    return here.parents[2]

REPO_ROOT = _duplexact_root()
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation" / "direct_replay" / "results" / "inference_results"

DR_ACTIONS = ["wait", "backchannel", "ignore", "turn_taking"]
SIM_ACTIONS = ["stop", "hold_floor", "respond_and_recover", "interrupt_user"]
ACTIONS = DR_ACTIONS + SIM_ACTIONS

MIN_AUDIO_S = 0.30      # below this, don't bother calling the API
ASR_MODEL = "whisper-1"

# Repair pass (see _repair_empty_segments). A `model_speech_segment` that came
# back with no words at all is only re-transcribed if it is at least this long
# and its RMS is at least this fraction of the whole file's -- below that it is
# genuinely quiet and the empty result was correct.
MIN_SLICE_S = 0.12
SLICE_ENERGY_RATIO = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0

_KEY_LOCK = threading.Lock()
_KEY_IDX = 0
_CLIENTS = None


def _openai_keys() -> list[str]:
    primary = (os.environ.get("OPENAI_API_KEY") or "").strip()
    fallback = (os.environ.get("OPENAI_API_KEY_FALLBACK") or "").strip()
    keys = []
    for k in (primary, fallback):
        if k and k not in keys:
            keys.append(k)
    return keys


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return True
    if "credit_balance" in text or "no credits remaining" in text:
        return True
    return getattr(exc, "code", None) == "insufficient_quota"


def _clients():
    global _CLIENTS
    if _CLIENTS is None:
        from openai import OpenAI
        keys = _openai_keys()
        if not keys:
            print("[asr_openai] OPENAI_API_KEY / OPENAI_API_KEY_FALLBACK not set", file=sys.stderr)
            sys.exit(1)
        _CLIENTS = [OpenAI(api_key=k) for k in keys]
        print(f"[asr_openai] OpenAI ready ({len(keys)} key(s))", flush=True)
    return _CLIENTS


def _client():
    clients = _clients()
    with _KEY_LOCK:
        return clients[_KEY_IDX], _KEY_IDX


def _advance_key(from_idx: int) -> bool:
    global _KEY_IDX
    clients = _clients()
    with _KEY_LOCK:
        if from_idx != _KEY_IDX:
            return True
        if _KEY_IDX + 1 >= len(clients):
            return False
        _KEY_IDX += 1
        print(
            f"[asr_openai] key {from_idx + 1} hit quota; "
            f"switched to key {_KEY_IDX + 1}/{len(clients)}",
            flush=True,
        )
        return True


def _empty_asr(reason: str) -> dict:
    return {"model": ASR_MODEL, "duration_s": 0.0, "text": "", "chunks": [], "segments": [], "skipped": True, "skip_reason": reason}


def _call_whisper(wav_path: Path) -> tuple[str, list, list]:
    """One transcription call with key-rotation + backoff. Raises on final failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        client, key_idx = _client()
        try:
            with open(wav_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=ASR_MODEL,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )
            words = getattr(resp, "words", None) or []
            segs = getattr(resp, "segments", None) or []
            chunks = [
                {"text": w.word if hasattr(w, "word") else w["word"],
                 "timestamp": [
                     w.start if hasattr(w, "start") else w["start"],
                     w.end if hasattr(w, "end") else w["end"],
                 ]}
                for w in words
            ]
            segments = [
                {"text": (s.text if hasattr(s, "text") else s["text"]).strip(),
                 "start": s.start if hasattr(s, "start") else s["start"],
                 "end":   s.end if hasattr(s, "end") else s["end"]}
                for s in segs
            ]
            return (resp.text or "").strip(), chunks, segments
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and _advance_key(key_idx):
                continue
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    raise last_exc if last_exc else RuntimeError("transcription failed")


def _model_speech_segments(wav_path: Path) -> list[tuple[float, float]]:
    p = wav_path.parent / "inference_info.json"
    if not p.exists():
        return []
    try:
        inf = (json.loads(p.read_text()).get("inference") or {})
        return [(float(a), float(b)) for a, b in (inf.get("model_speech_segments") or [])]
    except Exception:
        return []


def _has_word_in(chunks: list, on: float, off: float) -> bool:
    """Mirrors the overlap rule of metrics/run_metrics.py's words_in()."""
    for c in chunks:
        ts = c.get("timestamp") or [None, None]
        s = ts[0]
        if s is None:
            continue
        e = ts[1] if len(ts) > 1 and ts[1] is not None else s
        if s < off and e > on:
            return True
    return False


def _fill_unheard_segments(wav_path: Path, duration_s: float, result: dict) -> dict:
    """
    Transcribing the whole file silently loses model turns that sit alone in a
    long stretch of silence: `output.wav` is a single-speaker track, so a 30 s
    Whisper window can hold one 0.24 s "Yeah." and nothing else, and the window
    is discarded as no-speech. Measured on this benchmark that costs 2.6 % of
    simulator segments and 4.8 % of direct_replay ones, and it is heavily
    model-dependent (0.9 % for gpt-realtime, 24 % for lychee-fd), so it biases
    cross-model comparison rather than adding uniform noise.

    Worse than the lost words is how a lost segment reads downstream: an empty
    transcript makes is_backchannel() return None, which every caller treats as
    substantive speech -- so a quarter-second "Yeah." can satisfy a takeover
    criterion. Full-Duplex-Bench sidesteps the same problem by cropping to the
    region of interest before its ASR call (get_transcript/asr.py) and by
    scoring an empty transcript as "no turn taken".

    So any `model_speech_segment` the whole-file pass left wordless is
    re-transcribed on its own, and its words are merged back with the segment
    offset added. This runs inside the single transcription step on purpose:
    one invocation of this script yields a correct asr.json, with no follow-up
    repair stage to remember.
    """
    segs = _model_speech_segments(wav_path)
    if not segs:
        return result
    chunks = result.get("chunks") or []
    missing = [(s, e) for s, e in segs
               if e > s and not _has_word_in(chunks, s, e)]
    if not missing:
        return result

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    whole_rms = float(np.sqrt((audio ** 2).mean())) if len(audio) else 0.0

    filled: list[dict] = []
    for s, e in missing:
        sl = audio[int(s * sr):int(e * sr)]
        if len(sl) < int(MIN_SLICE_S * sr):
            continue
        if whole_rms <= 0 or float(np.sqrt((sl ** 2).mean())) < SLICE_ENERGY_RATIO * whole_rms:
            continue        # genuinely quiet -- the empty result was correct
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            sf.write(tmp, sl, sr)
            text, words, _ = _call_whisper(Path(tmp))
        except Exception as exc:
            filled.append({"start": s, "end": e, "error": f"{type(exc).__name__}: {exc}"})
            continue
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        if not text:
            continue
        for w in words:
            ts = w.get("timestamp") or [0.0, 0.0]
            lo = (ts[0] or 0.0) + s
            hi = (ts[1] if len(ts) > 1 and ts[1] is not None else ts[0] or 0.0) + s
            # A slice-local timestamp can run past the slice on the last word.
            chunks.append({"text": w.get("text", ""),
                           "timestamp": [min(lo, e), min(hi, e)]})
        filled.append({"start": s, "end": e, "text": text})

    if not filled:
        return result
    chunks.sort(key=lambda c: (c.get("timestamp") or [0.0])[0])
    result["chunks"] = chunks
    # `text` stays the verbatim whole-file transcript (it keeps Whisper's
    # punctuation); everything that scores per segment reads `chunks`.
    result["filled_segments"] = filled
    return result


def _transcribe_one(wav_path: Path) -> dict:
    try:
        duration_s = sf.info(str(wav_path)).duration
    except Exception as exc:
        return _empty_asr(f"unreadable_wav:{exc}")

    if duration_s < MIN_AUDIO_S:
        return _empty_asr("too_short")

    try:
        text, chunks, segments = _call_whisper(wav_path)
    except Exception as exc:
        return _empty_asr(f"api_error:{type(exc).__name__}: {exc}")

    result = {
        "model": ASR_MODEL,
        "duration_s": duration_s,
        "text": text,
        "chunks": chunks,
        "segments": segments,
        "skipped": False,
    }
    return _fill_unheard_segments(wav_path, duration_s, result)


def discover_output_wavs(
    results_root: Path,
    action: Optional[str],
    l3_condition: Optional[str],
    model: Optional[str],
) -> list[Path]:
    """
    Walk to every `output.wav`, filtering by directory NAME rather than by
    depth, because the two result trees this has to serve nest differently:
    direct_replay is <action>/<l3_condition>/<model>/<sample>/, the simulator
    is <action>/<model>/<sample>/ (it has no l3_condition level). A batch run
    written to its own --results_root has no <action> level at all, so
    `action` is optional and simply selects a subdirectory when given.
    """
    scan_root = (results_root / action) if action else results_root
    if not scan_root.is_dir():
        return []
    wavs: list[Path] = []
    for wav in sorted(scan_root.rglob("output.wav")):
        parts = wav.relative_to(scan_root).parts
        if model and model not in parts:
            continue
        if l3_condition and l3_condition not in parts:
            continue
        wavs.append(wav)
    return wavs


def run(
    results_root: Path,
    action: str,
    l3_condition: Optional[str],
    model: Optional[str],
    workers: int,
    overwrite: bool,
    limit: Optional[int],
    verbose: bool,
) -> None:
    _clients()

    wavs = discover_output_wavs(results_root, action, l3_condition, model)
    todo = []
    n_skipped = 0
    for wav in wavs:
        asr_path = wav.parent / "asr.json"
        if asr_path.exists() and not overwrite:
            n_skipped += 1
            continue
        todo.append(wav)
    if limit is not None:
        todo = todo[:limit]

    print(f"[asr_openai] action={action or 'ROOT'} l3_condition={l3_condition or 'ALL'} "
          f"model={model or 'ALL'} candidates={len(wavs)} todo={len(todo)} "
          f"(skipped {n_skipped} already done)")

    n_ok = n_skip_short = n_err = 0

    def _job(wav_path: Path):
        result = _transcribe_one(wav_path)
        (wav_path.parent / "asr.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return wav_path, result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_job, w): w for w in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            wav_path, result = fut.result()
            sample_label = f"{wav_path.parent.parent.parent.name}/{wav_path.parent.parent.name}/{wav_path.parent.name}"
            if result.get("skipped"):
                if result.get("skip_reason") == "too_short":
                    n_skip_short += 1
                else:
                    n_err += 1
                    print(f"  [WARN] {sample_label}: {result.get('skip_reason')}", file=sys.stderr)
            else:
                n_ok += 1
            if verbose or i % 25 == 0 or i == len(todo):
                print(f"  … {i}/{len(todo)}  ok={n_ok} short={n_skip_short} err={n_err}")

    print(f"[asr_openai] done: ok={n_ok} too_short={n_skip_short} errors={n_err} "
          f"(+{n_skipped} already cached)")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", default=None, choices=ACTIONS,
                        help="Omit when --results_root already points below the action level.")
    parser.add_argument("--l3_condition", default=None, help="Restrict to one l3_condition directory name.")
    parser.add_argument("--model", default=None, help="Restrict to one model slug directory name.")
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run(
        results_root=args.results_root,
        action=args.action,
        l3_condition=args.l3_condition,
        model=args.model,
        workers=args.workers,
        overwrite=args.overwrite,
        limit=args.limit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
