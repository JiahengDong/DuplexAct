#!/usr/bin/env python3
"""
Loqui Pause Candidate Extractor

Extracts two types of single-speaker floor segments from a Loqui .trs recording:
  1. pause        – speaker has one or more <P> markers; enough words on both sides
  2. floor_holding – speaker holds floor with NO <P> markers, sufficient words & duration

Key design decisions vs. CORAAL:
  • Overlap detection: no timestamp comparison needed.  Multi-speaker turns
    (speaker="spkA spkB") already mark all overlap events in TRS.  Keeping
    only single-speaker turns is sufficient.
  • No punctuation checks: TRS transcripts contain no sentence-final
    punctuation, so inter- vs intra-sentential distinction is not applicable.
  • No pause duration: <P> carries no duration value, so no range filter.

Pipeline:
  Step 1 – Timing filter : scan .trs single-speaker turns
  Step 2 – GPT filter    : verify semantic quality via OpenAI API
  Step 3 – Audio export  : per-speaker mono clips from stereo .san.wav

Usage:
  python extract_pause_loqui.py \\
      --file_id 1 \\
      --trs_path .../Transcriptions/1.final.trs \\
      --wav_path .../Transcriptions/1.san.wav \\
      --results_dir .../results/pauses_loqui
"""

import argparse
import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

# A turn with ≤ 3 words is a backchannel (consistent across all scripts).
MIN_SUBSTANTIVE_WORDS = 4

# Both sides of a pause must be substantive; total turn must also clear this.
MIN_TURN_WORDS = MIN_SUBSTANTIVE_WORDS * 2   # = 8

# Floor-holding: same total-word minimum as pause turns.
FLOOR_HOLD_MIN_WORDS = MIN_TURN_WORDS

# Floor-holding turn must span at least this long (seconds).
FLOOR_HOLD_MIN_DUR = 3.0

AUDIO_SR = 16000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL       = "gpt-5.4-2026-03-05"
QUOTA_EXIT_CODE = 2

# ── Regex helpers ─────────────────────────────────────────────────────────────

# Pause marker (<P> or <p>)
_PAUSE_RE = re.compile(r'<[Pp]>', re.IGNORECASE)

# Non-standard speech / PII — reject entire turn containing any of these
_NON_STANDARD_RE = re.compile(
    r'<(?:BN|VB|IR|CHKL|NU|NS|G|ST|TN|TP|TC|CT)[^>]*>', re.IGNORECASE)
_PII_RE = re.compile(
    r'<(?:LASTNAME|FIRSTNAME|FULLNAME|ADDRESS[^>]*|STREETNAME|TOWN|'
    r'ZIPCODE|PHONENUMBER|LN-[^>]*|FN-[^>]*)>',
    re.IGNORECASE)

# Markup stripped before word-counting / transcript building:
#   [os / eo] / oe] bracket markers — keep content inside (same as interruption script)
#   \\ Transcriber line-end markers
#   <B> breath marker
_STRIP_RE = re.compile(
    r'\[os\b|\beo\]|\boe\]'   # overlap bracket markers (keep content)
    r'|<[Bb]>',               # <B> breath marker
    re.IGNORECASE)

# Valid speaker names
_VALID_SPKR_RE = re.compile(r'\b(?:caller|librarian)\b', re.IGNORECASE)


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Strip all markup and normalise whitespace, keeping spoken words."""
    t = _PAUSE_RE.sub(' ', raw)
    t = _STRIP_RE.sub(' ', t)
    t = re.sub(r'<[^>]+>', ' ', t)   # any remaining tags
    t = re.sub(r'&\w+;', ' ', t)     # XML entities (&lt; etc.)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def count_words(raw: str) -> int:
    return len(clean_text(raw).split())


def has_non_standard(raw: str) -> bool:
    return bool(_NON_STANDARD_RE.search(raw) or _PII_RE.search(raw))


# ── TRS parser ────────────────────────────────────────────────────────────────

def parse_trs(path: Path) -> tuple[dict, list[dict]]:
    """
    Parse a .final.trs file into single-speaker valid-speaker turns.

    Overlap detection strategy
    ──────────────────────────
    TRS explicitly encodes every overlap as a multi-speaker turn
    (speaker="spkA spkB").  By keeping ONLY single-speaker turns we already
    exclude all overlap events — no timestamp comparison is necessary.

    Returns
    ──────────────────────────
    valid_spkrs : {speaker_id → name}
    turns       : list of turn dicts with keys:
                    spkr_id, spkr_name, turn_start, turn_end,
                    raw_text   – full undecoded text (XML entities decoded by ET)
                    chunks     – list of {raw, sync_start, sync_end}

    Example (file 1, librarian turn at t=98.739):
    ──────────────────────────────────────────────
      raw_text  = 'i can tell you the- <P> numbers if you want \\\\ but uh'
      chunks    = [
        {'raw': 'i can tell you the- <P> numbers if you want \\\\ ', 'sync_start': 98.739, 'sync_end': 101.098},
        {'raw': 'but uh',                                             'sync_start': 101.098, 'sync_end': 101.61},
      ]
      → _split_on_pauses(raw_text):
          [{'kind':'speech', 'raw':'i can tell you the- '},
           {'kind':'pause',  'raw':'<P>'},
           {'kind':'speech', 'raw':' numbers if you want \\\\ but uh'}]
      → words before pause: 4  ✓   words after: 5  ✓  → PASSED as 'pause'
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        print(f"  WARN: XML parse error in {path.name}: {e}")
        return {}, []

    root = tree.getroot()

    valid_spkrs = {}
    for el in root.findall('.//Speaker'):
        sid  = el.get('id', '')
        name = el.get('name', '')
        if _VALID_SPKR_RE.search(name):
            valid_spkrs[sid] = name

    turns = []
    for turn_el in root.iter('Turn'):
        t_start  = float(turn_el.get('startTime', 0))
        t_end    = float(turn_el.get('endTime',   0))
        spk_attr = (turn_el.get('speaker') or '').split()

        # Single-speaker, valid speaker → no overlap by definition
        valid_in = [s for s in spk_attr if s in valid_spkrs]
        if len(valid_in) != 1:
            continue
        spkr_id = valid_in[0]

        # Rebuild text with Sync boundaries tracked
        sync_times: list[float] = [t_start]
        texts_between: list[str] = []
        current_text = turn_el.text or ''

        for child in turn_el:
            if child.tag == 'Sync':
                sync_times.append(float(child.get('time', t_start)))
                texts_between.append(current_text)
                current_text = child.tail or ''
            else:
                # Who blocks are unexpected in single-speaker turns; keep content
                inner = (child.text or '') + (child.tail or '')
                # Strip [os...eo] bracket markers but keep content inside
                inner = re.sub(r'\[os\b.*?(?:eo\]|oe\])', r' \g<0> ', inner, flags=re.DOTALL)
                inner = re.sub(r'\[os\b|\beo\]|\boe\]', ' ', inner)
                current_text += inner
        texts_between.append(current_text)

        chunk_list: list[dict] = []
        for i, txt in enumerate(texts_between):
            s_start = sync_times[i] if i < len(sync_times) else t_start
            s_end   = sync_times[i + 1] if i + 1 < len(sync_times) else t_end
            # Strip Transcriber line-end markers (\\ at end of lines) and
            # normalise whitespace so each chunk reads as continuous speech.
            txt_clean = re.sub(r'\\\\', ' ', txt)           # \\ → space
            txt_clean = re.sub(r'\s+', ' ', txt_clean).strip()
            if txt_clean:
                chunk_list.append({'raw': txt_clean, 'sync_start': s_start, 'sync_end': s_end})

        raw_text = ' '.join(c['raw'] for c in chunk_list)
        turns.append({
            'spkr_id':    spkr_id,
            'spkr_name':  valid_spkrs[spkr_id],
            'turn_start': t_start,
            'turn_end':   t_end,
            'raw_text':   raw_text,
            'chunks':     chunk_list,
        })

    return valid_spkrs, turns


# ── Pause item splitter ────────────────────────────────────────────────────────

def _split_on_pauses(raw: str) -> list[dict]:
    """
    Split raw text on <P> markers.
    Returns alternating speech / pause items:
      [{'kind':'speech','raw':...}, {'kind':'pause','raw':'<P>'}, ...]
    """
    items: list[dict] = []
    parts = _PAUSE_RE.split(raw)
    for i, part in enumerate(parts):
        if part.strip():
            items.append({'kind': 'speech', 'raw': part})
        if i < len(parts) - 1:
            items.append({'kind': 'pause', 'raw': '<P>'})
    return items


# ── Step 1: Timing Filters ────────────────────────────────────────────────────

def find_pause(turns: list[dict], file_id: str) -> tuple[list[dict], dict]:
    """
    Single-speaker turns that contain at least one <P> marker with enough
    words on both sides.

    Rejection criteria (in order applied):
      rej_non_standard : turn contains non-standard/PII markers
      rej_no_pauses    : no <P> found
      rej_too_short    : total word count < MIN_TURN_WORDS (8)
      rej_before_short : words before first <P> < MIN_SUBSTANTIVE_WORDS (4)
      rej_after_short  : words after  last  <P> < MIN_SUBSTANTIVE_WORDS (4)
    """
    candidates = []
    stats = dict(
        total_turns      = len(turns),
        rej_non_standard = 0,
        rej_no_pauses    = 0,
        rej_too_short    = 0,
        rej_before_short = 0,
        rej_after_short  = 0,
        passed           = 0,
    )

    for i, turn in enumerate(turns):
        raw = turn['raw_text']

        if has_non_standard(raw):
            stats['rej_non_standard'] += 1
            continue

        items = _split_on_pauses(raw)
        pause_idxs = [j for j, it in enumerate(items) if it['kind'] == 'pause']

        if not pause_idxs:
            stats['rej_no_pauses'] += 1
            continue

        full_text = ' '.join(it['raw'] for it in items if it['kind'] == 'speech')
        if count_words(full_text) < MIN_TURN_WORDS:
            stats['rej_too_short'] += 1
            continue

        first_pi = pause_idxs[0]
        last_pi  = pause_idxs[-1]

        text_before = ' '.join(it['raw'] for it in items[:first_pi] if it['kind'] == 'speech')
        text_after  = ' '.join(it['raw'] for it in items[last_pi + 1:] if it['kind'] == 'speech')

        if count_words(text_before) < MIN_SUBSTANTIVE_WORDS:
            stats['rej_before_short'] += 1
            continue
        if count_words(text_after) < MIN_SUBSTANTIVE_WORDS:
            stats['rej_after_short'] += 1
            continue

        # Build clean transcript with [PAUSE] markers
        transcript_parts = []
        for j, it in enumerate(items):
            if it['kind'] == 'speech':
                t = clean_text(it['raw'])
                if t:
                    transcript_parts.append(t)
            else:
                transcript_parts.append('[PAUSE]')

        prev_end = turns[i - 1]['turn_end'] if i > 0 else None
        candidates.append({
            'type':            'pause',
            'file_id':         file_id,
            'spkr_id':         turn['spkr_id'],
            'spkr_name':       turn['spkr_name'],
            'n_pauses':        len(pause_idxs),
            'clip_start':      round(turn['turn_start'], 3),
            'clip_end':        round(turn['turn_end'],   3),
            'prev_turn_end':   round(prev_end, 3) if prev_end is not None else None,
            'full_transcript': ' '.join(transcript_parts),
            'gpt_valid':       None,
            'gpt_reason':      None,
            'gpt_details':     None,
        })
        stats['passed'] += 1

    return candidates, stats


def find_floor_holding(turns: list[dict], file_id: str) -> tuple[list[dict], dict]:
    """
    Single-speaker turns with NO <P> markers, sufficient word count and duration.

    Rejection criteria (in order applied):
      rej_non_standard   : turn contains non-standard/PII markers
      rej_has_pauses     : turn contains <P>
      rej_too_short      : word count < FLOOR_HOLD_MIN_WORDS (8)
      rej_duration_short : duration < FLOOR_HOLD_MIN_DUR (3.0 s)
    """
    candidates = []
    stats = dict(
        total_turns        = len(turns),
        rej_non_standard   = 0,
        rej_has_pauses     = 0,
        rej_too_short      = 0,
        rej_duration_short = 0,
        passed             = 0,
    )

    for i, turn in enumerate(turns):
        raw = turn['raw_text']

        if has_non_standard(raw):
            stats['rej_non_standard'] += 1
            continue

        if _PAUSE_RE.search(raw):
            stats['rej_has_pauses'] += 1
            continue

        n_words = count_words(raw)
        if n_words < FLOOR_HOLD_MIN_WORDS:
            stats['rej_too_short'] += 1
            continue

        dur = turn['turn_end'] - turn['turn_start']
        if dur < FLOOR_HOLD_MIN_DUR:
            stats['rej_duration_short'] += 1
            continue

        utt      = clean_text(raw)
        prev_end = turns[i - 1]['turn_end'] if i > 0 else None
        candidates.append({
            'type':            'floor_holding',
            'file_id':         file_id,
            'spkr_id':         turn['spkr_id'],
            'spkr_name':       turn['spkr_name'],
            'n_words':         n_words,
            'duration':        round(dur, 3),
            'clip_start':      round(turn['turn_start'], 3),
            'clip_end':        round(turn['turn_end'],   3),
            'prev_turn_end':   round(prev_end, 3) if prev_end is not None else None,
            'utterance':       utt,
            'full_transcript': utt,
            'gpt_valid':       None,
            'gpt_reason':      None,
            'gpt_details':     None,
        })
        stats['passed'] += 1

    return candidates, stats


# ── Step 2: GPT filter ────────────────────────────────────────────────────────

PAUSE_PROMPT = """\
You are evaluating a natural conversation segment for research on spoken dialogue systems.

A speaker pauses {n_pauses} time(s) during their turn:

{transcript}

Answer the following questions in order:

1. COMPLETE UTTERANCE: Does this utterance form a complete, coherent speech act?
   - The words before and after each pause must together form a meaningful whole.
   - If the speaker appears cut off or interrupted, answer false.

2. NATURAL PAUSE: Is every pause positioned in a meaningful location in the speech?
   - The words before each pause should form a substantive phrase (not just 1-2 filler words).
   - The words after each pause should continue the utterance naturally.

3. SELF-CONTAINED: Can this utterance be understood WITHOUT any prior context?
   - It should introduce its own topic clearly enough to stand alone.
   - Reject if it relies on something said earlier (unnamed "it/that/they", or is clearly a
     fragment of a longer answer to an unseen question).

4. PAUSE TYPE (answer only if questions 1-3 are all true):
   Note: this is spoken dialogue — there is NO punctuation in the transcript.
   You must judge pause position purely from the meaning of the surrounding words.
   Classify every [PAUSE] marker:
   - inter_sentential: the words BEFORE the pause form a complete, self-sufficient
     meaning unit (a full idea or statement has ended), AND the words AFTER start
     a new unit. The pause sits at a natural sentence boundary.
   - intra_sentential: the words BEFORE the pause leave the meaning incomplete
     (e.g. mid-phrase, mid-clause, or after a connective like "and"/"but"/"so"
     that needs the following words to make sense).
   Then:
   - If ALL pauses are inter_sentential → pause_type = "inter_sentential"
   - If ALL pauses are intra_sentential → pause_type = "intra_sentential"
   - If pauses are a mix of both        → pause_type = "mixed"
   A turn is only valid when pause_type is NOT "mixed".
   If any of questions 1-3 is false, set pause_type = null.

Respond ONLY with JSON (no markdown):
{{"complete_utterance": true/false, "natural_pause": true/false, "self_contained": true/false, "pause_type": "inter_sentential"|"intra_sentential"|"mixed"|null, "is_valid": true/false, "reason": "one sentence"}}"""

FLOOR_PROMPT = """\
You are evaluating a natural conversation segment for research on spoken dialogue systems.

A speaker held the floor for the following utterance (the other speaker only inserted brief backchannels):

Utterance: "{utterance}"

Evaluate:
1. Is this utterance semantically complete and meaningful?
2. Does it represent a coherent, self-contained speech act (not a fragment)?
3. SELF-CONTAINED: Can this utterance be understood WITHOUT any prior context?
   - Reject if it refers to unnamed "it/that/they" whose referent is unknown, or is clearly
     a continuation of an answer to an unseen question.
4. Overall, is this a valid floor-holding sample suitable for benchmarking a dialogue model?

Respond ONLY with JSON (no markdown):
{{"is_complete": true/false, "is_coherent": true/false, "self_contained": true/false, "is_valid": true/false, "reason": "one sentence"}}"""


def build_gpt_message(candidate: dict) -> str:
    if candidate['type'] == 'pause':
        return PAUSE_PROMPT.format(
            n_pauses=candidate.get('n_pauses', 1),
            transcript=candidate['full_transcript'])
    else:
        return FLOOR_PROMPT.format(utterance=candidate['full_transcript'])


class QuotaExceededError(RuntimeError):
    pass


def gpt_filter(candidates: list[dict], client: OpenAI,
               max_valid: int | None = None) -> tuple[list[dict], dict]:
    total   = len(candidates)
    n_valid = 0
    gpt_stats = dict(total_sent=total, valid=0, rejected=0, criteria={})

    print(f"  Sending {total} candidates to GPT ({GPT_MODEL})"
          + (f" (cap={max_valid})" if max_valid else "") + "...")
    for i, cand in enumerate(candidates):
        if max_valid and n_valid >= max_valid:
            print(f"    [{i+1}/{total}] cap reached — skipping rest")
            break
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": build_gpt_message(cand)}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            cand['gpt_valid']   = result.get('is_valid', False)
            cand['gpt_reason']  = result.get('reason', '')
            cand['gpt_details'] = {k: v for k, v in result.items()
                                   if k not in ('is_valid', 'reason')}
        except Exception as e:
            err_str = str(e)
            if 'insufficient_quota' in err_str or '429' in err_str:
                cand['gpt_valid']  = None
                cand['gpt_reason'] = f"API error: {e}"
                raise QuotaExceededError(str(e))
            print(f"    [{i+1}/{total}] WARN GPT call failed: {e}")
            cand['gpt_valid']  = None
            cand['gpt_reason'] = f"API error: {e}"
            continue

        # For pause turns: override is_valid if pause_type is mixed or null
        if cand['type'] == 'pause' and cand.get('gpt_valid'):
            pt = (cand.get('gpt_details') or {}).get('pause_type')
            if pt in ('mixed', None):
                cand['gpt_valid'] = False
                cand['gpt_reason'] = f"pause_type={pt} (inconsistent or unclassifiable)"
            else:
                # Promote the type: inter_sentential or intra_sentential
                cand['pause_type'] = pt

        if cand['gpt_valid']:
            n_valid += 1
            gpt_stats['valid'] += 1
        else:
            gpt_stats['rejected'] += 1
            for key, val in (cand.get('gpt_details') or {}).items():
                if val is False:
                    gpt_stats['criteria'][key] = gpt_stats['criteria'].get(key, 0) + 1
            # also track mixed-pause rejections
            if cand['type'] == 'pause':
                pt = (cand.get('gpt_details') or {}).get('pause_type')
                if pt == 'mixed':
                    gpt_stats['criteria']['pause_type_mixed'] = \
                        gpt_stats['criteria'].get('pause_type_mixed', 0) + 1

        status = 'valid' if cand['gpt_valid'] else 'reject'
        pt_str = f" [{cand.get('pause_type','')}]" if cand.get('pause_type') else ''
        print(f"    [{i+1}/{total}] {status}{pt_str}  — {cand.get('full_transcript', '')[:70]}")

    return candidates, gpt_stats


# ── Step 3: Audio Export ───────────────────────────────────────────────────────

def export_audio_clip(audio: np.ndarray, sr: int,
                      start_s: float, end_s: float,
                      out_path: Path,
                      prev_turn_end: float | None = None) -> None:
    clip_start = max(start_s, prev_turn_end) if prev_turn_end is not None else start_s
    s = max(0, int(clip_start * sr))
    e = min(len(audio), int(end_s * sr))
    sf.write(str(out_path), audio[s:e], sr)


def export_candidates(candidates: list[dict],
                      audio_ch0: np.ndarray, audio_ch1: np.ndarray, sr: int,
                      spkr_channel: dict,
                      results_dir: Path) -> None:
    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue

        file_id   = cand.get('file_id', 'unknown')
        spkr_id   = cand.get('spkr_id', '')
        ch        = spkr_channel.get(spkr_id, 0)
        audio     = audio_ch0 if ch == 0 else audio_ch1
        n         = cand.get('n_pauses', 1)
        cand_type = cand.get('type', 'pause')

        if cand_type == 'floor_holding':
            out_dir = results_dir / 'floor_holding'
            base    = f"{file_id}_floor_{round(cand['clip_start'], 2)}"
        else:
            # pause candidates: use the GPT-assigned pause_type as folder
            pt      = cand.get('pause_type', 'pause')
            out_dir = results_dir / pt / f"n{n}"
            base    = f"{file_id}_{pt[:5]}_{round(cand['clip_start'], 2)}"

        out_dir.mkdir(parents=True, exist_ok=True)
        full_path = out_dir / f"{base}_full.wav"
        export_audio_clip(audio, sr, cand['clip_start'], cand['clip_end'],
                          full_path, cand.get('prev_turn_end'))
        cand['audio_full'] = str(full_path)


# ── Main ──────────────────────────────────────────────────────────────────────

PAUSE_TYPES = ('inter_sentential', 'intra_sentential', 'floor_holding')


def main():
    parser = argparse.ArgumentParser(
        description="Extract pause candidates from a Loqui .trs recording.")
    parser.add_argument("--file_id",            required=True)
    parser.add_argument("--trs_path",           type=Path, required=True)
    parser.add_argument("--wav_path",           type=Path, required=True)
    parser.add_argument("--results_dir",        type=Path, required=True)
    parser.add_argument("--max_valid_per_type", type=int, default=500)
    parser.add_argument("--no_gpt",             action="store_true")
    parser.add_argument("--no_audio",           action="store_true")
    parser.add_argument("--force",              action="store_true")
    args = parser.parse_args()

    out_dir    = args.results_dir / args.file_id
    valid_path = out_dir / "valid.json"

    if valid_path.exists() and not args.force:
        print(f"{args.file_id}: already done (use --force to redo).")
        data = json.loads(valid_path.read_text())
        print(f"VALID_JSON:{json.dumps(data)}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse ─────────────────────────────────────────────────────────────────
    valid_spkrs, turns = parse_trs(args.trs_path)
    if not turns:
        print(f"{args.file_id}: no valid single-speaker turns found.")
        print("TIMING_STATS:{}")
        return

    print(f"{args.file_id}: {len(valid_spkrs)} valid speakers, {len(turns)} single-spkr turns")

    spkr_channel: dict[str, int] = {
        sid: (0 if 'caller' in name.lower() else 1)
        for sid, name in valid_spkrs.items()
    }

    # ── Step 1: Timing filters ────────────────────────────────────────────────
    print("\n── Step 1: Timing filters ──")
    pause_cands, t_stats_pause = find_pause(turns,         args.file_id)
    floor_cands, t_stats_floor = find_floor_holding(turns, args.file_id)

    print(f"  pause        : {len(pause_cands)}")
    print(f"  floor_holding: {len(floor_cands)}")
    print(f"  SEND_TO_LLM pause: {len(pause_cands)}")
    print(f"  SEND_TO_LLM floor_holding: {len(floor_cands)}")

    timing_stats = {"pause": t_stats_pause, "floor_holding": t_stats_floor}
    print(f"TIMING_STATS:{json.dumps(timing_stats)}")

    f1_dir = out_dir / "filter1_timing"
    f1_dir.mkdir(exist_ok=True)
    (f1_dir / "pause.json").write_text(json.dumps(pause_cands, indent=2))
    (f1_dir / "floor_holding.json").write_text(json.dumps(floor_cands, indent=2))
    (f1_dir / "all.json").write_text(json.dumps(pause_cands + floor_cands, indent=2))

    if args.no_gpt or (not pause_cands and not floor_cands):
        print("  Skipping GPT step.")
        return

    # ── Step 2: GPT filter ────────────────────────────────────────────────────
    print(f"\n── Step 2: GPT filter ({GPT_MODEL}) ──")
    client = OpenAI(api_key=OPENAI_API_KEY)
    cap    = args.max_valid_per_type

    g_stats_pause = g_stats_floor = {}
    try:
        print("  [pause]")
        pause_cands, g_stats_pause = gpt_filter(pause_cands, client, max_valid=cap)
        print("  [floor_holding]")
        floor_cands, g_stats_floor = gpt_filter(floor_cands, client, max_valid=cap)
    except QuotaExceededError as e:
        print(f"  Quota exceeded: {e}")

    pause_valid = [c for c in pause_cands if c.get('gpt_valid')]
    floor_valid = [c for c in floor_cands if c.get('gpt_valid')]

    inter_valid = [c for c in pause_valid if c.get('pause_type') == 'inter_sentential']
    intra_valid = [c for c in pause_valid if c.get('pause_type') == 'intra_sentential']

    print(f"\n  Valid after GPT:")
    print(f"    pause (inter_sentential): {len(inter_valid)}/{len(pause_cands)}")
    print(f"    pause (intra_sentential): {len(intra_valid)}/{len(pause_cands)}")
    print(f"    floor_holding           : {len(floor_valid)}/{len(floor_cands)}")
    print(f"  VALID_AFTER_LLM inter_sentential: {len(inter_valid)}")
    print(f"  VALID_AFTER_LLM intra_sentential: {len(intra_valid)}")
    print(f"  VALID_AFTER_LLM floor_holding: {len(floor_valid)}")

    gpt_stats = {"pause": g_stats_pause, "floor_holding": g_stats_floor}
    print(f"GPT_STATS:{json.dumps(gpt_stats)}")

    f2_dir = out_dir / "filter2_gpt"
    f2_dir.mkdir(exist_ok=True)
    (f2_dir / "pause.json").write_text(json.dumps(pause_cands, indent=2))
    (f2_dir / "floor_holding.json").write_text(json.dumps(floor_cands, indent=2))
    (f2_dir / "all_with_gpt_decisions.json").write_text(
        json.dumps(pause_cands + floor_cands, indent=2))

    valid_all = pause_valid + floor_valid
    # Tag floor_holding candidates with their type for consistent aggregation
    for c in floor_valid:
        c.setdefault('pause_type', 'floor_holding')
    valid_path.write_text(json.dumps(valid_all, indent=2))

    # ── Step 3: Audio export ──────────────────────────────────────────────────
    if args.no_audio or not valid_all:
        print("  Skipping audio export.")
        print(f"VALID_JSON:{json.dumps(valid_all)}")
        return

    print(f"\n── Step 3: Audio export ──")
    raw, sr_orig = sf.read(str(args.wav_path), always_2d=True)
    ch0 = librosa.resample(raw[:, 0].astype(np.float32), orig_sr=sr_orig, target_sr=AUDIO_SR)
    ch1 = librosa.resample(raw[:, 1].astype(np.float32), orig_sr=sr_orig, target_sr=AUDIO_SR) \
          if raw.shape[1] > 1 else ch0
    print(f"  Audio: {len(ch0)/AUDIO_SR:.1f}s @ {AUDIO_SR}Hz")

    export_candidates(pause_valid + floor_valid, ch0, ch1, AUDIO_SR, spkr_channel, args.results_dir)

    total_clips = sum(1 for c in valid_all if c.get('audio_full'))
    print(f"  Exported {total_clips} clips → {args.results_dir}/")
    print(f"VALID_JSON:{json.dumps(valid_all)}")


if __name__ == "__main__":
    main()
