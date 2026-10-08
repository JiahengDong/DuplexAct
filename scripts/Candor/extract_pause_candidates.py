#!/usr/bin/env python3
"""
CANDOR Pause Candidate Extractor

Extracts three types of single-floor segments from a CANDOR conversation:
  1. inter_sentential  – speaker pauses between two complete sentences
                         (other speaker may insert a short backchannel in the gap)
  2. intra_sentential  – speaker pauses mid-sentence (word-level gap within a turn)
  3. floor_holding     – speaker holds floor for a full turn, no interruption
                         (backchannels from other speaker are allowed)

Pipeline:
  Step 1 – Timing filter  : extract candidates from CSV + JSON timestamps
  Step 2 – GPT filter     : verify semantic completeness via OpenAI API
  Step 3 – Audio export   : extract .wav clips using librosa + soundfile

Usage:
  python extract_pause_candidates.py --conv_dir <path_to_conversation_dir>
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from openai import OpenAI

# ── Default Configuration ──────────────────────────────────────────────────────
DEFAULT_CONV_DIR = Path(
    "/data/gpfs/projects/punim2960/jiahengd/dataset/Candor"
    "/part_001/0d4a795e-7026-4d98-b6d0-739a2c7c1661"
)

PAUSE_MIN             = 0.2    # min pause gap for both inter- and intra-sentential (s)
PAUSE_MAX             = 1.0    # max pause gap (longer = likely recording gap)
INTER_PAUSE_MIN       = PAUSE_MIN
INTER_PAUSE_MAX       = PAUSE_MAX
INTRA_PAUSE_MIN       = PAUSE_MIN
INTRA_PAUSE_MAX       = PAUSE_MAX
BACKCHANNEL_MAX_WORDS = 3      # other speaker turn considered a backchannel if <= this
BACKCHANNEL_MAX_DUR   = 2.0    # and duration <= this (s)

# A turn with ≤ 3 words is a backchannel; substantive content requires > 3 words.
MIN_SUBSTANTIVE_WORDS = 4      # minimum words on each side of a pause
MIN_TURN_WORDS        = MIN_SUBSTANTIVE_WORDS * 2   # 8 — minimum total words for candidates

# Floor-holding turn has no pause, same total-word minimum applies.
FLOOR_HOLD_MIN_WORDS  = MIN_TURN_WORDS   # 8 — consistent with CORAAL / AMI / Loqui
FLOOR_HOLD_MIN_DUR    = 3.0    # and duration >= this (s)
AUDIO_PADDING         = 0.0    # seconds of silence padding around each clip
MAX_OVERLAP           = 1.5    # max tolerated overlap (s) with both prev and next turn
AUDIO_SR              = 16000  # output sample rate for saved clips
BC_PAUSE_MARGIN       = 0.3    # backchannel within this many seconds of the pause → skip

TARGET_PER_CATEGORY   = 500    # stop GPT calls for a category once this many are valid
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL      = "gpt-5.4-2026-03-05"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_at_pause_marker(full: str) -> tuple[str, str]:
    """Split transcript at first [PAUSE ...] marker → (pre_text, post_text)."""
    parts = re.split(r'\[PAUSE[^\]]*\]', full, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return full.strip(), ''


SENT_FINAL = re.compile(r'[.?!]\s*$')

def ends_sentence(text: str) -> bool:
    return bool(SENT_FINAL.search(str(text).strip()))

def is_backchannel_turn(row) -> bool:
    """
    Length-based backchannel check: short, non-question turns are backchannels.
    """
    utt = str(row.get('utterance', '')).strip()
    if utt.endswith('?'):
        return False
    dur = float(row['stop']) - float(row['start'])
    if dur > BACKCHANNEL_MAX_DUR:
        return False
    return int(row.get('n_words', 999)) <= BACKCHANNEL_MAX_WORDS

def no_large_overlap(row, prev_row=None, next_row=None,
                     max_overlap: float = MAX_OVERLAP) -> bool:
    """
    Bidirectional overlap check with configurable tolerance.

    A turn is accepted if:
      - The PREVIOUS turn ended no more than max_overlap seconds AFTER this
        turn started (backward overlap check).
      - The NEXT turn starts no more than max_overlap seconds BEFORE this
        turn ended (forward overlap check).
    Both checks must pass; missing neighbours are treated as passing.
    """
    if prev_row is not None:
        backward = float(row['start']) - float(prev_row['stop'])
        if backward < -max_overlap:
            return False
    if next_row is not None:
        forward = float(next_row['start']) - float(row['stop'])
        if forward < -max_overlap:
            return False
    return True


# Keep old name as alias so floor_holding callers still compile
no_genuine_interruption = no_large_overlap

def build_speaker_words(trans_json: dict, ch_to_speaker: dict) -> dict[str, list]:
    """
    Build a per-speaker word list from transcribe_output.json, attaching
    trailing punctuation to each word token.
    Each word: {start, end, word, sent_end (bool)}
    """
    channels = trans_json['results']['channel_labels']['channels']
    speaker_words: dict[str, list] = {}
    for ch_idx, ch in enumerate(channels):
        spk = ch_to_speaker.get(f"ch_{ch_idx}", f"ch_{ch_idx}")
        words = []
        items = ch['items']
        for j, item in enumerate(items):
            if item['type'] != 'pronunciation':
                continue
            punct = ''
            if j + 1 < len(items) and items[j + 1]['type'] == 'punctuation':
                punct = items[j + 1]['alternatives'][0]['content']
            words.append({
                'start':    float(item['start_time']),
                'end':      float(item['end_time']),
                'word':     item['alternatives'][0]['content'],
                'sent_end': punct in {'.', '?', '!'},
            })
        speaker_words[spk] = words
    return speaker_words


def trim_short_sentences(words: list[dict], min_words: int = 4) -> tuple | None:
    """
    Turn-level trim: strip leading and trailing sentences with fewer than
    `min_words` words (i.e. ≤ 3 words by default).

    A "sentence" here is a run of words ending at a sent_end word.  The last
    run (no sent_end punctuation) is also treated as a sentence.

    Example: "Right. Nice. Okay. Yeah I have heard of Huntsville..."
      → "Right."(1), "Nice."(1), "Okay."(1) are all ≤ 3 → stripped
      → "Yeah I have heard of Huntsville..." (many) → new clip_start

    Returns (trimmed_words, new_start, new_end) or None if nothing
    substantive (> min_words-1 words) remains after trimming.
    """
    # Split words into sentence groups
    sentences: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        current.append(w)
        if w['sent_end']:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    if not sentences:
        return None

    # Strip leading short sentences
    s = 0
    while s < len(sentences) and len(sentences[s]) < min_words:
        s += 1

    # Strip trailing short sentences
    e = len(sentences) - 1
    while e >= s and len(sentences[e]) < min_words:
        e -= 1

    if s > e:
        return None

    trimmed = [w for sent in sentences[s:e + 1] for w in sent]
    return trimmed, trimmed[0]['start'], trimmed[-1]['end']

def has_other_speech_in(spk: str, t_start: float, t_end: float,
                         back: pd.DataFrame, margin: float = 0.1) -> bool:
    """
    True if another speaker has a non-backchannel turn starting strictly inside
    (t_start + margin, t_end - margin).

    Backbiter already absorbed all backchannel turns into the main speaker's row,
    so any remaining Backbiter row from another speaker is by definition non-backchannel.
    No extra backchannel check needed.
    """
    for _, row in back.iterrows():
        if row['speaker'] == spk:
            continue
        r_start = float(row['start'])
        if t_start + margin < r_start < t_end - margin:
            return True
    return False


def is_question_turn(row) -> bool:
    """True if the backbiter turn contains at least one question mark."""
    return int(row.get('questions', 0)) > 0



# ── Step 1: Timing Filters ─────────────────────────────────────────────────────

def find_inter_sentential(speaker_words: dict[str, list],
                          back: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Inter-sentential pauses within a single backbiter turn.
    Returns (candidates, timing_stats).
    """
    conv_id = back.attrs.get('conv_id', '')
    turns   = back.reset_index(drop=True)
    candidates = []
    stats = dict(
        total_turns          = len(turns),
        rej_no_words         = 0,   # no words found in turn window
        rej_trim_failed      = 0,   # trim_short_sentences returned None
        rej_overlap          = 0,   # next turn starts before clip_end
        rej_mixed_pauses     = 0,   # intra-sentential gap found (purity)
        rej_pre_gap_too_large= 0,   # gap > INTER_PAUSE_MAX in turn
        rej_no_pauses        = 0,   # no qualifying inter-sentential pause
        rej_before_short     = 0,   # words before first pause < MIN_SUBSTANTIVE_WORDS
        rej_after_short      = 0,   # words after last pause  < MIN_SUBSTANTIVE_WORDS
        passed               = 0,
    )

    for i, turn_row in turns.iterrows():
        spk        = turn_row['speaker']
        turn_start = float(turn_row['start'])
        turn_stop  = float(turn_row['stop'])

        raw_words = [w for w in speaker_words.get(spk, [])
                     if turn_start <= w['start'] and w['end'] <= turn_stop]
        if not raw_words:
            stats['rej_no_words'] += 1
            continue

        result = trim_short_sentences(raw_words)
        if result is None:
            stats['rej_trim_failed'] += 1
            continue
        turn_words, clip_start, clip_end = result

        # ── Overlap check ─────────────────────────────────────────────────────
        next_row = turns.iloc[i + 1] if i + 1 < len(turns) else None
        if next_row is not None and float(next_row['start']) < clip_end - MAX_OVERLAP:
            stats['rej_overlap'] += 1
            continue

        # ── Purity: any intra-sentential gap >= PAUSE_MIN disqualifies the turn
        has_intra = any(
            not turn_words[k]['sent_end']
            and turn_words[k + 1]['start'] - turn_words[k]['end'] >= INTRA_PAUSE_MIN
            for k in range(len(turn_words) - 1)
        )
        if has_intra:
            stats['rej_mixed_pauses'] += 1
            continue

        # ── Find all inter-sentential pauses ───────────────────────────────────
        pauses        = []
        has_large_gap = False
        for k in range(len(turn_words) - 1):
            if turn_words[k]['sent_end']:
                gap = turn_words[k + 1]['start'] - turn_words[k]['end']
                if gap < INTER_PAUSE_MIN:
                    continue
                if gap > INTER_PAUSE_MAX:
                    has_large_gap = True
                    break
                pauses.append({
                    'word_idx':    k,
                    'pause_start': round(turn_words[k]['end'], 3),
                    'pause_end':   round(turn_words[k + 1]['start'], 3),
                    'duration':    round(gap, 3),
                })
        if has_large_gap:
            stats['rej_pre_gap_too_large'] += 1
            continue
        if not pauses:
            stats['rej_no_pauses'] += 1
            continue

        # ── Words before first pause >= MIN_SUBSTANTIVE_WORDS ─────────────────
        first_pause_k = pauses[0]['word_idx']
        if first_pause_k + 1 < MIN_SUBSTANTIVE_WORDS:
            stats['rej_before_short'] += 1
            continue

        # ── Words after last pause >= MIN_SUBSTANTIVE_WORDS ───────────────────
        last_pause_k = pauses[-1]['word_idx']
        words_after = len(turn_words) - (last_pause_k + 1)
        if words_after < MIN_SUBSTANTIVE_WORDS:
            stats['rej_after_short'] += 1
            continue

        # ── Build full transcript with pause markers ───────────────────────────
        parts = []
        seg_start = 0
        for p in pauses:
            k = p['word_idx']
            parts.append(' '.join(w['word'] for w in turn_words[seg_start:k + 1]))
            parts.append(f"[PAUSE {p['duration']:.2f}s]")
            seg_start = k + 1
        parts.append(' '.join(w['word'] for w in turn_words[seg_start:]))
        full_transcript = ' '.join(parts)

        # Split at first [PAUSE ...] to get pre/post pause text
        _pre, _post = _split_at_pause_marker(full_transcript)

        prev_row = turns.iloc[i - 1] if i > 0 else None
        stats['passed'] += 1
        candidates.append({
            'type':             'inter_sentential',
            'conv_id':          conv_id,
            'speaker':          spk,
            'turn_id':          int(turn_row['turn_id']),
            'n_pauses':         len(pauses),
            'clip_start':       round(clip_start, 3),
            'clip_end':         round(clip_end, 3),
            'prev_turn_end':    round(float(prev_row['stop']), 3) if prev_row is not None else None,
            'next_turn_start':  round(float(next_row['start']), 3) if next_row is not None else None,
            'pauses':           pauses,
            'full_transcript':  full_transcript,
            'pre_pause_text':   _pre,
            'post_pause_text':  _post,
            'gpt_valid':        None,
            'gpt_reason':       None,
            'gpt_details':      None,
        })

    return candidates, stats


def find_floor_holding(back: pd.DataFrame,
                       speaker_words: dict[str, list]) -> tuple[list[dict], dict]:
    """
    Turns where the speaker holds the floor without genuine interruption.
    Any word-level gap >= PAUSE_MIN disqualifies (consistent with AMI / CORAAL).
    Returns (candidates, timing_stats).
    """
    candidates = []
    turns = back.reset_index(drop=True)
    conv_id = back.attrs.get('conv_id', '')
    stats = dict(
        total_turns        = len(turns),
        rej_too_short      = 0,   # n_words < FLOOR_HOLD_MIN_WORDS
        rej_duration_short = 0,   # duration < FLOOR_HOLD_MIN_DUR
        rej_no_words       = 0,   # no words found in turn window
        rej_trim_failed    = 0,   # trim_short_sentences returned None
        rej_trim_short     = 0,   # trimmed word count < FLOOR_HOLD_MIN_WORDS
        rej_trim_dur_short = 0,   # trimmed duration < FLOOR_HOLD_MIN_DUR
        rej_overlap        = 0,   # next turn starts before clip_end
        rej_other_speech   = 0,   # non-backchannel speaker inside clip
        rej_has_pauses     = 0,   # turn contains gap >= PAUSE_MIN
        passed             = 0,
    )

    for i, row in turns.iterrows():
        dur = float(row['stop']) - float(row['start'])
        if int(row.get('n_words', 0)) < FLOOR_HOLD_MIN_WORDS:
            stats['rej_too_short'] += 1
            continue
        if dur < FLOOR_HOLD_MIN_DUR:
            stats['rej_duration_short'] += 1
            continue

        spk = row['speaker']
        turn_start = float(row['start'])
        turn_stop  = float(row['stop'])

        raw_words = [w for w in speaker_words.get(spk, [])
                     if turn_start <= w['start'] and w['end'] <= turn_stop]
        if not raw_words:
            stats['rej_no_words'] += 1
            continue

        trimmed = trim_short_sentences(raw_words)
        if trimmed is None:
            stats['rej_trim_failed'] += 1
            continue
        trimmed_words, clip_start, clip_end = trimmed

        if len(trimmed_words) < FLOOR_HOLD_MIN_WORDS:
            stats['rej_trim_short'] += 1
            continue
        if clip_end - clip_start < FLOOR_HOLD_MIN_DUR:
            stats['rej_trim_dur_short'] += 1
            continue

        next_row = turns.iloc[i + 1] if i + 1 < len(turns) else None
        if next_row is not None and float(next_row['start']) < clip_end - MAX_OVERLAP:
            stats['rej_overlap'] += 1
            continue

        if has_other_speech_in(spk, clip_start, clip_end, back):
            stats['rej_other_speech'] += 1
            continue

        # Any gap >= PAUSE_MIN disqualifies (purity — consistent with AMI/CORAAL)
        has_pause = any(
            trimmed_words[k + 1]['start'] - trimmed_words[k]['end'] >= PAUSE_MIN
            for k in range(len(trimmed_words) - 1)
        )
        if has_pause:
            stats['rej_has_pauses'] += 1
            continue

        utt = ' '.join(w['word'] for w in trimmed_words)
        prev_turn = turns.iloc[i - 1] if i > 0 else None
        stats['passed'] += 1
        candidates.append({
            'type':              'floor_holding',
            'conv_id':           conv_id,
            'speaker':           spk,
            'turn_id':           int(row['turn_id']),
            'clip_start':        round(clip_start, 3),
            'clip_end':          round(clip_end, 3),
            'duration':          round(clip_end - clip_start, 3),
            'n_words':           len(trimmed_words),
            'backchannel_count': int(row.get('backchannel_count', 0)),
            'prev_turn_end':     round(float(prev_turn['stop']), 3) if prev_turn is not None else None,
            'next_turn_start':   round(float(next_row['start']), 3) if next_row is not None else None,
            'utterance':         utt,
            'full_transcript':   utt,
            'gpt_valid':         None,
            'gpt_reason':        None,
        })

    return candidates, stats


def find_intra_sentential(speaker_words: dict[str, list],
                          back: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Intra-sentential pauses within individual sentences of a backbiter turn.
    One candidate per qualifying sentence. Returns (candidates, timing_stats).
    """
    conv_id = back.attrs.get('conv_id', '')
    turns   = back.reset_index(drop=True)
    candidates = []
    seen: set = set()
    stats = dict(
        total_turns           = len(turns),
        total_sentences       = 0,
        rej_no_words          = 0,   # no words found in turn window
        rej_trim_failed       = 0,   # trim_short_sentences returned None
        rej_overlap           = 0,   # next turn starts before clip_end
        rej_sentence_short    = 0,   # sentence < MIN_TURN_WORDS
        rej_sentence_large_gap= 0,   # gap > INTRA_PAUSE_MAX within sentence
        rej_no_pauses         = 0,   # sentence has no qualifying pause
        rej_before_short      = 0,   # words before first pause < MIN_SUBSTANTIVE_WORDS
        rej_after_short       = 0,   # words after last pause  < MIN_SUBSTANTIVE_WORDS
        passed                = 0,
    )

    for i, turn_row in turns.iterrows():
        spk        = turn_row['speaker']
        turn_start = float(turn_row['start'])
        turn_stop  = float(turn_row['stop'])

        raw_words = [w for w in speaker_words.get(spk, [])
                     if turn_start <= w['start'] and w['end'] <= turn_stop]
        if not raw_words:
            stats['rej_no_words'] += 1
            continue

        result = trim_short_sentences(raw_words)
        if result is None:
            stats['rej_trim_failed'] += 1
            continue
        turn_words, _, clip_end = result

        # ── Overlap check ─────────────────────────────────────────────────────
        next_row = turns.iloc[i + 1] if i + 1 < len(turns) else None
        if next_row is not None and float(next_row['start']) < clip_end - MAX_OVERLAP:
            stats['rej_overlap'] += 1
            continue

        # ── Split clean turn into sentences ────────────────────────────────────
        sentences: list[list[dict]] = []
        current: list[dict] = []
        for w in turn_words:
            current.append(w)
            if w['sent_end']:
                sentences.append(current)
                current = []
        # trailing fragment without closing punctuation is discarded

        prev_row = turns.iloc[i - 1] if i > 0 else None

        for sent_words in sentences:
            stats['total_sentences'] += 1

            if len(sent_words) < MIN_TURN_WORDS:
                stats['rej_sentence_short'] += 1
                continue

            # Find ALL intra-sentential pauses in this sentence
            pauses        = []
            has_large_gap = False
            for k in range(len(sent_words) - 1):
                gap = sent_words[k + 1]['start'] - sent_words[k]['end']
                if gap < INTRA_PAUSE_MIN:
                    continue
                if gap > INTRA_PAUSE_MAX:
                    has_large_gap = True
                    break
                pauses.append({
                    'word_idx':    k,
                    'pause_start': round(sent_words[k]['end'], 3),
                    'pause_end':   round(sent_words[k + 1]['start'], 3),
                    'duration':    round(gap, 3),
                })
            if has_large_gap:
                stats['rej_sentence_large_gap'] += 1
                continue
            if not pauses:
                stats['rej_no_pauses'] += 1
                continue

            # ── Words before first pause >= MIN_SUBSTANTIVE_WORDS ─────────────
            first_pause_k = pauses[0]['word_idx']
            if first_pause_k + 1 < MIN_SUBSTANTIVE_WORDS:
                stats['rej_before_short'] += 1
                continue

            # ── Words after last pause >= MIN_SUBSTANTIVE_WORDS ───────────────
            last_pause_k = pauses[-1]['word_idx']
            words_after = len(sent_words) - (last_pause_k + 1)
            if words_after < MIN_SUBSTANTIVE_WORDS:
                stats['rej_after_short'] += 1
                continue

            clip_start = sent_words[0]['start']
            clip_end_s = sent_words[-1]['end']

            key = (spk, round(clip_start, 2))
            if key in seen:
                continue
            seen.add(key)

            # ── Build transcript with pause markers ────────────────────────────
            parts = []
            seg_start = 0
            for p in pauses:
                k = p['word_idx']
                parts.append(' '.join(w['word'] for w in sent_words[seg_start:k + 1]))
                parts.append(f"[PAUSE {p['duration']:.2f}s]")
                seg_start = k + 1
            parts.append(' '.join(w['word'] for w in sent_words[seg_start:]))
            full_transcript = ' '.join(parts)

            _pre, _post = _split_at_pause_marker(full_transcript)
            stats['passed'] += 1
            candidates.append({
                'type':             'intra_sentential',
                'conv_id':          conv_id,
                'speaker':          spk,
                'turn_id':          int(turn_row['turn_id']),
                'n_pauses':         len(pauses),
                'clip_start':       round(clip_start, 3),
                'clip_end':         round(clip_end_s, 3),
                'prev_turn_end':    round(float(prev_row['stop']), 3) if prev_row is not None else None,
                'next_turn_start':  round(float(next_row['start']), 3) if next_row is not None else None,
                'pauses':           pauses,
                'full_transcript':  full_transcript,
                'pre_pause_text':   _pre,
                'post_pause_text':  _post,
                'gpt_valid':        None,
                'gpt_reason':       None,
                'gpt_details':      None,
            })

    return candidates, stats


# ── Step 2: GPT Completeness Filter ───────────────────────────────────────────

INTER_PROMPT = """\
You are evaluating a natural conversation segment for research on spoken dialogue systems.

A speaker holds the floor across {n_sents} sentences separated by {n_pauses} pause(s):

{transcript}

Answer three questions:

1. COMPLETE SENTENCES: Are all sentences complete and uninterrupted?
   - Each sentence must be a full thought, not a fragment or a filler phrase.
   - If the speaker appears to have been cut off mid-sentence, answer false.

2. NATURAL PAUSES: Are the pauses positioned naturally at sentence boundaries?
   - Each pause must fall after a substantive sentence, not after a 1–2 word fragment.
   - A sentence starting with a conjunction like "and" / "but" after a pause is acceptable
     only if the preceding sentence is complete on its own.

3. SELF-CONTAINED: Can this segment be understood and responded to WITHOUT any prior context?
   - It should introduce its own topic or information clearly enough to stand alone.
   - Reject if it heavily relies on something said earlier (e.g. starts with "so as I said",
     refers to unnamed "it/that/they" whose referent is unknown, or is a direct answer to a
     question not present in the segment).
   - A segment that could serve as a natural conversation opener or a standalone statement
     should be accepted.

Respond ONLY with JSON (no markdown):
{{"complete_sentences": true/false, "natural_pause": true/false, "self_contained": true/false, "is_valid": true/false, "reason": "one sentence"}}"""

INTRA_PROMPT = """\
You are evaluating a natural conversation segment for research on spoken dialogue systems.

A speaker pauses {n_pauses} time(s) within a single sentence:

{transcript}

Answer three questions:

1. COMPLETE SENTENCE: Is this a complete, uninterrupted sentence?
   - The full sentence must be a coherent, finished thought (not cut off mid-speech).
   - If the speaker appears to have been interrupted, answer false.

2. NATURAL PAUSES: Are the pauses in meaningful mid-sentence positions?
   - Words before each pause must form a substantive phrase (not just a filler word).
   - Words after each pause must continue the sentence naturally.
   - If a pause falls AFTER a connective word like "and" or "but" (i.e. the connective comes
     just before the pause marker), this is acceptable — the speaker said the connective and
     then hesitated before continuing.
   - If a pause falls BEFORE a connective word like "and" or "but" (i.e. the connective is
     the first word after the pause marker), AND the words before the pause already form a
     complete, standalone sentence on their own, then this is NOT acceptable — it reads more
     like an inter-sentential boundary than a genuine mid-sentence hesitation.

3. SELF-CONTAINED: Can this sentence be understood and responded to WITHOUT any prior context?
   - It should be interpretable on its own, without needing to know what was said before.
   - Reject if it refers to unnamed "it/that/they" whose referent is unknown, or is clearly
     a fragment of a longer answer to an unseen question.

Respond ONLY with JSON (no markdown):
{{"complete_sentence": true/false, "natural_pause": true/false, "self_contained": true/false, "is_valid": true/false, "reason": "one sentence"}}"""

FLOOR_PROMPT = """\
You are evaluating a natural conversation segment for research on spoken dialogue systems.

A speaker held the floor for the following utterance (the other speaker only inserted brief backchannels):

Utterance: "{utterance}"

Evaluate:
1. Is this utterance semantically complete and meaningful?
2. Does it represent a coherent, self-contained speech act (not a fragment)?
3. SELF-CONTAINED: Can this utterance be understood and responded to WITHOUT any prior context?
   - It should introduce its own topic or content clearly enough to stand alone.
   - Reject if it heavily relies on something said earlier (e.g. refers to unnamed "it/that/they"
     whose referent is unknown, or is clearly a continuation of an answer to an unseen question).
4. Overall, is this a valid floor-holding sample suitable for benchmarking a dialogue model?

Respond ONLY with JSON (no markdown):
{{"is_complete": true/false, "is_coherent": true/false, "self_contained": true/false, "is_valid": true/false, "reason": "one sentence"}}"""


def build_gpt_message(candidate: dict) -> str:
    t = candidate['type']
    n = candidate.get('n_pauses', 1)
    if t == 'inter_sentential':
        return INTER_PROMPT.format(
            n_sents=n + 1,
            n_pauses=n,
            transcript=candidate['full_transcript'],
        )
    elif t == 'intra_sentential':
        return INTRA_PROMPT.format(
            n_pauses=n,
            transcript=candidate['full_transcript'],
        )
    else:
        return FLOOR_PROMPT.format(utterance=candidate['full_transcript'])


class QuotaExceededError(RuntimeError):
    """Raised when the OpenAI API returns an insufficient_quota (429) error."""
    pass


def gpt_filter(candidates: list[dict], client: OpenAI,
               retry_nulls_only: bool = False,
               max_valid: int | None = None) -> tuple[list[dict], dict]:
    """Send each candidate to GPT for completeness/validity check.

    Stops early once `max_valid` validated samples have been found.
    Returns (candidates, gpt_stats).
    """
    pending = [c for c in candidates if not retry_nulls_only or c.get('gpt_valid') is None]
    total   = len(pending)
    n_valid = 0
    gpt_stats = dict(total_sent=total, valid=0, rejected=0, criteria={})

    print(f"  Sending {total} candidates to GPT ({GPT_MODEL})"
          + (f" (cap={max_valid})" if max_valid else "") + "...")
    for i, cand in enumerate(pending):
        if max_valid is not None and n_valid >= max_valid:
            print(f"    [{i+1}/{total}] cap reached ({max_valid} valid) — skipping rest")
            break
        msg = build_gpt_message(cand)
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": msg}],
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
                raise QuotaExceededError(
                    f"OpenAI quota exceeded at candidate {i} "
                    f"(type={cand.get('type')}, conv={cand.get('conv_id')})"
                ) from e
            print(f"    [WARN] GPT call failed for candidate {i}: {e}")
            cand['gpt_valid']  = None
            cand['gpt_reason'] = f"API error: {e}"
            continue
        if cand['gpt_valid']:
            n_valid += 1
            gpt_stats['valid'] += 1
        else:
            gpt_stats['rejected'] += 1
            for key, val in (cand.get('gpt_details') or {}).items():
                if val is False:
                    gpt_stats['criteria'][key] = gpt_stats['criteria'].get(key, 0) + 1
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{total} done  ({n_valid} valid so far)")

    return candidates, gpt_stats


# ── Step 3: Audio Export ───────────────────────────────────────────────────────

def export_audio_clip(audio: np.ndarray, sr: int, start_s: float, end_s: float,
                      out_path: Path, ch_idx: int = 0,
                      padding: float = AUDIO_PADDING,
                      prev_turn_end: float | None = None,
                      next_turn_start: float | None = None) -> str:
    """
    Slice [start_s - padding, end_s + padding] — capped so padding never
    crosses into the adjacent turns — and save one WAV file (speaker's
    single channel only).

    Returns the file path as a string.
    """
    pad_start = start_s - padding
    pad_end   = end_s   + padding

    # Cap padding at adjacent turn boundaries so we never bleed into them
    if prev_turn_end is not None:
        pad_start = max(pad_start, prev_turn_end)
    if next_turn_start is not None:
        pad_end = min(pad_end, next_turn_start)

    start_sample = max(0, int(pad_start * sr))
    end_sample   = min(audio.shape[-1], int(pad_end * sr))

    if audio.ndim == 2:
        ch_clip = audio[ch_idx, start_sample:end_sample]
    else:
        ch_clip = audio[start_sample:end_sample]

    sf.write(str(out_path), ch_clip, sr)
    return str(out_path)


def export_candidates(candidates: list[dict], audio: np.ndarray, sr: int,
                      audio_dir: Path, pause_type: str,
                      spk_to_ch: dict[str, int],
                      results_dir: Path | None = None) -> None:
    """Export audio only for GPT-validated candidates.

    For pause types with a pause count (inter/intra sentential), clips are
    placed in subdirectories by number of pauses:
      audio/intra_sentential/n1/   ← 1 pause
      audio/intra_sentential/n2/   ← 2 pauses
      ...
    Floor-holding clips go directly into audio/floor_holding/.

    When results_dir is provided the same files are also copied there:
      results_dir/<pause_type>/n<k>/   or   results_dir/floor_holding/
    """
    import shutil

    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue

        conv_id = cand.get('conv_id', 'unknown')
        spk     = cand.get('speaker', '')
        ch_idx  = spk_to_ch.get(spk, 0)
        n       = cand.get('n_pauses', 1)

        if pause_type == 'inter_sentential':
            out_dir = audio_dir / pause_type
            base    = f"{conv_id}_inter_{round(cand['clip_start'], 2)}"
        elif pause_type == 'intra_sentential':
            out_dir = audio_dir / pause_type
            base    = f"{conv_id}_intra_{round(cand['clip_start'], 2)}"
        else:
            out_dir = audio_dir / pause_type
            base    = f"{conv_id}_floor_tid{cand['turn_id']}"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{base}.wav"
        export_audio_clip(audio, sr, cand['clip_start'], cand['clip_end'],
                          out_path, ch_idx=ch_idx,
                          prev_turn_end=cand.get('prev_turn_end'),
                          next_turn_start=cand.get('next_turn_start'))
        cand['audio_file'] = str(out_path)

        if results_dir is not None:
            global_dir = results_dir / pause_type
            global_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, global_dir / out_path.name)
            # Write metadata JSON alongside the WAV (same stem, .json extension)
            meta = {k: cand[k] for k in (
                'type', 'conv_id', 'speaker', 'turn_id', 'n_pauses',
                'clip_start', 'clip_end', 'prev_turn_end', 'next_turn_start',
                'pauses', 'full_transcript',
                'pre_pause_text', 'post_pause_text',
                'gpt_valid', 'gpt_reason', 'gpt_details',
            ) if k in cand}
            meta['audio_full'] = str(global_dir / out_path.name)
            json_path = global_dir / (out_path.stem + '.json')
            with open(json_path, 'w') as _jf:
                json.dump(meta, _jf, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def save_results(data: list[dict], path: Path, label: str) -> None:
    path = path.with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [{label}] Saved {len(data)} entries → {path.name}")


def main(conv_dir: Path, gpt_only: bool = False,
         results_dir: Path | None = None,
         skip_gpt: bool = False,
         skip_types: set[str] | None = None) -> None:
    """
    skip_types: set of type names to exclude from GPT judging and audio export
                (e.g. {'inter_sentential'} when that type's global cap is reached).
    """
    conv_id    = conv_dir.name
    output_dir = conv_dir / "pause_candidates"
    f1_dir     = output_dir / "filter1_timing"
    f2_dir     = output_dir / "filter2_gpt"
    audio_path = conv_dir / "processed" / f"{conv_id}.mp3"
    chmap_path = conv_dir / "processed" / "channel_map.json"

    timing_stats: dict[str, dict] = {}

    if gpt_only:
        # ── GPT-only mode: reload existing filter1 results ─────────────────────
        print(f"[gpt_only] Loading existing filter1 results for {conv_id}...")
        inter = json.load(open(f1_dir / "inter_sentential.json"))
        floor = json.load(open(f1_dir / "floor_holding.json"))
        intra = json.load(open(f1_dir / "intra_sentential.json"))

        if (f2_dir / "all_with_gpt_decisions.json").exists():
            prev = {(c['type'], c.get('clip_start'), c.get('turn_id')): c
                    for c in json.load(open(f2_dir / "all_with_gpt_decisions.json"))}
            for lst in (inter, floor, intra):
                for c in lst:
                    key = (c['type'], c.get('clip_start'), c.get('turn_id'))
                    if key in prev and prev[key].get('gpt_valid') is not None:
                        c['gpt_valid']   = prev[key]['gpt_valid']
                        c['gpt_reason']  = prev[key]['gpt_reason']
                        c['gpt_details'] = prev[key].get('gpt_details')

        chmap  = json.load(open(chmap_path))
        json_path = conv_dir / "transcription" / "transcribe_output.json"
        trans_json = json.load(open(json_path))
        ch_to_speaker: dict[str, str] = {
            "ch_0": chmap["L"],
            "ch_1": chmap["R"],
        }
        spk_to_ch = {spk: int(key.split('_')[1]) for key, spk in ch_to_speaker.items()}

    else:
        # ── Full mode: wipe and reprocess from scratch ─────────────────────────
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"Deleted previous results in {output_dir.relative_to(conv_dir)}/")
        output_dir.mkdir(exist_ok=True)

        back_path  = conv_dir / "transcription" / "transcript_backbiter.csv"
        json_path  = conv_dir / "transcription" / "transcribe_output.json"

        print("Loading transcripts...")
        back  = pd.read_csv(back_path)
        back.attrs['conv_id'] = conv_id

        trans_json = json.load(open(json_path))
        chmap      = json.load(open(chmap_path))
        ch_to_speaker = {
            "ch_0": chmap["L"],
            "ch_1": chmap["R"],
        }
        print(f"  Channel mapping: {ch_to_speaker}")
        spk_to_ch = {spk: int(key.split('_')[1]) for key, spk in ch_to_speaker.items()}

        print("Building per-speaker word list...")
        speaker_words = build_speaker_words(trans_json, ch_to_speaker)
        for spk, wds in speaker_words.items():
            print(f"  {spk}: {len(wds)} words")

        print("\n── Step 1: Timing filters ──")
        inter, inter_stats = find_inter_sentential(speaker_words, back)
        floor, floor_stats = find_floor_holding(back, speaker_words)
        intra, intra_stats = find_intra_sentential(speaker_words, back)
        timing_stats = {
            'inter_sentential': inter_stats,
            'floor_holding':    floor_stats,
            'intra_sentential': intra_stats,
        }

        print(f"  inter_sentential : {len(inter):4d}  (passed timing)")
        print(f"  floor_holding    : {len(floor):4d}  (passed timing)")
        print(f"  intra_sentential : {len(intra):4d}  (passed timing)")

        # Machine-parseable lines consumed by batch script
        for cat, st in timing_stats.items():
            print(f"SEND_TO_LLM {cat}: {st['passed']}", flush=True)
        print(f"TIMING_STATS:{json.dumps(timing_stats)}", flush=True)

        print("\nSaving Filter 1 results (timing filter)...")
        save_results(inter, f1_dir / "inter_sentential.json", "inter")
        save_results(floor, f1_dir / "floor_holding.json",    "floor")
        save_results(intra, f1_dir / "intra_sentential.json", "intra")
        save_results(inter + floor + intra, f1_dir / "all.json", "all")

        if skip_gpt:
            print("\n[skip_gpt] Timing-only run finished.")
            return

    # ── Step 2: GPT completeness filter ───────────────────────────────────────
    print(f"\n── Step 2: GPT filter ({GPT_MODEL}) ──")
    client = OpenAI(api_key=OPENAI_API_KEY)
    gpt_all: dict[str, dict] = {}

    _skip = skip_types or set()
    for cat, lst in (('inter_sentential', inter), ('floor_holding', floor), ('intra_sentential', intra)):
        if cat in _skip:
            print(f"  [{cat}] SKIPPED (global cap reached)", flush=True)
            gpt_all[cat] = {'total_sent': 0, 'valid': 0, 'rejected': 0, 'criteria': {}}
            continue
        print(f"  [{cat}]")
        _, cat_gpt_stats = gpt_filter(lst, client, retry_nulls_only=gpt_only,
                                      max_valid=TARGET_PER_CATEGORY)
        gpt_all[cat] = cat_gpt_stats
        n_valid = sum(1 for c in lst if c.get('gpt_valid'))
        print(f"    valid: {n_valid}  (cap={TARGET_PER_CATEGORY})", flush=True)

    inter_valid = [c for c in inter if c.get('gpt_valid')]
    floor_valid = [c for c in floor if c.get('gpt_valid')]
    intra_valid = [c for c in intra if c.get('gpt_valid')]

    print(f"\n  Valid after GPT filter:")
    print(f"    inter_sentential : {len(inter_valid)}/{len(inter)}")
    print(f"    floor_holding    : {len(floor_valid)}/{len(floor)}")
    print(f"    intra_sentential : {len(intra_valid)}/{len(intra)}")

    # Machine-parseable lines consumed by batch script
    for cat, st in gpt_all.items():
        print(f"VALID_AFTER_LLM {cat}: {st['valid']}", flush=True)
    print(f"GPT_STATS:{json.dumps(gpt_all)}", flush=True)

    print("\nSaving Filter 2 results (GPT filter)...")
    save_results(inter, f2_dir / "inter_sentential.json", "inter (all+decisions)")
    save_results(floor, f2_dir / "floor_holding.json",    "floor (all+decisions)")
    save_results(intra, f2_dir / "intra_sentential.json", "intra (all+decisions)")
    save_results(inter + floor + intra, f2_dir / "all_with_gpt_decisions.json", "all+decisions")

    # ── Step 3: Audio export ───────────────────────────────────────────────────
    print(f"\n── Step 3: Audio export ──")
    print(f"Loading audio from {audio_path.name} (this may take ~30s)...")
    audio, sr = librosa.load(str(audio_path), sr=AUDIO_SR, mono=False)
    print(f"  Audio loaded: {audio.shape}, sr={sr}, duration={audio.shape[-1]/sr:.1f}s")

    audio_dir = output_dir / "audio"
    if 'inter_sentential' not in _skip:
        export_candidates(inter, audio, sr, audio_dir, 'inter_sentential', spk_to_ch, results_dir)
    if 'floor_holding' not in _skip:
        export_candidates(floor, audio, sr, audio_dir, 'floor_holding',    spk_to_ch, results_dir)
    if 'intra_sentential' not in _skip:
        export_candidates(intra, audio, sr, audio_dir, 'intra_sentential', spk_to_ch, results_dir)
    if results_dir:
        print(f"  Mirrored to global results → {results_dir}")

    total_clips = len(inter_valid) + len(floor_valid) + len(intra_valid)
    print(f"  Exported {total_clips} audio clips → {audio_dir.relative_to(conv_dir)}/")

    print("\n── Summary ──")
    print(f"  Output directory: {output_dir.relative_to(conv_dir)}/")
    print(f"  filter1_timing/  ← transcripts after timing filter only")
    print(f"  filter2_gpt/     ← transcripts after GPT filter (+ GPT decisions)")
    print(f"  audio/           ← .wav clips for GPT-validated candidates only")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pause candidates from CANDOR conversation.")
    parser.add_argument("--conv_dir",    type=Path, default=DEFAULT_CONV_DIR,
                        help="Path to a single CANDOR conversation directory")
    parser.add_argument("--gpt_only",    action="store_true",
                        help="Skip timing filter; re-run GPT+audio on existing filter1 results")
    parser.add_argument("--skip_gpt",   action="store_true",
                        help="Run timing filter only; do not call GPT or export audio")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Global results directory — audio and JSON mirrored here")
    parser.add_argument("--skip_types", nargs="*", default=[],
                        choices=["inter_sentential", "intra_sentential", "floor_holding"],
                        help="Sub-action types to skip GPT judging and audio export for "
                             "(used by batch script when global cap is already reached)")
    args = parser.parse_args()
    try:
        main(args.conv_dir, gpt_only=args.gpt_only,
             results_dir=args.results_dir, skip_gpt=args.skip_gpt,
             skip_types=set(args.skip_types))
    except QuotaExceededError as e:
        print(f"\n[QUOTA EXCEEDED] {e}", flush=True)
        raise SystemExit(2)
