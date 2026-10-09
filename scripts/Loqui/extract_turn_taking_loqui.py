#!/usr/bin/env python3
"""
Loqui Turn-Taking Extractor

Extracts 2-turn turn-taking pairs (Speaker A → Speaker B) from a Loqui
recording. Turn 1 is used alone as the model-side input, so it must be a
clean, complete utterance. Turn 2 confirms that a genuine response followed.

Two structural patterns are detected:

  Pattern A — Sequential (no overlap turn between T1 and T2):
    T1 (single A) → T2 (single B)
    Categories: take_turn_immediately  (0 ≤ gap ≤ 200 ms)
                take_turn_with_latency (200 ms < gap ≤ 3.0 s)

  Pattern B — Overlap via overlap turn:
    T1 (single A) → overlap(A + B) → T2 (single B)
    Category:       take_turn_with_overlap
    B's Who-block in the overlap turn must be ≥ MIN_SUBSTANTIVE_WORDS (≥ 4)
    to distinguish it from a backchannel (≤ 3 words).

Note: No sentence-final punctuation check is applied (Loqui has no
punctuation). GPT's turn1_complete criterion compensates.

All numerical thresholds are identical to CORAAL:
  MIN_SUBSTANTIVE_WORDS = 4
  IMMEDIATE_MAX_GAP     = 0.2 s
  MAX_LATENCY_GAP       = 3.0 s
  MAX_OVERLAP           = 1.5 s
  PAUSE_MIN / PAUSE_MAX = 0.2 / 1.0 s

Pipeline:
  Step 1 – Timing filter : find candidate pairs from .trs transcript
  Step 2 – GPT filter    : verify content quality and category appropriateness
  Step 3 – Audio export  : stereo .san.wav → per-speaker mono clips

Usage:
  python extract_turn_taking_loqui.py \\
      --file_id 37 \\
      --trs_path .../37.final.trs \\
      --wav_path .../37.san.wav \\
      --results_dir .../results/turn_taking_loqui
"""

import argparse
import os
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

try:
    import librosa
    import numpy as np
    import soundfile as sf
    from openai import OpenAI
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

import xml.etree.ElementTree as ET

# ── Configuration (identical to CORAAL) ────────────────────────────────────────

TIMESTAMP_NOISE_TOLERANCE = 0.05   # annotation imprecision slack (s)
PAUSE_MIN             = 0.2        # min pause duration to flag as disfluency
PAUSE_MAX             = 1.0        # max pause duration to flag (above = recording gap)
MIN_SUBSTANTIVE_WORDS = 4          # T1 and T2 minimum words
IMMEDIATE_MAX_GAP     = 0.2        # gap ≤ 200 ms → take_turn_immediately
MAX_LATENCY_GAP       = 3.0        # gap ≤ 3.0 s  → take_turn_with_latency
MAX_OVERLAP           = 1.5        # overlap turn duration ≤ 1.5 s → valid overlap
AUDIO_SR              = 16000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL      = "gpt-5.4-2026-03-05"
QUOTA_EXIT_CODE = 2

TARGET_PER_CATEGORY = 500

CATEGORIES = (
    "take_turn_immediately",
    "take_turn_with_latency",
    "take_turn_with_overlap",
)

# ── Loqui-specific tag regexes ──────────────────────────────────────────────────

_PAUSE_RE = re.compile(r'<P>', re.IGNORECASE)

_NON_STANDARD_RE = re.compile(
    r'<(?:BN|VB|IR|CHKL|NU|NS|G|ST|TN|TP|TC|CT)[^>]*>', re.IGNORECASE)
_PII_RE = re.compile(
    r'<(?:LASTNAME|FIRSTNAME|FULLNAME|ADDRESS[^>]*|STREETNAME|TOWN|'
    r'ZIPCODE|PHONENUMBER|LN-[^>]*|FN-[^>]*)>',
    re.IGNORECASE)
_STRIP_RE = re.compile(
    r'\[os\b|\beo\]|\boe\]|<[Bb]>', re.IGNORECASE)

# Matches caller/librarian but excludes non-speaking-*-activity participants
_VALID_SPKR_RE = re.compile(
    r'^(?!non-speaking)\S*\b(?:caller|librarian)\b', re.IGNORECASE)

# ── Text helpers ───────────────────────────────────────────────────────────────

def turn_timed(start: float, end: float) -> str:
    """Return a coarse timing string for a turn (TRS has no word-level timestamps)."""
    return f"[{start:.3f}s – {end:.3f}s]"


def _strip_raw(text: str) -> str:
    t = re.sub(r'\\\\', ' ', text)
    return re.sub(r'\s+', ' ', t).strip()


def clean_text(raw: str) -> str:
    t = _PAUSE_RE.sub(' ', raw)
    t = _STRIP_RE.sub(' ', t)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'&\w+;', ' ', t)
    t = re.sub(r'\\\\', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def count_words(raw: str) -> int:
    return len(clean_text(raw).split())


def has_non_standard(raw: str) -> bool:
    return bool(_NON_STANDARD_RE.search(raw) or _PII_RE.search(raw))


# ── TRS parser ─────────────────────────────────────────────────────────────────

def _collect_single_text(turn_el) -> str:
    chunks: list[str] = []
    buf = _strip_raw(turn_el.text or '')
    for child in turn_el:
        if child.tag == 'Sync':
            if buf:
                chunks.append(buf)
            buf = _strip_raw(child.tail or '')
        else:
            tail = _strip_raw(child.tail or '')
            if tail:
                buf += (' ' + tail) if buf else tail
    if buf:
        chunks.append(buf)
    return ' '.join(c for c in chunks if c)


def parse_trs_full(path: Path) -> tuple[dict, list[dict]]:
    """
    Parse a .final.trs file into ALL valid-speaker turns — both
    single-speaker and multi-speaker overlap turns — in chronological order.

    Single-speaker turn  (is_overlap=False):
        spkr_id, spkr_name, turn_start, turn_end, raw_text

    Multi-speaker overlap turn  (is_overlap=True):
        speakers, turn_start, turn_end,
        who_blocks: {spkr_id → raw_text}  (from <Who nb="N" /> tail)
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        print(f"  WARN: XML parse error in {path.name}: {e}")
        return {}, []

    root = tree.getroot()

    valid_spkrs: dict[str, str] = {}
    for el in root.findall('.//Speaker'):
        sid  = el.get('id', '')
        name = el.get('name', '')
        if _VALID_SPKR_RE.search(name):
            valid_spkrs[sid] = name

    turns: list[dict] = []
    for turn_el in root.iter('Turn'):
        t_start  = float(turn_el.get('startTime', 0))
        t_end    = float(turn_el.get('endTime',   0))
        spk_attr = (turn_el.get('speaker') or '').split()

        valid_in = [s for s in spk_attr if s in valid_spkrs]
        if not valid_in:
            continue

        if len(spk_attr) == 1:
            if len(valid_in) != 1:
                continue
            turns.append({
                'is_overlap': False,
                'spkr_id':    valid_in[0],
                'spkr_name':  valid_spkrs[valid_in[0]],
                'turn_start': t_start,
                'turn_end':   t_end,
                'raw_text':   _collect_single_text(turn_el),
            })
        else:
            if set(valid_in) != set(spk_attr):
                continue
            who_blocks: dict[str, str] = {}
            for who_el in turn_el.findall('Who'):
                nb  = int(who_el.get('nb', 1)) - 1
                if 0 <= nb < len(spk_attr):
                    sid = spk_attr[nb]
                    if sid in valid_spkrs:
                        raw = re.sub(r'\s+', ' ',
                                     _strip_raw(who_el.tail or '')).strip()
                        if raw:
                            who_blocks[sid] = raw
            if not who_blocks:
                full = _collect_single_text(turn_el)
                for sid in valid_in:
                    who_blocks[sid] = full
            turns.append({
                'is_overlap': True,
                'speakers':   spk_attr,
                'turn_start': t_start,
                'turn_end':   t_end,
                'who_blocks': who_blocks,
            })

    return valid_spkrs, turns


# ── Step 1: Timing Filter ──────────────────────────────────────────────────────

def find_turn_taking_patterns(turns: list[dict],
                              file_id: str) -> tuple[list[dict], dict]:
    """
    Scan the full turn list for two patterns:

    Pattern A — Sequential (produces immediately / latency):
        turns[i]   : single A  (T1)
        turns[i+1] : single B  (T2, different speaker)
        gap = T2.turn_start − T1.turn_end
          0 ≤ gap ≤ IMMEDIATE_MAX_GAP  → take_turn_immediately
          IMMEDIATE_MAX_GAP < gap ≤ MAX_LATENCY_GAP → take_turn_with_latency

    Pattern B — Overlap turn (produces overlap):
        turns[i]   : single A       (T1)
        turns[i+1] : overlap(A + B) (B's Who-block ≥ MIN_SUBSTANTIVE_WORDS)
        turns[i+2] : single B       (T2)
        overlap_duration = overlap_turn.turn_end − overlap_turn.turn_start
          ≤ MAX_OVERLAP → take_turn_with_overlap

    T1 quality (both patterns):
        no PII / non-standard tags, no <P>, ≥ MIN_SUBSTANTIVE_WORDS

    T2 quality (both patterns):
        ≥ MIN_SUBSTANTIVE_WORDS (following CORAAL: no PII check on T2)
    """
    stats = dict(
        pairs_checked           = 0,
        rej_same_speaker        = 0,
        rej_t2_too_short        = 0,
        rej_t1_too_short        = 0,
        rej_t1_has_pause        = 0,
        rej_t1_anon             = 0,
        rej_t2_words_short      = 0,   # always 0 in Loqui — no word-level timestamps for T2
        rej_gap_too_large       = 0,
        rej_overlap_too_large   = 0,
        # Pattern B specific
        rej_ovl_b_too_short     = 0,
        rej_duplicate           = 0,
        passed                  = 0,
        by_category             = {c: 0 for c in CATEGORIES},
    )

    candidates: list[dict] = []
    seen: set = set()

    for i in range(len(turns) - 1):
        t1 = turns[i]

        # T1 must be single-speaker
        if t1.get('is_overlap'):
            continue

        stats['pairs_checked'] += 1

        # ── T1 quality ────────────────────────────────────────────────────────
        t1_raw = t1['raw_text']
        if has_non_standard(t1_raw):
            stats['rej_t1_anon'] += 1
            continue
        if _PAUSE_RE.search(t1_raw):
            stats['rej_t1_has_pause'] += 1
            continue
        if count_words(t1_raw) < MIN_SUBSTANTIVE_WORDS:
            stats['rej_t1_too_short'] += 1
            continue

        spk_a = t1['spkr_id']

        # ─────────────────────────────────────────────────────────────────────
        # Pattern A: T1(single A) → T2(single B)
        # ─────────────────────────────────────────────────────────────────────
        nxt = turns[i + 1]
        if not nxt.get('is_overlap'):
            spk_b = nxt['spkr_id']
            if spk_b == spk_a:
                stats['rej_same_speaker'] += 1
                continue

            t2_raw = nxt['raw_text']
            if count_words(t2_raw) < MIN_SUBSTANTIVE_WORDS:
                stats['rej_t2_too_short'] += 1
                continue

            gap = nxt['turn_start'] - t1['turn_end']

            if gap < 0:
                # Timestamp annotation noise — treat as immediate if within tolerance
                if abs(gap) <= TIMESTAMP_NOISE_TOLERANCE:
                    gap = 0.0
                else:
                    # Real overlap between consecutive single-speaker turns
                    # is unusual in TRS — skip (Pattern B handles structural overlaps)
                    stats['rej_gap_too_large'] += 1
                    continue

            if gap <= IMMEDIATE_MAX_GAP:
                category = "take_turn_immediately"
            elif gap <= MAX_LATENCY_GAP:
                category = "take_turn_with_latency"
            else:
                stats['rej_gap_too_large'] += 1
                continue

            key = (file_id, spk_a, round(t1['turn_start'], 2), 'seq')
            if key in seen:
                stats['rej_duplicate'] += 1
                continue
            seen.add(key)

            stats['passed'] += 1
            stats['by_category'][category] += 1

            candidates.append({
                'type':       'turn_taking',
                'pattern':    'sequential',
                'file_id':    file_id,
                'spk_a':      spk_a,
                'spk_b':      spk_b,
                'spk_a_name': t1['spkr_name'],
                'spk_b_name': nxt['spkr_name'],

                't1_idx':  i,
                't2_idx':  i + 1,

                't1_start': round(t1['turn_start'],  3),
                't1_end':   round(t1['turn_end'],    3),
                't2_start': round(nxt['turn_start'], 3),
                't2_end':   round(nxt['turn_end'],   3),
                'gap':      round(gap, 3),
                'category': category,

                't1_text':  clean_text(t1_raw),
                't2_text':  clean_text(t2_raw),
                't1_timed': turn_timed(t1['turn_start'], t1['turn_end']),
                't2_timed': turn_timed(nxt['turn_start'], nxt['turn_end']),

                'gpt_valid':            None,
                'turn1_complete':       None,
                'turn1_self_contained': None,
                'turn2_responds':       None,
                'category_appropriate': None,
                'gpt_reason':           None,
            })

        # ─────────────────────────────────────────────────────────────────────
        # Pattern B: T1(single A) → overlap(A + B) → T2(single B)
        # ─────────────────────────────────────────────────────────────────────
        elif nxt.get('is_overlap') and (i + 2) < len(turns):
            ovl = nxt
            t2  = turns[i + 2]

            if t2.get('is_overlap'):
                continue

            # Both A and B must be in the overlap turn
            ovl_spks = ovl.get('speakers', [])
            if spk_a not in ovl_spks:
                continue
            # B is the other speaker in the overlap turn
            other = [s for s in ovl_spks if s != spk_a]
            if len(other) != 1:
                continue
            spk_b = other[0]

            # T2 must be single B
            if t2['spkr_id'] != spk_b:
                continue

            # B's Who-block must be substantive (≥ MIN_SUBSTANTIVE_WORDS)
            # This distinguishes turn-taking from backchannel (≤ 3 words)
            who = ovl.get('who_blocks', {})
            b_raw = who.get(spk_b, '')
            if count_words(b_raw) < MIN_SUBSTANTIVE_WORDS:
                stats['rej_ovl_b_too_short'] += 1
                continue

            # Overlap duration must not exceed MAX_OVERLAP
            ovl_dur = ovl['turn_end'] - ovl['turn_start']
            if ovl_dur > MAX_OVERLAP:
                stats['rej_overlap_too_large'] += 1
                continue

            # T2 word count
            t2_raw = t2['raw_text']
            if count_words(t2_raw) < MIN_SUBSTANTIVE_WORDS:
                stats['rej_t2_too_short'] += 1
                continue

            # spk_b must be a valid speaker
            from_valid = {s: nxt['who_blocks'].get(s, '') for s in ovl_spks}  # noqa
            # Get spk_b_name from the valid_spkrs (need to look it up)
            # We can get it from t2 since t2 is spk_b
            spk_b_name = t2['spkr_name']

            key = (file_id, spk_a, round(t1['turn_start'], 2), 'ovl')
            if key in seen:
                stats['rej_duplicate'] += 1
                continue
            seen.add(key)

            stats['passed'] += 1
            stats['by_category']['take_turn_with_overlap'] += 1

            a_raw_ovl = who.get(spk_a, '')

            candidates.append({
                'type':       'turn_taking',
                'pattern':    'overlap',
                'file_id':    file_id,
                'spk_a':      spk_a,
                'spk_b':      spk_b,
                'spk_a_name': t1['spkr_name'],
                'spk_b_name': spk_b_name,

                't1_idx':  i,
                'ovl_idx': i + 1,
                't2_idx':  i + 2,

                't1_start':  round(t1['turn_start'],  3),
                't1_end':    round(t1['turn_end'],     3),
                'ovl_start': round(ovl['turn_start'],  3),
                'ovl_end':   round(ovl['turn_end'],    3),
                't2_start':  round(t2['turn_start'],   3),
                't2_end':    round(t2['turn_end'],     3),
                'gap':       round(-(ovl_dur), 3),  # negative = overlap
                'ovl_dur':   round(ovl_dur, 3),
                'category':  'take_turn_with_overlap',

                't1_text':      clean_text(t1_raw),
                'ovl_a_text':   clean_text(a_raw_ovl),
                'ovl_b_text':   clean_text(b_raw),
                't2_text':      clean_text(t2_raw),
                't1_timed':     turn_timed(t1['turn_start'],  t1['turn_end']),
                't2_timed':     turn_timed(t2['turn_start'],  t2['turn_end']),

                'gpt_valid':            None,
                'turn1_complete':       None,
                'turn1_self_contained': None,
                'turn2_responds':       None,
                'category_appropriate': None,
                'gpt_reason':           None,
            })

    return candidates, stats


# ── Step 2: GPT Filter ─────────────────────────────────────────────────────────

# Category guidance identical to CORAAL
CATEGORY_GUIDANCE = {
    "take_turn_immediately": (
        "This pair was identified from timing as TAKE-TURN-IMMEDIATELY: Speaker B "
        "began shortly after Speaker A completed the turn, with no meaningful "
        "overlap. Keep it only if the content supports a prompt post-completion "
        "response. Specifically, Turn 1 should already make the main communicative "
        "goal and the general response frame clear before its end, while one or more "
        "final response-critical details remain pending until the final segment. "
        "These final details may include a date, time, person, place, option, "
        "quantity, target entity, polarity, or other slot needed to instantiate an "
        "already anticipated response. Once these details are supplied and Speaker A "
        "finishes, Speaker B can plausibly launch a prepared response immediately "
        "without needing to revise its response strategy.\n\n"
        "Examples that fit: a scheduling request whose date or time is supplied at "
        "the end; a request whose recipient or target item is supplied at the end; "
        "a question whose answer format is already clear but whose final option or "
        "entity is only given in the final phrase.\n\n"
        "Do NOT keep the pair if: (1) the final content changes, reverses, or "
        "substantially constrains what the response should be; (2) the response "
        "target remains genuinely unclear until the end; (3) the turn is merely "
        "simple and received a fast response by chance, without a meaningful final "
        "slot-completion structure; (4) Speaker B's response is a backchannel, "
        "continuation, or non-substantive acknowledgement.\n\n"
        "Key test: before the final detail appears, could Speaker B already prepare "
        "the response structure, while still needing to wait for that detail before "
        "safely speaking? After the final detail appears, would the same response "
        "strategy remain valid without further re-planning?"
    ),
    "take_turn_with_latency": (
        "This pair was identified from timing as TAKE-TURN-WITH-DELAY: Speaker B "
        "began after Speaker A completed the turn, following a noticeable positive "
        "gap. Keep it only if the final portion of Turn 1 creates a genuine "
        "content-based reason why Speaker B could not fully prepare a stable "
        "substantive response before turn completion. The decisive information "
        "should arrive very late, ideally in the final words or final short phrase, "
        "leaving insufficient time for Speaker B to update the response plan before "
        "Speaker A finishes.\n\n"
        "The late information must do more than fill a predictable slot: it should "
        "determine, revise, invalidate, or materially redirect the response plan "
        "itself. Typical qualifying patterns include a short late correction, "
        "negation, exception, condition, contrast, qualification, or decision cue "
        "that changes what Speaker B should say or do.\n\n"
        "Examples that fit: 'Could you review the introduction—actually, the full "
        "draft?'; 'I can meet tomorrow, but only after five'; 'We should book "
        "Monday—unless the conference starts Sunday'.\n\n"
        "Do NOT keep the pair if: (1) the revising content appears early enough "
        "that Speaker B could complete planning before Turn 1 ends; (2) the delay "
        "is only due to speaker style or hesitation; (3) the final content merely "
        "provides a slot that fits an already prepared frame; (4) Speaker B's "
        "response does not actually depend on the late information.\n\n"
        "Key test: does a short, late final update alter the response that Speaker B "
        "would otherwise have given, while leaving too little time to reliably revise "
        "that response before Turn 1 ends?"
    ),
    "take_turn_with_overlap": (
        "This pair was identified from timing as TAKE-TURN-WITH-OVERLAP: Speaker B "
        "began before Speaker A had acoustically finished (both speakers were active "
        "in an overlapping turn). Keep it only if the overlap is a safe content-based "
        "early entry rather than an interruption. Before Speaker B begins, Turn 1 "
        "must already have supplied all information needed for a correct and "
        "appropriate substantive response. The remaining Speaker A speech must be "
        "non-response-changing closing material — a hedge, repetition, trailing "
        "phrase, or prosodic wrap-up.\n\n"
        "Examples that fit: 'Thursday afternoon works best for me, I think—' "
        "followed by 'Yeah, Thursday works for me too'.\n\n"
        "Do NOT keep the pair if: (1) Speaker A is still providing a missing entity, "
        "condition, polarity marker, or other response-changing information; (2) the "
        "overlap occurs before the main question or request is complete; (3) Speaker "
        "B's reply would become incorrect after hearing the remaining speech; (4) the "
        "overlap is a backchannel, repair, or interruption.\n\n"
        "Key test: if the remaining suffix of Turn 1 were removed, would Speaker B's "
        "response still be equally correct and appropriate?"
    ),
}

GPT_PROMPT = """\
You are a dialogue analysis expert evaluating a 2-turn turn-taking pattern for \
research on conversational turn-taking. This is from a telephone conversation \
between a caller and a librarian (no punctuation in transcript — spoken dialogue).

Turn 1 will be used ALONE as the user-side input to a model, so it must stand on \
its own. Turn 2 only confirms that a genuine response followed; it may be incomplete.

--- Turn 1 (Speaker A) ---
"{t1}"
  Turn span: {t1_timed}

--- Turn 2 (Speaker B — response) ---
"{t2}"
  Turn span: {t2_timed}

This pair's assigned turn-taking category (from timing — you do NOT change it) is: \
{category}

---
Evaluate each field in order. Complete ALL fields even if an earlier answer is \
false — just set is_valid to false at the end.

1. TURN1_COMPLETE: Is Turn 1 a complete, well-formed utterance (not a mid-sentence \
fragment or incomplete thought)? Note: there is no punctuation — judge from content \
and grammar alone.

2. TURN1_SELF_CONTAINED: Can Turn 1 be fully understood and responded to WITHOUT \
any prior conversational context? Be strict:
   - Reject if it contains references whose meaning is unclear without context \
(e.g. "that one", "it", "those people").
   - Reject if it begins with an acknowledgement then transitions into a topic \
that requires prior context.
   - Reject if it is clearly a direct answer to an unseen question.
   - Reject if it is a reaction or emotional response that only makes sense as a \
reply to something just said ("Are you serious?", "Really?", "No way!").
   - Accept only if the content introduces a complete, interpretable thought that \
something could naturally respond to.

3. TURN2_RESPONDS: Is Turn 2 a genuine response aimed at Turn 1 — answering, \
reacting to, agreeing/disagreeing with, or asking a follow-up about Turn 1? \
Turn 2 may be incomplete, but it must be directed at Turn 1, not an unrelated remark.

4. CATEGORY_APPROPRIATE: For the assigned category "{category}", judge ONLY from \
the content of Turn 1 whether this pair is a clean, content-appropriate example:

{category_guidance}

Answer true if Turn 1's content genuinely fits "{category}", false otherwise.

5. IS_VALID: Set to true ONLY if ALL of the above are true.

Respond ONLY with JSON (no markdown):
{{
  "turn1_complete": true/false,
  "turn1_self_contained": true/false,
  "turn2_responds": true/false,
  "category_appropriate": true/false,
  "is_valid": true/false,
  "reason": "one sentence summary"
}}"""


class QuotaExceededError(RuntimeError):
    pass


def gpt_classify(candidates: list[dict], client,
                 max_valid_per_category: int | None = None
                 ) -> tuple[list[dict], dict]:
    total = len(candidates)
    print(f"  Sending {total} candidates to GPT ({GPT_MODEL})"
          + (f" (cap={max_valid_per_category}/category)"
             if max_valid_per_category else "") + "...")

    cat_valid: dict[str, int] = {c: 0 for c in CATEGORIES}

    gpt_stats = dict(
        total_sent   = total,
        total_judged = 0,
        valid        = 0,
        rejected     = 0,
        criteria     = {},
        by_category  = {c: 0 for c in CATEGORIES},
    )

    for i, cand in enumerate(candidates):
        cat = cand.get('category', 'unknown')

        if max_valid_per_category is not None:
            if cat_valid.get(cat, 0) >= max_valid_per_category:
                cand['gpt_valid']  = False
                cand['gpt_reason'] = "skipped: category budget reached"
                if all(cat_valid.get(c, 0) >= max_valid_per_category
                       for c in CATEGORIES):
                    print(f"    [{i+1}/{total}] all category budgets reached — stopping.")
                    break
                continue

        # For Pattern B, include the overlap context in the prompt
        if cand.get('pattern') == 'overlap':
            t2_shown = (f"[During overlap with A: \"{cand.get('ovl_b_text','')}\"] "
                        f"{cand['t2_text']}")
        else:
            t2_shown = cand['t2_text']

        msg = GPT_PROMPT.format(
            t1                = cand['t1_text'],
            t1_timed          = cand.get('t1_timed', ''),
            t2                = t2_shown,
            t2_timed          = cand.get('t2_timed', ''),
            category          = cat,
            category_guidance = CATEGORY_GUIDANCE.get(cat, ''),
        )

        try:
            resp = client.chat.completions.create(
                model    = GPT_MODEL,
                messages = [{"role": "user", "content": msg}],
                temperature      = 0,
                response_format  = {"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            cand['gpt_valid']            = result.get('is_valid', False)
            cand['gpt_reason']           = result.get('reason', '')
            cand['turn1_complete']       = result.get('turn1_complete')
            cand['turn1_self_contained'] = result.get('turn1_self_contained')
            cand['turn2_responds']       = result.get('turn2_responds')
            cand['category_appropriate'] = result.get('category_appropriate')
        except Exception as e:
            err = str(e)
            if 'insufficient_quota' in err or '429' in err or 'rate_limit' in err.lower():
                cand['gpt_valid']  = None
                cand['gpt_reason'] = f"API error: {e}"
                raise QuotaExceededError(str(e))
            print(f"    [{i+1}/{total}] WARN GPT failed: {e}")
            cand['gpt_valid']  = None
            cand['gpt_reason'] = f"API error: {e}"
            continue

        gpt_stats['total_judged'] += 1

        if cand['gpt_valid']:
            gpt_stats['valid'] += 1
            cat_valid[cat] = cat_valid.get(cat, 0) + 1
            gpt_stats['by_category'][cat] += 1
        else:
            gpt_stats['rejected'] += 1
            for key in ('turn1_complete', 'turn1_self_contained',
                        'turn2_responds', 'category_appropriate'):
                if cand.get(key) is False:
                    gpt_stats['criteria'][key] = \
                        gpt_stats['criteria'].get(key, 0) + 1

        status = '✓' if cand['gpt_valid'] else '✗'
        print(f"    [{i+1}/{total}] {status} [{cat[:4]}] — {cand['t1_text'][:70]}")
        sys.stdout.flush()

    print(f"GPT_STATS:{json.dumps(gpt_stats)}")
    sys.stdout.flush()
    return candidates, gpt_stats


# ── Step 3: Audio Export ───────────────────────────────────────────────────────

def export_candidates(candidates: list[dict],
                      ch0: 'np.ndarray', ch1: 'np.ndarray', sr: int,
                      spkr_channel: dict[str, int],
                      results_dir: Path) -> None:
    """
    Export per-speaker mono clips for each GPT-validated candidate into:

      results_dir/<category>/
        <file_id>_tt<t1_idx>_t1.wav    — T1 full turn (A's channel)
        <file_id>_tt<t1_idx>_t2.wav    — T2 full turn (B's channel)
        <file_id>_tt<t1_idx>_full.wav  — T1 start → T2 end (A's channel)

    Hierarchy: category/ only (consistent with CORAAL).
    """
    n0, n1 = len(ch0), len(ch1)

    def clip(audio: 'np.ndarray', n: int, st: float, en: float) -> 'np.ndarray':
        s = max(0, int(st * sr))
        e = min(n, int(en * sr))
        return audio[s:e]

    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue

        category = cand.get('category', 'unknown')
        out_dir  = results_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)

        spk_a = cand['spk_a']
        spk_b = cand['spk_b']
        ch_a  = spkr_channel.get(spk_a, 0)
        ch_b  = spkr_channel.get(spk_b, 1)
        aud_a = ch0 if ch_a == 0 else ch1
        aud_b = ch0 if ch_b == 0 else ch1
        n_a   = n0 if ch_a == 0 else n1
        n_b   = n0 if ch_b == 0 else n1

        base = f"{cand['file_id']}_tt{cand['t1_idx']:04d}"
        clips_spec = [
            ('t1',   aud_a, n_a, cand['t1_start'], cand['t1_end']),
            ('t2',   aud_b, n_b, cand['t2_start'], cand['t2_end']),
            ('full', aud_a, n_a, cand['t1_start'], cand['t2_end']),
        ]
        for tag, audio, n_samples, st, en in clips_spec:
            sf.write(str(out_dir / f"{base}_{tag}.wav"),
                     clip(audio, n_samples, st, en), sr)

        cand['audio_dir'] = str(out_dir)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract turn-taking pairs from a Loqui recording.")
    parser.add_argument("--file_id",              required=True)
    parser.add_argument("--trs_path",             type=Path, required=True)
    parser.add_argument("--wav_path",             type=Path, required=True)
    parser.add_argument("--results_dir",          type=Path, required=True)
    parser.add_argument("--max_valid_per_category", type=int,
                        default=TARGET_PER_CATEGORY)
    parser.add_argument("--no_gpt",               action="store_true")
    parser.add_argument("--no_audio",             action="store_true")
    parser.add_argument("--force",                action="store_true")
    args = parser.parse_args()

    valid_path = args.results_dir / args.file_id / "valid.json"
    if not args.force and valid_path.exists():
        print(f"{args.file_id}: already done (use --force to redo).")
        data = json.loads(valid_path.read_text())
        print(f"TIMING_STATS:{{}}")
        print(f"GPT_STATS:{{}}")
        print(f"VALID_JSON:{json.dumps(data)}")
        return

    # ── Parse TRS ─────────────────────────────────────────────────────────────
    valid_spkrs, turns = parse_trs_full(args.trs_path)
    if not turns:
        print(f"{args.file_id}: no turns found.")
        print("TIMING_STATS:{}")
        print("VALID_JSON:[]")
        return

    n_single  = sum(1 for t in turns if not t.get('is_overlap'))
    n_overlap = sum(1 for t in turns if     t.get('is_overlap'))
    print(f"{args.file_id}: {len(valid_spkrs)} valid speakers, "
          f"{n_single} single-spkr + {n_overlap} overlap turns")

    # Channel map: caller → ch0 (left), librarian → ch1 (right)
    spkr_channel: dict[str, int] = {
        sid: (0 if 'caller' in name.lower() else 1)
        for sid, name in valid_spkrs.items()
    }

    # ── Step 1: Timing filter ─────────────────────────────────────────────────
    print(f"\n── Step 1: Timing filter ──")
    candidates, timing_stats = find_turn_taking_patterns(turns, args.file_id)
    by_cat = timing_stats['by_category']
    print(f"  Candidates: {len(candidates)}  " +
          "  ".join(f"{k}={v}" for k, v in by_cat.items() if v))
    print(f"TIMING_STATS:{json.dumps(timing_stats)}")

    out_dir = args.results_dir / args.file_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing_all.json").write_text(json.dumps(candidates, indent=2))

    if args.no_gpt or not candidates:
        print("  Skipping GPT step.")
        print("VALID_JSON:[]")
        return

    # ── Step 2: GPT filter ────────────────────────────────────────────────────
    print(f"\n── Step 2: GPT filter ({GPT_MODEL}) ──")
    client = OpenAI(api_key=OPENAI_API_KEY)

    gpt_stats: dict = {}
    try:
        candidates, gpt_stats = gpt_classify(
            candidates, client,
            max_valid_per_category=args.max_valid_per_category,
        )
    except QuotaExceededError as e:
        print(f"  Quota exceeded: {e}")
        (out_dir / "gpt_partial.json").write_text(json.dumps(candidates, indent=2))
        sys.exit(QUOTA_EXIT_CODE)

    valid = [c for c in candidates if c.get('gpt_valid')]
    print(f"\n  Valid after GPT: {len(valid)}/{len(candidates)}")
    for cat in CATEGORIES:
        n = sum(1 for c in valid if c.get('category') == cat)
        if n:
            print(f"    {cat}: {n}")

    for cat in CATEGORIES:
        n = sum(1 for c in valid if c.get('category') == cat)
        print(f"  VALID_AFTER_LLM {cat}: {n}")

    (out_dir / "gpt_all.json").write_text(json.dumps(candidates, indent=2))
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.write_text(json.dumps(valid, indent=2))

    # ── Step 3: Audio export ──────────────────────────────────────────────────
    if args.no_audio or not valid:
        print("  Skipping audio export.")
        print(f"VALID_JSON:{json.dumps(valid)}")
        return

    print(f"\n── Step 3: Audio export ──")
    raw, sr_orig = sf.read(str(args.wav_path), always_2d=True)
    ch0 = librosa.resample(
        raw[:, 0].astype(np.float32), orig_sr=sr_orig, target_sr=AUDIO_SR)
    ch1 = librosa.resample(
        raw[:, 1].astype(np.float32), orig_sr=sr_orig, target_sr=AUDIO_SR) \
        if raw.shape[1] > 1 else ch0
    print(f"  Audio: {len(ch0)/AUDIO_SR:.1f}s @ {AUDIO_SR}Hz")

    export_candidates(valid, ch0, ch1, AUDIO_SR, spkr_channel, args.results_dir)
    print(f"  Exported {len(valid)} × 3 clips → {args.results_dir}/")
    print(f"VALID_JSON:{json.dumps(valid)}")


if __name__ == "__main__":
    main()
