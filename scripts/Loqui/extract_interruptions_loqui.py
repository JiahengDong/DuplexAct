#!/usr/bin/env python3
"""
Loqui Interruption & Recovery Extractor

Mirrors the CORAAL/Candor extractors: every timing and GPT validity
criterion can be individually toggled off via --skip_* / --min_* flags, and the
three INTERRUPTION_STYLE conditions (floor intent, pragmatic completeness,
mid-flow onset) are evaluated as separate booleans rather than a single
hard/none label.

In Loqui the T2+T3 overlap is encoded as a SINGLE multi-speaker XML turn:

  <Turn speaker="spkA">          ← T1  (A complete utterance)
  <Turn speaker="spkB">          ← T2-pre  (B starts responding, single-speaker)
  <Turn speaker="spkB spkA">     ← OVERLAP ZONE
    <Who nb="1"/> B-pre [os B-overlap eo]     ← T2 continuation + T2-post
    <Who nb="2"/> [os A-overlap eo] A-post    ← T3
  <Turn speaker="spkB">          ← T4  (B recovery)

nb=1 = speaker being interrupted (B); nb=2 = interrupter (A).

The T2-pre single-speaker turn may be absent when B barely spoke before A cut in.

Pipeline:
  Step 1 – Timing filter : scan all turns for the 4-piece pattern
  Step 2 – GPT filter    : verify semantic quality; classify style/type
  Step 3 – Audio export  : per-channel mono clips + stereo full clip

Usage:
  python extract_interruptions_loqui.py \\
      --file_id 1 \\
      --trs_path .../Transcriptions/1.final.trs \\
      --wav_path .../Transcriptions/1.san.wav \\
      --results_dir .../results/interruptions_loqui \\
      --skip_gpt_t1_self_contained --skip_gpt_t2_responds
"""

import argparse
import os
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

# Segment boundaries carry annotation imprecision; used as slack in comparisons.
TIMESTAMP_NOISE_TOLERANCE = 0.05

MIN_INTERRUPTION_OVERLAP  = 0.3

# Any overlap turn longer than this is a floor-takeover, not a brief interruption.
MAX_T3_DURATION = 10.0

# Maximum allowed overlap at the T2/T1 and T4/T3 boundaries.
# Loqui XML structure largely enforces ordering, but we keep this configurable
# for robustness and consistency with CORAAL/Candor extractors.
MAX_OVERLAP_TOLERANCE = 1.5

# A turn with ≤ 3 words is a backchannel (consistent across all scripts).
MIN_SUBSTANTIVE_WORDS = 4   # > MAX_BC_WORDS (3)

AUDIO_SR = 16000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL       = "gpt-5.4-2026-03-05"
QUOTA_EXIT_CODE = 2

TARGET_PER_TYPE = 500

RECOVERY_TYPES = ["correct", "enrich", "shorten", "repeat",
                  "shift_topic", "ask_followup", "none"]

# ── Per-run filter configuration ──────────────────────────────────────────────

@dataclass
class FilterConfig:
    # ── Timing criteria (False = criterion is skipped) ────────────────────────
    check_t2_no_overlap_t1:  bool  = True   # --skip_t2_overlap_t1
    max_t2_t1_overlap:       float = MAX_OVERLAP_TOLERANCE
    check_t3_overlap:        bool  = True   # --skip_t3_overlap
    min_t3_overlap:          float = MIN_INTERRUPTION_OVERLAP
    check_t4_no_overlap_t3:  bool  = True   # --skip_t4_overlap_t3
    max_t4_t3_overlap:       float = MAX_OVERLAP_TOLERANCE
    check_t1_min_words:      bool  = True   # --skip_t1_words
    check_t1_pauses:         bool  = True   # --skip_t1_pauses
    t1_pause_uppercase_only: bool  = False  # --t1_allow_lowercase_pause
    check_t2_not_question:   bool  = True   # --skip_t2_question
    check_t2_min_words:      bool  = True   # --skip_t2_words
    check_t2_pre_words:      bool  = True   # --skip_t2_pre_words
    check_t3_pauses:         bool  = True   # --skip_t3_pauses
    t3_pause_uppercase_only: bool  = False  # --t3_allow_lowercase_pause
    check_t3_max_duration:   bool  = True   # --skip_t3_duration
    check_t4_min_words:      bool  = True   # --skip_t4_words
    check_non_standard:      bool  = True   # --skip_non_standard

    # ── Backchannel overlap merge ─────────────────────────────────────────────
    do_bc_merge: bool = True   # --no_bc_merge

    # ── GPT validity criteria (False = criterion ignored in is_valid) ─────────
    gpt_require_t1_complete:           bool = True   # --skip_gpt_t1_complete
    gpt_require_t1_self_contained:     bool = True   # --skip_gpt_t1_self_contained
    gpt_require_t2_responds:           bool = True   # --skip_gpt_t2_responds
    gpt_require_t3_floor_intent:       bool = True   # --skip_gpt_t3_floor_intent
    gpt_require_t3_pragmatic_complete: bool = True   # --skip_gpt_t3_pragmatic_complete
    gpt_require_t3_mid_flow_onset:     bool = True   # --skip_gpt_t3_mid_flow
    gpt_require_t4_meaningful:         bool = True   # --skip_gpt_t4_meaningful

    def describe(self) -> str:
        skipped_t = [l for a, l in [
            ('check_t2_no_overlap_t1',  'T2/T1 no-overlap'),
            ('check_t3_overlap',        'T3 overlap'),
            ('check_t4_no_overlap_t3',  'T4/T3 no-overlap'),
            ('check_t1_min_words',      'T1 min-words'),
            ('check_t1_pauses',         'T1 pauses (uppercase-only)' if self.t1_pause_uppercase_only else 'T1 pauses'),
            ('check_t2_not_question',   'T2 not-question'),
            ('check_t2_min_words',      'T2 min-words'),
            ('check_t2_pre_words',      'T2 pre-words'),
            ('check_t3_pauses',         'T3 pauses (uppercase-only)' if self.t3_pause_uppercase_only else 'T3 pauses'),
            ('check_t3_max_duration',   'T3 duration'),
            ('check_t4_min_words',      'T4 min-words'),
            ('check_non_standard',      'non-standard speech'),
        ] if not getattr(self, a)]
        skipped_g = [l for a, l in [
            ('gpt_require_t1_complete',           't1_complete'),
            ('gpt_require_t1_self_contained',     't1_self_contained'),
            ('gpt_require_t2_responds',           't2_responds'),
            ('gpt_require_t3_floor_intent',       't3_floor_intent'),
            ('gpt_require_t3_pragmatic_complete', 't3_pragmatic_complete'),
            ('gpt_require_t3_mid_flow_onset',     't3_mid_flow_onset'),
            ('gpt_require_t4_meaningful',         't4_meaningful'),
        ] if not getattr(self, a)]
        lines = [
            f"  bc_merge          = {'ON' if self.do_bc_merge else 'OFF'} "
            f"(overlap: ≤{BC_MERGE_MAX_WORDS}w/≤{BC_MERGE_MAX_GAP}s; "
            f"single: ≤{BC_SINGLE_MERGE_MAX_WORDS}w/≤{BC_SINGLE_MERGE_MAX_DUR}s)",
            f"  min_t3_overlap    = {self.min_t3_overlap}s",
            f"  max_t2_t1_overlap = {self.max_t2_t1_overlap}s",
            f"  max_t4_t3_overlap = {self.max_t4_t3_overlap}s",
        ]
        if skipped_t:
            lines.append(f"  SKIPPED timing : {', '.join(skipped_t)}")
        if skipped_g:
            lines.append(f"  SKIPPED gpt    : {', '.join(skipped_g)}")
        if not skipped_t and not skipped_g:
            lines.append("  (all criteria active — strict mode)")
        return '\n'.join(lines)

# ── Text helpers ──────────────────────────────────────────────────────────────

# Markers that should be stripped before counting words / checking punctuation
_STRIP_RE = re.compile(
    r'<(?:P|p|B|b|BN|bn|NU|nu|G|NS|VB|IR|CHKL|ST|/ST|TN|TP|TC|CT)[^>]*>'
    r'|/(?:RD|ST)-[^\s<>]+'         # redaction markup (if present)
    r'|\[os\b|\beo\]|\boe\]'        # overlap bracket fragments
    r'|\\\\',                        # Transcriber line-end marker
    re.IGNORECASE)

# PII redaction tags — any <ALL-CAPS…> token not in the strip set above
_PII_RE   = re.compile(r'<[A-Z][A-Z0-9_ -]{2,}(?:-\d+)?>', re.IGNORECASE)
_PAUSE_RE = re.compile(r'<[pP]>')
_SENT_FINAL_RE = re.compile(r'[.?!]\s*$')

# Telephone-event markers — a turn containing any of these is call-disrupted
_PHONE_RE = re.compile(r'<(?:TN|TP|TC|CT)[^>]*>', re.IGNORECASE)

# Non-speech sound markers (breath noise, verbal fillers, irregular, clicking)
_NON_SPEECH_RE = re.compile(r'<(?:BN|VB|IR|CHKL)[^>]*>', re.IGNORECASE)

# Self-talk span
_ST_RE = re.compile(r'<ST>(.*?)</ST>', re.IGNORECASE | re.DOTALL)


def clean_text(raw: str) -> str:
    """Remove all non-lexical markers; return plain word string."""
    t = _STRIP_RE.sub(' ', raw)
    t = _PII_RE.sub(' ', t)
    t = _ST_RE.sub(' ', t)
    # Remove remaining XML-like tags
    t = re.sub(r'<[^>]+>', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def count_words(text: str) -> int:
    return len(clean_text(text).split())


def has_pause(text: str) -> bool:
    """True if the text contains a <P> or <p> pause marker (anywhere)."""
    return bool(_PAUSE_RE.search(text))


_SYNC_RE = re.compile(r'<Sync[^>]*/>', re.IGNORECASE)

def _word_count_raw(text: str) -> int:
    """Count words in raw TRS text (strips sync markers and XML tags)."""
    t = _SYNC_RE.sub(' ', text)
    t = re.sub(r'<[^>]+>', ' ', t)
    return len(t.split())


_PAUSE_RE_UPPER = re.compile(r'<P>')   # uppercase only (<P> = longer pause)

def has_internal_pause(text: str, uppercase_only: bool = False) -> bool:
    """True ONLY if a pause occurs in the MIDDLE of the turn.

    A pause is considered 'internal' (problematic) only when there are
    content words both BEFORE and AFTER it.  Leading pauses (speaker
    thinking before speaking) and trailing pauses are allowed.

    Args:
        uppercase_only: if True, only <P> (longer pauses) are checked;
                        <p> (short pauses) are ignored and treated as silence.
    """
    pattern = _PAUSE_RE_UPPER if uppercase_only else _PAUSE_RE
    pauses = list(pattern.finditer(text))
    if not pauses:
        return False
    for m in pauses:
        before = text[:m.start()]
        after  = text[m.end():]
        if _word_count_raw(before) >= 1 and _word_count_raw(after) >= 1:
            return True   # genuine mid-turn pause
    return False


def has_phone_event(text: str) -> bool:
    """True if the text contains a telephone-disruption marker."""
    return bool(_PHONE_RE.search(text))


def has_pii(text: str) -> bool:
    """True if the text contains a PII redaction tag."""
    return bool(_PII_RE.search(text))


def has_self_talk(text: str) -> bool:
    """True if the text contains a <ST>…</ST> self-talk span."""
    return bool(_ST_RE.search(text))


def has_unclear(text: str) -> bool:
    """True if the text contains <NU>, <G>, or <NS> (unclear/garbled speech)."""
    return bool(re.search(r'<(?:NU|G|NS)[^>]*>', text, re.IGNORECASE))


def has_non_speech_sound(text: str) -> bool:
    """True if the text contains a non-speech sound marker (BN, VB, IR, CHKL)."""
    return bool(_NON_SPEECH_RE.search(text))


def _non_standard_category(text: str) -> str | None:
    """Return the first non-standard speech sub-category found, or None.

    Check order (first match wins, counted once per rejection):
      ns_unclear    – unintelligible speech   (<NU>, <G>, <NS>)
      ns_non_speech – non-speech sounds       (<BN>, <VB>, <IR>, <CHKL>)
      ns_self_talk  – self-talk span          (<ST>…</ST>)
      ns_phone      – telephone/call events   (<TN>, <TP>, <TC>, <CT>)
      ns_pii        – PII redaction tags      (/RD-*, <LASTNAME-*>, …)
    """
    if has_unclear(text):          return 'ns_unclear'
    if has_non_speech_sound(text): return 'ns_non_speech'
    if has_self_talk(text):        return 'ns_self_talk'
    if has_phone_event(text):      return 'ns_phone'
    if has_pii(text):              return 'ns_pii'
    return None


def ends_sentence(text: str) -> bool:
    """True if cleaned text ends with sentence-final punctuation."""
    return bool(_SENT_FINAL_RE.search(clean_text(text).rstrip()))


# ── .trs XML parser ───────────────────────────────────────────────────────────

def get_primary_speakers(raw: str) -> tuple[str | None, str | None]:
    """
    Return (caller_id, librarian_id) from the <Speakers> section.
    Takes the FIRST caller and FIRST librarian found.
    """
    caller_id = librarian_id = None
    for m in re.finditer(r'<Speaker\s([^>]+)>', raw):
        attrs = m.group(1)
        sid_m  = re.search(r'\bid="([^"]*)"', attrs)
        name_m = re.search(r'\bname="([^"]*)"', attrs)
        if not sid_m or not name_m:
            continue
        sid  = sid_m.group(1)
        name = name_m.group(1).lower()
        if 'caller' in name and caller_id is None:
            caller_id = sid
        elif ('librarian' in name or 'libarian' in name) and librarian_id is None:
            librarian_id = sid
    return caller_id, librarian_id


def parse_trs(path: Path) -> tuple[list[dict], str, str]:
    """
    Parse a .trs file and return (turns, caller_id, librarian_id).

    Each turn dict has:
      spkrs      : frozenset of speaker IDs in this turn
      st         : float  start time
      en         : float  end time
      is_overlap : bool   multi-speaker turn?
      raw_body   : str    raw XML body (entities already decoded)

    For overlap turns:
      who1_id    : speaker ID of nb=1 (being interrupted)
      who1_text  : full decoded text for nb=1
      who2_id    : speaker ID of nb=2 (interrupter)
      who2_text  : full decoded text for nb=2
    """
    raw = path.read_text(encoding="iso-8859-1", errors="replace")
    # Decode XML entities once
    decoded = re.sub(r'&lt;', '<', re.sub(r'&gt;', '>', raw))

    caller_id, librarian_id = get_primary_speakers(raw)
    valid_spkrs = {s for s in (caller_id, librarian_id) if s}

    turns = []
    for tm in re.finditer(
            r'<Turn\s([^>]+)>(.*?)</Turn>', decoded, re.DOTALL):
        attrs = tm.group(1)
        body  = tm.group(2)

        spk_m = re.search(r'\bspeaker="([^"]*)"', attrs)
        st_m  = re.search(r'\bstartTime="([\d.]+)"', attrs)
        en_m  = re.search(r'\bendTime="([\d.]+)"',   attrs)
        if not (spk_m and st_m and en_m):
            continue

        spk_ids = spk_m.group(1).split()
        spkrs   = frozenset(spk_ids)

        # Skip turns involving any non-primary speaker
        if not spkrs.issubset(valid_spkrs):
            continue
        if not spkrs:
            continue

        st = float(st_m.group(1))
        en = float(en_m.group(1))
        is_overlap = len(spkrs) > 1

        turn = dict(spkrs=spkrs, st=st, en=en,
                    is_overlap=is_overlap, raw_body=body)

        if is_overlap:
            # Parse <Who nb="1"> and <Who nb="2"> sections
            who_blocks = re.split(r'<Who nb="(\d+)"\s*/>', body)
            # who_blocks = [pre, nb, text, nb, text, ...]
            who_texts: dict[int, str] = {}
            for j in range(1, len(who_blocks) - 1, 2):
                nb  = int(who_blocks[j])
                txt = who_blocks[j + 1] if j + 1 < len(who_blocks) else ''
                # A single multi-speaker turn may have repeated <Who nb="1"> blocks
                # (multiple sync points with renewed attribution). Concatenate.
                who_texts[nb] = who_texts.get(nb, '') + ' ' + txt

            # nb=1 is the speaker started first (being interrupted); nb=2 = interrupter
            # Map nb → speaker ID from the order in the speaker= attribute
            # spk_ids[0] is the first-listed speaker → nb=1
            nb1_id = spk_ids[0] if len(spk_ids) > 0 else None
            nb2_id = spk_ids[1] if len(spk_ids) > 1 else None

            turn['who1_id']   = nb1_id
            turn['who1_text'] = who_texts.get(1, '')
            turn['who2_id']   = nb2_id
            turn['who2_text'] = who_texts.get(2, '')

        turns.append(turn)

    return turns, caller_id, librarian_id


# ── Single-speaker backchannel merge ─────────────────────────────────────────

# When A speaks, B gives a short backchannel (≤ this many words), then A
# continues — merge the two A-segments into one so backward scan finds the
# full T1.
BC_SINGLE_MERGE_MAX_WORDS = 3
# Max total duration of the B-backchannel turn.  Telephone backchannels
# ("okay", "mm-hm", "yes") are typically under 2 seconds.
BC_SINGLE_MERGE_MAX_DUR   = 2.0


def merge_single_speaker_backchannels(turns: list[dict]) -> tuple[list[dict], int]:
    """
    Scan for the pattern:
        single-speaker A-turn  →  short single-speaker B-turn  →  single-speaker A-turn

    If B's turn is a backchannel (≤ BC_SINGLE_MERGE_MAX_WORDS words AND
    duration ≤ BC_SINGLE_MERGE_MAX_DUR seconds), merge the two A-turns into
    one continuous A-turn and drop the B-turn.  The B-content is stored in
    `_bc_absorbed` for reference.

    This mirrors the CORAAL `merge_backchannel_turns()` logic applied to
    Loqui's single-speaker turn list.

    Returns (merged_turns, n_merges).
    """
    if not turns:
        return turns, 0

    changed = True
    n_merges = 0

    while changed:
        changed = False
        result: list[dict] = []
        i = 0
        while i < len(turns):
            # Need a triple: A(single) → B(single, short) → A(single)
            if (i + 2 < len(turns)
                    and not turns[i]['is_overlap']
                    and not turns[i+1]['is_overlap']
                    and not turns[i+2]['is_overlap']
                    and len(turns[i]['spkrs']) == 1
                    and len(turns[i+1]['spkrs']) == 1
                    and len(turns[i+2]['spkrs']) == 1
                    and turns[i]['spkrs'] == turns[i+2]['spkrs']       # same A speaker
                    and turns[i]['spkrs'] != turns[i+1]['spkrs']       # B is different
                    and count_words(turns[i+1]['raw_body']) <= BC_SINGLE_MERGE_MAX_WORDS
                    and (turns[i+1]['en'] - turns[i+1]['st']) <= BC_SINGLE_MERGE_MAX_DUR):

                # Merge: A1 + A2, absorb B backchannel
                merged = dict(
                    spkrs      = turns[i]['spkrs'],
                    st         = turns[i]['st'],
                    en         = turns[i+2]['en'],
                    is_overlap = False,
                    raw_body   = turns[i]['raw_body'] + ' ' + turns[i+2]['raw_body'],
                    _bc_absorbed = turns[i+1]['raw_body'],
                )
                result.append(merged)
                n_merges += 1
                changed = True
                i += 3
            else:
                result.append(turns[i])
                i += 1
        turns = result

    return turns, n_merges


# ── Overlap-turn helpers ──────────────────────────────────────────────────────

_OS_RE = re.compile(r'\[os\s(.*?)(?:eo\]|oe\])', re.DOTALL)


def split_overlap_text(text: str) -> tuple[str, str, str]:
    """
    Split a Who-block text into (pre_overlap, overlap, post_overlap).

    pre_overlap  : text before the first [os ... eo] span
    overlap      : content of the [os ... eo] span(s) — concatenated if multiple
    post_overlap : text after the last [os ... eo] span
    """
    parts = _OS_RE.split(text)
    # _OS_RE.split gives: [pre, group1, post, group2, post2, ...]
    # With capturing group: [outside0, inside0, outside1, inside1, outside2, ...]
    if len(parts) == 1:
        # No [os...eo] found — all pre-overlap
        return text.strip(), '', ''

    pre  = parts[0]
    post = parts[-1]
    # Odd indices (1, 3, ...) are captured overlap contents
    overlaps = [parts[i] for i in range(1, len(parts) - 1, 2)]
    return pre.strip(), ' '.join(overlaps).strip(), post.strip()


def segs_summary(text: str) -> str:
    """Return a compact timed summary of text for GPT (sync points as markers)."""
    return clean_text(text)


# ── Backchannel overlap merge ─────────────────────────────────────────────────

BC_MERGE_MAX_WORDS = 3      # overlap content ≤ this many words → backchannel
BC_MERGE_MAX_GAP   = 0.200  # max gap (s) between main speaker's segments to merge


def merge_backchannel_overlaps(turns: list[dict]) -> tuple[list[dict], int]:
    """
    Scan for overlap turns where one speaker's [os...eo] content is a backchannel
    (≤ BC_MERGE_MAX_WORDS words) AND that speaker continues speaking after the
    overlap (has post-overlap text).  In this case the overlap is NOT a real
    interruption — the "main" speaker barely paused and the other speaker merely
    interjected briefly.

    These overlap turns are rewritten as single-speaker turns of the dominant
    speaker (the one who continues), so that `find_interruption_patterns` can
    detect proper ABAB patterns across them.

    Returns (merged_turns, n_bc_merged).
    """
    result = []
    n_merged = 0

    for turn in turns:
        if not turn['is_overlap']:
            result.append(turn)
            continue

        who1_text = turn.get('who1_text', '')
        who2_text = turn.get('who2_text', '')

        who1_pre, who1_ov, who1_post = split_overlap_text(who1_text)
        who2_pre, who2_ov, who2_post = split_overlap_text(who2_text)

        nb1_words = count_words(who1_ov)  # B's overlap content
        nb2_words = count_words(who2_ov)  # A's overlap content

        nb1_has_continuation = count_words(who1_post) > 0
        nb2_has_continuation = count_words(who2_post) > 0

        dominant_spk = None
        bc_text      = None

        # Case 1: nb=1 (B) has a backchannel overlap AND continues after
        #   → B is the main speaker; A's brief words were the backchannel
        if nb1_words <= BC_MERGE_MAX_WORDS and nb1_has_continuation:
            dominant_spk = turn['who1_id']
            # Merged body: B's pre-overlap + B's post-overlap text
            merged_body = (who1_pre + ' ' + who1_post).strip()
            bc_text     = who2_ov or who1_ov   # save for reference
        # Case 2: nb=2 (A) has a backchannel overlap AND continues after
        #   → A is the main speaker; B's brief words were the backchannel
        elif nb2_words <= BC_MERGE_MAX_WORDS and nb2_has_continuation:
            dominant_spk = turn['who2_id']
            merged_body  = (who2_pre + ' ' + who2_post).strip()
            bc_text      = who1_ov or who2_ov
        # Case 3: BOTH sides are tiny and neither continues → very short exchange
        #   → treat as backchannel for nb=1 (rewrite as nb=1's single-speaker turn)
        elif nb1_words <= BC_MERGE_MAX_WORDS and nb2_words <= BC_MERGE_MAX_WORDS:
            dominant_spk = turn['who1_id']
            merged_body  = (who1_pre + ' ' + who1_post).strip()
            bc_text      = who2_ov

        if dominant_spk is not None:
            n_merged += 1
            merged_turn = dict(
                spkrs      = frozenset({dominant_spk}),
                st         = turn['st'],
                en         = turn['en'],
                is_overlap = False,
                raw_body   = merged_body,
                _bc_merged = True,
                _bc_text   = bc_text,
            )
            result.append(merged_turn)
        else:
            result.append(turn)

    return result, n_merged


# ── Step 1: Timing Filter ─────────────────────────────────────────────────────

def find_interruption_patterns(turns: list[dict],
                               file_id: str,
                               caller_id: str,
                               librarian_id: str,
                               cfg: FilterConfig) -> tuple[list[dict], dict]:
    """
    Pivot on each overlap turn and expand outward by merging consecutive
    same-speaker turns.  The 4-piece pattern is assembled as:

      T1  = all consecutive A-only turns ending just before B started speaking
      T2  = all consecutive B-only turns before overlap  +  B's overlap portion
      T3  = A's overlap portion  +  all consecutive A-only turns after overlap
      T4  = first B-only turn after T3 ends

    Returns (candidates, timing_stats).
    """
    stats = {
        'windows_checked':   0,
        'overlap_not_two':   0,
        'not_abab':          0,
        't2_overlaps_t1':    0,
        't3_overlap_short':  0,
        't4_overlaps_t3':    0,
        't1_few_words':      0,
        't1_has_pauses':     0,
        't2_few_words':      0,
        't2_pre_words_short':0,
        't2_is_question':    0,
        't3_has_pauses':     0,
        't3_too_long':       0,
        't3_empty':          0,
        't4_few_words':      0,
        'non_standard':      0,
        'ns_unclear':        0,
        'ns_non_speech':     0,
        'ns_self_talk':      0,
        'ns_phone':          0,
        'ns_pii':            0,
        'duplicate':         0,
        'passed':            0,
    }

    candidates = []
    seen: set  = set()
    n = len(turns)
    spk_b_set: frozenset
    spk_a_set: frozenset

    for i, ov_turn in enumerate(turns):
        if not ov_turn['is_overlap']:
            continue

        stats['windows_checked'] += 1

        spkrs = ov_turn['spkrs']
        if len(spkrs) != 2:
            stats['overlap_not_two'] += 1
            continue

        spk_b = ov_turn['who1_id']   # nb=1 = interrupted speaker (B)
        spk_a = ov_turn['who2_id']   # nb=2 = interrupter (A)
        if spk_b is None or spk_a is None:
            stats['overlap_not_two'] += 1
            continue

        spk_b_set = frozenset({spk_b})
        spk_a_set = frozenset({spk_a})
        ov_start  = ov_turn['st']
        ov_end    = ov_turn['en']

        # ── Backward pass: collect T2-pre then T1 ─────────────────────────
        t2_pre_turns: list[dict] = []
        j = i - 1
        while j >= 0 and not turns[j]['is_overlap'] \
                and turns[j]['spkrs'] == spk_b_set:
            t2_pre_turns.insert(0, turns[j])
            j -= 1

        t1_turns: list[dict] = []
        while j >= 0 and not turns[j]['is_overlap'] \
                and turns[j]['spkrs'] == spk_a_set:
            t1_turns.insert(0, turns[j])
            j -= 1

        if not t1_turns:
            stats['not_abab'] += 1
            continue

        # ── Forward pass: collect T3-post then find T4 ────────────────────
        t3_post_turns: list[dict] = []
        j = i + 1
        while j < n and not turns[j]['is_overlap'] \
                and turns[j]['spkrs'] == spk_a_set:
            t3_post_turns.append(turns[j])
            j += 1

        t4_turn: dict | None = None
        if j < n and not turns[j]['is_overlap'] \
                and turns[j]['spkrs'] == spk_b_set:
            t4_turn = turns[j]

        if t4_turn is None:
            stats['not_abab'] += 1
            continue

        # ── Timing ────────────────────────────────────────────────────────
        t1_start = t1_turns[0]['st']
        t1_end   = t1_turns[-1]['en']
        t2_start = t2_pre_turns[0]['st'] if t2_pre_turns else ov_start
        t2_end   = ov_end
        t3_start = ov_start
        t3_end   = t3_post_turns[-1]['en'] if t3_post_turns else ov_end
        t4_start = t4_turn['st']
        t4_end   = t4_turn['en']
        t2_pre_duration = ov_start - t2_start

        # ── Overlap boundary checks ───────────────────────────────────────
        # T2 must not start too far before T1 ends (annotation noise allowed)
        if cfg.check_t2_no_overlap_t1:
            if t2_start < t1_end - cfg.max_t2_t1_overlap:
                stats['t2_overlaps_t1'] += 1
                continue

        # T3 must overlap T2 by at least min_t3_overlap
        # In Loqui, T3 starts at ov_start and T2 ends at ov_end.
        # The overlap duration is ov_end - ov_start.
        if cfg.check_t3_overlap:
            overlap_dur = ov_end - ov_start
            if overlap_dur < cfg.min_t3_overlap - TIMESTAMP_NOISE_TOLERANCE:
                stats['t3_overlap_short'] += 1
                continue

        # T4 must not start too far before T3 ends
        if cfg.check_t4_no_overlap_t3:
            if t4_start < t3_end - cfg.max_t4_t3_overlap:
                stats['t4_overlaps_t3'] += 1
                continue

        # ── Text assembly ─────────────────────────────────────────────────
        who1_pre, who1_ov, _ = split_overlap_text(ov_turn['who1_text'])

        t1_body      = ' '.join(t['raw_body'] for t in t1_turns)
        t2_pre_body  = ' '.join(t['raw_body'] for t in t2_pre_turns)
        t2_full_pre  = t2_pre_body + ' ' + who1_pre
        t3_ov_body   = ov_turn['who2_text']
        t3_cont_body = ' '.join(t['raw_body'] for t in t3_post_turns)
        t3_body      = t3_ov_body + (' ' + t3_cont_body if t3_cont_body else '')
        t4_body      = t4_turn['raw_body']

        # ── Per-criterion filters ─────────────────────────────────────────
        # NOTE: Loqui TRS annotations have no punctuation; T1 sentence-final
        # check is not applicable. Turn completeness is verified by GPT.

        if cfg.check_t1_min_words:
            if count_words(t1_body) < MIN_SUBSTANTIVE_WORDS:
                stats['t1_few_words'] += 1
                continue

        if cfg.check_t1_pauses:
            # Allow leading/trailing pauses (thinking before/after speaking);
            # only reject if a pause occurs mid-turn (between content words).
            if has_internal_pause(t1_body, uppercase_only=cfg.t1_pause_uppercase_only):
                stats['t1_has_pauses'] += 1
                continue

        if cfg.check_non_standard:
            ns_cat = _non_standard_category(t1_body)
            if ns_cat:
                stats['non_standard'] += 1
                stats[ns_cat] += 1
                continue

        if cfg.check_t2_not_question:
            t2_last_text = who1_pre if who1_pre.strip() else t2_pre_body
            if clean_text(t2_last_text).rstrip().endswith('?'):
                stats['t2_is_question'] += 1
                continue

        if cfg.check_t2_min_words:
            t2_full_text = (t2_full_pre + ' ' + who1_ov).strip()
            if count_words(t2_full_text) < MIN_SUBSTANTIVE_WORDS:
                stats['t2_few_words'] += 1
                continue

        if cfg.check_t2_pre_words:
            if count_words(t2_full_pre) < MIN_SUBSTANTIVE_WORDS:
                stats['t2_pre_words_short'] += 1
                continue

        if cfg.check_t3_pauses:
            # Same as T1: only reject mid-turn pauses.
            if has_internal_pause(t3_ov_body, uppercase_only=cfg.t3_pause_uppercase_only):
                stats['t3_has_pauses'] += 1
                continue

        if cfg.check_t3_max_duration:
            if t3_end - t3_start > MAX_T3_DURATION:
                stats['t3_too_long'] += 1
                continue

        if count_words(t3_body) == 0:
            stats['t3_empty'] += 1
            continue

        if cfg.check_t4_min_words:
            if count_words(t4_body) < MIN_SUBSTANTIVE_WORDS:
                stats['t4_few_words'] += 1
                continue

        if cfg.check_non_standard:
            ns_cat = _non_standard_category(t4_body)
            if ns_cat:
                stats['non_standard'] += 1
                stats[ns_cat] += 1
                continue

        # ── De-duplicate ──────────────────────────────────────────────────
        key = (file_id, spk_a, round(t1_start, 2))
        if key in seen:
            stats['duplicate'] += 1
            continue
        seen.add(key)

        stats['passed'] += 1

        candidates.append({
            'type':    'interruption',
            'source':  'loqui',
            'file_id': file_id,
            'spk_a':   spk_a,
            'spk_b':   spk_b,
            'ov_idx':  i,

            't1_start':         round(t1_start,        3),
            't1_end':           round(t1_end,           3),
            't2_start':         round(t2_start,         3),
            't2_end':           round(t2_end,            3),
            'ov_start':         round(ov_start,          3),
            'ov_end':           round(ov_end,            3),
            't3_start':         round(t3_start,          3),
            't3_end':           round(t3_end,            3),
            't4_start':         round(t4_start,          3),
            't4_end':           round(t4_end,            3),
            'overlap_duration': round(ov_end - ov_start, 3),
            't2_pre_duration':  round(t2_pre_duration,   3),

            't1_text':         clean_text(t1_body),
            't2_pre_text':     clean_text(t2_full_pre),
            't2_overlap_text': clean_text(who1_ov),
            't3_text':         clean_text(t3_body),
            't4_text':         clean_text(t4_body),

            # GPT fields (populated later)
            'gpt_valid':                  None,
            'turn1_complete':             None,
            'turn1_self_contained':       None,
            'turn2_responds':             None,
            'turn2_interrupted':          None,
            't3_floor_intent':            None,
            't3_pragmatic_complete':      None,
            't3_mid_flow_onset':          None,
            'interruption_style':         None,
            'interruption_type':          None,
            'turn4_meaningful':           None,
            'gpt_reason':                 None,
        })

    return candidates, stats


# ── Step 2: GPT Filter ────────────────────────────────────────────────────────

_LOQUI_CTX = (
    "The conversation is from the Loqui corpus — a dyadic telephone conversation "
    "between a library patron and a librarian."
)

_GPT_HEADER = """\
You are a dialogue analysis expert evaluating a 4-turn conversational pattern \
for research on interruption and recovery in spoken dialogue.

""" + _LOQUI_CTX + """

Here are four consecutive turns from a conversation, with segment-level \
timestamps to help you assess timing precisely:

Turn 1 (Speaker A):
"{t1}"

Turn 2 (Speaker B — response):
"{t2_pre}"
  (B continued during overlap: "{t2_ov}")
  [Turn 2 ran {t2_start:.2f}s–{t2_end:.2f}s; Turn 3 started at {t3_start:.2f}s, \
i.e. {t2_pre_nwords} content words into Turn 2, with {ov_dur:.2f}s of overlap remaining]

Turn 3 (Speaker A — started while B was speaking, overlap duration {ov_dur:.2f}s):
"{t3}"

Turn 4 (Speaker B — recovery/response):
"{t4}"

---
Evaluate each turn in order. Complete ALL fields even if an earlier answer is \
false — set is_valid to false at the end.
"""

_Q_T1_COMPLETE = """\
── TURN 1 (Speaker A) ──────────────────────────────────────────────────────────

1. TURN1_COMPLETE: Is Turn 1 a complete, well-formed utterance (not a fragment)?
"""

_Q_T1_SELF_CONTAINED = """\
2. TURN1_SELF_CONTAINED: Can Turn 1 be fully understood and responded to WITHOUT \
any prior conversational context? Be strict:
   - Reject if it contains references whose meaning is unclear without context \
(e.g. "that one", "it", "those people", bare numbers like "four or five").
   - Reject if it begins with an acknowledgement then transitions into a topic \
requiring prior context.
   - Reject if it is clearly a direct answer to an unseen question.
   - Reject if it is a reaction that only makes sense as a reply to something just \
said ("Are you serious?", "Really?", "No way!").
   - Accept only if the content introduces a complete, interpretable thought on its own.
"""

_Q_T2_RESPONDS = """\
── TURN 2 (Speaker B) ──────────────────────────────────────────────────────────

3. TURN2_RESPONDS: Is Turn 2 a genuine, direct DECLARATIVE response to Turn 1?

Turn 2 must directly address the content of Turn 1 and primarily take the form of \
a declarative answer, explanation, statement, or narrative continuation.

Answer false if Turn 2 is primarily:
   - a question directed to Speaker A, including a follow-up question;
   - a topic shift or unrelated remark;
   - a backchannel, acknowledgement, or short reaction;
   - an imperative, request, or instruction rather than a declarative response.

A brief tag question at the end does not automatically make this false, provided \
the main communicative action of Turn 2 is still a declarative response.

4. TURN2_INTERRUPTED: Was Turn 2 cut off before reaching a natural ending?

Turn 3 always overlaps the end of Turn 2 (only overlapping cases are sent to you), \
so you do NOT need Speaker B to keep talking after Turn 3 begins — B may stop \
immediately. Judge ONLY whether Turn 2 stopped at a non-natural point.

Answer true if Turn 2's final words leave the thought clearly unfinished — e.g. \
it trails off mid-clause, ends on a dangling conjunction or preposition, ends with \
an incomplete predicate, or stops in the middle of an idea B was still developing.

Answer false if Turn 2 had already reached a natural completion point — a complete \
clause or sentence that stands on its own — so Speaker A simply took the next turn \
with slight overlap.
"""

_Q_T3_FLOOR_INTENT = """\
── TURN 3 (Speaker A — interruption) ───────────────────────────────────────────

Identify the floor-taking onset in Turn 3 — the first substantive word that \
attempts to take the floor (ignore leading "yeah", "uh-huh", "okay", etc.).

5. T3_FLOOR_INTENT: Does Turn 3 contribute meaningful content that requires or \
invites a response (question, correction, challenge, counterpoint, substantive \
information)? Pure backchannels ("uh huh", "right") and wrap-up phrases \
("okay that's cool") do NOT qualify. Answer true/false.
"""

_Q_T3_PRAGMATIC_COMPLETE = """\
6. T3_PRAGMATIC_COMPLETE: Does Turn 3 form a complete conversational action \
(not a fragment that fails to deliver a full communicative move)? Answer true/false.
"""

_Q_T3_MID_FLOW = """\
7. T3_MID_FLOW_ONSET: Does the onset of Turn 3 occur while Turn 2 is still \
substantively in progress (i.e. B had not yet reached a natural completion point \
before A started)? Answer true/false.
"""

_Q_T4_MEANINGFUL = """\
── TURN 4 (Speaker B — recovery) ───────────────────────────────────────────────

8. TURN4_MEANINGFUL: Does Turn 4 provide a substantive response to the content or \
communicative action of Turn 3?

Answer true only if Turn 4 performs meaningful conversational work, such as:
   - answering A's question with actual information or explanation;
   - explicitly revising or correcting B's earlier content;
   - repeating or rephrasing the requested content;
   - providing the requested elaboration;
   - summarizing or shortening the previous explanation;
   - substantively following A's newly introduced topic;
   - asking a genuine clarification question needed to understand A's interruption.

Answer false for generic, content-free, or merely affiliative responses, including:
   - "Yeah", "Right", "Okay", "Uh huh"
   - "You're right" without stating what was wrong
   - "I see", "Fair enough", "That makes sense"
   - "Sorry" without a substantive follow-up.

9. RECOVERY_TYPE: classify as one of:
   "correct"      – A corrects an error; B acknowledges or revises.
   "enrich"       – A asks for more detail; B elaborates.
   "shorten"      – A signals enough detail; B wraps up.
   "repeat"       – A asks B to repeat; B repeats.
   "shift_topic"  – A redirects to a different subject; B follows.
   "ask_followup" – B asks A to clarify the interruption.
   "none"         – Turn 4 is not substantive or no type fits.
"""


def build_gpt_prompt(cfg: FilterConfig) -> str:
    """Assemble GPT prompt from active question blocks based on FilterConfig."""
    parts = [_GPT_HEADER]
    if cfg.gpt_require_t1_complete:
        parts.append(_Q_T1_COMPLETE)
    if cfg.gpt_require_t1_self_contained:
        parts.append(_Q_T1_SELF_CONTAINED)
    if cfg.gpt_require_t2_responds:
        parts.append(_Q_T2_RESPONDS)
    if cfg.gpt_require_t3_floor_intent:
        parts.append(_Q_T3_FLOOR_INTENT)
    if cfg.gpt_require_t3_pragmatic_complete:
        parts.append(_Q_T3_PRAGMATIC_COMPLETE)
    if cfg.gpt_require_t3_mid_flow_onset:
        parts.append(_Q_T3_MID_FLOW)
    if cfg.gpt_require_t4_meaningful:
        parts.append(_Q_T4_MEANINGFUL)

    active_validity = []
    if cfg.gpt_require_t1_complete:         active_validity.append("turn1_complete=true")
    if cfg.gpt_require_t1_self_contained:   active_validity.append("turn1_self_contained=true")
    if cfg.gpt_require_t2_responds:         active_validity.append("turn2_responds=true")
    if cfg.gpt_require_t3_floor_intent:     active_validity.append("t3_floor_intent=true")
    if cfg.gpt_require_t3_pragmatic_complete: active_validity.append("t3_pragmatic_complete=true")
    if cfg.gpt_require_t3_mid_flow_onset:   active_validity.append("t3_mid_flow_onset=true")
    if cfg.gpt_require_t4_meaningful:       active_validity.append("turn4_meaningful=true")

    validity_cond = ", ".join(active_validity) if active_validity else "none (all GPT criteria skipped)"

    # Build JSON schema with only the fields that were actually asked about.
    # Fields not in the prompt are omitted so GPT does not invent values for them.
    json_fields: list[str] = []
    if cfg.gpt_require_t1_complete:             json_fields.append('  "turn1_complete": true/false,')
    if cfg.gpt_require_t1_self_contained:       json_fields.append('  "turn1_self_contained": true/false,')
    if cfg.gpt_require_t2_responds:             json_fields.append('  "turn2_responds": true/false,')
    # turn2_interrupted is always asked (it determines recovery_type)
    json_fields.append('  "turn2_interrupted": true/false,')
    if cfg.gpt_require_t3_floor_intent:         json_fields.append('  "t3_floor_intent": true/false,')
    if cfg.gpt_require_t3_pragmatic_complete:   json_fields.append('  "t3_pragmatic_complete": true/false,')
    if cfg.gpt_require_t3_mid_flow_onset:       json_fields.append('  "t3_mid_flow_onset": true/false,')
    if cfg.gpt_require_t4_meaningful:           json_fields.append('  "turn4_meaningful": true/false,')
    json_fields += [
        '  "recovery_type": "none"|"correct"|"enrich"|"shorten"|"repeat"|"shift_topic"|"ask_followup",',
        '  "is_valid": true/false,',
        '  "reason": "one sentence summary"',
    ]
    json_schema = '\n'.join(json_fields)

    parts.append(f"""\
── VALIDITY ─────────────────────────────────────────────────────────────────────

IS_VALID: true ONLY if ALL hold:
  {validity_cond}

Respond ONLY with JSON (no markdown):
{{{{
{json_schema}
}}}}""")
    return '\n'.join(parts)


class QuotaExceededError(RuntimeError):
    pass


def recompute_validity(cand: dict, cfg: FilterConfig) -> bool:
    """Derive is_valid from individual GPT field answers and FilterConfig."""
    if cfg.gpt_require_t1_complete and not cand.get('turn1_complete'):
        return False
    if cfg.gpt_require_t1_self_contained and not cand.get('turn1_self_contained'):
        return False
    if cfg.gpt_require_t2_responds and not cand.get('turn2_responds'):
        return False
    if cfg.gpt_require_t3_floor_intent and not cand.get('t3_floor_intent'):
        return False
    if cfg.gpt_require_t3_pragmatic_complete and not cand.get('t3_pragmatic_complete'):
        return False
    if cfg.gpt_require_t3_mid_flow_onset and not cand.get('t3_mid_flow_onset'):
        return False
    if cfg.gpt_require_t4_meaningful and not cand.get('turn4_meaningful'):
        return False
    return True


def gpt_classify(candidates: list[dict], client: OpenAI, cfg: FilterConfig,
                 max_valid_per_type: int | None = None) -> tuple[list[dict], dict]:
    prompt_template = build_gpt_prompt(cfg)
    total = len(candidates)
    print(f"  Sending {total} candidates to GPT ({GPT_MODEL})"
          + (f" (cap/type={max_valid_per_type})" if max_valid_per_type else "") + "...")

    valid_by_type: dict[str, int] = {t: 0 for t in RECOVERY_TYPES}

    for i, cand in enumerate(candidates):
        # Cap check: stop if all recovery types are full
        if max_valid_per_type is not None:
            if all(valid_by_type.get(t, 0) >= max_valid_per_type for t in RECOVERY_TYPES):
                print(f"    [{i+1}/{total}] all type caps reached — stopping.")
                break

        msg = prompt_template.format(
            t1            = cand['t1_text'],
            t2_pre        = cand['t2_pre_text'] or '(nothing before overlap)',
            t2_ov         = cand['t2_overlap_text'] or '(none)',
            t2_start      = cand.get('t2_start', 0.0),
            t2_end        = cand.get('t2_end', 0.0),
            t3_start      = cand.get('t3_start', 0.0),
            t2_pre_nwords = len((cand.get('t2_pre_text') or '').split()),
            t3            = cand['t3_text'],
            t4            = cand['t4_text'],
            ov_dur        = cand['overlap_duration'],
        )

        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": msg}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)

            cand['turn1_complete']         = result.get('turn1_complete')
            cand['turn1_self_contained']   = result.get('turn1_self_contained')
            cand['turn2_responds']         = result.get('turn2_responds')
            cand['turn2_interrupted']      = result.get('turn2_interrupted')
            cand['t3_floor_intent']        = result.get('t3_floor_intent')
            cand['t3_pragmatic_complete']  = result.get('t3_pragmatic_complete')
            cand['t3_mid_flow_onset']      = result.get('t3_mid_flow_onset')
            cand['turn4_meaningful']       = result.get('turn4_meaningful')
            cand['interruption_type']      = result.get('recovery_type', 'none')
            cand['gpt_reason']             = result.get('reason', '')

            # Derive interruption_style from the three T3 booleans
            if (cand.get('t3_floor_intent') and
                    cand.get('t3_pragmatic_complete') and
                    cand.get('t3_mid_flow_onset')):
                cand['interruption_style'] = 'hard'
            else:
                cand['interruption_style'] = 'none'

            cand['gpt_valid'] = recompute_validity(cand, cfg)

        except Exception as e:
            err = str(e)
            if 'insufficient_quota' in err or '429' in err:
                cand['gpt_valid']  = None
                cand['gpt_reason'] = f"API error: {e}"
                raise QuotaExceededError(str(e))
            print(f"    [{i+1}/{total}] WARN GPT failed: {e}")
            cand['gpt_valid']  = None
            cand['gpt_reason'] = f"API error: {e}"
            continue

        style = cand['interruption_style']
        rtype = cand['interruption_type'] or 'none'
        if cand['gpt_valid']:
            valid_by_type[rtype] = valid_by_type.get(rtype, 0) + 1

        flag = '✓' if cand['gpt_valid'] else '✗'
        print(f"    [{i+1}/{total}] {flag} [{style}] — {cand['t1_text'][:65]}")

    judged = [c for c in candidates if c.get('gpt_valid') is not None]
    gpt_stats = {
        'total_sent':                       len(candidates),
        'total_judged':                     len(judged),
        'turn1_complete_false':             sum(1 for c in judged if c.get('turn1_complete')        is False),
        'turn1_self_contained_false':       sum(1 for c in judged if c.get('turn1_self_contained')  is False),
        'turn2_responds_false':             sum(1 for c in judged if c.get('turn2_responds')        is False),
        'turn2_interrupted_false':          sum(1 for c in judged if c.get('turn2_interrupted')     is False),
        't3_floor_intent_false':            sum(1 for c in judged if c.get('t3_floor_intent')       is False),
        't3_pragmatic_complete_false':      sum(1 for c in judged if c.get('t3_pragmatic_complete') is False),
        't3_mid_flow_onset_false':          sum(1 for c in judged if c.get('t3_mid_flow_onset')     is False),
        'turn4_meaningful_false':           sum(1 for c in judged if c.get('turn4_meaningful')      is False),
        'interruption_style':               dict(Counter(c.get('interruption_style', '?') for c in judged)),
        'recovery_type':                    dict(Counter(c.get('interruption_type',   '?') for c in judged)),
        'valid_by_type':                    {k: v for k, v in valid_by_type.items() if v > 0},
        'valid_hard':                       sum(1 for c in judged if c.get('gpt_valid') and c.get('interruption_style') == 'hard'),
    }

    print(f"  VALID_AFTER_LLM hard: {gpt_stats['valid_hard']}")
    print(f"GPT_STATS:{json.dumps(gpt_stats)}")
    return candidates, gpt_stats


# ── Step 3: Audio Export ──────────────────────────────────────────────────────

def load_audio(wav_path: Path, caller_id: str, librarian_id: str,
               spk_a: str) -> tuple[np.ndarray, int, int]:
    """
    Load stereo .san.wav and return (audio_2ch, sr, ch_a).

    Loqui .san.wav: stereo telephone recording.
    We assign ch=0 to the first-listed speaker in the XML (typically caller=spk1)
    and ch=1 to the second (typically librarian=spk2).

    ch_a is the channel for speaker A (the interrupter) in this candidate.
    """
    audio_raw, sr_orig = sf.read(str(wav_path), always_2d=True)
    # Resample to AUDIO_SR, keep stereo
    if sr_orig != AUDIO_SR:
        ch0 = librosa.resample(audio_raw[:, 0].astype(np.float32),
                               orig_sr=sr_orig, target_sr=AUDIO_SR)
        ch1 = librosa.resample(audio_raw[:, 1].astype(np.float32),
                               orig_sr=sr_orig, target_sr=AUDIO_SR)
        audio = np.stack([ch0, ch1])   # shape (2, N)
        sr    = AUDIO_SR
    else:
        audio = audio_raw.T.astype(np.float32)   # (2, N)
        sr    = sr_orig

    # Channel assignment: spk1 (caller) → ch0, spk2 (librarian) → ch1
    # Determine which speaker is spk1 by checking caller_id
    ch_caller    = 0
    ch_librarian = 1
    if spk_a == caller_id:
        ch_a = ch_caller
    else:
        ch_a = ch_librarian

    return audio, sr, ch_a


def _write_clip(audio: np.ndarray, sr: int, ch: int,
                st: float, en: float, path: Path) -> None:
    s = max(0, int(st * sr))
    e = min(audio.shape[1], int(en * sr))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio[ch, s:e], sr)


def _write_stereo(audio: np.ndarray, sr: int,
                  st: float, en: float, path: Path) -> None:
    s = max(0, int(st * sr))
    e = min(audio.shape[1], int(en * sr))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio[:, s:e].T, sr)


def export_candidates(candidates: list[dict], audio: np.ndarray, sr: int,
                      caller_id: str, librarian_id: str,
                      results_dir: Path) -> None:
    """
    Export per-channel mono clips + full stereo for each valid candidate.

      results_dir/<style>/<recovery_type>/
        <file_id>_ov<ov_idx>_t1.wav      ← T1 (Speaker A mono)
        <file_id>_ov<ov_idx>_t2.wav      ← T2 (Speaker B mono, full pre+overlap)
        <file_id>_ov<ov_idx>_t3.wav      ← T3 (overlap zone; A's channel)
        <file_id>_ov<ov_idx>_t4.wav      ← T4 (Speaker B mono)
        <file_id>_ov<ov_idx>_full_stereo.wav
    """
    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue

        style  = cand.get('interruption_style', 'unknown')
        rtype  = cand.get('interruption_type',  'unknown')
        out_dir = results_dir / style / rtype
        out_dir.mkdir(parents=True, exist_ok=True)

        base = f"{cand['file_id']}_ov{cand['ov_idx']:04d}"

        spk_a = cand['spk_a']
        ch_a  = 0 if spk_a == caller_id else 1
        ch_b  = 1 - ch_a

        clips = [
            (cand['t1_start'], cand['t1_end'],  f"{base}_t1.wav",  ch_a,  False),
            (cand['t2_start'], cand['t2_end'],  f"{base}_t2.wav",  ch_b,  False),
            (cand['ov_start'], cand['ov_end'],  f"{base}_t3.wav",  ch_a,  False),
            (cand['t4_start'], cand['t4_end'],  f"{base}_t4.wav",  ch_b,  False),
            (cand['t1_start'], cand['t4_end'],  f"{base}_full_stereo.wav", None, True),
        ]
        for st, en, fname, ch, stereo in clips:
            dest = out_dir / fname
            if stereo:
                _write_stereo(audio, sr, st, en, dest)
            else:
                _write_clip(audio, sr, ch, st, en, dest)
        cand['audio_dir'] = str(out_dir)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_json(data: list[dict], path: Path, label: str) -> None:
    path = path.with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [{label}] Saved {len(data)} entries → {path.name}")


# ── Main body ─────────────────────────────────────────────────────────────────

def _main_body(args) -> None:
    f1_dir = args.out_dir / "filter1_timing"
    f2_dir = args.out_dir / "filter2_gpt"

    # ── Build FilterConfig ────────────────────────────────────────────────────
    cfg = FilterConfig(
        do_bc_merge                     = not args.no_bc_merge,
        check_t2_no_overlap_t1          = not args.skip_t2_overlap_t1,
        max_t2_t1_overlap               = args.max_t2_t1_overlap,
        check_t3_overlap                = not args.skip_t3_overlap,
        min_t3_overlap                  = args.min_t3_overlap,
        check_t4_no_overlap_t3          = not args.skip_t4_overlap_t3,
        max_t4_t3_overlap               = args.max_t4_t3_overlap,
        check_t1_min_words              = not args.skip_t1_words,
        check_t1_pauses                 = not args.skip_t1_pauses,
        t1_pause_uppercase_only         = args.t1_allow_lowercase_pause,
        check_t2_not_question           = not args.skip_t2_question,
        check_t2_min_words              = not args.skip_t2_words,
        check_t2_pre_words              = not args.skip_t2_pre_words,
        check_t3_pauses                 = not args.skip_t3_pauses,
        t3_pause_uppercase_only         = args.t3_allow_lowercase_pause,
        check_t3_max_duration           = not args.skip_t3_duration,
        check_t4_min_words              = not args.skip_t4_words,
        check_non_standard              = not args.skip_non_standard,
        gpt_require_t1_complete         = not args.skip_gpt_t1_complete,
        gpt_require_t1_self_contained   = not args.skip_gpt_t1_self_contained,
        gpt_require_t2_responds         = not args.skip_gpt_t2_responds,
        gpt_require_t3_floor_intent     = not args.skip_gpt_t3_floor_intent,
        gpt_require_t3_pragmatic_complete = not args.skip_gpt_t3_pragmatic_complete,
        gpt_require_t3_mid_flow_onset   = not args.skip_gpt_t3_mid_flow,
        gpt_require_t4_meaningful       = not args.skip_gpt_t4_meaningful,
    )

    print(f"\nFilter configuration:\n{cfg.describe()}")

    # ── Step 1: Timing filter ─────────────────────────────────────────────────
    print(f"\n── Step 1: Timing filter ──")
    turns, caller_id, librarian_id = parse_trs(args.trs_path)
    print(f"{args.file_id}: {len(turns)} primary-speaker turns  "
          f"(caller={caller_id}, librarian={librarian_id})")

    if caller_id is None or librarian_id is None:
        print("  WARN: could not identify both speakers — skipping file.")
        print("TIMING_STATS:{}")
        print("VALID_JSON:[]")
        return

    # ── Backchannel overlap merge (pre-processing) ────────────────────────────
    if cfg.do_bc_merge:
        turns, n_bc_overlap = merge_backchannel_overlaps(turns)
        print(f"  BC-merge (overlap): rewrote {n_bc_overlap} backchannel overlap(s) as single-speaker turns")

        # Also merge single-speaker A-turns that are separated by a short B-turn
        # (e.g.  A: "I need a book"  B: "mm-hm"  A: "about trains"
        #  → A: "I need a book about trains")
        turns, n_bc_single = merge_single_speaker_backchannels(turns)
        print(f"  BC-merge (single):  merged {n_bc_single} short-turn backchannel(s) into adjacent same-speaker turns")
    else:
        print("  BC-merge: OFF")

    candidates, timing_stats = find_interruption_patterns(
        turns, args.file_id, caller_id, librarian_id, cfg)

    print(f"  Candidates found: {len(candidates)}")
    print(f"TIMING_STATS:{json.dumps(timing_stats)}")
    print(f"  SEND_TO_LLM interruption: {len(candidates)}")

    save_json(candidates, f1_dir / "all", "timing")

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
            candidates, client, cfg,
            max_valid_per_type=args.max_valid_per_type,
        )
    except QuotaExceededError as e:
        print(f"  Quota exceeded: {e}", file=sys.stderr)
        save_json(candidates, f2_dir / "all_with_gpt_decisions", "all+decisions")
        sys.exit(QUOTA_EXIT_CODE)

    valid = [c for c in candidates if c.get('gpt_valid')]
    print(f"\n  Valid after GPT: {len(valid)}/{len(candidates)}")
    print(f"  VALID_AFTER_LLM hard: {sum(1 for c in valid if c.get('interruption_style')=='hard')}")

    save_json(candidates, f2_dir / "all_with_gpt_decisions", "all+decisions")
    if valid:
        save_json(valid, f2_dir / "valid", "valid")

    # ── Step 3: Audio export ──────────────────────────────────────────────────
    if valid and not args.no_audio:
        audio_dest = args.results_dir if args.results_dir else args.out_dir / "audio"

        if audio_dest.exists():
            stale = list(audio_dest.rglob(f"{args.file_id}_ov*"))
            for fp in stale:
                fp.unlink()
            if stale:
                print(f"  Removed {len(stale)} stale clip(s) for {args.file_id}")

        print(f"\n── Step 3: Audio export ──")
        print(f"Loading {args.wav_path.name}...")
        audio, sr, _ = load_audio(args.wav_path, caller_id, librarian_id,
                                   valid[0]['spk_a'])
        print(f"  Audio: {audio.shape}, sr={sr}")

        export_candidates(valid, audio, sr, caller_id, librarian_id, audio_dest)
        print(f"  Exported {len(valid)} × 5 clips → {audio_dest}/")
    else:
        print("  Skipping audio export.")

    print(f"\nDone. {len(valid)} valid candidates.")
    print(f"VALID_JSON:{json.dumps(valid)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract interruption patterns from a Loqui .trs recording.")
    parser.add_argument("--file_id",       required=True)
    parser.add_argument("--trs_path",      type=Path, required=True)
    parser.add_argument("--wav_path",      type=Path, required=True)
    parser.add_argument("--out_dir",       type=Path, default=None)
    parser.add_argument("--results_dir",   type=Path, default=None)
    parser.add_argument("--max_valid_per_type", type=int, default=TARGET_PER_TYPE)

    # ── Timing skip flags ─────────────────────────────────────────────────────
    parser.add_argument("--skip_t2_overlap_t1",  action="store_true")
    parser.add_argument("--max_t2_t1_overlap",   type=float, default=MAX_OVERLAP_TOLERANCE)
    parser.add_argument("--skip_t3_overlap",      action="store_true")
    parser.add_argument("--min_t3_overlap",       type=float, default=MIN_INTERRUPTION_OVERLAP)
    parser.add_argument("--skip_t4_overlap_t3",  action="store_true")
    parser.add_argument("--max_t4_t3_overlap",   type=float, default=MAX_OVERLAP_TOLERANCE)
    parser.add_argument("--skip_t1_words",        action="store_true")
    parser.add_argument("--skip_t1_pauses",            action="store_true")
    parser.add_argument("--t1_allow_lowercase_pause",  action="store_true",
                        help="Allow <p> (short) pauses in T1; only <P> (long) pauses "
                             "cause rejection. Default: both <p> and <P> count.")
    parser.add_argument("--skip_t2_question",     action="store_true")
    parser.add_argument("--skip_t2_words",        action="store_true")
    parser.add_argument("--skip_t2_pre_words",    action="store_true")
    parser.add_argument("--skip_t3_pauses",            action="store_true")
    parser.add_argument("--t3_allow_lowercase_pause",  action="store_true",
                        help="Allow <p> (short) pauses in T3; only <P> (long) pauses "
                             "cause rejection. Default: both <p> and <P> count.")
    parser.add_argument("--skip_t3_duration",     action="store_true")
    parser.add_argument("--skip_t4_words",        action="store_true")
    parser.add_argument("--skip_non_standard",    action="store_true")

    # ── GPT skip flags ────────────────────────────────────────────────────────
    parser.add_argument("--skip_gpt_t1_complete",           action="store_true")
    parser.add_argument("--skip_gpt_t1_self_contained",     action="store_true")
    parser.add_argument("--skip_gpt_t2_responds",           action="store_true")
    parser.add_argument("--skip_gpt_t3_floor_intent",       action="store_true")
    parser.add_argument("--skip_gpt_t3_pragmatic_complete", action="store_true")
    parser.add_argument("--skip_gpt_t3_mid_flow",           action="store_true")
    parser.add_argument("--skip_gpt_t4_meaningful",         action="store_true")

    parser.add_argument("--no_bc_merge", action="store_true",
                        help="Disable backchannel overlap merge pre-processing")
    parser.add_argument("--no_gpt",   action="store_true")
    parser.add_argument("--no_audio", action="store_true")
    args = parser.parse_args()

    _tmpdir_ctx = None
    if args.out_dir is None:
        _tmpdir_ctx = tempfile.TemporaryDirectory(
            prefix=f"loqui_intr_{args.file_id}_")
        args.out_dir = Path(_tmpdir_ctx.name)

    try:
        _main_body(args)
    finally:
        if _tmpdir_ctx is not None:
            _tmpdir_ctx.cleanup()


if __name__ == "__main__":
    main()
