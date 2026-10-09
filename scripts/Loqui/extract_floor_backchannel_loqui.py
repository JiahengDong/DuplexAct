#!/usr/bin/env python3
"""
Loqui Floor-Holding + Backchannel Extractor 

Extracts a 4-turn sequential A-B-A-B structure where Speaker A holds the
floor across a short backchannel from Speaker B:

  T0       : single-speaker B  – B's prior turn (trigger for A, ≥ MIN_T0_WORDS)
  T_before : single-speaker A  – A begins response, holds floor (≥ MIN_BEFORE_WORDS)
  T_bc     : single-speaker B  – B's backchannel (≤ MAX_BC_WORDS, no <P>)
  T_after  : single-speaker A  – A resumes (≥ MIN_AFTER_WORDS, no <P>)

Key design alignment with CORAAL:
  • Backchannels in Loqui appear overwhelmingly as sequential single-speaker
    turns (as confirmed by TSCA annotations: ~800 [Backchannel] labels across
    82 files), NOT as overlap-encoded turns.  We therefore scan the
    single-speaker-only turn list exactly as in CORAAL.
  • Multi-speaker overlap turns are filtered out by parse_trs — they are
    NOT the target pattern and their presence between turns does not break
    A's floor hold.
  • No ab_gap timing constraint (no word-level timestamps in Loqui TRS);
    an optional MAX_BC_GAP check uses turn-level timestamps as a loose guard.
  • Pause check (check_t1_pauses) uses has_internal_pause() — only rejects
    turns with <P> between content words, tolerating leading/trailing pauses.

Labels assigned by GPT (identical to CORAAL/CANDOR):
  floor_holding_label : continue_current | continue_shorten | continue_simplify | none
  backchannel_type    : acknowledgement | understanding | agreement |
                        uncertainty | disagreement | none

Pipeline:
  Step 1 – Timing filter : scan single-speaker turns for the 4-turn pattern
  Step 2 – GPT filter    : verify semantic quality and assign both labels
  Step 3 – Audio export  : per-speaker mono clips from stereo .san.wav

Usage:
  python extract_floor_backchannel_loqui.py \\
      --file_id 1 \\
      --trs_path .../Transcriptions/1.final.trs \\
      --wav_path .../Transcriptions/1.san.wav \\
      --results_dir .../results/floor_backchannel_loqui \\
      [--skip_t1_pauses] [--skip_t1_anon] [--skip_t0_words] [--skip_t0_pauses]
"""

import argparse
import os
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

MAX_BC_WORDS     = 3   # B's backchannel must be this many words or fewer
MIN_T0_WORDS     = 4   # B trigger must be substantive (> MAX_BC_WORDS)
MIN_DURING_WORDS = 2   # A must continue through the overlap (A's Who block)
MIN_AFTER_WORDS  = 4   # A must resume after the overlap

AUDIO_SR = 16000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL       = "gpt-5.4-2026-03-05"
QUOTA_EXIT_CODE = 2

TARGET_PER_LABEL = 500

FLOOR_HOLDING_LABELS = [
    "continue_current", "continue_shorten", "continue_simplify", "none"]
BACKCHANNEL_TYPES = [
    "acknowledgement", "understanding", "agreement",
    "uncertainty", "disagreement", "none"]

FLOOR_HOLDING_GUIDANCE = {
    "continue_current": (
        "A keeps going on the same track with no obvious change in plan, scope, or "
        "level of detail. The backchannel was simply absorbed and A continues the "
        "answer as if it had not occurred."
    ),
    "continue_shorten": (
        "After the backchannel, A noticeably wraps up, compresses, or cuts the "
        "answer short — moving to a conclusion, dropping planned detail, or "
        "hurrying to finish. Often follows a BC that signals the listener already "
        "agrees or has understood."
    ),
    "continue_simplify": (
        "After the backchannel, A rephrases, clarifies, gives an example, or "
        "explains more plainly — restating the same point in simpler terms. Often "
        "follows a BC that signals uncertainty or confusion."
    ),
    "none": (
        "A's continuation after the backchannel does not clearly fit any of the "
        "above — the turn is too short to judge, the BC occurs at the very end, "
        "or A's response is ambiguous. Use this only as a last resort."
    ),
}

BACKCHANNEL_TYPE_GUIDANCE = {
    "acknowledgement": (
        "A minimal 'I'm still listening / received' signal without committing to a "
        "stance — e.g. 'mhm', 'uh huh', 'right', 'okay'. Keeps the channel open "
        "but does not claim comprehension, agreement, or doubt."
    ),
    "understanding": (
        "Signals that the listener has grasped the content — e.g. 'oh I see', "
        "'got it', 'ah okay', 'that makes sense'. Stronger than bare acknowledgement."
    ),
    "agreement": (
        "Signals alignment or endorsement of A's point — e.g. 'yeah exactly', "
        "'totally', 'for sure', 'absolutely'. Only valid when A actually made a "
        "claim to agree on."
    ),
    "uncertainty": (
        "Signals hesitation, doubt, or confusion — e.g. 'hmm', 'really?', 'wait', "
        "a doubtful 'uh'. The listener is following but not convinced or unclear."
    ),
    "disagreement": (
        "Signals pushback or non-alignment — e.g. 'no', 'I don't think so', "
        "'mm not really', 'eh'. The listener does not accept A's point."
    ),
    "none": (
        "The backchannel does not clearly fit any of the above — ambiguous, "
        "multi-functional, or social meaning unclear from context. Last resort."
    ),
}


# ── FilterConfig ───────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """Controls which timing and GPT criteria are active.

    Timing flags
    ────────────
    check_t0             – master switch for all T0-side checks (speaker pattern,
                           anonymisation, word count, pause)
    check_t0_min_words   – T0 must have >= min_t0_words content words
    check_t0_pauses      – T0 must have no <P> pause tag
    check_t1_pauses      – T_before or T_after must have no internal <P> (between
                           content words; leading/trailing pauses are tolerated)
    check_t1_anon        – T_before and T_after must contain no anonymisation marker
                           (T1 = A's speech; if T1 is not used as a model prompt it
                           is safe to allow anonymised turns here)
    check_after_min_words– T_after must have >= min_after_words words

    Thresholds
    ──────────
    min_t0_words / min_after_words

    GPT flags
    ─────────
    gpt_require_t0_complete       – T0 is a complete utterance
    gpt_require_t0_self_contained – T0 is self-contained (no prior context needed)
    gpt_require_t1_complete       – combined T1 (T_before + T_after) is complete
    gpt_require_t1_self_contained – combined T1 is self-contained
    """
    # ── Timing filter ─────────────────────────────────────────────────────────
    check_t0:              bool = True   # master switch for all T0-side checks
    check_t0_min_words:    bool = True   # T0 word-count gate
    check_t0_pauses:       bool = True   # T0 no-pause gate
    t0_pause_uppercase_only: bool = False  # if True, only <P> (uppercase) rejects T0;
                                           # <p> (lowercase, shorter pause) is tolerated
    check_t1_pauses:       bool = True   # T_before or T_after no-internal-pause
    t1_pause_uppercase_only: bool = False  # if True, only <P> (uppercase) rejects T1;
                                           # <p> (lowercase, shorter pause) is tolerated
    check_t1_anon:         bool = True   # T_before / T_after no anonymisation marker
    check_after_min_words: bool = True   # T_after word-count gate
    allow_t_before_overlap: bool = False # allow T_before to be an overlap turn
                                         # (B must still speak > MAX_BC_WORDS in it)

    # Threshold overrides
    min_t0_words:    int   = MIN_T0_WORDS
    min_after_words: int   = MIN_AFTER_WORDS

    # ── GPT criteria ──────────────────────────────────────────────────────────
    gpt_require_t0_complete:        bool = True
    gpt_require_t0_self_contained:  bool = True
    gpt_require_t1_complete:        bool = True
    gpt_require_t1_self_contained:  bool = True

    # Task context — used to decide which GPT fields to ask and how to cap
    task: str = 'both'   # 'backchannel' | 'floor_holding' | 'both'

    def describe(self) -> str:
        skipped_t = [l for a, l in [
            ('check_t0',               'T0 checks (master)'),
            ('check_t0_min_words',     'T0 min-words'),
            ('check_t0_pauses',        'T0 no-pause'),
            ('check_t1_pauses',        'T1 no-internal-pause'),
            ('check_t1_anon',          'T1 no-anonymisation'),
            ('check_after_min_words',  'after-BC min-words'),
        ] if not getattr(self, a)]
        if self.allow_t_before_overlap:
            skipped_t.append('T_before single-speaker-only (overlap T_before allowed)')
        if self.check_t0_pauses and self.t0_pause_uppercase_only:
            skipped_t.append('T0 lowercase-<p> tolerated (uppercase <P> still rejects)')
        if self.check_t1_pauses and self.t1_pause_uppercase_only:
            skipped_t.append('T1 lowercase-<p> tolerated (uppercase <P> still rejects)')
        skipped_g = [l for a, l in [
            ('gpt_require_t0_complete',       't0_complete'),
            ('gpt_require_t0_self_contained', 't0_self_contained'),
            ('gpt_require_t1_complete',       't1_complete'),
            ('gpt_require_t1_self_contained', 't1_self_contained'),
        ] if not getattr(self, a)]
        lines = [
            f"  min_t0_words    = {self.min_t0_words}",
            f"  min_after_words = {self.min_after_words}",
        ]
        if skipped_t:
            lines.append(f"  TIMING skipped  : {', '.join(skipped_t)}")
        if skipped_g:
            lines.append(f"  GPT skipped     : {', '.join(skipped_g)}")
        if not skipped_t and not skipped_g:
            lines.append("  (all criteria active)")
        return '\n'.join(lines)


# ── Regex helpers ──────────────────────────────────────────────────────────────

_PAUSE_RE       = re.compile(r'<[Pp]>', re.IGNORECASE)   # matches <P> and <p>
_PAUSE_UPPER_RE = re.compile(r'<P>')                       # matches only <P> (case-sensitive)
_OS_RE     = re.compile(r'\[os\b', re.IGNORECASE)
_EO_RE     = re.compile(r'\b(?:eo|oe)\]', re.IGNORECASE)

_NON_STANDARD_RE = re.compile(
    r'<(?:BN|VB|IR|CHKL|NU|NS|G|ST|TN|TP|TC|CT)[^>]*>', re.IGNORECASE)
_PII_RE = re.compile(
    r'<(?:LASTNAME|FIRSTNAME|FULLNAME|ADDRESS[^>]*|STREETNAME|TOWN|'
    r'ZIPCODE|PHONENUMBER|LN-[^>]*|FN-[^>]*)>',
    re.IGNORECASE)

_STRIP_RE = re.compile(
    r'\[os\b|\beo\]|\boe\]'
    r'|<[Bb]>',
    re.IGNORECASE)

# Matches caller/librarian but excludes "non-speaking-*-activity" participants
_VALID_SPKR_RE = re.compile(r'^(?!non-speaking)\S*\b(?:caller|librarian)\b', re.IGNORECASE)


# ── Text helpers ───────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Strip all TRS markup and normalise whitespace, keeping spoken words."""
    t = _PAUSE_RE.sub(' ', raw)
    t = _STRIP_RE.sub(' ', t)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'&\w+;', ' ', t)
    t = re.sub(r'\\\\', ' ', t)       # Transcriber line-end markers
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def count_words(raw: str) -> int:
    return len(clean_text(raw).split())


def has_non_standard(raw: str) -> bool:
    return bool(_NON_STANDARD_RE.search(raw) or _PII_RE.search(raw))


def has_internal_pause(raw: str, uppercase_only: bool = False) -> bool:
    """Return True if <P> appears between content words (not only leading/trailing).

    Leading/trailing pauses are tolerated — these are common in Loqui (thinking pauses,
    librarian looking up information) and do not indicate the turn is disfluent.
    Only a mid-utterance pause would break the floor-holding pattern.

    When uppercase_only=True, only uppercase <P> (longer/marked pause) triggers a
    rejection; lowercase <p> (short pause) is tolerated even mid-utterance.
    """
    # Remove leading pauses (both cases)
    stripped = re.sub(r'^(\s*<[Pp]>\s*)+', '', raw)
    # Remove trailing pauses (both cases)
    stripped = re.sub(r'(\s*<[Pp]>\s*)+$', '', stripped)
    p_re = _PAUSE_UPPER_RE if uppercase_only else _PAUSE_RE
    return bool(p_re.search(stripped))


def _strip_raw(text: str) -> str:
    """Strip \\ markers and normalise whitespace."""
    t = re.sub(r'\\\\', ' ', text)
    return re.sub(r'\s+', ' ', t).strip()


# ── TRS parser (full — single + overlap turns) ─────────────────────────────────

def _collect_single_text(turn_el) -> str:
    """Extract raw text from a single-speaker <Turn> element."""
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
    Parse a .final.trs file into ALL valid-speaker turns — both single-speaker
    and multi-speaker overlap turns — preserving original order.

    Single-speaker turn  (is_overlap=False):
      spkr_id, spkr_name, turn_start, turn_end, raw_text

    Multi-speaker overlap turn  (is_overlap=True):
      speakers     : [spkr_id, ...]
      turn_start, turn_end
      who_blocks   : {spkr_id → raw_text}
        — content comes from the TAIL of each self-closing <Who nb="N" /> tag.

    Only turns where every listed speaker is a valid caller/librarian are kept.
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
            # ── Single-speaker turn ───────────────────────────────────────────
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
            # ── Multi-speaker overlap turn ────────────────────────────────────
            if set(valid_in) != set(spk_attr):
                continue   # non-caller/librarian speaker present
            who_blocks: dict[str, str] = {}
            for who_el in turn_el.findall('Who'):
                nb  = int(who_el.get('nb', 1)) - 1   # 0-indexed
                if 0 <= nb < len(spk_attr):
                    sid = spk_attr[nb]
                    if sid in valid_spkrs:
                        raw = re.sub(r'\s+', ' ', _strip_raw(who_el.tail or '')).strip()
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


# ── Same-speaker turn aggregation ─────────────────────────────────────────────

_MAX_SAME_SPK_MERGE_GAP = 0.02   # seconds — absorb zero-gap transcription splits

def aggregate_consecutive_same_speaker(turns: list[dict]) -> tuple[list[dict], int]:
    """
    Merge consecutive non-overlap single-speaker turns from the same speaker
    when they are immediately adjacent (gap ≤ _MAX_SAME_SPK_MERGE_GAP).

    This handles transcription tool artifacts where one continuous utterance is
    split into two adjacent Turn elements with the same speaker and zero gap.
    It does NOT merge turns separated by an overlap turn.

    Returns (merged_turns, n_merges).
    """
    if not turns:
        return turns, 0

    merged: list[dict] = []
    n_merges = 0
    i = 0
    while i < len(turns):
        cur = turns[i]
        if (not cur.get('is_overlap')
                and i + 1 < len(turns)
                and not turns[i + 1].get('is_overlap')
                and turns[i + 1]['spkr_id'] == cur['spkr_id']
                and turns[i + 1]['turn_start'] - cur['turn_end'] <= _MAX_SAME_SPK_MERGE_GAP):
            nxt = turns[i + 1]
            merged.append({
                'is_overlap': False,
                'spkr_id':   cur['spkr_id'],
                'spkr_name': cur['spkr_name'],
                'turn_start': cur['turn_start'],
                'turn_end':   nxt['turn_end'],
                'raw_text':   (cur['raw_text'] + ' ' + nxt['raw_text']).strip(),
            })
            n_merges += 1
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged, n_merges


# ── Step 1: Timing Filter ──────────────────────────────────────────────────────

def find_floor_bc_patterns(turns: list[dict],
                           file_id: str,
                           task: str = 'backchannel',
                           cfg: 'FilterConfig | None' = None,
                           ) -> tuple[list[dict], dict]:
    """
    Scan the FULL turn list (single + overlap) for the floor-hold + BC pattern.
    Two encodings of the backchannel are detected simultaneously:

      Pattern A — Sequential (B speaks after A pauses briefly):
        turns[i-1] : single B  (T0 — quality-gated for floor_holding/both only)
        turns[i]   : single A  (T_before, ≥ MIN_BEFORE_WORDS, no internal <P>)
        turns[i+1] : single B  (T_bc, ≤ MAX_BC_WORDS, no <P>)
        turns[i+2] : single A  (T_after, ≥ MIN_AFTER_WORDS, no internal <P>)

      Pattern B — Overlap (B speaks simultaneously while A continues):
        turns[i-1] : single B       (T0 — quality-gated for floor_holding/both only)
        turns[i]   : single A       (T_before)
        turns[i+1] : overlap A+B    (B's Who block ≤ MAX_BC_WORDS;
                                     A's Who block ≥ MIN_DURING_WORDS)
        turns[i+2] : single A       (T_after)

      Pattern C — Who-block split (B's BC is embedded in an overlap turn; A's
        T_before is the text in A's Who block BEFORE the [os marker):
        turns[i-1] : any (T0 context / floor_holding quality gate)
        turns[i]   : overlap A+B where:
                        A's Who block has ≥ MIN_T0_WORDS before [os  (T_before)
                        B's Who block (stripped) ≤ MAX_BC_WORDS       (BC)
        turns[i+1] : single A       (T_after, starts within MAX_WHOBLOCK_GAP)

    bc_subtype values: 'sequential' | 'overlap' | 'whoblock'

    For task='backchannel', T0 checks are skipped unless cfg.check_t0 is set.
    For task='floor_holding' or 'both', the full T0 quality filter applies.
    """
    if cfg is None:
        cfg = FilterConfig(
            check_t0           = task in ('floor_holding', 'both'),
            check_t0_min_words = task in ('floor_holding', 'both'),
            check_t0_pauses    = task in ('floor_holding', 'both'),
        )

    stats = {
        'windows_checked':   0,
        # Pattern-level rejections
        'not_baba':          0,
        'before_ovl_b_short': 0,   # overlap T_before where B ≤ MAX_BC_WORDS (ambiguous)
        # T0 rejections (floor_holding/both only)
        't0_non_std':        0,
        't0_few_words':      0,
        't0_has_pause':      0,
        # T_before rejections (anonymisation only; pause unified below)
        'before_non_std':    0,
        # BC rejections — sequential
        'seq_bc_non_std':    0,
        'seq_bc_empty':      0,
        'seq_bc_too_many':   0,
        'seq_bc_has_pause':  0,
        # BC rejections — overlap
        'ovl_bc_not_b':      0,
        'ovl_bc_non_std':    0,
        'ovl_bc_too_many':   0,
        'ovl_bc_empty':      0,
        'ovl_bc_has_pause':  0,
        'ovl_during_short':  0,
        'ovl_during_pause':  0,
        # T1 unified internal-pause (T_before or T_after)
        't1_has_pause':      0,
        # T_after rejections (shared)
        'after_non_std':     0,
        'after_few_words':   0,
        # Final (A/B/seq)
        'duplicate':         0,
        'passed_sequential': 0,
        'passed_overlap':    0,
        # Pattern C — Who-block split
        'who_bc_empty':      0,   # B's Who block is empty after stripping
        'who_bc_too_many':   0,   # B's Who block > MAX_BC_WORDS
        'who_bc_has_pause':  0,   # B's Who block contains <P>
        'who_bc_non_std':    0,   # B's Who block has non-standard markup
        'who_before_short':  0,   # A's text before [os < MIN_T0_WORDS
        'who_t0_few_words':  0,   # T0 too few words (floor_holding)
        'who_t0_has_pause':  0,   # T0 has pause (floor_holding)
        'who_t0_non_std':    0,   # T0 has anonymisation (floor_holding)
        'who_t1_anon':       0,   # T_before or T_after has anonymisation
        'who_t1_pause':      0,   # T_before or T_after has internal <P>
        'who_after_missing': 0,   # no suitable T_after turn found
        'who_after_non_std': 0,   # T_after has non-standard markup
        'who_after_short':   0,   # T_after < min_after_words
        'passed_whoblock':   0,
        'passed':            0,
    }

    # Maximum gap (s) between overlap turn end and the next A turn for Pattern C.
    MAX_WHOBLOCK_GAP = 2.0

    candidates: list[dict] = []
    seen: set = set()

    for i in range(1, len(turns) - 2):
        t0_el       = turns[i - 1]
        t_before_el = turns[i]
        bc_el       = turns[i + 1]
        t_after_el  = turns[i + 2]

        stats['windows_checked'] += 1

        # T_after must always be single-speaker A
        if t_after_el.get('is_overlap'):
            stats['not_baba'] += 1
            continue

        # Derive spk_a and before_raw.
        # By default T_before must be single-speaker A (strict).
        # When cfg.allow_t_before_overlap is True, T_before may also be an
        # overlap A+B turn, provided B speaks > MAX_BC_WORDS there.
        t_before_is_overlap = t_before_el.get('is_overlap', False)
        if t_before_is_overlap and not cfg.allow_t_before_overlap:
            stats['not_baba'] += 1
            continue

        if not t_before_is_overlap:
            spk_a      = t_before_el['spkr_id']
            before_raw = t_before_el.get('raw_text', '')
        else:
            # Overlap T_before: A's portion is our T_before content.
            spk_a          = t_after_el['spkr_id']
            ovl_spks_b4    = t_before_el.get('speakers', [])
            who_b4         = t_before_el.get('who_blocks', {})
            if len(ovl_spks_b4) != 2 or spk_a not in ovl_spks_b4:
                stats['not_baba'] += 1
                continue
            spk_b_in_b4 = next(s for s in ovl_spks_b4 if s != spk_a)
            if count_words(who_b4.get(spk_b_in_b4, '')) <= MAX_BC_WORDS:
                # B only said ≤ MAX_BC_WORDS in T_before → ambiguous (B is BCing A),
                # not a substantive overlap that could serve as T_before.
                stats['before_ovl_b_short'] += 1
                continue
            before_raw = who_b4.get(spk_a, '')

        if cfg.check_t0:
            # floor_holding/both: T0 must be single-speaker B (full B-A-B-A)
            if t0_el.get('is_overlap'):
                stats['not_baba'] += 1
                continue
            spk_b      = t0_el['spkr_id']
            spk_b_name = t0_el.get('spkr_name', spk_b)
            if spk_a == spk_b or t_after_el['spkr_id'] != spk_a:
                stats['not_baba'] += 1
                continue
        else:
            # backchannel: T0 is context only — derive spk_b from the BC element
            spk_b_name = ''
            if not bc_el.get('is_overlap'):
                spk_b      = bc_el.get('spkr_id', '')
                spk_b_name = bc_el.get('spkr_name', spk_b)
            else:
                ovl_spks = bc_el.get('speakers', [])
                spk_b = next((s for s in ovl_spks if s != spk_a), '') \
                        if len(ovl_spks) == 2 else ''
            if not spk_b or spk_b == spk_a or t_after_el['spkr_id'] != spk_a:
                stats['not_baba'] += 1
                continue

        # Determine backchannel sub-type from bc_el
        bc_subtype: str | None = None
        bc_raw     = ''
        during_raw = ''

        if not bc_el.get('is_overlap'):
            # ── Sequential candidate ──────────────────────────────────────────
            if bc_el['spkr_id'] == spk_b:
                bc_subtype = 'sequential'
                bc_raw = bc_el['raw_text']
        else:
            # ── Overlap candidate ─────────────────────────────────────────────
            ovl_spks = bc_el.get('speakers', [])
            if len(ovl_spks) == 2 and spk_a in ovl_spks and spk_b in ovl_spks:
                bc_subtype = 'overlap'
                who = bc_el.get('who_blocks', {})
                bc_raw     = who.get(spk_b, '')
                during_raw = who.get(spk_a, '')

        if bc_subtype is None:
            stats['not_baba'] += 1
            continue

        # ── T0 quality (only when cfg.check_t0) ──────────────────────────────
        if t0_el.get('is_overlap'):
            who = t0_el.get('who_blocks', {})
            t0_raw = ' '.join(who.values())
        else:
            t0_raw = t0_el.get('raw_text', '')
        if cfg.check_t0:
            if has_non_standard(t0_raw):
                stats['t0_non_std'] += 1
                continue
            if cfg.check_t0_min_words and count_words(t0_raw) < cfg.min_t0_words:
                stats['t0_few_words'] += 1
                continue
            _t0_p_re = _PAUSE_UPPER_RE if cfg.t0_pause_uppercase_only else _PAUSE_RE
            if cfg.check_t0_pauses and _t0_p_re.search(t0_raw):
                stats['t0_has_pause'] += 1
                continue

        # ── T_before: anonymisation check (skippable via check_t1_anon) ───────
        # before_raw already set above (from raw_text or A's who_block for overlap T_before)
        if cfg.check_t1_anon and has_non_standard(before_raw):
            stats['before_non_std'] += 1
            continue

        # ── BC quality (sub-type specific; always enforced — defines what BC is) ──
        if bc_subtype == 'sequential':
            if has_non_standard(bc_raw):
                stats['seq_bc_non_std'] += 1
                continue
            bc_words = count_words(bc_raw)
            if bc_words == 0:
                stats['seq_bc_empty'] += 1
                continue
            if bc_words > MAX_BC_WORDS:
                stats['seq_bc_too_many'] += 1
                continue
            if _PAUSE_RE.search(bc_raw):
                stats['seq_bc_has_pause'] += 1
                continue
            bc_start = bc_el['turn_start']
            bc_end   = bc_el['turn_end']
            gap      = t_after_el['turn_start'] - t_before_el['turn_end']

        else:  # overlap
            if not bc_raw:
                stats['ovl_bc_not_b'] += 1
                continue
            if has_non_standard(bc_raw):
                stats['ovl_bc_non_std'] += 1
                continue
            bc_words = count_words(bc_raw)
            if bc_words == 0:
                stats['ovl_bc_empty'] += 1
                continue
            if bc_words > MAX_BC_WORDS:
                stats['ovl_bc_too_many'] += 1
                continue
            if _PAUSE_RE.search(bc_raw):
                stats['ovl_bc_has_pause'] += 1
                continue
            if count_words(during_raw) < MIN_DURING_WORDS:
                stats['ovl_during_short'] += 1
                continue
            if _PAUSE_RE.search(during_raw):
                stats['ovl_during_pause'] += 1
                continue
            bc_start = bc_el['turn_start']
            bc_end   = bc_el['turn_end']
            gap      = t_after_el['turn_start'] - t_before_el['turn_end']

        # ── T1 internal-pause check (skippable) ──────────────────────────────
        # Uses has_internal_pause() — only rejects if <P> sits between content words.
        after_raw = t_after_el['raw_text']
        if cfg.check_t1_pauses and (
            has_internal_pause(before_raw, cfg.t1_pause_uppercase_only) or
            has_internal_pause(after_raw,  cfg.t1_pause_uppercase_only)
        ):
            stats['t1_has_pause'] += 1
            continue

        # ── T_after quality ───────────────────────────────────────────────────
        if cfg.check_t1_anon and has_non_standard(after_raw):
            stats['after_non_std'] += 1
            continue
        if cfg.check_after_min_words and count_words(after_raw) < cfg.min_after_words:
            stats['after_few_words'] += 1
            continue

        # ── De-duplicate ──────────────────────────────────────────────────────
        key = (file_id, spk_a, round(t_before_el['turn_start'], 2), bc_subtype)
        if key in seen:
            stats['duplicate'] += 1
            continue
        seen.add(key)

        stats[f'passed_{bc_subtype}'] += 1
        stats['passed'] += 1

        cand = {
            'type':        'floor_backchannel',
            'bc_subtype':  bc_subtype,
            'file_id':     file_id,
            'spk_a':       spk_a,
            'spk_b':       spk_b,
            'spk_a_name':  t_before_el.get('spkr_name', t_after_el.get('spkr_name', spk_a)),
            'spk_b_name':  spk_b_name,

            't0_idx':      i - 1,
            'before_idx':  i,
            'bc_idx':      i + 1,
            'after_idx':   i + 2,

            't0_start':     round(t0_el['turn_start'],       3),
            't0_end':       round(t0_el['turn_end'],         3),
            'before_start': round(t_before_el['turn_start'], 3),
            'before_end':   round(t_before_el['turn_end'],   3),
            'bc_start':     round(bc_start,                  3),
            'bc_end':       round(bc_end,                    3),
            'after_start':  round(t_after_el['turn_start'],  3),
            'after_end':    round(t_after_el['turn_end'],    3),

            'bc_words':    bc_words,
            'bc_duration': round(bc_end - bc_start, 3),
            'bc_gap':      round(gap, 3),

            't0_text':     clean_text(t0_raw),
            'before_text': clean_text(before_raw),
            'bc_text':     clean_text(bc_raw),
            'after_text':  clean_text(after_raw),

            'gpt_valid':             None,
            'gpt_task':              None,
            'turn0_complete':        None,
            'turn0_self_contained':  None,
            'turn1_complete':        None,
            'turn1_self_contained':  None,
            'floor_holding_label':   None,
            'backchannel_type':      None,
            'gpt_reason':            None,
        }
        if bc_subtype == 'overlap':
            cand['during_text'] = clean_text(during_raw)

        candidates.append(cand)

    # ── Pattern C: Who-block split ────────────────────────────────────────────
    # Scan overlap turns where A's Who block contains text before [os (T_before)
    # and B's Who block is a short backchannel. T_after must be the immediately
    # following single-speaker A turn (within MAX_WHOBLOCK_GAP seconds).
    for i in range(1, len(turns) - 1):
        ovl_el      = turns[i]
        t_after_el  = turns[i + 1]

        if not ovl_el.get('is_overlap'):
            continue
        if t_after_el.get('is_overlap'):
            continue

        spk_attr = ovl_el.get('speakers', [])
        if len(spk_attr) != 2:
            continue
        who = ovl_el.get('who_blocks', {})

        # Try both A/B role assignments
        matched = False
        for spk_a, spk_b in [(spk_attr[0], spk_attr[1]),
                              (spk_attr[1], spk_attr[0])]:
            if matched:
                break
            if t_after_el.get('spkr_id') != spk_a:
                continue

            a_who_raw = who.get(spk_a, '')
            b_who_raw = who.get(spk_b, '')

            # B's Who block → BC text
            os_in_b = _OS_RE.search(b_who_raw)
            eo_in_b = _EO_RE.search(b_who_raw)
            # Strip [os/eo] markers to get clean BC text
            bc_raw = re.sub(r'\[os\b|\b(?:eo|oe)\]', ' ', b_who_raw, flags=re.IGNORECASE)
            bc_raw = re.sub(r'\\\\', ' ', bc_raw)
            bc_raw = re.sub(r'\s+', ' ', bc_raw).strip()

            if has_non_standard(bc_raw):
                stats['who_bc_non_std'] += 1
                continue
            bc_words = count_words(bc_raw)
            if bc_words == 0:
                stats['who_bc_empty'] += 1
                continue
            if bc_words > MAX_BC_WORDS:
                stats['who_bc_too_many'] += 1
                continue
            if _PAUSE_RE.search(b_who_raw):
                stats['who_bc_has_pause'] += 1
                continue

            # A's Who block → T_before is text before [os
            os_in_a = _OS_RE.search(a_who_raw)
            if not os_in_a:
                continue  # no [os in A's Who → not Pattern C
            before_raw_who = a_who_raw[:os_in_a.start()]
            before_raw_who = re.sub(r'\\\\', ' ', before_raw_who)
            before_raw_who = re.sub(r'\s+', ' ', before_raw_who).strip()

            before_words = count_words(before_raw_who)
            if before_words < cfg.min_t0_words:
                stats['who_before_short'] += 1
                continue

            # Gap between overlap turn end and T_after start
            gap = t_after_el['turn_start'] - ovl_el['turn_end']
            if gap > MAX_WHOBLOCK_GAP:
                stats['who_after_missing'] += 1
                continue

            # T0 (the turn immediately before the overlap)
            t0_el = turns[i - 1]
            if t0_el.get('is_overlap'):
                t0_raw = ' '.join(t0_el.get('who_blocks', {}).values())
            else:
                t0_raw = t0_el.get('raw_text', '')
            t0_spkr = (t0_el.get('spkr_id') if not t0_el.get('is_overlap') else '')

            if cfg.check_t0:
                if t0_el.get('is_overlap') or t0_spkr != spk_b:
                    continue  # T0 must be single-speaker B
                if has_non_standard(t0_raw):
                    stats['who_t0_non_std'] += 1
                    continue
                if cfg.check_t0_min_words and count_words(t0_raw) < cfg.min_t0_words:
                    stats['who_t0_few_words'] += 1
                    continue
                _t0_p_re = _PAUSE_UPPER_RE if cfg.t0_pause_uppercase_only else _PAUSE_RE
                if cfg.check_t0_pauses and _t0_p_re.search(t0_raw):
                    stats['who_t0_has_pause'] += 1
                    continue
                spk_b_name = t0_el.get('spkr_name', spk_b)
            else:
                spk_b_name = ''

            # T_after quality
            after_raw = t_after_el['raw_text']
            if cfg.check_t1_anon and (has_non_standard(before_raw_who) or
                                       has_non_standard(after_raw)):
                stats['who_t1_anon'] += 1
                continue
            if cfg.check_t1_pauses and (
                has_internal_pause(before_raw_who, cfg.t1_pause_uppercase_only) or
                has_internal_pause(after_raw,      cfg.t1_pause_uppercase_only)
            ):
                stats['who_t1_pause'] += 1
                continue
            if cfg.check_t1_anon and has_non_standard(after_raw):
                stats['who_after_non_std'] += 1
                continue
            if cfg.check_after_min_words and count_words(after_raw) < cfg.min_after_words:
                stats['who_after_short'] += 1
                continue

            # De-duplicate
            key = (file_id, spk_a, round(ovl_el['turn_start'], 2), 'whoblock')
            if key in seen:
                stats['duplicate'] += 1
                continue
            seen.add(key)

            stats['passed_whoblock'] += 1
            stats['passed'] += 1
            matched = True

            cand = {
                'type':        'floor_backchannel',
                'bc_subtype':  'whoblock',
                'file_id':     file_id,
                'spk_a':       spk_a,
                'spk_b':       spk_b,
                'spk_a_name':  '',
                'spk_b_name':  spk_b_name,

                't0_idx':      i - 1,
                'before_idx':  i,
                'bc_idx':      i,       # same turn as before (overlap)
                'after_idx':   i + 1,

                't0_start':     round(t0_el['turn_start'],     3),
                't0_end':       round(t0_el['turn_end'],       3),
                # T_before audio ≈ full overlap turn (A's channel)
                'before_start': round(ovl_el['turn_start'],    3),
                'before_end':   round(ovl_el['turn_end'],      3),
                # BC audio ≈ full overlap turn (B's channel); same window
                'bc_start':     round(ovl_el['turn_start'],    3),
                'bc_end':       round(ovl_el['turn_end'],      3),
                'after_start':  round(t_after_el['turn_start'], 3),
                'after_end':    round(t_after_el['turn_end'],   3),

                'bc_words':    bc_words,
                'bc_duration': round(ovl_el['turn_end'] - ovl_el['turn_start'], 3),
                'bc_gap':      round(gap, 3),

                't0_text':     clean_text(t0_raw),
                'before_text': clean_text(before_raw_who),
                'bc_text':     clean_text(bc_raw),
                'after_text':  clean_text(after_raw),

                'gpt_valid':             None,
                'gpt_task':              None,
                'turn0_complete':        None,
                'turn0_self_contained':  None,
                'turn1_complete':        None,
                'turn1_self_contained':  None,
                'floor_holding_label':   None,
                'backchannel_type':      None,
                'gpt_reason':            None,
            }
            candidates.append(cand)

    return candidates, stats


# ── Step 2: GPT Classification ─────────────────────────────────────────────────

def _label_block(guidance: dict) -> str:
    return "\n".join(f"   - {name}: {text}" for name, text in guidance.items())


_EPISODE_HEADER = """\
You are a dialogue analysis expert labelling a floor-holding + backchannel \
episode from a telephone conversation between a caller and a librarian.

Structure of the episode:
  Turn 0  (Speaker B) is the previous turn that prompted Speaker A to respond.
  Turn 1a (Speaker A) is the start of A's response — A holds the floor.
  BACKCHANNEL (Speaker B) — {bc_subtype_desc}
  Turn 1b (Speaker A) is A's continuation immediately after the backchannel.

Note: there is NO punctuation in this transcript (spoken dialogue).
Turn 1a{during_note} + Turn 1b together form one continuous floor-holding turn by A.

--- Turn 0 (Speaker B — the trigger) ---
"{t0}"

--- Turn 1a (Speaker A — before the backchannel) ---
"{before}"

--- BACKCHANNEL (Speaker B — {bc_words} word(s)) ---
"{bc}"{during_block}

--- Turn 1b (Speaker A — continues after the backchannel) ---
"{after}"

---
Evaluate each field in order. Complete ALL fields even if an earlier answer \
is false — just set is_valid to false at the end.\
"""

# ── Modular GPT question blocks ────────────────────────────────────────────────
# Use {n} for auto-numbering (replaced by .format(n=n) per block).
# Use {{floor_holding_block}} / {{backchannel_block}} so they survive that call
# and become {floor_holding_block} / {backchannel_block} for the final .format().

_Q_T0_COMPLETE = """\
{n}. TURN0_COMPLETE: Is Turn 0 (the trigger) a complete, well-formed utterance \
(not a fragment)?

"""

_Q_T0_SELF_CONTAINED = """\
{n}. TURN0_SELF_CONTAINED: Can Turn 0 be fully understood WITHOUT any prior \
conversational context?
   - Reject if it requires knowing what was said before to make sense.
   - Reject if it is only interpretable as a reply to an unseen utterance.
   - Accept if it introduces a complete, interpretable thought or question on its own.

"""

_Q_T1_COMPLETE = """\
{n}. TURN1_COMPLETE: Is the combined Turn 1 (Turn 1a + Turn 1b) a complete, \
well-formed utterance (not a fragment)?

"""

_Q_T1_SELF_CONTAINED = """\
{n}. TURN1_SELF_CONTAINED: Can the combined Turn 1 (Turn 1a + Turn 1b) be fully \
understood WITHOUT any prior conversational context? Be strict:
   - Reject if it contains references unclear without context (e.g. "that one", \
"it", "those").
   - Reject if it is only interpretable as a direct answer to an unseen question, \
or a bare reaction.
   - Accept only if it introduces a complete, interpretable thought.

"""

_Q_FLOOR_HOLDING = """\
{n}. FLOOR_HOLDING_LABEL: Compare what A said AFTER the backchannel (Turn 1b) to \
what A was doing BEFORE it (Turn 1a), in light of the backchannel, and choose \
exactly one:
{{floor_holding_block}}

"""

_Q_BACKCHANNEL_TYPE = """\
{n}. BACKCHANNEL_TYPE: From the combined Turn 1 context and the backchannel \
content, choose the single best type:
{{backchannel_block}}

"""


def build_gpt_prompt(cfg: 'FilterConfig') -> str:
    """Assemble the GPT prompt from active question blocks.
    FLOOR_HOLDING_LABEL and BACKCHANNEL_TYPE are always asked (classification labels).
    The *_complete / *_self_contained questions are only asked when their criterion
    is active so GPT does not invent values for skipped fields.
    """
    question_blocks = []
    if cfg.gpt_require_t0_complete:       question_blocks.append(_Q_T0_COMPLETE)
    if cfg.gpt_require_t0_self_contained: question_blocks.append(_Q_T0_SELF_CONTAINED)
    if cfg.gpt_require_t1_complete:       question_blocks.append(_Q_T1_COMPLETE)
    if cfg.gpt_require_t1_self_contained: question_blocks.append(_Q_T1_SELF_CONTAINED)
    if cfg.task != 'backchannel':
        question_blocks.append(_Q_FLOOR_HOLDING)
    question_blocks.append(_Q_BACKCHANNEL_TYPE)

    # Auto-number questions; each block uses {n} → replaced via .format(n=n).
    # {{floor_holding_block}} / {{backchannel_block}} survive this call and become
    # {floor_holding_block} / {backchannel_block} for the final .format().
    n = 1
    numbered_parts = []
    for blk in question_blocks:
        numbered_parts.append(blk.format(n=n))
        n += 1

    # Validity condition summary
    active_validity = []
    if cfg.gpt_require_t0_complete:       active_validity.append("turn0_complete=true")
    if cfg.gpt_require_t0_self_contained: active_validity.append("turn0_self_contained=true")
    if cfg.gpt_require_t1_complete:       active_validity.append("turn1_complete=true")
    if cfg.gpt_require_t1_self_contained: active_validity.append("turn1_self_contained=true")
    validity_cond = (", ".join(active_validity)
                     if active_validity else "none (all GPT criteria skipped — always valid)")

    # JSON schema — only include fields that were actually asked
    json_fields: list[str] = []
    if cfg.gpt_require_t0_complete:       json_fields.append('  "turn0_complete": true/false,')
    if cfg.gpt_require_t0_self_contained: json_fields.append('  "turn0_self_contained": true/false,')
    if cfg.gpt_require_t1_complete:       json_fields.append('  "turn1_complete": true/false,')
    if cfg.gpt_require_t1_self_contained: json_fields.append('  "turn1_self_contained": true/false,')
    if cfg.task != 'backchannel':
        json_fields.append('  "floor_holding_label": "continue_current|continue_shorten|continue_simplify|none",')
    json_fields += [
        '  "backchannel_type": "acknowledgement|understanding|agreement|uncertainty|disagreement|none",',
        '  "is_valid": true/false,',
        '  "reason": "one sentence summary"',
    ]
    json_schema = '\n'.join(json_fields)

    # Use {{ / }} so these literal braces survive the final .format() call.
    validity_block = (
        f"{n}. IS_VALID: Set to true ONLY if ALL hold: {validity_cond}.\n\n"
        f"Respond ONLY with JSON (no markdown):\n"
        f"{{{{\n{json_schema}\n}}}}"
    )

    return (
        _EPISODE_HEADER
        + "\n\n"
        + "".join(numbered_parts)
        + validity_block
    )


def recompute_validity(cand: dict, cfg: 'FilterConfig') -> bool:
    """Compute is_valid from individual GPT fields + FilterConfig (ignores GPT's own is_valid)."""
    if cfg.gpt_require_t0_complete and not cand.get('turn0_complete'):
        return False
    if cfg.gpt_require_t0_self_contained and not cand.get('turn0_self_contained'):
        return False
    if cfg.gpt_require_t1_complete and not cand.get('turn1_complete'):
        return False
    if cfg.gpt_require_t1_self_contained and not cand.get('turn1_self_contained'):
        return False
    return True


class QuotaExceededError(RuntimeError):
    pass


def gpt_classify(candidates: list[dict], client: OpenAI,
                 cfg: 'FilterConfig | None' = None,
                 task: str = 'backchannel',
                 max_valid_per_label: int | None = None) -> tuple[list[dict], dict]:
    if cfg is None:
        cfg = FilterConfig()

    total = len(candidates)
    print(f"  SEND_TO_LLM floor_bc (task={task}): {total}")

    floor_block  = _label_block(FLOOR_HOLDING_GUIDANCE)
    bc_block     = _label_block(BACKCHANNEL_TYPE_GUIDANCE)
    prompt_tmpl  = build_gpt_prompt(cfg)
    label_counts: dict[str, int] = {}

    for i, cand in enumerate(candidates):
        if max_valid_per_label is not None:
            all_full = all(
                label_counts.get(lbl, 0) >= max_valid_per_label
                for lbl in FLOOR_HOLDING_LABELS
            )
            if all_full:
                print(f"    [{i+1}/{total}] all label budgets reached — stopping.")
                break

        if cand['bc_subtype'] == 'overlap':
            bc_subtype_desc = "B speaks SIMULTANEOUSLY with A (overlap turn)"
            during_note     = " (including the overlap portion)"
            during_block    = (f'\n--- A continues during overlap ---\n'
                               f'"{cand.get("during_text", "")}"')
        else:
            bc_subtype_desc = "a short sequential turn B inserts while A briefly pauses"
            during_note     = ""
            during_block    = ""

        msg = prompt_tmpl.format(
            t0                  = cand['t0_text'],
            before              = cand['before_text'],
            bc                  = cand['bc_text'],
            bc_words            = cand['bc_words'],
            after               = cand['after_text'],
            bc_subtype_desc     = bc_subtype_desc,
            during_note         = during_note,
            during_block        = during_block,
            floor_holding_block = floor_block,
            backchannel_block   = bc_block,
        )

        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": msg}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)

            # Store only the fields we actually asked for
            if cfg.gpt_require_t0_complete:
                cand['turn0_complete']       = result.get('turn0_complete')
            if cfg.gpt_require_t0_self_contained:
                cand['turn0_self_contained'] = result.get('turn0_self_contained')
            if cfg.gpt_require_t1_complete:
                cand['turn1_complete']       = result.get('turn1_complete')
            if cfg.gpt_require_t1_self_contained:
                cand['turn1_self_contained'] = result.get('turn1_self_contained')
            if cfg.task != 'backchannel':
                cand['floor_holding_label'] = result.get('floor_holding_label')
            cand['backchannel_type']     = result.get('backchannel_type')
            cand['gpt_reason']           = result.get('reason', '')
            cand['gpt_task']             = task

            valid = recompute_validity(cand, cfg)

            # Budget key: backchannel_type for backchannel task, floor_holding_label otherwise
            if cfg.task == 'backchannel':
                budget_key = cand.get('backchannel_type') or 'none'
            else:
                budget_key = cand.get('floor_holding_label') or 'none'
            if valid and max_valid_per_label is not None:
                if label_counts.get(budget_key, 0) >= max_valid_per_label:
                    valid = False
                    cand['gpt_reason'] += ' [demoted: budget reached]'

            cand['gpt_valid'] = valid
            if valid:
                label_counts[budget_key] = label_counts.get(budget_key, 0) + 1

            status = '✓' if valid else '✗'
            print(f"    [{i+1}/{total}] {status} "
                  + (f"floor={cand.get('floor_holding_label','?')} " if cfg.task != 'backchannel' else "")
                  + f"bc={cand.get('backchannel_type','?')} "
                  + f"— {cand['before_text'][:50]}")

        except Exception as e:
            err = str(e)
            if 'insufficient_quota' in err or 'rate_limit' in err.lower() or '429' in err:
                cand['gpt_valid']  = None
                cand['gpt_reason'] = f"API error: {e}"
                raise QuotaExceededError(
                    f"OpenAI quota exceeded at candidate {i}") from e
            print(f"    [WARN] GPT failed for candidate {i}: {e}")
            cand['gpt_valid']  = None
            cand['gpt_reason'] = f"API error: {e}"

    judged = [c for c in candidates if c.get('gpt_valid') is not None]
    gpt_stats: dict = {
        'total_sent':       total,
        'total_judged':     len(judged),
        'backchannel_type': dict(Counter(c.get('backchannel_type', '?') for c in judged)),
        'total_valid':      sum(1 for c in judged if c.get('gpt_valid')),
    }
    if cfg.task != 'backchannel':
        gpt_stats['floor_holding_label'] = dict(Counter(c.get('floor_holding_label', '?') for c in judged))
        gpt_stats['valid_by_label']      = dict(Counter(
            c.get('floor_holding_label', '?')
            for c in judged if c.get('gpt_valid')
        ))
    if cfg.gpt_require_t1_complete:
        gpt_stats['turn1_complete_false'] = sum(1 for c in judged if c.get('turn1_complete') is False)
    if cfg.gpt_require_t0_complete:
        gpt_stats['turn0_complete_false'] = sum(1 for c in judged if c.get('turn0_complete') is False)
    if cfg.gpt_require_t0_self_contained:
        gpt_stats['turn0_self_contained_false'] = sum(1 for c in judged if c.get('turn0_self_contained') is False)
    if cfg.gpt_require_t1_self_contained:
        gpt_stats['turn1_self_contained_false'] = sum(1 for c in judged if c.get('turn1_self_contained') is False)

    print(f"  VALID_AFTER_LLM: {gpt_stats['total_valid']}")
    print(f"GPT_STATS:{json.dumps(gpt_stats)}")
    return candidates, gpt_stats


# ── Step 3: Audio Export ───────────────────────────────────────────────────────

def export_candidates(candidates: list[dict],
                      ch0: np.ndarray, ch1: np.ndarray, sr: int,
                      spkr_channel: dict[str, int],
                      results_dir: Path,
                      task: str = 'both') -> None:
    """
    Export audio clips for each GPT-validated candidate into:

      results_dir/<backchannel_type>/                            (backchannel task)
      results_dir/<floor_holding_label>/<bc_subtype>/<bc_type>/  (floor_holding/both)
        <file_id>_bc<bc_idx>_t0.wav       — B trigger (B's channel)
        <file_id>_bc<bc_idx>_before.wav   — A before BC (A's channel)
        <file_id>_bc<bc_idx>_bc.wav       — BC turn / BC Who-block (B's channel)
        <file_id>_bc<bc_idx>_after.wav    — A after BC (A's channel)
        <file_id>_bc<bc_idx>_full.wav     — T0 start → T_after end (A's channel)
    """
    n0, n1 = len(ch0), len(ch1)

    def clip(audio: np.ndarray, n: int, start_s: float, end_s: float) -> np.ndarray:
        s = max(0, int(start_s * sr))
        e = min(n, int(end_s   * sr))
        return audio[s:e]

    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue

        bc_type    = cand.get('backchannel_type',    'unknown') or 'unknown'
        if task == 'backchannel':
            out_dir = results_dir / bc_type
        else:
            floor_lbl  = cand.get('floor_holding_label', 'unknown') or 'unknown'
            bc_subtype = cand.get('bc_subtype',          'unknown') or 'unknown'
            out_dir    = results_dir / floor_lbl / bc_subtype / bc_type
        out_dir.mkdir(parents=True, exist_ok=True)

        spk_a = cand['spk_a']
        spk_b = cand['spk_b']
        ch_a  = spkr_channel.get(spk_a, 0)
        ch_b  = spkr_channel.get(spk_b, 1)
        aud_a = ch0 if ch_a == 0 else ch1
        aud_b = ch0 if ch_b == 0 else ch1
        n_a, n_b = (n0, n1) if ch_a == 0 else (n1, n0)

        base = f"{cand['file_id']}_bc{cand['bc_idx']:04d}"
        clips = [
            ('t0',     aud_b, n_b, cand['t0_start'],     cand['t0_end']),
            ('before', aud_a, n_a, cand['before_start'], cand['before_end']),
            ('bc',     aud_b, n_b, cand['bc_start'],     cand['bc_end']),
            ('after',  aud_a, n_a, cand['after_start'],  cand['after_end']),
            ('full',   aud_a, n_a, cand['t0_start'],     cand['after_end']),
        ]
        for tag, audio, n_samples, st, en in clips:
            sf.write(str(out_dir / f"{base}_{tag}.wav"),
                     clip(audio, n_samples, st, en), sr)

        cand['audio_dir'] = str(out_dir)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract floor-holding + backchannel episodes from a Loqui recording.")
    parser.add_argument("--file_id",             required=True)
    parser.add_argument("--trs_path",            type=Path, required=True)
    parser.add_argument("--wav_path",            type=Path, required=True)
    parser.add_argument("--results_dir",         type=Path, required=True)
    parser.add_argument("--max_valid_per_label", type=int,
                        default=TARGET_PER_LABEL)
    parser.add_argument("--task", choices=['backchannel', 'floor_holding', 'both'],
                        default='backchannel',
                        help="GPT evaluation task: 'backchannel' validates T1 "
                             "self-containedness; 'floor_holding' validates T0 "
                             "self-containedness + T1 completeness; 'both' requires all.")
    parser.add_argument("--no_gpt",   action="store_true")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--force",    action="store_true")

    # ── Timing skip flags ──────────────────────────────────────────────────────
    parser.add_argument("--skip_t0",          action="store_true",
                        help="Skip all T0-side timing checks (master switch).")
    parser.add_argument("--skip_t0_words",    action="store_true",
                        help="Skip T0 min-word-count gate.")
    parser.add_argument("--skip_t0_pauses",   action="store_true",
                        help="Skip T0 pause gate entirely (allow <P> and <p>).")
    parser.add_argument("--t0_pause_uppercase_only", action="store_true",
                        help="Tolerate short pauses in T0: only uppercase <P> rejects T0; "
                             "lowercase <p> (brief hesitation) is allowed. "
                             "Ignored if --skip_t0_pauses is set.")
    parser.add_argument("--skip_t1_pauses",   action="store_true",
                        help="Skip T_before/T_after internal-pause gate "
                             "(leading/trailing pauses are always tolerated; "
                             "this skips the mid-utterance pause check too).")
    parser.add_argument("--t1_pause_uppercase_only", action="store_true",
                        help="Tolerate short pauses in T1: only uppercase <P> rejects "
                             "T_before/T_after; lowercase <p> (brief hesitation) is allowed. "
                             "Ignored if --skip_t1_pauses is set.")
    parser.add_argument("--skip_t1_anon",     action="store_true",
                        help="Skip T_before/T_after anonymisation-marker gate "
                             "(T1 = A's speech, not used as model input).")
    parser.add_argument("--skip_after_words", action="store_true",
                        help="Skip T_after min-word-count gate.")
    parser.add_argument("--allow_t_before_overlap", action="store_true",
                        help="Allow T_before to be an overlap turn (A+B) when B speaks "
                             "> MAX_BC_WORDS words in it. Default: T_before must be a "
                             "single-speaker A turn.")

    # ── Threshold overrides ────────────────────────────────────────────────────
    parser.add_argument("--min_t0_words",    type=int,   default=MIN_T0_WORDS)
    parser.add_argument("--min_after_words", type=int,   default=MIN_AFTER_WORDS)

    # ── GPT skip flags ─────────────────────────────────────────────────────────
    parser.add_argument("--skip_gpt_t0_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t0_self_contained", action="store_true")
    parser.add_argument("--skip_gpt_t1_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t1_self_contained", action="store_true")

    args = parser.parse_args()

    # ── Build FilterConfig ─────────────────────────────────────────────────────
    is_floor = args.task in ('floor_holding', 'both')
    cfg = FilterConfig(
        check_t0              = is_floor and not args.skip_t0,
        check_t0_min_words    = is_floor and not args.skip_t0 and not args.skip_t0_words,
        check_t0_pauses       = is_floor and not args.skip_t0 and not args.skip_t0_pauses,
        t0_pause_uppercase_only = (is_floor and not args.skip_t0 and not args.skip_t0_pauses
                                   and args.t0_pause_uppercase_only),
        check_t1_pauses          = not args.skip_t1_pauses,
        t1_pause_uppercase_only  = (not args.skip_t1_pauses and args.t1_pause_uppercase_only),
        check_t1_anon            = not args.skip_t1_anon,
        check_after_min_words = not args.skip_after_words,
        allow_t_before_overlap = args.allow_t_before_overlap,
        min_t0_words          = args.min_t0_words,
        min_after_words       = args.min_after_words,
        gpt_require_t0_complete       = is_floor and not args.skip_gpt_t0_complete,
        gpt_require_t0_self_contained = is_floor and not args.skip_gpt_t0_self_contained,
        gpt_require_t1_complete       = not args.skip_gpt_t1_complete,
        gpt_require_t1_self_contained = (
            args.task in ('backchannel', 'both') and not args.skip_gpt_t1_self_contained
        ),
        task = args.task,
    )

    print(f"\n── FilterConfig ──")
    print(cfg.describe())

    out_dir    = args.results_dir / args.file_id
    valid_path = out_dir / "valid.json"

    if valid_path.exists() and not args.force:
        print(f"{args.file_id}: already done (use --force to redo).")
        data = json.loads(valid_path.read_text())
        print(f"VALID_JSON:{json.dumps(data)}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse ──────────────────────────────────────────────────────────────────
    valid_spkrs, turns = parse_trs_full(args.trs_path)
    if not turns:
        print(f"{args.file_id}: no turns found.")
        print("TIMING_STATS:{}")
        print("VALID_JSON:[]")
        return

    # Merge consecutive same-speaker single turns (transcription artifacts)
    turns, n_agg = aggregate_consecutive_same_speaker(turns)

    n_single  = sum(1 for t in turns if not t.get('is_overlap'))
    n_overlap = sum(1 for t in turns if     t.get('is_overlap'))
    print(f"{args.file_id}: {len(valid_spkrs)} valid speakers, "
          f"{n_single} single-spkr + {n_overlap} overlap turns"
          + (f" ({n_agg} same-spkr turns merged)" if n_agg else ""))

    spkr_channel: dict[str, int] = {
        sid: (0 if 'caller' in name.lower() else 1)
        for sid, name in valid_spkrs.items()
    }

    # ── Step 1: Timing filter ──────────────────────────────────────────────────
    print(f"\n── Step 1: Timing filter ──")
    candidates, timing_stats = find_floor_bc_patterns(
        turns, args.file_id, task=args.task, cfg=cfg)
    n_seq = timing_stats.get('passed_sequential', 0)
    n_ovl = timing_stats.get('passed_overlap',    0)
    print(f"  Candidates found: {len(candidates)} (sequential={n_seq}, overlap={n_ovl})")
    print(f"TIMING_STATS:{json.dumps(timing_stats)}")

    f1_dir = out_dir / "filter1_timing"
    f1_dir.mkdir(exist_ok=True)
    (f1_dir / "all.json").write_text(json.dumps(candidates, indent=2))

    if args.no_gpt or not candidates:
        print("  Skipping GPT step.")
        print("VALID_JSON:[]")
        return

    # ── Step 2: GPT filter ─────────────────────────────────────────────────────
    print(f"\n── Step 2: GPT filter ({GPT_MODEL}) ──")
    client = OpenAI(api_key=OPENAI_API_KEY)

    gpt_stats: dict = {}
    try:
        candidates, gpt_stats = gpt_classify(
            candidates, client,
            cfg=cfg,
            task=args.task,
            max_valid_per_label=args.max_valid_per_label)
    except QuotaExceededError as e:
        print(f"  Quota exceeded: {e}")
        (out_dir / "filter2_gpt").mkdir(exist_ok=True)
        ((out_dir / "filter2_gpt") / "all_with_gpt_decisions.json").write_text(
            json.dumps(candidates, indent=2))
        import sys; sys.exit(QUOTA_EXIT_CODE)

    valid = [c for c in candidates if c.get('gpt_valid')]
    by_label: Counter = Counter(c.get('floor_holding_label', '?') for c in valid)
    by_bc:    Counter = Counter(c.get('backchannel_type',    '?') for c in valid)
    print(f"\n  Valid after GPT: {len(valid)}/{len(candidates)}")
    if by_label:
        print("    floor_holding_label: " +
              "  ".join(f"{k}={v}" for k, v in sorted(by_label.items())))
    if by_bc:
        print("    backchannel_type: " +
              "  ".join(f"{k}={v}" for k, v in sorted(by_bc.items())))

    f2_dir = out_dir / "filter2_gpt"
    f2_dir.mkdir(exist_ok=True)
    (f2_dir / "all_with_gpt_decisions.json").write_text(
        json.dumps(candidates, indent=2))
    if valid:
        (f2_dir / "valid.json").write_text(json.dumps(valid, indent=2))

    valid_path.write_text(json.dumps(valid, indent=2))

    # ── Step 3: Audio export ───────────────────────────────────────────────────
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

    export_candidates(valid, ch0, ch1, AUDIO_SR, spkr_channel, args.results_dir, task=args.task)
    n_clips = sum(5 for c in valid if c.get('audio_dir'))
    print(f"  Exported {len(valid)} × 5 clips → {args.results_dir}/")
    print(f"VALID_JSON:{json.dumps(valid)}")


if __name__ == "__main__":
    main()
