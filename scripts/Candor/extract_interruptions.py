#!/usr/bin/env python3
"""
CANDOR Interruption & Recovery Extractor — Backbiter Edition

Uses transcript_backbiter.csv instead of transcript_cliffhanger.csv.

Key differences from the Cliffhanger version:
  - Backbiter already absorbed all backchannel turns, so no backchannel
    filtering is needed: any surviving row is substantive speech.
  - Turns with fewer than MIN_WORDS_FILTER words are removed upfront.
  - Overlap is detected directly via the `overlap` column (True = this turn
    started before the previous turn ended).
  - T2 must NOT overlap T1 (overlap col = False).
  - T3 MUST overlap T2 (overlap col = True) with magnitude >= MIN_INTERRUPTION_OVERLAP.
  - T1 must NOT have internal pauses (word-to-word gap >= PAUSE_MIN); a pause
    inside T1 indicates a natural turn boundary rather than a single continuous
    utterance, so those candidates are rejected.

4-turn pattern:
  Turn1  Speaker A  – complete utterance (sentence-final punct, substantive)
  Turn2  Speaker B  – response to T1 (no overlap with T1, not a question)
  Turn3  Speaker A  – overlaps T2 (genuine interruption)
  Turn4  Speaker B  – recovery / response after interruption

Usage:
  python extract_interruptions.py --conv_dir <path>
  python extract_interruptions.py --gpt_only --conv_dir <path>
"""

import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_CONV_DIR = Path(
    "/data/gpfs/projects/punim2960/jiahengd/dataset/Candor"
    "/part_001/0d4a795e-7026-4d98-b6d0-739a2c7c1661"
)

PAUSE_MIN                = 0.2   # min gap to count as a pause (s)
PAUSE_MAX                = 1.0   # max gap to count as a pause (s)
MIN_WORDS_FILTER         = 3     # pre-filter: drop any turn with fewer words
MIN_SUBSTANTIVE_WORDS    = 4     # minimum content words for T1 / T4
MIN_INTERRUPTION_OVERLAP = 0.3   # T3 must overlap T2 by at least this many seconds
TIMESTAMP_NOISE_TOLERANCE = 0.05 # slack margin for "must not overlap" checks (T2/T1, T4/T3)
# Maximum allowed overlap at the T2/T1 and T4/T3 boundaries when the check is active.
# Mirrors the turn-taking extractor's 1.5s threshold; GPT turn1_complete and the
# three T3 booleans act as the semantic safety net for any false positives.
MAX_OVERLAP_TOLERANCE     = 1.5
MAX_T3_DURATION          = 10.0  # longer T3 = floor takeover, not interruption
T2_MIN_WORDS             = 4     # Turn2 must have >= this many words total
T2_MIN_PRE_WORDS         = 4     # Turn2 must have >= this many words BEFORE overlap
AUDIO_SR                = 16000
TARGET_PER_TYPE         = 500    # cap per recovery type at aggregation
RECOVERY_TYPES          = ["correct", "enrich", "shorten", "repeat",
                            "shift_topic", "ask_followup", "none"]
QUOTA_EXIT_CODE         = 2


# ── Per-run filter configuration ───────────────────────────────────────────────
# Every flag defaults to on. Pass --skip_XXX or --min_XXX to turn a criterion off.
@dataclass
class FilterConfig:
    # ── Pre-filter ─────────────────────────────────────────────────────────────
    min_words_filter: int = MIN_WORDS_FILTER   # --min_words_filter N (0 = off)

    # ── Timing criteria (False = criterion is skipped) ─────────────────────────
    check_abab:             bool = True  # --skip_abab
    check_t2_no_overlap_t1: bool = True  # --skip_t2_overlap_t1
    max_t2_t1_overlap:     float = MAX_OVERLAP_TOLERANCE   # --max_t2_t1_overlap N
    check_t3_overlap:       bool = True  # --skip_t3_overlap
    min_t3_overlap:        float = MIN_INTERRUPTION_OVERLAP  # --min_t3_overlap N
    check_t4_no_overlap_t3: bool = True  # --skip_t4_overlap_t3
    max_t4_t3_overlap:     float = MAX_OVERLAP_TOLERANCE   # --max_t4_t3_overlap N
    check_t1_sent_final:    bool = True  # --skip_t1_sent_final
    check_t1_min_words:     bool = True  # --skip_t1_words
    check_t1_pauses:        bool = True  # --skip_t1_pauses
    check_t2_not_question:  bool = True  # --skip_t2_question
    check_t2_min_words:     bool = True  # --skip_t2_words
    check_t2_pre_words:     bool = True  # --skip_t2_pre_words
    check_t1_backchannel:   bool = True  # --skip_t1_backchannel
    check_t2_backchannel:   bool = True  # --skip_t2_backchannel
    check_t3_backchannel:   bool = True  # --skip_t3_backchannel
    check_t3_pauses:        bool = True  # --skip_t3_pauses
    check_t3_max_duration:  bool = True  # --skip_t3_duration
    check_t4_min_words:     bool = True  # --skip_t4_words

    # ── GPT validity criteria (False = criterion ignored when deciding is_valid) ─
    gpt_require_t1_complete:          bool = True  # --skip_gpt_t1_complete
    gpt_require_t1_self_contained:    bool = True  # --skip_gpt_t1_self_contained
    gpt_require_t2_responds:          bool = True  # --skip_gpt_t2_responds
    gpt_require_t3_floor_intent:      bool = True  # --skip_gpt_t3_floor_intent
    gpt_require_t3_pragmatic_complete: bool = True  # --skip_gpt_t3_pragmatic_complete
    gpt_require_t3_mid_flow_onset:    bool = True  # --skip_gpt_t3_mid_flow
    gpt_require_t4_meaningful:        bool = True  # --skip_gpt_t4_meaningful

    def active_timing_checks(self) -> list[str]:
        """Return human-readable list of active timing criteria (for reports)."""
        mapping = [
            ('check_abab',              'ABAB alternation'),
            ('check_t2_no_overlap_t1',  'T2 must not overlap T1'),
            ('check_t3_overlap',        f'T3 overlap >= {self.min_t3_overlap}s'),
            ('check_t4_no_overlap_t3',  'T4 must not overlap T3'),
            ('check_t1_sent_final',     'T1 sentence-final punctuation'),
            ('check_t1_min_words',      'T1 minimum words'),
            ('check_t1_pauses',         'T1 no internal pauses'),
            ('check_t1_backchannel',    'T1 no backchannel from B'),
            ('check_t2_not_question',   'T2 not a question'),
            ('check_t2_min_words',      'T2 minimum words'),
            ('check_t2_pre_words',      'T2 pre-overlap words'),
            ('check_t2_backchannel',    'T2 no backchannel from A'),
            ('check_t3_backchannel',    'T3 no backchannel from B'),
            ('check_t3_pauses',         'T3 no internal pauses'),
            ('check_t3_max_duration',   'T3 duration limit'),
            ('check_t4_min_words',      'T4 minimum words'),
        ]
        return [label for attr, label in mapping if getattr(self, attr)]

    def active_gpt_checks(self) -> list[str]:
        mapping = [
            ('gpt_require_t1_complete',           'turn1_complete'),
            ('gpt_require_t1_self_contained',     'turn1_self_contained'),
            ('gpt_require_t2_responds',           'turn2_responds'),
            ('gpt_require_t3_floor_intent',       't3_floor_intent'),
            ('gpt_require_t3_pragmatic_complete', 't3_pragmatic_complete'),
            ('gpt_require_t3_mid_flow_onset',     't3_mid_flow_onset'),
            ('gpt_require_t4_meaningful',         'turn4_meaningful'),
        ]
        return [label for attr, label in mapping if getattr(self, attr)]

    def describe(self) -> str:
        lines = [f"  min_words_filter  = {self.min_words_filter}",
                 f"  min_t3_overlap    = {self.min_t3_overlap}s",
                 f"  max_t2_t1_overlap = {self.max_t2_t1_overlap}s",
                 f"  max_t4_t3_overlap = {self.max_t4_t3_overlap}s"]
        skipped_timing = [l for a, l in [
            ('check_abab', 'ABAB'), ('check_t2_no_overlap_t1', 'T2/T1 no-overlap'),
            ('check_t3_overlap', 'T3 overlap'), ('check_t4_no_overlap_t3', 'T4/T3 no-overlap'),
            ('check_t1_sent_final', 'T1 sent-final'), ('check_t1_min_words', 'T1 min-words'),
            ('check_t1_pauses', 'T1 pauses'), ('check_t1_backchannel', 'T1 backchannel'),
            ('check_t2_not_question', 'T2 not-question'),
            ('check_t2_min_words', 'T2 min-words'), ('check_t2_pre_words', 'T2 pre-words'),
            ('check_t2_backchannel', 'T2 backchannel'),
            ('check_t3_backchannel', 'T3 backchannel'),
            ('check_t3_pauses', 'T3 pauses'), ('check_t3_max_duration', 'T3 duration'),
            ('check_t4_min_words', 'T4 min-words'),
        ] if not getattr(self, a)]
        skipped_gpt = [l for a, l in [
            ('gpt_require_t1_complete',           't1_complete'),
            ('gpt_require_t1_self_contained',     't1_self_contained'),
            ('gpt_require_t2_responds',           't2_responds'),
            ('gpt_require_t3_floor_intent',       't3_floor_intent'),
            ('gpt_require_t3_pragmatic_complete', 't3_pragmatic_complete'),
            ('gpt_require_t3_mid_flow_onset',     't3_mid_flow_onset'),
            ('gpt_require_t4_meaningful',         't4_meaningful'),
        ] if not getattr(self, a)]
        if skipped_timing:
            lines.append(f"  SKIPPED timing : {', '.join(skipped_timing)}")
        if skipped_gpt:
            lines.append(f"  SKIPPED gpt    : {', '.join(skipped_gpt)}")
        if not skipped_timing and not skipped_gpt:
            lines.append("  (all criteria active — strict mode)")
        return '\n'.join(lines)


def recompute_validity(candidate: dict, cfg: FilterConfig) -> bool:
    """
    Compute is_valid purely from GPT's individual field-level answers.
    GPT no longer produces an is_valid field — this is the single source of truth.

    Returns False if the candidate has not been classified yet
    (turn2_interrupted is None, meaning GPT has not responded).
    """
    # Sentinel: if the always-evaluated field is None, GPT hasn't run yet
    if candidate.get('turn2_interrupted') is None:
        return False

    if cfg.gpt_require_t1_complete and not candidate.get('turn1_complete', False):
        return False
    if cfg.gpt_require_t1_self_contained and not candidate.get('turn1_self_contained', False):
        return False
    if cfg.gpt_require_t2_responds and not candidate.get('turn2_responds', False):
        return False
    if cfg.gpt_require_t3_floor_intent and not candidate.get('t3_floor_intent', False):
        return False
    if cfg.gpt_require_t3_pragmatic_complete and not candidate.get('t3_pragmatic_complete', False):
        return False
    if cfg.gpt_require_t3_mid_flow_onset and not candidate.get('t3_mid_flow_onset', False):
        return False
    if cfg.gpt_require_t4_meaningful and not candidate.get('turn4_meaningful', False):
        return False
    return True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL      = "gpt-5.4-2026-03-05"


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_speaker_words(trans_json: dict, ch_to_speaker: dict) -> dict[str, list]:
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


def get_turn_words(row, all_words: list[dict]) -> list[dict] | None:
    t_start = float(row['start'])
    t_stop  = float(row['stop'])
    words = [w for w in all_words if t_start <= w['start'] and w['end'] <= t_stop]
    return words if words else None



def words_to_text(words: list[dict]) -> str:
    return ' '.join(w['word'] for w in words)


def words_to_timed(words: list[dict]) -> str:
    return ' '.join(f'{w["word"]}({w["start"]:.2f}-{w["end"]:.2f}s)' for w in words)


def backchannel_from_row(row, expected_speaker: str | None = None,
                         before_time: float | None = None) -> list[dict]:
    """Return a timing dict for the backchannel embedded in a backbiter row.

    The backbiter CSV stores at most one backchannel window per turn in the
    columns: backchannel_speaker, backchannel_start, backchannel_stop.
    Returns an empty list if none is present, the speaker doesn't match, or
    the backchannel starts at or after `before_time` (used for T2 to exclude
    backchannels that start at/after T3 onset).

    The returned dict has {'start': float, 'end': float} so it is compatible
    with silence_other_speaker() and export_stereo_clip_masked().
    """
    import math
    count = int(row.get('backchannel_count', 0) or 0)
    if count == 0:
        return []
    bc_spk = row.get('backchannel_speaker')
    if expected_speaker and bc_spk != expected_speaker:
        return []
    try:
        bc_start = float(row['backchannel_start'])
        bc_end   = float(row['backchannel_stop'])
    except (TypeError, ValueError):
        return []
    if math.isnan(bc_start) or math.isnan(bc_end):
        return []
    if before_time is not None and bc_start >= before_time:
        return []
    return [{'start': bc_start, 'end': bc_end}]


# ── Step 1: Timing Filter ──────────────────────────────────────────────────────

def find_interruption_patterns(back: pd.DataFrame,
                                speaker_words: dict[str, list],
                                cfg: FilterConfig | None = None,
                                ) -> tuple[list[dict], dict]:
    """
    Scan backbiter turns for the 4-turn interruption pattern:
      T1(A) → T2(B) → T3(A, overlaps T2) → T4(B)
    Returns (candidates, timing_stats).
    Each criterion can be individually disabled via FilterConfig.
    """
    if cfg is None:
        cfg = FilterConfig()

    turns   = back.reset_index(drop=True)
    conv_id = back.attrs.get('conv_id', '')
    candidates = []
    seen: set = set()
    stats = {
        'windows_checked':    0,
        'not_abab':           0,
        't2_overlaps_t1':     0,
        't3_overlap_small':   0,
        't4_overlaps_t3':     0,
        't1_no_sent_final':   0,
        't1_few_words':       0,
        't1_has_pauses':      0,
        't1_has_backchannel': 0,
        't2_is_question':     0,
        't2_few_words':       0,
        't2_pre_words_short': 0,
        't2_has_backchannel': 0,
        't3_has_backchannel': 0,
        't3_has_pauses':      0,
        't3_too_long':        0,
        't3_empty':           0,
        't4_few_words':       0,
        'duplicate':          0,
        'passed':             0,
    }

    for i in range(len(turns) - 3):
        t1_row = turns.iloc[i]
        t2_row = turns.iloc[i + 1]
        t3_row = turns.iloc[i + 2]
        t4_row = turns.iloc[i + 3]

        spk_a = t1_row['speaker']
        spk_b = t2_row['speaker']

        stats['windows_checked'] += 1

        # ── Enforce strict A-B-A-B alternation ───────────────────────────────
        if cfg.check_abab:
            if spk_a == spk_b or t3_row['speaker'] != spk_a or t4_row['speaker'] != spk_b:
                stats['not_abab'] += 1
                continue

        # Use timestamps directly — the `overlap` column was computed against
        # the original unfiltered DataFrame and is unreliable after pre-filtering.
        t1_end   = float(t1_row['stop'])
        t2_start = float(t2_row['start'])
        t2_end   = float(t2_row['stop'])
        t3_start = float(t3_row['start'])
        t3_end   = float(t3_row['stop'])

        # ── T2/T1 overlap check ───────────────────────────────────────────────
        # Reject only when T2 starts more than max_t2_t1_overlap before T1 ends.
        # Default 1.5s (matching turn-taking); GPT turn1_complete filters the rest.
        if cfg.check_t2_no_overlap_t1:
            if t2_start < t1_end - cfg.max_t2_t1_overlap:
                stats['t2_overlaps_t1'] += 1
                continue

        # ── T3 MUST overlap T2 (the actual interruption) ──────────────────────
        overlap = t2_end - t3_start
        if cfg.check_t3_overlap:
            if overlap < cfg.min_t3_overlap:
                stats['t3_overlap_small'] += 1
                continue

        # ── T4/T3 overlap check ───────────────────────────────────────────────
        # Reject only when T4 starts more than max_t4_t3_overlap before T3 ends.
        # Default 1.5s; GPT turn4_meaningful and T3 checks filter false positives.
        if cfg.check_t4_no_overlap_t3:
            if float(t4_row['start']) < t3_end - cfg.max_t4_t3_overlap:
                stats['t4_overlaps_t3'] += 1
                continue

        # ── T1: get words ─────────────────────────────────────────────────────
        t1_words = get_turn_words(t1_row, speaker_words.get(spk_a, []))
        if not t1_words:
            stats['t1_few_words'] += 1
            continue

        # ── T1: sentence-final punctuation ────────────────────────────────────
        if cfg.check_t1_sent_final:
            if not t1_words[-1]['sent_end']:
                stats['t1_no_sent_final'] += 1
                continue

        # ── T1: minimum content words ─────────────────────────────────────────
        if cfg.check_t1_min_words:
            if len(t1_words) < MIN_SUBSTANTIVE_WORDS:
                stats['t1_few_words'] += 1
                continue

        # ── T1: no internal pauses ────────────────────────────────────────────
        if cfg.check_t1_pauses:
            if any(
                t1_words[k + 1]['start'] - t1_words[k]['end'] >= PAUSE_MIN
                for k in range(len(t1_words) - 1)
            ):
                stats['t1_has_pauses'] += 1
                continue

        # ── T1: backchannel from B during T1 ─────────────────────────────────
        # Use the backbiter-recorded backchannel window (backchannel_start/stop).
        # Always detect — stored on candidate for audio masking even when skipped.
        bc_b_in_t1 = backchannel_from_row(t1_row, expected_speaker=spk_b)
        if cfg.check_t1_backchannel and bc_b_in_t1:
            stats['t1_has_backchannel'] += 1
            continue

        # ── T2: not a question ────────────────────────────────────────────────
        if cfg.check_t2_not_question:
            if bool(t2_row.get('end_question', False)):
                stats['t2_is_question'] += 1
                continue

        # ── T2: minimum total words ───────────────────────────────────────────
        if cfg.check_t2_min_words:
            if int(t2_row.get('n_words', 0)) < T2_MIN_WORDS:
                stats['t2_few_words'] += 1
                continue

        # ── T2: get words, check pre-overlap word count ───────────────────────
        t2_words = get_turn_words(t2_row, speaker_words.get(spk_b, []))
        if not t2_words:
            stats['t2_few_words'] += 1
            continue
        t2_pre_words  = [w for w in t2_words if w['end'] <= t3_start]
        t2_post_words = [w for w in t2_words if w['start'] >= t3_start]
        if cfg.check_t2_pre_words:
            if len(t2_pre_words) < T2_MIN_PRE_WORDS:
                stats['t2_pre_words_short'] += 1
                continue

        # ── T2: backchannel from A before T3 onset ───────────────────────────
        # Use the backbiter-recorded backchannel window for T2; exclude any
        # backchannel that starts at or after t3_start (that would be T3 itself).
        bc_a_in_t2 = backchannel_from_row(t2_row, expected_speaker=spk_a,
                                           before_time=t3_start)
        if cfg.check_t2_backchannel and bc_a_in_t2:
            stats['t2_has_backchannel'] += 1
            continue

        # ── T3: backchannel from B during T3 ─────────────────────────────────
        bc_b_in_t3 = backchannel_from_row(t3_row, expected_speaker=spk_b)
        if cfg.check_t3_backchannel and bc_b_in_t3:
            stats['t3_has_backchannel'] += 1
            continue

        # ── T3: no internal pauses ────────────────────────────────────────────
        t3_words = get_turn_words(t3_row, speaker_words.get(spk_a, []))
        if not t3_words:
            stats['t3_empty'] += 1
            continue
        if cfg.check_t3_pauses:
            if any(
                t3_words[k + 1]['start'] - t3_words[k]['end'] >= PAUSE_MIN
                for k in range(len(t3_words) - 1)
            ):
                stats['t3_has_pauses'] += 1
                continue

        # ── T3: duration limit (not a full floor take-over) ───────────────────
        if cfg.check_t3_max_duration:
            if t3_end - t3_start > MAX_T3_DURATION:
                stats['t3_too_long'] += 1
                continue

        # ── T3: has content ───────────────────────────────────────────────────
        t3_txt = words_to_text(t3_words)
        if not t3_txt.strip():
            stats['t3_empty'] += 1
            continue

        # ── T4: get words, minimum content words ─────────────────────────────
        t4_words = get_turn_words(t4_row, speaker_words.get(spk_b, []))
        if not t4_words:
            stats['t4_few_words'] += 1
            continue
        if cfg.check_t4_min_words:
            if len(t4_words) < MIN_SUBSTANTIVE_WORDS:
                stats['t4_few_words'] += 1
                continue

        # ── De-duplicate ──────────────────────────────────────────────────────
        key = (conv_id, spk_a, round(float(t1_row['start']), 2))
        if key in seen:
            stats['duplicate'] += 1
            continue
        seen.add(key)

        stats['passed'] += 1
        candidates.append({
            'type':    'interruption',
            'source':  'backbiter',
            'conv_id': conv_id,
            'spk_a':   spk_a,
            'spk_b':   spk_b,

            # Turn IDs
            't1_turn_id': int(t1_row['turn_id']),
            't2_turn_id': int(t2_row['turn_id']),
            't3_turn_id': int(t3_row['turn_id']),
            't4_turn_id': int(t4_row['turn_id']),

            # Timing
            't1_start': round(t1_words[0]['start'], 3),
            't1_end':   round(t1_words[-1]['end'], 3),
            't2_start': round(float(t2_row['start']), 3),
            't2_end':   round(t2_end, 3),
            't3_start': round(t3_start, 3),
            't3_end':   round(t3_end, 3),
            't4_start': round(float(t4_row['start']), 3),
            't4_end':   round(float(t4_row['stop']), 3),
            'overlap_duration':        round(overlap, 3),
            'yield_latency':           round(t2_end - t3_start, 3),
            'interruption_onset_time': None,
            'interruption_onset_word': None,
            'turn2_yield_word':        None,
            'turn2_yield_time':        None,

            # Transcripts
            't1_text':         words_to_text(t1_words),
            't2_text':         words_to_text(t2_words),
            't2_pre_overlap':  words_to_text(t2_pre_words),
            't2_post_overlap': words_to_text(t2_post_words),
            't3_text':         words_to_text(t3_words),
            't4_text':         words_to_text(t4_words),

            # Word-level timestamps for GPT
            't1_timed':      words_to_timed(t1_words),
            't2_pre_timed':  words_to_timed(t2_pre_words),
            't2_post_timed': words_to_timed(t2_post_words),
            't3_timed':      words_to_timed(t3_words),
            't4_timed':      words_to_timed(t4_words),

            # Detected backchannel words (non-empty only when check was skipped).
            # Used for audio masking during export.
            'bc_b_in_t1': bc_b_in_t1,
            'bc_a_in_t2': bc_a_in_t2,
            'bc_b_in_t3': bc_b_in_t3,

            # GPT fields (filled in Step 2)
            'gpt_valid':               None,
            'turn1_complete':          None,
            'turn1_self_contained':    None,
            'turn2_responds':          None,
            'turn2_interrupted':       None,
            't3_floor_intent':         None,
            't3_pragmatic_complete':   None,
            't3_mid_flow_onset':       None,
            'turn4_meaningful':        None,
            'interruption_style':      None,   # set in code: 'hard' / 'none'
            'interruption_type':       None,
            'gpt_reason':              None,
        })

    return candidates, stats


# ── Step 2: GPT Classification ─────────────────────────────────────────────────

_CANDOR_CTX = (
    "The conversation is from the CANDOR corpus — a dyadic spoken dialogue "
    "between two strangers recorded over a video-call platform."
)

# Text blocks for each optional question (keyed by criterion name)
_Q_T1_COMPLETE = """\
{n}. TURN1_COMPLETE: Is Turn 1 a complete, well-formed utterance (not a fragment)?
"""

_Q_T1_SELF_CONTAINED = """\
{n}. TURN1_SELF_CONTAINED: Can Turn 1 be fully understood and responded to WITHOUT \
any prior conversational context? Be strict:
   - Reject if it contains numbers, quantities, or references whose meaning is \
unclear without context (e.g. "four or five", "that one", "it").
   - Reject if it begins with a filler/acknowledgement then transitions into a \
topic that requires prior context to interpret.
   - Reject if it is clearly a direct answer to an unseen question.
   - Reject if it is a reaction or emotional response that only makes sense as a \
reply to something just said — e.g. "Are you serious?", "Are you kidding?", \
"Really?", "No way!", "That's crazy!", "Seriously?" — these have no interpretable \
meaning without the prior utterance.
   - Accept only if the content introduces a complete, interpretable thought on its own.
"""

_Q_T2_RESPONDS = """\
{n}. TURN2_RESPONDS: Is Turn 2 a genuine, direct DECLARATIVE response to Turn 1?

Turn 2 must directly address the content of Turn 1 and primarily take the form of \
a declarative answer, explanation, statement, or narrative continuation.

Answer false if Turn 2 is primarily:
   - a question directed to Speaker A, including a follow-up question;
   - a topic shift or unrelated remark;
   - a backchannel, acknowledgement, or short reaction;
   - an imperative, request, or instruction rather than a declarative response.

A brief tag question at the end does not automatically make this false, provided \
the main communicative action of Turn 2 is still a declarative response.
"""

_Q_T2_INTERRUPTED = """\
{n}. TURN2_INTERRUPTED: Was Turn 2 cut off before reaching a natural ending?

Turn 3 always overlaps the end of Turn 2 (only overlapping cases are sent to you), \
so you do NOT need Speaker B to keep talking after Turn 3 begins — B may stop \
immediately. Judge ONLY whether Turn 2 stopped at a non-natural point.

Answer true if Turn 2's final words (the point where B stopped) leave the thought \
clearly unfinished — e.g. it trails off mid-clause, ends on a dangling conjunction \
or preposition, ends with an incomplete predicate, or stops in the middle of an \
idea B was still developing, so that B evidently still had more to say.

Answer false if Turn 2 had already reached a natural completion point — a complete \
clause or sentence (subject + verb + complement) that stands on its own, or B was \
plainly wrapping up — so Speaker A simply took the next turn with slight overlap \
(normal turn-taking).
"""

_Q_T3_FLOOR_INTENT = """\
{n}. T3_FLOOR_INTENT: Does Turn 3 carry genuine floor-taking intent at its onset?

First identify the FLOOR-TAKING ONSET: the first substantive word or phrase in \
Turn 3 that attempts to take the floor. Ignore leading fillers, backchannels, and \
acknowledgements such as "yeah," "uh-huh," "right," or "okay."

Answer true if at the floor-taking onset Turn 3 contributes meaningful content \
that requires or invites a response from Speaker B — e.g. asks a question, \
requests clarification, makes a correction, challenges B's claim, introduces a \
counterpoint, or adds substantive information.

Answer false if Turn 3 consists only of:
   - Pure backchannels: "Uh huh", "Yeah", "Mm hmm", "Right", "Okay"
   - Short approval/reaction phrases: "That's cool", "Nice", "Interesting", \
"Oh wow", "Good for you", "That makes sense", "Okay cool"
   - Wrap-up signals: "Okay that's cool", "Yeah that's great", "Sounds good"
   - Empathetic acknowledgements: "I know", "I know right", "I can imagine", \
"That must be hard", "I totally get that", "I understand"

Short questions such as "Why?", "Which one?", "What do you mean?", and "How so?" \
DO qualify.
"""

_Q_T3_PRAGMATIC_COMPLETE = """\
{n}. T3_PRAGMATIC_COMPLETE: Does Turn 3 form a complete conversational action?

Answer true if Turn 3, even if not a full grammatical sentence, is interpretable \
as a self-standing communicative act — e.g. a short question, correction, or \
challenge that can stand alone.
Answer false if Turn 3 is an unintelligible fragment or trailing phrase that \
cannot function as an independent conversational move.
"""

_Q_T3_MID_FLOW_ONSET = """\
{n}. T3_MID_FLOW_ONSET: Does Turn 3's floor-taking onset occur while Turn 2 is \
still substantially in progress?

Answer true if Turn 2 had not yet reached a natural ending when Turn 3 began — \
i.e. B was clearly still mid-thought at the interruption onset (consistent with \
TURN2_INTERRUPTED).
Answer false if Turn 2 had already completed or was about to complete a full, \
self-standing thought before Turn 3's onset began — making this normal \
overlapping turn-taking rather than a disruptive interruption.
"""

_Q_T4_MEANINGFUL = """\
{n}. TURN4_MEANINGFUL: Does Turn 4 provide a substantive response to the content or \
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
"""

_Q_RECOVERY_TYPE = """\
{n}. RECOVERY_TYPE: Based on the content of Turn 3 and B's response in Turn 4, \
classify the recovery as one of:
   - "correct"     – A corrects an error; B acknowledges or revises.
   - "enrich"      – A asks for more detail; B elaborates on the same point.
   - "shorten"     – A signals B should be more concise; B wraps up.
   - "repeat"      – A asks B to repeat something; B repeats.
   - "shift_topic" – A redirects to a different subject; B follows the new topic.
   - "ask_followup"– After being interrupted, B asks A to clarify A's interruption.
   - "none"        – Turn 4 is not substantive, or no recovery type can be assigned.
"""


def build_gpt_prompt(skipped_gpt: frozenset[str] = frozenset()) -> str:
    """
    Build the GPT evaluation prompt, omitting questions for skipped criteria so
    they don't influence GPT's reasoning on other questions.

    skipped_gpt: frozenset of criterion names:
        't1_complete', 't1_self_contained', 't2_responds',
        't3_floor_intent', 't3_pragmatic_complete', 't3_mid_flow_onset',
        't4_meaningful'
    """
    # Decide which optional questions to include
    include_t1_complete            = 't1_complete'            not in skipped_gpt
    include_t1_self_contained      = 't1_self_contained'      not in skipped_gpt
    include_t2_responds            = 't2_responds'            not in skipped_gpt
    include_t3_floor_intent        = 't3_floor_intent'        not in skipped_gpt
    include_t3_pragmatic_complete  = 't3_pragmatic_complete'  not in skipped_gpt
    include_t3_mid_flow_onset      = 't3_mid_flow_onset'      not in skipped_gpt
    include_t4_meaningful          = 't4_meaningful'          not in skipped_gpt

    # Build ordered list of (question_block, json_field) pairs
    # Always-on questions: TURN2_INTERRUPTED, INTERRUPTION_STYLE, RECOVERY_TYPE
    questions: list[str] = []
    json_fields: list[str] = []

    # ── Turn 1 ────────────────────────────────────────────────────────────────
    t1_section_needed = include_t1_complete or include_t1_self_contained
    if t1_section_needed:
        questions.append("\n── TURN 1 (Speaker A) ──────────────────────────────────────────────────────────\n")
    if include_t1_complete:
        questions.append(_Q_T1_COMPLETE)
        json_fields.append('  "turn1_complete": true/false,')
    if include_t1_self_contained:
        questions.append(_Q_T1_SELF_CONTAINED)
        json_fields.append('  "turn1_self_contained": true/false,')

    # ── Turn 2 ────────────────────────────────────────────────────────────────
    questions.append("\n── TURN 2 (Speaker B) ──────────────────────────────────────────────────────────\n")
    if include_t2_responds:
        questions.append(_Q_T2_RESPONDS)
        json_fields.append('  "turn2_responds": true/false,')
    questions.append(_Q_T2_INTERRUPTED)
    json_fields.append('  "turn2_interrupted": true/false,')

    # ── Turn 3 ────────────────────────────────────────────────────────────────
    questions.append("\n── TURN 3 (Speaker A — interruption) ───────────────────────────────────────────\n")
    if include_t3_floor_intent:
        questions.append(_Q_T3_FLOOR_INTENT)
        json_fields.append('  "t3_floor_intent": true/false,')
    if include_t3_pragmatic_complete:
        questions.append(_Q_T3_PRAGMATIC_COMPLETE)
        json_fields.append('  "t3_pragmatic_complete": true/false,')
    if include_t3_mid_flow_onset:
        questions.append(_Q_T3_MID_FLOW_ONSET)
        json_fields.append('  "t3_mid_flow_onset": true/false,')

    # ── Turn 4 ────────────────────────────────────────────────────────────────
    questions.append("\n── TURN 4 (Speaker B — recovery) ───────────────────────────────────────────────\n")
    if include_t4_meaningful:
        questions.append(_Q_T4_MEANINGFUL)
        json_fields.append('  "turn4_meaningful": true/false,')
    questions.append(_Q_RECOVERY_TYPE)
    json_fields.append('  "recovery_type": "none"|"correct"|"enrich"|"shorten"|"repeat"|"shift_topic"|"ask_followup",')

    # ── Renumber questions sequentially ───────────────────────────────────────
    n = 1
    numbered: list[str] = []
    for block in questions:
        if block.startswith('\n──'):  # section header, no number
            numbered.append(block)
        else:
            numbered.append(block.format(n=n))
            n += 1

    questions_block = "".join(numbered)

    # is_valid is NOT included — validity is computed in code from individual fields.
    json_fields.append('  "reason": "one sentence explaining the key decision"')
    json_block = "{{\n" + "\n".join(json_fields) + "\n}}"

    prompt = (
        "You are a dialogue analysis expert evaluating a 4-turn conversational pattern "
        "for research on interruption and recovery in spoken dialogue.\n\n"
        + _CANDOR_CTX + "\n\n"
        "Here are four consecutive turns from a conversation, with segment-level "
        "timestamps to help you assess timing precisely:\n\n"
        'Turn 1 (Speaker A):\n"{t1}"\n  Segment timestamps: {t1_timed}\n\n'
        'Turn 2 (Speaker B — response):\n"{t2}"\n'
        "  Segment timestamps: {t2_pre_timed} ... {t2_post_timed}\n"
        "  [Turn 2 ran {t2_start:.2f}s–{t2_end:.2f}s; Turn 3 started at {t3_start:.2f}s, "
        "i.e. {t2_pre_words} content words into Turn 2, with {overlap:.2f}s of overlap remaining]\n\n"
        'Turn 3 (Speaker A — started at {t3_start:.2f}s, {overlap:.2f}s before Turn 2 ended):\n"{t3}"\n'
        "  Segment timestamps: {t3_timed}\n\n"
        'Turn 4 (Speaker B — recovery/response):\n"{t4}"\n'
        "  Segment timestamps: {t4_timed}\n\n"
        "---\n"
        + questions_block
        + "\nRespond ONLY with JSON (no markdown):\n"
        + json_block
    )
    return prompt


class QuotaExceededError(RuntimeError):
    pass


def gpt_classify(candidates: list[dict], client: OpenAI,
                 cfg: FilterConfig | None = None,
                 retry_nulls_only: bool = False,
                 max_valid_per_type: int | None = None,
                 skipped_gpt: frozenset[str] = frozenset()) -> tuple[list[dict], dict]:
    from collections import Counter

    # Drop candidates where B had already stopped before A started (no real overlap)
    for cand in candidates:
        if cand.get('gpt_valid') is None and cand.get('yield_latency', 1) <= 0:
            cand['gpt_valid'] = False
            cand['gpt_reason'] = "pre-filter: yield_latency <= 0 (B ended before A started)"

    if cfg is None:
        cfg = FilterConfig()

    pending = [c for c in candidates
               if not retry_nulls_only or c.get('turn2_interrupted') is None]
    total = len(pending)
    if skipped_gpt:
        print(f"  Skipped GPT criteria (removed from prompt): {', '.join(sorted(skipped_gpt))}")
    print(f"  Sending {total} candidates to GPT ({GPT_MODEL})"
          + (f" (cap={max_valid_per_type}/type)" if max_valid_per_type else "") + "...")
    valid_by_type: dict[str, int] = {}

    # Build the prompt template once for this run (same skipped set for all candidates)
    prompt_template = build_gpt_prompt(skipped_gpt)

    for i, cand in enumerate(pending):
        # Stop early when every known recovery type has hit the cap
        if (max_valid_per_type is not None
                and len(valid_by_type) >= len(RECOVERY_TYPES)
                and all(valid_by_type.get(t, 0) >= max_valid_per_type
                        for t in RECOVERY_TYPES)):
            print(f"    [{i+1}/{total}] all type caps reached — stopping.")
            break

        msg = prompt_template.format(
            t1=cand['t1_text'],
            t1_timed=cand.get('t1_timed', ''),
            t2=cand['t2_text'],
            t2_pre=cand['t2_pre_overlap'] or '(nothing before overlap)',
            t2_pre_timed=cand.get('t2_pre_timed', '(none)'),
            t2_post=cand['t2_post_overlap'] or '(none)',
            t2_post_timed=cand.get('t2_post_timed', '(none)'),
            t2_start=cand.get('t2_start', 0.0),
            t2_end=cand.get('t2_end', 0.0),
            t2_pre_words=len((cand.get('t2_pre_overlap') or '').split()),
            t3=cand['t3_text'],
            t3_timed=cand.get('t3_timed', ''),
            t3_start=cand['t3_start'],
            t4=cand['t4_text'],
            t4_timed=cand.get('t4_timed', ''),
            overlap=cand['overlap_duration'],
        )
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": msg}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            # Store individual field answers from GPT
            cand['turn1_complete']          = result.get('turn1_complete')
            cand['turn1_self_contained']    = result.get('turn1_self_contained')
            cand['turn2_responds']          = result.get('turn2_responds')
            cand['turn2_interrupted']       = result.get('turn2_interrupted')
            cand['t3_floor_intent']         = result.get('t3_floor_intent')
            cand['t3_pragmatic_complete']   = result.get('t3_pragmatic_complete')
            cand['t3_mid_flow_onset']       = result.get('t3_mid_flow_onset')
            cand['turn4_meaningful']        = result.get('turn4_meaningful')
            cand['interruption_type']       = result.get('recovery_type')
            cand['gpt_reason']              = result.get('reason', '')
            # interruption_style is derived in code: "hard" when no active T3
            # check returned False (None = skipped, treated as pass).
            cand['interruption_style'] = (
                'hard'
                if (cand.get('t3_floor_intent')       is not False
                    and cand.get('t3_pragmatic_complete') is not False
                    and cand.get('t3_mid_flow_onset')     is not False)
                else 'none'
            )
            # onset/yield times come directly from timing data, not GPT
            cand['interruption_onset_time'] = cand['t3_start']
            cand['interruption_onset_word'] = cand['t3_timed'].split('(')[0].strip() if cand.get('t3_timed') else None
            cand['turn2_yield_time']        = cand['t2_end']
            cand['turn2_yield_word']        = cand['t2_pre_timed'].rsplit('(', 1)[0].rsplit(' ', 1)[-1].strip() if cand.get('t2_pre_timed') else None

            # Validity is computed from individual fields — not from GPT's is_valid
            cand['gpt_valid'] = recompute_validity(cand, cfg)

            rtype = cand.get('interruption_type') or 'none'
            valid = cand['gpt_valid']
            t3_flags = (
                f"fi={'T' if cand.get('t3_floor_intent') else 'F'}"
                f" pc={'T' if cand.get('t3_pragmatic_complete') else 'F'}"
                f" mf={'T' if cand.get('t3_mid_flow_onset') else 'F'}"
            )
            print(f"    [{i+1}/{total}] {'✓' if valid else '✗'} "
                  f"{t3_flags} type={rtype} "
                  f"— {cand['t1_text'][:60]}")

            if valid:
                valid_by_type[rtype] = valid_by_type.get(rtype, 0) + 1

        except Exception as e:
            err_str = str(e)
            if 'insufficient_quota' in err_str or '429' in err_str or 'rate_limit' in err_str.lower():
                cand['gpt_valid']  = None
                cand['gpt_reason'] = f"API error: {e}"
                raise QuotaExceededError(
                    f"OpenAI quota exceeded at candidate {i} conv={cand.get('conv_id')}"
                ) from e
            print(f"    [WARN] GPT failed for candidate {i}: {e}")
            cand['gpt_valid']  = None
            cand['gpt_reason'] = f"API error: {e}"

    # Judged = candidates where GPT has set turn2_interrupted (always-evaluated field)
    judged = [c for c in candidates if c.get('turn2_interrupted') is not None]
    gpt_stats = {
        'total_sent':                   total,
        'total_judged':                 len(judged),
        'turn1_complete_false':         sum(1 for c in judged if c.get('turn1_complete')          is False),
        'turn1_self_contained_false':   sum(1 for c in judged if c.get('turn1_self_contained')    is False),
        'turn2_responds_false':         sum(1 for c in judged if c.get('turn2_responds')          is False),
        'turn2_interrupted_false':      sum(1 for c in judged if c.get('turn2_interrupted')       is False),
        't3_floor_intent_false':        sum(1 for c in judged if c.get('t3_floor_intent')         is False),
        't3_pragmatic_complete_false':  sum(1 for c in judged if c.get('t3_pragmatic_complete')   is False),
        't3_mid_flow_onset_false':      sum(1 for c in judged if c.get('t3_mid_flow_onset')       is False),
        'turn4_meaningful_false':       sum(1 for c in judged if c.get('turn4_meaningful')        is False),
        'recovery_type':                dict(Counter(c.get('interruption_type', '?') for c in judged)),
        'valid_by_type':                valid_by_type,
    }

    for rtype, n in sorted(valid_by_type.items()):
        print(f"  VALID_AFTER_LLM {rtype}: {n}")
    print(f"GPT_STATS:{json.dumps(gpt_stats)}")
    return candidates, gpt_stats


# ── Step 3: Audio Export ───────────────────────────────────────────────────────

def silence_other_speaker(clip: np.ndarray, sr: int,
                          clip_start_s: float,
                          other_words: list[dict]) -> np.ndarray:
    """Zero out frames in `clip` where the other speaker's words fall.

    `clip` is a 1-D mono array starting at `clip_start_s`.
    `other_words` are word dicts (start/end in absolute conversation seconds)
    from the other speaker — derived from transcribe_output.json.
    Any frame whose absolute time overlaps a word window is set to zero,
    removing microphone bleed from the other speaker.
    """
    clip = clip.copy()
    n = len(clip)
    for w in other_words:
        w_start = w['start'] - clip_start_s
        w_end   = w['end']   - clip_start_s
        s = max(0, int(w_start * sr))
        e = min(n, int(w_end   * sr))
        if s < e:
            clip[s:e] = 0.0
    return clip


def export_clip(audio: np.ndarray, sr: int,
                full_start: float, full_end: float,
                turn_start: float, turn_end: float,
                out_path: Path, ch_idx: int,
                other_words: list[dict] | None = None) -> None:
    """Export a single-channel clip spanning [full_start, full_end].

    Only the frames in [turn_start, turn_end] contain actual audio; every
    other frame is zeroed (silence padding).  This keeps all mono clips
    time-aligned with the full stereo clip.

    other_words are silenced within the turn window to remove mic bleed.
    Absolute timestamps in other_words are relative to full_start.
    """
    fs = max(0, int(full_start * sr))
    fe = min(audio.shape[-1], int(full_end * sr))
    full_clip = (audio[ch_idx, fs:fe] if audio.ndim == 2 else audio[fs:fe]).copy()
    n = len(full_clip)

    # Build output: silence everywhere, copy turn audio into the right slice
    result = np.zeros(n, dtype=full_clip.dtype)
    ts = max(0, min(n, int((turn_start - full_start) * sr)))
    te = max(0, min(n, int((turn_end   - full_start) * sr)))
    result[ts:te] = full_clip[ts:te]

    # Remove mic bleed from the other speaker within the active turn window
    if other_words:
        result = silence_other_speaker(result, sr, full_start, other_words)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), result, sr)


def export_stereo_clip(audio: np.ndarray, sr: int,
                       start_s: float, end_s: float,
                       out_path: Path) -> None:
    s = max(0, int(start_s * sr))
    e = min(audio.shape[-1], int(end_s * sr))
    clip = audio[:, s:e].T if audio.ndim == 2 else audio[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr)


def export_stereo_clip_masked(audio: np.ndarray, sr: int,
                               start_s: float, end_s: float,
                               out_path: Path,
                               mask_per_ch: list[tuple[int, list[dict]]] | None = None
                               ) -> None:
    """Export stereo clip, silencing detected backchannel words per channel.

    mask_per_ch: list of (ch_idx, word_list) pairs.  For each pair, every word
    in word_list is zeroed out in that channel.  Words use absolute conversation
    timestamps; start_s is subtracted to convert to clip-relative offsets.
    All other frames are kept untouched.
    """
    s = max(0, int(start_s * sr))
    e = min(audio.shape[-1], int(end_s * sr))
    if audio.ndim == 2:
        ch0 = audio[0, s:e].copy()
        ch1 = audio[1, s:e].copy()
        if mask_per_ch:
            for ch_idx, words in mask_per_ch:
                target = ch0 if ch_idx == 0 else ch1
                n = len(target)
                for w in words:
                    w_s = max(0, int((w['start'] - start_s) * sr))
                    w_e = min(n,  int((w['end']   - start_s) * sr))
                    if w_s < w_e:
                        target[w_s:w_e] = 0.0
        clip = np.stack([ch0, ch1], axis=1)   # (samples, 2) for soundfile
    else:
        clip = audio[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr)


def export_interruption_candidates(candidates: list[dict],
                                   audio: np.ndarray, sr: int,
                                   audio_dir: Path,
                                   spk_to_ch: dict[str, int],
                                   speaker_words: dict[str, list],
                                   results_dir: Path | None = None) -> None:
    """Export audio clips with other-speaker backchannel frames silenced.

    For each mono clip (T1–T4), words from the other speaker that fall inside
    the clip window are zeroed out — removing microphone bleed from listener
    backchannels. The stereo full-sequence clip is exported as-is.
    """
    import shutil
    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue
        style = cand.get('interruption_style', 'unknown')
        itype = cand.get('interruption_type', 'unknown')
        out_dir = audio_dir / style / itype
        out_dir.mkdir(parents=True, exist_ok=True)
        conv_id = cand['conv_id']
        base    = f"{conv_id}_t{cand['t1_turn_id']}"
        ch_a = spk_to_ch.get(cand['spk_a'], 0)
        ch_b = spk_to_ch.get(cand['spk_b'], 1)

        # Words from the other speaker, used to silence mic-bleed in mono clips
        words_a = speaker_words.get(cand['spk_a'], [])
        words_b = speaker_words.get(cand['spk_b'], [])

        def b_words_in(t0, t1):
            return [w for w in words_b if w['start'] < t1 and w['end'] > t0]

        def a_words_in(t0, t1):
            return [w for w in words_a if w['start'] < t1 and w['end'] > t0]

        # ── Mono clips — all span [t1_start, t4_end] with silence padding ────
        # Each file is the same length as the stereo clip; only the turn's own
        # frames carry audio, everything else is zeroed.
        # T3 bleed-silencing is skipped: B's last word overlaps T3's onset,
        # silencing it would cut A's first word.
        full_start = cand['t1_start']
        full_end   = cand['t4_end']
        mono_clips = [
            (cand['t1_start'], cand['t1_end'], f"{base}_t1.wav", ch_a, b_words_in),
            (cand['t2_start'], cand['t2_end'], f"{base}_t2.wav", ch_b, a_words_in),
            (cand['t3_start'], cand['t3_end'], f"{base}_t3.wav", ch_a, None),
            (cand['t4_start'], cand['t4_end'], f"{base}_t4.wav", ch_b, a_words_in),
        ]
        for turn_start, turn_end, fname, ch, other_fn in mono_clips:
            dest = out_dir / fname
            other = other_fn(turn_start, turn_end) if other_fn else None
            export_clip(audio, sr, full_start, full_end,
                        turn_start, turn_end, dest, ch, other_words=other)

        # ── Stereo full-sequence clip — mask backchannel words per channel ────
        # When a backchannel check was skipped, the candidate carries the detected
        # words. We zero those words in each speaker's own channel so the stereo
        # output is clean without discarding the candidate.
        bc_b_t1 = cand.get('bc_b_in_t1', [])
        bc_a_t2 = cand.get('bc_a_in_t2', [])
        bc_b_t3 = cand.get('bc_b_in_t3', [])
        mask_per_ch: list[tuple[int, list[dict]]] = []
        if bc_b_t1:
            mask_per_ch.append((ch_b, bc_b_t1))   # silence B's channel during T1 BC
        if bc_a_t2:
            mask_per_ch.append((ch_a, bc_a_t2))   # silence A's channel during T2 BC
        if bc_b_t3:
            mask_per_ch.append((ch_b, bc_b_t3))   # silence B's channel during T3 BC
        stereo_dest = out_dir / f"{base}_full_stereo.wav"
        export_stereo_clip_masked(
            audio, sr,
            cand['t1_start'], cand['t4_end'],
            stereo_dest,
            mask_per_ch=mask_per_ch or None,
        )

        # Mirror all exported files to the global results dir
        if results_dir is not None:
            global_dir = results_dir / style / itype
            global_dir.mkdir(parents=True, exist_ok=True)
            for src in out_dir.glob(f"{base}_*.wav"):
                shutil.copy2(src, global_dir / src.name)

        cand['audio_dir'] = str(out_dir)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_results(data: list[dict], path: Path, label: str) -> None:
    path = path.with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [{label}] Saved {len(data)} entries → {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(conv_dir: Path, gpt_only: bool = False,
         no_gpt: bool = False,
         results_dir: Path | None = None,
         max_valid: int | None = None,
         cfg: FilterConfig | None = None,
         conv_subdir: str = "interruption_candidates_relaxed") -> None:
    if cfg is None:
        cfg = FilterConfig()
    conv_id    = conv_dir.name
    output_dir = conv_dir / conv_subdir
    f1_dir     = output_dir / "filter1_timing"
    f2_dir     = output_dir / "filter2_gpt"
    audio_dir  = output_dir / "audio"
    audio_path = conv_dir / "processed" / f"{conv_id}.mp3"
    chmap_path = conv_dir / "processed" / "channel_map.json"
    back_path  = conv_dir / "transcription" / "transcript_backbiter.csv"
    json_path  = conv_dir / "transcription" / "transcribe_output.json"

    # ── Load shared data ───────────────────────────────────────────────────────
    trans_json = json.load(open(json_path))
    chmap      = json.load(open(chmap_path))

    # channel_map.json maps L/R -> participant ID; L=ch_0, R=ch_1
    ch_to_speaker: dict[str, str] = {
        "ch_0": chmap["L"],
        "ch_1": chmap["R"],
    }
    spk_to_ch = {spk: int(key.split('_')[1]) for key, spk in ch_to_speaker.items()}
    print(f"  Channel mapping: {ch_to_speaker}")

    # Build speaker_words here so it is available in both gpt_only and full modes
    speaker_words = build_speaker_words(trans_json, ch_to_speaker)

    if gpt_only:
        print(f"[gpt_only] Loading existing timing results for {conv_id}...")
        candidates = json.load(open(f1_dir / "all.json"))
        if (f2_dir / "all_with_gpt_decisions.json").exists():
            prev = {(c['conv_id'], c['t1_turn_id']): c
                    for c in json.load(open(f2_dir / "all_with_gpt_decisions.json"))}
            for c in candidates:
                key = (c['conv_id'], c['t1_turn_id'])
                if key in prev and prev[key].get('gpt_valid') is not None:
                    for field in ('gpt_valid', 'turn1_complete', 'turn1_self_contained',
                                  'turn2_responds', 'turn2_interrupted',
                                  'interruption_style', 'interruption_type', 'gpt_reason'):
                        c[field] = prev[key].get(field)
    else:
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"Deleted previous results in {output_dir.name}/")
        output_dir.mkdir(exist_ok=True)

        print("Loading backbiter transcript...")
        back = pd.read_csv(back_path)
        back.attrs['conv_id'] = conv_id

        # Pre-filter: drop any turn with fewer than cfg.min_words_filter words.
        before = len(back)
        if cfg.min_words_filter > 0:
            back = back[back['n_words'] >= cfg.min_words_filter].reset_index(drop=True)
        print(f"  Pre-filter (min_words={cfg.min_words_filter}): {before} → {len(back)} turns "
              f"(removed {before - len(back)} short turns)")

        print("Building per-speaker word list...")
        for spk, wds in speaker_words.items():
            print(f"  {spk}: {len(wds)} words")

        print("\n── Step 1: Timing filter ──")
        candidates, timing_stats = find_interruption_patterns(back, speaker_words, cfg)
        print(f"  Candidates found: {len(candidates)}")
        print(f"SEND_TO_LLM interruption: {len(candidates)}", flush=True)
        print(f"TIMING_STATS:{__import__('json').dumps(timing_stats)}", flush=True)

        print("\nSaving Filter 1 results...")
        save_results(candidates, f1_dir / "all.json", "timing")

    if no_gpt:
        print("\n── Skipping GPT step (--no_gpt) ──")
        print(f"\n── Summary ──")
        print(f"  Output: {output_dir.name}/")
        print(f"  filter1_timing/  ← timing filter results ({len(candidates)} candidates)")
        print("\nDone.")
        return

    # ── Step 2: GPT classification ─────────────────────────────────────────────
    print(f"\n── Step 2: GPT classification ({GPT_MODEL}) ──")
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Build the set of GPT criteria to skip (completely removed from the prompt)
    skipped_gpt: frozenset[str] = frozenset(k for k, v in {
        't1_complete':            not cfg.gpt_require_t1_complete,
        't1_self_contained':      not cfg.gpt_require_t1_self_contained,
        't2_responds':            not cfg.gpt_require_t2_responds,
        't3_floor_intent':        not cfg.gpt_require_t3_floor_intent,
        't3_pragmatic_complete':  not cfg.gpt_require_t3_pragmatic_complete,
        't3_mid_flow_onset':      not cfg.gpt_require_t3_mid_flow_onset,
        't4_meaningful':          not cfg.gpt_require_t4_meaningful,
    }.items() if v)
    try:
        candidates, gpt_stats = gpt_classify(candidates, client,
                                             cfg=cfg,
                                             retry_nulls_only=gpt_only,
                                             max_valid_per_type=max_valid,
                                             skipped_gpt=skipped_gpt)
    except QuotaExceededError as e:
        print(f"  QUOTA EXCEEDED: {e}")
        sys.exit(QUOTA_EXIT_CODE)

    # gpt_valid is already set inside gpt_classify() via recompute_validity()
    valid = [c for c in candidates if c.get('gpt_valid')]
    hard_v = [c for c in valid if c.get('interruption_style') == 'hard']
    print(f"\n  Valid after GPT: {len(valid)}/{len(candidates)}")
    if hard_v:
        by_type: dict = {}
        for c in hard_v:
            t = c.get('interruption_type', 'unknown')
            by_type[t] = by_type.get(t, 0) + 1
        print(f"    hard: " + "  ".join(f"{t}={n}" for t, n in by_type.items()))

    print("\nSaving Filter 2 results (GPT)...")
    save_results(candidates, f2_dir / "all_with_gpt_decisions.json", "all+decisions")
    if hard_v:
        save_results(hard_v, f2_dir / "hard.json", "hard")

    # ── Step 3: Audio export ───────────────────────────────────────────────────
    if valid:
        print(f"\n── Step 3: Audio export ──")
        print(f"Loading audio from {audio_path.name}...")
        audio, sr = librosa.load(str(audio_path), sr=AUDIO_SR, mono=False)
        print(f"  Audio: {audio.shape}, sr={sr}")
        export_interruption_candidates(candidates, audio, sr, audio_dir, spk_to_ch,
                                       speaker_words=speaker_words,
                                       results_dir=results_dir)
        print(f"  Exported {len(valid)} × 5 clips → {audio_dir.name}/")
        if results_dir:
            print(f"  Mirrored to global results → {results_dir}")
    else:
        print("\n  No valid candidates — skipping audio export.")

    # Save valid.json AFTER audio export so audio_dir is populated in each entry
    save_results(valid, f2_dir / "valid.json", "valid")

    print("\n── Summary ──")
    print(f"  Output: {output_dir.name}/")
    print(f"  filter1_timing/  ← timing filter results")
    print(f"  filter2_gpt/     ← GPT classifications")
    print(f"  audio/           ← <style>/<type>/<conv>_t1/t2/t3/t4/full_stereo.wav")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract interruption & recovery patterns.")

    # ── Core options ───────────────────────────────────────────────────────────
    parser.add_argument("--conv_dir",    type=Path, default=DEFAULT_CONV_DIR)
    parser.add_argument("--gpt_only",   action="store_true",
                        help="Skip timing filter; re-run GPT on existing filter1 results")
    parser.add_argument("--no_gpt",     action="store_true",
                        help="Run timing filter only; skip GPT classification and audio export")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Global results directory — audio mirrored here")
    parser.add_argument("--max_valid", type=int, default=None,
                        help="Stop GPT calls after this many valid candidates per recovery type")
    parser.add_argument("--conv_subdir", type=str, default="interruption_candidates_relaxed",
                        help="Per-conversation output subdirectory name (default: interruption_candidates_relaxed)")

    # ── Pre-filter threshold ───────────────────────────────────────────────────
    parser.add_argument("--min_words_filter", type=int, default=MIN_WORDS_FILTER,
                        help=f"Pre-filter: drop turns with fewer than N words (default {MIN_WORDS_FILTER}; 0=off)")

    # ── Timing criteria (each --skip_X disables the check) ────────────────────
    parser.add_argument("--skip_abab",         action="store_true",
                        help="Do NOT enforce A-B-A-B speaker alternation")
    parser.add_argument("--skip_t2_overlap_t1", action="store_true",
                        help="Do NOT reject T2 that starts before T1 ends")
    parser.add_argument("--max_t2_t1_overlap", type=float, default=MAX_OVERLAP_TOLERANCE,
                        help=f"Max T2-before-T1 overlap allowed (default {MAX_OVERLAP_TOLERANCE}s; 0.05 = strict)")
    parser.add_argument("--skip_t3_overlap",   action="store_true",
                        help="Do NOT require T3 to overlap T2 by min_t3_overlap seconds")
    parser.add_argument("--min_t3_overlap",    type=float, default=MIN_INTERRUPTION_OVERLAP,
                        help=f"Minimum T3/T2 overlap to count as an interruption (default {MIN_INTERRUPTION_OVERLAP}s)")
    parser.add_argument("--skip_t4_overlap_t3", action="store_true",
                        help="Do NOT reject T4 that starts before T3 ends")
    parser.add_argument("--max_t4_t3_overlap", type=float, default=MAX_OVERLAP_TOLERANCE,
                        help=f"Max T4-before-T3 overlap allowed (default {MAX_OVERLAP_TOLERANCE}s; 0.05 = strict)")
    parser.add_argument("--skip_t1_sent_final", action="store_true",
                        help="Do NOT require T1 to end with sentence-final punctuation")
    parser.add_argument("--skip_t1_words",     action="store_true",
                        help="Do NOT enforce T1 minimum word count")
    parser.add_argument("--skip_t1_pauses",    action="store_true",
                        help="Do NOT reject T1 with internal pauses")
    parser.add_argument("--skip_t1_backchannel", action="store_true",
                        help="Do NOT reject when B produces words during T1 (backchannel)")
    parser.add_argument("--skip_t2_question",  action="store_true",
                        help="Do NOT reject T2 that ends with a question mark")
    parser.add_argument("--skip_t2_words",     action="store_true",
                        help="Do NOT enforce T2 minimum total word count")
    parser.add_argument("--skip_t2_pre_words", action="store_true",
                        help="Do NOT enforce T2 minimum pre-overlap word count")
    parser.add_argument("--skip_t2_backchannel", action="store_true",
                        help="Do NOT reject when A produces words during T2 before T3 onset")
    parser.add_argument("--skip_t3_backchannel", action="store_true",
                        help="Do NOT reject when B produces words during T3 (backchannel over interruption)")
    parser.add_argument("--skip_t3_pauses",    action="store_true",
                        help="Do NOT reject T3 with internal pauses")
    parser.add_argument("--skip_t3_duration",  action="store_true",
                        help="Do NOT reject T3 that exceeds MAX_T3_DURATION")
    parser.add_argument("--skip_t4_words",     action="store_true",
                        help="Do NOT enforce T4 minimum word count")

    # ── GPT validity criteria ──────────────────────────────────────────────────
    parser.add_argument("--skip_gpt_t1_complete",       action="store_true",
                        help="Do NOT require GPT turn1_complete=true for is_valid")
    parser.add_argument("--skip_gpt_t1_self_contained", action="store_true",
                        help="Do NOT require GPT turn1_self_contained=true for is_valid")
    parser.add_argument("--skip_gpt_t2_responds",           action="store_true",
                        help="Do NOT require GPT turn2_responds=true for is_valid")
    parser.add_argument("--skip_gpt_t3_floor_intent",       action="store_true",
                        help="Do NOT require GPT t3_floor_intent=true for is_valid")
    parser.add_argument("--skip_gpt_t3_pragmatic_complete", action="store_true",
                        help="Do NOT require GPT t3_pragmatic_complete=true for is_valid")
    parser.add_argument("--skip_gpt_t3_mid_flow",           action="store_true",
                        help="Do NOT require GPT t3_mid_flow_onset=true for is_valid")
    parser.add_argument("--skip_gpt_t4_meaningful",         action="store_true",
                        help="Do NOT require GPT turn4_meaningful=true for is_valid")

    args = parser.parse_args()

    cfg = FilterConfig(
        min_words_filter            = args.min_words_filter,
        check_abab                  = not args.skip_abab,
        check_t2_no_overlap_t1      = not args.skip_t2_overlap_t1,
        max_t2_t1_overlap           = args.max_t2_t1_overlap,
        check_t3_overlap            = not args.skip_t3_overlap,
        min_t3_overlap              = args.min_t3_overlap,
        check_t4_no_overlap_t3      = not args.skip_t4_overlap_t3,
        max_t4_t3_overlap           = args.max_t4_t3_overlap,
        check_t1_sent_final         = not args.skip_t1_sent_final,
        check_t1_min_words          = not args.skip_t1_words,
        check_t1_pauses             = not args.skip_t1_pauses,
        check_t1_backchannel        = not args.skip_t1_backchannel,
        check_t2_not_question       = not args.skip_t2_question,
        check_t2_min_words          = not args.skip_t2_words,
        check_t2_pre_words          = not args.skip_t2_pre_words,
        check_t2_backchannel        = not args.skip_t2_backchannel,
        check_t3_backchannel        = not args.skip_t3_backchannel,
        check_t3_pauses             = not args.skip_t3_pauses,
        check_t3_max_duration       = not args.skip_t3_duration,
        check_t4_min_words          = not args.skip_t4_words,
        gpt_require_t1_complete     = not args.skip_gpt_t1_complete,
        gpt_require_t1_self_contained = not args.skip_gpt_t1_self_contained,
        gpt_require_t2_responds          = not args.skip_gpt_t2_responds,
        gpt_require_t3_floor_intent      = not args.skip_gpt_t3_floor_intent,
        gpt_require_t3_pragmatic_complete= not args.skip_gpt_t3_pragmatic_complete,
        gpt_require_t3_mid_flow_onset    = not args.skip_gpt_t3_mid_flow,
        gpt_require_t4_meaningful        = not args.skip_gpt_t4_meaningful,
    )

    main(args.conv_dir, gpt_only=args.gpt_only, no_gpt=args.no_gpt,
         results_dir=args.results_dir, max_valid=args.max_valid, cfg=cfg,
         conv_subdir=args.conv_subdir)
