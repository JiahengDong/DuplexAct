#!/usr/bin/env python3
"""
CANDOR Floor-Holding + Backchannel Extractor — Backbiter Edition

Extracts a 2-turn structure for studying (a) how Speaker A HOLDS THE FLOOR when
Speaker B drops a backchannel mid-turn, and (b) what kind of BACKCHANNEL Speaker B
signals. A single candidate carries BOTH labels:

  Turn 0 : Speaker B finished speaking            (the trigger for A)
  Turn 1 : Speaker A responding                   (A holds the floor)
             ├─ response_before_backchannel  (A words before the backchannel)
             ├─ BACKCHANNEL  (B, a single continuous backchannel during A's turn)
             └─ response_after_backchannel   (A words after the backchannel — A continues)
  floor_holding_label : continue_current | continue_shorten | continue_simplify | none
  backchannel_type    : acknowledgement | understanding | agreement | uncertainty | disagreement | none

Everything decidable from timestamps/transcripts is done with rules; anything
requiring language/context understanding is delegated to GPT.

Data sources (per conversation):
  - transcript_backbiter.csv : clean turns (backchannels already separated out)
  - transcribe_output.json   : word-level timestamps per channel (used to locate
                               the exact backchannel words and to split Turn 1)

Key idea: the Backbiter transcript already labels each turn's backchannels in
dedicated columns (text, count, speaker, aggregate start/stop). We use those columns
to confirm a backchannel by Speaker B, then take the per-word timing from
transcribe_output within the recorded span and CLUSTER it with a merge-gap:
continuous words collapse into one backchannel, and we require exactly ONE
backchannel (reject if more than one distinct backchannel remains).

Usage:
  python extract_floor_backchannel_backbiter.py --conv_dir <path>
  python extract_floor_backchannel_backbiter.py --gpt_only --conv_dir <path>
"""

import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from collections import Counter
from pathlib import Path
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_CONV_DIR = Path(
    "/data/gpfs/projects/punim2960/jiahengd/dataset/Candor"
    "/part_001/0d4a795e-7026-4d98-b6d0-739a2c7c1661"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL      = "gpt-5.4-2026-03-05"

PAUSE_MIN        = 0.2   # min gap to count as a pause within a turn (s)
SPAN_TOL         = 0.05  # timing tolerance for boundary comparisons (s)
MIN_T1_WORDS    = 4   # Turn1 (A) must be substantive (> MAX_BC_WORDS)
MIN_AFTER_WORDS = 4   # A must genuinely CONTINUE after the backchannel (floor hold)
MIN_T0_WORDS    = 4   # Turn0 (B trigger) must be substantive (> MAX_BC_WORDS)
AUDIO_SR         = 16000

FLOOR_HOLDING_LABELS = ["continue_current", "continue_shorten", "continue_simplify", "none"]
BACKCHANNEL_TYPES    = ["acknowledgement", "understanding", "agreement",
                        "uncertainty", "disagreement", "none"]

# Guidance for the floor-holding label: how A's speech AFTER the backchannel relates
# to what A was doing BEFORE it, in light of the backchannel B gave.
FLOOR_HOLDING_GUIDANCE = {
    "continue_current": (
        "A keeps going on the same track with no obvious change in plan, scope, or "
        "level of detail. The backchannel was simply absorbed (a routine listener "
        "signal) and A continues the answer as if it had not occurred."
    ),
    "continue_shorten": (
        "After the backchannel, A noticeably wraps up, compresses, or cuts the answer "
        "short — e.g. moving to a conclusion, dropping planned detail, or hurrying to "
        "finish. This often follows a backchannel that signals the listener already "
        "agrees or has understood, so further elaboration feels unnecessary."
    ),
    "continue_simplify": (
        "After the backchannel, A rephrases, clarifies, gives an example, or explains "
        "more plainly — restating the same point in simpler terms rather than just "
        "wrapping up. This often follows a backchannel that signals uncertainty, "
        "confusion, or disagreement, prompting A to make the point clearer."
    ),
    "none": (
        "A's continuation after the backchannel does not clearly fit any of the above "
        "categories — e.g. the turn is too short to judge, the backchannel occurs at "
        "the very end of Turn 1, or A's response is ambiguous. Use this as a last resort."
    ),
}

# Guidance for backchannel type, judged from Turn 1 context + the backchannel content.
BACKCHANNEL_TYPE_GUIDANCE = {
    "acknowledgement": (
        "A minimal 'I'm still listening / received' signal that conveys attention "
        "without committing to a stance — e.g. 'mhm', 'uh huh', 'right', 'okay'. It "
        "keeps the channel open but does not claim comprehension, agreement, or doubt."
    ),
    "understanding": (
        "Signals that the listener has grasped/followed the content — e.g. 'oh I see', "
        "'got it', 'ah okay', 'that makes sense'. Stronger than bare acknowledgement: "
        "it claims comprehension of what A is saying."
    ),
    "agreement": (
        "Signals alignment or endorsement of A's point — e.g. 'yeah exactly', 'totally', "
        "'for sure', 'absolutely', 'so true'. The listener is not just following but "
        "actively agreeing with the stance/claim. This is only valid when A actually made some claim that could be agreed on."
    ),
    "uncertainty": (
        "Signals hesitation, doubt, or being unsure/confused — e.g. 'hmm', 'I'm not "
        "sure', 'really?', 'wait', a doubtful 'uh'. The listener is following but not "
        "convinced or not fully clear."
    ),
    "disagreement": (
        "Signals pushback, objection, or non-alignment — e.g. 'no', 'I don't think so', "
        "'mm not really', 'eh'. The listener indicates they do not accept A's point."
    ),
    "none": (
        "The backchannel does not clearly fit any of the above types — e.g. it is "
        "ambiguous, multi-functional, or its social meaning is unclear from context. "
        "Use this as a last resort only when no other type is a reasonable fit."
    ),
}


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


def get_turn_words(row, all_words: list[dict]) -> list[dict]:
    t_start = float(row['start'])
    t_stop  = float(row['stop'])
    return [w for w in all_words if t_start <= w['start'] and w['end'] <= t_stop]


def words_to_text(words: list[dict]) -> str:
    return ' '.join(w['word'] for w in words)


def words_to_timed(words: list[dict]) -> str:
    return ' '.join(f'{w["word"]}({w["start"]:.2f}-{w["end"]:.2f}s)' for w in words)


def has_internal_pause(words: list[dict], pause_min: float = PAUSE_MIN) -> bool:
    """True if any consecutive pair of words is separated by a gap >= pause_min."""
    return any(words[k + 1]['start'] - words[k]['end'] >= pause_min
               for k in range(len(words) - 1))


# ── Step 1: Timing / transcript filter ─────────────────────────────────────────

def find_floor_backchannel_patterns(back: pd.DataFrame,
                                    speaker_words: dict[str, list],
                                    task: str = 'backchannel',
                                    ) -> tuple[list[dict], dict]:
    """
    Scan consecutive Backbiter rows for the structure:
        Turn0(B trigger) → Turn1(A, holds floor with ONE backchannel from B).

    Rules applied here (timestamps / transcript only):
      - Turn1 is Speaker A, >= MIN_T1_WORDS words, and has NO pause anywhere in it.
      - No turn-level overlap: Turn1 starts only after Turn0 ends (A did not cut B
        off) and the next turn starts only after Turn1 ends (A was not cut off), so
        Turn1 is a clean, uninterrupted floor hold. (The backchannel still overlaps
        Turn1 — it is absorbed by Backbiter and is not a turn.)
      - A backchannel by B is confirmed by the Backbiter columns: backchannel_speaker
        must be B, backchannel_count must be exactly 1 (Backbiter already segmented
        separate events), and backchannel_start/stop give the timing.
      - A speaks before the backchannel (>= MIN_BEFORE_WORDS) AND continues after it
        (>= MIN_AFTER_WORDS) — the latter is what makes it floor-holding.

    For task='backchannel', T0 is kept as context only — no T0 quality check
    gates candidates (not same-speaker, not word count, not internal pause,
    not T1-starts-after-T0). For task='floor_holding' or 'both', the full
    T0 quality filter applies.
    """
    check_t0 = task in ('floor_holding', 'both')
    turns   = back.reset_index(drop=True)
    conv_id = back.attrs.get('conv_id', '')
    candidates = []
    seen: set = set()

    stats = {
        'windows_checked':  0,
        'same_speaker':     0,   # T0 and T1 are the same speaker
        't0_few_words':     0,   # T0 < MIN_T0_WORDS
        't0_has_pause':     0,   # T0 has internal pause
        't1_few_words':     0,   # T1 < MIN_T1_WORDS (not a backchannel)
        't1_has_pause':     0,   # T1 has internal pause
        't1_overlap':       0,   # T1 overlaps T0 or is interrupted by others
        'no_bc_annotation': 0,   # Backbiter BC annotation missing/invalid
        'after_few_words':  0,   # words after BC < MIN_AFTER_WORDS
        'duplicate':        0,
        'passed':           0,
    }

    for i in range(1, len(turns)):
        t0_row = turns.iloc[i - 1]
        t1_row = turns.iloc[i]

        stats['windows_checked'] += 1

        spk_a = t1_row['speaker']
        spk_b = t0_row['speaker']

        # For floor_holding/both we need a proper B-A alternation; for backchannel
        # the preceding turn's speaker is irrelevant — T0 is context only.
        if check_t0:
            if spk_a == spk_b:
                stats['same_speaker'] += 1
                continue

        a_all = speaker_words.get(spk_a, [])
        b_all = speaker_words.get(spk_b, [])

        # ── Turn0: quality gates only for floor_holding / both ──────────────────
        # For backchannel, T0 is fetched for the GPT prompt but never gates
        # candidates (skips: same-speaker, few-words, internal pause,
        # anonymisation, T1-starts-after-T0).
        t0_words = get_turn_words(t0_row, b_all)
        if check_t0:
            if len(t0_words) < MIN_T0_WORDS:
                stats['t0_few_words'] += 1
                continue
            if has_internal_pause(t0_words):
                stats['t0_has_pause'] += 1
                continue

        # ── Turn1 (A): long enough, no pause anywhere in it ───────────────────
        t1_words = get_turn_words(t1_row, a_all)
        if len(t1_words) < MIN_T1_WORDS:
            stats['t1_few_words'] += 1
            continue
        if has_internal_pause(t1_words):
            stats['t1_has_pause'] += 1
            continue

        t1_start = t1_words[0]['start']
        t1_stop  = t1_words[-1]['end']

        # ── No turn-level overlap: Turn1 must not be interrupted ───────────────
        # For floor_holding/both: Turn1 must start after Turn0 has finished.
        # For backchannel: that boundary is irrelevant — only check that T1 is
        # not cut short by an external speaker.
        if check_t0 and t0_words:
            if t1_start < t0_words[-1]['end'] - 0.3:
                stats['t1_overlap'] += 1
                continue
        # …and the next DIFFERENT-speaker turn must not start before Turn1 ends.
        # Skip if the next row is still Speaker A.
        if i + 1 < len(turns):
            next_row = turns.iloc[i + 1]
            if next_row['speaker'] != spk_a:
                if float(next_row['start']) < t1_stop - SPAN_TOL:
                    stats['t1_overlap'] += 1
                    continue

        # ── Backchannel: read directly from Backbiter columns ────────────────
        # backchannel_speaker confirms B gave the backchannel.
        # backchannel_count == 1 means exactly one continuous backchannel event
        # (Backbiter already did the segmentation).
        # backchannel_start/stop give the timing; backchannel gives the text.
        bc_spk = t1_row['backchannel_speaker']
        if pd.isna(bc_spk) or bc_spk != spk_b:
            stats['no_bc_annotation'] += 1
            continue
        bc_count = t1_row['backchannel_count']
        if pd.isna(bc_count) or int(bc_count) != 1:
            stats['no_bc_annotation'] += 1
            continue  # require exactly one continuous backchannel
        bc_col_start = t1_row['backchannel_start']
        bc_col_stop  = t1_row['backchannel_stop']
        if pd.isna(bc_col_start) or pd.isna(bc_col_stop):
            stats['no_bc_annotation'] += 1
            continue
        bc_start = float(bc_col_start)
        bc_end   = float(bc_col_stop)
        bc_text  = str(t1_row['backchannel'])

        # Pull B's per-word timestamps within the span for the GPT prompt only
        bc_words = [w for w in b_all
                    if w['start'] < bc_end + SPAN_TOL
                    and w['end']  > bc_start - SPAN_TOL]

        # ── Split Turn1 around the backchannel ────────────────────────────────
        before = [w for w in t1_words if w['start'] < bc_start]
        after  = [w for w in t1_words if w['start'] >= bc_end]
        if not before:
            stats['before_no_words'] = stats.get('before_no_words', 0) + 1
            continue  # BC started before T1's first word — degenerate case
        if len(after) < MIN_AFTER_WORDS:
            stats['after_few_words'] += 1
            continue  # A must genuinely continue → floor holding

        # ── De-duplicate ──────────────────────────────────────────────────────
        key = (conv_id, spk_a, round(t1_start, 2))
        if key in seen:
            stats['duplicate'] += 1
            continue
        seen.add(key)

        stats['passed'] += 1
        candidates.append({
            'type':     'floor_backchannel',
            'source':   'backbiter',
            'conv_id':  conv_id,
            'spk_a':    spk_a,
            'spk_b':    spk_b,

            't0_turn_id': int(t0_row['turn_id']),
            't1_turn_id': int(t1_row['turn_id']),

            # Timing
            't0_start': round(float(t0_row['start']), 3),
            't0_end':   round(float(t0_row['stop']), 3),
            't1_start': round(t1_start, 3),
            't1_end':   round(t1_stop, 3),
            'bc_start': round(bc_start, 3),
            'bc_end':   round(bc_end, 3),
            'before_end':  round(before[-1]['end'], 3),
            'after_start': round(after[0]['start'], 3),
            'n_backchannel_words': len(bc_words),

            # Transcripts
            't0_text':     words_to_text(t0_words),
            't1_text':     words_to_text(t1_words),
            't1_timed':    words_to_timed(t1_words),
            'before_text': words_to_text(before),
            'bc_text':     bc_text,
            'bc_timed':    words_to_timed(bc_words) if bc_words else '',
            'after_text':  words_to_text(after),

            # GPT fields (filled in Step 2)
            'gpt_valid':             None,
            'gpt_task':              None,
            'turn0_complete':        None,
            'turn0_self_contained':  None,
            'turn1_complete':        None,
            'turn1_self_contained':  None,
            'floor_holding_label':   None,
            'backchannel_type':      None,
            'gpt_reason':                 None,
        })

    return candidates, stats


# ── Step 2: GPT Classification ─────────────────────────────────────────────────

def _label_block(guidance: dict) -> str:
    return "\n".join(f"   - {name}: {text}" for name, text in guidance.items())


_EPISODE_HEADER = """\
You are a dialogue analysis expert labelling a floor-holding + backchannel episode \
for research on conversational behaviour.

Structure of the episode:
  Turn 0 (Speaker B) is the previous turn that prompted Speaker A to respond.
  Turn 1 (Speaker A) is A's response. WHILE A is speaking, Speaker B drops a single \
short BACKCHANNEL, and A keeps the floor and continues afterwards.

--- Turn 0 (Speaker B — the trigger) ---
"{t0}"

--- Turn 1 (Speaker A — holds the floor), full text with word timestamps ---
"{t1}"
  Word timestamps: {t1_timed}

  A said BEFORE the backchannel:
  "{before}"

  BACKCHANNEL from Speaker B (during A's turn):
  "{bc}"
  Backchannel timestamps: {bc_timed}

  A continued AFTER the backchannel:
  "{after}"

---
Evaluate each field in order. Complete ALL fields even if an earlier answer is \
false — just set is_valid to false at the end.\
"""

# ── Task: backchannel signal ───────────────────────────────────────────────────
# Validity: Turn 1 must be complete AND self-contained.
# Use case: give Turn 1 to a model, ask it to generate a backchannel response.

GPT_PROMPT_BACKCHANNEL = _EPISODE_HEADER + """

1. TURN1_COMPLETE: Is Turn 1 (the full Speaker A text) a complete, well-formed \
utterance (not a fragment)?

2. TURN1_SELF_CONTAINED: Can Turn 1 be fully understood WITHOUT any prior \
conversational context (Turn 0 is given only for your understanding; do NOT rely on \
it)? Be strict:
   - Reject if it contains references whose meaning is unclear without context \
(e.g. "that one", "it", "those").
   - Reject if it is clearly only interpretable as a direct answer to an unseen \
question, or a bare reaction ("Are you serious?", "Really?").
   - Accept only if it introduces a complete, interpretable thought.

3. FLOOR_HOLDING_LABEL: Compare what A said AFTER the backchannel to what A was \
doing BEFORE it, in light of the backchannel, and choose exactly one:
{floor_holding_block}

4. BACKCHANNEL_TYPE: From Turn 1's context and the backchannel content, choose the \
single best type of the backchannel B gave:
{backchannel_block}

5. IS_VALID: Set to true ONLY if turn1_complete=true and turn1_self_contained=true.

Respond ONLY with JSON (no markdown):
{{
  "turn1_complete": true/false,
  "turn1_self_contained": true/false,
  "floor_holding_label": "continue_current|continue_shorten|continue_simplify|none",
  "backchannel_type": "acknowledgement|understanding|agreement|uncertainty|disagreement|none",
  "is_valid": true/false,
  "reason": "one sentence summary"
}}"""

# ── Task: floor-holding simulation ────────────────────────────────────────────
# Validity: Turn 0 must be complete AND self-contained; Turn 1 must be complete.
# Use case: give Turn 0 to a model as prompt, model responds, replay BC mid-response.

GPT_PROMPT_FLOOR_HOLDING = _EPISODE_HEADER + """

1. TURN0_COMPLETE: Is Turn 0 (the trigger) a complete, well-formed utterance \
(not a fragment)?

2. TURN0_SELF_CONTAINED: Can Turn 0 be fully understood WITHOUT any prior \
conversational context?
   - Reject if it requires knowing what was said before to make sense.
   - Reject if it is only interpretable as a reply to an unseen utterance.
   - Accept if it introduces a complete, interpretable thought or question \
on its own.

3. TURN1_COMPLETE: Is Turn 1 (the full Speaker A text) a complete, well-formed \
utterance (not a fragment)?

4. FLOOR_HOLDING_LABEL: Compare what A said AFTER the backchannel to what A was \
doing BEFORE it, in light of the backchannel, and choose exactly one:
{floor_holding_block}

5. BACKCHANNEL_TYPE: From Turn 1's context and the backchannel content, choose the \
single best type of the backchannel B gave:
{backchannel_block}

6. IS_VALID: Set to true ONLY if turn0_complete=true AND \
turn0_self_contained=true AND turn1_complete=true.

Respond ONLY with JSON (no markdown):
{{
  "turn0_complete": true/false,
  "turn0_self_contained": true/false,
  "turn1_complete": true/false,
  "floor_holding_label": "continue_current|continue_shorten|continue_simplify|none",
  "backchannel_type": "acknowledgement|understanding|agreement|uncertainty|disagreement|none",
  "is_valid": true/false,
  "reason": "one sentence summary"
}}"""

# ── Task: both ────────────────────────────────────────────────────────────────
# Validity: all four criteria must hold (intersection of both tasks).

GPT_PROMPT_BOTH = _EPISODE_HEADER + """

1. TURN0_COMPLETE: Is Turn 0 (the trigger) a complete, well-formed utterance \
(not a fragment)?

2. TURN0_SELF_CONTAINED: Can Turn 0 be fully understood WITHOUT any prior \
conversational context?
   - Reject if it requires prior context to make sense.
   - Accept if it introduces a complete, interpretable thought or question \
on its own.

3. TURN1_COMPLETE: Is Turn 1 (the full Speaker A text) a complete, well-formed \
utterance (not a fragment)?

4. TURN1_SELF_CONTAINED: Can Turn 1 be fully understood WITHOUT any prior \
conversational context (Turn 0 is given only for your understanding)? Be strict:
   - Reject if it contains references whose meaning is unclear without context \
(e.g. "that one", "it", "those").
   - Reject if it is clearly only interpretable as a direct answer to an unseen \
question, or a bare reaction.
   - Accept only if it introduces a complete, interpretable thought.

5. FLOOR_HOLDING_LABEL: Compare what A said AFTER the backchannel to what A was \
doing BEFORE it, in light of the backchannel, and choose exactly one:
{floor_holding_block}

6. BACKCHANNEL_TYPE: From Turn 1's context and the backchannel content, choose the \
single best type of the backchannel B gave:
{backchannel_block}

7. IS_VALID: Set to true ONLY if ALL of turn0_complete, turn0_self_contained, \
turn1_complete, and turn1_self_contained are true.

Respond ONLY with JSON (no markdown):
{{
  "turn0_complete": true/false,
  "turn0_self_contained": true/false,
  "turn1_complete": true/false,
  "turn1_self_contained": true/false,
  "floor_holding_label": "continue_current|continue_shorten|continue_simplify|none",
  "backchannel_type": "acknowledgement|understanding|agreement|uncertainty|disagreement|none",
  "is_valid": true/false,
  "reason": "one sentence summary"
}}"""

_TASK_PROMPT = {
    "backchannel":   GPT_PROMPT_BACKCHANNEL,
    "floor_holding": GPT_PROMPT_FLOOR_HOLDING,
    "both":          GPT_PROMPT_BOTH,
}


class QuotaExceededError(RuntimeError):
    pass


def gpt_classify(candidates: list[dict], client: OpenAI,
                 task: str = 'backchannel',
                 retry_nulls_only: bool = False,
                 max_valid: int | None = None,
                 cat_budget: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    """Label candidates with GPT. Budget (if given) is keyed on floor_holding_label;
    since the label is only known after labelling, over-budget valids are demoted to
    invalid, and we stop once every budgeted label is full."""
    pending = [c for c in candidates
               if not retry_nulls_only or c.get('gpt_valid') is None]
    total = len(pending)
    print(f"  Sending {total} candidates to GPT ({GPT_MODEL}, task={task})...")
    if cat_budget is not None:
        print(f"    Per-type budget: {cat_budget}")
    floor_block   = _label_block(FLOOR_HOLDING_GUIDANCE)
    bc_block      = _label_block(BACKCHANNEL_TYPE_GUIDANCE)
    prompt_tmpl   = _TASK_PROMPT[task]
    n_valid       = 0
    n_valid_cat: dict[str, int] = {}
    for i, cand in enumerate(pending):
        if max_valid is not None and n_valid >= max_valid:
            print(f"    Reached max_valid={max_valid}; stopping GPT calls early.")
            break
        if cat_budget is not None and all(
                n_valid_cat.get(c, 0) >= b for c, b in cat_budget.items()):
            print("    All type budgets met; stopping GPT calls early.")
            break

        msg = prompt_tmpl.format(
            t0=cand['t0_text'],
            t1=cand['t1_text'],
            t1_timed=cand.get('t1_timed', ''),
            before=cand['before_text'],
            bc=cand['bc_text'],
            bc_timed=cand.get('bc_timed', ''),
            after=cand['after_text'],
            floor_holding_block=floor_block,
            backchannel_block=bc_block,
        )
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": msg}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)

            if task in ('floor_holding', 'both'):
                cand['turn0_complete']       = result.get('turn0_complete')
                cand['turn0_self_contained'] = result.get('turn0_self_contained')
            cand['turn1_complete']       = result.get('turn1_complete')
            if task in ('backchannel', 'both'):
                cand['turn1_self_contained'] = result.get('turn1_self_contained')
            cand['floor_holding_label']  = result.get('floor_holding_label')
            cand['backchannel_type']     = result.get('backchannel_type')
            cand['gpt_reason']           = result.get('reason', '')
            cand['gpt_task']             = task

            if task == 'backchannel':
                valid = bool(result.get('turn1_complete') and
                             result.get('turn1_self_contained'))
            elif task == 'floor_holding':
                valid = bool(result.get('turn0_complete') and
                             result.get('turn0_self_contained') and
                             result.get('turn1_complete'))
            else:  # both
                valid = bool(result.get('turn0_complete') and
                             result.get('turn0_self_contained') and
                             result.get('turn1_complete') and
                             result.get('turn1_self_contained'))

            floor_label = cand['floor_holding_label'] or 'none'
            if valid and cat_budget is not None:
                if n_valid_cat.get(floor_label, 0) >= cat_budget.get(floor_label, 0):
                    valid = False
                    cand['gpt_reason'] = (cand.get('gpt_reason', '') +
                                          " [demoted: floor-label budget reached]")
            cand['gpt_valid'] = valid
            if valid:
                n_valid += 1
                n_valid_cat[floor_label] = n_valid_cat.get(floor_label, 0) + 1
        except Exception as e:
            err_str = str(e)
            if 'insufficient_quota' in err_str:
                cand['gpt_valid'] = None
                cand['gpt_reason'] = f"API error: {e}"
                raise QuotaExceededError(
                    f"OpenAI quota exceeded at candidate {i} conv={cand.get('conv_id')}"
                ) from e
            print(f"    [WARN] GPT failed for candidate {i}: {e}")
            cand['gpt_valid']  = None
            cand['gpt_reason'] = f"API error: {e}"
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(pending)} done")

    judged = [c for c in candidates if c.get('gpt_valid') is not None]
    gpt_stats: dict = {
        'total_sent':           total,
        'total_judged':         len(judged),
        'turn1_complete_false': sum(1 for c in judged if c.get('turn1_complete') is False),
        'floor_holding_label':  dict(Counter(c.get('floor_holding_label', '?') for c in judged)),
        'backchannel_type':     dict(Counter(c.get('backchannel_type', '?')     for c in judged)),
        'valid_by_label':       dict(Counter(
            c.get('floor_holding_label', '?')
            for c in judged if c.get('gpt_valid')
        )),
        'total_valid':          sum(1 for c in judged if c.get('gpt_valid')),
    }
    if task in ('floor_holding', 'both'):
        gpt_stats['turn0_complete_false']       = sum(1 for c in judged if c.get('turn0_complete')       is False)
        gpt_stats['turn0_self_contained_false'] = sum(1 for c in judged if c.get('turn0_self_contained') is False)
    if task in ('backchannel', 'both'):
        gpt_stats['turn1_self_contained_false'] = sum(1 for c in judged if c.get('turn1_self_contained') is False)

    print(f"  VALID_AFTER_LLM: {gpt_stats['total_valid']}")
    print(f"GPT_STATS:{json.dumps(gpt_stats)}")
    return candidates, gpt_stats


# ── Step 3: Audio Export ───────────────────────────────────────────────────────

def silence_other_speaker(clip: np.ndarray, sr: int,
                          clip_start_s: float,
                          other_words: list[dict]) -> np.ndarray:
    """Zero out frames in `clip` where the other speaker's words fall."""
    clip = clip.copy()
    n = len(clip)
    for w in other_words:
        s = max(0, int((w['start'] - clip_start_s) * sr))
        e = min(n, int((w['end']   - clip_start_s) * sr))
        if s < e:
            clip[s:e] = 0.0
    return clip


def export_clip(audio: np.ndarray, sr: int,
                start_s: float, end_s: float,
                out_path: Path, ch_idx: int,
                other_words: list[dict] | None = None) -> None:
    s = max(0, int(start_s * sr))
    e = min(audio.shape[-1], int(end_s * sr))
    clip = audio[ch_idx, s:e] if audio.ndim == 2 else audio[s:e]
    if other_words:
        clip = silence_other_speaker(clip, sr, start_s, other_words)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr)


def export_stereo_clip(audio: np.ndarray, sr: int,
                       start_s: float, end_s: float,
                       out_path: Path) -> None:
    s = max(0, int(start_s * sr))
    e = min(audio.shape[-1], int(end_s * sr))
    clip = audio[:, s:e].T if audio.ndim == 2 else audio[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr)


def export_candidates(candidates: list[dict],
                      audio: np.ndarray, sr: int,
                      audio_dir: Path,
                      spk_to_ch: dict[str, int],
                      speaker_words: dict[str, list],
                      results_dir: Path | None = None) -> None:
    """Export per valid candidate, into <backchannel_type>/:
       t0 (B trigger), t1_full (A, B silenced), backchannel (B only), full_stereo."""
    import shutil
    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue
        category = cand.get('backchannel_type', 'unknown')
        out_dir  = audio_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)
        conv_id = cand['conv_id']
        base    = f"{conv_id}_t{cand['t1_turn_id']}"
        ch_a = spk_to_ch.get(cand['spk_a'], 0)
        ch_b = spk_to_ch.get(cand['spk_b'], 1)

        words_a = speaker_words.get(cand['spk_a'], [])
        words_b = speaker_words.get(cand['spk_b'], [])

        def b_in(t0, t1):
            return [w for w in words_b if w['start'] < t1 and w['end'] > t0]

        def a_in(t0, t1):
            return [w for w in words_a if w['start'] < t1 and w['end'] > t0]

        clips = [
            # t0: B's trigger turn (A's mic-bleed silenced)
            (cand['t0_start'], cand['t0_end'], f"{base}_t0.wav",          ch_b,  False, a_in),
            # t1_full: A's full turn (B's words silenced so only A's speech remains)
            (cand['t1_start'], cand['t1_end'], f"{base}_t1_full.wav",     ch_a,  False, b_in),
            # backchannel: B's full channel from T1 start to T1 end (no silencing)
            # so the backchannel word is audible in context on B's mic track.
            (cand['t1_start'], cand['t1_end'], f"{base}_backchannel.wav", ch_b,  False, None),
            # full_stereo: T0 start → T1 end (both channels, no silencing)
            (cand['t0_start'], cand['t1_end'], f"{base}_full_stereo.wav", None,  True,  None),
        ]
        for t_start, t_end, fname, ch, stereo, other_fn in clips:
            dest = out_dir / fname
            if stereo:
                export_stereo_clip(audio, sr, t_start, t_end, dest)
            else:
                other = other_fn(t_start, t_end) if other_fn else None
                export_clip(audio, sr, t_start, t_end, dest, ch, other_words=other)
            if results_dir is not None:
                global_dir = results_dir / category
                global_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, global_dir / fname)
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
         cat_budget: dict[str, int] | None = None,
         task: str = 'backchannel') -> None:
    conv_id    = conv_dir.name
    output_dir = conv_dir / "floor_backchannel_backbiter"
    f1_dir     = output_dir / "filter1_timing"
    f2_dir     = output_dir / "filter2_gpt"
    audio_dir  = output_dir / "audio"
    audio_path = conv_dir / "processed" / f"{conv_id}.mp3"
    chmap_path = conv_dir / "processed" / "channel_map.json"
    back_path  = conv_dir / "transcription" / "transcript_backbiter.csv"
    json_path  = conv_dir / "transcription" / "transcribe_output.json"

    trans_json = json.load(open(json_path))
    chmap      = json.load(open(chmap_path))

    ch_to_speaker: dict[str, str] = {"ch_0": chmap["L"], "ch_1": chmap["R"]}
    spk_to_ch = {spk: int(key.split('_')[1]) for key, spk in ch_to_speaker.items()}
    print(f"  Channel mapping: {ch_to_speaker}")

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
                    for field in ('gpt_valid', 'gpt_task',
                                  'turn0_complete', 'turn0_self_contained',
                                  'turn1_complete', 'turn1_self_contained',
                                  'floor_holding_label', 'backchannel_type', 'gpt_reason'):
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

        print("Building per-speaker word list...")
        for spk, wds in speaker_words.items():
            print(f"  {spk}: {len(wds)} words")

        print("\n── Step 1: Timing / transcript filter ──")
        candidates, timing_stats = find_floor_backchannel_patterns(back, speaker_words, task=task)
        print(f"  Candidates found: {len(candidates)}")
        print(f"TIMING_STATS:{json.dumps(timing_stats)}", flush=True)
        sys.stdout.flush()

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
    candidates, _gpt_stats = gpt_classify(candidates, client, task=task,
                                          retry_nulls_only=gpt_only,
                                          max_valid=max_valid, cat_budget=cat_budget)

    valid = [c for c in candidates if c.get('gpt_valid')]
    print(f"\n  Valid after GPT: {len(valid)}/{len(candidates)}")
    by_floor: dict = {}
    by_bc:    dict = {}
    for c in valid:
        by_floor[c['floor_holding_label']] = by_floor.get(c['floor_holding_label'], 0) + 1
        by_bc[c['backchannel_type']]       = by_bc.get(c['backchannel_type'], 0) + 1
    if by_floor:
        print("    floor_holding_label: " +
              "  ".join(f"{k}={v}" for k, v in by_floor.items()))
    if by_bc:
        print("    backchannel_type: " +
              "  ".join(f"{k}={v}" for k, v in by_bc.items()))

    print("\nSaving Filter 2 results (GPT)...")
    save_results(candidates, f2_dir / "all_with_gpt_decisions.json", "all+decisions")

    # ── Step 3: Audio export ───────────────────────────────────────────────────
    if valid:
        print(f"\n── Step 3: Audio export ──")
        print(f"Loading audio from {audio_path.name}...")
        audio, sr = librosa.load(str(audio_path), sr=AUDIO_SR, mono=False)
        print(f"  Audio: {audio.shape}, sr={sr}")
        export_candidates(candidates, audio, sr, audio_dir, spk_to_ch,
                          speaker_words=speaker_words, results_dir=results_dir)
        print(f"  Exported {len(valid)} × 4 clips → {audio_dir.name}/")
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
    print(f"  audio/           ← <backchannel_type>/<conv>_t0/t1_full/backchannel/full_stereo.wav")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract floor-holding + backchannel episodes using Backbiter transcript.")
    parser.add_argument("--conv_dir",    type=Path, default=DEFAULT_CONV_DIR)
    parser.add_argument("--gpt_only",    action="store_true",
                        help="Skip timing filter; re-run GPT on existing filter1 results")
    parser.add_argument("--no_gpt",      action="store_true",
                        help="Run timing filter only; skip GPT classification and audio export")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Global results directory — audio mirrored here")
    parser.add_argument("--max_valid",   type=int, default=None,
                        help="Stop GPT calls after this many valid candidates")
    parser.add_argument("--cat_budget",  type=str, default=None,
                        help='JSON dict of per-backchannel_type valid budgets, e.g. '
                             '\'{"acknowledgement": 5}\'')
    parser.add_argument("--task", choices=['backchannel', 'floor_holding', 'both'],
                        default='backchannel',
                        help="GPT evaluation task: 'backchannel' validates T1 "
                             "self-containedness; 'floor_holding' validates T0 "
                             "self-containedness + T1 completeness; 'both' requires all.")
    args = parser.parse_args()
    cat_budget = json.loads(args.cat_budget) if args.cat_budget else None
    main(args.conv_dir, gpt_only=args.gpt_only, no_gpt=args.no_gpt,
         results_dir=args.results_dir,
         max_valid=args.max_valid, cat_budget=cat_budget, task=args.task)
