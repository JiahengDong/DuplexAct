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
  python extract_interruptions_backbiter.py --conv_dir <path>
  python extract_interruptions_backbiter.py --gpt_only --conv_dir <path>
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
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
MAX_T3_DURATION          = 10.0  # longer T3 = floor takeover, not interruption
T2_MIN_WORDS             = 4     # Turn2 must have >= this many words total
T2_MIN_PRE_WORDS         = 4     # Turn2 must have >= this many words BEFORE overlap
AUDIO_SR                = 16000
TARGET_PER_TYPE         = 500    # cap per recovery type at aggregation
RECOVERY_TYPES          = ["correct", "enrich", "shorten", "repeat",
                            "shift_topic", "ask_followup", "none"]
QUOTA_EXIT_CODE         = 2
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


# ── Step 1: Timing Filter ──────────────────────────────────────────────────────

def find_interruption_patterns(back: pd.DataFrame,
                                speaker_words: dict[str, list]) -> tuple[list[dict], dict]:
    """
    Scan backbiter turns for the 4-turn interruption pattern:
      T1(A) → T2(B) → T3(A, overlaps T2) → T4(B)
    Returns (candidates, timing_stats).
    """
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
        't2_is_question':     0,
        't2_few_words':       0,
        't2_pre_words_short': 0,
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

        # ── T2 must NOT start while T1 is still running ───────────────────────
        if t2_start < t1_end - TIMESTAMP_NOISE_TOLERANCE:
            stats['t2_overlaps_t1'] += 1
            continue

        # ── T3 MUST overlap T2 (the actual interruption) ──────────────────────
        overlap = t2_end - t3_start
        if overlap < MIN_INTERRUPTION_OVERLAP:
            stats['t3_overlap_small'] += 1
            continue

        # ── T4 must start after T3 ends ───────────────────────────────────────
        if float(t4_row['start']) < t3_end - TIMESTAMP_NOISE_TOLERANCE:
            stats['t4_overlaps_t3'] += 1
            continue

        # ── T1: get words ─────────────────────────────────────────────────────
        t1_words = get_turn_words(t1_row, speaker_words.get(spk_a, []))
        if not t1_words:
            stats['t1_few_words'] += 1
            continue

        # ── T1: sentence-final punctuation ────────────────────────────────────
        if not t1_words[-1]['sent_end']:
            stats['t1_no_sent_final'] += 1
            continue

        # ── T1: minimum content words ─────────────────────────────────────────
        if len(t1_words) < MIN_SUBSTANTIVE_WORDS:
            stats['t1_few_words'] += 1
            continue

        # ── T1: no internal pauses ────────────────────────────────────────────
        if any(
            t1_words[k + 1]['start'] - t1_words[k]['end'] >= PAUSE_MIN
            for k in range(len(t1_words) - 1)
        ):
            stats['t1_has_pauses'] += 1
            continue

        # ── T2: not a question ────────────────────────────────────────────────
        if bool(t2_row.get('end_question', False)):
            stats['t2_is_question'] += 1
            continue

        # ── T2: minimum total words ───────────────────────────────────────────
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
        if len(t2_pre_words) < T2_MIN_PRE_WORDS:
            stats['t2_pre_words_short'] += 1
            continue

        # ── T3: no internal pauses ────────────────────────────────────────────
        t3_words = get_turn_words(t3_row, speaker_words.get(spk_a, []))
        if not t3_words:
            stats['t3_empty'] += 1
            continue
        if any(
            t3_words[k + 1]['start'] - t3_words[k]['end'] >= PAUSE_MIN
            for k in range(len(t3_words) - 1)
        ):
            stats['t3_has_pauses'] += 1
            continue

        # ── T3: duration limit (not a full floor take-over) ───────────────────
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
        if not t4_words or len(t4_words) < MIN_SUBSTANTIVE_WORDS:
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

            # GPT fields (filled in Step 2)
            'gpt_valid':            None,
            'turn1_complete':       None,
            'turn1_self_contained': None,
            'turn2_responds':       None,
            'turn2_interrupted':    None,
            'turn4_meaningful':     None,
            'interruption_style':   None,
            'interruption_type':    None,
            'gpt_reason':           None,
        })

    return candidates, stats


# ── Step 2: GPT Classification ─────────────────────────────────────────────────
# Identical prompt to the Cliffhanger version — same semantic checks apply.

_CANDOR_CTX = (
    "The conversation is from the CANDOR corpus — a dyadic spoken dialogue "
    "between two strangers recorded over a video-call platform."
)

GPT_PROMPT = """\
You are a dialogue analysis expert evaluating a 4-turn conversational pattern \
for research on interruption and recovery in spoken dialogue.

""" + _CANDOR_CTX + """

Here are four consecutive turns from a conversation, with segment-level \
timestamps to help you assess timing precisely:

Turn 1 (Speaker A):
"{t1}"
  Segment timestamps: {t1_timed}

Turn 2 (Speaker B — response):
"{t2}"
  Segment timestamps: {t2_pre_timed} ... {t2_post_timed}
  [Turn 2 ran {t2_start:.2f}s–{t2_end:.2f}s; Turn 3 started at {t3_start:.2f}s, \
i.e. {t2_pre_words} content words into Turn 2, with {overlap:.2f}s of overlap remaining]

Turn 3 (Speaker A — started at {t3_start:.2f}s, {overlap:.2f}s before Turn 2 ended):
"{t3}"
  Segment timestamps: {t3_timed}

Turn 4 (Speaker B — recovery/response):
"{t4}"
  Segment timestamps: {t4_timed}

---
Evaluate each turn in order below. Complete ALL fields even if an earlier answer \
is false — just set is_valid to false at the end.

── TURN 1 (Speaker A) ──────────────────────────────────────────────────────────

1. TURN1_COMPLETE: Is Turn 1 a complete, well-formed utterance (not a fragment)?

2. TURN1_SELF_CONTAINED: Can Turn 1 be fully understood and responded to WITHOUT \
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

Answer true if Turn 2's final words (the point where B stopped) leave the thought \
clearly unfinished — e.g. it trails off mid-clause, ends on a dangling conjunction \
or preposition, ends with an incomplete predicate, or stops in the middle of an \
idea B was still developing, so that B evidently still had more to say.

Answer false if Turn 2 had already reached a natural completion point — a complete \
clause or sentence (subject + verb + complement) that stands on its own, or B was \
plainly wrapping up — so Speaker A simply took the next turn with slight overlap \
(normal turn-taking).

── TURN 3 (Speaker A — interruption) ───────────────────────────────────────────

5. INTERRUPTION_STYLE: Determine whether Turn 3 is a meaningful floor-taking \
interruption of Turn 2. (We only classify hard interruptions — see below.)

First identify the FLOOR-TAKING ONSET: the first substantive word or phrase in \
Turn 3 that attempts to take the floor. Ignore leading fillers, backchannels, and \
acknowledgements such as "yeah," "uh-huh," "right," or "okay."

Turn 3 qualifies as an interruption only if ALL conditions below hold:

(a) FLOOR-TAKING INTENT: At the floor-taking onset, Turn 3 contributes meaningful \
content that requires or invites a response from Speaker B. It may ask a question, \
request clarification, make a correction, challenge B's claim, introduce a \
counterpoint, or add substantive information.

The following alone do NOT qualify:
     • Pure backchannels: "Uh huh", "Yeah", "Mm hmm", "Right", "Okay"
     • Short approval/reaction phrases: "That's cool", "Nice", "Interesting", \
"Oh wow", "Good for you", "That makes sense", "Okay cool"
     • Wrap-up signals: "Okay that's cool", "Yeah that's great", "Sounds good"
     • Empathetic acknowledgements: "I know", "I know right", "I can imagine", \
"That must be hard", "I totally get that", "I understand"

Questions such as "Why?", "Which one?", "What do you mean?", and "How so?" \
DO qualify because they request a meaningful response.

(b) PRAGMATIC COMPLETENESS: Turn 3 forms a complete conversational action, even \
if it is not a full grammatical sentence. A short but interpretable question, \
correction, or challenge can qualify.

(c) MID-FLOW ONSET: The floor-taking onset occurs while Turn 2 is still \
substantively in progress — i.e. Turn 2 had not yet reached a natural ending when \
Turn 3 began (consistent with TURN2_INTERRUPTED above). 
Reject if Turn 2 had already completed or almost about to complete a full, self-standing thought before \
the onset.

If all three conditions hold, classify:
   - "hard": Turn 2 stops at a non-natural point — B's final words are \
syntactically or semantically unfinished, such as a cut-off clause, dangling \
conjunction, incomplete predicate, filler/disfluency, or abandoned phrase. \
(turn2_interrupted should be true)
   - "none": Any of the three conditions above fails, OR Turn 2 had already \
reached a natural completion before A started (overlapping turn-taking, not a \
disruptive interruption).

── TURN 4 (Speaker B — recovery) ───────────────────────────────────────────────

6. TURN4_MEANINGFUL: Does Turn 4 provide a substantive response to the content or \
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

7. RECOVERY_TYPE: Based on the content of Turn 3 and B's response in Turn 4, \
classify the recovery as one of:
   - "correct"     – A corrects an error; B acknowledges or revises.
   - "enrich"      – A asks for more detail; B elaborates on the same point.
   - "shorten"     – A signals B should be more concise; B wraps up.
   - "repeat"      – A asks B to repeat something; B repeats.
   - "shift_topic" – A redirects to a different subject; B follows the new topic.
   - "ask_followup"– After being interrupted, B asks A to clarify A's interruption.
   - "none"        – Turn 4 is not substantive, or no recovery type can be assigned.

── VALIDITY ────────────────────────────────────────────────────────────────────

8. IS_VALID: Set to true ONLY if ALL of the following hold:
   turn1_complete=true, turn1_self_contained=true, turn2_responds=true,
   interruption_style="hard", turn4_meaningful=true.

Respond ONLY with JSON (no markdown):
{{
  "turn1_complete": true/false,
  "turn1_self_contained": true/false,
  "turn2_responds": true/false,
  "turn2_interrupted": true/false,
  "interruption_style": "hard"|"none",
  "turn4_meaningful": true/false,
  "recovery_type": "none"|"correct"|"enrich"|"shorten"|"repeat"|"shift_topic"|"ask_followup",
  "is_valid": true/false,
  "reason": "one sentence summary"
}}"""


class QuotaExceededError(RuntimeError):
    pass


def gpt_classify(candidates: list[dict], client: OpenAI,
                 retry_nulls_only: bool = False,
                 max_valid_per_type: int | None = None) -> tuple[list[dict], dict]:
    from collections import Counter

    # Drop candidates where B had already stopped before A started (no real overlap)
    for cand in candidates:
        if cand.get('gpt_valid') is None and cand.get('yield_latency', 1) <= 0:
            cand['gpt_valid'] = False
            cand['gpt_reason'] = "pre-filter: yield_latency <= 0 (B ended before A started)"

    pending = [c for c in candidates
               if not retry_nulls_only or c.get('gpt_valid') is None]
    total = len(pending)
    print(f"  Sending {total} candidates to GPT ({GPT_MODEL})"
          + (f" (cap={max_valid_per_type}/type)" if max_valid_per_type else "") + "...")
    valid_by_type: dict[str, int] = {}

    for i, cand in enumerate(pending):
        # Stop early when every known recovery type has hit the cap
        if (max_valid_per_type is not None
                and len(valid_by_type) >= len(RECOVERY_TYPES)
                and all(valid_by_type.get(t, 0) >= max_valid_per_type
                        for t in RECOVERY_TYPES)):
            print(f"    [{i+1}/{total}] all type caps reached — stopping.")
            break

        msg = GPT_PROMPT.format(
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
            cand['gpt_valid']            = result.get('is_valid', False)
            cand['turn1_complete']       = result.get('turn1_complete')
            cand['turn1_self_contained'] = result.get('turn1_self_contained')
            cand['turn2_responds']       = result.get('turn2_responds')
            cand['turn2_interrupted']    = result.get('turn2_interrupted')
            cand['turn4_meaningful']     = result.get('turn4_meaningful')
            cand['interruption_style']   = result.get('interruption_style')
            cand['interruption_type']    = result.get('recovery_type')
            cand['gpt_reason']           = result.get('reason', '')
            # onset/yield times come directly from timing data, not GPT
            cand['interruption_onset_time'] = cand['t3_start']
            cand['interruption_onset_word'] = cand['t3_timed'].split('(')[0].strip() if cand.get('t3_timed') else None
            cand['turn2_yield_time']        = cand['t2_end']
            cand['turn2_yield_word']        = cand['t2_pre_timed'].rsplit('(', 1)[0].rsplit(' ', 1)[-1].strip() if cand.get('t2_pre_timed') else None

            style = cand.get('interruption_style', 'none')
            rtype = cand.get('interruption_type') or 'none'
            valid = cand.get('gpt_valid', False)
            print(f"    [{i+1}/{total}] {'✓' if valid else '✗'} "
                  f"style={style} type={rtype} "
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

    judged = [c for c in candidates if c.get('gpt_valid') is not None]
    gpt_stats = {
        'total_sent':                 total,
        'total_judged':               len(judged),
        'turn1_complete_false':       sum(1 for c in judged if c.get('turn1_complete')       is False),
        'turn1_self_contained_false': sum(1 for c in judged if c.get('turn1_self_contained') is False),
        'turn2_responds_false':       sum(1 for c in judged if c.get('turn2_responds')       is False),
        'turn2_interrupted_false':    sum(1 for c in judged if c.get('turn2_interrupted')    is False),
        'turn4_meaningful_false':     sum(1 for c in judged if c.get('turn4_meaningful')     is False),
        'interruption_style':         dict(Counter(c.get('interruption_style', '?') for c in judged)),
        'recovery_type':              dict(Counter(c.get('interruption_type',   '?') for c in judged)),
        'valid_by_type':              valid_by_type,
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
                start_s: float, end_s: float,
                out_path: Path, ch_idx: int,
                other_words: list[dict] | None = None) -> None:
    """Export a single-channel clip, optionally silencing other-speaker frames."""
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

        # Words from the other speaker, used to silence bleed frames
        words_a = speaker_words.get(cand['spk_a'], [])
        words_b = speaker_words.get(cand['spk_b'], [])

        def b_words_in(t0, t1):
            return [w for w in words_b if w['start'] < t1 and w['end'] > t0]

        def a_words_in(t0, t1):
            return [w for w in words_a if w['start'] < t1 and w['end'] > t0]

        # (t_start, t_end, filename, ch_idx, is_stereo, other_speaker_words_fn)
        # T3 is the interruption — it intentionally overlaps with T2, so B's last
        # word bleeds into T3's window. Silencing would cut off the start of T3's
        # speech. Skip silencing for T3; keep it raw.
        clips = [
            (cand['t1_start'], cand['t1_end'],  f"{base}_t1.wav",          ch_a, False, b_words_in),
            (cand['t2_start'], cand['t2_end'],  f"{base}_t2.wav",          ch_b, False, a_words_in),
            (cand['t3_start'], cand['t3_end'],  f"{base}_t3.wav",          ch_a, False, None),
            (cand['t4_start'], cand['t4_end'],  f"{base}_t4.wav",          ch_b, False, a_words_in),
            (cand['t1_start'], cand['t4_end'],  f"{base}_full_stereo.wav", None, True,  None),
        ]
        for t_start, t_end, fname, ch, stereo, other_fn in clips:
            dest = out_dir / fname
            if stereo:
                export_stereo_clip(audio, sr, t_start, t_end, dest)
            else:
                other = other_fn(t_start, t_end) if other_fn else None
                export_clip(audio, sr, t_start, t_end, dest, ch, other_words=other)
            if results_dir is not None:
                global_dir = results_dir / style / itype
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
         max_valid: int | None = None) -> None:
    conv_id    = conv_dir.name
    output_dir = conv_dir / "interruption_candidates_backbiter"
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

        # Pre-filter: drop any turn with fewer than MIN_WORDS_FILTER words.
        # Backbiter already absorbed backchannels, so this removes only genuine
        # short non-bc fragments ("No,", "Well,") that are too thin to be
        # meaningful T1/T2/T4 candidates.
        before = len(back)
        back = back[back['n_words'] >= MIN_WORDS_FILTER].reset_index(drop=True)
        print(f"  Pre-filter: {before} → {len(back)} turns (removed {before - len(back)} "
              f"with < {MIN_WORDS_FILTER} words)")

        print("Building per-speaker word list...")
        for spk, wds in speaker_words.items():
            print(f"  {spk}: {len(wds)} words")

        print("\n── Step 1: Timing filter ──")
        candidates, timing_stats = find_interruption_patterns(back, speaker_words)
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
    try:
        candidates, gpt_stats = gpt_classify(candidates, client,
                                             retry_nulls_only=gpt_only,
                                             max_valid_per_type=max_valid)
    except QuotaExceededError as e:
        print(f"  QUOTA EXCEEDED: {e}")
        sys.exit(QUOTA_EXIT_CODE)

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
        description="Extract interruption & recovery patterns using Backbiter transcript.")
    parser.add_argument("--conv_dir",    type=Path, default=DEFAULT_CONV_DIR)
    parser.add_argument("--gpt_only",   action="store_true",
                        help="Skip timing filter; re-run GPT on existing filter1 results")
    parser.add_argument("--no_gpt",     action="store_true",
                        help="Run timing filter only; skip GPT classification and audio export")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Global results directory — audio mirrored here")
    parser.add_argument("--max_valid", type=int, default=None,
                        help="Stop GPT calls after this many valid candidates per recovery type")
    args = parser.parse_args()
    main(args.conv_dir, gpt_only=args.gpt_only, no_gpt=args.no_gpt,
         results_dir=args.results_dir, max_valid=args.max_valid)
