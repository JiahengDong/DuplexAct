#!/usr/bin/env python3
"""
CANDOR Interruption & Recovery Extractor

Finds 4-turn patterns of the form:

  Turn1  Speaker A  – fluent, self-contained utterance (no internal pauses,
                       leading backchannels stripped from the start)
  Turn2  Speaker B  – substantive response to Turn1 (not a pure backchannel)
  Turn3  Speaker A  – overlaps with Turn2 (interruption attempt)
  Turn4  Speaker B  – recovery / response after the interruption

GPT then classifies:
  • turn1_complete / turn1_self_contained
  • turn2_interrupted  (was Turn2 genuinely cut off, not just handing over?)
  • interruption_style : "hard" (abrupt cut-off) | "soft" (Turn2 wraps up quickly)
  •   recovery_type      : correct | enrich | shorten | repeat |
                        shift_topic | ask_followup | answer_followup

Audio export:
  • One .wav per turn (speaker's own channel)
  • One stereo .wav of the full 4-turn window (both speakers)

Usage:
  python extract_interruptions.py --conv_dir <path>
  python extract_interruptions.py --gpt_only --conv_dir <path>
"""

import json
import os
import re
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

PAUSE_MIN               = 0.2   # min gap to count as a pause (s)
PAUSE_MAX               = 1.0   # max gap to count as a pause (s)
MIN_INTERRUPTION_OVERLAP = 0.3  # Turn3 must start this many seconds before Turn2 ends
MAX_T3_DURATION          = 10.0 # Turn3 must not exceed this duration — longer means full
                                 # floor takeover, not an interruption
BACKCHANNEL_MAX_WORDS   = 3     # turn with <= this many words is a backchannel candidate
BACKCHANNEL_MAX_DUR     = 2.0   # and duration <= this (s)
T1_MIN_WORDS            = 6     # Turn1 must have >= this many substantive words
T1_MIN_DUR              = 2.0   # Turn1 must last >= this many seconds
T2_MIN_WORDS            = 4     # Turn2 must have >= this many words
T2_MIN_PRE_WORDS        = 4     # Turn2 must have >= this many words BEFORE the overlap
T2_MIN_POST_WORDS       = 3     # Turn2 must have >= this many words AFTER T3 starts
T3_MIN_WORDS            = 5     # Turn3 must have >= this many words (rejects "Okay that's cool")
                                 # (ensures interruption hits a meaningful mid-flow moment,
                                 #  not just the very tail of T2)
T3_MIN_GAP_FROM_T1      = 2.0   # T3 must start at least this many seconds after T1 ends
                                 # (prevents T3 from being A just continuing their own turn)
AUDIO_SR                = 16000
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL      = "gpt-5.4-2026-03-05"


# ── Helpers (shared logic with extract_pause_candidates) ───────────────────────

def is_backchannel_turn(row) -> bool:
    utt = str(row.get('utterance', '')).strip()
    if utt.endswith('?'):
        return False
    dur = float(row['stop']) - float(row['start'])
    if dur > BACKCHANNEL_MAX_DUR:
        return False
    return int(row.get('n_words', 999)) <= BACKCHANNEL_MAX_WORDS


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
    """Return words belonging to this turn's time window."""
    t_start = float(row['start'])
    t_stop  = float(row['stop'])
    words = [w for w in all_words if t_start <= w['start'] and w['end'] <= t_stop]
    return words if words else None


def trim_short_sentences(words: list[dict], min_words: int = 4) -> tuple | None:
    """
    Strip leading AND trailing sentences with fewer than min_words.
    Returns (trimmed_words, start, end) or None if nothing substantive remains.
    """
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

    s = 0
    while s < len(sentences) and len(sentences[s]) < min_words:
        s += 1
    e = len(sentences) - 1
    while e >= s and len(sentences[e]) < min_words:
        e -= 1
    if s > e:
        return None

    trimmed = [w for sent in sentences[s:e + 1] for w in sent]
    return trimmed, trimmed[0]['start'], trimmed[-1]['end']


def has_internal_pause(words: list[dict]) -> bool:
    """True if any consecutive word gap is in [PAUSE_MIN, PAUSE_MAX]."""
    return any(
        PAUSE_MIN <= words[k + 1]['start'] - words[k]['end'] <= PAUSE_MAX
        for k in range(len(words) - 1)
    )


def words_to_text(words: list[dict]) -> str:
    return ' '.join(w['word'] for w in words)


def words_to_timed(words: list[dict]) -> str:
    """Compact repr: word(start-end) for each word."""
    return ' '.join(f'{w["word"]}({w["start"]:.2f}-{w["end"]:.2f}s)' for w in words)


# ── Step 1: Timing Filter ──────────────────────────────────────────────────────

def find_interruption_patterns(cliff: pd.DataFrame,
                                speaker_words: dict[str, list]) -> list[dict]:
    """
    Scan cliffhanger turns for the 4-turn interruption pattern:
      T1(A) → T2(B, substantive) → T3(A, overlaps T2) → T4(B)

    Constraints:
      T1: no internal pauses, leading backchannel content stripped, substantive
      T2: not a pure backchannel, substantive
      T3: starts before T2 ends by >= MIN_INTERRUPTION_OVERLAP
      T4: any response from B
    """
    turns    = cliff.reset_index(drop=True)
    conv_id  = cliff.attrs.get('conv_id', '')
    candidates = []
    seen: set = set()

    for i in range(len(turns) - 3):
        t1_row = turns.iloc[i]
        t2_row = turns.iloc[i + 1]
        t3_row = turns.iloc[i + 2]
        t4_row = turns.iloc[i + 3]

        spk_a = t1_row['speaker']
        spk_b = t2_row['speaker']

        # ── Enforce alternating A-B-A-B pattern ───────────────────────────────
        if spk_a == spk_b:
            continue
        if t3_row['speaker'] != spk_a:
            continue
        if t4_row['speaker'] != spk_b:
            continue

        # ── Turn2 must not be a pure backchannel or a question ───────────────
        if is_backchannel_turn(t2_row):
            continue
        if int(t2_row.get('n_words', 0)) < T2_MIN_WORDS:
            continue
        if str(t2_row.get('utterance', '')).strip().endswith('?'):
            continue

        # ── Turn3 must genuinely overlap Turn2 ────────────────────────────────
        t2_end   = float(t2_row['stop'])
        t3_start = float(t3_row['start'])
        t1_end   = float(t1_row['stop'])
        overlap  = t2_end - t3_start
        if overlap < MIN_INTERRUPTION_OVERLAP:
            continue

        # ── T3 must start well after T1 ends (not just A continuing their turn)
        if t3_start < t1_end + T3_MIN_GAP_FROM_T1:
            continue

        # ── Turn1 must not itself overlap with the previous turn ──────────────
        if i > 0:
            prev_end = float(turns.iloc[i - 1]['stop'])
            if float(t1_row['start']) < prev_end - 0.1:
                continue

        # ── Turn2 must not overlap with Turn1 (T2 interrupting T1 is wrong pattern)
        t2_start = float(t2_row['start'])
        if t2_start < t1_end - 0.1:
            continue

        # ── Turn1: trim both leading and trailing short sentences ─────────────
        raw_t1 = get_turn_words(t1_row, speaker_words.get(spk_a, []))
        if not raw_t1:
            continue
        result = trim_short_sentences(raw_t1)
        if result is None:
            continue
        t1_words, _, _ = result

        if len(t1_words) < T1_MIN_WORDS:
            continue
        t1_dur = t1_words[-1]['end'] - t1_words[0]['start']
        if t1_dur < T1_MIN_DUR:
            continue

        # ── Turn1 must end with sentence-final punctuation (truly complete) ───
        if not t1_words[-1]['sent_end']:
            continue

        # ── Turn1 must have no internal pauses (fluent, uninterrupted speech) ─
        if has_internal_pause(t1_words):
            continue

        # ── Turn3: check duration, get words, check minimum length ───────────
        t3_end      = float(t3_row['stop'])
        t3_duration = t3_end - t3_start
        if t3_duration > MAX_T3_DURATION:
            continue  # Full floor takeover, not a brief interruption
        t3_words = get_turn_words(t3_row, speaker_words.get(spk_a, []))
        if not t3_words:
            continue
        if len(t3_words) < T3_MIN_WORDS:
            continue

        # ── Turn2: cap at t3_end — only words B spoke while T3 was active ────
        # Words B spoke after t3_end are discarded (diarization artifact where
        # B's CSV turn extends past T3 because B was already responding to T3).
        t2_all_words = get_turn_words(t2_row, speaker_words.get(spk_b, []))
        if not t2_all_words:
            continue
        t2_words = [w for w in t2_all_words if w['start'] < t3_end]
        if not t2_words:
            continue

        # ── Turn4: must start after T3 ends (clean fresh B response) ─────────
        if float(t4_row['start']) <= t3_end:
            continue
        # ── Turn4: must be a meaningful response (not a pure backchannel) ─────
        if is_backchannel_turn(t4_row):
            continue
        t4_words = get_turn_words(t4_row, speaker_words.get(spk_b, []))
        if not t4_words:
            continue

        # ── De-duplicate (same T1 start should only appear once) ──────────────
        key = (conv_id, spk_a, round(float(t1_row['start']), 2))
        if key in seen:
            continue
        seen.add(key)

        # Words spoken by T2 BEFORE T3 starts (pre-overlap) and DURING T3 (post-overlap)
        # Both sets are within the capped t2_words (start < t3_end) by construction.
        t2_pre_words  = [w for w in t2_words if w['end'] <= t3_start]
        t2_post_words = [w for w in t2_words if w['start'] >= t3_start]

        # ── T2 must have spoken enough words before the overlap ───────────────
        if len(t2_pre_words) < T2_MIN_PRE_WORDS:
            continue

        # ── T2 must still have words when T3 starts ───────────────────────────
        if not t2_post_words:
            continue

        # ── T2 must also have enough words during T3 (mid-flow interruption) ──
        if len(t2_post_words) < T2_MIN_POST_WORDS:
            continue

        candidates.append({
            'type':     'interruption',
            'conv_id':  conv_id,
            'spk_a':    spk_a,
            'spk_b':    spk_b,

            # Turn IDs
            't1_turn_id': int(t1_row['turn_id']),
            't2_turn_id': int(t2_row['turn_id']),
            't3_turn_id': int(t3_row['turn_id']),
            't4_turn_id': int(t4_row['turn_id']),

            # Timing
            't1_start': round(t1_words[0]['start'], 3),
            't1_end':   round(t1_words[-1]['end'], 3),
            't2_start': round(float(t2_row['start']), 3),
            # t2_end is the last capped word's end (≤ t3_end), not the CSV row stop
            't2_end':   round(t2_words[-1]['end'], 3),
            't3_start': round(t3_start, 3),
            't3_end':   round(t3_end, 3),
            't4_start': round(float(t4_row['start']), 3),
            't4_end':   round(float(t4_row['stop']), 3),
            'overlap_duration':         round(overlap, 3),
            # Yield latency: initially t2_end - t3_start; refined by GPT onset/yield times
            'yield_latency':            round(t2_words[-1]['end'] - t3_start, 3),
            'interruption_onset_time':  None,   # filled by GPT
            'interruption_onset_word':  None,   # filled by GPT
            'turn2_yield_word':         None,   # filled by GPT
            'turn2_yield_time':         None,   # filled by GPT

            # Transcripts (T2 is capped at t3_end — only words while T3 was active)
            't1_text':          words_to_text(t1_words),
            't2_text':          words_to_text(t2_words),
            't2_pre_overlap':   words_to_text(t2_pre_words),
            't2_post_overlap':  words_to_text(t2_post_words),
            't3_text':          words_to_text(t3_words),
            't4_text':          words_to_text(t4_words),

            # Word-level timestamps (for GPT context)
            't1_timed':         words_to_timed(t1_words),
            't2_pre_timed':     words_to_timed(t2_pre_words),
            't2_post_timed':    words_to_timed(t2_post_words),
            't3_timed':         words_to_timed(t3_words),
            't4_timed':         words_to_timed(t4_words),

            # GPT fields (filled in Step 2)
            'gpt_valid':               None,
            'turn1_complete':          None,
            'turn1_self_contained':    None,
            'turn2_responds':          None,
            'turn2_interrupted':       None,
            'turn4_meaningful':        None,
            'interruption_style':    None,  # "hard" | "soft"
            'interruption_type':     None,  # correct|enrich|shorten|repeat|
                                            # shift_topic|ask_followup|answer_followup
            'interruption_onset_word': None,
            'gpt_reason':            None,
        })

    return candidates


# ── Step 2: GPT Classification ─────────────────────────────────────────────────

GPT_PROMPT = """\
You are a dialogue analysis expert evaluating a 4-turn conversational pattern \
for research on interruption and recovery in spoken dialogue.

Here are four consecutive turns from a conversation, with word-level timestamps \
to help you assess timing precisely:

Turn 1 (Speaker A):
"{t1}"
  Word timestamps: {t1_timed}

Turn 2 (Speaker B — response):
"{t2}"
  Words BEFORE Turn 3 started (at {t3_start:.2f}s): "{t2_pre}"
    Timed: {t2_pre_timed}
  Words AFTER Turn 3 started (spoken during overlap): "{t2_post}"
    Timed: {t2_post_timed}

Turn 3 (Speaker A — started at {t3_start:.2f}s, {overlap:.2f}s before Turn 2 ended):
"{t3}"
  Word timestamps: {t3_timed}

Turn 4 (Speaker B — recovery/response):
"{t4}"
  Word timestamps: {t4_timed}

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
   - Accept only if the content introduces a complete, interpretable thought on its own.

── TURN 2 (Speaker B) ──────────────────────────────────────────────────────────

3. TURN2_RESPONDS: Is Turn 2 a genuine, direct declarative response to Turn 1 — \
NOT a topic shift, NOT an unrelated remark, and NOT a question? If Turn 2 is a \
question directed back at Speaker A (even a follow-up question), answer false.

4. TURN2_INTERRUPTED: Was Turn 2 genuinely cut off mid-thought by Turn 3 — i.e. \
Speaker B clearly had more to say when Turn 3 started? Use the word timestamps \
to judge: if Speaker B's last words before the overlap show an incomplete thought \
and Speaker B continued speaking after Turn 3 started, it was interrupted. Answer \
false if Turn 2 had already reached a natural completion point and Speaker A just \
took the next turn with a slight overlap (normal turn-taking).

4b. TURN2_YIELD_WORD: Identify the last SUBSTANTIVE word Speaker B said in Turn 2 \
— i.e. the actual yield point where B gave up the floor. Exclude trailing fillers \
("um", "uh", "so", "like", "I mean") at the very end. This is symmetric to \
INTERRUPTION_ONSET: just as T3 may begin with backchannels, T2 may end with fillers \
after the real content stops. Report as turn2_yield_word (word text) and \
turn2_yield_time (its END timestamp).

── TURN 3 (Speaker A — interruption) ───────────────────────────────────────────

5. INTERRUPTION_ONSET: Look at Turn 3's word timestamps. Turn 3 may begin with \
overlapping backchannels ("Yeah", "Uh huh", "Right", "Okay") before Speaker A \
launches into substantive floor-taking content. Identify the FIRST WORD of Turn 3 \
that marks the genuine start of the interruption — where Speaker A begins their \
real content. Report its word text and its start timestamp from the word timestamps \
provided. If Turn 3 begins with substantive content immediately, report the first word.

6. INTERRUPTION_STYLE: Classify how Turn 3 interrupts Turn 2. First verify all \
three conditions — if any fail, set interruption_style to "none":
   (a) FLOOR-TAKING INTENT: Turn 3 (at the onset identified above) must show clear \
intent to take the floor by contributing SUBSTANTIVE content — starting a new \
thought, asking a question, making a correction, or adding new information. \
The following do NOT qualify, even if more than one word:
     • Pure backchannels: "Uh huh", "Yeah", "Mm hmm", "Right", "Okay"
     • Short approval/reaction phrases: "That's cool", "Nice", "Interesting", \
"Oh wow", "Good for you", "That makes sense", "Okay cool"
     • Wrap-up signals that only acknowledge without adding: "Okay that's cool", \
"Yeah that's great", "Sounds good"
   Turn 3 must introduce something genuinely new — a question, a correction, a \
counter-point, or a substantive statement — not just signal that A has heard enough.
   (b) COMPLETENESS: Turn 3 must be a complete, well-formed utterance. A fragment \
or a thought that trails off without resolution does not qualify.
   (c) MID-FLOW ONSET: The interruption onset must land during a genuinely \
mid-flow moment of Turn 2 — not at the very start (Speaker B had barely begun) \
and not at the very tail (Speaker B was already wrapping up a completed thought). \
Use the word timestamps: check that there are meaningful words both BEFORE and \
AFTER the onset in Turn 2.

   If all three conditions pass, classify:
   - "hard": Speaker B stopped abruptly — the capped T2 content ends without \
reaching a syntactic boundary. Signs of "hard": sentence trails off mid-clause, \
ends with a dangling conjunction ("and", "but", "so", "because"), ends with a \
filler/disfluency ("um", "uh", "like", "you know"), or ends with an incomplete \
predicate that has no object/complement. Use "hard" when B clearly did NOT finish \
their thought before yielding.
   - "soft": Speaker B noticed Turn 3's signal and voluntarily wrapped up their \
thought before yielding — the capped T2 ends at a natural clause or sentence \
boundary (subject + verb + complement all present, no dangling filler). B completed \
the thought faster than they otherwise would have, but the ending is syntactically \
self-contained. Use "soft" only when B's final words form a grammatically complete \
clause, not just a trail-off.
   - "none": Any of the three conditions (a/b/c) above failed.

── TURN 4 (Speaker B — recovery) ───────────────────────────────────────────────

7. TURN4_MEANINGFUL: Is Turn 4 a meaningful, substantive engagement with what \
Turn 3 said — NOT a pure backchannel ("Uh huh", "Yeah", "Mm hmm", brief \
affirmations)? Turn 4 must show that Speaker B actually responded to the content \
of the interruption, not just acknowledged it.

8. RECOVERY_TYPE: Based on Turn 3's content and Turn 4's response, classify the \
type of recovery Speaker B performs:
   - "correct"         – A interrupts to correct an error in what B said → B \
acknowledges or corrects themselves in Turn 4.
   - "enrich"          – A explicitly asks B to elaborate on their current point \
(e.g. "tell me more", "what do you mean exactly") — NOT a new topic → B expands \
on the same point in Turn 4.
   - "shorten"         – A signals they've heard enough or want to move on \
(e.g. "yeah yeah I get it", "okay okay") → B wraps up more briefly than intended.
   - "repeat"          – A asks B to repeat or clarify something unclear \
(e.g. "sorry what?", "say that again") → B repeats or rephrases in Turn 4.
   - "shift_topic"     – A steers to a new subject unrelated to what B was saying \
→ B follows the new topic in Turn 4.
   - "ask_followup"    – B (after being interrupted) asks A a follow-up question in \
Turn 4 — B is the one asking.
   - "answer_followup" – A's Turn 3 contained a direct question for B → B answers \
it in Turn 4.

── VALIDITY ────────────────────────────────────────────────────────────────────

9. IS_VALID: Set to true ONLY if ALL of the following hold:
   turn1_complete=true, turn1_self_contained=true, turn2_responds=true, \
turn2_interrupted=true, interruption_style in ("hard","soft"), turn4_meaningful=true.

Respond ONLY with JSON (no markdown):
{{
  "turn1_complete": true/false,
  "turn1_self_contained": true/false,
  "turn2_responds": true/false,
  "turn2_interrupted": true/false,
  "turn2_yield_word": "<last substantive word of Turn 2>",
  "turn2_yield_time": <end timestamp of that word in seconds>,
  "interruption_onset_word": "<first substantive word of Turn 3>",
  "interruption_onset_time": <start timestamp of that word in seconds>,
  "interruption_style": "hard"|"soft"|"none",
  "turn4_meaningful": true/false,
  "recovery_type": "correct"|"enrich"|"shorten"|"repeat"|"shift_topic"|"ask_followup"|"answer_followup",
  "is_valid": true/false,
  "reason": "one sentence summary"
}}"""


class QuotaExceededError(RuntimeError):
    pass


def gpt_classify(candidates: list[dict], client: OpenAI,
                 retry_nulls_only: bool = False) -> list[dict]:
    pending = [c for c in candidates
               if not retry_nulls_only or c.get('gpt_valid') is None]
    print(f"  Sending {len(pending)} candidates to GPT ({GPT_MODEL})...")
    for i, cand in enumerate(pending):
        msg = GPT_PROMPT.format(
            t1=cand['t1_text'],
            t1_timed=cand.get('t1_timed', ''),
            t2=cand['t2_text'],
            t2_pre=cand['t2_pre_overlap'] or '(nothing before overlap)',
            t2_pre_timed=cand.get('t2_pre_timed', '(none)'),
            t2_post=cand['t2_post_overlap'] or '(none)',
            t2_post_timed=cand.get('t2_post_timed', '(none)'),
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
            cand['turn2_responds']    = result.get('turn2_responds')
            cand['turn2_interrupted'] = result.get('turn2_interrupted')
            cand['turn4_meaningful']  = result.get('turn4_meaningful')
            cand['interruption_style']   = result.get('interruption_style')
            cand['interruption_type']    = result.get('recovery_type')
            cand['gpt_reason']           = result.get('reason', '')
            # Interruption onset (first substantive word of T3)
            onset_word = result.get('interruption_onset_word')
            onset_time = result.get('interruption_onset_time')
            cand['interruption_onset_word'] = onset_word
            cand['interruption_onset_time'] = onset_time
            # T2 yield point (last substantive word of T2)
            yield_word = result.get('turn2_yield_word')
            yield_time = result.get('turn2_yield_time')
            cand['turn2_yield_word'] = yield_word
            cand['turn2_yield_time'] = yield_time
            # Recompute yield latency: from interruption onset → T2 yield point
            if onset_time is not None and yield_time is not None:
                try:
                    cand['yield_latency'] = round(float(yield_time) - float(onset_time), 3)
                except (TypeError, ValueError):
                    pass
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
    return candidates


# ── Step 3: Audio Export ───────────────────────────────────────────────────────

def export_clip(audio: np.ndarray, sr: int,
                start_s: float, end_s: float,
                out_path: Path, ch_idx: int) -> None:
    """Save a single-channel clip (speaker's own channel)."""
    s = max(0, int(start_s * sr))
    e = min(audio.shape[-1], int(end_s * sr))
    clip = audio[ch_idx, s:e] if audio.ndim == 2 else audio[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr)


def export_stereo_clip(audio: np.ndarray, sr: int,
                       start_s: float, end_s: float,
                       out_path: Path) -> None:
    """Save a stereo clip covering the full 4-turn window (both speakers)."""
    s = max(0, int(start_s * sr))
    e = min(audio.shape[-1], int(end_s * sr))
    if audio.ndim == 2:
        clip = audio[:, s:e].T  # shape (samples, channels) for soundfile
    else:
        clip = audio[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr)


def export_interruption_candidates(candidates: list[dict],
                                   audio: np.ndarray, sr: int,
                                   audio_dir: Path,
                                   spk_to_ch: dict[str, int],
                                   results_dir: Path | None = None) -> None:
    """Export audio for GPT-validated candidates.

    Files are written into the per-conversation audio_dir.  When results_dir is
    provided the same files are also copied there, giving a global view:
        results_dir/<style>/<type>/<conv_id>_t<n>_{t1,t2,t3,t4,full_stereo}.wav
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

        clips = [
            (cand['t1_start'], cand['t1_end'],  f"{base}_t1.wav",          ch_a,  False),
            (cand['t2_start'], cand['t2_end'],  f"{base}_t2.wav",          ch_b,  False),
            (cand['t3_start'], cand['t3_end'],  f"{base}_t3.wav",          ch_a,  False),
            (cand['t4_start'], cand['t4_end'],  f"{base}_t4.wav",          ch_b,  False),
            (cand['t1_start'], cand['t4_end'],  f"{base}_full_stereo.wav", None,  True),
        ]

        for t_start, t_end, fname, ch, stereo in clips:
            dest = out_dir / fname
            if stereo:
                export_stereo_clip(audio, sr, t_start, t_end, dest)
            else:
                export_clip(audio, sr, t_start, t_end, dest, ch)

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
         results_dir: Path | None = None) -> None:
    conv_id    = conv_dir.name
    output_dir = conv_dir / "interruption_candidates"
    f1_dir     = output_dir / "filter1_timing"
    f2_dir     = output_dir / "filter2_gpt"
    audio_dir  = output_dir / "audio"
    audio_path = conv_dir / "processed" / f"{conv_id}.mp3"
    chmap_path = conv_dir / "processed" / "channel_map.json"
    cliff_path = conv_dir / "transcription" / "transcript_cliffhanger.csv"
    json_path  = conv_dir / "transcription" / "transcribe_output.json"

    # ── Load shared data (always needed) ──────────────────────────────────────
    cliff      = pd.read_csv(cliff_path)
    trans_json = json.load(open(json_path))
    chmap      = json.load(open(chmap_path))
    cliff.attrs['conv_id'] = conv_id

    channels: list = trans_json['results']['channel_labels']['channels']
    ch_to_speaker: dict[str, str] = {}
    for ch_idx, ch in enumerate(channels):
        first_word_start = float(ch['items'][0]['start_time'])
        match = cliff[abs(cliff['start'] - first_word_start) < 0.1]
        ch_to_speaker[f"ch_{ch_idx}"] = (match.iloc[0]['speaker'] if not match.empty
                                          else list(chmap.values())[ch_idx])
    spk_to_ch = {spk: int(key.split('_')[1]) for key, spk in ch_to_speaker.items()}

    if gpt_only:
        # ── GPT-only: reload timing results, preserve existing GPT decisions ──
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
        # ── Full mode: wipe and reprocess ────────────────────────────────────
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"Deleted previous results in {output_dir.name}/")
        output_dir.mkdir(exist_ok=True)

        print("Building per-speaker word list...")
        speaker_words = build_speaker_words(trans_json, ch_to_speaker)
        for spk, wds in speaker_words.items():
            print(f"  {spk}: {len(wds)} words")

        print("\n── Step 1: Timing filter ──")
        candidates = find_interruption_patterns(cliff, speaker_words)
        print(f"  Candidates found: {len(candidates)}")

        print("\nSaving Filter 1 results...")
        save_results(candidates, f1_dir / "all.json", "timing")

    # ── Step 2: GPT classification ─────────────────────────────────────────────
    print(f"\n── Step 2: GPT classification ({GPT_MODEL}) ──")
    client = OpenAI(api_key=OPENAI_API_KEY)
    candidates = gpt_classify(candidates, client, retry_nulls_only=gpt_only)

    valid = [c for c in candidates if c.get('gpt_valid')]
    print(f"\n  Valid after GPT: {len(valid)}/{len(candidates)}")

    # Breakdown by style and type
    for style in ('hard', 'soft'):
        by_type = {}
        for c in valid:
            if c.get('interruption_style') == style:
                t = c.get('interruption_type', 'unknown')
                by_type[t] = by_type.get(t, 0) + 1
        if by_type:
            print(f"    {style}: " + "  ".join(f"{t}={n}" for t, n in by_type.items()))

    print("\nSaving Filter 2 results (GPT)...")
    save_results(candidates, f2_dir / "all_with_gpt_decisions.json", "all+decisions")
    save_results(valid,      f2_dir / "valid.json",                  "valid")

    # Per style/type breakdown files
    for style in ('hard', 'soft'):
        subset = [c for c in valid if c.get('interruption_style') == style]
        if subset:
            save_results(subset, f2_dir / f"{style}.json", style)

    # ── Step 3: Audio export ───────────────────────────────────────────────────
    if valid:
        print(f"\n── Step 3: Audio export ──")
        print(f"Loading audio from {audio_path.name}...")
        audio, sr = librosa.load(str(audio_path), sr=AUDIO_SR, mono=False)
        print(f"  Audio: {audio.shape}, sr={sr}")
        export_interruption_candidates(candidates, audio, sr, audio_dir, spk_to_ch,
                                       results_dir=results_dir)
        print(f"  Exported {len(valid)} × 5 clips → {audio_dir.name}/")
        if results_dir:
            print(f"  Mirrored to global results → {results_dir}")
    else:
        print("\n  No valid candidates — skipping audio export.")

    print("\n── Summary ──")
    print(f"  Output: {output_dir.name}/")
    print(f"  filter1_timing/  ← timing filter results")
    print(f"  filter2_gpt/     ← GPT classifications")
    print(f"  audio/           ← <style>/<type>/<conv>_t1/t2/t3/t4/full_stereo.wav")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract interruption & recovery patterns.")
    parser.add_argument("--conv_dir",    type=Path, default=DEFAULT_CONV_DIR)
    parser.add_argument("--gpt_only",    action="store_true",
                        help="Skip timing filter; re-run GPT on existing filter1 results")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Global results directory — audio is mirrored here under "
                             "<style>/<type>/; JSON is aggregated by the batch script")
    args = parser.parse_args()
    try:
        main(args.conv_dir, gpt_only=args.gpt_only, results_dir=args.results_dir)
    except QuotaExceededError as e:
        print(f"\n[QUOTA EXCEEDED] {e}", flush=True)
        raise SystemExit(2)
