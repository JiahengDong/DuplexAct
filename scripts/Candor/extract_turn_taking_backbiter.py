#!/usr/bin/env python3
"""
CANDOR Turn-Taking Extractor — Backbiter Edition

Extracts simple 2-turn turn-taking pairs (Speaker A → Speaker B) for benchmarking
turn-taking behaviour. Turn 1 is used alone as the user-side input to a model, so
it must be a complete, self-contained utterance. Turn 2 is only used to categorise
the kind of turn-taking that follows; it may be incomplete but must be a genuine
response to Turn 1.

Everything that can be decided from timestamps/transcripts is done with rules;
anything requiring language/context understanding is delegated to GPT.

2-turn pattern:
  Turn1  Speaker A  – complete, self-contained sentence, NO internal pause
  Turn2  Speaker B  – response to T1 (> 3 words), pauses allowed, may be incomplete

Backchannels are already absorbed by the Backbiter transcript, so any surviving
row is substantive speech.

Timing categories (decided purely from timestamps):
  - take_turn_immediately : 0 ≤ gap ≤ 200 ms between T1 end and T2 start
  - take_turn_with_latency: gap > 200 ms
  - take_turn_with_overlap: T2 starts before T1 ends (T1 is complete / about to
                            finish and B steps in with a > 3-word response — not
                            an interruption)

Usage:
  python extract_turn_taking_backbiter.py --conv_dir <path>
  python extract_turn_taking_backbiter.py --gpt_only --conv_dir <path>
"""

import json
import os
import argparse
import sys
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

PAUSE_MIN               = 0.2   # min gap to count as a pause within a turn (s)
MIN_WORDS_FILTER        = 3     # pre-filter: drop any turn with fewer words
MIN_SUBSTANTIVE_WORDS   = 4     # T1 and T2 minimum content words (> MAX_BC_WORDS = 3)
IMMEDIATE_MAX_GAP       = 0.2   # ≤ this gap = "immediate" (200 ms)
MAX_LATENCY_GAP         = 3.0   # gaps larger than this are not turn-taking responses
MAX_OVERLAP             = 1.5   # max T1/T2 overlap to still count as turn-taking
AUDIO_SR                = 16000

QUOTA_EXIT_CODE     = 2
TARGET_PER_CATEGORY = 500   # global cap per timing category

CATEGORIES = ("take_turn_immediately", "take_turn_with_latency", "take_turn_with_overlap")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL      = "gpt-5.4-2026-03-05"

# ── Dataset Context ───────────────────────────────────────────────────────────

_CANDOR_CTX = (
    "This dialogue is from the CANDOR corpus (Conversation: A Naturalistic Dataset "
    "Of Recorded conversations), a collection of spontaneous dyadic conversations "
    "between two strangers conducted over video call. Speech is transcribed at the "
    "word level with timestamps. Turns may contain natural disfluencies and "
    "false starts. Punctuation is present in the transcription."
)

# Per-category guidance: given the timing-assigned category, GPT judges whether
# THIS pair is a clean, content-appropriate example of that category (it does NOT
# recategorise). Keyed by category → human-readable criterion injected into the prompt.

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
        "without needing to revise its response strategy. "
        "\n\n"
        "Examples that fit: a scheduling request whose date or time is supplied at "
        "the end; a request whose recipient or target item is supplied at the end; "
        "a question whose answer format is already clear but whose final option or "
        "entity is only given in the final phrase. "
        "\n\n"
        "Do NOT keep the pair if: (1) the final content changes, reverses, or "
        "substantially constrains what the response should be; (2) the response "
        "target remains genuinely unclear until the end; (3) the turn is merely "
        "simple and received a fast response by chance, without a meaningful final "
        "slot-completion structure; (4) Speaker B's response is a backchannel, "
        "continuation, or non-substantive acknowledgement. "
        "\n\n"
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
        "Speaker A finishes. "
        "\n\n"
        "The late information must do more than fill a predictable slot: it should "
        "determine, revise, invalidate, or materially redirect the response plan "
        "itself. Typical qualifying patterns include a short late correction, "
        "negation, exception, condition, contrast, qualification, or decision cue "
        "that changes what Speaker B should say or do. The final update should be "
        "part of the same topic and interactional goal, rather than a long appended "
        "explanation or a switch to a completely different topic. "
        "\n\n"
        "Examples that fit: 'Could you review the introduction—actually, the full "
        "draft?'; 'I can meet tomorrow, but only after five'; 'We should book "
        "Monday—unless the conference starts Sunday'; 'The answer is yes, provided "
        "it stays under five hundred.' In each case, the final short phrase changes "
        "the scope, constraint, feasibility, or response strategy too late for a "
        "fully prepared immediate response. "
        "\n\n"
        "Do NOT keep the pair if: (1) the revising or decision-changing content "
        "appears early enough that Speaker B could reasonably complete planning "
        "before Turn 1 ends; (2) the delay could be explained only by speaker style, "
        "hesitation, distraction, social delicacy, or generic response difficulty; "
        "(3) the final content merely provides a date, person, place, or other slot "
        "that fits an already prepared response frame; (4) the final content is only "
        "closing, repetition, or hedging; (5) Speaker B's response does not actually "
        "depend on or react to the late information. "
        "\n\n"
        "Key test: does a short, late final update alter the response that Speaker B "
        "would otherwise have given, while leaving too little time to reliably revise "
        "that response before Turn 1 ends?"
    ),

    "take_turn_with_overlap": (
        "This pair was identified from timing as TAKE-TURN-WITH-OVERLAP: Speaker B "
        "began before Speaker A had acoustically finished. Keep it only if the "
        "overlap is a safe content-based early entry rather than an interruption. "
        "Before Speaker B begins, Turn 1 must already have supplied all information "
        "needed for a correct and appropriate substantive response, including the "
        "main goal, target entity or entities, relevant constraints, polarity, and "
        "decision-relevant content. The remaining Speaker A speech must be "
        "non-response-changing closing material, such as a hedge, repetition, "
        "trailing phrase, discourse-closing expression, or prosodic wrap-up. "
        "\n\n"
        "Examples that fit: 'Thursday afternoon works best for me, I think—' "
        "followed by 'Yeah, Thursday works for me too'; 'So the only realistic "
        "option is Friday, basically—' followed by an aligned response during the "
        "final closing phrase. "
        "\n\n"
        "Do NOT keep the pair if: (1) Speaker A is still providing a missing entity, "
        "option, condition, polarity marker, contrast, correction, or other "
        "response-changing information; (2) the overlap occurs before the main "
        "question, request, decision, or stance is complete; (3) Speaker B's reply "
        "would become incorrect or materially different after hearing the remaining "
        "speech; (4) the overlap is a backchannel, collaborative completion, repair, "
        "floor seizure, or interruption rather than a genuine substantive response; "
        "(5) Speaker A continues with a new proposition after Speaker B enters. "
        "\n\n"
        "Key test: if the remaining suffix of Turn 1 were removed, would Speaker B's "
        "response still be equally correct and appropriate? If yes, the overlap may "
        "be a valid candidate. If no, reject it."
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

def find_turn_taking_patterns(back: pd.DataFrame,
                              speaker_words: dict[str, list]
                              ) -> tuple[list[dict], dict]:
    """
    Scan consecutive Backbiter rows for a 2-turn turn-taking pair:
      T1(A) → T2(B), A != B.

    Rejection order (consistent with CORAAL):
      1. same speaker
      2. T2 < MIN_SUBSTANTIVE_WORDS (cheap gate from row metadata)
      3. T1 empty (no word-level timestamps found)
      4. T1 < MIN_SUBSTANTIVE_WORDS
      5. T1 no sentence-final punctuation
      6. T1 has internal pause (word gap >= PAUSE_MIN)
      7. T2 empty or < MIN_SUBSTANTIVE_WORDS (definitive check from timestamps)
      8. gap/overlap out of range
      9. duplicate

    Returns (candidates, timing_stats).
    """
    turns   = back.reset_index(drop=True)
    conv_id = back.attrs.get('conv_id', '')

    stats = dict(
        pairs_checked         = 0,
        rej_same_speaker      = 0,
        rej_t2_too_short      = 0,
        rej_t1_too_short      = 0,
        rej_t1_incomplete     = 0,
        rej_t1_has_pause      = 0,
        rej_t2_words_short    = 0,
        rej_gap_too_large     = 0,
        rej_overlap_too_large = 0,
        rej_duplicate         = 0,
        passed                = 0,
        by_category           = {c: 0 for c in CATEGORIES},
    )

    candidates: list[dict] = []
    seen: set = set()

    for i in range(len(turns) - 1):
        t1_row = turns.iloc[i]
        t2_row = turns.iloc[i + 1]

        spk_a = t1_row['speaker']
        spk_b = t2_row['speaker']

        stats['pairs_checked'] += 1

        # ── 1. Same speaker ───────────────────────────────────────────────────
        if spk_a == spk_b:
            stats['rej_same_speaker'] += 1
            continue

        # ── 2. T2 cheap gate (row metadata) ──────────────────────────────────
        if int(t2_row.get('n_words', 0)) < MIN_SUBSTANTIVE_WORDS:
            stats['rej_t2_too_short'] += 1
            continue

        # ── 3. T1 words (needed for all subsequent T1 checks) ─────────────────
        t1_words = get_turn_words(t1_row, speaker_words.get(spk_a, []))

        # ── 4. T1 minimum word count (covers both zero-word and short turns) ──
        if not t1_words or len(t1_words) < MIN_SUBSTANTIVE_WORDS:
            stats['rej_t1_too_short'] += 1
            continue

        # ── 5. T1 must end with sentence-final punctuation ────────────────────
        if not t1_words[-1]['sent_end']:
            stats['rej_t1_incomplete'] += 1
            continue

        # ── 6. T1 must have no internal pause (word gap >= PAUSE_MIN) ─────────
        if any(
            t1_words[k + 1]['start'] - t1_words[k]['end'] >= PAUSE_MIN
            for k in range(len(t1_words) - 1)
        ):
            stats['rej_t1_has_pause'] += 1
            continue

        # ── 7. T2 definitive word count (from timestamps) ─────────────────────
        t2_words = get_turn_words(t2_row, speaker_words.get(spk_b, []))
        if not t2_words or len(t2_words) < MIN_SUBSTANTIVE_WORDS:
            stats['rej_t2_words_short'] += 1
            continue

        t1_start = t1_words[0]['start']
        t1_end   = t1_words[-1]['end']
        t2_start = t2_words[0]['start']
        t2_end   = t2_words[-1]['end']
        gap      = t2_start - t1_end

        # ── 8. Gap/overlap and category assignment ────────────────────────────
        if gap < 0:
            if -gap > MAX_OVERLAP:
                stats['rej_overlap_too_large'] += 1
                continue
            category = "take_turn_with_overlap"
        elif gap <= IMMEDIATE_MAX_GAP:
            category = "take_turn_immediately"
        elif gap <= MAX_LATENCY_GAP:
            category = "take_turn_with_latency"
        else:
            stats['rej_gap_too_large'] += 1
            continue

        # ── 9. De-duplicate ───────────────────────────────────────────────────
        key = (conv_id, spk_a, round(float(t1_row['start']), 2))
        if key in seen:
            stats['rej_duplicate'] += 1
            continue
        seen.add(key)

        stats['passed'] += 1
        stats['by_category'][category] += 1

        candidates.append({
            'type':     'turn_taking',
            'source':   'backbiter',
            'conv_id':  conv_id,
            'spk_a':    spk_a,
            'spk_b':    spk_b,

            't1_turn_id': int(t1_row['turn_id']),
            't2_turn_id': int(t2_row['turn_id']),

            't1_start': round(t1_start, 3),
            't1_end':   round(t1_end,   3),
            't2_start': round(t2_start, 3),
            't2_end':   round(t2_end,   3),
            'gap':      round(gap,       3),
            'category': category,

            't1_text':  words_to_text(t1_words),
            't2_text':  words_to_text(t2_words),
            't1_timed': words_to_timed(t1_words),
            't2_timed': words_to_timed(t2_words),

            'gpt_valid':            None,
            'turn1_complete':       None,
            'turn1_self_contained': None,
            'turn2_responds':       None,
            'category_appropriate': None,
            'gpt_reason':           None,
        })

    return candidates, stats


# ── Step 2: GPT Classification ─────────────────────────────────────────────────

GPT_PROMPT = """\
You are a dialogue analysis expert evaluating a 2-turn turn-taking pattern for \
research on conversational turn-taking.

""" + _CANDOR_CTX + """

Turn 1 will be used ALONE as the user-side input to a model, so it must stand on \
its own. Turn 2 is only used to confirm that a genuine response followed; it may \
be incomplete.

Here are the two consecutive turns, with word-level timestamps:

Turn 1 (Speaker A):
"{t1}"
  Word timestamps: {t1_timed}

Turn 2 (Speaker B — response):
"{t2}"
  Word timestamps: {t2_timed}

This pair's assigned turn-taking category (from timing — you do NOT change it) is: \
{category}

---
Evaluate each field in order. Complete ALL fields even if an earlier answer is \
false — just set is_valid to false at the end.

1. TURN1_COMPLETE: Is Turn 1 a complete, well-formed utterance (not a fragment)?

2. TURN1_SELF_CONTAINED: Can Turn 1 be fully understood and responded to WITHOUT \
any prior conversational context? Be strict:
   - Reject if it contains references whose meaning is unclear without context \
(e.g. "that one", "it", "four or five").
   - Reject if it begins with a filler/acknowledgement then transitions into a \
topic that requires prior context to interpret.
   - Reject if it is clearly a direct answer to an unseen question.
   - Reject if it is a reaction or emotional response that only makes sense as a \
reply to something just said — e.g. "Are you serious?", "Really?", "No way!", \
"That's crazy!" — these have no interpretable meaning without the prior utterance.
   - Accept only if the content introduces a complete, interpretable thought that \
something could naturally respond to.

3. TURN2_RESPONDS: Is Turn 2 a genuine response aimed at Turn 1 — answering, \
reacting to, agreeing/disagreeing with, elaborating on, or asking a follow-up \
about Turn 1? Turn 2 may be incomplete, but it must be directed at Turn 1 and not \
an unrelated remark or a topic shift to something disconnected from Turn 1.

4. CATEGORY_APPROPRIATE: For the assigned category "{category}", judge ONLY from \
the content of Turn 1 (ignore how fast Speaker B actually replied — speaker speed \
varies between individuals) whether this pair is a clean, content-appropriate \
example of that category.

{category_guidance}

Answer true if Turn 1's content genuinely fits the assigned category "{category}", \
false otherwise (e.g. the timing was just speaker variance, not an ideal example).

5. IS_VALID: Set to true ONLY if turn1_complete=true, turn1_self_contained=true, \
turn2_responds=true, and category_appropriate=true.

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


def gpt_classify(candidates: list[dict], client: OpenAI,
                 retry_nulls_only: bool = False,
                 max_valid_per_category: int | None = None,
                 ) -> tuple[list[dict], dict]:
    """
    Send each candidate to GPT for validity + category-appropriateness check.
    Stops a category once max_valid_per_category is reached.
    Returns (candidates, gpt_stats).
    """
    pending = [c for c in candidates
               if not retry_nulls_only or c.get('gpt_valid') is None]
    total = len(pending)
    print(f"  Sending {total} candidates to GPT ({GPT_MODEL})"
          + (f" (cap={max_valid_per_category}/category)" if max_valid_per_category else "") + "...")

    cat_valid: dict[str, int] = {c: 0 for c in CATEGORIES}

    gpt_stats = dict(
        total_sent   = total,
        total_judged = 0,
        valid        = 0,
        rejected     = 0,
        criteria     = {},
        by_category  = {c: 0 for c in CATEGORIES},
    )

    for i, cand in enumerate(pending):
        cat = cand.get('category', 'unknown')

        # Skip if this category's budget is already met
        if max_valid_per_category is not None:
            if cat_valid.get(cat, 0) >= max_valid_per_category:
                cand['gpt_valid']  = False
                cand['gpt_reason'] = "skipped: category budget reached"
                if all(cat_valid.get(c, 0) >= max_valid_per_category for c in CATEGORIES):
                    print(f"    [{i+1}/{total}] all category budgets reached — stopping.")
                    break
                continue

        msg = GPT_PROMPT.format(
            t1                = cand['t1_text'],
            t1_timed          = cand.get('t1_timed', ''),
            t2                = cand['t2_text'],
            t2_timed          = cand.get('t2_timed', ''),
            category          = cat,
            category_guidance = CATEGORY_GUIDANCE.get(cat, ''),
        )
        try:
            resp = client.chat.completions.create(
                model           = GPT_MODEL,
                messages        = [{"role": "user", "content": msg}],
                temperature     = 0,
                response_format = {"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            cand['gpt_valid']            = result.get('is_valid', False)
            cand['turn1_complete']       = result.get('turn1_complete')
            cand['turn1_self_contained'] = result.get('turn1_self_contained')
            cand['turn2_responds']       = result.get('turn2_responds')
            cand['category_appropriate'] = result.get('category_appropriate')
            cand['gpt_reason']           = result.get('reason', '')

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
            continue

        gpt_stats['total_judged'] += 1

        if cand['gpt_valid']:
            gpt_stats['valid'] += 1
            cat_valid[cat] = cat_valid.get(cat, 0) + 1
            gpt_stats['by_category'][cat] = gpt_stats['by_category'].get(cat, 0) + 1
        else:
            gpt_stats['rejected'] += 1
            for key in ('turn1_complete', 'turn1_self_contained',
                        'turn2_responds', 'category_appropriate'):
                if cand.get(key) is False:
                    gpt_stats['criteria'][key] = gpt_stats['criteria'].get(key, 0) + 1

        status = '✓' if cand['gpt_valid'] else '✗'
        print(f"    [{i+1}/{total}] {status} [{cat[:4]}] — {cand['t1_text'][:70]}")

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


def export_turn_taking_candidates(candidates: list[dict],
                                  audio: np.ndarray, sr: int,
                                  audio_dir: Path,
                                  spk_to_ch: dict[str, int],
                                  speaker_words: dict[str, list],
                                  results_dir: Path | None = None) -> None:
    """Export T1 (benchmark input), T2, and full-stereo clips per valid candidate.

    Mono clips silence the other speaker's frames to remove mic bleed. T1 is the
    clean user-side input; T2 confirms the response.
    """
    import shutil
    for cand in candidates:
        if not cand.get('gpt_valid'):
            continue
        category = cand.get('category', 'unknown')
        out_dir  = audio_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)
        conv_id = cand['conv_id']
        base    = f"{conv_id}_t{cand['t1_turn_id']}"
        ch_a = spk_to_ch.get(cand['spk_a'], 0)
        ch_b = spk_to_ch.get(cand['spk_b'], 1)

        words_a = speaker_words.get(cand['spk_a'], [])
        words_b = speaker_words.get(cand['spk_b'], [])

        def b_words_in(t0, t1):
            return [w for w in words_b if w['start'] < t1 and w['end'] > t0]

        def a_words_in(t0, t1):
            return [w for w in words_a if w['start'] < t1 and w['end'] > t0]

        clips = [
            (cand['t1_start'], cand['t1_end'], f"{base}_t1.wav",          ch_a,  False, b_words_in),
            (cand['t2_start'], cand['t2_end'], f"{base}_t2.wav",          ch_b,  False, a_words_in),
            (cand['t1_start'], cand['t2_end'], f"{base}_full_stereo.wav", None,  True,  None),
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
         max_valid_per_category: int | None = None) -> None:
    conv_id    = conv_dir.name
    output_dir = conv_dir / "turn_taking_backbiter"
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

    timing_stats: dict = {}

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
                                  'turn2_responds', 'category_appropriate', 'gpt_reason'):
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

        before = len(back)
        back = back[back['n_words'] >= MIN_WORDS_FILTER].reset_index(drop=True)
        print(f"  Pre-filter: {before} → {len(back)} turns (removed {before - len(back)} "
              f"with < {MIN_WORDS_FILTER} words)")

        print("Building per-speaker word list...")
        for spk, wds in speaker_words.items():
            print(f"  {spk}: {len(wds)} words")

        print("\n── Step 1: Timing filter ──")
        candidates, timing_stats = find_turn_taking_patterns(back, speaker_words)
        by_cat = timing_stats['by_category']
        print(f"  Candidates: {len(candidates)}  " +
              "  ".join(f"{k}={v}" for k, v in by_cat.items() if v))
        print(f"TIMING_STATS:{json.dumps(timing_stats)}")
        for cat in CATEGORIES:
            print(f"  SEND_TO_LLM {cat}: {by_cat.get(cat, 0)}")
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

    gpt_stats: dict = {}
    try:
        candidates, gpt_stats = gpt_classify(
            candidates, client,
            retry_nulls_only=gpt_only,
            max_valid_per_category=max_valid_per_category,
        )
    except QuotaExceededError as e:
        print(f"  QUOTA EXCEEDED: {e}")
        save_results(candidates, f2_dir / "all_with_gpt_decisions.json", "all+decisions")
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
    sys.stdout.flush()

    print("\nSaving Filter 2 results (GPT)...")
    save_results(candidates, f2_dir / "all_with_gpt_decisions.json", "all+decisions")

    # ── Step 3: Audio export ───────────────────────────────────────────────────
    if valid:
        print(f"\n── Step 3: Audio export ──")
        print(f"Loading audio from {audio_path.name}...")
        audio, sr = librosa.load(str(audio_path), sr=AUDIO_SR, mono=False)
        print(f"  Audio: {audio.shape}, sr={sr}")
        export_turn_taking_candidates(candidates, audio, sr, audio_dir, spk_to_ch,
                                      speaker_words=speaker_words,
                                      results_dir=results_dir)
        print(f"  Exported {len(valid)} × 3 clips → {audio_dir.name}/")
        if results_dir:
            print(f"  Mirrored to global results → {results_dir}")
    else:
        print("\n  No valid candidates — skipping audio export.")

    # Save valid.json AFTER audio export so audio_dir is populated in each entry
    save_results(valid, f2_dir / "valid.json", "valid")

    print(f"\nVALID_JSON:{json.dumps(valid)}")
    print("\n── Summary ──")
    print(f"  Output: {output_dir.name}/")
    print(f"  filter1_timing/  ← timing filter results")
    print(f"  filter2_gpt/     ← GPT classifications")
    print(f"  audio/           ← <category>/<conv>_t1/t2/full_stereo.wav")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 2-turn turn-taking pairs using Backbiter transcript.")
    parser.add_argument("--conv_dir",              type=Path, default=DEFAULT_CONV_DIR)
    parser.add_argument("--gpt_only",              action="store_true",
                        help="Skip timing filter; re-run GPT on existing filter1 results")
    parser.add_argument("--no_gpt",                action="store_true",
                        help="Run timing filter only; skip GPT classification and audio export")
    parser.add_argument("--results_dir",           type=Path, default=None,
                        help="Global results directory — audio mirrored here")
    parser.add_argument("--max_valid_per_category", type=int, default=TARGET_PER_CATEGORY,
                        help="Stop GPT calls per category after this many valid candidates")
    args = parser.parse_args()
    main(args.conv_dir, gpt_only=args.gpt_only, no_gpt=args.no_gpt,
         results_dir=args.results_dir,
         max_valid_per_category=args.max_valid_per_category)
