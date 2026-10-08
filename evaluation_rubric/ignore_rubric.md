# Ignore — Level-2 Rubric

## Overview

| Field | Value |
|---|---|
| **Level-2 action** | Ignore |
| **Metric family** | SR — Silence Rate |
| **Timing variable** | None |
| **Judge required** | Yes — R2 requires LLM / human judge (condition-specific prompt) |

**Base success (B_i):** No model speech onset inside the ignore interval.  
**Reliability (R_i):** Mean of two items — responsive (auto), no_leakage (LLM).

---

## Level-3 Conditions

| Condition | Description |
|---|---|
| `background_speech` | Background speech plays during the user's addressed turn; model must not respond to it |
| `side_conversation` | User addresses another person mid-turn (non-directed speech); model must not respond to it |

**Data aliases** (synthesized dataset names → canonical):

| Dataset name | Canonical |
|---|---|
| `ignore_background_speech_during_user_speech` | `background_speech` |
| `ignore_other_directed_speech_during_user_speech` | `side_conversation` |

---

## Timestamps Required

| Key | Description |
|---|---|
| `t_other_start` | Onset of the background / non-directed speech |
| `t_other_end` | Offset of the background / non-directed speech |
| `t_user_end` | When the user's addressed turn ends (required for R1 responsive) |

---

## B_i — Base Success (Automatic)

**Definition:** B_i = 1 iff no model speech onset falls inside `[t_other_start, t_other_end + GRACE_S]`.

Any onset inside that window is treated as a spurious takeover triggered by the ignored speech.

**Failure tag:** `spurious_takeover:onset=<t>s`

---

## Reliability Items

### R1 — Responsive (Automatic)

**Definition:** responsive = 1 iff the model's first onset falls within the valid response window around the end of the user's addressed turn:

```
[t_user_end − RESPONSE_OVERLAP_MAX_S,  t_user_end + RESPONSE_LATENCY_MAX_S]
```

This checks that the model still responds to what the user actually said, despite the intervening ignored speech.

**Applicable to:** Both conditions.

---

### R2 — No Leakage (LLM / Human Judge)

**Definition:** no_leakage = 1 iff the model's response does not reference, echo, paraphrase, or incorporate content from the ignored speech.

Two condition-specific judge prompts are used:

**`background_speech` prompt:**  
*Does the model's response to the user's addressed speech avoid referencing, reacting to, or incorporating content from the background speech? Answer YES if the model's response is based entirely on the user's addressed utterance and contains no trace of the background speech content. Answer NO if the model's response echoes, paraphrases, or is visibly shaped by what was said in the background.*

**`side_conversation` prompt:**  
*Does the model's response to the user's addressed speech avoid referencing, reacting to, or incorporating content from the side conversation the user had with another person? Answer YES if the model responds only to the content the user directed at it, ignoring what was said to the other person. Answer NO if the model incorporates, references, or appears influenced by the content of the user's side conversation.*

| | |
|---|---|
| **Context required** | `transcript_user`, `transcript_model`, `transcript_context` |
| **Applicable to** | Both conditions |

---

## Tunable Intervals

All constants live at the top of `actions/ignore.py`.

| Constant | Default | Meaning |
|---|---|---|
| `GRACE_S` | 0.50 s | Grace period after `t_other_end`; onsets still counted as spurious takeover |
| `RESPONSE_OVERLAP_MAX_S` | 0.20 s | Max pre-emption before `t_user_end` for responsive check |
| `RESPONSE_LATENCY_MAX_S` | 2.00 s | Max delay after `t_user_end` for responsive check |

---

## Aggregation

1. **Instance → L3:** Mean B_i, mean R_i, criterion means, coverage (fraction with `t_user_end` available).
2. **L3 → L2:** Macro-average B_c and R_c across the two conditions (equal weight by default).
