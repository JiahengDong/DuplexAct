# Wait — Level-2 Rubric

## Overview

| Field | Value |
|---|---|
| **Level-2 action** | Wait (Silent Wait) |
| **Metric family** | SR — Silence Rate |
| **Timing variable** | None |
| **Judge required** | No — all scores are auto-computed |

**Base success (B_i):** No **substantive** model speech begins inside the wait interval. A short backchannel (≤ 3 words) is permitted and does not fail B_i.  
**Reliability (R_i):** Single auto-computed item — responsive.

---

## Level-3 Conditions

| Condition | Description | Wait window end |
|---|---|---|
| `inter_sentential_pause` | Pause between sentences; model must wait until user resumes | `t_wait_end` (user resumes) |
| `intra_sentential_pause` | Pause within a sentence; model must wait for sentence completion | `t_wait_end` (user resumes) |
| `continuous_floor_hold` | No acoustic pause; user verbally holds floor (e.g. "hold on"); model must wait for explicit release | `t_release` (verbal cue) |

---

## Timestamps Required

| Key | Description | Used by |
|---|---|---|
| `t_wait_start` | Onset of the pause or hold event | B_i |
| `t_wait_end` | End of the wait window (user resumes / sentence completes) | B_i, responsive (pause conditions) |
| `t_release` | Explicit verbal release cue (`continuous_floor_hold` only; serves as `t_wait_end` when absent) | B_i, responsive |
| `t_user_end` | When the user's full turn ends | responsive |

---

## B_i — Base Success (Automatic)

**Definition:** B_i = 1 iff no **substantive** model speech onset falls inside `[t_wait_start, t_wait_end]`.

A model speech segment with onset inside the window is treated as a backchannel (acceptable) if its word count is **≤ 3 words**, and as a spurious takeover (B_i = 0) if its word count is **> 3 words**.

**Backchannel threshold:** `BACKCHANNEL_MAX_WORDS = 3` (exclusive). Adjust at the top of `wait.py`.

**Word count source:** `metadata["segment_transcripts"]` — a list of transcript strings aligned index-by-index to `model_speech_segments`. If not provided, the segment is conservatively treated as substantive (word count unknown → fails B_i).

**Failure tags:**
- `takeover_during_wait:onset=<t>s` — substantive speech began during the wait interval
- `missing_segment_transcript:onset=<t>s` — onset found but transcript unavailable; treated as substantive

---

## Reliability Item (R1)

Auto-computed from VAD segments and timestamps. No LLM call is needed.

### R1 — Responsive

| | |
|---|---|
| **Applicable to** | All three conditions |

**Definition:** responsive = 1 iff the model's first speech onset falls within the condition-specific valid response window after the pause / release ends.

This is a pure **"did it ever respond"** check, not a response-*speed* check — response latency/quality after a wait interval is smooth_turn_taking's concern, not wait's. Accordingly the window **end** is the full time the model had available to respond (`metadata.padding_end`, i.e. the end of the recorded/padded clip), not a short fixed latency cap. Only the window **start** differs by condition, and only by a small boundary-tolerance grace period.

**Response windows:**

| Condition | Window | Interpretation |
|---|---|---|
| `inter_sentential_pause` | `[t_wait_end + PAUSE_START_GRACE_S, padding_end]` | Respond anytime from just after pause end to the end of the sample's recorded/padded clip |
| `intra_sentential_pause` | `[t_wait_end + PAUSE_START_GRACE_S, padding_end]` | Same as above |
| `continuous_floor_hold` | `[t_user_end − FLOOR_HOLD_OVERLAP_MAX_S, padding_end]` | Respond any time from just before the natural end of user speech to the end of the recorded/padded clip; slight overlap allowed |

**Why not a short latency cap?** An earlier version capped the window end at `t_user_end + 2.0s`. Debugging showed this systematically penalized models whose turn-detection needs longer confirmed-silence before committing to a response on `continuous_floor_hold` specifically (no acoustic pause to key off of) — e.g. GPT-Realtime's `semantic_vad` genuinely responds at a median lag of ~3.8s post-`t_user_end` on this condition (vs. ~0.5–1.5s for Gemini/MiniCPM), which is a legitimate, successful response, not a failure to respond. Since *how fast* is already measured elsewhere, `responsive` was widened to just ask *whether at all*.

---

## Tunable Intervals

All numerical boundaries live at the top of `actions/wait.py` and should be adjusted before benchmark release.

| Constant | Default | Meaning |
|---|---|---|
| `BACKCHANNEL_MAX_WORDS` | 3 | Max word count for a permitted backchannel during the wait interval (B_i) |
| `FLOOR_HOLD_OVERLAP_MAX_S` | 0.20 s | Max pre-emption before `t_user_end` (continuous_floor_hold responsive window start) |
| `PAUSE_START_GRACE_S` | 0.20 s | Grace period after `t_wait_end` (inter/intra_sentential responsive window start) |

Responsive window **end** is always `metadata.padding_end` (fallback: `t_user_end` if unavailable) — no separate tunable constant.

---

## Aggregation

Per the framework (Section 4.2):

1. **Instance → L3:** Mean B_i, mean R_i, criterion means, coverage over all instances of the same condition.
2. **L3 → L2:** Macro-average B_c and R_c across the three conditions (equal weight by default).

Coverage = fraction of instances where `t_user_end` was available (responsive check possible).
