# Smooth Turn Taking Rubric

## Overview

| Field | Value |
|---|---|
| L2 Action | `smooth_turn_taking` |
| Metric family | TOR — Take-Over Rate |
| B_i computation | Automatic (VAD / timestamps) |
| T_i | Headline score (automatic) |
| Reliability items | 3 (all LLM / human judge, condition-specific prompts) |

The model is expected to take the conversational floor after the user finishes speaking. Three conditions test different timing styles — immediate response, delayed deliberate response, and slight-overlap response — representing a tolerance spectrum rather than strict distinct categories.

---

## Level-3 Conditions

| Condition | Description | Accepted onset window |
|---|---|---|
| `immediate_take_turn` | Model responds immediately after user finishes | `[0.0 s, +0.50 s]` relative to `t_user_end` |
| `delayed_take_turn` | Model responds after a deliberate pause (correction / reclarification window) | `[0.0 s, +3.00 s]` relative to `t_user_end` |
| `slight_overlap_take_turn` | Model begins speaking close to the end of the user's turn (symmetric window) | `[−0.50 s, +0.50 s]` relative to `t_user_end` |

---

## Required Timestamps

| Key | Description |
|---|---|
| `t_user_end` | Offset of the user's turn (required for B_i and T_i) |

---

## B_i — Binary Success (Automatic)

**B_i = 1** iff:

```
t_model_start − t_user_end  ∈  [w_lo − BOUNDARY_TOLERANCE,  w_hi + BOUNDARY_TOLERANCE]
```

where `[w_lo, w_hi]` is the condition's window from the table above.

### Tunable intervals

| Constant | Default | Meaning |
|---|---|---|
| `L3_WINDOWS["immediate_take_turn"]` | `[0.0 s, 0.50 s]` | Onset within half a second of user end |
| `L3_WINDOWS["delayed_take_turn"]` | `[0.0 s, 3.00 s]` | Onset up to 3 seconds after user end |
| `L3_WINDOWS["slight_overlap_take_turn"]` | `[−0.50 s, +0.50 s]` | Onset within ±0.5 s of user end |
| `BOUNDARY_TOLERANCE` | `0.05 s` | Symmetric buffer at each window boundary to absorb annotation jitter |

### Failure tags

| Tag | Meaning |
|---|---|
| `missing_timestamp:t_user_end` | `t_user_end` not provided |
| `no_event:model_speech_absent` | No model speech segments present |
| `window_miss_early` | Model onset was earlier than the window lower bound |
| `window_miss_late` | Model onset was later than the window upper bound |

---

## T_i — Timing (Headline Score, Automatic)

```
T_response = t_model_start − t_user_end   (signed seconds)
```

| Value | Meaning |
|---|---|
| Negative | Model began before the user finished (overlap) |
| Zero | Exact alignment with user end |
| Positive | Model began after the user finished (gap) |

T_i is reported as a headline score. Combined with B_i, it shows not just whether the onset was inside the window, but how far into or outside the window it fell.

---

## Reliability Items (all LLM / human judge, condition-specific prompts)

All three items are always applicable. Prompts differ by condition to reflect the different expectations of each timing style.

### R1 — `relevance`

Does the model's response address the user's intent?

- **`immediate_take_turn`**: Judge assesses whether the response addresses the user's intent as established in their utterance (intent is typically clear before the final entity).
- **`delayed_take_turn`**: Judge assesses relevance accounting for any correction or reclarification the model provides — the response should stay on-subject even if it corrects a misunderstanding.
- **`slight_overlap_take_turn`**: Judge assesses whether the response addresses the user's topic correctly based on the content of the user's speech.

**YES** — response directly addresses the user's question, topic, or underlying need.  
**NO** — response is off-topic, generic, or misses the user's intent.

---

### R2 — `information_completeness`

Does the response capture the right information given the condition's expectations?

| Condition | Expected behaviour |
|---|---|
| `immediate_take_turn` | Captures **all important information**, especially the final key entity or point the user just completed |
| `delayed_take_turn` | Includes only the **correct and necessary** information — may correct or reclarify, does not need to carry forward irrelevant earlier content |
| `slight_overlap_take_turn` | Captures all important information **available up to the point** the model began speaking |

**YES** — response reflects the right information for the condition.  
**NO** — response misses key content, repeats irrelevant content uncritically, or omits a needed correction.

---

### R3 — `helpfulness`

Does the model's response effectively solve or address the user's request?

| Condition | Expected behaviour |
|---|---|
| `immediate_take_turn` | Provides a direct, actionable answer that fulfils the user's need |
| `delayed_take_turn` | Provides an accurate, precise answer — correcting misunderstandings or resolving ambiguities where relevant |
| `slight_overlap_take_turn` | Correctly understands the user's need and provides a helpful, accurate response |

**YES** — response addresses the user's need effectively.  
**NO** — response is evasive, inaccurate, incomplete, or misreads the request.

---

## R_i Aggregation

```
R_i = mean(R1, R2, R3)      # all three always applicable
```

---

## Data Aliases

| Synthesized label | Canonical condition |
|---|---|
| `take_turn_immediately` | `immediate_take_turn` |
| `take_turn_with_delay` | `delayed_take_turn` |
| `slight_overlap_take_turn` | `slight_overlap_take_turn` |
