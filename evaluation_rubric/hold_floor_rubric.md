# Hold Floor Rubric

## Overview

| Field | Value |
|---|---|
| L2 Action | `hold_floor` |
| Metric family | FHR — Floor-Holding Rate |
| B_i computation | Automatic (VAD / timestamps) |
| T_i | Diagnostic only (max silence gap in window) |
| Reliability items | 2 (both LLM / human judge) |

The model is presented with an **interfering cue** (background speech, a backchannel, or user speech directed elsewhere) while it is already speaking. A successful Hold Floor means the model continues its turn through the cue without releasing the floor.

---

## Level-3 Conditions

| Condition | Description |
|---|---|
| `continue_with_background_speech` | Background speech occurs while the model is talking; model must keep speaking without pausing |
| `continue_with_other_directed_speech` | User briefly addresses someone else while the model is talking; model must keep speaking |
| `continue_current_response` | User gives a neutral backchannel (e.g. "mm-hmm"); model continues the response as-is |
| `continue_but_shorten` | User gives a backchannel implying "wrap up"; model continues but shortens the response |
| `continue_with_simplification` | User gives a backchannel implying "simplify"; model continues but simplifies the response |

---

## Required Timestamps

| Key | Description |
|---|---|
| `t_cue_start` | Onset of the interfering cue (required) |
| `t_cue_end` | Offset of the interfering cue (optional; defaults to `t_cue_start` if absent) |

---

## B_i — Binary Success (Automatic)

**B_i = 1** iff the maximum silence gap observed in the observation window **does not exceed** `MAX_GAP_S`.

**Observation window:** `[t_cue_start, t_cue_end + GRACE_S]`

**Maximum silence gap:** the longest uninterrupted silence in the observation window, including any leading silence before the first model segment and trailing silence after the last.

### Tunable intervals

| Constant | Default | Meaning |
|---|---|---|
| `MAX_GAP_S` | `0.50 s` | Maximum permissible silence gap (applies to all five conditions) |
| `GRACE_S` | `0.50 s` | Grace period added after `t_cue_end` to extend the observation window |

### Failure tags

| Tag | Meaning |
|---|---|
| `missing_timestamp:t_cue_start` | `t_cue_start` not provided |
| `no_event:model_speech_absent` | No model speech segments present |
| `floor_released:gap=Xs>threshold=Ys` | Observed max gap exceeded the threshold |

---

## T_i — Timing Diagnostic (Not a Headline Score)

`score_t` returns the **observed maximum silence gap** (seconds) in the observation window, for diagnostic purposes only. It is stored in `confidence_metadata["max_gap_s"]` and included in `evidence_spans`, but is not used as a headline ranking metric.

---

## Reliability Items

Both items require an **LLM or human judge**.

### R1 — `continue_intent`

> After the interfering cue, does the model maintain the intended content direction of its response?

- **YES** — model continues on the same topic, goal, or thread it was pursuing before the cue, even if it shortens or simplifies the delivery.
- **NO** — model abandons the current thread, changes topic without cause, or restarts from the beginning.
- Applies to all five conditions (same prompt).

### R2 — `semantic_steering`

Condition-specific question about whether the model responded to the cue's *implied intent* appropriately:

| Condition | Expected behaviour |
|---|---|
| `continue_with_background_speech` | Model's content is **unaffected** by the background speech |
| `continue_with_other_directed_speech` | Model's content continues **unaffected** by the user's side conversation |
| `continue_current_response` | Model proceeds **without unexpected restructuring** |
| `continue_but_shorten` | Model **visibly shortens or condenses** its response |
| `continue_with_simplification` | Model **visibly simplifies** its language or content |

---

## R_i Aggregation

```
R_i = mean(e_1, e_2)      # both items always applicable
```

---

## Data Aliases

| Synthesized label | Canonical condition |
|---|---|
| `Continue_with_background_speech` | `continue_with_background_speech` |
| `Continue_with_other_directed_speech` | `continue_with_other_directed_speech` |
