# Respond & Recover Rubric

## Overview

| Field | Value |
|---|---|
| L2 Action | `respond_and_recover` |
| Metric family | TOR — Take-Over Rate |
| B_i computation | Automatic (VAD / timestamps) |
| T_i | Headline score (automatic) |
| Reliability items | 4 (all LLM / human judge; R3 and R4 not applicable for `shift_topic`) |

The model is interrupted mid-turn by the user. After the interruption ends, the model is expected to recover by performing a specific recovery action. The Level-3 condition labels describe **what the model does** as its recovery, not what the user said.

---

## Level-3 Conditions (Model Recovery Types)

| Condition | Model recovery action |
|---|---|
| `enrich` | Model enriches its own response — expands or adds detail based on the interruption |
| `shorten` | Model shortens its own response — condenses based on the interruption |
| `repeat` | Model repeats a specific segment of its own prior response |
| `shift_topic` | Model shifts to a new topic prompted by the interruption |
| `ask_follow_up` | Model poses a follow-up question based on the interruption |

---

## Required Timestamps

| Key | Description |
|---|---|
| `t_interruption_end` | When the user's interruption ends — the recovery reference point (required) |
| `t_interruption_start` | When the user's interruption began (optional; for context) |

---

## B_i — Binary Success (Automatic)

**B_i = 1** iff the model's first speech onset *after* `t_interruption_end` falls within:

```
[t_interruption_end + w_lo − tol,  t_interruption_end + w_hi + tol]
```

### Tunable interval (shared across all conditions)

| Constant | Default | Meaning |
|---|---|---|
| `RECOVERY_WINDOW` | `(−0.10 s, +3.00 s)` | Valid onset window relative to `t_interruption_end`; small negative permits slight overlap with interruption end |
| `BOUNDARY_TOLERANCE` | `0.05 s` | Symmetric buffer at window boundaries |

### Failure tags

| Tag | Meaning |
|---|---|
| `missing_timestamp:t_interruption_end` | Reference timestamp not provided |
| `no_event:no_model_speech_after_interruption` | No model speech after the interruption |
| `window_miss_early` | Model speech onset was before the window lower bound |
| `window_miss_late:T=Xs` | Model speech onset was after the 3 s upper bound |

---

## T_i — Timing (Headline Score, Automatic)

```
T_recovery = t_model_start − t_interruption_end   (signed seconds)
```

| Value | Meaning |
|---|---|
| Negative | Model began before the interruption fully ended |
| Zero | Exact alignment with interruption end |
| Positive | Model began after a gap following the interruption |

---

## Reliability Items

All items require an **LLM or human judge**.

### R1 — `recover_type_match` (condition-specific)

Did the model perform the correct recovery action for this test case?

| Condition | Expected behaviour |
|---|---|
| `enrich` | Post-interruption speech is visibly richer or more detailed than before |
| `shorten` | Post-interruption speech is visibly shorter or more concise |
| `repeat` | Model reproduces or closely paraphrases the specific prior segment |
| `shift_topic` | Model transitions to the new topic and engages with it |
| `ask_follow_up` | Model poses a relevant, well-formed follow-up question |

- **YES** — model performed the expected recovery type.
- **NO** — model performed a different recovery type or ignored the cue.

---

### R2 — `addresses_interruption`

Did the model's recovery actually handle the content of the user's interruption?

- **YES** — the recovery explicitly deals with what the user said (incorporates, answers, adjusts, or acknowledges it).
- **NO** — the model proceeds as if the interruption did not occur.

Applies to all five conditions.

---

### R3 — `natural_reentry`

Does the model re-enter the original content at a semantically natural position, rather than restarting from the beginning?

- **YES** — model resumes at a coherent, logical point without unnecessary repetition or jarring jumps.
- **NO** — model restarts the entire response unnecessarily, skips critical content, or re-enters mid-sentence confusingly.

**Not applicable for `shift_topic`** — no re-entry into the original content is expected when the topic intentionally changes.

---

### R4 — `returns_to_thread`

After addressing the interruption, does the model return to the original conversation thread?

- **YES** — model transitions back to and continues or completes the original topic or task.
- **NO** — model stays on the interruption topic indefinitely or ends without returning.

**Not applicable for `shift_topic`** — the topic is intentionally changed; returning to the original thread is not the expected behaviour.

---

## R_i Aggregation

```
# All conditions except shift_topic:
R_i = mean(R1, R2, R3, R4)

# shift_topic only:
R_i = mean(R1, R2)
```

---

## Data Aliases

| Synthesized label | Canonical condition |
|---|---|
| `enrich_response_after_interruption` | `enrich` |
| `shorten_response_after_interruption` | `shorten` |
| `repeat_response_after_interruption` | `repeat` |
| `shift_topic_after_interruption` | `shift_topic` |
| `ask_follow_up_after_interruption` | `ask_follow_up` |
