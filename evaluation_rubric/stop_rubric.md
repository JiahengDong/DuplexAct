# Stop — Level-2 Rubric

## Overview

| Field | Value |
|---|---|
| **Level-2 action** | Stop |
| **Metric family** | YR — Yield Rate |
| **Timing variable** | L_yield = t_model_yield_end − t_user_int_start |
| **Judge required** | Yes — R1 requires LLM / human judge |

**Base success (B_i):** While the model is speaking, the user interrupts mid-sentence and the model yields the floor within the valid yield window.  
**Reliability (R_i):** Single binary evidence item — was the stop caused by the interruption?

---

## Level-3 Condition

| Condition | Description |
|---|---|
| `mid_sentence_stop` | User interrupts the model mid-sentence; model must stop promptly after the interruption onset |

---

## Timestamps Required

| Key | Description |
|---|---|
| `t_user_int_start` | Onset of the user's interrupting speech |
| `t_model_active` | A time known to be inside the model's active speech segment (optional; defaults to `t_user_int_start`) |

---

## B_i — Base Success (Automatic)

**Definition:** B_i = 1 iff the model's yield end falls within the valid yield window:

```
t_model_yield_end ∈ [t_user_int_start + YIELD_W_LO, t_user_int_start + YIELD_W_HI]
```

`t_model_yield_end` is the offset of the model speech segment that was active at `t_user_int_start`.

**Key constraint:** `YIELD_W_LO = 0.0` — the model can only yield **at or after** the interruption onset. A yield before `t_user_int_start` cannot be attributed to the interruption and is not a valid stop.

**Failure tags:**

| Tag | Meaning |
|---|---|
| `yield_too_late:L_yield=<t>s` | Model kept speaking beyond `YIELD_W_HI` after interruption |
| `yield_before_interruption:L_yield=<t>s` | Model's active segment ended before `t_user_int_start` (only reachable when `t_model_active` is provided earlier than `t_user_int_start`) |
| `no_event:model_not_speaking_at_interruption` | No model segment was active at `t_user_int_start`; model had already finished |
| `missing_timestamp:t_user_int_start` | Required timestamp absent |

---

## T_i — Timing Variable (Automatic)

```
L_yield = t_model_yield_end − t_user_int_start   (seconds, signed)
```

Smaller non-negative value is better — the model should stop as soon as possible after the interruption begins. `T_i_raw` is stored in `InstanceRecord` and reported at the Level-3 stage as a timing summary (median and IQR across instances).

---

## Reliability Item

### R1 — Release Attributable (LLM / Human Judge)

| | |
|---|---|
| **Applicable to** | All instances |
| **Auto-computable** | No — requires reasoning about the model's speech content |

**Definition:** R1 = 1 if the model's stop is attributable to the user's interruption rather than a coincidental natural sentence end.

**Judge question:**  
*Is the model's floor release attributable to the user's interruption, rather than a coincidental natural sentence end? Answer YES if the model was mid-sentence or mid-thought when it stopped, and the stop timing is consistent with a response to the user's interruption. Answer NO if the model had already reached a natural completion point (sentence end, topic boundary) before or at the moment it stopped, suggesting the interruption had no causal effect on the stop.*

| | |
|---|---|
| **Positive evidence** | Model stops mid-utterance; the stopping point is not a natural sentence boundary |
| **Failure evidence** | Model reaches a natural sentence or clause end just before stopping; stop appears coincidental |
| **Context required** | `transcript_model`, `transcript_user`, `timestamps` |

---

## Tunable Intervals

Both constants live at the top of `actions/stop.py`.

| Constant | Default | Meaning |
|---|---|---|
| `YIELD_W_LO` | 0.00 s | Earliest valid yield end relative to `t_user_int_start` (must be ≥ 0) |
| `YIELD_W_HI` | 2.00 s | Latest valid yield end relative to `t_user_int_start` |

---

## Aggregation

1. **Instance → L3:** Mean B_i, timing median [IQR] of L_yield, mean R_i, coverage.
2. **L3 → L2:** Single condition; L2 result equals the L3 result directly.
