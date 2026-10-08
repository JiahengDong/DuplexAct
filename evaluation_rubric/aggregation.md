# Score Aggregation

## Overview

Evaluation proceeds in three stages:

```
EvalInstance  →  InstanceRecord  →  L3Result  →  L2Result
  (input)          (per instance)    (per condition)  (per action)
```

---

## Stage 1 — Instance Level

Each instance is scored independently and produces an `InstanceRecord`.

| Field | Computation |
|---|---|
| `B_i` | Auto: 1 if the required event occurred in the valid window, 0 otherwise |
| `T_i_raw` | Auto: signed timing offset in seconds (headline; `None` for silence-based actions) |
| `e_i[k]` | Binary score for criterion k — auto-computed or LLM / human judge |
| `applicability_mask[k]` | Whether criterion k applies to this instance (some criteria are condition-specific) |
| `R_i` | `mean(e_i[k])` over criteria where `applicability_mask[k] = True` |

---

## Stage 2 — Level-3 Aggregation

All instances sharing the same `(l2_action, l3_condition)` are grouped. Instance counts are equalised across conditions before aggregation, so micro and macro averaging are equivalent.

| Output | Formula |
|---|---|
| `B_c` | `mean(B_i)` — success rate for this condition |
| `R_c` | `mean(R_i)` — mean reliability score for this condition |
| `timing_median` | `median(T_i_raw)` over instances with timing defined; absent for silence-based actions |
| `timing_IQR` | `(Q1, Q3)` of `T_i_raw` (reported when ≥ 4 timed instances available) |
| `criterion_means[k]` | `mean(e_i[k])` across instances where criterion k is applicable |
| `n_instances` | Total instances in this condition group |

---

## Stage 3 — Level-2 Aggregation

L3 results are macro-averaged across conditions (each condition contributes equally).

| Output | Formula |
|---|---|
| `B_l2` | `mean(B_c)` across all L3 conditions |
| `R_l2` | `mean(R_c)` across all L3 conditions |
| `timing_summary` | `median` of per-condition `timing_median` values + per-condition breakdown; absent when no condition carries timing |

---

## Reliability Checklist per L2 Action

The number of reliability criteria differs across actions, which affects what R_i represents.

| L2 Action | # Criteria | Criteria | Judge required |
|---|---|---|---|
| Wait | 1 | `responsive` | Auto |
| Stop | 1 | `release_attributable` | LLM |
| Ignore | 2 | `responsive`, `no_leakage` | Auto + LLM |
| Hold Floor | 2 | `continue_intent`, `semantic_steering` | LLM, LLM |
| Interrupt User | 2 | `interruption_position`, `content_validity` | LLM, LLM |
| Smooth Turn Taking | 3 | `relevance`, `information_completeness`, `helpfulness` | LLM, LLM, LLM |
| Respond & Recover | 2–4 | `recover_type_match`, `addresses_interruption`, `natural_reentry`†, `returns_to_thread`† | LLM × 2–4 |
| Backchannel | TBD | TBD | TBD |

† `natural_reentry` and `returns_to_thread` are not applicable for the `shift_topic` condition.

---

## Limitations

### 1. Reliability criteria are author-defined, not empirically validated

The reliability checklists for each action were designed by the authors based on the rubric framework. There is no established ground truth for what constitutes a sufficient or necessary set of reliability criteria for each action. Different annotators or researchers might define different criteria, and the current set may under-cover or over-specify what matters for reliable behaviour.

### 2. Equal weighting of criteria within R_i

At the instance level, `R_i = mean(e_i[k])` treats all applicable criteria as equally important. In practice, some criteria may have a larger influence on interaction quality than others. For example, `release_attributable` in Stop is the single criterion and carries full weight, while `helpfulness` in Smooth Turn Taking is one of three equally weighted items. No mechanism currently exists to assign differential importance to criteria.

### 3. R_i is not comparable across L2 actions

The number of criteria differs by action (1 to 4 among currently finalised actions). An R_i of 0.75 for Stop (1 criterion) carries a different evidential weight than R_i of 0.75 for Respond & Recover (up to 4 criteria). R_l2 values across different actions should not be directly compared as if they were on the same scale. Cross-action comparisons should be interpreted with the criterion count in mind.

### 4. Timing axis is absent for silence-based actions

Wait, Ignore, and Stop measure success as the absence of an event (no onset in the forbidden window). There is no meaningful timing offset to report for these actions — `T_i_raw = None` throughout, so no timing summary is produced at L3 or L2. Timing-based analysis is only available for event-based actions (Smooth Turn Taking, Interrupt User, Respond & Recover, Hold Floor, Backchannel).
