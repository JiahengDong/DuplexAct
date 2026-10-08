# Turn-Taking: How We Measure "Model Speech" and Response Onset

## Overview

| Field | Value |
|---|---|
| Applies to | `smooth_turn_taking` (all three L3 conditions) |
| Scope | Design rationale for `t_model_start` selection and `model_speech_segments` detection — *not* a redefinition of the formal rubric (see `smooth_turn_taking_rubric.md` for B_i/T_i/R_i formulas) |
| Code | `evaluation/actions/smooth_turn_taking.py`, `evaluation/direct_replay/inference/realtime_common.py`, `local_common.py` |

This note exists because the rubric file defines *what* B_i/T_i require (an onset inside a timing window), but not *how* we decide which of a model's possibly-many speech segments counts as "the" onset being scored, or how "speech" is detected for models in the first place. Both of those are implementation decisions with real failure modes we hit in practice (Moshi), so we document the reasoning here rather than leaving it implicit in the code.

---

## 1. What `turn_taking` is — and deliberately is not — measuring

`smooth_turn_taking` answers one narrow question: *once the user has finished speaking, does the model correctly take the floor within the expected timing window, and how good is the response?* It does **not** attempt to also detect whether the model inappropriately spoke *during* the user's turn — that failure mode already has a dedicated test.

This separation is intentional and structural, not incidental:

- `wait` (specifically the `floor_holding` L3 condition, plus `inter_sentential`/`intra_sentential`) tests exactly "does the model stay silent while the user continuously holds the floor" — its `score_b` fails the instance if **any** model onset falls inside `[t_wait_start, t_wait_end]`.
- `turn_taking` tests "does the model correctly seize the floor at the right moment" — its `score_b` only checks whether a valid onset exists *near* `t_user_end`, without caring whether the model also spoke earlier in the clip.
- `wait` and `turn_taking` are scored on **disjoint sample sets** (different clips under `data/wait/...` vs `data/turn_taking/...`), not the same clip scored two different ways. So there is no double-counting or gap: a model's tendency to prematurely interrupt is fully captured by `wait`'s conditions, and `turn_taking` is free to ignore that question entirely and focus on timing/quality of the *correct* take-turn moment.

A consequence worth being explicit about: a model that both (a) interrupts early, well before `t_user_end`, *and* (b) also correctly responds again right at `t_user_end` within the valid window, scores **B_i = 1** for `turn_taking` on that instance. That is correct under this design — (a) is out of scope here and is `wait`'s job on its own samples.

---

## 2. The problem this caused in practice: Moshi's greeting

`smooth_turn_taking.py`'s onset-picking originally just took the *earliest* speech segment anywhere in the clip as "the" response:

```python
def _first_model_onset(instance):
    if not instance.model_speech_segments:
        return None
    return min(seg[0] for seg in instance.model_speech_segments)
```

This is fine for models that only speak once, in response to the user. It breaks for Moshi, which — being a native full-duplex model — always emits an unprompted greeting in the first ~0.5s of *every* session, completely independent of the conversation content. That greeting is technically "a model speech segment," so the naive picker grabbed it as `t_model_start` for every instance, producing a median `T_response` around **−12s** (i.e. the model appearing to "respond" 12 seconds before the user even finished speaking) — an artifact of pre-roll noise, not a real turn-taking measurement.

We are not the first to hit this: `FD-Bench`'s own `benchmarking.py` explicitly special-cases it —

```python
if self.model_name == "Moshi":
    ai_starts = ai_starts[1:]
    ai_ends = ai_ends[1:]
```

— unconditionally discarding the first AI segment for this one model. We chose not to copy that approach (see §3) because it's a hard-coded, model-specific carve-out that only works if the greeting is always exactly the first segment and never recurs; a window-based rule generalizes to any model with pre-roll artifacts without needing a per-model exception list.

---

## 3. Our fix: a `min_onset` guard anchored to `t_user_end`

`_first_model_onset` now takes an optional `min_onset` and discards any candidate segment starting before it — the earliest *qualifying* onset, not the earliest onset in the whole clip:

```python
PRE_ROLL_GUARD_S = 1.0

def _first_model_onset(instance, min_onset=None):
    if not instance.model_speech_segments:
        return None
    candidates = [seg[0] for seg in instance.model_speech_segments]
    if min_onset is not None:
        candidates = [t for t in candidates if t >= min_onset]
    if not candidates:
        return None
    return min(candidates)
```

`score_b` / `score_t` call it with `min_onset = t_user_end - PRE_ROLL_GUARD_S`. Why `1.0s`:

- The widest onset window across all three L3 conditions is `slight_overlap_take_turn`'s `[-0.50s, +0.50s]` (plus `BOUNDARY_TOLERANCE = 0.05s`), so nothing any condition considers a genuine response can start earlier than `t_user_end - 0.55s`.
- `1.0s` sits comfortably beyond that worst case, so it never risks excluding a real early-overlap response.
- Moshi's greeting sits at ~0.45s from session start, while `t_user_end` is typically tens of seconds into the clip — miles outside a 1s guard — so it's reliably excluded.

Critically, the guard doesn't ask *why* an earlier segment exists (greeting, genuine early interruption, noise) — anything before `t_user_end - 1.0s` is out of scope for "the" turn-taking response by construction, consistent with §1: this function's only job is to find a candidate response to `t_user_end`, not to audit everything the model said earlier in the clip.

**Result after the fix** (Moshi, all three conditions re-scored): B_i values changed from artifacts near-0 to plausible values, and median `T_response` flipped from ≈ -12s to a small positive/near-zero number consistent with Moshi's real response latency once the greeting is filtered out.

---

## 4. How `model_speech_segments` itself is detected

Independent of onset *selection* above, segments themselves are produced by one of three tiers depending on what signal a model exposes (see `realtime_common.py` / `local_common.py` docstrings for full detail):

| Tier | Mechanism | Used by | Why |
|---|---|---|---|
| 1. Automatic presence/absence | `SynchronizedRecorder` opens/closes a segment based on literal PCM bytes arriving in its queue | Closed-source realtime APIs (GPT-Realtime, Grok Voice, Gemini Live, Qwen Omni), most open-source turn-based servers | Default — no VAD needed when "queue empty" genuinely means "not talking" |
| 2. Explicit native signal (`recorder.set_speaking(...)`) | Adapter drives segment boundaries from the model's own per-frame "am I producing a word right now" signal | Moshi (text-token cadence, `MOSHI_SILENCE_GRACE_S = 0.8s` intra-utterance pause tolerance), MiniCPM-o (`is_listen`) | Needed for models that emit a continuous audio signal with no literal digital silence to detect |
| 3. Post-hoc RMS-energy thresholding | `BatchReplayClientBase._energy_segments()`, `SILENCE_RMS_DBFS=-40`, merges gaps ≤0.3s, drops runs <0.15s | Pure batch-call models with no incremental signal at all (e.g. Voila-Autonomous, unless a native `streamer=` hook is available) | Last resort — coarsest, heuristic substitute for a real "is speaking" signal |

This matters for §3 because the *quality* of `t_model_start` is only as good as the underlying segmentation tier: Moshi's greeting is a real, correctly-detected tier-2 segment (the fix in §3 was about onset *selection*, not fixing bad segmentation), whereas a tier-3 model's segment boundaries carry more inherent uncertainty from the energy threshold itself.

---

## 5. Known limitations

- **`PRE_ROLL_GUARD_S` is a single global constant**, not tuned per-model or per-condition. It was sized against the widest L3 window (`slight_overlap_take_turn`) plus a comfortable margin, so it should be safe for any current condition, but a future L3 condition with a lower bound below −1.0s would need this revisited.
- **Edge case (theoretical, not observed in any of the 5 models evaluated so far):** if a model began speaking well before `t_user_end` and never stopped — one continuous segment straddling the guard boundary rather than two discrete ones — the guard would exclude that segment entirely (its start is before the cutoff) and find no later candidate, scoring `no_event:model_speech_absent`. This under-counts a model that happens to still be talking at the right moment but never exhibits a discrete "restart." We have not seen this pattern in practice (Moshi's greeting is a short, self-contained segment that ends well before the guard boundary), so we have not built special handling for it.
- **This fix only affects onset *selection*, not the `wait`/`floor_holding` samples or scoring** — those are unaffected by anything in this document, by design (§1).
