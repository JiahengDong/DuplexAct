# Appendix: Evaluation Protocols and Action Metrics

Source of truth for scoring: `evaluation/scoring/`.
This note is written so it can be copied into the paper appendix with light editing.

Every Level-2 action is scored as two 0/1 halves:

- **Immediate** — did the model do the right thing in the annotated timing interval?
- **Subsequent** — did later behaviour correctly use or respect what happened during or after that interval?

Subsequent is **conditional** on Immediate: denominator is Immediate $=1$. Immediate fail $\rightarrow$ N/A (excluded, not $S=0$). On direct replay, no later response among Immediate-success trials is $S=0$, not N/A; a later response that fails the action check is also $S=0$; $S=1$ only if the later response meets the requirement.

Immediate intervals are condition-specific. Subsequent wording is shared across conditions of the same action, except where a condition prompt is needed because the cue itself is underspecified (Hold floor, Interrupt, Wait). Scores are never inherited from the old B/T/R scorers.

Unless noted, Immediate is automatic from time-aligned `model_speech_segments` on the shared `input`/`output` timeline. Subsequent is a text LLM judge (`gpt-5.4`, `YES`/`NO` JSON, `temperature=0`) except Stop Subsequent, which is automatic from segments. Backchannel is the exception on Immediate: it is judged in two stages (Section I.9), a text screen for whether a backchannel occurred at all (`gpt-5.4`) followed by the only listening judge (`gpt-audio`) for its function.

---

## A. Shared operational rules

**Backchannel vs takeover.** A model speech segment is a *backchannel* if its full ASR transcript has $\le 3$ words. Empty ASR on a VAD segment is 0 words and is treated as a backchannel. A missing or length-mismatched transcript cannot be proven a backchannel, so an overlapping VAD segment counts as a takeover. Word count is always on the **whole segment**, never on the slice that happens to fall inside a short pause.

**Take-turn-with-overlap tail.** The last $1.0\,\mathrm{s}$ of user speech (or of a cue) is the take-turn-with-overlap lookback. Onset there is allowed as a legitimate next-turn start. It is **not** Interrupt Immediate success, and it is **not** a Stop Subsequent reclaim.

**Timeline.** $t=0$ is stream start unless an action re-anchors (user-speech end, cue onset, interruption offset, user onset). `input` and `output` share the same clock. Trailing padding after user-speech end is outside Immediate unless the action says otherwise.

**Headline population.** Listen actions (Wait, Backchannel, Ignore, Take turn) and Interrupt: all scored samples. Stop / Hold floor / Recover: the cue landed while the model was **speaking** — `cue_class = speaking`, i.e. a post-lead-in model turn overlaps cue onset (`evaluation/scoring/cue_landing.py`). **Forced is allowed.** Cue in silence (`no_reply`) or after the model already finished (`finished`) is coverage, not in the Immediate denominator. Transport-error trials are dropped, not scored as silence.

---

## B. Which protocol each action uses

| Action | $N$ | Protocol | Why |
|---|---:|---|---|
| Wait | 600 | Direct replay | Model is listening; withhold a response during a pause or a continuous turn. |
| Backchannel | 400 | Direct replay | Model is listening; insert a short continuer without taking the floor. |
| Ignore | 400 | Direct replay | Model is listening; do not respond to a distractor in the user’s clip. |
| Take turn | 600 | Direct replay | Model is listening; enter after (or slightly before) user-speech end. |
| Stop | 200 | Simulator (cue) | Model already holds the floor; a user interruption must make it yield. |
| Hold floor | 891 | Simulator (cue) | Model already holds the floor; a non-takeover cue must not make it yield. |
| Recover | 1200 | Simulator (cue) | After a user interruption ends, the model must re-enter and adapt. |
| Interrupt | 400 | Simulator (barge-in) | User holds the floor; the model must enter before the authored deadline. |

Direct replay covers listen-side actions. The simulator covers speak-side actions: cue injection while the model is talking (Stop / Hold floor / Recover) or live barge-in while the user is talking (Interrupt). Sections C–D are the interaction pipelines (clocks, I/O, what is live vs fixed). Scoring is Section E onward. Checkpoints and API IDs are **Section J**.

| | Direct replay | Simulator |
|---|---|---|
| Actions | Wait, Backchannel, Ignore, Take turn | Stop, Hold floor, Recover, Interrupt |
| User audio | One fixed `input.wav` | Growable source: lead-in then cue, or one full user utterance |
| What is live | Nothing | Cue time, or whether to stop the user (Interrupt) |
| Observation after the event | $15\,\mathrm{s}$ padding baked into the file | $15\,\mathrm{s}$ after cue / user clip drains |
| $t=0$ | First streamed sample (not connect) | Same |
| Model recording | `output.wav` on that clock | Same, plus stereo `conversation.wav` |

---

## C. Direct replay (listen-side pipeline)

**Role.** The model is the listener. The user side is a single, fully known clip. We stream that clip in real time and record the model on the **same clock**. Nothing is decided live: no cue injection, no branching.

**Why this protocol.** Wait, Backchannel, Ignore, and Take turn test whether the model *withholds*, *gives a short continuer*, *ignores a distractor*, or *enters after the user*. The user audio is fixed in advance; a live simulator would only add variance.

### C.1 Clip construction

`input.wav` is 16 kHz mono PCM16 as prepared, then resampled per model at send time. Content starts at $t=0$. A fixed $15\,\mathrm{s}$ of trailing digital silence (`padding_time`) is baked into the file.

| Quantity | Definition |
|---|---|
| `clip_start` / `clip_end` | Addressed user speech (before padding) |
| `padding_time` | Always $15.0\,\mathrm{s}$ |
| `input_duration` $=$ `padding_end` | Content + padding. This **is** the observation window |

There is no extra post-input grace. Once the last input sample (including padding) has been sent, the receive task is cancelled. Speech the model would have continued past that cutoff is out of scope by definition — the same bound that open-source file-length models inherently have.

**Per-action content (before padding).**

| Action | What is in `input.wav` | Immediate interval lives in |
|---|---|---|
| Wait | User clip, optionally with an annotated pause | Pause span, or the whole clip if there is no pause |
| Backchannel | User’s ongoing turn with a human reference BC point (`bc_start` / `bc_end`) | Whole user speech; no window around the human point |
| Ignore | Addressed user speech plus overlapping background or other-directed speech | Distractor span |
| Take turn | User turn T1 | Windows relative to T1 end |

### C.2 Session loop

One sample $=$ one session. Handshake (WebSocket / local server) happens **before** $t=0$. Connection events are logged with negative timestamps and do not shift the scoring clock.

1. Connect and send the session config (system prompt, voice, provider VAD). No explicit `response.create` on listen-side APIs: turn-taking is the system under test.
2. At the first input chunk, freeze a single monotonic origin. Recorder, event log, and send-deadline pacing share that origin.
3. Stream `input.wav` in model-native chunks (e.g. 20 ms for GPT/Grok, 80 ms for Moshi). Each chunk $n$ is sent at absolute deadline $t_0 + n\cdot\mathrm{chunk}$; sleep is to that deadline, not a relative `sleep(chunk)` after send, so encoding/network overhead cannot accumulate into replay drift.
4. Write `output.wav` at wall-clock pace on the same origin: silence wherever the model produced no audio. Incoming API audio that arrives faster than realtime is queued and played at $1\times$; it is not written as soon as it lands.
5. Hard stop at `input_duration`. Transport errors fail the sample; they are not scored as silence.

Provider turn-detection (`semantic_vad` / `server_vad`) only decides when that API *starts a response*. It is not the scored onset. Scored `(on, off)` come from the recorder (Section C.3).

If the provider signals that an in-progress response was cancelled because the replayed user started speaking (`speech_started` while a response is active), the recorder drops queued-but-unwritten model audio and mutes until new audio arrives. The **input** stream is never truncated: the file keeps playing on its fixed schedule.

### C.3 Speech segments on the shared clock

`model_speech_segments` are `(onset, offset)` on the output timeline. Three mechanisms, one field:

| Mechanism | Used when | How |
|---|---|---|
| Audio presence (default) | Turn-based realtime APIs | A segment is open while model PCM is in the recorder queue |
| Native speaking flag | Continuous duplex (Moshi-style) | `set_speaking(True/False)` from the model’s own token / phase stream; these models emit audio with no digital silence |
| Windowed RMS | Lychee-FD | Recomputed from `output.wav` after the session (50 ms windows, RMS 0.01, merge gaps $\le 1.0\,\mathrm{s}$, drop blips $< 0.15\,\mathrm{s}$) |

Word-level ASR (`whisper-1`) is aligned to these segments. Native text, when the client exposes it on the same clock, is used instead of Whisper for models whose `output.wav` is mostly padding.

### C.4 Artifacts

Per sample: `output.wav`, `inference_info.json` (`instance` + `inference`), `events.jsonl`. Scoring reads `model_speech_segments`, per-segment transcripts, and `inference.input_duration_s`. Direct replay does not write a two-channel conversation mix: `input.wav` is already the user side.

---

## D. Simulator (speak-side pipeline)

The simulator is a tick-loop dual-clock design (audio time vs wall-clock time), following $\tau^2$-bench’s interaction loop, **except** that the simulated user never speaks via LLM+TTS. For cross-model consistency we replay a fixed, pre-recorded, WER- and speaker-checked clip. The only live decision is *when* to start (cue injection) or *when to stop* (Interrupt truncate).

Native codec frame size is **not** the simulated user’s reaction time. Every system under test is polled at the same wall-clock cadence.

### D.0 Shared apparatus

**Growable input source.** Unlike direct replay’s static `input.wav`, the total input length is not known at $t=0$. A dynamic PCM (or token) source is primed with the first clip. `pull(n)` always returns exactly $n$ bytes immediately: real queued audio if present, otherwise digital silence. Waiting for a trigger therefore sounds like a silent user, not a frozen stream. `append` queues the next clip; `truncate_pending` drops the unsent tail (Interrupt only); `finish` marks that nothing else will be appended.

**Decision model.** `gpt-5.4`, `temperature=0`, JSON object. Polled every $\mathrm{POLL}=1.0\,\mathrm{s}$ of wall time for **every** model. The loop is serial (`sleep(P)` then await the LLM), so realized cadence is $\approx 1.0 + 1.2\,\mathrm{s}$. That think time sits on the system-under-test’s wall clock; we do not have $\tau^2$-bench’s `conversation.item.truncate` as a universal primitive.

**Ear-aligned transcript (closed-source only).** Several realtime APIs generate audio $4$–$5.5\times$ faster than playback, and their text stream runs ahead of that audio. The trigger / stop LLM must decide from what the simulated listener has **heard**, not from what the socket has already produced. Released characters $= r_{\mathrm{model}}\times$ played speech seconds, cut on a word boundary, monotone (never un-said). Rates are per-voice medians (e.g. GPT $15.9$, Grok $16.6$, Qwen $19.7$, Gemini $17.3$ chars/s). Open-source clients are frame-synchronous and are not gated.

**Playback-layer barge-in flush (closed-source cue path).** When the provider’s own VAD reports that the injected cue has started (`speech_started`) and there is audible or queued model speech, the recorder drops the unplayed backlog so the playback clock catches up to generation. From that instant, “is the model still speaking” is the model’s post-cue decision, not buffer drainage. We do **not** send `response.cancel` (that would decide the yield for the model). `conversation.item.truncate` is sent only where the protocol supports it (OpenAI); xAI / DashScope get client-side discard only; Gemini truncates server-side itself.

**Shared $t=0$.** Same rule as direct replay: first streamed sample, not connect. `output.wav` is model audio only. A stereo `conversation.wav` is built afterwards (ch0 $=$ simulator clips at their true offsets, ch1 $=$ `output.wav`) so a reviewer can hear both sides on one clock.

### D.1 Cue injection — Stop, Hold floor, Recover

**Layout.** `lead_in.wav` + `cue.wav` + `metadata.json`. Same physical layout for all three actions. Cue text / reference fields differ by action; they are normalized before the trigger LLM sees them. They are a **pattern** for when to cut in, not a script the model must match.

**Procedure.**

1. Stream `lead_in.wav` so the model starts an in-progress turn. Do not poll and do not inject during lead-in.
2. After lead-in ends, wait until the model has produced at least $1.0\,\mathrm{s}$ of *post-lead-in* speech (speech that overlapped the lead-in does not count).
3. Poll the decision LLM: “inject the overlapping clip NOW?” Preference order: mid-clause; a locus whose *timing/type* resembles the human reference; sentence-final is allowed, never refused solely because the text ends in `.?!`. If the reply already has intelligible content but has diverged from the reference, inject now rather than waiting for a lexical match.
4. Caps, both measured **after lead-in** (not from stream $t=0$):
   - $T_{\mathrm{onset}}=5\,\mathrm{s}$ of wall time with **no** post-lead-in speech $\rightarrow$ force-inject (`last_reason=onset_timeout`).
   - $T_{\mathrm{search}}=8\,\mathrm{s}$ of wall time from the first post-lead-in onset $\rightarrow$ force-inject whether the model is still talking or already silent (`last_reason=search_timeout`).
5. On trigger (LLM or force), `append(cue.wav)` onto the live source. The remaining lead-in tail is not pre-mixed.
6. When the source has drained the cue, observe $15\,\mathrm{s}$ of trailing silence (`POST_CUE_OBSERVATION_S`), matching direct-replay padding. Then stop the recorder on elapsed wall-clock, not on lead-in length.

**What is live vs fixed.** Cue audio is fixed. Only injection time is live. Force-injected trials are `forced=true`. They stay in the Immediate headline if the cue still overlapped post-lead-in speech (`cue_class=speaking`); `immediate_unforced` is a diagnostic only.

**Action-specific trigger wording.** Stop: the clip is a user interruption meant to make the model yield. Recover: the same kind of interruption, but we will score adaptation after it ends — inject during the reply, not after it is already over. Hold floor: the clip is **not** a takeover (backchannel, background, or other-directed); inject while the model is still talking, or the trial is invalid.

**Timestamps exposed to scorers** (all from the `source="cue"` span):

| Action | Keys | Meaning |
|---|---|---|
| Stop | `t_cue_start` / `t_user_int_start` | Cue onset |
| Hold floor | `t_cue_start`, `t_cue_end` | Cue span |
| Recover | `t_interruption_start`, `t_interruption_end` | Same cue span |

### D.2 Barge-in — Interrupt (one continuous user utterance)

Roles are reversed: the simulator holds the floor; the model may cut in. There is no lead-in/cue pair.

**Layout on disk.** `user_speech.wav` (prefix up to the designed point) + `branch_ignored.wav` (the **full** utterance, `transcript_user_full`) + `branch_acknowledged.wav` + `metadata.json`. Playback uses the full clip (`branch_ignored.wav`). The two `branch_*.wav` files are **not** played in the current protocol; they remain in the sample dir as the original two-outcome recording. Dataset `branch_label` (what the human conversation did) is never used — the live truncate is based on the model under test.

**Procedure.**

1. From $t=0$, stream the full user utterance continuously. Authored `t_target` is the designed interrupt point and is **Timing only**. The clip is never split at `t_target` and is never replayed.
2. Poll the same decision LLM **only while** the model has been continuously speaking for $\ge 1.0\,\mathrm{s}$: “is the model trying to TAKE THE TURN (user should stop NOW)?”
   - `stop=true`: any intentional take-turn speech (opinion, question, correction, offer, …). Off-topic still counts. Do **not** require a match to `reference_interruption_text` or the condition label.
   - `stop=false` only for greeting / backchannel / filler / empty transcript. When in doubt between take-turn and continuer, `stop=true`.
3. **Acknowledged.** On `stop=true`, drop the unsent user-speech tail immediately (`truncate_pending`). Nothing else is appended. `t_model_onset_detected` is the wall time of that decision. This is **not** Interrupt Immediate success — it only stops the simulated user so the session can continue.
4. **Ignored.** If user-clip duration $+2.0\,\mathrm{s}$ grace elapses with no LLM stop, leave the full utterance in place. Filler-only model speech correctly lands here.
5. When the source is finished and drained, observe $15\,\mathrm{s}$ (`POST_BRANCH_OBSERVATION_S`).

**Scoring clock.** Interrupt Immediate uses $t_{\mathrm{user\_end}}=$ `inference.input_duration_s` (the wav that was actually sent — the full utterance), **not** live truncation (`simulator` `user_speech.end` / `t_model_onset_detected`) and **not** authored `t_target`. $\Delta=$ first in-window onset $-$ `t_target`. Cache key `interrupt_window = input_duration_minus_1`.

### D.3 Why not direct replay here

Stop / Hold floor / Recover require a cue **while the model already holds the floor**; the injection time depends on that model’s own reply, so it cannot be pre-baked into `input.wav`. Interrupt requires a live decision to stop the user when the model takes the turn; a fully pre-mixed user clip would keep talking over a successful barge-in.

### D.4 Artifacts

Same trio as direct replay, plus `conversation.wav`, merged `conversation` / `conversation_text`, and `inference.trigger` (cue path) or `inference.monitor` (Interrupt: `outcome` $\in$ `{acknowledged, ignored}`). Scoring reads the timestamp keys in D.1–D.2 and the shared `model_speech_segments`.

---

## E. Action metrics

Immediate / timing equations, windows, and fail tags are collected in **Section H**. This section is the per-action definition plus Subsequent wording. Judge prompt strings are **Section I**.

### E.1 Wait — $N=600$ (direct replay)

**Definition.** When the model should wait while listening, it should withhold a substantive response during the designated waiting interval and, when a subsequent response is called for, respond in a way that appropriately reflects the user’s continuing contribution.

| Condition | Data dir | $n$ | Immediate interval |
|---|---|---:|---|
| Inter-sentential pause | `inter_sentential` | 200 | `[pause_start, pause_end]` |
| Intra-sentential pause | `intra_sentential` | 200 | `[pause_start, pause_end]` |
| No pause | `floor_holding` | 200 | whole user speech `[clip_start, clip_end]` |

**Immediate — silence rate ($1-$ takeover).** Pass if no *substantive* model turn **overlaps** the interval. Overlap, not onset: speech that started before the interval and continued into it is a takeover. A $\le 3$-word backchannel is not a takeover. No-pause has **no** take-turn-with-overlap grace. Trailing padding after `clip_end` is outside Immediate.

Diagnostic only (not in the headline): pause-onset takeover — a new substantive onset *started* inside the pause. Do not average this with no-pause Immediate.

**Subsequent — continuation uptake.** Conditional on Immediate. Denominator = Immediate $=1$. Numerator = later response that is relevant and uses some user content from during or after the wait. No later response or an inappropriate later response $\rightarrow S=0$. Immediate fail $\rightarrow$ N/A. Same judge question for all three conditions; only the condition prompt changes.

> Is the model’s response relevant, and does it include *some* information from the user’s continuing contribution during or after the wait interval? Complete uptake is not required.

Judge sees: condition prompt, user speech during or after the wait, model response.

---

### E.2 Backchannel — $N=400$ (direct replay)

**Definition.** When the user still holds the floor, the model should give brief feedback of the required function without taking over, and any later response should fit that feedback.

| Condition | $n$ | Required function |
|---|---:|---|
| Affirmative | 200 | Acknowledgement / understanding / agreement |
| Uncertain | 200 | Hesitation / confusion / need for clarification |

**Immediate.** No window around the human `bc_start`. User speech is `[clip_start, clip_end]`.

1. **Candidate detection** (automatic). A segment of $\le 3$ words whose onset falls in that interval. Type is ignored. The first such event is the candidate. This is a detector, **not** a score: it over-fires on the truncated openings of full turns, so it is reported only as the `rule` column of the decomposition in `IMMEDIATE.md`.
2. **Stage 1 — existence** (`gpt-5.4`, text; Section I.9). Rules on whether the candidate is a genuine, self-contained backchannel rather than the clipped start of a turn, unintelligible audio, or speech with no listener-feedback function.
3. **Stage 2 — function match** (`gpt-audio`; only listening judge). Scored only when stage 1 returned YES. Hears the candidate span cropped from `output.wav` ($\pm 2$ s) with its transcript withheld, and makes a forced binary choice between `affirmative` and `uncertain`. The condition label is **not** shown.

**Immediate $=1$ iff stage 1 confirms the backchannel and the stage 2 label matches the condition.** Subsequent consistency on this action remains the text `gpt-5.4` judge and is gated on stage 1 only, not on function match.

Diagnostic: $\Delta = t_{\mathrm{BC}} - t_{\mathrm{target}}$ vs the human reference. Not scored.

**Subsequent — feedback–response consistency.** Text LLM, conditional on stage 1 (a backchannel occurred), not on function match. No later model speech $\rightarrow S=0$. No condition prompt and no gold `bc_text`.

> Is the model’s later speech consistent with the understanding or stance displayed in its backchannel? Yes if it fits that backchannel, or is justified by what the user said afterwards.

Judge sees: user speech before and overlapping the BC; user speech after the BC and before the later response; the BC; later model speech.

---

### E.3 Ignore — $N=400$ (direct replay; all synthesized)

**Definition.** While listening, the model should not respond to input that is irrelevant to the ongoing interaction, and it should not let that input alter its later response to the addressed user.

| Condition | $n$ | Distractor timestamps |
|---|---:|---|
| Background speech | 200 | `bg_speech_*` $\rightarrow$ `[t_other_start, t_other_end]` |
| Other-directed speech | 200 | `side_speech_*` $\rightarrow$ `[t_other_start, t_other_end]` |

**Immediate — silence rate.** Same helper as Wait, on the distractor span. Overlap, not onset. No grace after `t_other_end`. Trailing addressed user speech after the distractor is outside Immediate. Old “respond after user end” is Take turn, not Ignore Immediate.

**Subsequent — irrelevant-input exclusion.** Conditional on Immediate. From `t_other_end` onward, **all** overlapping model speech is judged. Immediate fail $\rightarrow$ N/A. No speech after `t_other_end` $\rightarrow S=0$. Do **not** score whether the reply addresses the user — only whether it contains or is shaped by the ignored speech. No condition label and no gold `speaker_b_text`.

> Does the model’s reply stay free of the other speech? Yes if it neither contains that speech nor is influenced by it (no quoting, echoing, answering, or treating it as a fact or task).

---

### E.4 Take turn — $N=600$ (direct replay)

**Definition.** When the model should take a turn, it should begin a substantive response within the appropriate timing window and continue by addressing the user’s latest communicative intent, while respecting the information and constraints currently in effect.

$t=0$ is user-speech end (`t1_end` / `t_user_end`).

| Window | Relative to user end | Role |
|---|---|---|
| Observation | $[-2.0,\,5.0]\,\mathrm{s}$ | Where we look for takeovers. Not a success criterion. |
| Immediate | $[0,\,0.5]\,\mathrm{s}$ | Prompt after user end. |
| Delayed | $[0,\,3.0]\,\mathrm{s}$ | Longer gap still succeeds. |
| Overlap | $[-1.0,\,0.5]\,\mathrm{s}$ | May enter in the last $1.0\,\mathrm{s}$ of user speech; overlap is not required. |

200 samples per window (400 synthesized + 200 real).

**Immediate — timing-conditioned takeover.** Pass if **any** valid takeover onset (backchannel excluded) falls in the condition success window. It need not be the first takeover. Bounds inclusive.

**Subsequent — intent under user constraints.** Conditional on Immediate. Immediate fail $\rightarrow$ N/A. An in-window response that fails the intent check $\rightarrow S=0$.

> Does the model’s response address the user’s intent by respecting the information and constraints in the user’s speech?

---

### E.5 Stop — $N=200$ (simulator cue; all real; one condition)

**Definition.** When an intervention requires the model to yield, it should stop its ongoing speech and refrain from continuing or renewing its claim to the current turn while the intervention is still underway.

Condition: `direct_interruption` only. $t=0$ is cue onset. Cue span is `[t_cue_start, t_cue_end]`.

**Immediate — stop rate.** No time window. Pass if the model was speaking at interrupt onset **and** that turn ends afterwards. Still talking at recording end $=$ did not stop. Not speaking at the cue is scored `not_speaking_at_cue` on the raw check, but those trials (and `finished` / `no_reply`) are **out of the headline denominator**. Headline is speaking-at-cue; **forced allowed**. `never_stopped` stays a real zero.

**Diagnostic:** yield latency $= t_{\mathrm{yield}} - t_{\mathrm{cue\_start}}$. Slow stops still count as stops.

**Subsequent — yielding maintained.** Automatic from segments. Gated on Immediate: only scored after a first stop at or before user offset; else N/A. After that first stop, no new substantive onset before $\mathrm{user\ offset}-1.0\,\mathrm{s}$. Onset in the last $1.0\,\mathrm{s}$ is take-turn-with-overlap and is not a reclaim. A $\le 3$-word backchannel is not a reclaim. Speech after user offset is allowed.

---

### E.6 Hold floor — $N=891$ (simulator cue)

**Definition.** When the model already holds the floor and a non-takeover cue arrives, it should keep speaking through the cue and adapt (or preserve) the ongoing turn as that cue called for.

Inverse of Stop Immediate. $t=0$ is cue onset. All five conditions share the same hold window.

**Immediate — floor-holding rate.** Window $[t_{\mathrm{cue\_start}},\, t_{\mathrm{cue\_end}}+1.0]$. The extra second is a yielding tail: short continuer cues would otherwise be shorter than the allowed gap, so a full yield inside the raw cue would still pass. Pass if the longest silence in that window (including lead from cue onset and trail to window end) is $\le 0.50\,\mathrm{s}$. No speech in the window $=$ fail. Observed max gap is also a diagnostic. Headline among speaking-at-cue; **forced allowed**.

| Condition | $n$ | Required adjustment (shown to the Subsequent judge; exact string in I.4) |
|---|---:|---|
| `continue_current_response` | 200 | Continue the current response without unnecessary change. |
| `continue_but_shorten` | 131 | Shorten or condense the ongoing response. |
| `continue_with_simplification` | 200 | Simplify the language or content of the ongoing response. |
| `continue_with_background_speech` | 200 | Preserve the current content; do not treat background speech as addressed input. |
| `continue_with_other_directed_speech` | 160 | Preserve the current content; do not treat other-directed speech as addressed input. |

**Subsequent — continuation adaptation.** Gated on Immediate. Split at $t_{\mathrm{cue\_end}}$. N/A if Immediate failed or there is no speech after cue end. The judge **is** shown the required adjustment because real cues are often “Yes” / “Hmm” / inaudible. Gold `reference_after_text` is not shown.

> Relative to what the model was saying before and during the cue, does its speech after the cue ended meet this condition’s requirement (keep going, shorten, simplify, or ignore a distractor)?

---

### E.7 Recover — $N=1200$ (simulator cue)

**Definition.** After an interruption, the model should re-enter within the recovery window and adapt its next contribution to what the interruption required.

$t=0$ is interruption offset (`t_interruption_end`). All six conditions share one window.

| Window | Relative to interruption end | Role |
|---|---|---|
| Observation $=$ Immediate success | $[-2.0,\,5.0]\,\mathrm{s}$ | Matches take-turn observation; Recover is not a take-turn test. |

200 samples per condition: enrich, shorten, repeat, shift topic, follow-up, correct.

**Immediate — take-over rate.** Pass if any valid takeover onset (backchannel excluded) falls in that window. Headline among speaking-at-cue; **forced allowed**. Cue in silence or after the model already finished is coverage (Take turn, not Recover).

**Subsequent — recovery adaptation.** Gated on Immediate; else N/A. Judge the in-window response from the interrupted contribution and the user interruption only. **No** condition label, condition prompt, or `reference_recovery_text`. Conditions are for aggregation only.

> Does the model’s response in the observation window make the adjustment this interruption required? No if it ignores the interruption, adjusts incorrectly, or resumes obsolete content.

---

### E.8 Interrupt — $N=400$ (simulator barge-in; all synthesized)

**Definition.** When the user is still speaking and an intervention is required, the model should enter before that contribution ends and do the work the condition asked for, not merely announce an interruption.

$t=0$ is user onset (`t_user_start`). The Immediate deadline is the **streamed** user-clip duration $t_{\mathrm{user\_end}}$ (`inference.input_duration_s` — the wav that was actually sent), **not** live truncation (`user_speech.end` / `t_model_onset_detected`) and **not** authored `t_target`. `t_target` is Timing only. Cache key `interrupt_window = input_duration_minus_1`. Both conditions share the same Immediate window.

| Window | Absolute | Role |
|---|---|---|
| Observation | `[t_user_start, t_user_end]` | Played user utterance. |
| Success | `[t_user_start, t_user_end $-$ 1.0]` | Must barge in before the last $1.0\,\mathrm{s}$. |

| Condition | $n$ | What the interruption must do |
|---|---:|---|
| Help | 200 | Offer the specific assistance or constraint the situation requires. |
| Safety | 200 | Flag the specific safety risk and say why it is dangerous or what to do instead. |

**Immediate — timing-conditioned interrupt.** Pass if any valid takeover onset falls in the success window. Onset in the last $1.0\,\mathrm{s}$ of the played clip, or after $t_{\mathrm{user\_end}}$, is take-turn-with-overlap / late — Immediate fail. Headline is all scored Interrupt samples (not a cue-into-speech action).

Diagnostic: $\Delta =$ first in-window onset $-$ `t_target` (falls back to $t_{\mathrm{user\_end}}$ if `t_target` is missing). Negative $=$ before the designed point. Not the headline.

**Subsequent — intervention fulfilment.** Gated on Immediate; else N/A. Shared judge question plus a per-condition prompt (why this user must be interrupted). Do **not** show `reference_interruption_text`.

> Did this interruption do the work this condition required, rather than merely announcing that an interruption is happening?

---

## F. Subsequent gating (quick reference)

| Action | Immediate headline | Subsequent | Gated on Immediate? | N/A when |
|---|---|---|---|---|
| Wait | Silence rate | Continuation uptake (LLM) | Yes | Immediate fail. No later response is $S=0$ |
| Backchannel | Screened BC $\land$ function match | Feedback–response consistency (LLM) | Yes (stage 1 confirmed a BC; not function match) | Stage 1 NO (no genuine BC). No later speech is $S=0$ |
| Ignore | Silence rate | Irrelevant-input exclusion (LLM) | Yes | Immediate fail. No speech after distractor is $S=0$ |
| Take turn | In-window takeover | Intent under constraints (LLM) | Yes | Immediate fail |
| Stop | Stop rate | Yielding maintained (auto) | Yes | No first stop during the cue |
| Hold floor | Floor-holding rate | Continuation adaptation (LLM) | Yes | Immediate fail, or no speech after cue end |
| Recover | Take-over rate | Recovery adaptation (LLM) | Yes | No in-window takeover |
| Interrupt | In-window barge-in | Intervention fulfilment (LLM) | Yes | No in-window interrupt |

---

## G. Constants

| Name | Value | Used by |
|---|---|---|
| Backchannel word cap | 3 | All actions that distinguish BC vs takeover |
| Overlap lookback | $1.0\,\mathrm{s}$ | Take turn (overlap), Interrupt, Stop Subsequent |
| Take-turn observation | $[-2,\,5]\,\mathrm{s}$ vs user end | Take turn |
| Recover window | $[-2,\,5]\,\mathrm{s}$ vs interruption end | Recover |
| Hold-floor yielding tail | $1.0\,\mathrm{s}$ after cue end | Hold floor Immediate |
| Hold-floor max gap | $0.50\,\mathrm{s}$ | Hold floor Immediate |
| Direct-replay padding | $15\,\mathrm{s}$ | Wait, Backchannel, Ignore, Take turn |
| Post-cue / post-utterance observation | $15\,\mathrm{s}$ | Simulator (after cue or user clip drains) |
| Cue search $T_{\mathrm{onset}}$ / $T_{\mathrm{search}}$ | $5\,\mathrm{s}$ / $8\,\mathrm{s}$ after lead-in | Stop, Hold floor, Recover |
| Simulator poll interval | $1.0\,\mathrm{s}$ | All simulator actions |
| Min post-lead-in / min speaking before poll | $1.0\,\mathrm{s}$ | Cue injection and Interrupt |
| Interrupt success tail | $1.0\,\mathrm{s}$ before $t_{\mathrm{user\_end}}$ | Interrupt Immediate (`input_duration_minus_1`) |
| Interrupt grace after user clip | $2.0\,\mathrm{s}$ | Interrupt ignored outcome only (not Immediate) |
| Never-stopped epsilon | $0.05\,\mathrm{s}$ | Stop Immediate (`t_yield \ge t_{rec}-0.05`) |

---

## H. Immediate and timing metrics (as implemented)

Source: `evaluation/scoring/*.py`. Immediate is $0/1$ from `model_speech_segments` `(on, off)` on the shared clock, plus per-segment ASR when a backchannel vs takeover decision is required. Bounds are **inclusive** unless noted. Overlap is half-open in the usual interval sense: $on < t_{hi}$ and $off > t_{lo}$.

Headline Immediate: listen actions and Interrupt = all scored samples. Stop / Hold floor / Recover = speaking-at-cue (`cue_class=speaking`); **forced allowed**. `immediate_unforced` is a diagnostic. Cue in silence or after the model already finished is coverage.

### H.1 Shared predicates

| Symbol | Definition |
|---|---|
| $S$ | Model speech segments $\{(on_i, off_i)\}$ |
| $w(s)$ | Word count of the **whole-segment** ASR (`str.split()`). Empty ASR $=0$. |
| $\mathrm{BC}(s)$ | $\mathrm{true}$ if $w(s)\le 3$; $\mathrm{false}$ if transcript usable and $w(s)>3$; $\mathrm{unknown}$ if transcript missing or length-mismatched |
| $\mathrm{overlap}(s,I)$ | $on < I_{hi} \land off > I_{lo}$ |
| $\mathrm{TakeoverOnsets}$ | $\{on : \mathrm{BC}(s)\ne\mathrm{true}\}$ (unknown counts as takeover) |
| $\mathrm{SubstantiveOverlap}$ | $\mathrm{BC}(s)$ is $\mathrm{unknown}$ or $\mathrm{false}$ (Wait / Ignore silence) |
| $\mathrm{cue\_class}$ | Stop / Hold / Recover: `speaking` / `no_reply` / `finished` (`cue_landing.py`). Headline $=$ `speaking` |

Word count is never taken on the slice that happens to fall inside a short pause.

### H.2 Master Immediate table

| Action | Check | Score | Scored interval | Pass | Fail tags | Diagnostic (not scored) |
|---|---|---|---|---|---|---|
| Wait | `silence` | $1-$ takeover | condition interval (below) | no $\mathrm{SubstantiveOverlap}$ | `takeover_during_wait` | pause-onset takeover |
| Ignore | `silence` | $1-$ takeover | $[t_{\mathrm{other\_start}}, t_{\mathrm{other\_end}}]$ | same helper as Wait | `takeover_during_wait` | — |
| Backchannel | candidate $+$ stage 1 | gate | $[t_{\mathrm{clip\_start}}, t_{\mathrm{clip\_end}}]$ | first onset with $\mathrm{BC}=\mathrm{true}$, then `gpt-5.4` returns YES | `no_backchannel`, stage-1 NO | $\Delta=t_{\mathrm{BC}}-t_{\mathrm{target}}$ |
| Backchannel | `function_match` | $0/1$ | candidate span $\pm 2$ s of `output.wav` (`gpt-audio`) | label $=$ condition | mismatch | — |
| Take turn | `timing_conditioned_takeover` | $0/1$ | per-condition window vs $t_{\mathrm{end}}$ | any $\mathrm{TakeoverOnset}$ in window | `no_takeover`, `no_takeover_in_success_window` | first in-window $\Delta=on-t_{\mathrm{end}}$ |
| Recover | `timing_conditioned_takeover` | $0/1$ | $[-2.0, 5.0]$ vs $t_{\mathrm{int\_end}}$ | same | same | same |
| Interrupt | `timing_conditioned_interrupt` | $0/1$ | $[t_{\mathrm{user\_start}}, t_{\mathrm{user\_end}}-1.0]$ | any $\mathrm{TakeoverOnset}$ in window | same | $\Delta=on-t_{\mathrm{target}}$ |
| Stop | `stop_rate` | $0/1$ | after $t_{\mathrm{cue\_start}}$, no $t_{hi}$ | speaking at cue **and** that turn ends | `not_speaking_at_cue`, `never_stopped`, `missing_cue` | $L=t_{\mathrm{yield}}-t_{\mathrm{cue\_start}}$ |
| Hold floor | `floor_holding_rate` | $0/1$ | $[t_{\mathrm{cue\_start}}, t_{\mathrm{cue\_end}}+1.0]$ | $\mathrm{maxgap}\le 0.50$ | `floor_released`, `missing_cue` | observed $\mathrm{maxgap}$ |

Stop / Hold floor / Recover: the table is the raw check. Headline Immediate keeps only `cue_class=speaking` (forced allowed). Interrupt is all scored samples.

### H.3 Windows (absolute or vs anchor)

$t=0$ is stream start unless the action re-anchors.

| Action | Condition | Anchor $t_0$ | Immediate window | Observation (if different) |
|---|---|---|---|---|
| Wait | `inter_sentential_pause` | stream | $[\texttt{pause\_start}, \texttt{pause\_end}]$ | — |
| Wait | `intra_sentential_pause` | stream | $[\texttt{pause\_start}, \texttt{pause\_end}]$ | — |
| Wait | `no_pause` | stream | $[\texttt{clip\_start}, \texttt{clip\_end}]$ | — |
| Ignore | background | stream | $[\texttt{bg\_speech\_start\_s}, \texttt{bg\_speech\_end\_s}]$ | — |
| Ignore | other-directed | stream | $[\texttt{side\_speech\_start\_s}, \texttt{side\_speech\_end\_s}]$ | — |
| Backchannel | both | stream | $[\texttt{clip\_start}, \texttt{clip\_end}]$ | — |
| Take turn | `take_turn_immediately` | $t_{\mathrm{end}}=$ `t1_end` | $[t_{\mathrm{end}}+0.0,\, t_{\mathrm{end}}+0.5]$ | $[t_{\mathrm{end}}-2.0,\, t_{\mathrm{end}}+5.0]$ |
| Take turn | `take_turn_with_delay` | $t_{\mathrm{end}}$ | $[t_{\mathrm{end}}+0.0,\, t_{\mathrm{end}}+3.0]$ | same observation |
| Take turn | `take_turn_with_overlap` | $t_{\mathrm{end}}$ | $[t_{\mathrm{end}}-1.0,\, t_{\mathrm{end}}+0.5]$ | same observation |
| Recover | all six | $t_{\mathrm{int\_end}}$ | $[t_{\mathrm{int\_end}}-2.0,\, t_{\mathrm{int\_end}}+5.0]$ | $=$ Immediate |
| Interrupt | help / safety | stream | $[t_{\mathrm{user\_start}},\, t_{\mathrm{user\_end}}-1.0]$ | $[t_{\mathrm{user\_start}}, t_{\mathrm{user\_end}}]$ |
| Stop | `direct_interruption` | cue onset | $[t_{\mathrm{cue\_start}}, +\infty)$ (no $t_{hi}$) | cue Subsequent: $[t_{\mathrm{cue\_start}}, t_{\mathrm{cue\_end}}]$ |
| Hold floor | all five | cue onset | $[t_{\mathrm{cue\_start}},\, t_{\mathrm{cue\_end}}+1.0]$ | Subsequent split at $t_{\mathrm{cue\_end}}$ |

Take-turn / Recover / Interrupt: **any** in-window takeover onset passes; it need not be the first takeover on the recording. Window bounds inclusive.

No extra grace after Ignore $t_{\mathrm{other\_end}}$ or Wait `clip_end`. Trailing $15\,\mathrm{s}$ padding is outside those Immediate intervals. Interrupt Immediate uses streamed $t_{\mathrm{user\_end}}$ (`input_duration_s`), not live truncation and not authored `t_target`.

### H.4 Per-action score (code)

**Wait / Ignore silence** (`score_immediate_silence`)

$$
I = 0 \iff \exists s\in S:\ \mathrm{overlap}(s,I_{\mathrm{imm}})\ \land\ \mathrm{SubstantiveOverlap}(s),\quad\text{else } I=1.
$$

Wait diagnostic only (`score_pause_onset_takeover`): $1$ iff a **new** substantive onset starts inside the pause. Do not average with no-pause Immediate.

**Backchannel rate** (`score_immediate_backchannel`)

$$
I_{\mathrm{rate}} = 1 \iff \exists s:\ t_{\mathrm{clip\_start}} \le on \le t_{\mathrm{clip\_end}}\ \land\ \mathrm{BC}(s)=\mathrm{true}.
$$

First such event is the BC candidate. $\Delta = t_{\mathrm{BC}}-t_{\mathrm{target}}$ vs human `bc_start`. $I_{\mathrm{rate}}$ is the detector only; the reported Immediate adds the two judging stages of Section I.9, both conditional on $I_{\mathrm{rate}}=1$:

$$
I_{\mathrm{backchannel}} = 1 \iff I_{\mathrm{rate}}=1\ \land\ \mathrm{screen}=\mathrm{YES}\ \land\ \mathrm{function}=\mathrm{condition}.
$$

**Take turn / Recover** (`score_immediate_takeover`)

$$
I = 1 \iff \exists\, on\in\mathrm{TakeoverOnsets}:\ lo \le on-t_0 \le hi.
$$

| | $t_0$ | $(lo, hi)$ |
|---|---|---|
| Take turn immediate | `t1_end` | $(0.0, 0.5)$ |
| Take turn delay | `t1_end` | $(0.0, 3.0)$ |
| Take turn overlap | `t1_end` | $(-1.0, 0.5)$ |
| Recover (all) | `t_interruption_end` | $(-2.0, 5.0)$ |

Returned $\Delta$ is the **first in-window** $on-t_0$, or the first takeover’s $\Delta$ on fail.

**Interrupt** (`score_immediate_interrupt`)

$$
I = 1 \iff \exists\, on\in\mathrm{TakeoverOnsets}:\ t_{\mathrm{user\_start}} \le on \le t_{\mathrm{user\_end}}-1.0.
$$

$t_{\mathrm{user\_end}}=$ `inference.input_duration_s`. $\Delta = on^{\star} - t_{\mathrm{target}}$ for the first in-window onset (falls back to $t_{\mathrm{user\_end}}$ if `t_target` is missing). Negative $=$ before the designed point. Onset in $(t_{\mathrm{user\_end}}-1.0, t_{\mathrm{user\_end}}]$ is take-turn-with-overlap, not Interrupt success.

**Stop Immediate** (`score_stop_rate`)

$$
t_{\mathrm{yield}} = \min\{off : on \le t_{\mathrm{cue\_start}} \le off\}.
$$

| | $I$ | Tag |
|---|---|---|
| no containing segment | $0$ | `not_speaking_at_cue` |
| $t_{\mathrm{yield}} \ge t_{\mathrm{rec}}-0.05$ | $0$ | `never_stopped` |
| else | $1$ | — |

$L = t_{\mathrm{yield}}-t_{\mathrm{cue\_start}}$. No success deadline: a slow stop is still a stop.

**Hold floor Immediate** (`score_immediate_hold_floor`)

Window $W=[t_{\mathrm{cue\_start}}, t_{\mathrm{cue\_end}}+1.0]$. Clip $S$ to $W$. Then

$$
\mathrm{maxgap} =
\begin{cases}
|W| & \text{no speech in }W \\
\max\!\big(s_1-W_{lo},\ \max_i(s_{i+1,lo}-s_{i,hi}),\ W_{hi}-s_{n,hi}\big) & \text{otherwise.}
\end{cases}
$$

$$
I = 1 \iff \mathrm{maxgap} \le 0.50.
$$

No speech $\Rightarrow$ $\mathrm{maxgap}=|W|>0.50$ $\Rightarrow$ fail. The $1.0\,\mathrm{s}$ tail is so a short continuer cue cannot be shorter than the allowed gap.

### H.5 Stop Subsequent (automatic timing; no LLM)

Gated on Immediate. Cue $C=[t_{\mathrm{cue\_start}}, t_{\mathrm{cue\_end}}]$.

| | Subsequent |
|---|---|
| no $t_{\mathrm{yield}}$ | N/A (`no_first_stop`) |
| $t_{\mathrm{yield}} > t_{\mathrm{cue\_end}}$ | N/A (`first_stop_after_cue`) |
| new onset $on\in(t_{\mathrm{yield}},\, t_{\mathrm{cue\_end}}-1.0]$ and $\mathrm{BC}\ne\mathrm{true}$ | $0$ (`renewed_speech_during_cue`) |
| else | $1$ |

Onset in the last $1.0\,\mathrm{s}$ of the cue is take-turn-with-overlap, not a reclaim. Speech after $t_{\mathrm{cue\_end}}$ is allowed. $\mathrm{BC}=\mathrm{true}$ is not a reclaim.

### H.6 Subsequent applicability (timing gates only)

LLM wording is Section I. Direct Replay: Immediate fail is N/A; no later response among Immediate $=1$ is $S=0$. Simulator rows are extra N/A gates. Stop / Hold floor / Recover Subsequent is further restricted to speaking-at-cue (same as Immediate headline; forced allowed).

| Action | Subsequent scored when | N/A |
|---|---|---|
| Wait | Immediate $=1$ | Immediate fail. No later response is $S=0$ |
| Ignore | Immediate $=1$ | Immediate fail. No speech after $t_{\mathrm{other\_end}}$ is $S=0$ |
| Take turn | Immediate $=1$ | `immediate_fail` |
| Backchannel | Immediate rate $=1$ | Immediate fail (no BC). No later speech is $S=0$ |
| Hold floor | Immediate $=1$ and speech after $t_{\mathrm{cue\_end}}$ | `immediate_fail`, `no_speech_after_cue` |
| Recover | Immediate $=1$ | `immediate_fail` |
| Interrupt | Immediate $=1$ | `immediate_fail` |
| Stop | Section H.5 | Section H.5 |

---

## I. Judge prompts (as implemented)

Source: `evaluation/scoring/*.py` and `run_metrics.py`. There is **no** single shared GPT-5.4 system prompt. Each action has its own `JUDGE_SYSTEM_PROMPT`. The shared contract is:

- Stem: `You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark.`
- Use **only** the given evidence. Ambiguous or insufficient $\rightarrow$ **NO**.
- Text output: `{"answer": "YES"|"NO", "reason": "<one short sentence>"}`.
- API: `gpt-5.4`, `temperature=0`, `response_format={"type": "json_object"}`. No `json_schema` object is passed. Score $1$ iff `answer` uppercased starts with `Y`.
- Stop has no LLM judge (Subsequent is automatic from segments).

Brace placeholders below are filled at runtime. Empty text uses the fallback shown in parentheses.

### I.1 Shared API wrapper

Text Subsequent (`call_text_judge`): system = that action’s `JUDGE_SYSTEM_PROMPT`; user = `build_judge_prompt(...)`.

Backchannel is judged by `bc_two_stage_judge.py`, not by the wrapper above. Stage 1 is an ordinary text call (`gpt-5.4-2026-03-05`, `temperature=0`). Stage 2 calls `gpt-audio` with `temperature=0`, `modalities=["text"]` and no `response_format` — that model rejects `json_object`, so the JSON shape is prompt-enforced and parsed leniently. Its user content is two parts: the text context block plus `{type: "input_audio", input_audio: {data: <cropped output.wav base64>, format: "wav"}}`. Full prompts in Section I.9.

### I.2 Wait — `wait.py`

**System**

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. You will be shown the wait condition, the user's speech during or after that wait interval, and the model's later response. Answer a single yes/no question using ONLY the evidence given. If the evidence is ambiguous or insufficient, answer NO. Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

**Condition prompts** (`CONDITION_PROMPTS`; this is the only Wait-side change across conditions)

```
inter_sentential_pause:
The wait interval is a pause between sentences. The user's continuing contribution is what they say after that pause (the next sentence and anything that follows until the model responds).

intra_sentential_pause:
The wait interval is a pause inside a sentence. The user's continuing contribution is what they say after that pause (the rest of the sentence and anything that follows until the model responds).

no_pause:
There is no annotated pause. The wait interval is the user's continuous turn. The user's continuing contribution is that turn (and anything said after it, if any) until the model responds.
```

**User message** (`build_judge_prompt`)

```
Action: wait
Condition: {condition_id}

Condition:
{CONDITION_PROMPTS[condition_id] or '(not specified)'}

User speech during or after the wait interval:
{user_during_or_after_wait or '(none)'}

Model response:
{model_response or '(model produced no speech)'}

Question: Is the model's response relevant, and does it include some information from the user's continuing contribution during or after the wait interval? Answer YES if the response is on-topic and uses at least some of that later or overlapping user content. Complete uptake of every detail is not required. Answer NO if the response is irrelevant, or if it ignores that continuing contribution.

Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

Fields: `no_pause` $\rightarrow$ `full_transcript`; inter/intra $\rightarrow$ `post_pause_text`. Model side: all model text.

### I.3 Take turn — `take_turn.py`

Same question for every condition. The user message carries `condition_id` only; there is no condition-specific paragraph.

**System**

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. You will be shown one user utterance and the model's spoken response, and asked a single yes/no question. Answer using ONLY the evidence given. If the evidence is ambiguous, insufficient, or the model produced no relevant speech, answer NO. Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

**User message** (`build_judge_prompt`)

```
Action: take_turn
Condition: {condition_id}

User speech:
{user_speech}

Model response:
{model_response or '(model produced no speech)'}

Question: Does the model's response address the user's intent by respecting the information and constraints in the user's speech? Answer YES if the response takes up what the user was asking or doing, and it uses (does not ignore or contradict) the facts, limits, and revisions in that speech. Answer NO if it misses the intent, or if it ignores or contradicts information or constraints from the user's speech.

Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

Fields: `t1_text` (else `full_transcript`); all model text.

### I.4 Hold floor (Maintain turn) — `hold_floor.py`

**System**

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. You will be shown what the model was saying before and during a cue, what it said after the cue ended, and what this condition required. Answer a single yes/no question using ONLY the evidence given. If the evidence is ambiguous or insufficient, answer NO. Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

**Required adjustment** (fourth column of `CONDITION_ROWS`; this is what the judge sees)

```
continue_with_background_speech:
Preserve the current content; do not treat background speech as addressed input.

continue_with_other_directed_speech:
Preserve the current content; do not treat other-directed speech as addressed input.

continue_current_response:
Continue the current response without unnecessary change.

continue_but_shorten:
Shorten or condense the ongoing response.

continue_with_simplification:
Simplify the language or content of the ongoing response.
```

**User message** (`build_judge_prompt`)

```
Action: hold_floor
Condition: {condition_id}

What this condition requires:
{required_adjustment or '(not specified)'}

Cue:
{cue or '(none)'}

Model speech before and during the cue:
{model_through_cue or '(none)'}

Model speech after the cue ended:
{model_after_cue or '(none)'}

Question: Relative to what the model was saying before and during the cue, does its speech after the cue ended meet this condition's requirement? Answer YES if it keeps the same thread and does what the condition asked (keep going, shorten, simplify, or ignore a distractor). Answer NO if it abandons the thread, takes up a cue that should be ignored, or fails to make the change the condition required.

Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

Fields: `cue_text` / `reference_interruption_text`; model text split at $t_{\mathrm{cue\_end}}$. No `reference_after_text`, no timestamps.

### I.5 Interrupt — `interrupt.py`

**System**

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. You will be shown why an interruption is needed, the user's speech, and the model's interruption, and asked a single yes/no question. Answer using ONLY the evidence given. If the evidence is ambiguous or insufficient, answer NO. Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

**Condition prompts** (`CONDITION_PROMPTS`)

```
interrupt_for_safety:
The user is describing a dangerous or hazardous behaviour. An interruption is needed to flag the specific safety risk and to say why it is dangerous or what to do instead.

interrupt_for_help:
The user is proceeding in a direction where timely help would matter. An interruption is needed to offer the specific assistance or constraint the current situation requires.
```

**User message** (`build_judge_prompt`)

```
Action: interrupt
Condition: {condition_id}

Condition (why an interruption is needed):
{CONDITION_PROMPTS[condition_id] or '(not specified)'}

User speech (what was underway):
{user_speech}

Model interruption:
{model_interruption or '(model produced no speech)'}

Question: Did this interruption do the work this condition required, rather than merely announcing that an interruption is happening? Answer YES if the model's speech addresses the obstacle or action described in the condition. Answer NO if it only announces the interruption, is generic, off-topic, or misses the issue.

Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

Fields: `transcript_user`; first in-window takeover text. Do not show `reference_interruption_text`.

### I.6 Ignore — `ignore.py`

No condition row.

**System**

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. You will be shown the user's addressed speech, other speech that should be ignored, and the model's speech (already restricted to after that other speech ended). Answer a single yes/no question using ONLY the evidence given. If the evidence is ambiguous or insufficient, answer NO. Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

**User message** (`build_judge_prompt`)

```
Action: ignore

User speech addressed to the model:
{addressed_user_speech}

Other speech (should be ignored):
{ignored_speech}

Model speech (after the other speech ended):
{model_response or '(none)'}

Question: Does the model's reply stay free of the other speech? Answer YES if it neither contains that other speech nor is influenced by it: no quoting, echoing, answering, or treating it as a fact or task. Answer NO if the reply includes the other speech or is shaped by it.

Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

Fields: `speaker_a_text`; `other_text`; model speech after $t_{\mathrm{other\_end}}$.

### I.7 Recover — `recover.py`

No condition id and no `required_adjustment` in the judge. All six conditions share this prompt.

**System**

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. You will be shown the interrupted contribution, the user's interruption, and the model's recovery in the observation window, and asked a single yes/no question. Answer using ONLY the evidence given. If the evidence is ambiguous or insufficient, answer NO. Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

**User message** (`build_judge_prompt`)

```
Action: recover

Interrupted contribution (what the model was saying):
{interrupted_contribution or '(not shown)'}

User interruption:
{interruption}

Model recovery (speech with onset in the observation window):
{model_recovery or '(model produced no speech)'}

Question: Does the model's response in the observation window make the adjustment this interruption required? Answer YES if it adapts as the user's interruption called for, given what the model was saying. Answer NO if it ignores the interruption, adjusts incorrectly, or resumes obsolete content as if the interruption had not happened.

Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

Fields: model text before interruption end; cue / `reference_interruption_text`; in-window model recovery. No `reference_recovery_text`.

### I.8 Backchannel Subsequent (text, GPT-5.4) — `backchannel.py`

No condition prompt and no gold `bc_text`.

**System** (`SUBSEQUENT_JUDGE_SYSTEM_PROMPT`)

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. You will be shown the user's speech before and overlapping the model's backchannel, the user's speech after that backchannel and before the model's later response, the backchannel itself, and the later model speech. Answer a single yes/no question using ONLY the evidence given. If the evidence is ambiguous or insufficient, answer NO. Return strict JSON only: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

**User message** (`build_judge_prompt`)

```
Action: backchannel

User speech before and overlapping the model backchannel:
{user_before_and_overlap or '(none)'}

User speech after the backchannel and before the model's later response:
{user_between or '(none)'}

Model backchannel:
{model_backchannel or '(none)'}

Model later speech:
{model_later or '(none)'}

Question: Is the model's later speech consistent with the understanding or stance displayed in its backchannel? Answer YES if later speech fits that backchannel, or is justified by what the user said afterwards. Answer NO if later speech contradicts the backchannel without such justification.

Reply with JSON: {"answer": "YES"|"NO", "reason": "<one short sentence>"}.
```

Fields: `before_text` + `bc_text`; `after_text`; detected BC text; model speech after the BC.

### I.9 Backchannel two-stage judging — `bc_two_stage_judge.py`

Backchannel Immediate is not the ≤3-word rule. The rule only nominates
candidates; a segment counts as a success only if **stage 1** confirms it is a
genuine backchannel **and** stage 2's label matches the condition. Stage 1 is
text-only (`gpt-5.4-2026-03-05`, `temperature=0`); stage 2 listens
(`gpt-audio`, `temperature=0`, `modalities=["text"]`) and runs only on segments
stage 1 confirmed. Neither stage is shown the condition label, the gold human
`bc_text`, or `bc_type_original`.

Both stages receive the same context block, built by `build_context`. It gives
the user's speaking interval and transcript, then every model segment in order
with the silence preceding each, with the candidate marked. The silence before
and after the candidate is stated explicitly because it is what separates a
self-contained backchannel from the truncated opening of a full turn:

```
Action: backchannel

The user speaks continuously from <start>s to <end>s.

User transcript:
<user transcript>

Model output segments, in order, with the silence before each one:
  [<on>s - <off>s] (start) <text>
  [<on>s - <off>s] (<gap>s silence before) <text>   <<< THE MARKED SEGMENT
  ...

The marked segment runs <on>s to <off>s, i.e. <d>s into the user's turn. The next model segment begins <g>s later.
  [if that gap is < 0.5s, instead: "...begins only <g>s later, so check whether the two are one continuous utterance."]
  [if no later segment: "No further model speech follows."]

Question: <stage question>

Reply with JSON: <stage reply shape>
```

**Stage 1 system** (`SCREEN_SYSTEM`) — the candidate's transcript **is** shown.

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. The user holds the floor for the whole interval. A detector has flagged one short model segment as a possible backchannel, and you must rule on whether it really is one. A backchannel is brief listener feedback produced while the other speaker keeps the floor -- an acknowledgement, a continuer, or a signal of trouble.
The model is expected to backchannel during the user's turn AND to answer properly later, so the mere existence of later model speech does NOT make the marked segment a turn attempt. What matters is whether the marked segment is self-contained. Treat it as the truncated opening of a full turn only when the following segment resumes the same utterance with no real pause, so that reading them together gives one continuous sentence.
Answer NO when the segment runs on into the next one as a single sentence, when it carries no listener-feedback function, or when it has no intelligible speech. Use ONLY the evidence given. Return strict JSON only: {"is_backchannel": "YES"|"NO", "reason": "<one short sentence>"}.
```

**Stage 1 question** (`SCREEN_QUESTION`) and reply shape

```
Is the marked segment a genuine backchannel? Answer YES if it is self-contained brief listener feedback that leaves the floor with the user, even when the model speaks again later. Answer NO if it runs straight on into the next segment as one continuous sentence, carries no feedback function, or contains no intelligible speech.

{"is_backchannel": "YES"|"NO", "reason": "<one short sentence>"}
```

**Stage 2 system** (`FUNCTION_SYSTEM`) — forced binary, no `unknown`.

```
You are a strict, consistent evaluator for a full-duplex spoken dialogue benchmark. The user holds the floor and the model produces one short backchannel inside the user's turn. Decide whether that backchannel signals an affirmative function (acknowledgement, understanding, or agreement) or an uncertain function (hesitation, confusion, or a need for clarification). Judge only the marked backchannel, not the model's later speech. Base the decision on how it sounds -- intonation, pitch contour, hesitation. This is a forced binary choice: you must answer affirmative or uncertain. There is no unknown option, so if the audio is ambiguous commit to whichever of the two the sound leans towards. Return strict JSON only: {"function": "affirmative"|"uncertain", "reason": "<one short sentence>"}.
```

**Stage 2 question** (`FUNCTION_QUESTION`) and reply shape

```
Does the marked backchannel signal affirmative or uncertain? Answer affirmative if it acknowledges, shows understanding, or agrees. Answer uncertain if it shows hesitation, confusion, or a need for clarification. You must pick one of the two.

{"function": "affirmative"|"uncertain", "reason": "<one short sentence>"}
```

In stage 2 the candidate's own transcript is **replaced** in the context block
by `<<< THE MARKED SEGMENT - listen to the attached audio for it`, so the label
cannot be read off the words; the surrounding segments stay visible as context.
The attachment is `output.wav` cropped to the candidate span padded by
`AUDIO_PAD_S = 2.0` s on each side, re-encoded as 16-bit PCM WAV and passed as
`input_audio`. `gpt-audio` accepts no `response_format`, so the JSON is only as
firm as the prompt and is parsed leniently; when the model stalls instead of
answering (e.g. "let me listen..."), its reply is appended to the conversation
and the question re-asked, which breaks the determinism that would otherwise
reproduce the stall at `temperature=0`.

Post-hoc match: `backchannel_affirmative` $\leftrightarrow$ `affirmative`;
`backchannel_uncertainty` $\leftrightarrow$ `uncertain`. Immediate $=1$ iff
stage 1 returned YES **and** the stage 2 label matches. Per-sample verdicts and
the rule/screened/function decomposition: `BC_TWO_STAGE.md`.

---

## J. Systems under test (checkpoints and API IDs)

Source: `evaluation/direct_replay/inference/model_registry.py` and each client’s `DEFAULT_MODEL`. One checkpoint / API ID per slug; no multi-revision sweep. Hugging Face repos are loaded by repo id (no pinned commit in the client). Closed-source IDs are whatever that API served during **2026-09-06 – 2026-09-13** (UTC). `grok-voice-latest` is a floating alias.

**Protocol.** DR = direct replay (Wait / Backchannel / Ignore / Take turn). Sim = simulator (Stop / Hold floor / Recover / Interrupt). Simulator production roster requires a live text signal for the trigger LLM; Lychee-FD has a cue client but is DR-only in the reported tables. VITA-1.5 / Voila / Mini-Omni2 are DR-only (no incremental live-text driver for cue injection).

### J.1 Closed-source realtime APIs

| Report slug | API model ID | Endpoint | Voice | Protocol |
|---|---|---|---|---|
| `gpt-realtime-2.1` | `gpt-realtime-2.1` | `wss://api.openai.com/v1/realtime` | `alloy` | DR + Sim |
| `gemini-3.1-flash-live` | `gemini-3.1-flash-live-preview` | Google GenAI, `vertexai=False` (API key, not Vertex) | — (no voice field) | DR + Sim |
| `qwen3.5-omni-flash-realtime` | `qwen3.5-omni-flash-realtime` | `wss://{DASHSCOPE_WS_HOST}/api-ws/v1/realtime` (intl / Singapore) | `Ethan` | DR + Sim |
| `grok-voice` | `grok-voice-latest` | `wss://api.x.ai/v1/realtime` | `eve` | DR + Sim |

Turn-taking is the provider VAD (GPT/Qwen `semantic_vad`; Grok `server_vad`; Gemini Live audio). We do not set temperature / top-$p$ on these APIs.

### J.2 Open-source checkpoints (Hugging Face)

Weights live under `opensource_models/<slug>/` (isolated venv + HF cache). Extra columns are additional repos or files the client **must** load besides the main checkpoint.

| Report slug | Main checkpoint | Extra weights / files | Protocol |
|---|---|---|---|
| `moshi` | `kyutai/moshiko-pytorch-bf16` | — | DR + Sim |
| `flm-audio` | `CofeAI/FLM-Audio` (same repo as Mimi and LM) | — | DR + Sim |
| `bayling-duplex` | `BayLing-Models/BayLing-Duplex` | — | DR + Sim |
| `raon-speechchat` | `KRAFTON/Raon-SpeechChat-9B` | — | DR + Sim |
| `minicpm-o-4.5` | `openbmb/MiniCPM-o-4_5` | official `assets/HT_ref_audio.wav` (same repo) as reference speaker | DR + Sim |
| `freeze-omni` | `VITA-MLLM/Freeze-Omni` | frozen LLM `Qwen/Qwen2-7B-Instruct` | DR + Sim |
| `lychee-fd` | `HIT-TMG/Lychee-FD` | Token2Wav vocoder from `stepfun-ai/Step-Audio-2-mini` | DR only |
| `vita-1.5` | `VITA-MLLM/VITA-1.5` | audio encoder and InternViT via that repo’s `config.json` (official 2-vLLM server, repo commit `35d064a`, **patched at the session boundary — see J.4**) | DR only |
| `mini-omni2` | `gpt-omni/mini-omni2` | — | DR only |
| `voila-autonomous` | `maitrix-org/Voila-autonomous-preview` | — | DR only |

### J.3 Judges and ASR (not systems under test)

| Role | Model ID |
|---|---|
| Subsequent text judge; simulator trigger / Interrupt stop LLM | OpenAI `gpt-5.4`, `temperature=0`, JSON object |
| Backchannel stage 1, backchannel existence (text) | OpenAI `gpt-5.4-2026-03-05`, `temperature=0` |
| Backchannel stage 2, function match (audio) | OpenAI `gpt-audio`, `temperature=0`, `modalities=["text"]` |
| Model-speech ASR (when native text is unused) | OpenAI `whisper-1` (`verbose_json`, word + segment timestamps) |

### J.4 The one modification to a system under test (VITA-1.5 session boundary)

VITA-1.5 is the only system whose serving code we changed. The patch clears the
server's two app-scoped TTS queues at the session hooks that already exist; it
touches no weights, no decoding parameter, no VAD threshold, and no prompt.
Applied by `pod_vita_patch_server.py` to `web_demo/server.py` of the official
repo at commit `35d064a`, against which all three of its anchors match exactly
once (the script aborts rather than guess if the file differs, and re-parses the
result before keeping it).

**What was wrong.** The official server holds `TTS_QUEUE` and
`TTS_OUTPUT_QUEUE` in `app.config`, i.e. process-wide, shared by every session
and both LLM workers. `handle_reset_state` clears the conversation *text*
history only; `handle_recording_started` calls `interrupt()`/`reset()` on the
*per-session* `GlobalParams`. Neither touches the audio queues, and the only
full clear is `cleanup_resources()` at process shutdown. Because TTS is
delivered pull-on-write — `handle_audio` drains the queue and emits only in
response to an inbound chunk — whatever a sample fails to pull before
disconnecting stays queued and is handed to the *next* session on its first
200 ms chunk. Before the patch this hit 1039 of 1984 samples (52.4%),
which received audio within 50 ms of streaming start, before the first input
chunk had finished sending; 1038 of those saw a `stop_tts` in the same instant.
Pull-on-write then kept audio flowing for every subsequent chunk, so the
recorder never saw a silence gap long enough to close the segment: median 26.9 s,
97.7% over 5 s, 987 of 1039 with exactly one segment. All four direct-replay
cells were unscoreable (Wait and Ignore failing at `onset=0.053 s`, Take turn
`no_takeover_in_success_window` 579/597, Backchannel `no_backchannel` 389/397).
Those numbers were withheld entirely and are not reported anywhere.

**Why this protocol surfaces it when FD-Bench's does not.** Both differences
are verifiable in the copy of FD-Bench's own VITA driver vendored at
`FD-Bench/examples/vita/vita/web_demo/client-audio.py`, which drives this same
server for the same kind of batch replay. First, the unit of measurement: an
FD-Bench item is one long multi-round conversation streamed in a single
session, and its metrics are round-indexed against the user turns inside that
conversation. Ours is one short cue per sample, one session per sample, with
the server kept warm across 2000 sessions — so process-wide state crossing a
session boundary is state crossing a *measurement unit* boundary, and an
artifact at t≈0.05 s does not perturb one round among many, it destroys the
entire sample, because the whole score is the onset of the first model segment
relative to a single cue. Second, client pacing: FD-Bench's client stops
emitting `audio` the moment the input file is exhausted and then
`time.sleep(10)` before `recording-stopped`, whereas we emit 200 ms chunks
continuously for the full input duration. Under pull-on-write, emitting is
pulling, so continuous emission both retrieves the leftover immediately and
keeps the stream unbroken — which is precisely what fuses it into one
continuous ~27 s "response" at t≈0.

**Why we did not simply adopt their pacing instead.** The same pull-on-write
property means the server cannot deliver anything while the client is silent,
even with a response already queued, so a response landing inside that 10 s
window is silently lost. Continuous emission is what makes our onset
measurements faithful; it is both the reason we saw this artifact and the
reason we could not avoid it by loosening the client.

**What we deliberately left alone.** The module-level `last_tts_model_id` is
*not* reset. Zeroing it would make the next generation's first queue item
mismatch and be skipped, dropping the opening sentence of the response
(`outputs_queue` is filled per punctuation mark) and shifting the measured
onset. That leaves one pre-existing issue unaddressed: when consecutive samples
land on different workers, the first TTS item is skipped regardless. It affects
the pre-patch data equally and is not something this patch introduces or fixes.

**Verification.** On the patched server (the source of every
VITA-1.5 number reported here), first-segment onset < 0.1 s is 0 of 2000,
against 52.4% before. The model's characteristic profile is unchanged and
simply now measurable: 1824 of 2000 samples (91.2%) are a single segment with a
median of roughly 22 s of speech, arriving well after the cue rather than at
t=0.

