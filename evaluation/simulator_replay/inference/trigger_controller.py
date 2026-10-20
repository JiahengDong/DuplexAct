#!/usr/bin/env python3
"""
Shared live-injection trigger logic for simulator-driven actions
(stop, respond_and_recover, hold_floor).

The simulator follows tau2-bench's tick-loop / dual-clock design (audio time
vs wall-clock time), but instead of generating the user's speech with
LLM+TTS it replays a fixed, pre-recorded cue clip, so every model hears the
same audio. The only thing decided live is WHEN to start playing the clip.

That decision is made by an LLM (DECISION_MODEL) polled at a fixed cadence
(POLL_INTERVAL_S), given the model's own live transcript and the reference
case from the source human conversation (metadata.json's
reference_original_response_text / reference_interruption_text /
gpt_reason / interruption_type) as grounding for what "the right moment"
looks like. One controller instance drives one sample's single injection.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from realtime_common import DynamicPCMSource, SynchronizedRecorder  # noqa: E402

DECISION_MODEL = "gpt-5.4"

# The poll interval is a property of the simulated user, not the model under
# test, so every model gets the same wall-clock cadence (and the same
# "patience") regardless of its native frame size. The loop is serial
# sleep(P) then await LLM (~1.0-1.5s), so the effective cadence is about
# P + 1.2s; 1.0s matches the 1s text chunk of the slowest-chunked model
# and MIN_SPEAKING_S.
TAU2_TICK_DURATION_S       = 0.20   # tau2-bench's own default tick size
TAU2_CHECK_INTERVAL_TICKS  = 10     # tau2-bench's tick ratio; ignored by poll_interval_for_native_chunk
POLL_INTERVAL_S = 1.0             # ALL models (open- and closed-source)
MIN_SPEAKING_S = 1.0      # don't ask the LLM before the model has said this much
                          # of its *post-lead-in* reply (speech during lead-in does
                          # not count, and we do not poll until lead-in has finished)
# Wait clocks start when the lead-in has finished playing, not at stream t=0:
#   T_ONSET_S   -- wall clock after lead-in with ZERO post-lead-in speech
#                  -> force-inject.
#   T_SEARCH_S  -- wall clock from the model's first post-lead-in onset;
#                  force-inject whether it is still talking or already
#                  silent. The LLM may trigger earlier while it is speaking.
# interrupt_user does not use these (see interrupt_controller.py).
T_ONSET_S = 5.0
T_SEARCH_S = 8.0
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0

# ── Ear-aligned transcript release ────────────────────────────────────────
# The trigger LLM must decide "is this a good moment to cut in" from what the
# simulated listener has actually HEARD, not from what the provider has
# already generated. Those differ for two reasons:
#
#   (a) Realtime APIs deliver audio faster than realtime and the recorder
#       plays it at 1x, so unplayed audio sits queued.
#   (b) Within one response the TEXT stream also runs ahead of that
#       response's own audio, sometimes delivering a whole turn's transcript
#       almost at once.
#
# Because of (b), tagging each delta with "audio generated so far" is not
# enough, and tau2-bench's byte-proportional split (played/received) would
# release the whole turn for a provider that streams audio at realtime. No
# provider exposes word-level timings, so the release budget is derived from
# played speech duration and the model's speaking rate:
#
#     released_chars = SPEECH_CHARS_PER_S[model] * recorder.played_speech_s()
#
# clamped to the text actually received and monotone (never un-say
# something already shown to the LLM).
#
# Rates are the median of (transcript chars) / (audio seconds) per response.
# They are a property of each model's own TTS voice, not of the simulated
# user, so unlike POLL_INTERVAL_S they may differ per model.
SPEECH_CHARS_PER_S = {
    "gpt-realtime-2.1":            15.9,
    "grok-voice":                  16.6,
    "qwen3.5-omni-flash-realtime": 19.7,
    "gemini-3.1-flash-live":       17.3,
}


def speech_rate_for(model_slug: Optional[str]) -> Optional[float]:
    """
    Release rate for `model_slug`, or None to leave the transcript ungated.

    EAR_ALIGN_TRANSCRIPT=0 disables the gate for every model, so the gate's
    effect can be measured separately from the playback-layer flush.
    """
    if os.environ.get("EAR_ALIGN_TRANSCRIPT", "1").strip() in ("0", "false", "no"):
        return None
    return SPEECH_CHARS_PER_S.get(model_slug or "")
# Open-source clients pass no rate and are not gated: they are
# frame-synchronous (one output frame per input frame, with input paced at
# wall clock), so their text cannot run ahead of played audio and the gate
# would be a no-op.


def poll_interval_for_native_chunk(native_chunk_s: float,
                                    ticks: int = TAU2_CHECK_INTERVAL_TICKS) -> float:
    """
    Returns POLL_INTERVAL_S for every model.

    The trigger controller is part of the simulated user, not the system
    under test, so every client uses the same wall-clock interval.
    `native_chunk_s` / `ticks` are ignored.
    """
    return POLL_INTERVAL_S


# ── Trigger-LLM key failover ──────────────────────────────────────────────
# Primary: OPENAI_API_KEY. Optional backup: OPENAI_API_KEY_FALLBACK.
# On insufficient_quota / "exceeded your current quota" the process
# switches to the backup. After PRIMARY_REQUERY_EVERY_SAMPLES finished
# samples on the backup, the next chat call probes the primary again and
# switches back if it succeeds. TPM 429s are not treated as quota errors;
# those retry on the current key.

PRIMARY_REQUERY_EVERY_SAMPLES = int(os.environ.get("OPENAI_PRIMARY_REQUERY_EVERY_SAMPLES", "50"))
_REPO_ENV = Path(__file__).resolve().parents[3] / ".env"
_TRIGGER_LLM_CLIENT = None


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        code = err.get("code") if isinstance(err, dict) else None
        if code == "insufficient_quota":
            return True
    code = getattr(exc, "code", None)
    return code == "insufficient_quota"


class _ChatCompletionsProxy:
    def __init__(self, owner: "TriggerLLMClient"):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create(**kwargs)


class _ChatProxy:
    def __init__(self, owner: "TriggerLLMClient"):
        self.completions = _ChatCompletionsProxy(owner)


class TriggerLLMClient:
    """Duck-types openai.OpenAI for `.chat.completions.create`, with sticky quota failover."""

    def __init__(self, primary_key: str, fallback_key: Optional[str] = None):
        from openai import OpenAI
        self._OpenAI = OpenAI
        self._keys = [primary_key] + ([fallback_key] if fallback_key else [])
        self._idx = 0
        self._client = OpenAI(api_key=self._keys[0])
        self.chat = _ChatProxy(self)
        self._samples_on_fallback = 0
        self._probe_primary_next = False
        n = len(self._keys)
        print(f"[trigger] OpenAI client ready ({n} key(s); "
              f"{'fallback armed' if n > 1 else 'no fallback'}; "
              f"requery primary every {PRIMARY_REQUERY_EVERY_SAMPLES} fallback samples)",
              file=sys.stderr)

    def on_sample_finished(self) -> None:
        """Count a completed sample. After N on fallback, probe primary on the next chat call."""
        if self._idx == 0 or len(self._keys) < 2:
            self._samples_on_fallback = 0
            return
        self._samples_on_fallback += 1
        if self._samples_on_fallback >= PRIMARY_REQUERY_EVERY_SAMPLES:
            self._probe_primary_next = True
            self._samples_on_fallback = 0
            print(f"[trigger] {PRIMARY_REQUERY_EVERY_SAMPLES} fallback samples done; "
                  f"will probe primary key on next poll", file=sys.stderr)

    def _switch_to(self, idx: int, reason: str) -> None:
        self._idx = idx
        self._client = self._OpenAI(api_key=self._keys[idx])
        self._samples_on_fallback = 0
        self._probe_primary_next = False
        print(f"[trigger] {reason} (now key {idx + 1}/{len(self._keys)})", file=sys.stderr)

    def _create(self, **kwargs):
        if self._probe_primary_next and self._idx != 0 and len(self._keys) > 1:
            self._probe_primary_next = False
            print("[trigger] probing primary OpenAI key (may have been recharged)",
                  file=sys.stderr)
            probe = self._OpenAI(api_key=self._keys[0])
            try:
                resp = probe.chat.completions.create(**kwargs)
                self._switch_to(0, "primary OpenAI key has quota again; switched back")
                return resp
            except Exception as exc:
                if _is_quota_error(exc):
                    print("[trigger] primary still over quota; staying on fallback",
                          file=sys.stderr)
                else:
                    print(f"[trigger] primary probe failed ({type(exc).__name__}); "
                          f"staying on fallback", file=sys.stderr)

        last_exc: Optional[BaseException] = None
        while self._idx < len(self._keys):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if _is_quota_error(exc) and self._idx + 1 < len(self._keys):
                    self._switch_to(
                        self._idx + 1,
                        "OpenAI key hit quota; switched to fallback",
                    )
                    continue
                raise
        raise last_exc  # type: ignore[misc]


def make_trigger_llm_client():
    """
    Process-wide singleton so a quota switch sticks across samples in a job.
    Returns None if OPENAI_API_KEY is unset.
    """
    global _TRIGGER_LLM_CLIENT
    if _TRIGGER_LLM_CLIENT is not None:
        return _TRIGGER_LLM_CLIENT
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ENV)
    except Exception:
        pass
    primary = os.environ.get("OPENAI_API_KEY", "").strip()
    fallback = os.environ.get("OPENAI_API_KEY_FALLBACK", "").strip()
    if not primary:
        return None
    _TRIGGER_LLM_CLIENT = TriggerLLMClient(primary, fallback or None)
    return _TRIGGER_LLM_CLIENT


_SHARED_TRIGGER_RULES = (
    "The model under test provides a live transcript of its current turn; "
    "use that text. "
    "Preferences, strongest first: "
    "(1) prefer a MID-CLAUSE / mid-phrase point; "
    "(2) prefer a locus whose injection PATTERN resembles the reference "
    "(timing and type), not matching its lexical content. "
    "A sentence-final point (ends_at_sentence_boundary=true) is ALLOWED. "
    "It is only a weaker preference than mid-clause -- never refuse to "
    "trigger solely because the text currently ends in '.', '?', or '!'. "
    "You will be given ends_at_sentence_boundary, already computed "
    "programmatically (regex for ending in '.', '?', or '!'). Trust that "
    "boolean exactly; do not re-derive it, and do not contradict it in "
    "your reason. "
    "If the model has already produced some intelligible meaning, but that "
    "content has clearly diverged from the reference, inject NOW at a "
    "suitable point (mid-clause if available, otherwise sentence-final is "
    "fine). Do not keep waiting for an analogous content match that will "
    "not arrive, and do not wait until the turn sounds fully finished. "
    "Do not trigger too early -- before the model has said anything "
    "substantive (a word or two of filler is not enough). "
    "Your one-sentence reason MUST restate the exact "
    "ends_at_sentence_boundary value you were given. "
    'Return strict JSON only: {"trigger": true|false, "reason": "<one short sentence>"}.'
)

STOP_TRIGGER_SYSTEM_PROMPT = (
    "You control WHEN a pre-recorded USER INTERRUPTION audio clip should be "
    "played into the model under test's current reply. The clip is addressed "
    "to the model and is meant to cut in so we can measure whether it "
    "stops talking. "
    "The reference is a PATTERN for when/how a human interruption is "
    "injected (for example cutting in mid-thought vs after a short beat). "
    "It is NOT a script: do not wait for the model to discuss the same "
    "topic or reuse the same wording as the reference original response. "
    + _SHARED_TRIGGER_RULES
)

RECOVER_TRIGGER_SYSTEM_PROMPT = (
    "You control WHEN a pre-recorded USER INTERRUPTION audio clip should be "
    "played into the model under test's current reply. The clip is addressed "
    "to the model. After it plays we measure how the model recovers "
    "(continues, corrects, enriches, etc.) -- not whether the conversation "
    "simply ends. Inject during the reply, at a suitable cut-in point; "
    "do not wait until the turn is already over. "
    "The reference is a PATTERN for when/how that interruption lands. "
    "It is NOT a script: do not wait for the model to discuss the same "
    "topic or reuse the same wording as the reference original response. "
    + _SHARED_TRIGGER_RULES
)

HOLD_FLOOR_TRIGGER_SYSTEM_PROMPT = (
    "You control WHEN a pre-recorded overlapping audio clip should be "
    "played while the model under test is speaking. This clip is NOT a "
    "user interruption that takes the turn. Depending on the condition it "
    "is either a short backchannel (e.g. 'mm-hmm', 'yes') from a listener "
    "who is NOT taking the floor, background speech in the environment, "
    "or other-directed speech between other people. "
    "The test is whether the model KEEPS TALKING through that overlap. "
    "Therefore you MUST inject while the model is still in the middle of "
    "its reply. Injecting after it has already stopped makes the trial "
    "invalid. "
    "The reference is a PATTERN for when that overlapping audio occurred "
    "in the source conversation. It is NOT a script: do not wait for the "
    "model to discuss the same topic or reuse the same wording. "
    + _SHARED_TRIGGER_RULES
)

# Alias for the stop wording; prompts are normally picked via action.
TRIGGER_SYSTEM_PROMPT = STOP_TRIGGER_SYSTEM_PROMPT

_PROMPT_BY_ACTION = {
    "stop": STOP_TRIGGER_SYSTEM_PROMPT,
    "respond_and_recover": RECOVER_TRIGGER_SYSTEM_PROMPT,
    "hold_floor": HOLD_FLOOR_TRIGGER_SYSTEM_PROMPT,
}

_HOLD_FLOOR_CUE_KIND = {
    "continue_current_response": (
        "a short backchannel from a listener who is not taking the turn"
    ),
    "continue_but_shorten": (
        "a short backchannel from a listener who is not taking the turn"
    ),
    "continue_with_simplification": (
        "a short backchannel from a listener who is not taking the turn"
    ),
    "continue_with_background_speech": (
        "background speech in the environment, not addressed to the model"
    ),
    "continue_with_other_directed_speech": (
        "other-directed speech between other people, not addressed to the model"
    ),
}

_SENTENCE_BOUNDARY_RE = re.compile(r'[.?!]\s*$')


def ends_at_sentence_boundary(text: str) -> bool:
    """
    Whether `text` ends in '.', '?' or '!'. Computed here and handed to the
    trigger LLM as a trusted given, because the LLM is unreliable at judging
    this from raw text. Sentence-final is a weaker preference, not a veto.
    """
    return bool(_SENTENCE_BOUNDARY_RE.search(text.rstrip())) if text else False


class LiveTriggerController:
    """
    Polls every POLL_INTERVAL_S while the model is speaking its *post-lead-in*
    reply; once the LLM decides "now", appends `cue_pcm` to `source` and
    records `t_user_int_start` on the sample clock (`now_fn`). One-shot:
    `run()` resolves after an LLM trigger or a forced injection.

    Does not poll or inject during lead-in. Once the model starts its
    post-lead-in reply, T_SEARCH_S of wall clock later the cue is
    force-injected whether that reply is still going or already silent.
    """

    def __init__(
        self,
        recorder: SynchronizedRecorder,
        source: DynamicPCMSource,
        cue_pcm: bytes,
        now_fn,
        reference_context: dict,
        client,
        lead_in_duration_s: float,
        decision_model: str = DECISION_MODEL,
        poll_interval_s: float = POLL_INTERVAL_S,
        min_speaking_s: float = MIN_SPEAKING_S,
        t_onset_s: float = T_ONSET_S,
        t_search_s: float = T_SEARCH_S,
        event_logger=None,
        speech_chars_per_s: Optional[float] = None,
    ):
        self.recorder = recorder
        self.source = source
        self.cue_pcm = cue_pcm
        self.now_fn = now_fn
        self.reference_context = reference_context
        self.client = client
        self.lead_in_duration_s = float(lead_in_duration_s)
        self.decision_model = decision_model
        self.poll_interval_s = poll_interval_s
        self.min_speaking_s = min_speaking_s
        self.t_onset_s = t_onset_s
        self.t_search_s = t_search_s
        self.event_logger = event_logger
        self.speech_chars_per_s = speech_chars_per_s

        # Everything the provider has sent us this sample, vs. the prefix of
        # it whose audio the listener has actually heard. When
        # speech_chars_per_s is None the two are the same string.
        self._text_full = ""
        self._released_chars = 0
        self._release_audit: dict = {}
        self._triggered = False
        action = str(
            (reference_context or {}).get("action") or "stop"
        ).strip()
        if action not in _PROMPT_BY_ACTION:
            action = "stop"
        self.action = action
        self.system_prompt = _PROMPT_BY_ACTION[action]

    # ── model-client hook: call this on every text token / transcript delta
    def on_text(self, text: str, at_s: Optional[float] = None) -> None:
        # No separator is inserted: callers emit tokens that already carry
        # their own leading space at word boundaries (" day", ",", " how").
        # A caller whose tokens do not self-space must normalize first.
        if text:
            self._text_full += text

    @property
    def transcript_so_far(self) -> str:
        """
        The part of this turn's text whose audio the listener has already
        heard -- this is what the trigger LLM sees. Ungated (== everything
        received) when speech_chars_per_s is None.
        """
        if self.speech_chars_per_s is None:
            return self._text_full.strip()
        return self._text_full[:self._released_chars].strip()

    def _release_heard_text(self) -> None:
        """
        Advance the released prefix to match played speech. Monotone: a
        character shown to the LLM once is never withdrawn, even if a
        subsequent interrupt() makes the budget shrink.
        """
        if self.speech_chars_per_s is None:
            return
        played = self.recorder.played_speech_s()
        budget = min(len(self._text_full),
                     int(self.speech_chars_per_s * played))
        # Cut on a word boundary so the released text (and
        # `ends_at_sentence_boundary` on it) never ends mid-word.
        if 0 < budget < len(self._text_full):
            cut = self._text_full.rfind(" ", 0, budget + 1)
            budget = cut if cut > 0 else 0
        self._released_chars = max(self._released_chars, budget)
        # Snapshot for the poll log, since the following LLM call takes ~1s
        # and re-reading afterwards would not describe what the LLM saw.
        self._release_audit = {
            "heard_chars": self._released_chars,
            "generated_chars": len(self._text_full),
            "played_speech_s": round(played, 3),
        }

    def on_interrupt(self) -> None:
        """
        Model-client hook: call right after recorder.interrupt(), i.e. when
        queued-but-unplayed audio has just been discarded on barge-in. The
        text belonging to that discarded audio must go with it -- the listener
        never heard it, and (where the protocol supports it) we tell the server
        the same thing via conversation.item.truncate. Without this the
        abandoned tail would sit in front of the next response's text and
        permanently absorb its release budget.
        """
        if self.speech_chars_per_s is None:
            return
        self._release_heard_text()
        self._text_full = self._text_full[:self._released_chars]

    def _build_user_message(self, speaking_duration: float) -> str:
        ref = self.reference_context
        boundary = ends_at_sentence_boundary(self.transcript_so_far)
        l3 = ref.get("l3_condition") or ""
        header = (
            f"Action: {self.action}\n"
            f"Condition: {l3}\n"
            f"Model has been continuously speaking for {speaking_duration:.1f}s "
            f"so far in its reply AFTER the user finished talking "
            f"(speech that overlapped the user's lead-in is excluded).\n"
            f"Text generated so far this turn: "
            f"{self.transcript_so_far or '(empty so far)'}\n"
            f"ends_at_sentence_boundary (COMPUTED -- trust this exactly, do not "
            f"re-derive it): {str(boundary).lower()}\n\n"
        )
        if self.action == "hold_floor":
            cue_kind = _HOLD_FLOOR_CUE_KIND.get(
                l3, "overlapping audio that is not a turn-taking interruption"
            )
            return (
                header
                + "The clip to inject is "
                + cue_kind
                + ".\n"
                + "Reference case (injection PATTERN, not a script):\n"
                + f"- Model speech before the overlapping clip in the source: "
                + f"\"{ref.get('reference_before_text') or ref.get('reference_original_response_text', '')}\"\n"
                + f"- Cue text: \"{ref.get('cue_text') or ref.get('reference_interruption_text', '')}\"\n"
                + f"- Why this was a valid overlap point: {ref.get('gpt_reason', '')}\n\n"
                + "Treat the reference as when/how to overlap, not as wording "
                "the model under test must match. If the reply already has "
                "intelligible content but has diverged, prefer injecting now "
                "while it is still speaking.\n"
                "Should we inject the overlapping clip NOW?"
            )
        clip_name = (
            "interruption clip" if self.action == "stop"
            else "interruption clip (recover trial)"
        )
        return (
            header
            + "Reference case (interruption PATTERN, not a script):\n"
            + "- Original response before the interruption happened: "
            + f"\"{ref.get('reference_original_response_text', '')}\"\n"
            + "- What the interruption said in the real conversation: "
            + f"\"{ref.get('reference_interruption_text', '')}\"\n"
            + "- Why this was a valid interruption point in that conversation: "
            + f"{ref.get('gpt_reason', '')}\n"
            + f"- Interruption type: {ref.get('interruption_type', '')}\n\n"
            + "Treat the reference as when/how to cut in, not as wording "
            "the model under test must match. If the reply already has "
            "intelligible content but has diverged, prefer injecting now "
            "rather than waiting.\n"
            f"Should we inject the {clip_name} NOW?"
        )

    def _ask_llm(self, speaking_duration: float) -> tuple[bool, str]:
        user_msg = self._build_user_message(speaking_duration)
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.decision_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                parsed = json.loads(raw)
                trigger = bool(parsed.get("trigger", False))
                reason = str(parsed.get("reason", ""))
                return trigger, reason
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_S * attempt)
        return False, f"llm_error:{type(last_exc).__name__}: {last_exc}"

    def _log(self, etype: str, data: Optional[dict] = None) -> None:
        if self.event_logger is not None:
            self.event_logger.log("trigger", etype, data)

    def _result(self, *, t_user_int_start, num_polls: int, forced: bool,
                injected: bool, last_reason: str, speaking_after_lead_s: float,
                elapsed_after_lead_s: float,
                reply_onset_s: Optional[float] = None) -> dict:
        return {
            "t_user_int_start": (
                None if t_user_int_start is None else round(t_user_int_start, 4)
            ),
            "num_polls": num_polls,
            "forced": forced,
            "injected": injected,
            "last_reason": last_reason,
            "poll_interval_s": self.poll_interval_s,
            "lead_in_duration_s": round(self.lead_in_duration_s, 4),
            "speaking_after_lead_s": round(speaking_after_lead_s, 4),
            "elapsed_after_lead_s": round(elapsed_after_lead_s, 4),
            "reply_onset_s": (
                None if reply_onset_s is None else round(reply_onset_s, 4)
            ),
            "t_onset_s": self.t_onset_s,
            "t_search_s": self.t_search_s,
            "action": self.action,
        }

    def _inject(self, now: float, num_polls: int, forced: bool, reason: str,
                speaking_after_lead_s: float, elapsed_after_lead_s: float,
                reply_onset_s: Optional[float] = None) -> dict:
        self.source.append(self.cue_pcm)
        log_name = "forced_trigger" if forced else "triggered"
        self._log(log_name, {
            "t": now,
            "reason": reason,
            "speaking_after_lead_s": round(speaking_after_lead_s, 3),
            "elapsed_after_lead_s": round(elapsed_after_lead_s, 3),
            "reply_onset_s": None if reply_onset_s is None else round(reply_onset_s, 3),
        })
        return self._result(
            t_user_int_start=now,
            num_polls=num_polls,
            forced=forced,
            injected=True,
            last_reason=reason,
            speaking_after_lead_s=speaking_after_lead_s,
            elapsed_after_lead_s=elapsed_after_lead_s,
            reply_onset_s=reply_onset_s,
        )

    async def run(self) -> dict:
        """
        Returns a dict with t_user_int_start / num_polls / forced / injected /
        last_reason plus the post-lead-in timing fields.

        Outcomes:
          injected=True, forced=False -- LLM said trigger now.
          injected=True, forced=True, last_reason=search_timeout -- T_SEARCH_S
            wall clock since the first post-lead-in onset, still talking or
            already silent.
          injected=True, forced=True, last_reason=onset_timeout -- T_ONSET_S
            after lead-in with no post-lead-in speech.
        """
        num_polls = 0
        lead_end = self.lead_in_duration_s
        while True:
            await asyncio.sleep(self.poll_interval_s)
            now = self.now_fn()
            if now < lead_end:
                continue

            elapsed_after = now - lead_end
            acc = self.recorder.speaking_duration_between(lead_end, now)
            reply_onset = self.recorder.first_speech_start_after(lead_end, now)
            speaking_duration = self.recorder.current_speaking_duration(at_s=now)
            speaking = speaking_duration > 0.0

            if reply_onset is not None and (now - reply_onset) >= self.t_search_s:
                return self._inject(
                    now, num_polls, True, "search_timeout", acc, elapsed_after,
                    reply_onset_s=reply_onset,
                )

            if reply_onset is None and elapsed_after >= self.t_onset_s:
                return self._inject(
                    now, num_polls, True, "onset_timeout", acc, elapsed_after,
                    reply_onset_s=None,
                )

            if not speaking:
                continue

            open_start = now - speaking_duration
            post_this = now - max(open_start, lead_end)
            if post_this < self.min_speaking_s:
                continue

            num_polls += 1
            self._release_heard_text()
            trigger, reason = await asyncio.to_thread(self._ask_llm, post_this)
            self._log("poll", {
                "speaking_duration": round(post_this, 3),
                "speaking_after_lead_s": round(acc, 3),
                "reply_onset_s": None if reply_onset is None else round(reply_onset, 3),
                "trigger": trigger,
                "reason": reason,
                "ends_at_sentence_boundary": ends_at_sentence_boundary(self.transcript_so_far),
                # How much of the provider's text was audible when the LLM
                # was asked.
                **self._release_audit,
            })
            if trigger:
                return self._inject(
                    now, num_polls, False, reason, acc, elapsed_after,
                    reply_onset_s=reply_onset,
                )
