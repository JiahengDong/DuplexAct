#!/usr/bin/env python3
"""
Live monitor for the `interrupt_user` action.

The simulator plays one continuous user utterance (the dataset's
branch_ignored.wav, i.e. transcript_user_full) and the model may barge in.
Whether the simulated user stops talking is an LLM judgment rather than "the
model started speaking", because many duplex models emit early filler or
backchannels that are not real interruptions.

While the model is speaking, the LLM (DECISION_MODEL, shared with
LiveTriggerController) is polled with "is the model trying to take the turn?"
Greetings, backchannels and filler -> stop=false. Any other intentional
take-turn speech, on-topic or not -> stop=true, and the rest of the user audio
is dropped with truncate_pending(). The model is not required to match
reference_interruption_text or l3_condition.

Outcomes:
  ACKNOWLEDGED -- the LLM said stop; the pending user-speech tail is dropped.
  IGNORED      -- the utterance finished (plus a short grace) without a stop.

`t_target` is a scoring annotation only; the clip is never cut there. Scores
come from the recorder's model_speech_segments vs metadata t_target, not from
this controller's poll timestamps.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import DynamicPCMSource, SynchronizedRecorder  # noqa: E402
from trigger_controller import (  # noqa: E402
    DECISION_MODEL, POLL_INTERVAL_S, MIN_SPEAKING_S, MAX_RETRIES, RETRY_BACKOFF_S,
    ends_at_sentence_boundary,
)

GRACE_AFTER_TARGET_S = 2.0
MAX_POST_ONSET_WAIT_S = 20.0

STOP_SYSTEM_PROMPT = (
    "You are the USER-side of a spoken-dialogue simulator. You are currently "
    "playing a pre-recorded user utterance into the model under test. The "
    "model may start speaking over you (barge-in). Your ONLY job is to decide "
    "whether the model is trying to TAKE THE TURN -- i.e. whether YOU (the "
    "user) should STOP talking right now so the model can speak. "
    "stop=true if the model's text so far is an intentional attempt to take "
    "the floor: any real utterance meant as the model's own turn -- an "
    "opinion, suggestion, question to the user, correction, instruction, "
    "offer, comment, or other content-bearing speech. Off-topic, rambling, "
    "or unhelpful still counts, as long as it is take-turn speech rather "
    "than a social continuer. Do NOT require the speech to match the "
    "user's topic, to offer help, or to resemble the reference "
    "interruption. "
    "stop=false ONLY if the model is greeting, backchanneling, stalling, "
    "or producing filler with no turn-taking intent (examples: 'how are "
    "you', 'yeah', 'mm-hm', 'I see', 'okay', 'go on', 'that's true', "
    "'that makes sense', a hello, or an empty/unavailable transcript). "
    "A short backchannel is NOT an interruption. A short but real "
    "take-turn (a genuine question, suggestion, or statement) IS. "
    "When in doubt between take-turn vs greeting/backchannel/filler, "
    "answer stop=true. "
    'Return strict JSON only: {"stop": true|false, "reason": "<one short sentence>"}.'
)


class InterruptMonitorController:
    """
    LLM-judged truncate-user-speech controller for interrupt_user only.
    One-shot: run() resolves once with acknowledged or ignored.
    """

    def __init__(
        self,
        recorder: SynchronizedRecorder,
        source: DynamicPCMSource,
        user_speech_duration: float,
        branch_acknowledged_pcm: bytes,
        branch_ignored_pcm: bytes,
        now_fn,
        reference_context: Optional[dict] = None,
        llm_client=None,
        decision_model: str = DECISION_MODEL,
        event_logger=None,
        poll_interval_s: float = POLL_INTERVAL_S,
        min_speaking_s: float = MIN_SPEAKING_S,
        grace_after_target_s: float = GRACE_AFTER_TARGET_S,
        max_post_onset_wait_s: float = MAX_POST_ONSET_WAIT_S,
        speech_chars_per_s: Optional[float] = None,
    ):
        self.recorder = recorder
        self.source = source
        self.user_speech_duration = user_speech_duration
        self.branch_acknowledged_pcm = branch_acknowledged_pcm
        self.branch_ignored_pcm = branch_ignored_pcm
        self.now_fn = now_fn
        self.reference_context = reference_context or {}
        self.client = llm_client
        self.decision_model = decision_model
        self.event_logger = event_logger
        self.poll_interval_s = poll_interval_s
        self.min_speaking_s = min_speaking_s
        self.grace_after_target_s = grace_after_target_s
        self.max_post_onset_wait_s = max_post_onset_wait_s
        self.speech_chars_per_s = speech_chars_per_s

        # As in LiveTriggerController: all text the provider has sent vs. the
        # prefix whose audio the listener has heard. The LLM only sees the
        # heard prefix, matching the audio-based speaking gate.
        self._text_full = ""
        self._released_chars = 0
        self._release_audit: dict = {}

    def on_text(self, text: str, at_s: Optional[float] = None) -> None:
        # Same concatenation rule as LiveTriggerController.on_text: Moshi
        # tokens already carry their own leading space at word boundaries.
        if text:
            self._text_full += text

    @property
    def transcript_so_far(self) -> str:
        """The part of this turn's text the listener has already heard, which is
        what the monitor LLM sees. Ungated when speech_chars_per_s is None."""
        if self.speech_chars_per_s is None:
            return self._text_full.strip()
        return self._text_full[:self._released_chars].strip()

    def _release_heard_text(self) -> None:
        """Advance the released prefix to match played speech, cutting on a word
        boundary so `ends_at_sentence_boundary` describes a real position in the
        utterance. Monotone: text shown once is never withdrawn."""
        if self.speech_chars_per_s is None:
            return
        played = self.recorder.played_speech_s()
        budget = min(len(self._text_full),
                     int(self.speech_chars_per_s * played))
        if 0 < budget < len(self._text_full):
            cut = self._text_full.rfind(" ", 0, budget + 1)
            budget = cut if cut > 0 else 0
        self._released_chars = max(self._released_chars, budget)
        self._release_audit = {
            "heard_chars": self._released_chars,
            "generated_chars": len(self._text_full),
            "played_speech_s": round(played, 3),
        }

    def on_interrupt(self) -> None:
        """Model-client hook: call right after recorder.interrupt(). The text
        belonging to the discarded audio goes with it, otherwise the abandoned
        tail sits in front of the next response and absorbs its release
        budget."""
        if self.speech_chars_per_s is None:
            return
        self._release_heard_text()
        self._text_full = self._text_full[:self._released_chars]

    def _log(self, etype: str, data: Optional[dict] = None) -> None:
        if self.event_logger is not None:
            self.event_logger.log("interrupt_monitor", etype, data)

    def _build_user_message(self, speaking_duration: float, elapsed: float) -> str:
        ref = self.reference_context
        return (
            f"Elapsed since user started speaking: {elapsed:.1f}s "
            f"(user utterance natural end ≈ {self.user_speech_duration:.1f}s).\n"
            f"Model has been continuously speaking for {speaking_duration:.1f}s "
            f"in its current turn.\n"
            f"Model text so far this turn: "
            f"{self.transcript_so_far or '(not available)'}\n"
            f"ends_at_sentence_boundary (COMPUTED): "
            f"{str(ends_at_sentence_boundary(self.transcript_so_far)).lower()}\n\n"
            f"What the user is currently saying (context only; do not require "
            f"the model to be on this topic): "
            f"\"{ref.get('transcript_user_full') or ref.get('transcript_user', '')}\"\n"
            f"Ignore the following reference except as an example of take-turn "
            f"speech -- do NOT require the model to match it: "
            f"\"{ref.get('reference_interruption_text', '')}\"\n\n"
            f"Is the model taking a turn (stop=true), or only greeting / "
            f"backchanneling / filler (stop=false)?"
        )

    def _ask_llm(self, speaking_duration: float, elapsed: float) -> tuple[bool, str]:
        if self.client is None:
            return False, "llm_unavailable"
        user_msg = self._build_user_message(speaking_duration, elapsed)
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.decision_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": STOP_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                parsed = json.loads(raw)
                stop = bool(parsed.get("stop", False))
                reason = str(parsed.get("reason", ""))
                return stop, reason
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_S * attempt)
        return False, f"llm_error:{type(last_exc).__name__}: {last_exc}"

    def _finish_ignored(self, t_end: float) -> dict:
        # The full utterance is already in the source; nothing is replayed.
        self._log("ignored", {"t": round(t_end, 4), "outcome": "ignored"})
        return {
            "t_model_onset_detected": None,
            "t_branch_appended": None,
            "outcome": "ignored",
            "bytes_truncated": 0,
            "forced": False,
            "last_reason": "window_elapsed_no_llm_stop",
        }

    def _finish_acknowledged(self, t_onset: float, bytes_truncated: int) -> dict:
        self._log("acknowledged", {
            "t": round(t_onset, 4), "bytes_truncated": bytes_truncated,
            "outcome": "acknowledged",
        })
        return {
            "t_model_onset_detected": round(t_onset, 4),
            "t_branch_appended": None,
            "outcome": "acknowledged",
            "bytes_truncated": bytes_truncated,
            "forced": False,
        }

    async def run(self) -> dict:
        start = self.now_fn()
        total_wait = self.user_speech_duration + self.grace_after_target_s
        num_polls = 0
        last_reason = ""

        while True:
            await asyncio.sleep(self.poll_interval_s)
            now = self.now_fn()
            elapsed = now - start
            speaking_duration = self.recorder.current_speaking_duration(at_s=now)

            if elapsed >= total_wait:
                result = self._finish_ignored(now)
                result["num_polls"] = num_polls
                if last_reason:
                    result["last_reason"] = last_reason
                return result

            if speaking_duration < self.min_speaking_s:
                continue
            if self.client is None:
                continue

            self._release_heard_text()
            num_polls += 1
            stop, reason = await asyncio.to_thread(self._ask_llm, speaking_duration, elapsed)
            last_reason = reason
            self._log("poll", {
                "speaking_duration": round(speaking_duration, 3),
                "elapsed": round(elapsed, 3),
                "stop": stop,
                "reason": reason,
                "transcript_so_far": self.transcript_so_far,
                "ends_at_sentence_boundary": ends_at_sentence_boundary(self.transcript_so_far),
                "num_polls": num_polls,
                **({"release": dict(self._release_audit)} if self._release_audit else {}),
            })
            if not stop:
                continue

            t_onset = now
            bytes_truncated = self.source.truncate_pending()
            self._log("interrupt_detected", {
                "t": round(t_onset, 4),
                "bytes_truncated": bytes_truncated,
                "reason": reason,
                "num_polls": num_polls,
            })
            result = self._finish_acknowledged(t_onset, bytes_truncated)
            result["last_reason"] = last_reason
            result["num_polls"] = num_polls
            return result
