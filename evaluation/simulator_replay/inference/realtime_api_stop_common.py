#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/realtime_api_stop_common.py

Shared base class for `stop`-action simulator clients against closed-source
"GA Realtime schema" WebSocket APIs -- currently OpenAI GPT-Realtime and
xAI Grok Voice, whose direct_replay clients (infer_gpt_realtime.py,
infer_grok_voice.py) are already near-verbatim twins of each other (same
message shapes: session.update / response.created / input_audio_buffer.
speech_started / response.done / response.output_audio.delta /
response.audio.delta / error -- see infer_grok_voice.py's own module
docstring: "Protocol ... is essentially the same GA schema as OpenAI's
Realtime API"). Extracting the shared parts here (rather than duplicating
this whole file twice, the way infer_stop_moshi.py / infer_stop_flm_audio.py
are still two full separate files, since THOSE differ enough in their
sphn/opus plumbing to not be worth a shared base) keeps the two per-model
files (infer_stop_gpt_realtime.py, infer_stop_grok_voice.py) down to just
their provider-specific session.update shape plus a few constants.

Each subclass must set:
  - MODEL_SLUG, OUTPUT_SR / INPUT_SR (as normal for RealtimeClientBase)
  - WS_URL (class attr, no `?model=` query string -- appended in
    _connect_and_stream from self.model_name, matching both original
    clients' own construction)
  - CHUNK_MS / API_SR (both providers currently use 20ms @ 24kHz, kept as
    class attrs rather than hardcoded here in case a future provider on
    this same schema differs)
  - build_session_update(self) -> dict -- the one genuinely
    provider-specific piece (GPT nests turn_detection/voice differently
    from Grok -- see each client's own module for the exact shape)

Poll interval: every simulator client (open- and closed-source) uses
trigger_controller.POLL_INTERVAL_S (1.0s) as the simulated-user check
cadence. LiveTriggerController is constructed below without a
poll_interval_s override for exactly this reason.

Live text signal: both original direct_replay clients currently drop
transcript deltas outright (an explicit "ignore other delta streams"
branch) -- this is the MINOR ADJUST flagged in MODEL_COMPATIBILITY.md.
This file stops ignoring them: any event type in TRANSCRIPT_DELTA_TYPES is
decoded and fed to the trigger controller via ctrl.on_text(), mirroring how
moshi/flm-audio's text tokens are handled. Both known alias names are
accepted, the same two-alias pattern already used for audio deltas in both
original clients.

This file is deliberately separate from infer_gpt_realtime.py and
infer_grok_voice.py (never imported by them, never touched by them) -- see
trigger_controller.py's own module docstring for the "always a separate
module, never a subclass override of the direct_replay client" rule.
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import (
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, classify_terminal_response_status,
    DynamicPCMSource, is_expected_cancel,
)
from trigger_controller import LiveTriggerController, speech_rate_for
from stop_common import POST_CUE_OBSERVATION_S

MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024

TRANSCRIPT_DELTA_TYPES = (
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
    "response.text.delta",
)
TRANSCRIPT_DONE_TYPES = (
    "response.output_audio_transcript.done",
    "response.audio_transcript.done",
)


class ProviderError(RuntimeError):
    pass


# One playback policy for every action: on barge-in, drop the queued-but-unheard
# backlog. These APIs emit audio 4.0-5.5x faster than realtime, so by the time
# the cue reaches the model's ear there is 3.1-7.0s of audio it generated before
# it could know about the cue -- audio that in a 1x system would not exist yet.
# Dropping it makes the playback clock catch up to the generation clock, which
# is the whole point: from that instant on, whether the listener hears speech is
# the model's own post-cue decision (its continued deltas arrive within ~0.1s at
# these generation speeds) rather than buffer drainage. That is what lets stop,
# hold_floor and respond_and_recover all be read off the ongoing speech.
#
# What must NOT come with the flush is the R-03 response-id blacklist -- see the
# speech_started handler.
#
# Still open, and deferred to the same round as interrupt_user and the scoring
# fields: a model that finished generating its whole turn before the cue reached
# its ear (GPT 46%, Grok 70%, Qwen 100% of hold_floor samples) never had the
# stop-or-continue decision available to it, so those samples should be gated
# out as not-applicable rather than scored as failures. That is an eligibility
# question on the scoring side, not a playback-policy one.


class RealtimeAPIStopClientBase(RealtimeClientBase):
    WS_URL = ""
    CHUNK_MS = 20
    API_SR = 24000
    # Does this provider accept `conversation.item.truncate`, i.e. can we tell
    # its server how much of an assistant item the listener actually heard?
    # Only OpenAI does. xAI documents cancellation but not output truncation
    # (pydantic-ai reports supports_output_truncation=False for it, and
    # livekit/agents#6391 shows sending it there desyncs the server-side item
    # list and then drops every later item); DashScope's Qwen-Omni client
    # event list has no truncate event at all. Gemini needs none -- its server
    # does barge-in itself and truncates its own context. tau2-bench makes the
    # same split: server-side truncate for OpenAI only, client-side discard
    # for all four.
    SUPPORTS_ITEM_TRUNCATE = False

    def __init__(self, api_key, model_name, system_prompt, verbose,
                 cue_pcm, reference_context, llm_client):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.cue_pcm = cue_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result = {}
        self.text_tokens = []
        self.simulator_speech_segments = []

    def build_session_update(self):
        raise NotImplementedError

    def websocket_url(self) -> str:
        return self.WS_URL + "?model=" + self.model_name

    async def _connect_and_stream(self, input_pcm, input_duration, recorder,
                                   event_logger, start_streaming):
        chunk_samples = self.API_SR * self.CHUNK_MS // 1000
        chunk_bytes = chunk_samples * 2

        url = self.websocket_url()
        key = self.api_key
        if "api.openai.com" in (self.WS_URL or ""):
            from openai_keys import bearer_key
            key = bearer_key(self.api_key)
        headers = {"Authorization": "Bearer " + key}

        response_active = False
        expecting_cancel = False
        current_response_id = None
        # Barge-in bookkeeping. `current_item_id` is the assistant item whose
        # audio is currently being played out, and `item_play_start_s` is where
        # on the recorder's played-speech axis that item's playback begins --
        # snapshotted as (already played) + (still queued) when the item's
        # first delta lands, because its predecessor may still be draining.
        # audio_end_ms for conversation.item.truncate is the difference.
        current_item_id = None
        item_play_start_s = 0.0
        got_transcript_delta = False
        provider_errors = []
        # Dedupes a confirmed provider quirk found at full-scale (0/1905
        # direct_replay samples ever hit it -- that pipeline never sends
        # interrupting audio mid-response, so this path was simply never
        # exercised there; 8/200 stop-simulator gpt-realtime-2.1 samples did):
        # OpenAI's server can emit a SECOND, stale `response.done` for a
        # response_id that had ALREADY received its real terminal event.
        # Confirmed pattern on every occurrence (Candor_stop_0028/0060,
        # AMI_stop_0139, Candor_stop_0026, ...): response.done(completed) for
        # response_id X, then ~150-250ms later -- after our injected cue audio
        # reached the server and its own VAD flagged turn_detected on it --
        # a DUPLICATE response.done(cancelled, reason=turn_detected) for that
        # SAME already-finished response_id X. Our own `expecting_cancel` flag
        # can't be True for it (no `speech_started` had arrived yet when the
        # first "completed" landed, so response_active was already False by
        # the time speech_started did fire), so this second event previously
        # misclassified as unexpected_cancelled and failed 8/200 otherwise-
        # clean samples. The first terminal status a response_id receives is
        # authoritative (audio for it is already fully recorded); any later
        # response.done for the SAME response_id is provider-side redundancy,
        # not a new failure, so it's logged but not re-classified/re-errored.
        finalized_response_ids: set = set()
        controller_holder = {"controller": None}
        stream_t0 = {"t": None}

        async with websockets.connect(
            url, additional_headers=headers, max_size=MAX_WS_MESSAGE_BYTES,
            ping_interval=20, ping_timeout=30,
        ) as ws:
            await ws.send(json.dumps(self.build_session_update()))
            event_logger.log("send", "session.update")

            async def receive_loop():
                nonlocal response_active, expecting_cancel, current_response_id
                nonlocal finalized_response_ids, got_transcript_delta
                nonlocal current_item_id, item_play_start_s
                async for raw_msg in ws:
                    event = json.loads(raw_msg)
                    etype = event.get("type", "")
                    now = None
                    if stream_t0["t"] is not None:
                        now = time.monotonic() - stream_t0["t"]

                    if etype == "response.output_audio.delta" or etype == "response.audio.delta":
                        pcm_bytes = base64.b64decode(event.get("delta", ""))
                        item_id = event.get("item_id")
                        if item_id is not None and item_id != current_item_id:
                            current_item_id = item_id
                            item_play_start_s = (recorder.played_speech_s()
                                                 + recorder.queued_unplayed_s())
                        await recorder.add(pcm_bytes, response_id=event.get("response_id"))
                        event_logger.log("recv", etype, n_bytes=len(pcm_bytes))

                    elif etype in TRANSCRIPT_DELTA_TYPES:
                        text = event.get("delta", "")
                        if text and now is not None:
                            got_transcript_delta = True
                            self.text_tokens.append((now, text))
                            ctrl = controller_holder["controller"]
                            if ctrl is not None:
                                ctrl.on_text(text, at_s=now)
                        event_logger.log("recv", etype, {"text": text})

                    elif etype in TRANSCRIPT_DONE_TYPES:
                        # Qwen (and some GA-schema providers) emit the full
                        # utterance transcript only at .done, with no
                        # incremental .delta. Skip if deltas already arrived
                        # so GPT/Grok don't concatenate the same turn twice.
                        text = event.get("transcript") or event.get("text") or ""
                        if text and now is not None and not got_transcript_delta:
                            self.text_tokens.append((now, text))
                            ctrl = controller_holder["controller"]
                            if ctrl is not None:
                                ctrl.on_text(text, at_s=now)
                        event_logger.log("recv", etype, {"text": text})

                    elif etype == "response.created":
                        response_active = True
                        current_response_id = event.get("response", {}).get("id")
                        got_transcript_delta = False
                        event_logger.log("recv", etype, {"response_id": current_response_id})

                    elif etype == "input_audio_buffer.speech_started":
                        # The provider's OWN server-side VAD has just told us
                        # the user is speaking. Stopping playback of audio the
                        # listener has not heard yet is the playback layer's
                        # job in any real deployment -- it is NOT us deciding
                        # the model yields the floor (that would be
                        # `response.cancel`, which we still never send).
                        #
                        # This used to be gated on `response_active`, i.e. on
                        # the response still being mid-generation. But these
                        # APIs emit audio 4-5.5x faster than realtime, so
                        # `response.done` usually lands BEFORE the cue does and
                        # the gate was already False: measured over
                        # respond_and_recover, 353/1092 GPT samples where the
                        # cue arrived while the model was audibly speaking took
                        # the else-branch, and Grok took it 724/724 times (its
                        # server never reports `cancelled`). Those samples then
                        # played the whole queued backlog through the cue, so
                        # the pre-cue turn and the recovery reply merged into
                        # one continuous segment with no onset in the scoring
                        # window: Recover Immediate was 15.9% for the un-flushed
                        # GPT group vs 78.9% for the flushed one. Whether the
                        # server happens to still be generating is a function
                        # of its generation speed, not of the model's duplex
                        # behaviour, so it must not gate this.
                        played_in_item_ms = int(1000 * max(
                            0.0, recorder.played_speech_s() - item_play_start_s
                        ))
                        # Only a barge-in if there is model speech to cut into:
                        # either audible right now, or queued and unheard.
                        # Otherwise (e.g. the VAD firing on the lead-in itself,
                        # before the model has said anything) there is nothing
                        # to truncate and interrupt() would only add a spurious
                        # entry to recorder.interruptions.
                        is_barge_in = (recorder.queued_unplayed_s() > 0.0
                                       or recorder.current_speaking_duration() > 0.0)
                        if is_barge_in:
                            # response_id is deliberately NOT passed. Dropping
                            # the queued backlog is what lets playback catch up
                            # to generation, so that from here on "is the model
                            # still speaking" reads the model's own post-cue
                            # decision instead of buffer drainage. Blacklisting
                            # the response id (R-03) would defeat exactly that:
                            # it also discards the response's CONTINUED deltas,
                            # which is how these providers express "I am keeping
                            # the floor" -- measured on hold_floor, 72-93% of
                            # post-cue audio is more deltas of the same
                            # response and 0% is a new one. The blacklist
                            # belongs on the provider's own cancellation (see
                            # response.done below), which is what R-03 was
                            # written for; predicting it from user speech onset
                            # is only correct for providers that always cancel,
                            # and Grok never reports cancelled at all.
                            recorder.interrupt(reason="input_speech_during_response")
                            ctrl = controller_holder["controller"]
                            if ctrl is not None:
                                ctrl.on_interrupt()
                        # `expecting_cancel` stays coupled to generation, not
                        # to the flush: setting it for a response that already
                        # reached response.done(completed) would corrupt the
                        # terminal-status classification below.
                        if response_active and is_barge_in:
                            expecting_cancel = True
                        event_logger.log("recv", etype, {
                            "barge_in": is_barge_in,
                            "response_active": response_active,
                            "flushed_bytes": recorder.bytes_flushed,
                            "played_in_item_ms": played_in_item_ms,
                        })
                        if is_barge_in:
                            response_active = False

                        # Sync the server's idea of what was heard with the
                        # client's playback, where the protocol allows it.
                        # audio_end_ms=0 is rejected ("Only model output audio
                        # messages can be truncated"), so skip that case.
                        if (is_barge_in
                                and self.SUPPORTS_ITEM_TRUNCATE
                                and current_item_id is not None
                                and played_in_item_ms > 0):
                            msg = {
                                "type": "conversation.item.truncate",
                                "item_id": current_item_id,
                                "content_index": 0,
                                "audio_end_ms": played_in_item_ms,
                            }
                            try:
                                await ws.send(json.dumps(msg))
                                event_logger.log("send", "conversation.item.truncate", {
                                    "item_id": current_item_id,
                                    "audio_end_ms": played_in_item_ms,
                                })
                            except Exception as exc:
                                event_logger.log("send", "conversation.item.truncate", {
                                    "item_id": current_item_id,
                                    "audio_end_ms": played_in_item_ms,
                                    "error": repr(exc),
                                })

                    elif etype == "response.done":
                        response = event.get("response", {})
                        status = response.get("status", "unknown")
                        resp_id = response.get("id")
                        response_active = False

                        if resp_id is not None and resp_id in finalized_response_ids:
                            # Stale duplicate terminal event for a response_id
                            # that already got its real terminal status --
                            # see finalized_response_ids' own comment above
                            # for the confirmed root cause. Logged for
                            # visibility, never re-classified/re-errored.
                            expecting_cancel = False
                            event_logger.log("recv", etype, {
                                "status": status, "status_label": "duplicate_terminal_ignored",
                                "response_id": resp_id,
                                "status_details": response.get("status_details"),
                            })
                        else:
                            if resp_id is not None:
                                finalized_response_ids.add(resp_id)
                            details = response.get("status_details") or {}
                            expected = is_expected_cancel(
                                expecting_cancel, status, details,
                            )
                            label, is_error = classify_terminal_response_status(
                                status, expected_cancel=expected)
                            expecting_cancel = False
                            if status == "cancelled" and resp_id is not None:
                                # R-03, now anchored on the provider's own
                                # signal rather than predicted from user speech
                                # onset: this response really is dead, so a
                                # delta still in flight for it must not resume
                                # playback of audio the cancellation suppressed.
                                recorder.interrupt(reason="provider_cancelled",
                                                   response_id=resp_id)
                            event_logger.log("recv", etype, {
                                "status": status, "status_label": label,
                                "response_id": resp_id,
                                "status_details": response.get("status_details"),
                            })
                            if is_error:
                                provider_errors.append({
                                    "message": "response.done terminal status=" + repr(status) + " (" + label + ")",
                                    "response_id": resp_id,
                                    "status_details": response.get("status_details"),
                                })

                    elif etype == "error":
                        err = event.get("error", {})
                        msg = err.get("message", str(event))
                        code = err.get("code")
                        event_logger.log("recv", "error", {"message": msg, "code": code})
                        if code == "input_audio_buffer_commit_empty":
                            pass
                        else:
                            provider_errors.append({"message": msg, "code": code})
                            if self.verbose:
                                print("  [ERROR] " + msg, file=sys.stderr)

                    elif etype.endswith(".delta"):
                        pass

                    else:
                        event_logger.log("recv", etype)

            recv_task = asyncio.create_task(receive_loop())

            loop = asyncio.get_event_loop()
            send_t0 = await start_streaming()
            stream_t0["t"] = send_t0

            def now_fn():
                return time.monotonic() - stream_t0["t"]

            source = DynamicPCMSource(initial_pcm=input_pcm.astype(np.int16).tobytes())

            controller = LiveTriggerController(
                recorder=recorder,
                source=source,
                cue_pcm=self.cue_pcm,
                now_fn=now_fn,
                reference_context=self.reference_context,
                client=self.llm_client,
                lead_in_duration_s=input_duration,
                event_logger=event_logger,
                speech_chars_per_s=speech_rate_for(self.MODEL_SLUG),
            )
            controller_holder["controller"] = controller
            trigger_task = asyncio.create_task(controller.run())

            def _on_trigger_done(t):
                source.finish()

            trigger_task.add_done_callback(_on_trigger_done)

            cue_drained_at = None
            i = 0
            try:
                while True:
                    chunk = source.pull(chunk_bytes)
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }))
                    event_logger.log("send", "input_audio_buffer.append", n_bytes=len(chunk))

                    if source.is_done and cue_drained_at is None:
                        cue_drained_at = now_fn()
                        event_logger.log("send", "cue_drained", {"t": cue_drained_at})
                    if cue_drained_at is not None and now_fn() - cue_drained_at >= POST_CUE_OBSERVATION_S:
                        break

                    i += 1
                    await sleep_until_deadline(loop, send_t0 + i * (self.CHUNK_MS / 1000))
            finally:
                try:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    event_logger.log("send", "input_audio_buffer.commit")
                except Exception:
                    pass

                if not trigger_task.done():
                    trigger_task.cancel()
                    try:
                        await trigger_task
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
                    try:
                        self.trigger_result = trigger_task.result()
                    except Exception as e:
                        self.trigger_result = {"error": type(e).__name__ + ": " + str(e)}

                self.simulator_speech_segments = [
                    {
                        "start": 0.0,
                        "end": round(input_duration, 4),
                        "text": self.reference_context.get("transcript_user", ""),
                        "source": "lead_in",
                    }
                ]
                t_trigger = self.trigger_result.get("t_user_int_start")
                if t_trigger is not None:
                    cue_duration = len(self.cue_pcm) / 2 / self.API_SR
                    self.simulator_speech_segments.append({
                        "start": round(t_trigger, 4),
                        "end": round(t_trigger + cue_duration, 4),
                        "text": self.reference_context.get("reference_interruption_text", ""),
                        "source": "cue",
                    })

                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

        if provider_errors:
            raise ProviderError(
                str(len(provider_errors)) + " server error event(s); first: " + str(provider_errors[0])
            )
