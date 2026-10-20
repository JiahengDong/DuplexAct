#!/usr/bin/env python3
"""
Shared base class for simulator-replay clients against WebSocket APIs that use
the OpenAI GA Realtime schema (OpenAI GPT-Realtime, xAI Grok Voice): session.update,
response.created, input_audio_buffer.speech_started, response.done,
response.output_audio.delta / response.audio.delta, error.

Each subclass must set:
  - MODEL_SLUG, OUTPUT_SR / INPUT_SR (as for RealtimeClientBase)
  - WS_URL, without the `?model=` query string (appended from self.model_name)
  - CHUNK_MS / API_SR (20 ms at 24 kHz by default)
  - build_session_update(self) -> dict, the provider-specific session config

The simulated user is polled every trigger_controller.POLL_INTERVAL_S, the same
cadence as every other simulator client. Transcript deltas (any type in
TRANSCRIPT_DELTA_TYPES) are fed to the trigger controller via ctrl.on_text().
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


# Playback policy for every action: on barge-in, drop the queued-but-unheard
# backlog. These APIs emit audio several times faster than realtime, so when the
# cue arrives there are seconds of audio generated before the model could know
# about it. Dropping it makes playback catch up to generation, so from then on
# whether the listener hears speech reflects the model's own post-cue decision
# rather than buffer drainage. The flush does not blacklist the response id;
# see the speech_started handler.


class RealtimeAPIStopClientBase(RealtimeClientBase):
    WS_URL = ""
    CHUNK_MS = 20
    API_SR = 24000
    # Whether the provider accepts `conversation.item.truncate` to report how
    # much of an assistant item the listener heard. Only OpenAI does: on xAI it
    # desyncs the server-side item list (livekit/agents#6391), and DashScope's
    # Qwen-Omni has no truncate event.
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
        # Barge-in bookkeeping. `current_item_id` is the assistant item being
        # played out; `item_play_start_s` is where its playback begins on the
        # recorder's played-speech axis, taken as (played) + (queued) when its
        # first delta lands, since its predecessor may still be draining.
        # audio_end_ms for conversation.item.truncate is the difference.
        current_item_id = None
        item_play_start_s = 0.0
        got_transcript_delta = False
        provider_errors = []
        # OpenAI can emit a second, stale `response.done` for a response_id that
        # already got its terminal event: response.done(completed), then a
        # duplicate response.done(cancelled, reason=turn_detected) once the cue
        # trips the server VAD. The first terminal status per response_id is
        # authoritative; later ones are logged but not classified as errors.
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
                        # The provider's server-side VAD reports user speech.
                        # Stopping playback of unheard audio is the playback
                        # layer's job; it does not make the model yield the
                        # floor (that would be `response.cancel`, never sent).
                        #
                        # Not gated on `response_active`: these APIs generate
                        # faster than realtime, so `response.done` often lands
                        # before the cue while audio is still playing. Whether
                        # the server is still generating reflects its speed,
                        # not its duplex behaviour.
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
                            # response_id is not passed: blacklisting it would
                            # also discard the response's continued deltas,
                            # which is how these providers keep the floor.
                            # The blacklist is applied only on the provider's
                            # own cancellation (see response.done below).
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
                            # Stale duplicate terminal event; see
                            # finalized_response_ids above.
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
                                # The response is cancelled, so a delta still in
                                # flight for it must not resume playback.
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
