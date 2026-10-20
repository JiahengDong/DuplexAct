#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/realtime_api_interrupt_common.py

interrupt_user sibling of realtime_api_stop_common.RealtimeAPIStopClientBase.
Never imported by, never edits, the stop file. Same GA Realtime WS receive
loop; InterruptMonitorController instead of LiveTriggerController.
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, classify_terminal_response_status, DynamicPCMSource,
    is_expected_cancel,
)
from interrupt_controller import InterruptMonitorController  # noqa: E402
from trigger_controller import speech_rate_for  # noqa: E402
from interrupt_user_common import (  # noqa: E402
    POST_BRANCH_OBSERVATION_S, build_interrupt_simulator_speech_segments,
)
from realtime_api_stop_common import (  # noqa: E402  -- read-only constants / ProviderError
    MAX_WS_MESSAGE_BYTES, TRANSCRIPT_DELTA_TYPES, TRANSCRIPT_DONE_TYPES, ProviderError,
)


class RealtimeAPIInterruptClientBase(RealtimeClientBase):
    WS_URL = ""
    CHUNK_MS = 20
    API_SR = 24000
    # Same provider split as realtime_api_stop_common: only OpenAI accepts
    # conversation.item.truncate. See that file's comment for why xAI and
    # DashScope must not be sent one.
    SUPPORTS_ITEM_TRUNCATE = False

    def __init__(self, api_key, model_name, system_prompt, verbose,
                 branch_acknowledged_pcm, branch_ignored_pcm, reference_context,
                 llm_client=None):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.branch_acknowledged_pcm = branch_acknowledged_pcm
        self.branch_ignored_pcm = branch_ignored_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.monitor_result = {}
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
        # Barge-in bookkeeping, same as realtime_api_stop_common: the assistant
        # item currently being played out, and where in played_speech_s its
        # audio began, so conversation.item.truncate can report heard-ms.
        current_item_id = None
        item_play_start_s = 0.0
        got_transcript_delta = False
        provider_errors = []
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
                        # Identical policy to realtime_api_stop_common's handler,
                        # and for the same reason: the provider has told us the
                        # user is speaking, so its server-side context is
                        # truncated there and our queued-but-unheard backlog is
                        # our buffering, not the model. That the model is the one
                        # meant to be interrupting in THIS action changes neither
                        # what the provider did nor what the listener hears. If
                        # the provider then abandons the response because the
                        # user kept talking, that is the model failing to sustain
                        # its interruption -- a real outcome, and one the backlog
                        # would otherwise paper over.
                        played_in_item_ms = int(1000 * max(
                            0.0, recorder.played_speech_s() - item_play_start_s
                        ))
                        is_barge_in = (recorder.queued_unplayed_s() > 0.0
                                       or recorder.current_speaking_duration() > 0.0)
                        if is_barge_in:
                            # response_id deliberately NOT passed: blacklisting
                            # here would also discard the response's CONTINUED
                            # deltas, which is how a model expresses that it is
                            # holding on to its interruption. The blacklist lives
                            # on the provider's own cancellation in response.done.
                            recorder.interrupt(reason="input_speech_during_response")
                            ctrl = controller_holder["controller"]
                            if ctrl is not None:
                                ctrl.on_interrupt()
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
                                # The provider's own signal that this response is
                                # dead, so a delta still in flight for it must not
                                # resume playback of suppressed audio. Anchored
                                # here rather than predicted from speech onset,
                                # since Grok never reports cancelled at all.
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
                        if code != "input_audio_buffer_commit_empty":
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
            monitor = InterruptMonitorController(
                recorder=recorder, source=source,
                user_speech_duration=input_duration,
                branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                branch_ignored_pcm=self.branch_ignored_pcm,
                now_fn=now_fn, reference_context=self.reference_context,
                llm_client=self.llm_client, event_logger=event_logger,
                speech_chars_per_s=speech_rate_for(self.MODEL_SLUG),
            )
            controller_holder["controller"] = monitor
            monitor_task = asyncio.create_task(monitor.run())
            monitor_task.add_done_callback(lambda t: source.finish())

            branch_drained_at = None
            i = 0
            try:
                while True:
                    chunk = source.pull(chunk_bytes)
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }))
                    event_logger.log("send", "input_audio_buffer.append", n_bytes=len(chunk))
                    if source.is_done and branch_drained_at is None:
                        branch_drained_at = now_fn()
                        event_logger.log("send", "branch_drained", {"t": branch_drained_at})
                    if (branch_drained_at is not None
                            and now_fn() - branch_drained_at >= POST_BRANCH_OBSERVATION_S):
                        break
                    i += 1
                    await sleep_until_deadline(loop, send_t0 + i * (self.CHUNK_MS / 1000))
            finally:
                try:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    event_logger.log("send", "input_audio_buffer.commit")
                except Exception:
                    pass
                if not monitor_task.done():
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
                    try:
                        self.monitor_result = monitor_task.result()
                    except Exception as e:
                        self.monitor_result = {"error": type(e).__name__ + ": " + str(e)}

                self.simulator_speech_segments = build_interrupt_simulator_speech_segments(
                    monitor_result=self.monitor_result,
                    reference_context=self.reference_context,
                    input_duration=input_duration,
                    branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                    branch_ignored_pcm=self.branch_ignored_pcm,
                    pcm_sr=self.API_SR,
                )
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

        if provider_errors:
            raise ProviderError(
                str(len(provider_errors)) + " server error event(s); first: " + str(provider_errors[0])
            )
