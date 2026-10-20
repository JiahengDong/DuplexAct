#!/usr/bin/env python3
"""
direct_replay/inference/infer_qwen_realtime.py
==================================================
Qwen3.5-Omni-Flash-Realtime client (Alibaba Cloud Model Studio / DashScope,
Singapore region), built on the shared `RealtimeClientBase` /
`SynchronizedRecorder` infrastructure in realtime_common.py.

Protocol (per Alibaba Cloud Model Studio "Qwen-Omni-Realtime" docs, Aug 2026)
closely mirrors OpenAI's Realtime API, with these differences:
  - WS URL is workspace-specific:
      wss://{host}/api-ws/v1/realtime?model=qwen3.5-omni-flash-realtime
    where {host} is a per-workspace domain (see DASHSCOPE_WS_HOST env var),
    NOT a shared public endpoint. Region is baked into the host
    (ap-southeast-1 = Singapore; cn-beijing = China). Only Singapore and
    China (Beijing) regions exist for this model as of Aug 2026.
  - session.update fields are flat: "modalities" (not "output_modalities"),
    "input_audio_format"/"output_audio_format": "pcm" (not nested under
    audio.input/output.format), "voice" at top level, "instructions" for the
    system prompt.
  - turn_detection: {"type": "semantic_vad", "threshold", "silence_duration_ms"}
    -- semantic_vad is the officially recommended mode for this model family
    (helps avoid false-positive turn triggers from backchannel/background
    noise -- directly relevant to our wait/backchannel/ignore actions).
  - Model audio arrives via "response.audio.delta" (vs OpenAI's
    "response.output_audio.delta").
  - Client-side barge-in via response.cancel: DashScope's own reference
    client explicitly sends "response.cancel" on speech_started to interrupt
    an in-progress response -- there is no documented server-side
    auto-truncation for this model family. An earlier version of this client
    deliberately did NOT send response.cancel (reasoning that doing so would
    be "our own scaffolding" rather than the model's behavior), but that
    made Qwen's client contract asymmetric with GPT/Grok (both of which DO
    get their responses truncated on barge-in, whether by server auto-cancel
    or our own interrupt-on-signal) and diverged from Alibaba's own reference
    integration -- see realtime_direct_replay_code_review.pdf F-09. We now
    send response.cancel ourselves on speech_started-during-response, exactly
    matching the reference client, and call recorder.interrupt() at the same
    moment, consistent with every other provider in this benchmark.
  - Terminal response status is now classified (code review R-06): Qwen's
    `response.done` is OpenAI-Realtime-shaped too (carries a `status` field)
    but this client used to ignore it entirely, only using the event to flip
    `response_active` back to False. It now goes through the same
    `classify_terminal_response_status()` as GPT/Grok -- a "failed" or
    "incomplete" terminal response, or a "cancelled" one we didn't already
    explain via our own response.cancel, fails the sample.
  - Late deltas can no longer resume a cancelled response (code review
    R-03): audio-delta events' `response_id` is now passed to
    `recorder.add()`, and our own `response.cancel` send now also tells
    `recorder.interrupt()` which response id it's cancelling. A straggler
    delta for that id that was already in flight before DashScope
    processed our cancel is dropped instead of resuming playback -- this
    matters more for Qwen than GPT/Grok since our cancellation is
    client-initiated (see module docstring above) rather than an automatic
    server-side truncation, so the race window is real.

Requirements:
    DASHSCOPE_API_KEY   workspace-scoped API key (starts with "sk-ws-...")
    DASHSCOPE_WS_HOST    workspace-specific host, e.g.
                         ws-xxxxxxxxxxxxxxxx.ap-southeast-1.maas.aliyuncs.com
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

try:
    import websockets
except ImportError as e:
    # Raise rather than sys.exit(1) (F-20): SystemExit isn't caught by
    # model_registry.available_models()'s `except Exception`.
    raise ImportError("websockets not installed. Run: pip install websockets") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import (
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, classify_terminal_response_status,
    is_expected_cancel,
)

DEFAULT_MODEL = "qwen3.5-omni-flash-realtime"
DEFAULT_VOICE = "Ethan"

API_SR           = 16_000   # input PCM16 must be 16kHz mono
OUTPUT_SR        = 24_000   # output PCM16 is 24kHz mono
BYTES_PER_SAMPLE = 2

CHUNK_MS      = 20
CHUNK_SAMPLES = API_SR * CHUNK_MS // 1000
CHUNK_BYTES   = CHUNK_SAMPLES * BYTES_PER_SAMPLE

# See infer_gpt_realtime.py's MAX_WS_MESSAGE_BYTES docstring (F-22).
MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024


class ProviderError(RuntimeError):
    """Raised when DashScope sends an explicit `error` event (see F-04)."""

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class QwenRealtimeClient(RealtimeClientBase):
    MODEL_SLUG = "qwen3.5-omni-flash-realtime"
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = API_SR

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool = False, ws_host: str | None = None,
                 voice: str = DEFAULT_VOICE):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.ws_host = ws_host or os.getenv("DASHSCOPE_WS_HOST", "")
        self.voice = voice
        if not self.ws_host:
            raise RuntimeError(
                "DASHSCOPE_WS_HOST not set. Qwen realtime requires a "
                "workspace-specific host (see module docstring)."
            )

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder,
                                   event_logger: EventLogger, start_streaming) -> None:
        url = f"wss://{self.ws_host}/api-ws/v1/realtime?model={self.model_name}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response_active = False
        expecting_cancel = False   # R-06: set when barge-in already recorded an interrupt
        current_response_id: Optional[str] = None   # R-03
        provider_errors: list[dict] = []

        async with websockets.connect(
            url,
            additional_headers=headers,
            max_size=MAX_WS_MESSAGE_BYTES,
            ping_interval=20,
            ping_timeout=30,
        ) as ws:

            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": self.voice,
                    "instructions": self.system_prompt,
                    "input_audio_format": "pcm",
                    "output_audio_format": "pcm",
                    "turn_detection": {
                        "type": "semantic_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 800,
                    },
                },
            }
            await ws.send(json.dumps(session_update))
            event_logger.log("send", "session.update")

            async def receive_loop():
                nonlocal response_active, expecting_cancel, current_response_id
                async for raw_msg in ws:
                    event = json.loads(raw_msg)
                    etype = event.get("type", "")

                    if etype == "response.audio.delta":
                        pcm_bytes = base64.b64decode(event.get("delta", ""))
                        # R-03: see infer_gpt_realtime.py -- pass the delta's
                        # own response_id so a straggler for an
                        # already-cancelled response is dropped.
                        await recorder.add(pcm_bytes, response_id=event.get("response_id"))
                        event_logger.log("recv", etype, n_bytes=len(pcm_bytes))

                    elif etype == "response.created":
                        response_active = True
                        current_response_id = event.get("response", {}).get("id")
                        event_logger.log("recv", etype, {"response_id": current_response_id})

                    elif etype == "input_audio_buffer.speech_started":
                        if response_active:
                            # Client-side barge-in, matching Alibaba's own
                            # reference client (see module docstring / F-09):
                            # explicitly cancel the in-progress response and
                            # clear anything already queued for playback.
                            await ws.send(json.dumps({"type": "response.cancel"}))
                            event_logger.log("send", "response.cancel",
                                              {"trigger": "speech_started_during_response"})
                            recorder.interrupt(reason="input_speech_during_response",
                                                response_id=current_response_id)
                            response_active = False
                            expecting_cancel = True   # R-06: our own response.cancel
                            # should make the next response.done "cancelled"
                            event_logger.log("recv", etype, {"barge_in": True})
                        else:
                            event_logger.log("recv", etype, {"barge_in": False})

                    elif etype == "response.done":
                        response = event.get("response", {})
                        status = response.get("status", "unknown")
                        response_active = False
                        # R-06: classify instead of just logging. Qwen often
                        # cancels with reason=turn_detected before
                        # speech_started; is_expected_cancel covers that
                        # (same as the simulator clients).
                        label, is_error = classify_terminal_response_status(
                            status, expected_cancel=is_expected_cancel(
                                expecting_cancel, status,
                                response.get("status_details"),
                            ))
                        expecting_cancel = False
                        event_logger.log("recv", etype, {
                            "status": status,
                            "status_label": label,
                            "response_id": response.get("id"),
                            "status_details": response.get("status_details"),
                        })
                        if is_error:
                            provider_errors.append({
                                "message": f"response.done terminal status={status!r} "
                                           f"({label})",
                                "response_id": response.get("id"),
                                "status_details": response.get("status_details"),
                            })

                    elif etype == "error":
                        err = event.get("error", {})
                        msg = err.get("message", str(event))
                        event_logger.log("recv", "error", {"message": msg, "code": err.get("code")})
                        provider_errors.append({"message": msg, "code": err.get("code")})
                        if self.verbose:
                            print(f"  [ERROR] {msg}", file=sys.stderr)

                    elif etype.endswith(".delta"):
                        pass  # ignore transcript deltas etc.

                    else:
                        event_logger.log("recv", etype)

            recv_task = asyncio.create_task(receive_loop())

            # t=0 for the recorder/event-log starts at the first audio chunk,
            # not at connect/session-update time.
            loop = asyncio.get_event_loop()
            # send_t0 IS stream_start["t"] (see start_streaming()'s
            # docstring in realtime_common.py) -- shared, not a second
            # independently-measured clock read.
            send_t0 = await start_streaming()
            pcm_bytes_all = input_pcm.tobytes()
            for i, off in enumerate(range(0, len(pcm_bytes_all), CHUNK_BYTES)):
                chunk = pcm_bytes_all[off: off + CHUNK_BYTES]
                if len(chunk) < CHUNK_BYTES:
                    chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
                await ws.send(json.dumps({
                    "type":  "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))
                event_logger.log("send", "input_audio_buffer.append", n_bytes=len(chunk))
                # Absolute deadline instead of a fixed post-send sleep (F-10).
                await sleep_until_deadline(loop, send_t0 + (i + 1) * (CHUNK_MS / 1000))

            # No manual commit/response.create needed: semantic_vad governs
            # turn segmentation and response creation autonomously, same
            # philosophy as the GPT-Realtime client.

            # Hard cutoff at input end: see infer_gpt_realtime.py.
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        if provider_errors:
            raise ProviderError(
                f"{len(provider_errors)} server error event(s); first: {provider_errors[0]}"
            )


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = QwenRealtimeClient(
        api_key=args.api_key,
        model_name=args.model,
        system_prompt=args.system_prompt,
        verbose=args.verbose,
        ws_host=args.ws_host,
        voice=args.voice,
    )
    result = await client.run_sample(
        input_wav=args.input_wav,
        output_wav=args.output_wav,
        events_path=args.events_path,
        input_duration=input_duration,
    )
    print(json.dumps(result, indent=2))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_wav", type=Path, required=True)
    parser.add_argument("--output_wav", type=Path, required=True)
    parser.add_argument("--events_path", type=Path, required=True)
    parser.add_argument("--input_duration", type=float, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--ws_host", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not args.api_key:
        print("Error: no DASHSCOPE_API_KEY found (env / .env / --api_key).", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
