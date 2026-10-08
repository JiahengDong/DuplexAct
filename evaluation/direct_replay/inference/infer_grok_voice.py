#!/usr/bin/env python3
"""
direct_replay/inference/infer_grok_voice.py
==============================================
xAI Grok Voice (Speech-to-Speech) client, built on the shared
`RealtimeClientBase` / `SynchronizedRecorder` infrastructure in
realtime_common.py.

Protocol (per docs.x.ai/developers/model-capabilities/audio/speech-to-speech,
Aug 2026) is essentially the same GA schema as OpenAI's Realtime API:
  - wss://api.x.ai/v1/realtime?model=grok-voice-latest
  - session.update: nested "audio.input/output.format.type/rate" (same shape
    as OpenAI's GA session.audio), "turn_detection": {"type": "server_vad",
    "silence_duration_ms": ...}, "instructions", "voice".
  - Model audio arrives via "response.output_audio.delta" (alias:
    "response.audio.delta", handled generically like the GPT client).
  - Docs explicitly recommend server_vad for "automatic, natural barge-in",
    so -- like GPT-Realtime -- we treat input_audio_buffer.speech_started
    while a response is in progress as a genuine interruption and call
    recorder.interrupt(); no explicit response.cancel needed from us.
  - No forced response.create: turn_detection (server_vad) governs response
    creation autonomously from the continuously appended input stream.

Correctness fixes from realtime_direct_replay_code_review.pdf:
  - Server `error` events now raise ProviderError instead of only being
    logged, so an API-side failure is recorded in the sample's `errors` list
    rather than potentially looking like clean model silence (F-04).
  - Post-input cutoff (design decision, supersedes an earlier drain_until_quiet
    heuristic): reception is cancelled immediately once input playback
    (including padding) ends -- a hard, fixed window boundary applied
    uniformly across every provider and both model classes (F-01 revisited).
  - Send pacing uses absolute deadlines instead of a fixed post-send sleep,
    so per-frame overhead can't accumulate into replay drift (F-10).
  - Terminal response status is now classified (code review R-06): this
    client used to ignore `response.done`'s `status` entirely, only using
    the event to flip `response_active` back to False. It now goes through
    the same `classify_terminal_response_status()` as GPT/Qwen -- a
    "failed" or "incomplete" terminal response, or a "cancelled" one that
    wasn't already explained by a recorded barge-in, fails the sample.
  - Late deltas can no longer resume a cancelled response (code review
    R-03): audio-delta events' `response_id` is now passed to
    `recorder.add()`, and `recorder.interrupt()` is told which response id
    server_vad just truncated, so a straggler delta for that id is dropped
    instead of resuming playback.
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

from realtime_common import (
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, classify_terminal_response_status,
    is_expected_cancel,
)

REALTIME_WS_URL = "wss://api.x.ai/v1/realtime"
DEFAULT_MODEL   = "grok-voice-latest"
DEFAULT_VOICE   = "eve"

API_SR           = 24_000   # xAI's recommended default PCM16 rate (both directions)
BYTES_PER_SAMPLE = 2

CHUNK_MS      = 20
CHUNK_SAMPLES = API_SR * CHUNK_MS // 1000
CHUNK_BYTES   = CHUNK_SAMPLES * BYTES_PER_SAMPLE

# See infer_gpt_realtime.py's MAX_WS_MESSAGE_BYTES docstring (F-22).
MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024


class ProviderError(RuntimeError):
    """Raised when xAI sends an explicit `error` event (see F-04)."""

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class GrokVoiceClient(RealtimeClientBase):
    MODEL_SLUG = "grok-voice"
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool = False, voice: str = DEFAULT_VOICE):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.voice = voice

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder,
                                   event_logger: EventLogger, start_streaming) -> None:
        url = f"{REALTIME_WS_URL}?model={self.model_name}"
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
                    "voice": self.voice,
                    "instructions": self.system_prompt,
                    # No threshold/silence_duration_ms override: use xAI's
                    # documented defaults (threshold 0.85, energy-based
                    # server_vad) so we aren't hand-tuning sensitivity in
                    # either direction relative to how a real user would
                    # configure the API out of the box.
                    "turn_detection": {
                        "type": "server_vad",
                    },
                    "audio": {
                        "input":  {"format": {"type": "audio/pcm", "rate": API_SR}},
                        "output": {"format": {"type": "audio/pcm", "rate": API_SR}},
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

                    if etype in ("response.output_audio.delta", "response.audio.delta"):
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
                            # Per docs: server_vad gives "automatic, natural
                            # barge-in" -- the server truncates the in-progress
                            # response on its own when new speech arrives.
                            recorder.interrupt(reason="input_speech_during_response",
                                                response_id=current_response_id)
                            response_active = False
                            expecting_cancel = True   # R-06: server_vad's own
                            # auto-truncation should make the next
                            # response.done "cancelled"
                            event_logger.log("recv", etype, {"barge_in": True})
                        else:
                            event_logger.log("recv", etype, {"barge_in": False})

                    elif etype == "response.done":
                        response = event.get("response", {})
                        status = response.get("status", "unknown")
                        response_active = False
                        # R-06: see infer_gpt_realtime.py -- classify instead
                        # of just logging so failed/incomplete/unexplained
                        # cancelled responses fail the sample.
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

            # No manual commit/response.create: server_vad governs turn
            # segmentation and response creation autonomously.

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

    client = GrokVoiceClient(
        api_key=args.api_key,
        model_name=args.model,
        system_prompt=args.system_prompt,
        verbose=args.verbose,
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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.api_key = args.api_key or os.getenv("XAI_API_KEY", "")
    if not args.api_key:
        print("Error: no XAI_API_KEY found (env / .env / --api_key).", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
