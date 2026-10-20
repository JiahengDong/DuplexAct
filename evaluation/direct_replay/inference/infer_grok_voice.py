#!/usr/bin/env python3
"""
xAI Grok Voice (speech-to-speech) client, built on `RealtimeClientBase` /
`SynchronizedRecorder` in realtime_common.py.

The protocol (docs.x.ai speech-to-speech) follows OpenAI's GA Realtime schema:
  - wss://api.x.ai/v1/realtime?model=grok-voice-latest
  - session.update with nested "audio.input/output.format", "turn_detection"
    ({"type": "server_vad"}), "instructions" and "voice".
  - Model audio arrives as "response.output_audio.delta" (alias
    "response.audio.delta").
  - server_vad handles barge-in: input_audio_buffer.speech_started during a
    response is treated as an interruption (recorder.interrupt()); no
    response.cancel or response.create is sent.

Behaviour:
  - Server `error` events raise ProviderError, so API failures are recorded in
    the sample's `errors` rather than looking like model silence.
  - Reception stops as soon as input playback (including padding) ends.
  - Sends are paced on absolute deadlines so per-frame overhead does not drift.
  - `response.done` status goes through `classify_terminal_response_status()`:
    failed, incomplete, or unexplained cancelled responses fail the sample.
  - Audio deltas carry their `response_id`, so a late delta for an interrupted
    response is dropped instead of resuming playback.
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
    # Raise rather than sys.exit(1): SystemExit isn't caught by
    # model_registry.available_models()'s `except Exception`.
    raise ImportError("websockets not installed. Run: pip install websockets") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

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

# See MAX_WS_MESSAGE_BYTES in infer_gpt_realtime.py.
MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024


class ProviderError(RuntimeError):
    """Raised when xAI sends an explicit `error` event."""

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
        expecting_cancel = False   # set when barge-in already recorded an interrupt
        current_response_id: Optional[str] = None
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
                    # documented server_vad defaults.
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
                        # Pass the delta's response_id so a straggler for an
                        # already-cancelled response is dropped.
                        await recorder.add(pcm_bytes, response_id=event.get("response_id"))
                        event_logger.log("recv", etype, n_bytes=len(pcm_bytes))

                    elif etype == "response.created":
                        response_active = True
                        current_response_id = event.get("response", {}).get("id")
                        event_logger.log("recv", etype, {"response_id": current_response_id})

                    elif etype == "input_audio_buffer.speech_started":
                        if response_active:
                            # server_vad truncates the in-progress response
                            # on its own when new speech arrives.
                            recorder.interrupt(reason="input_speech_during_response",
                                                response_id=current_response_id)
                            response_active = False
                            expecting_cancel = True   # next response.done
                            # should be "cancelled"
                            event_logger.log("recv", etype, {"barge_in": True})
                        else:
                            event_logger.log("recv", etype, {"barge_in": False})

                    elif etype == "response.done":
                        response = event.get("response", {})
                        status = response.get("status", "unknown")
                        response_active = False
                        # Failed, incomplete, or unexplained cancelled
                        # responses fail the sample.
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
            # send_t0 is the same clock origin as stream_start["t"]
            # (see start_streaming() in realtime_common.py).
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
                # Absolute deadline so per-chunk overhead does not accumulate.
                await sleep_until_deadline(loop, send_t0 + (i + 1) * (CHUNK_MS / 1000))

            # No manual commit/response.create: server_vad governs turn
            # segmentation and response creation autonomously.

            # Hard cutoff at input end.
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
