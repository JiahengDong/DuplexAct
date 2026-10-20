#!/usr/bin/env python3
"""
OpenAI GPT-Realtime client (WebSocket), built on the shared
`RealtimeClientBase` / `SynchronizedRecorder` infrastructure in
realtime_common.py.

  - No explicit `response.create`: server turn detection (semantic_vad)
    decides when to respond, as a real full-duplex system would. A forced
    response would defeat wait/ignore/backchannel tests.
  - The receive loop runs for the whole input_duration, so multi-turn
    behavior (repeated backchannels, floor-holding) is captured.
  - `input_audio_buffer.speech_started` triggers `recorder.interrupt()` only
    when a response is in progress (a genuine barge-in).
  - Server `error` events, and `response.done` with status failed /
    incomplete / unexplained cancelled, fail the sample (ProviderError) so
    an API failure is never scored as the model staying silent.
  - The observation window is a hard cutoff at input_duration (including
    trailing padding), the same for every provider.
  - Send pacing uses absolute deadlines so per-frame overhead does not
    accumulate into replay drift.
  - Audio deltas carry their `response_id`, so the recorder drops late
    deltas from a response already cancelled by barge-in.

Standalone usage (single sample, for debugging):
    python evaluation/direct_replay/inference/infer_gpt_realtime.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4

For batch runs across the dataset, use run_inference.py + model_registry.py.
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
    # Raise rather than sys.exit(1) so model_registry.available_models()'s
    # `except Exception` can mark this model unavailable.
    raise ImportError("websockets not installed. Run: pip install websockets") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import (
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, classify_terminal_response_status,
    is_expected_cancel,
)

REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2.1"

API_SR           = 24_000   # OpenAI Realtime: PCM16 at 24kHz, both directions
BYTES_PER_SAMPLE = 2

CHUNK_MS      = 20   # send audio in 20ms frames
CHUNK_SAMPLES = API_SR * CHUNK_MS // 1000
CHUNK_BYTES   = CHUNK_SAMPLES * BYTES_PER_SAMPLE

# 16MB covers any realistic single realtime event (audio deltas are at most
# a few hundred ms of PCM) while keeping a frame-size guard.
MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024


class ProviderError(RuntimeError):
    """Raised when the provider reports an error or a failed response.
    RealtimeClientBase.run_sample() records it in the sample's `errors` list,
    so retry/skip logic and scoring never mistake an API-side failure for
    the model staying silent."""

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class GPTRealtimeClient(RealtimeClientBase):
    MODEL_SLUG = "gpt-realtime-2.1"
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder,
                                   event_logger: EventLogger, start_streaming) -> None:
        url = f"{REALTIME_WS_URL}?model={self.model_name}"
        from openai_keys import bearer_key
        headers = {"Authorization": f"Bearer {bearer_key(self.api_key)}"}

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
                    "type": "realtime",
                    "model": self.model_name,
                    "instructions": self.system_prompt,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": API_SR},
                            "turn_detection": {"type": "semantic_vad"},
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": API_SR},
                            "voice": "alloy",
                        },
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
                        # Pass the delta's response_id so the recorder can
                        # drop stragglers from an already-cancelled response.
                        await recorder.add(pcm_bytes, response_id=event.get("response_id"))
                        event_logger.log("recv", etype, n_bytes=len(pcm_bytes))

                    elif etype in ("response.created",):
                        response_active = True
                        current_response_id = event.get("response", {}).get("id")
                        event_logger.log("recv", etype, {"response_id": current_response_id})

                    elif etype == "input_audio_buffer.speech_started":
                        if response_active:
                            # Genuine barge-in: new input speech arrived while the
                            # model's response was still in progress. The server
                            # will auto-cancel the response (interrupt_response
                            # default for semantic_vad) -- reflect that now.
                            recorder.interrupt(reason="input_speech_during_response",
                                                response_id=current_response_id)
                            response_active = False
                            expecting_cancel = True   # the next response.done
                            # for this response should be "cancelled"
                            event_logger.log("recv", etype, {"barge_in": True})
                        else:
                            event_logger.log("recv", etype, {"barge_in": False})

                    elif etype == "response.done":
                        response = event.get("response", {})
                        status = response.get("status", "unknown")
                        response_active = False
                        # "failed" / "incomplete" / an unexplained "cancelled"
                        # fail the sample via provider_errors.
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
                        code = err.get("code")
                        event_logger.log("recv", "error", {"message": msg, "code": code})
                        if code == "input_audio_buffer_commit_empty":
                            # Benign under semantic_vad: the server auto-commits
                            # at end-of-turn, so the trailing commit below
                            # often lands on an already-flushed buffer.
                            if self.verbose:
                                print(f"  [INFO] {msg} (benign: server already auto-committed)",
                                      file=sys.stderr)
                        else:
                            # Record rather than raise here: an exception in
                            # recv_task would be lost when it is cancelled.
                            # The sender raises on `provider_errors` after
                            # streaming ends.
                            provider_errors.append({"message": msg, "code": code})
                            if self.verbose:
                                print(f"  [ERROR] {msg}", file=sys.stderr)

                    elif etype.endswith(".delta"):
                        pass  # ignore other delta streams (e.g. transcript deltas)

                    else:
                        event_logger.log("recv", etype)

            recv_task = asyncio.create_task(receive_loop())

            # ── Stream input audio in real time ────────────────────────────
            # No manual commit mid-stream: semantic_vad segments speech and
            # creates responses from the continuously appended audio.
            #
            # t=0 for the recorder/event-log timeline is the first audio
            # chunk, not connect time, so it aligns with input.wav's t=0 and
            # the sample's timestamps (pause_start/bc_start/...).
            loop = asyncio.get_event_loop()
            # send_t0 is stream_start["t"] itself, so the recorder clock, the
            # event log and this send schedule share one origin.
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
                # Absolute deadline so per-frame overhead can't accumulate
                # into replay drift.
                await sleep_until_deadline(loop, send_t0 + (i + 1) * (CHUNK_MS / 1000))

            # Flush any trailing partial buffer. No response.create: the
            # server decides whether/when to respond. This often lands on an
            # empty buffer (input_audio_buffer_commit_empty, handled above).
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            event_logger.log("send", "input_audio_buffer.commit")

            # Hard cutoff at input end (including trailing padding): the
            # observation window is over, so stop receiving immediately.
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        if provider_errors:
            # Raised after the websocket block exits so the connection is
            # always closed cleanly.
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

    client = GPTRealtimeClient(
        api_key=args.api_key,
        model_name=args.model,
        system_prompt=args.system_prompt,
        verbose=args.verbose,
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
    parser.add_argument("--input_duration", type=float, default=None,
                         help="Defaults to the actual duration of input_wav")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not args.api_key:
        print("Error: no OPENAI_API_KEY found (env / .env / --api_key).", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
