#!/usr/bin/env python3
"""
direct_replay/inference/infer_gpt_realtime.py
================================================
OpenAI GPT-Realtime client (WebSocket), built on the shared
`RealtimeClientBase` / `SynchronizedRecorder` infrastructure in
realtime_common.py.

Correctness fixes vs. the original implementation:
  - No more explicit `response.create`. Turn detection (semantic_vad) governs
    response creation autonomously, exactly like a real full-duplex system --
    critical for testing wait/ignore/backchannel, where a forced response
    would defeat the point of the test.
  - Continuous, full-duration capture: the receive loop runs for the entire
    input_duration (not just until the first response.done), so multi-turn
    behavior (e.g. repeated backchannels, floor-holding across several turns)
    is captured, not just a single response.
  - Genuine barge-in detection: `input_audio_buffer.speech_started` only
    triggers `recorder.interrupt()` when a response is actually in progress
    at that moment (response_active). Speech-started events that occur while
    no response is active are just normal turn-taking on the input stream and
    are logged but not treated as interruptions.
  - Server `error` events now fail the sample (raise ProviderError) instead
    of only being logged -- previously an API-side error could leave a
    sample looking like a clean "model stayed silent" result, contaminating
    wait/ignore scoring (code review F-04).
  - Post-input cutoff (design decision, supersedes an earlier drain_until_quiet
    heuristic): the receive task is cancelled immediately once every input
    sample -- including the trailing padding -- has been sent. The
    observation window is a hard, fixed cutoff at input_duration, applied
    uniformly across every provider and both model classes; we do not wait
    any further for new model audio once input is exhausted (F-01 revisited).
  - Send pacing uses absolute deadlines (`sleep_until_deadline`) instead of
    a fixed post-send sleep, so per-frame overhead can't accumulate into
    replay drift over a long sample (F-10).
  - Terminal response status is now classified (code review R-06):
    `response.done`'s `status` used to only flip `response_active` back to
    False, so a `failed` or `incomplete` terminal response was silently
    accepted as if it were an ordinary completed turn. It's now run through
    `classify_terminal_response_status()` -- "expected_cancelled" (a
    barge-in we already recorded via `recorder.interrupt()`) is still
    normal, but "failed", "incomplete", and an unexplained "cancelled" are
    appended to `provider_errors` and fail the sample, same as a server
    `error` event.
  - Late deltas can no longer resume a cancelled response (code review
    R-03): every audio-delta event carries the OpenAI-documented
    `response_id` field, which is now passed to `recorder.add()`, and
    `recorder.interrupt()` is now given the response id being cancelled.
    `SynchronizedRecorder` rejects any later delta naming an id it already
    recorded as cancelled instead of relying solely on `muted`, which used
    to unconditionally clear the instant ANY packet arrived -- including a
    straggler that was already in flight before the cancellation took
    effect.

Standalone usage (single sample, for debugging):
    python infer_gpt_realtime.py --input_wav path/to/input.wav \\
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
    # Raise rather than sys.exit(1) (code review F-20): SystemExit derives
    # from BaseException, not Exception, so model_registry.available_models()'s
    # `except Exception` couldn't catch it and this whole process would die
    # instead of just marking this one model unavailable.
    raise ImportError("websockets not installed. Run: pip install websockets") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import (
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, classify_terminal_response_status,
    is_expected_cancel,
)

REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
# gpt-realtime-2.1 is the current GA model (replaces gpt-4o-realtime-preview).
DEFAULT_MODEL = "gpt-realtime-2.1"

API_SR           = 24_000   # OpenAI Realtime: PCM16 at 24kHz, both directions
BYTES_PER_SAMPLE = 2

CHUNK_MS      = 20   # send audio in 20ms frames
CHUNK_SAMPLES = API_SR * CHUNK_MS // 1000
CHUNK_BYTES   = CHUNK_SAMPLES * BYTES_PER_SAMPLE

# Trusted provider, but disabling the websockets library's own frame-size
# guard entirely removes protection against a malformed/unexpectedly large
# message (code review F-22). 16MB comfortably covers any realistic single
# realtime event (audio deltas are at most a few hundred ms of PCM).
MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024


class ProviderError(RuntimeError):
    """Raised when the provider sends an explicit `error` event over the
    realtime WebSocket. Propagating this (instead of only logging it, see
    code review F-04) ensures RealtimeClientBase.run_sample() records it in
    the sample's `errors` list, so run_inference.py's retry/skip logic and
    downstream B_i/R_i scoring can no longer mistake an API-side failure for
    the model genuinely staying silent."""

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
                        # R-03: pass the delta's own response_id so the
                        # recorder can reject it if it belongs to a response
                        # we've already recorded as cancelled (a straggler
                        # already in flight before our interrupt() call
                        # took effect) rather than resuming playback.
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
                            expecting_cancel = True   # R-06: the response.done we
                            # see next for this response should be "cancelled"
                            event_logger.log("recv", etype, {"barge_in": True})
                        else:
                            event_logger.log("recv", etype, {"barge_in": False})

                    elif etype == "response.done":
                        response = event.get("response", {})
                        status = response.get("status", "unknown")
                        response_active = False
                        # R-06: classify the terminal status instead of just
                        # logging it -- "failed"/"incomplete"/an unexplained
                        # "cancelled" now fail the sample (provider_errors)
                        # instead of being silently accepted as valid output.
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
                            # Benign, expected under semantic_vad (see the
                            # trailing input_audio_buffer.commit below):
                            # semantic_vad auto-commits mid-stream as soon as
                            # it detects end-of-turn, so our own defensive
                            # end-of-file commit frequently lands on a buffer
                            # the server already flushed itself -- there is
                            # nothing left to commit, which is fine, not a
                            # failure. (Discovered via F-04's fix: this error
                            # was previously always silently swallowed, which
                            # is also why it went unnoticed until now.)
                            if self.verbose:
                                print(f"  [INFO] {msg} (benign: server already auto-committed)",
                                      file=sys.stderr)
                        else:
                            # Record but don't raise from inside the receive
                            # loop itself -- an exception raised here would
                            # just be a "Task exception was never retrieved"
                            # warning once recv_task is cancelled (F-04); the
                            # sender side checks `provider_errors` after
                            # streaming ends and raises there instead, so
                            # it's always observed.
                            provider_errors.append({"message": msg, "code": code})
                            if self.verbose:
                                print(f"  [ERROR] {msg}", file=sys.stderr)

                    elif etype.endswith(".delta"):
                        pass  # ignore other delta streams (e.g. transcript deltas)

                    else:
                        event_logger.log("recv", etype)

            recv_task = asyncio.create_task(receive_loop())

            # ── Stream input audio in real time ────────────────────────────
            # No manual commit mid-stream: turn_detection (semantic_vad) governs
            # speech segmentation and response creation autonomously from the
            # continuously appended raw audio, exactly as a live full-duplex
            # system would behave.
            #
            # t=0 for the recorder/event-log timeline starts HERE, at the first
            # audio chunk -- not at connect time -- so it aligns with
            # input.wav's own t=0 (and therefore with every sample's embedded
            # timestamps like pause_start/bc_start/side_speech_start_s).
            loop = asyncio.get_event_loop()
            # send_t0 IS stream_start["t"] itself (see start_streaming()'s
            # docstring) -- not a second, independently-measured clock read
            # -- so the recorder's expected_count() clock, the event log's
            # t=0, and this send schedule's origin are all the literal same
            # monotonic value.
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
                # Absolute deadline instead of a fixed post-send sleep, so
                # per-frame overhead (JSON/base64 encoding, network I/O,
                # event-loop scheduling) can't accumulate into replay drift
                # over the length of the sample (F-10).
                await sleep_until_deadline(loop, send_t0 + (i + 1) * (CHUNK_MS / 1000))

            # Flush any trailing partial buffer. No response.create: we let
            # the server decide autonomously whether/when to respond. Under
            # semantic_vad the server frequently auto-commits mid-stream
            # already, so this often lands on an empty buffer -- see the
            # input_audio_buffer_commit_empty handling above, which treats
            # that specific case as benign rather than a failure.
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            event_logger.log("send", "input_audio_buffer.commit")

            # Hard cutoff at input end: input_duration (including the
            # trailing padding) has now been fully sent, so the observation
            # window is over -- cancel reception immediately rather than
            # waiting any further for new model audio (see realtime_common.py
            # module docstring for the design rationale).
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        if provider_errors:
            # Raised AFTER the `async with websockets.connect(...)` block
            # closes cleanly, so the connection is always torn down properly
            # regardless of whether an error occurred (F-04).
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
