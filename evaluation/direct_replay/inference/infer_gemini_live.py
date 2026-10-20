#!/usr/bin/env python3
"""
direct_replay/inference/infer_gemini_live.py
================================================
Gemini 3.1 Flash Live (Preview) client, built on the shared
`RealtimeClientBase` / `SynchronizedRecorder` infrastructure in
realtime_common.py.

Ported from Full-Duplex-Bench's reference implementation
(Full-Duplex-Bench/v1_v1.5/model_inference/gemini/inference_gemini31_live.py,
itself a thin layer over inference_gemini25_native.py) so our results are
directly comparable in protocol/methodology.

Key characteristics preserved from the FDB reference:
  - genai.Client(vertexai=False, ...): the public API-key path is the only
    one that supports gemini-3.1-flash-live-preview today.
  - `sc.interrupted` maps directly to `recorder.interrupt()` -- Gemini's API
    tells us explicitly when the model's turn was barged-in on, unlike
    OpenAI's Realtime API where we have to infer it from speech_started +
    response-active state.

Deliberate departure from the FDB reference (code review R-01):
  - FDB's reference script opens a NEW `client.aio.live.connect()` session
    every time a turn ends (`turn_complete`/`generation_complete`/
    `interrupted`) and reconnects to send the remaining input. A Live
    session is designed to carry a full multi-turn conversation over one
    connection -- reconnecting per turn discards whatever conversational/VAD
    state the model had, and the handshake for the new connection burns real
    wall-clock time (empirically ~140ms median, up to ~1s, measured across
    ~2000 already-collected samples that reconnect on 97.8% of them) during
    which no input is being sent. Because the absolute send-deadline schedule
    is anchored once at the start of the sample and never resets, that dead
    time makes the next several chunks arrive at the server already overdue,
    so they get sent back-to-back with no pacing to catch back up -- exactly
    the "faster than real time" failure mode we guard against everywhere
    else in this benchmark. It also makes Gemini the only one of our four
    closed-source providers that ever reconnects mid-sample; GPT/Qwen/Grok
    all stream a sample's entire input over one persistent connection and
    let the provider's own turn detection (semantic_vad / server VAD) handle
    however many turns occur within it autonomously.
  - We now do the same for Gemini: ONE `client.aio.live.connect()` session
    for the whole sample. `interrupted`/`turn_complete`/`generation_complete`
    are logged as turn boundaries (each gets its own turn counter, used as
    the response identity for the R-03 late-delta guard below) but no longer
    end the session -- the receiver keeps listening across as many turns as
    occur, exactly like the receive loop in the GPT/Qwen/Grok clients. If the
    connection itself genuinely drops (a real network error, not a turn
    boundary), the exception now propagates and fails the sample outright,
    same as GPT/Qwen/Grok -- run_inference.py's existing sample-level
    retry-with-backoff (not a second, redundant retry loop inside this
    client) is what recovers from that.

Difference from the FDB reference (deliberate, for OUR cross-model study):
  - We DO pass a system_instruction (the same DEFAULT_SYSTEM_PROMPT shared
    with the other 4 model clients), for apples-to-apples comparability
    across models in this benchmark. FDB's own 3.1 script omits it
    ("minimum-config" philosophy, measuring the model fully out-of-the-box).
    VAD config is still omitted either way -- we rely on the API's
    documented realtime_input_config defaults, exactly like FDB.

Correctness fix from realtime_direct_replay_code_review.pdf (F-03):
  - t=0 (the moment `start_streaming()` establishes the recorder/event-log
    origin) used to be set BEFORE `client.aio.live.connect()` was called, so
    the WebSocket handshake consumed positive benchmark time before any
    audio was even sent -- unlike GPT/Qwen/Grok, which all connect first and
    only then call start_streaming(). This shifted every Gemini timestamp/
    latency measurement later relative to the other three providers.
    `start_streaming()` is now called immediately after the connection is
    open (matching every other client).
  - F-10: send pacing uses one absolute deadline schedule for the whole
    sample (anchored at that same t=0), instead of a fixed per-chunk sleep
    that would drift over a long sample.

Correctness fix from realtime_direct_replay_code_review_updated.pdf (R-10):
  - `mime_type` now declares the rate explicitly ("audio/pcm;rate=16000")
    instead of the bare "audio/pcm", matching Google's current examples --
    the SDK happened to infer 16kHz correctly either way, but the wire
    declaration was ambiguous.

Post-input cutoff (design decision, supersedes an earlier drain-with-timeout
heuristic on the receiver): once every input chunk -- including the trailing
padding -- has been sent, the receiver is cancelled immediately; we no longer
wait (bounded or otherwise) for a final `turn_complete` to fire. The
observation window is a hard, fixed cutoff at input_duration, applied
uniformly across every provider and both model classes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import (
    RealtimeClientBase, SynchronizedRecorder, EventLogger, sleep_until_deadline,
)

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"

SEND_SAMPLE_RATE    = 16_000   # Gemini Live: input PCM16 at 16kHz
RECEIVE_SAMPLE_RATE = 24_000   # Gemini Live: output PCM16 at 24kHz
SEND_CHUNK_SAMPLES  = 1024     # matches FDB reference (~64ms/chunk)
SEND_CHUNK_BYTES    = SEND_CHUNK_SAMPLES * 2

THINKING_LEVEL = "minimal"   # API default for gemini-3.1-flash-live-preview

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class GeminiLiveClient(RealtimeClientBase):
    MODEL_SLUG = "gemini-3.1-flash-live"
    OUTPUT_SR  = RECEIVE_SAMPLE_RATE
    INPUT_SR   = SEND_SAMPLE_RATE

    def _make_client(self) -> genai.Client:
        # Ensure routing through the public API-key path, not Vertex AI --
        # required for gemini-3.1-flash-live-preview (see module docstring).
        return genai.Client(vertexai=False, api_key=self.api_key)

    def _build_config(self) -> dict:
        cfg: dict = {
            "response_modalities": ["AUDIO"],
            "thinking_config": {"thinking_level": THINKING_LEVEL},
        }
        if self.system_prompt:
            cfg["system_instruction"] = self.system_prompt
        return cfg

    @staticmethod
    def _split_chunks(input_pcm: np.ndarray) -> list[bytes]:
        pcm_bytes = input_pcm.tobytes()
        chunks = []
        for off in range(0, len(pcm_bytes), SEND_CHUNK_BYTES):
            c = pcm_bytes[off: off + SEND_CHUNK_BYTES]
            if len(c) < SEND_CHUNK_BYTES:
                c = c + b"\x00" * (SEND_CHUNK_BYTES - len(c))
            chunks.append(c)
        return chunks

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder,
                                   event_logger: EventLogger, start_streaming) -> None:
        client = self._make_client()
        config = self._build_config()
        chunks = self._split_chunks(input_pcm)
        total = len(chunks)
        chunk_duration = SEND_CHUNK_SAMPLES / SEND_SAMPLE_RATE

        # R-01: one persistent Live session for the entire sample -- turn
        # boundaries no longer cause a reconnect (see module docstring).
        # turn_no increments on every interrupted/turn_complete/
        # generation_complete boundary and is used as that turn's response
        # identity for the R-03 late-delta guard below.
        turn_no = 0

        async with client.aio.live.connect(model=self.model_name, config=config) as sess:
            event_logger.log("send", "session.connect")

            async def receiver():
                nonlocal turn_no
                # google-genai sess.receive() ends on turn_complete /
                # interrupted. Re-enter so a later turn is heard.
                while True:
                    async for resp in sess.receive():
                        sc = resp.server_content
                        if not sc:
                            continue

                        turn_ended = False

                        if getattr(sc, "interrupted", False):
                            recorder.interrupt(reason="gemini_interrupted",
                                                response_id=f"turn-{turn_no}")
                            event_logger.log("recv", "interrupted", {"turn": turn_no})
                            turn_ended = True

                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    await recorder.add(part.inline_data.data,
                                                        response_id=f"turn-{turn_no}")
                                    event_logger.log("recv", "model_turn.audio", n_bytes=len(part.inline_data.data))
                                if getattr(part, "generation_complete", False):
                                    turn_ended = True

                        if getattr(sc, "turn_complete", False) or getattr(sc, "generation_complete", False):
                            turn_ended = True

                        if turn_ended:
                            event_logger.log("recv", "turn_complete", {"turn": turn_no})
                            turn_no += 1

            recv_task = asyncio.create_task(receiver())

            # t=0 for the recorder/event-log timeline starts HERE, right
            # after the connection is actually open -- matching GPT/Qwen/
            # Grok (F-03).
            loop = asyncio.get_event_loop()
            send_t0 = await start_streaming()
            idx = 0
            while idx < total:
                await sess.send_realtime_input(
                    audio={"data": chunks[idx], "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}"}
                )
                event_logger.log("send", "realtime_input.audio", n_bytes=len(chunks[idx]))
                idx += 1
                # Absolute deadline anchored once at send_t0 (F-10) -- one
                # persistent session now, so there's no reconnect handshake
                # to introduce dead time or reset the schedule's baseline.
                await sleep_until_deadline(loop, send_t0 + idx * chunk_duration)

            try:
                await sess.send_realtime_input(audio_stream_end=True)
                event_logger.log("send", "realtime_input.audio_stream_end")
            except Exception:
                pass

            # Hard cutoff at input end: every input chunk, including the
            # trailing padding, has now been sent -- cancel reception
            # immediately rather than waiting any further for a final
            # turn_complete (see module docstring).
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = GeminiLiveClient(
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
    import os
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_wav", type=Path, required=True)
    parser.add_argument("--output_wav", type=Path, required=True)
    parser.add_argument("--events_path", type=Path, required=True)
    parser.add_argument("--input_duration", type=float, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    if not args.api_key:
        print("Error: no GEMINI_API_KEY found (env / .env / --api_key).", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
