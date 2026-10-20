#!/usr/bin/env python3
"""
Gemini 3.1 Flash Live (Preview) client, built on the shared
`RealtimeClientBase` / `SynchronizedRecorder` infrastructure in
realtime_common.py.

Follows Full-Duplex-Bench's Gemini Live reference implementation for
protocol comparability, with these properties:
  - genai.Client(vertexai=False, ...): the public API-key path is the one
    that supports gemini-3.1-flash-live-preview.
  - `sc.interrupted` maps directly to `recorder.interrupt()`; Gemini reports
    barge-in explicitly.
  - ONE Live session for the whole sample. `interrupted` / `turn_complete` /
    `generation_complete` are logged as turn boundaries (each turn number is
    the response identity for the late-delta guard) but do not end the
    session. Reconnecting per turn, as the FDB reference does, would discard
    conversational/VAD state and stall input pacing during the handshake.
    A genuine connection drop fails the sample; run_inference.py retries.
  - A system_instruction (the same prompt as the other clients) is passed;
    VAD config is left at the API's realtime_input_config defaults.
  - t=0 is set right after the connection opens, and send pacing uses one
    absolute deadline schedule anchored there.
  - `mime_type` declares the rate explicitly ("audio/pcm;rate=16000").
  - The observation window is a hard cutoff at input_duration (including
    trailing padding): the receiver is cancelled as soon as all input is
    sent, the same for every provider.
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
        # Public API-key path, not Vertex AI (required for this model).
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

        # One persistent Live session for the entire sample. turn_no
        # increments on every interrupted/turn_complete/generation_complete
        # boundary and is that turn's response identity for the recorder's
        # late-delta guard.
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

            # t=0 for the recorder/event-log timeline starts here, after the
            # connection is open, as in the other clients.
            loop = asyncio.get_event_loop()
            send_t0 = await start_streaming()
            idx = 0
            while idx < total:
                await sess.send_realtime_input(
                    audio={"data": chunks[idx], "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}"}
                )
                event_logger.log("send", "realtime_input.audio", n_bytes=len(chunks[idx]))
                idx += 1
                # Absolute deadline anchored once at send_t0.
                await sleep_until_deadline(loop, send_t0 + idx * chunk_duration)

            try:
                await sess.send_realtime_input(audio_stream_end=True)
                event_logger.log("send", "realtime_input.audio_stream_end")
            except Exception:
                pass

            # Hard cutoff at input end: all input (including trailing
            # padding) has been sent, so stop receiving immediately.
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
