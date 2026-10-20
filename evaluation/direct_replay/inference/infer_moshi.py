#!/usr/bin/env python3
"""
Kyutai Moshi client. Launches a local `python -m moshi.server` once per
process (via `local_common.LocalServerManager`) and streams each sample
through its websocket API like a cloud realtime API.

Protocol (see kyutai-labs/moshi's moshi/moshi/client.py + server.py):
  - On connect, the server sends a single b"\\x00" handshake byte.
  - Both directions send audio as b"\\x01" + <opus bytes>, one packet per
    80ms (1920-sample @ 24kHz) frame -- Opus-encoded via the `sphn` package.
  - The server additionally sends b"\\x02" + <utf8 text> whenever it decodes
    a *non-padding* text token from its inner monologue, i.e. whenever it
    is actually producing a word/punctuation piece. Padding/silence frames
    send no text message at all.

Segmentation: Moshi emits audio continuously, so `model_speech_segments`
cannot come from audio presence. They are derived from the text-token
cadence instead (same 80ms granularity); MOSHI_SILENCE_GRACE_S is the
pause tolerated inside one utterance.

Standalone usage (single sample, for debugging):
    python infer_moshi.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import sphn
except ImportError as e:
    # Raise ImportError rather than sys.exit() so that
    # model_registry.available_models() can catch it and skip this model.
    raise ImportError("sphn not installed. Run: pip install sphn") from e

try:
    import websockets
except ImportError as e:
    raise ImportError("websockets not installed. Run: pip install websockets") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from local_common import LocalServerManager, env_for_model, opensource_root_for
from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger

MOSHI_SLUG = "moshi"
MOSHI_HOST = "localhost"
MOSHI_PORT = int(os.environ.get("MOSHI_PORT", "8998"))
DEFAULT_MODEL = "kyutai/moshiko-pytorch-bf16"   # moshi library's own default repo

API_SR         = 24_000
FRAME_SAMPLES  = 1_920                 # 80ms @ 24kHz -- Mimi's native frame size
FRAME_SEC      = FRAME_SAMPLES / API_SR

# Gap (seconds) in the text-token stream after which Moshi is considered to
# have stopped speaking. Clause pauses inside one utterance run ~0.5-0.6s, so
# this must exceed that to avoid splitting utterances. 10 Mimi frames.
MOSHI_SILENCE_GRACE_S = 0.8

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _pcm_f32_to_i16_bytes(pcm_f32: np.ndarray) -> bytes:
    return (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class MoshiClient(RealtimeClientBase):
    MODEL_SLUG = MOSHI_SLUG
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        await LocalServerManager.ensure(
            key=f"moshi:{self.model_name}",
            cmd=[
                sys.executable, "-m", "moshi.server",
                "--host", MOSHI_HOST, "--port", str(MOSHI_PORT),
                "--hf-repo", self.model_name,
                "--static", "none",   # skip web UI (no browser needed here)
            ],
            env=env_for_model(MOSHI_SLUG),
            host=MOSHI_HOST, port=MOSHI_PORT,
            # Per-job log file so concurrent servers on different ports
            # (MOSHI_PORT) do not interleave output.
            log_path=LOG_DIR / f"moshi_server.{os.environ.get('SLURM_JOB_ID', 'local')}.log",
            startup_timeout_s=900.0,
        )

        url = f"ws://{MOSHI_HOST}:{MOSHI_PORT}/api/chat"
        opus_writer = sphn.OpusStreamWriter(API_SR)
        opus_reader = sphn.OpusStreamReader(API_SR)

        speaking = False
        last_word_t = -1.0
        stream_t0: dict = {"t": None}

        async with websockets.connect(url, max_size=None, ping_interval=20,
                                       ping_timeout=30) as ws:
            first = await ws.recv()
            if not (isinstance(first, (bytes, bytearray)) and first[:1] == b"\x00"):
                event_logger.log("recv", "unexpected_first_message")
            else:
                event_logger.log("recv", "handshake")

            async def receive_loop():
                nonlocal speaking, last_word_t
                async for raw_msg in ws:
                    if stream_t0["t"] is None:
                        continue
                    if not isinstance(raw_msg, (bytes, bytearray)) or len(raw_msg) == 0:
                        continue
                    kind, payload = raw_msg[0], bytes(raw_msg[1:])
                    now = time.monotonic() - stream_t0["t"]

                    if kind == 1:  # audio (opus)
                        pcm_f32 = opus_reader.append_bytes(payload)
                        if pcm_f32.size:
                            await recorder.add(_pcm_f32_to_i16_bytes(pcm_f32))
                            event_logger.log("recv", "audio", n_bytes=len(payload))
                        # Audio arrives every ~80ms regardless of speech, so
                        # this doubles as the periodic silence-gap check.
                        if speaking and (now - last_word_t) > MOSHI_SILENCE_GRACE_S:
                            recorder.set_speaking(False, at_s=last_word_t + MOSHI_SILENCE_GRACE_S)
                            speaking = False
                            event_logger.log("recv", "speech_segment_end", {"t": now})

                    elif kind == 2:  # non-padding text token (a word/punctuation piece)
                        text = payload.decode("utf8", errors="ignore")
                        if not speaking:
                            recorder.set_speaking(True, at_s=now)
                            speaking = True
                            event_logger.log("recv", "speech_segment_start", {"t": now})
                        last_word_t = now
                        event_logger.log("recv", "text", {"text": text})

                    else:
                        event_logger.log("recv", "unknown_kind", {"kind": kind})

            recv_task = asyncio.create_task(receive_loop())

            # Switch to manual segmentation before any audio reaches the
            # recorder, so Moshi's priming audio is not counted as speech.
            recorder.set_speaking(False, at_s=0.0)

            await start_streaming()
            stream_t0["t"] = time.monotonic()

            pcm_f32_all = input_pcm.astype(np.float32) / 32768.0
            pad = (-len(pcm_f32_all)) % FRAME_SAMPLES
            if pad:
                pcm_f32_all = np.concatenate([pcm_f32_all, np.zeros(pad, dtype=np.float32)])

            for off in range(0, len(pcm_f32_all), FRAME_SAMPLES):
                frame = pcm_f32_all[off: off + FRAME_SAMPLES]
                opus_bytes = opus_writer.append_pcm(frame)
                if opus_bytes:
                    await ws.send(b"\x01" + opus_bytes)
                    event_logger.log("send", "audio", n_bytes=len(opus_bytes))
                await asyncio.sleep(FRAME_SEC)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

            if speaking:
                recorder.set_speaking(False, at_s=time.monotonic() - stream_t0["t"])


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = MoshiClient(
        api_key="",
        model_name=args.model,
        system_prompt="",
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
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_wav", type=Path, required=True)
    parser.add_argument("--output_wav", type=Path, required=True)
    parser.add_argument("--events_path", type=Path, required=True)
    parser.add_argument("--input_duration", type=float, default=None,
                         help="Defaults to the actual duration of input_wav")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                         help="HF repo id, e.g. kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
