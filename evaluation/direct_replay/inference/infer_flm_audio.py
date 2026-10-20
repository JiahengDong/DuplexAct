#!/usr/bin/env python3
"""
CofeAI/FLM-Audio client (local network server). FLM-Audio's reference server
(`flmaudio/server.py`) is derived from Kyutai's Moshi server and speaks the
same WebSocket protocol, so this client mirrors `infer_moshi.py`.

Protocol:
  - On connect, the server sends a single b"\\x00" handshake byte.
  - Both directions send audio as b"\\x01" + <opus bytes>, one packet per
    80ms (1920-sample @ 24kHz) Mimi frame, Opus-encoded with `sphn`.
  - The server sends b"\\x02" + <utf8 text> for every non-padding text token
    (a word/punctuation piece); these drive speech-segment detection.

FLM-Audio has no system-prompt mechanism; `self.system_prompt` is accepted
for interface uniformity but unused.

FLM-Audio pins `sphn==0.1.4`, whose Opus writer/reader are buffered:
`append_pcm()`/`append_bytes()` only feed data and return None, and output
must be polled with `read_bytes()`/`read_pcm()` (as server.py's own
`opus_loop()` does). The send/receive drain loops below handle this.

The same HF repo (`CofeAI/FLM-Audio`, which bundles the Mimi checkpoint) is
passed as both `--mimi-path` and `--model-path`, matching server.py's default.

Standalone usage (single sample, for debugging):
    python infer_flm_audio.py --input_wav path/to/input.wav \\
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
    # Raise rather than sys.exit(1) so model_registry can probe availability.
    raise ImportError("sphn not installed. Run: pip install sphn") from e

try:
    import websockets
except ImportError as e:
    raise ImportError("websockets not installed. Run: pip install websockets") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from local_common import LocalServerManager, env_for_model, opensource_root_for
from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger

FLM_AUDIO_SLUG = "flm-audio"
FLM_AUDIO_HOST = "localhost"
FLM_AUDIO_PORT = int(os.environ.get("FLM_AUDIO_PORT", "8990"))   # matches server.py's own default
DEFAULT_MODEL = "CofeAI/FLM-Audio"

API_SR         = 24_000
FRAME_SAMPLES  = 1_920                 # 80ms @ 24kHz -- same Mimi frame size as Moshi
FRAME_SEC      = FRAME_SAMPLES / API_SR

# Max gap between text tokens still treated as one utterance; same as
# MOSHI_SILENCE_GRACE_S since the text channel has the same semantics.
FLM_AUDIO_SILENCE_GRACE_S = 0.8

REPO_DIR = opensource_root_for(FLM_AUDIO_SLUG) / "repo"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _pcm_f32_to_i16_bytes(pcm_f32: np.ndarray) -> bytes:
    return (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class FLMAudioClient(RealtimeClientBase):
    MODEL_SLUG = FLM_AUDIO_SLUG
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        await LocalServerManager.ensure(
            key=f"flm-audio:{self.model_name}",
            cmd=[
                sys.executable, "-m", "flmaudio.server",
                "--host", FLM_AUDIO_HOST, "--port", str(FLM_AUDIO_PORT),
                "--mimi-path", self.model_name,
                "--model-path", self.model_name,
            ],
            cwd=REPO_DIR,   # flmaudio is a plain relative-import package, not pip-installed
            env=env_for_model(FLM_AUDIO_SLUG),
            host=FLM_AUDIO_HOST, port=FLM_AUDIO_PORT,
            # Per-job log file, so concurrent servers on different ports do
            # not interleave their output.
            log_path=LOG_DIR / f"flm_audio_server.{os.environ.get('SLURM_JOB_ID', 'local')}.log",
            startup_timeout_s=900.0,
        )

        url = f"ws://{FLM_AUDIO_HOST}:{FLM_AUDIO_PORT}/api/chat"
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
                        # append_bytes() only feeds the decoder; decoded pcm
                        # is drained by recv_drain_loop().
                        opus_reader.append_bytes(payload)
                        if speaking and (now - last_word_t) > FLM_AUDIO_SILENCE_GRACE_S:
                            recorder.set_speaking(False, at_s=last_word_t + FLM_AUDIO_SILENCE_GRACE_S)
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

            async def recv_drain_loop():
                # Mirrors server.py's opus_loop(): read_pcm() returns None or
                # an empty array until enough input is buffered.
                while True:
                    await asyncio.sleep(0.001)
                    pcm_f32 = opus_reader.read_pcm()
                    if pcm_f32 is None or pcm_f32.shape[-1] == 0:
                        continue
                    await recorder.add(_pcm_f32_to_i16_bytes(pcm_f32))
                    event_logger.log("recv", "audio", {"n_samples": int(pcm_f32.shape[-1])})

            async def send_drain_loop():
                # append_pcm() only feeds the encoder; read_bytes() drains
                # whatever opus it has produced so far.
                while True:
                    await asyncio.sleep(0.001)
                    opus_bytes = opus_writer.read_bytes()
                    if opus_bytes:
                        await ws.send(b"\x01" + opus_bytes)
                        event_logger.log("send", "audio", n_bytes=len(opus_bytes))

            recv_task = asyncio.create_task(receive_loop())
            recv_drain_task = asyncio.create_task(recv_drain_loop())
            send_drain_task = asyncio.create_task(send_drain_loop())

            # Switch the recorder to manual segmentation before any audio
            # arrives, so priming/padding audio before the first text token
            # is never registered as speech.
            recorder.set_speaking(False, at_s=0.0)

            await start_streaming()
            stream_t0["t"] = time.monotonic()

            pcm_f32_all = input_pcm.astype(np.float32) / 32768.0
            pad = (-len(pcm_f32_all)) % FRAME_SAMPLES
            if pad:
                pcm_f32_all = np.concatenate([pcm_f32_all, np.zeros(pad, dtype=np.float32)])

            for off in range(0, len(pcm_f32_all), FRAME_SAMPLES):
                frame = pcm_f32_all[off: off + FRAME_SAMPLES]
                # Transmitted by send_drain_loop().
                opus_writer.append_pcm(frame)
                await asyncio.sleep(FRAME_SEC)

            for task in (recv_task, recv_drain_task, send_drain_task):
                task.cancel()
            for task in (recv_task, recv_drain_task, send_drain_task):
                try:
                    await task
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

    client = FLMAudioClient(
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
                         help="HF repo id, e.g. CofeAI/FLM-Audio")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
