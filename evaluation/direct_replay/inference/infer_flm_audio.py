#!/usr/bin/env python3
"""
direct_replay/inference/infer_flm_audio.py
=============================================
CofeAI/flm-audio (FLM-Audio) client. Category A (local network server) --
the SAME wire protocol as Moshi, because FLM-Audio's own reference server
(`flmaudio/server.py`) is a close derivative of Kyutai's own Moshi server:
it bundles a vendored copy of Moshi's Mimi codec + streaming modules
verbatim under `flmaudio/third_party/moshi/` and reuses the exact same
WebSocket wire format. Read `infer_moshi.py` and `_shared_framework.md`
first; this file is deliberately a close structural twin of
`infer_moshi.py`, not a from-scratch design.

Architecture (arXiv:2509.02521): a 7B-parameter LLM backbone (initialized
from a pretrained text LM, not trained from scratch) + Mimi codec (24kHz,
12.5Hz frame rate, 8 audio codebooks/frame -- identical Mimi config to
Moshi's) + a small "depth transformer" that autoregressively produces the
8 audio tokens for each frame from the backbone's last hidden state (this
project's own equivalent of Moshi's RQ-Transformer depth decoder). Unlike
Moshi's separate listen/speak/pad text-stream design, FLM-Audio's paper
frames its main contribution as "natural monologues" -- a revised stream
organization that interleaves listening, speaking, and an internal
monologue text channel more contiguously than Moshi's original design --
but at the wire-protocol level (what our client actually has to speak) the
two servers are indistinguishable: same handshake byte, same `\\x01`+opus
audio framing in both directions, same `\\x02`+utf8-text framing for
non-padding text tokens.

Protocol (see cofe-ai/flm-audio's flmaudio/server.py + client.py, and
compare with kyutai-labs/moshi's own server.py/client.py):
  - On connect, the server sends a single b"\\x00" handshake byte.
  - Both directions send audio as b"\\x01" + <opus bytes>, one packet per
    80ms (1920-sample @ 24kHz) frame -- Opus-encoded via the same `sphn`
    package Moshi uses (FLM-Audio's own requirements-server.txt pins the
    identical `sphn==0.1.4`).
  - The server sends b"\\x02" + <utf8 text> whenever `decode_step()`
    decodes a text token that is (a) not one of the two special IDs it
    always skips (`0`, `3` -- BOS/EOS-equivalent bookkeeping tokens) and
    (b) not equal to `config.mm_token_info.text_wait_token_id` (FLM-Audio's
    own name for what Moshi calls the padding/silence text token). This is
    functionally identical to Moshi's own "only send \\x02 for genuine
    word/punctuation pieces" behavior, just under a different token-ID
    naming scheme in a differently-organized stream.

No system-prompt support: unlike Moshi (whose lack of a system-prompt
concept we already documented as "unused (no API concept)"), FLM-Audio's
`LMGen`/server.py expose no text-preamble or instruction-seeding mechanism
at all -- the server only ever feeds the user's own audio codes into the
cache, with no slot for an initial system message. `self.system_prompt` is
accepted (interface uniformity with every other client) but unused, same
situation as Moshi.

**`sphn` API version mismatch (adapter-side bug, not vendored code):**
FLM-Audio's own `requirements-server.txt` pins `sphn==0.1.4`, a materially
OLDER API than the `sphn==0.2.1` Moshi's venv uses -- 0.1.4's
`OpusStreamWriter`/`OpusStreamReader` are asynchronous/buffered
(`append_pcm()`/`append_bytes()` only *feed* data in and always return
`None`; you must separately poll `read_bytes()`/`read_pcm()` to retrieve
whatever has become available so far -- confirmed both by each method's
own docstring and by direct testing in this model's venv), whereas 0.2.1
collapses this into a single synchronous call that both feeds data in
*and* returns the result (confirmed the same way in Moshi's venv: 0.2.1's
`OpusStreamWriter`/`Reader` don't even expose `read_bytes`/`read_pcm` at
all). `infer_moshi.py`'s send/receive code -- written for 0.2.1 -- was
carried over near-verbatim when this file was built as Moshi's "close
structural twin," which silently broke against 0.1.4: `append_pcm()`'s
return was always `None`, so the `if opus_bytes:` guard on the send side
was always false and **no audio was ever actually sent to the server**;
symmetrically, `append_bytes()`'s return on the receive side was always
`None`, so `pcm_f32.size` raised `AttributeError` the moment any audio
message actually arrived. Fixed by adding the polling loops
FLM-Audio's *own* `server.py` already uses internally for this exact
purpose (`opus_loop()`/`audio_loop()`) -- see `_connect_and_stream()`
below.

No separate mimi/model repos: unlike Moshi (where `--hf-repo` names one
combined repo containing both), FLM-Audio's server.py already defaults
BOTH `--mimi-path` and `--model-path` to the same single `CofeAI/FLM-Audio`
repo (which bundles the Mimi checkpoint file
`tokenizer-e351c8d8-checkpoint125.safetensors` alongside the LLM's own
safetensors shards) -- we pass the same repo id for both, matching the
server's own default exactly.

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
    # Raise rather than sys.exit(1) (code review F-20, applied here too --
    # confirmed in practice to crash model_registry.available_models()).
    raise ImportError("sphn not installed. Run: pip install sphn") from e

try:
    import websockets
except ImportError as e:
    raise ImportError("websockets not installed. Run: pip install websockets") from e

from local_common import LocalServerManager, env_for_model, opensource_root_for
from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger

FLM_AUDIO_SLUG = "flm-audio"
FLM_AUDIO_HOST = "localhost"
FLM_AUDIO_PORT = int(os.environ.get("FLM_AUDIO_PORT", "8990"))   # matches server.py's own default
DEFAULT_MODEL = "CofeAI/FLM-Audio"

API_SR         = 24_000
FRAME_SAMPLES  = 1_920                 # 80ms @ 24kHz -- same Mimi frame size as Moshi
FRAME_SEC      = FRAME_SAMPLES / API_SR

# Same rationale and starting point as Moshi's own MOSHI_SILENCE_GRACE_S
# (see infer_moshi.py) -- FLM-Audio's text channel has the identical
# "one non-padding token per spoken word/punctuation piece" semantics, so
# the same intra-utterance-pause tolerance applies. Re-tune from real
# smoke-test output if FLM-Audio's own pacing turns out to differ from
# Moshi's.
FLM_AUDIO_SILENCE_GRACE_S = 0.8

REPO_DIR = opensource_root_for(FLM_AUDIO_SLUG) / "repo"
LOG_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs")


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
            # Per-job log file (job ID suffix) -- with per-node port isolation
            # (FLM_AUDIO_PORT, see slurm_infer_opensource.sh) two same-model
            # jobs can legitimately run concurrently on one node; a shared
            # log path would otherwise interleave their output.
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
                        # sphn==0.1.4: append_bytes() only feeds the encoder
                        # buffer, it does NOT return decoded pcm (see module
                        # docstring) -- decoded output is drained separately
                        # by recv_drain_loop() below, polling read_pcm().
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
                # Mirrors flmaudio/server.py's own opus_loop(): read_pcm()
                # returns None (no data yet) or an empty array (also no data
                # yet, per server.py's own defensive double-check) until the
                # decoder has enough buffered input, then returns whatever
                # pcm has accumulated so far -- decoupled in time from
                # whichever receive_loop() call fed in the bytes it decoded.
                while True:
                    await asyncio.sleep(0.001)
                    pcm_f32 = opus_reader.read_pcm()
                    if pcm_f32 is None or pcm_f32.shape[-1] == 0:
                        continue
                    await recorder.add(_pcm_f32_to_i16_bytes(pcm_f32))
                    event_logger.log("recv", "audio", {"n_samples": int(pcm_f32.shape[-1])})

            async def send_drain_loop():
                # Symmetric counterpart on the write side: append_pcm() below
                # also only feeds the encoder, it does NOT return the
                # encoded opus bytes (see module docstring) -- read_bytes()
                # drains whatever the encoder has produced so far.
                while True:
                    await asyncio.sleep(0.001)
                    opus_bytes = opus_writer.read_bytes()
                    if opus_bytes:
                        await ws.send(b"\x01" + opus_bytes)
                        event_logger.log("send", "audio", n_bytes=len(opus_bytes))

            recv_task = asyncio.create_task(receive_loop())
            recv_drain_task = asyncio.create_task(recv_drain_loop())
            send_drain_task = asyncio.create_task(send_drain_loop())

            # Same reasoning as infer_moshi.py: force manual segmentation
            # mode before any audio can reach recorder.add(), so the
            # recorder's automatic (audio-presence) tracking never briefly
            # mis-registers priming/padding audio as a false speech segment
            # during the window before FLM-Audio's first real word.
            recorder.set_speaking(False, at_s=0.0)

            await start_streaming()
            stream_t0["t"] = time.monotonic()

            pcm_f32_all = input_pcm.astype(np.float32) / 32768.0
            pad = (-len(pcm_f32_all)) % FRAME_SAMPLES
            if pad:
                pcm_f32_all = np.concatenate([pcm_f32_all, np.zeros(pad, dtype=np.float32)])

            for off in range(0, len(pcm_f32_all), FRAME_SAMPLES):
                frame = pcm_f32_all[off: off + FRAME_SAMPLES]
                # sphn==0.1.4: append_pcm() only feeds the encoder buffer
                # (always returns None) -- send_drain_loop() above drains
                # and actually transmits whatever it has produced so far.
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
