#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/infer_stop_flm_audio.py
===========================================================
CofeAI/FLM-Audio client for the `stop` Level-2 action, simulator-driven
(live LLM-triggered injection -- see trigger_controller.py's module
docstring for the full architecture rationale).

FLM-Audio's own direct_replay client (infer_flm_audio.py) documents that
its wire protocol is a byte-for-byte twin of Moshi's own (same handshake,
same kind==1 opus-audio / kind==2 non-padding-text-token framing, same
80ms/1920-sample Mimi frame -- FLM-Audio's server vendors Moshi's own Mimi
codec + streaming modules verbatim). So this file is, in turn, a close
structural twin of infer_stop_moshi.py -- same DynamicPCMSource sender
loop, same LiveTriggerController wiring, same text-token capture -- with
only the server-launch command and the two known sphn==0.1.4 API
differences (see infer_flm_audio.py's own module docstring: `append_pcm`/
`append_bytes` are async/buffered in this older sphn version, requiring the
separate `read_bytes()`/`read_pcm()` polling loops FLM-Audio's own
server.py already uses internally) carried over unchanged from
infer_flm_audio.py.

This file is deliberately a SEPARATE module from infer_flm_audio.py (not a
subclass override) so the already-validated direct_replay pipeline is
never touched -- see MODEL_COMPATIBILITY.md's "DIRECT PORT" entry for
flm-audio.

Standalone usage (single sample, for smoke testing):
    python infer_stop_flm_audio.py --sample_dir <path to Candor_stop_0038 etc> \\
        --output_wav out.wav --events_path events.jsonl
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import sphn
except ImportError as e:
    raise ImportError("sphn not installed. Run: pip install sphn") from e

try:
    import websockets
except ImportError as e:
    raise ImportError("websockets not installed. Run: pip install websockets") from e

from local_common import LocalServerManager, env_for_model, opensource_root_for  # noqa: E402
from realtime_common import (  # noqa: E402
    RealtimeClientBase,
    SynchronizedRecorder,
    EventLogger,
    DynamicPCMSource,
)
from trigger_controller import LiveTriggerController, poll_interval_for_native_chunk  # noqa: E402
from stop_common import stop_cli_main, POST_CUE_OBSERVATION_S  # noqa: E402

FLM_AUDIO_SLUG = "flm-audio"
FLM_AUDIO_HOST = "localhost"
FLM_AUDIO_PORT = int(os.environ.get("FLM_AUDIO_PORT", "8990"))   # matches infer_flm_audio.py's own default
DEFAULT_MODEL = "CofeAI/FLM-Audio"

API_SR        = 24_000
FRAME_SAMPLES = 1_920          # 80ms @ 24kHz -- same Mimi frame size as Moshi
FRAME_SEC     = FRAME_SAMPLES / API_SR
FRAME_BYTES   = FRAME_SAMPLES * 2   # int16

# Same derivation as MOSHI_POLL_INTERVAL_S in infer_stop_moshi.py -- FLM-
# Audio's native audio quantum is the identical 80ms Mimi frame, so the
# tau2-bench 10-tick ratio (see trigger_controller.py) lands on the same
# 0.8s value.
FLM_AUDIO_POLL_INTERVAL_S = poll_interval_for_native_chunk(FRAME_SEC)  # 1.0s unified user-side poll

# Same rationale and starting point as infer_flm_audio.py's own
# FLM_AUDIO_SILENCE_GRACE_S -- FLM-Audio's text channel has the identical
# "one non-padding token per spoken word/punctuation piece" semantics as
# Moshi's.
FLM_AUDIO_SILENCE_GRACE_S = 0.8

REPO_DIR = opensource_root_for(FLM_AUDIO_SLUG) / "repo"   # same as infer_flm_audio.py's own REPO_DIR
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _pcm_f32_to_i16_bytes(pcm_f32: np.ndarray) -> bytes:
    return (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class FLMAudioStopClient(RealtimeClientBase):
    MODEL_SLUG = FLM_AUDIO_SLUG
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, cue_pcm: bytes, reference_context: dict, llm_client):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.cue_pcm = cue_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result: dict = {}
        self.text_tokens: list[tuple[float, str]] = []
        self.simulator_speech_segments: list[dict] = []

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        # `input_pcm`/`input_duration` here are lead_in.wav's, per run_sample()'s
        # contract (same as infer_stop_moshi.py).
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
            log_path=LOG_DIR / f"flm_audio_server.{os.environ.get('SLURM_JOB_ID', 'local')}.log",
            startup_timeout_s=900.0,
        )

        url = f"ws://{FLM_AUDIO_HOST}:{FLM_AUDIO_PORT}/api/chat"
        opus_writer = sphn.OpusStreamWriter(API_SR)
        opus_reader = sphn.OpusStreamReader(API_SR)

        speaking = False
        last_word_t = -1.0
        stream_t0: dict = {"t": None}
        controller_holder: dict = {"controller": None}

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
                        # buffer, it does NOT return decoded pcm -- decoded
                        # output is drained separately by recv_drain_loop()
                        # below, polling read_pcm() (see infer_flm_audio.py's
                        # module docstring for the full sphn version-skew
                        # explanation).
                        opus_reader.append_bytes(payload)
                        if speaking and (now - last_word_t) > FLM_AUDIO_SILENCE_GRACE_S:
                            recorder.set_speaking(False, at_s=last_word_t + FLM_AUDIO_SILENCE_GRACE_S)
                            speaking = False
                            event_logger.log("recv", "speech_segment_end", {"t": now})

                    elif kind == 2:  # non-padding text token
                        text = payload.decode("utf8", errors="ignore")
                        if not speaking:
                            recorder.set_speaking(True, at_s=now)
                            speaking = True
                            event_logger.log("recv", "speech_segment_start", {"t": now})
                        last_word_t = now
                        self.text_tokens.append((now, text))
                        ctrl = controller_holder["controller"]
                        if ctrl is not None:
                            ctrl.on_text(text, at_s=now)
                        event_logger.log("recv", "text", {"text": text})

                    else:
                        event_logger.log("recv", "unknown_kind", {"kind": kind})

            async def recv_drain_loop():
                # Mirrors flmaudio/server.py's own opus_loop() -- see
                # infer_flm_audio.py's module docstring.
                while True:
                    await asyncio.sleep(0.001)
                    pcm_f32 = opus_reader.read_pcm()
                    if pcm_f32 is None or pcm_f32.shape[-1] == 0:
                        continue
                    await recorder.add(_pcm_f32_to_i16_bytes(pcm_f32))
                    event_logger.log("recv", "audio", {"n_samples": int(pcm_f32.shape[-1])})

            async def send_drain_loop():
                # Symmetric write-side counterpart -- append_pcm() below
                # only feeds the encoder (see infer_flm_audio.py's module
                # docstring); read_bytes() drains whatever it has produced.
                while True:
                    await asyncio.sleep(0.001)
                    opus_bytes = opus_writer.read_bytes()
                    if opus_bytes:
                        await ws.send(b"\x01" + opus_bytes)
                        event_logger.log("send", "audio", n_bytes=len(opus_bytes))

            recv_task = asyncio.create_task(receive_loop())
            recv_drain_task = asyncio.create_task(recv_drain_loop())
            send_drain_task = asyncio.create_task(send_drain_loop())

            recorder.set_speaking(False, at_s=0.0)

            await start_streaming()
            stream_t0["t"] = time.monotonic()

            def now_fn() -> float:
                return time.monotonic() - stream_t0["t"]

            source = DynamicPCMSource(initial_pcm=input_pcm.astype(np.int16).tobytes())

            controller = LiveTriggerController(
                recorder=recorder,
                source=source,
                cue_pcm=self.cue_pcm,
                now_fn=now_fn,
                reference_context=self.reference_context,
                client=self.llm_client,
                lead_in_duration_s=input_duration,
                event_logger=event_logger,
                poll_interval_s=FLM_AUDIO_POLL_INTERVAL_S,
            )
            controller_holder["controller"] = controller
            trigger_task = asyncio.create_task(controller.run())

            def _on_trigger_done(t: asyncio.Task) -> None:
                source.finish()

            trigger_task.add_done_callback(_on_trigger_done)

            cue_drained_at: float | None = None
            try:
                while True:
                    chunk_i16 = source.pull(FRAME_BYTES)
                    frame_f32 = np.frombuffer(chunk_i16, dtype=np.int16).astype(np.float32) / 32768.0
                    # sphn==0.1.4: append_pcm() only feeds the encoder buffer
                    # (always returns None) -- send_drain_loop() above drains
                    # and actually transmits whatever it has produced so far.
                    opus_writer.append_pcm(frame_f32)

                    if source.is_done and cue_drained_at is None:
                        cue_drained_at = now_fn()
                        event_logger.log("send", "cue_drained", {"t": cue_drained_at})
                    if cue_drained_at is not None and now_fn() - cue_drained_at >= POST_CUE_OBSERVATION_S:
                        break

                    await asyncio.sleep(FRAME_SEC)
            finally:
                if not trigger_task.done():
                    trigger_task.cancel()
                    try:
                        await trigger_task
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
                    try:
                        self.trigger_result = trigger_task.result()
                    except Exception as e:
                        self.trigger_result = {"error": f"{type(e).__name__}: {e}"}

                self.simulator_speech_segments = [
                    {
                        "start": 0.0,
                        "end": round(input_duration, 4),
                        "text": self.reference_context.get("transcript_user", ""),
                        "source": "lead_in",
                    }
                ]
                t_trigger = self.trigger_result.get("t_user_int_start")
                if t_trigger is not None:
                    cue_duration = len(self.cue_pcm) / 2 / API_SR
                    self.simulator_speech_segments.append({
                        "start": round(t_trigger, 4),
                        "end": round(t_trigger + cue_duration, 4),
                        "text": self.reference_context.get("reference_interruption_text", ""),
                        "source": "cue",
                    })

                for task in (recv_task, recv_drain_task, send_drain_task):
                    task.cancel()
                for task in (recv_task, recv_drain_task, send_drain_task):
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                if speaking:
                    recorder.set_speaking(False, at_s=now_fn())


def main():
    stop_cli_main(FLMAudioStopClient, FLM_AUDIO_SLUG, API_SR, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
