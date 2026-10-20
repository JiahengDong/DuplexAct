#!/usr/bin/env python3
"""
Moshi client for the `stop` action, with the cue injected live by an LLM
trigger (see trigger_controller.py).

Uses the same Moshi server and wire protocol as infer_moshi.py (kind==1 opus
audio, kind==2 non-padding text token). Instead of streaming a fixed input,
the sender streams `lead_in.wav` through a DynamicPCMSource, a concurrent
LiveTriggerController decides when to append `cue.wav`, and streaming then
continues for a fixed post-cue observation window.

Standalone usage (single sample):
    python infer_stop_moshi.py --sample_dir <stop sample dir> \\
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

from local_common import LocalServerManager, env_for_model  # noqa: E402
from realtime_common import (  # noqa: E402
    RealtimeClientBase,
    SynchronizedRecorder,
    EventLogger,
    DynamicPCMSource,
)
from trigger_controller import LiveTriggerController, poll_interval_for_native_chunk  # noqa: E402
from stop_common import stop_cli_main  # noqa: E402

MOSHI_SLUG = "moshi"
MOSHI_HOST = "localhost"
MOSHI_PORT = int(os.environ.get("MOSHI_PORT", "8998"))
DEFAULT_MODEL = "kyutai/moshiko-pytorch-bf16"

API_SR        = 24_000
FRAME_SAMPLES = 1_920          # 80ms @ 24kHz, Mimi's native frame size
FRAME_SEC     = FRAME_SAMPLES / API_SR
FRAME_BYTES   = FRAME_SAMPLES * 2   # int16

MOSHI_POLL_INTERVAL_S = poll_interval_for_native_chunk(FRAME_SEC)

MOSHI_SILENCE_GRACE_S = 0.8    # same as infer_moshi.py

# Observation window after the cue, matching the 15s of silence Direct Replay
# appends to every input.wav, so the model has room for its recovery turn.
POST_CUE_OBSERVATION_S = 15.0

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _pcm_f32_to_i16_bytes(pcm_f32: np.ndarray) -> bytes:
    return (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class MoshiStopClient(RealtimeClientBase):
    MODEL_SLUG = MOSHI_SLUG
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, cue_pcm: bytes, reference_context: dict, llm_client):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.cue_pcm = cue_pcm                    # int16 PCM bytes @ API_SR
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result: dict = {}             # populated by _connect_and_stream
        self.text_tokens: list[tuple[float, str]] = []   # (t, word/punct) as received
        self.model_speech_transcripts: list[str] = []     # aligned to inference["model_speech_segments"]
        self.simulator_speech_segments: list[dict] = []   # what the simulator played and when

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        # `input_pcm`/`input_duration` are lead_in.wav's; callers pass
        # lead_in.wav as run_sample()'s `input_wav`.
        await LocalServerManager.ensure(
            key=f"moshi:{self.model_name}",
            cmd=[
                sys.executable, "-m", "moshi.server",
                "--host", MOSHI_HOST, "--port", str(MOSHI_PORT),
                "--hf-repo", self.model_name,
                "--static", "none",
            ],
            env=env_for_model(MOSHI_SLUG),
            host=MOSHI_HOST, port=MOSHI_PORT,
            log_path=LOG_DIR / f"moshi_server.{os.environ.get('SLURM_JOB_ID', 'local')}.log",
            startup_timeout_s=900.0,
        )

        url = f"ws://{MOSHI_HOST}:{MOSHI_PORT}/api/chat"
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
                        pcm_f32 = opus_reader.append_bytes(payload)
                        if pcm_f32.size:
                            await recorder.add(_pcm_f32_to_i16_bytes(pcm_f32))
                            event_logger.log("recv", "audio", n_bytes=len(payload))
                        if speaking and (now - last_word_t) > MOSHI_SILENCE_GRACE_S:
                            recorder.set_speaking(False, at_s=last_word_t + MOSHI_SILENCE_GRACE_S)
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

            recv_task = asyncio.create_task(receive_loop())

            # Manual segmentation from t=0, as in infer_moshi.py, to avoid a
            # spurious segment during priming audio.
            recorder.set_speaking(False, at_s=0.0)

            await start_streaming()
            stream_t0["t"] = time.monotonic()

            def now_fn() -> float:
                return time.monotonic() - stream_t0["t"]

            source = DynamicPCMSource(initial_pcm=input_pcm.tobytes())

            controller = LiveTriggerController(
                recorder=recorder,
                source=source,
                cue_pcm=self.cue_pcm,
                now_fn=now_fn,
                reference_context=self.reference_context,
                client=self.llm_client,
                lead_in_duration_s=input_duration,
                event_logger=event_logger,
                poll_interval_s=MOSHI_POLL_INTERVAL_S,
            )
            controller_holder["controller"] = controller
            trigger_task = asyncio.create_task(controller.run())

            def _on_trigger_done(t: asyncio.Task) -> None:
                # One-shot for `stop`: once the controller has triggered, no
                # more content follows the cue. pull() keeps returning
                # silence; is_done starts the POST_CUE_OBSERVATION_S countdown.
                source.finish()

            trigger_task.add_done_callback(_on_trigger_done)

            cue_drained_at: float | None = None
            try:
                while True:
                    chunk_i16 = source.pull(FRAME_BYTES)
                    frame_f32 = np.frombuffer(chunk_i16, dtype=np.int16).astype(np.float32) / 32768.0
                    opus_bytes = opus_writer.append_pcm(frame_f32)
                    if opus_bytes:
                        await ws.send(b"\x01" + opus_bytes)
                        event_logger.log("send", "audio", n_bytes=len(opus_bytes))

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

                # Simulator-side timeline, the counterpart of
                # model_speech_segments, so the full conversation can be
                # reconstructed from inference_info.json. Built here because
                # the cue's start depends on the resolved trigger_result.
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
                    cue_duration = len(self.cue_pcm) / 2 / API_SR   # int16 @ API_SR
                    self.simulator_speech_segments.append({
                        "start": round(t_trigger, 4),
                        "end": round(t_trigger + cue_duration, 4),
                        "text": self.reference_context.get("reference_interruption_text", ""),
                        "source": "cue",
                    })

                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

                if speaking:
                    recorder.set_speaking(False, at_s=now_fn())


# The standalone CLI lives in stop_common.py. Moshi's word-level
# `self.text_tokens` work with `default_model_speech_transcripts()`, so no
# `compute_model_speech_transcripts()` override is needed.

def main():
    stop_cli_main(MoshiStopClient, MOSHI_SLUG, API_SR, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
