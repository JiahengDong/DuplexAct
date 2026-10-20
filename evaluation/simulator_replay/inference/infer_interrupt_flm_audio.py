#!/usr/bin/env python3
"""
FLM-Audio client for the `interrupt_user` action. Uses the same Moshi-style
wire protocol and sphn==0.1.4 drain loops as infer_stop_flm_audio.py, driven
by InterruptMonitorController instead of LiveTriggerController.
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
    RealtimeClientBase, SynchronizedRecorder, EventLogger, DynamicPCMSource,
)
from interrupt_controller import InterruptMonitorController  # noqa: E402
from interrupt_user_common import (  # noqa: E402
    interrupt_cli_main, POST_BRANCH_OBSERVATION_S, build_interrupt_simulator_speech_segments,
)
from trigger_controller import poll_interval_for_native_chunk  # noqa: E402

FLM_AUDIO_SLUG = "flm-audio"
FLM_AUDIO_HOST = "localhost"
FLM_AUDIO_PORT = int(os.environ.get("FLM_AUDIO_PORT", "8990"))
DEFAULT_MODEL = "CofeAI/FLM-Audio"

API_SR        = 24_000
FRAME_SAMPLES = 1_920
FRAME_SEC     = FRAME_SAMPLES / API_SR
FRAME_BYTES   = FRAME_SAMPLES * 2
FLM_AUDIO_POLL_INTERVAL_S = poll_interval_for_native_chunk(FRAME_SEC)
FLM_AUDIO_SILENCE_GRACE_S = 0.8

REPO_DIR = opensource_root_for(FLM_AUDIO_SLUG) / "repo"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _pcm_f32_to_i16_bytes(pcm_f32: np.ndarray) -> bytes:
    return (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class InterruptFLMAudioClient(RealtimeClientBase):
    MODEL_SLUG = FLM_AUDIO_SLUG
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, branch_acknowledged_pcm: bytes, branch_ignored_pcm: bytes,
                 reference_context: dict, llm_client=None):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.branch_acknowledged_pcm = branch_acknowledged_pcm
        self.branch_ignored_pcm = branch_ignored_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.monitor_result: dict = {}
        self.text_tokens: list[tuple[float, str]] = []
        self.simulator_speech_segments: list[dict] = []

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
            cwd=REPO_DIR,
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

                    if kind == 1:
                        opus_reader.append_bytes(payload)
                        if speaking and (now - last_word_t) > FLM_AUDIO_SILENCE_GRACE_S:
                            recorder.set_speaking(False, at_s=last_word_t + FLM_AUDIO_SILENCE_GRACE_S)
                            speaking = False
                            event_logger.log("recv", "speech_segment_end", {"t": now})

                    elif kind == 2:
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
                while True:
                    await asyncio.sleep(0.001)
                    pcm_f32 = opus_reader.read_pcm()
                    if pcm_f32 is None or pcm_f32.shape[-1] == 0:
                        continue
                    await recorder.add(_pcm_f32_to_i16_bytes(pcm_f32))
                    event_logger.log("recv", "audio", {"n_samples": int(pcm_f32.shape[-1])})

            async def send_drain_loop():
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
            monitor = InterruptMonitorController(
                recorder=recorder, source=source,
                user_speech_duration=input_duration,
                branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                branch_ignored_pcm=self.branch_ignored_pcm,
                now_fn=now_fn, reference_context=self.reference_context,
                llm_client=self.llm_client, event_logger=event_logger,
                poll_interval_s=FLM_AUDIO_POLL_INTERVAL_S,
            )
            controller_holder["controller"] = monitor
            monitor_task = asyncio.create_task(monitor.run())
            monitor_task.add_done_callback(lambda t: source.finish())

            branch_drained_at: float | None = None
            try:
                while True:
                    chunk_i16 = source.pull(FRAME_BYTES)
                    frame_f32 = np.frombuffer(chunk_i16, dtype=np.int16).astype(np.float32) / 32768.0
                    opus_writer.append_pcm(frame_f32)
                    if source.is_done and branch_drained_at is None:
                        branch_drained_at = now_fn()
                        event_logger.log("send", "branch_drained", {"t": branch_drained_at})
                    if branch_drained_at is not None and now_fn() - branch_drained_at >= POST_BRANCH_OBSERVATION_S:
                        break
                    await asyncio.sleep(FRAME_SEC)
            finally:
                if not monitor_task.done():
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
                    try:
                        self.monitor_result = monitor_task.result()
                    except Exception as e:
                        self.monitor_result = {"error": f"{type(e).__name__}: {e}"}

                self.simulator_speech_segments = build_interrupt_simulator_speech_segments(
                    monitor_result=self.monitor_result,
                    reference_context=self.reference_context,
                    input_duration=input_duration,
                    branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                    branch_ignored_pcm=self.branch_ignored_pcm,
                    pcm_sr=API_SR,
                )
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
    interrupt_cli_main(InterruptFLMAudioClient, FLM_AUDIO_SLUG, API_SR, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
