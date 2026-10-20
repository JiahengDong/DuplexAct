#!/usr/bin/env python3
"""
Moshi client for the simulator-driven `interrupt_user` action (see
interrupt_controller.py). Roles are reversed relative to stop/hold_floor/
respond_and_recover: the simulator plays `user_speech.wav` continuously and
watches for the model to barge in.

Uses the same Moshi server and wire protocol as infer_stop_moshi.py. The
differences from infer_stop_moshi.py's `_connect_and_stream`:
  1. `source`'s initial content is `user_speech.wav`, not `lead_in.wav`.
  2. The concurrent controller is InterruptMonitorController.
  3. `simulator_speech_segments` has an entry for user_speech (whose `end`
     is the actual truncation point) and one for the branch clip selected
     live.

Standalone usage (single sample):
    python infer_interrupt_moshi.py --sample_dir <path to an interrupt_user sample dir> \\
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
from interrupt_controller import InterruptMonitorController  # noqa: E402
from interrupt_user_common import interrupt_cli_main, POST_BRANCH_OBSERVATION_S  # noqa: E402
from trigger_controller import poll_interval_for_native_chunk  # noqa: E402

MOSHI_SLUG = "moshi"
MOSHI_HOST = "localhost"
MOSHI_PORT = int(os.environ.get("MOSHI_PORT", "8998"))
DEFAULT_MODEL = "kyutai/moshiko-pytorch-bf16"

API_SR        = 24_000
FRAME_SAMPLES = 1_920          # 80ms @ 24kHz -- Mimi's native frame size
FRAME_SEC     = FRAME_SAMPLES / API_SR
FRAME_BYTES   = FRAME_SAMPLES * 2   # int16

MOSHI_SILENCE_GRACE_S = 0.8    # same value as infer_stop_moshi.py
MOSHI_POLL_INTERVAL_S = poll_interval_for_native_chunk(FRAME_SEC)  # 1.0s unified user-side poll

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _pcm_f32_to_i16_bytes(pcm_f32: np.ndarray) -> bytes:
    return (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class InterruptMoshiClient(RealtimeClientBase):
    MODEL_SLUG = MOSHI_SLUG
    OUTPUT_SR  = API_SR
    INPUT_SR   = API_SR

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, branch_acknowledged_pcm: bytes, branch_ignored_pcm: bytes,
                 reference_context: dict, llm_client=None):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.branch_acknowledged_pcm = branch_acknowledged_pcm   # int16 PCM bytes @ API_SR
        self.branch_ignored_pcm = branch_ignored_pcm             # int16 PCM bytes @ API_SR
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.monitor_result: dict = {}             # populated by _connect_and_stream
        self.text_tokens: list[tuple[float, str]] = []   # (t, word/punct) as received
        self.simulator_speech_segments: list[dict] = []   # what the simulator played and when

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        # `input_pcm`/`input_duration` here are user_speech.wav's, per
        # interrupt_user_common.run_interrupt_sample_cli's contract.
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

            # Manual segmentation from t=0, so priming audio is never
            # registered as a speech segment.
            recorder.set_speaking(False, at_s=0.0)

            await start_streaming()
            stream_t0["t"] = time.monotonic()

            def now_fn() -> float:
                return time.monotonic() - stream_t0["t"]

            source = DynamicPCMSource(initial_pcm=input_pcm.tobytes())

            monitor = InterruptMonitorController(
                recorder=recorder,
                source=source,
                user_speech_duration=input_duration,
                branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                branch_ignored_pcm=self.branch_ignored_pcm,
                now_fn=now_fn,
                reference_context=self.reference_context,
                llm_client=self.llm_client,
                event_logger=event_logger,
                poll_interval_s=MOSHI_POLL_INTERVAL_S,
            )
            controller_holder["controller"] = monitor
            monitor_task = asyncio.create_task(monitor.run())

            def _on_monitor_done(t) -> None:
                # Once the selected branch is appended, mark the source
                # finished; pull() keeps returning silence and is_done starts
                # the POST_BRANCH_OBSERVATION_S countdown.
                source.finish()

            monitor_task.add_done_callback(_on_monitor_done)

            branch_drained_at: float | None = None
            try:
                while True:
                    chunk_i16 = source.pull(FRAME_BYTES)
                    frame_f32 = np.frombuffer(chunk_i16, dtype=np.int16).astype(np.float32) / 32768.0
                    opus_bytes = opus_writer.append_pcm(frame_f32)
                    if opus_bytes:
                        await ws.send(b"\x01" + opus_bytes)
                        event_logger.log("send", "audio", n_bytes=len(opus_bytes))

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

                # Simulator-side timeline. The user_speech segment ends at the
                # actual truncation point: input_duration if the model never
                # barged in, else where truncate_pending() cut it off.
                t_onset = self.monitor_result.get("t_model_onset_detected")
                t_user_speech_end = t_onset if t_onset is not None else input_duration
                self.simulator_speech_segments = [
                    {
                        "start": 0.0,
                        "end": round(t_user_speech_end, 4),
                        "text": self.reference_context.get("transcript_user", ""),
                        "source": "user_speech",
                    }
                ]
                t_branch = self.monitor_result.get("t_branch_appended")
                outcome = self.monitor_result.get("outcome")
                if t_branch is not None and outcome in ("acknowledged", "ignored"):
                    branch_pcm = (self.branch_acknowledged_pcm if outcome == "acknowledged"
                                  else self.branch_ignored_pcm)
                    branch_text_key = ("branch_acknowledged_text" if outcome == "acknowledged"
                                       else "branch_ignored_text")
                    branch_duration = len(branch_pcm) / 2 / API_SR   # int16 @ API_SR
                    self.simulator_speech_segments.append({
                        "start": round(t_branch, 4),
                        "end": round(t_branch + branch_duration, 4),
                        "text": self.reference_context.get(branch_text_key, ""),
                        "source": f"branch_{outcome}",
                    })

                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

                if speaking:
                    recorder.set_speaking(False, at_s=now_fn())


def main():
    interrupt_cli_main(InterruptMoshiClient, MOSHI_SLUG, API_SR, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
