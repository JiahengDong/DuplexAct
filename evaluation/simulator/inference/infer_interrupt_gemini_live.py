#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_interrupt_gemini_live.py

Gemini Live interrupt_user client. Separate from infer_stop_gemini_live.py.
Same live.connect session + output_audio_transcription; InterruptMonitorController.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "direct_replay" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, DynamicPCMSource,
)
from interrupt_controller import InterruptMonitorController  # noqa: E402
from trigger_controller import speech_rate_for  # noqa: E402
from interrupt_user_common import (  # noqa: E402
    interrupt_cli_main, POST_BRANCH_OBSERVATION_S, build_interrupt_simulator_speech_segments,
)

GEMINI_LIVE_SLUG = "gemini-3.1-flash-live"
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
SEND_SAMPLE_RATE    = 16_000
RECEIVE_SAMPLE_RATE = 24_000
SEND_CHUNK_SAMPLES  = 1024
SEND_CHUNK_BYTES    = SEND_CHUNK_SAMPLES * 2
THINKING_LEVEL = "minimal"
SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class InterruptGeminiLiveClient(RealtimeClientBase):
    MODEL_SLUG = GEMINI_LIVE_SLUG
    OUTPUT_SR = RECEIVE_SAMPLE_RATE
    INPUT_SR = SEND_SAMPLE_RATE

    def __init__(self, api_key, model_name, system_prompt, verbose,
                 branch_acknowledged_pcm, branch_ignored_pcm, reference_context,
                 llm_client=None):
        super().__init__(api_key, model_name, system_prompt or SYSTEM_PROMPT, verbose)
        self.branch_acknowledged_pcm = branch_acknowledged_pcm
        self.branch_ignored_pcm = branch_ignored_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.monitor_result = {}
        self.text_tokens = []
        self.simulator_speech_segments = []

    def _make_client(self) -> genai.Client:
        return genai.Client(vertexai=False, api_key=self.api_key)

    def _build_config(self) -> dict:
        cfg: dict = {
            "response_modalities": ["AUDIO"],
            "thinking_config": {"thinking_level": THINKING_LEVEL},
            "output_audio_transcription": types.AudioTranscriptionConfig(),
        }
        if self.system_prompt:
            cfg["system_instruction"] = self.system_prompt
        return cfg

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder,
                                   event_logger: EventLogger, start_streaming) -> None:
        client = self._make_client()
        config = self._build_config()
        chunk_duration = SEND_CHUNK_SAMPLES / SEND_SAMPLE_RATE
        stream_t0 = {"t": None}
        controller_holder = {"controller": None}
        turn_no = 0

        async with client.aio.live.connect(model=self.model_name, config=config) as sess:
            event_logger.log("send", "session.connect")

            def now_fn():
                return time.monotonic() - stream_t0["t"]

            async def receiver():
                nonlocal turn_no
                # google-genai sess.receive() ends on turn_complete /
                # interrupted. Re-enter so a later turn is heard.
                while True:
                    async for resp in sess.receive():
                        sc = resp.server_content
                        if not sc:
                            continue
                        now = now_fn() if stream_t0["t"] is not None else None
                        turn_ended = False
                        if getattr(sc, "interrupted", False):
                            recorder.interrupt(reason="gemini_interrupted",
                                                response_id=f"turn-{turn_no}")
                            # Text belonging to the audio just discarded was
                            # never heard, so it must not stay in front of the
                            # next turn's text -- see
                            # InterruptMonitorController.on_interrupt.
                            ctrl = controller_holder["controller"]
                            if ctrl is not None:
                                ctrl.on_interrupt()
                            event_logger.log("recv", "interrupted", {"turn": turn_no})
                            turn_ended = True
                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    await recorder.add(part.inline_data.data,
                                                        response_id=f"turn-{turn_no}")
                                    event_logger.log("recv", "model_turn.audio",
                                                      n_bytes=len(part.inline_data.data))
                                if getattr(part, "generation_complete", False):
                                    turn_ended = True
                        ot = getattr(sc, "output_transcription", None)
                        if ot is not None and ot.text:
                            self.text_tokens.append((now, ot.text))
                            ctrl = controller_holder["controller"]
                            if ctrl is not None and now is not None:
                                ctrl.on_text(ot.text, at_s=now)
                            event_logger.log("recv", "output_transcription", {"text": ot.text})
                        if getattr(sc, "turn_complete", False) or getattr(sc, "generation_complete", False):
                            turn_ended = True
                        if turn_ended:
                            event_logger.log("recv", "turn_complete", {"turn": turn_no})
                            turn_no += 1

            recv_task = asyncio.create_task(receiver())
            loop = asyncio.get_event_loop()
            send_t0 = await start_streaming()
            stream_t0["t"] = send_t0
            source = DynamicPCMSource(initial_pcm=input_pcm.astype(np.int16).tobytes())
            monitor = InterruptMonitorController(
                recorder=recorder, source=source,
                user_speech_duration=input_duration,
                branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                branch_ignored_pcm=self.branch_ignored_pcm,
                now_fn=now_fn, reference_context=self.reference_context,
                llm_client=self.llm_client, event_logger=event_logger,
                speech_chars_per_s=speech_rate_for(self.MODEL_SLUG),
            )
            controller_holder["controller"] = monitor
            monitor_task = asyncio.create_task(monitor.run())
            monitor_task.add_done_callback(lambda t: source.finish())

            branch_drained_at = None
            i = 0
            try:
                while True:
                    chunk = source.pull(SEND_CHUNK_BYTES)
                    await sess.send_realtime_input(
                        audio={"data": chunk, "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}"}
                    )
                    event_logger.log("send", "realtime_input.audio", n_bytes=len(chunk))
                    if source.is_done and branch_drained_at is None:
                        branch_drained_at = now_fn()
                        event_logger.log("send", "branch_drained", {"t": branch_drained_at})
                    if (branch_drained_at is not None
                            and now_fn() - branch_drained_at >= POST_BRANCH_OBSERVATION_S):
                        break
                    i += 1
                    await sleep_until_deadline(loop, send_t0 + i * chunk_duration)
            finally:
                try:
                    await sess.send_realtime_input(audio_stream_end=True)
                    event_logger.log("send", "realtime_input.audio_stream_end")
                except Exception:
                    pass
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
                        self.monitor_result = {"error": type(e).__name__ + ": " + str(e)}
                self.simulator_speech_segments = build_interrupt_simulator_speech_segments(
                    monitor_result=self.monitor_result,
                    reference_context=self.reference_context,
                    input_duration=input_duration,
                    branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                    branch_ignored_pcm=self.branch_ignored_pcm,
                    pcm_sr=SEND_SAMPLE_RATE,
                )
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass


def main():
    interrupt_cli_main(InterruptGeminiLiveClient, GEMINI_LIVE_SLUG, SEND_SAMPLE_RATE, DEFAULT_MODEL,
                       api_key_env="GEMINI_API_KEY", output_sr=RECEIVE_SAMPLE_RATE)


if __name__ == "__main__":
    main()
