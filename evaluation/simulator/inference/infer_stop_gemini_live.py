#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_stop_gemini_live.py

Gemini 3.1 Flash Live (Preview) client for the `stop` Level-2 action,
simulator-driven (live LLM-triggered injection -- see
trigger_controller.py's module docstring for the full architecture
rationale).

Unlike GPT-Realtime/Grok Voice (raw JSON-over-WebSocket, both sharing
realtime_api_stop_common.RealtimeAPIStopClientBase), Gemini Live is
accessed through the `google-genai` SDK's `client.aio.live.connect()`
session object (`sess.send_realtime_input(...)` / `async for resp in
sess.receive()`), so this file implements `_connect_and_stream` directly
against `RealtimeClientBase` rather than subclassing that shared base --
same category of provider (a true real-time streaming API, not a
round/chunk-based local model), just a different wire protocol.

Ported from direct_replay/inference/infer_gemini_live.py's connection/
session-config plumbing (see that file's own module docstring for why it's
ONE persistent `client.aio.live.connect()` session for the whole sample,
never reconnecting on turn boundaries -- R-01 fix); this file layers the
same `DynamicPCMSource` + `LiveTriggerController` pattern used by every
other stop-simulator client on top of it, and additionally turns ON
`output_audio_transcription` (the direct_replay client never requests it,
an explicit MINOR_ADJUST -- see MODEL_COMPATIBILITY.md) so the trigger
controller gets a real live-text signal via `sc.output_transcription.text`
instead of falling back to speaking-duration-only polling.

Poll interval: uses trigger_controller.POLL_INTERVAL_S's default (1.0s)
unmodified -- same reasoning as GPT-Realtime/Grok Voice in
realtime_api_stop_common.py's own docstring: this is a true real-time
streaming API, not a round/chunk-quantized local model, so there's no
"native chunk size" to derive a poll interval from.

This file is deliberately a SEPARATE module from
direct_replay/inference/infer_gemini_live.py (never touches it, never
imported by it), so the already-validated direct_replay pipeline is never
affected -- see MODEL_COMPATIBILITY.md's gemini-3.1-flash-live entry.

Standalone usage (single sample, for smoke testing):
    python infer_stop_gemini_live.py --sample_dir <path to Candor_stop_0038 etc> \\
        --output_wav out.wav --events_path events.jsonl
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

from realtime_common import (
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
    sleep_until_deadline, DynamicPCMSource,
)
from trigger_controller import LiveTriggerController, speech_rate_for
from stop_common import stop_cli_main, POST_CUE_OBSERVATION_S

GEMINI_LIVE_SLUG = "gemini-3.1-flash-live"
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"

SEND_SAMPLE_RATE    = 16_000   # Gemini Live: input PCM16 at 16kHz
RECEIVE_SAMPLE_RATE = 24_000   # Gemini Live: output PCM16 at 24kHz
SEND_CHUNK_SAMPLES  = 1024     # matches infer_gemini_live.py (~64ms/chunk)
SEND_CHUNK_BYTES    = SEND_CHUNK_SAMPLES * 2

THINKING_LEVEL = "minimal"   # matches infer_gemini_live.py's own default

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class GeminiLiveStopClient(RealtimeClientBase):
    MODEL_SLUG = GEMINI_LIVE_SLUG
    OUTPUT_SR = RECEIVE_SAMPLE_RATE
    INPUT_SR = SEND_SAMPLE_RATE

    def __init__(self, api_key, model_name, system_prompt, verbose,
                 cue_pcm, reference_context, llm_client):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.cue_pcm = cue_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result = {}
        self.text_tokens = []
        self.simulator_speech_segments = []

    def _make_client(self) -> genai.Client:
        # vertexai=False required for gemini-3.1-flash-live-preview -- see
        # infer_gemini_live.py's own module docstring.
        return genai.Client(vertexai=False, api_key=self.api_key)

    def _build_config(self) -> dict:
        cfg: dict = {
            "response_modalities": ["AUDIO"],
            "thinking_config": {"thinking_level": THINKING_LEVEL},
            # Turned ON (unlike infer_gemini_live.py) so the trigger
            # controller gets a real live-text signal -- see module
            # docstring's "additionally turns ON output_audio_transcription"
            # note.
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
                # interrupted. Re-enter so a later recover turn is heard.
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
                            # recover turn's text -- see
                            # LiveTriggerController.on_interrupt.
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

            controller = LiveTriggerController(
                recorder=recorder,
                source=source,
                cue_pcm=self.cue_pcm,
                now_fn=now_fn,
                reference_context=self.reference_context,
                client=self.llm_client,
                lead_in_duration_s=input_duration,
                event_logger=event_logger,
                speech_chars_per_s=speech_rate_for(self.MODEL_SLUG),
            )
            controller_holder["controller"] = controller
            trigger_task = asyncio.create_task(controller.run())

            def _on_trigger_done(t):
                source.finish()

            trigger_task.add_done_callback(_on_trigger_done)

            cue_drained_at = None
            i = 0
            try:
                while True:
                    chunk = source.pull(SEND_CHUNK_BYTES)
                    await sess.send_realtime_input(
                        audio={"data": chunk, "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}"}
                    )
                    event_logger.log("send", "realtime_input.audio", n_bytes=len(chunk))

                    if source.is_done and cue_drained_at is None:
                        cue_drained_at = now_fn()
                        event_logger.log("send", "cue_drained", {"t": cue_drained_at})
                    if cue_drained_at is not None and now_fn() - cue_drained_at >= POST_CUE_OBSERVATION_S:
                        break

                    i += 1
                    await sleep_until_deadline(loop, send_t0 + i * chunk_duration)
            finally:
                try:
                    await sess.send_realtime_input(audio_stream_end=True)
                    event_logger.log("send", "realtime_input.audio_stream_end")
                except Exception:
                    pass

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
                        self.trigger_result = {"error": type(e).__name__ + ": " + str(e)}

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
                    cue_duration = len(self.cue_pcm) / 2 / SEND_SAMPLE_RATE
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


def main():
    stop_cli_main(GeminiLiveStopClient, GEMINI_LIVE_SLUG, SEND_SAMPLE_RATE, DEFAULT_MODEL,
                  api_key_env="GEMINI_API_KEY", output_sr=RECEIVE_SAMPLE_RATE)


if __name__ == "__main__":
    main()
