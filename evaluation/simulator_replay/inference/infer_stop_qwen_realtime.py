#!/usr/bin/env python3
"""
Qwen3.5-Omni-Flash-Realtime (DashScope) client for the `stop` Level-2
action. Thin subclass of realtime_api_stop_common; session.update matches
direct_replay/inference/infer_qwen_realtime.py.

Differences from the GPT Realtime API:
  - WS URL is wss://{DASHSCOPE_WS_HOST}/api-ws/v1/realtime?model=...
  - session.update is flat (modalities/voice/pcm format strings)
  - input PCM is 16 kHz, output PCM is 24 kHz
  - live text comes from response.audio_transcript.delta, falling back to
    response.audio_transcript.done when Qwen sends no deltas

Like the GPT/Grok stop clients, this does not send response.cancel on
speech_started: cancelling client-side would implement "stop" for the
model instead of measuring it.

hold_floor / respond_and_recover reuse this client via run_action_batch.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_api_stop_common import RealtimeAPIStopClientBase
from stop_common import stop_cli_main

QWEN_SLUG = "qwen3.5-omni-flash-realtime"
DEFAULT_MODEL = "qwen3.5-omni-flash-realtime"
DEFAULT_VOICE = "Ethan"

INPUT_SR = 16_000
OUTPUT_SR = 24_000
API_SR = INPUT_SR

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class QwenRealtimeStopClient(RealtimeAPIStopClientBase):
    MODEL_SLUG = QWEN_SLUG
    OUTPUT_SR = OUTPUT_SR
    INPUT_SR = INPUT_SR
    CHUNK_MS = 20
    API_SR = API_SR

    def __init__(self, api_key, model_name, system_prompt, verbose,
                 cue_pcm, reference_context, llm_client,
                 voice=DEFAULT_VOICE, ws_host=None):
        super().__init__(api_key, model_name, system_prompt, verbose,
                         cue_pcm, reference_context, llm_client)
        self.voice = voice
        self.ws_host = ws_host or os.getenv("DASHSCOPE_WS_HOST", "")
        if not self.ws_host:
            raise RuntimeError(
                "DASHSCOPE_WS_HOST not set. Qwen realtime requires a host "
                "(e.g. dashscope-intl.aliyuncs.com)."
            )

    def websocket_url(self) -> str:
        return f"wss://{self.ws_host}/api-ws/v1/realtime?model={self.model_name}"

    def build_session_update(self):
        return {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self.voice,
                "instructions": SYSTEM_PROMPT,
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "turn_detection": {
                    "type": "semantic_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 800,
                },
            },
        }


def main():
    stop_cli_main(
        QwenRealtimeStopClient, QWEN_SLUG, API_SR, DEFAULT_MODEL,
        client_kwargs={"voice": DEFAULT_VOICE}, api_key_env="DASHSCOPE_API_KEY",
        output_sr=OUTPUT_SR,
    )


if __name__ == "__main__":
    main()
