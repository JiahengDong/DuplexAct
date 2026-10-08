#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_interrupt_grok_voice.py

xAI Grok Voice interrupt_user client. Thin subclass of
realtime_api_interrupt_common -- same session.update as infer_stop_grok_voice.py.
Never edits the stop client or the direct_replay grok client.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_api_interrupt_common import RealtimeAPIInterruptClientBase
from interrupt_user_common import interrupt_cli_main

GROK_VOICE_SLUG = "grok-voice"
DEFAULT_MODEL = "grok-voice-latest"
DEFAULT_VOICE = "eve"

REALTIME_WS_URL = "wss://api.x.ai/v1/realtime"
API_SR = 24000

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class InterruptGrokVoiceClient(RealtimeAPIInterruptClientBase):
    MODEL_SLUG = GROK_VOICE_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = API_SR
    WS_URL = REALTIME_WS_URL
    CHUNK_MS = 20
    API_SR = API_SR

    def __init__(self, api_key, model_name, system_prompt, verbose,
                 branch_acknowledged_pcm, branch_ignored_pcm, reference_context,
                 llm_client=None, voice=DEFAULT_VOICE):
        super().__init__(api_key, model_name, system_prompt, verbose,
                         branch_acknowledged_pcm, branch_ignored_pcm,
                         reference_context, llm_client)
        self.voice = voice

    def build_session_update(self):
        # Same shape as infer_stop_grok_voice.py / infer_grok_voice.py.
        return {
            "type": "session.update",
            "session": {
                "voice": self.voice,
                "instructions": SYSTEM_PROMPT,
                "turn_detection": {"type": "server_vad"},
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": API_SR}},
                    "output": {"format": {"type": "audio/pcm", "rate": API_SR}},
                },
            },
        }


def main():
    interrupt_cli_main(
        InterruptGrokVoiceClient, GROK_VOICE_SLUG, API_SR, DEFAULT_MODEL,
        client_kwargs={"voice": DEFAULT_VOICE}, api_key_env="XAI_API_KEY",
    )


if __name__ == "__main__":
    main()
