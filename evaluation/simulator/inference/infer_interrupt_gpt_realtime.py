#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_interrupt_gpt_realtime.py

GPT-Realtime interrupt_user client. Thin subclass of
realtime_api_interrupt_common -- same session.update as infer_stop_gpt_realtime.py.
Never edits the stop client.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_api_interrupt_common import RealtimeAPIInterruptClientBase
from interrupt_user_common import interrupt_cli_main

GPT_REALTIME_SLUG = "gpt-realtime-2.1"
DEFAULT_MODEL = "gpt-realtime-2.1"
REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
API_SR = 24000

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class InterruptGPTRealtimeClient(RealtimeAPIInterruptClientBase):
    MODEL_SLUG = GPT_REALTIME_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = API_SR
    WS_URL = REALTIME_WS_URL
    CHUNK_MS = 20
    API_SR = API_SR
    # OpenAI is the only provider of the four that accepts this -- see the
    # base class's SUPPORTS_ITEM_TRUNCATE comment.
    SUPPORTS_ITEM_TRUNCATE = True

    def build_session_update(self):
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model_name,
                "instructions": SYSTEM_PROMPT,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": API_SR},
                        "turn_detection": {"type": "semantic_vad"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": API_SR},
                        "voice": "alloy",
                    },
                },
            },
        }


def main():
    interrupt_cli_main(InterruptGPTRealtimeClient, GPT_REALTIME_SLUG, API_SR, DEFAULT_MODEL,
                       api_key_env="OPENAI_API_KEY")


if __name__ == "__main__":
    main()
