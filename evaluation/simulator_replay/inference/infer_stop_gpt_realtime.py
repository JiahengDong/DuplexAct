#!/usr/bin/env python3
"""
OpenAI GPT-Realtime client for the simulator-driven `stop` action (live
LLM-triggered cue injection; see trigger_controller.py). Thin subclass of
realtime_api_stop_common.RealtimeAPIStopClientBase, which documents what is
shared vs. provider-specific.

Standalone usage (single sample):
    python infer_stop_gpt_realtime.py --sample_dir <path to a stop sample dir> \\
        --output_wav out.wav --events_path events.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_api_stop_common import RealtimeAPIStopClientBase
from stop_common import stop_cli_main

GPT_REALTIME_SLUG = "gpt-realtime-2.1"
DEFAULT_MODEL = "gpt-realtime-2.1"

REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
API_SR = 24000

SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


class GPTRealtimeStopClient(RealtimeAPIStopClientBase):
    MODEL_SLUG = GPT_REALTIME_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = API_SR
    WS_URL = REALTIME_WS_URL
    CHUNK_MS = 20
    API_SR = API_SR
    # OpenAI is the only supported realtime provider that accepts this; see
    # the base class's SUPPORTS_ITEM_TRUNCATE comment.
    SUPPORTS_ITEM_TRUNCATE = True

    def build_session_update(self):
        # Same session as direct_replay's infer_gpt_realtime.py: semantic_vad
        # decides when to respond, with no forced response.create.
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
    stop_cli_main(GPTRealtimeStopClient, GPT_REALTIME_SLUG, API_SR, DEFAULT_MODEL,
                  api_key_env="OPENAI_API_KEY")


if __name__ == "__main__":
    main()
