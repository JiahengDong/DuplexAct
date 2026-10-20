#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/infer_stop_grok_voice.py

xAI Grok Voice client for the `stop` Level-2 action, simulator-driven
(live LLM-triggered injection -- see trigger_controller.py's module
docstring for the full architecture rationale). Thin subclass of
realtime_api_stop_common.RealtimeAPIStopClientBase -- see that module's
own docstring for what's shared vs. provider-specific (Grok Voice's wire
protocol is, per its own docs, essentially the same GA schema as OpenAI's
Realtime API -- see infer_grok_voice.py's own module docstring).

This file is deliberately a SEPARATE module from
direct_replay/inference/infer_grok_voice.py (never touches it, never
imported by it), so the already-validated direct_replay pipeline is never
affected -- see MODEL_COMPATIBILITY.md's grok-voice entry.

Standalone usage (single sample, for smoke testing):
    python infer_stop_grok_voice.py --sample_dir <path to Candor_stop_0038 etc> \\
        --output_wav out.wav --events_path events.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_api_stop_common import RealtimeAPIStopClientBase
from stop_common import stop_cli_main

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


class GrokVoiceStopClient(RealtimeAPIStopClientBase):
    MODEL_SLUG = GROK_VOICE_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = API_SR
    WS_URL = REALTIME_WS_URL
    CHUNK_MS = 20
    API_SR = API_SR

    def __init__(self, api_key, model_name, system_prompt, verbose,
                 cue_pcm, reference_context, llm_client, voice=DEFAULT_VOICE):
        super().__init__(api_key, model_name, system_prompt, verbose,
                          cue_pcm, reference_context, llm_client)
        self.voice = voice

    def build_session_update(self):
        # Same shape as infer_grok_voice.py's own session_update -- flatter
        # than GPT's (voice/instructions/turn_detection at the top level of
        # "session", not nested under "audio"). No threshold/silence_
        # duration_ms override, same as the original client -- use xAI's
        # documented server_vad defaults.
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
    stop_cli_main(GrokVoiceStopClient, GROK_VOICE_SLUG, API_SR, DEFAULT_MODEL,
                  client_kwargs={"voice": DEFAULT_VOICE}, api_key_env="XAI_API_KEY")


if __name__ == "__main__":
    main()
