#!/usr/bin/env python3
"""
direct_replay/inference/model_registry.py
=============================================
Central registry mapping model slugs (used as directory names throughout
`results/inference_results/<action>/<l3_condition>/<model_slug>/...`) to their
client class and required API key environment variable.

`run_inference.py` is the only place that should import this module -- model
client files (infer_*.py) never need to know about each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Type

from realtime_common import RealtimeClientBase

# Batch-replay clients (e.g. infer_voila.py) implement the same
# run_sample(...) contract as RealtimeClientBase but subclass
# local_common.BatchReplayClientBase instead -- both are valid client_cls
# values, run_inference.py only ever calls run_sample() on whichever it gets.


@dataclass
class ModelSpec:
    slug: str                      # directory-safe key, must match client.MODEL_SLUG
    model_name: str                # exact model identifier passed to the API
    client_cls: Optional[Type]     # None if not yet implemented (resolved lazily below)
    api_key_env: str               # environment variable holding the API key
    default_system_prompt: str


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally. "
    "Keep your responses concise."
)


def _load_gpt_realtime_cls():
    from infer_gpt_realtime import GPTRealtimeClient
    return GPTRealtimeClient


def _load_gemini_live_cls():
    from infer_gemini_live import GeminiLiveClient
    return GeminiLiveClient


def _load_qwen_realtime_cls():
    from infer_qwen_realtime import QwenRealtimeClient
    return QwenRealtimeClient


def _load_grok_voice_cls():
    from infer_grok_voice import GrokVoiceClient
    return GrokVoiceClient


def _load_moshi_cls():
    from infer_moshi import MoshiClient
    return MoshiClient


def _load_minicpm_o_cls():
    from infer_minicpm_o import MiniCPMDuplexClient
    return MiniCPMDuplexClient


def _load_voila_cls():
    from infer_voila import VoilaAutonomousClient
    return VoilaAutonomousClient


def _load_bayling_cls():
    from infer_bayling import BayLingDuplexClient
    return BayLingDuplexClient


def _load_raon_cls():
    from infer_raon import RaonSpeechChatClient
    return RaonSpeechChatClient


def _load_freeze_omni_cls():
    from infer_freeze_omni import FreezeOmniClient
    return FreezeOmniClient


def _load_mini_omni2_cls():
    from infer_mini_omni2 import MiniOmni2Client
    return MiniOmni2Client


def _load_vita15_cls():
    from infer_vita15 import VITA15Client
    return VITA15Client


def _load_flm_audio_cls():
    from infer_flm_audio import FLMAudioClient
    return FLMAudioClient


def _load_lychee_fd_cls():
    from infer_lychee_fd import LycheeFDClient
    return LycheeFDClient


def _load_nemotron_voicechat_cls():
    from infer_nemotron_voicechat import NemotronVoiceChatClient
    return NemotronVoiceChatClient


# Lazily-resolved so importing this module never requires every model's SDK
# (e.g. google-genai, boto3, sphn) to be installed just to run one model's
# batch -- doubly important for open-source models, since each one lives in
# its own venv (see opensource_models/<slug>/venv/) and most of their
# packages simply won't be importable from any other model's environment.
_LAZY_LOADERS: dict[str, Callable[[], Type[RealtimeClientBase]]] = {
    "gpt-realtime-2.1":            _load_gpt_realtime_cls,
    "gemini-3.1-flash-live":       _load_gemini_live_cls,
    "qwen3.5-omni-flash-realtime": _load_qwen_realtime_cls,
    "grok-voice":                  _load_grok_voice_cls,
    "moshi":                       _load_moshi_cls,
    "minicpm-o-4.5":               _load_minicpm_o_cls,
    "voila-autonomous":            _load_voila_cls,
    "bayling-duplex":              _load_bayling_cls,
    "raon-speechchat":             _load_raon_cls,
    "freeze-omni":                 _load_freeze_omni_cls,
    "mini-omni2":                  _load_mini_omni2_cls,
    "vita-1.5":                    _load_vita15_cls,
    "flm-audio":                   _load_flm_audio_cls,
    "lychee-fd":                   _load_lychee_fd_cls,
    "nemotron-voicechat":          _load_nemotron_voicechat_cls,
    # "nova-2-sonic":                  lambda: __import__("infer_nova_sonic").NovaSonicClient,
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt-realtime-2.1": ModelSpec(
        slug="gpt-realtime-2.1",
        model_name="gpt-realtime-2.1",
        client_cls=None,   # resolved lazily via _LAZY_LOADERS
        api_key_env="OPENAI_API_KEY",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    ),
    "gemini-3.1-flash-live": ModelSpec(
        slug="gemini-3.1-flash-live",
        model_name="gemini-3.1-flash-live-preview",
        client_cls=None,
        api_key_env="GEMINI_API_KEY",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    ),
    "nova-2-sonic": ModelSpec(
        slug="nova-2-sonic",
        model_name="amazon.nova-2-sonic-v1:0",
        client_cls=None,
        api_key_env="AWS_PROFILE",   # Bedrock uses AWS credential chain, not a single key
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    ),
    "grok-voice": ModelSpec(
        slug="grok-voice",
        model_name="grok-voice-latest",
        client_cls=None,
        api_key_env="XAI_API_KEY",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    ),
    "qwen3.5-omni-flash-realtime": ModelSpec(
        slug="qwen3.5-omni-flash-realtime",
        model_name="qwen3.5-omni-flash-realtime",
        client_cls=None,
        api_key_env="DASHSCOPE_API_KEY",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    ),
    # ── Open-source, locally-hosted models (Tier 1) ──────────────────────
    "moshi": ModelSpec(
        slug="moshi",
        model_name="kyutai/moshiko-pytorch-bf16",   # HF repo id passed to --hf-repo
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",     # locally hosted, no API key needed
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,   # unused by Moshi (no system prompt support)
    ),
    "minicpm-o-4.5": ModelSpec(
        slug="minicpm-o-4.5",
        model_name="openbmb/MiniCPM-o-4_5",   # HF repo id passed to AutoModel.from_pretrained
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt="Streaming Omni Conversation.",   # used as prefix_system_prompt
    ),
    "voila-autonomous": ModelSpec(
        slug="voila-autonomous",
        model_name="maitrix-org/Voila-autonomous-preview",   # HF repo id passed to load_model()
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt="",   # Voila's "instruction" field; autonomous ckpt has no chat system-prompt convention
    ),
    "bayling-duplex": ModelSpec(
        slug="bayling-duplex",
        model_name="BayLing-Models/BayLing-Duplex",   # HF repo id passed to BayLingDuplex(model_path=...)
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt="",   # BayLingDuplex has no system-prompt/instruction parameter at all
    ),
    "raon-speechchat": ModelSpec(
        slug="raon-speechchat",
        model_name="KRAFTON/Raon-SpeechChat-9B",   # HF repo id passed to AutoModel.from_pretrained
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,   # real chat-style system prompt, same convention as closed-source clients
    ),
    "freeze-omni": ModelSpec(
        slug="freeze-omni",
        model_name="VITA-MLLM/Freeze-Omni",   # + Qwen/Qwen2-7B-Instruct frozen LLM backbone (hardcoded in infer_freeze_omni.py)
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt="You are a helpful voice assistant. Your answer should be "
                               "coherent, natural, simple, complete. Do not answer too long.",   # matches official demo's own "role" default
    ),
    "mini-omni2": ModelSpec(
        slug="mini-omni2",
        model_name="gpt-omni/mini-omni2",   # HF repo id, downloaded as a flat ckpt_dir
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt="",   # run_AT_batch_stream's prompt scaffolding has no system-prompt slot at all
    ),
    "vita-1.5": ModelSpec(
        slug="vita-1.5",
        model_name="VITA-MLLM/VITA-1.5",   # + VITA-1.5_AudioEnc / InternViT-300M-448px (auto-resolved via config.json)
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,   # real chat-template system-message slot ("lang" modality), same convention as closed-source clients
    ),
    "flm-audio": ModelSpec(
        slug="flm-audio",
        model_name="CofeAI/FLM-Audio",   # single repo, passed as both --mimi-path and --model-path
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt="",   # no text-preamble/instruction-seeding mechanism at all (same situation as Moshi)
    ),
    "lychee-fd": ModelSpec(
        slug="lychee-fd",
        model_name="HIT-TMG/Lychee-FD",   # + stepfun-ai/Step-Audio-2-mini's token2wav/ vocoder (hardcoded path in infer_lychee_fd.py)
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt="",   # run_chunk_dialogue_inference() has no per-call system-prompt parameter
    ),
    "nemotron-voicechat": ModelSpec(
        slug="nemotron-voicechat",
        model_name="nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",   # HF repo id, downloaded to a flat local dir
        client_cls=None,
        api_key_env="LOCAL_NO_KEY",
        default_system_prompt=(
            "You are an AI voice assistant developed by NVIDIA. "
            "Your name is NVIDIA Voice Chat. "
            "Answer in a spoken, conversational style rather than a written one. "
            "Do not repeat the same sentence over and over again. "
            "Start the conversation by greeting the user."
        ),   # verbatim copy of the official offline_voicechat_infer.py's own DEFAULT_SYSTEM_PROMPT
    ),
}


def get_model_spec(slug: str) -> ModelSpec:
    if slug not in MODEL_SPECS:
        raise KeyError(
            f"Unknown model slug {slug!r}. Available: {sorted(MODEL_SPECS)}"
        )
    return MODEL_SPECS[slug]


def get_client_cls(slug: str) -> Type[RealtimeClientBase]:
    if slug not in _LAZY_LOADERS:
        raise NotImplementedError(
            f"Model {slug!r} is registered but its client is not implemented yet."
        )
    return _LAZY_LOADERS[slug]()


def available_models() -> list[str]:
    """Slugs with a working client implementation (importable now)."""
    out = []
    for slug in MODEL_SPECS:
        if slug in _LAZY_LOADERS:
            try:
                _LAZY_LOADERS[slug]()
                out.append(slug)
            except Exception:
                pass
    return out


def all_registered_models() -> list[str]:
    """All known slugs, including ones not yet implemented."""
    return sorted(MODEL_SPECS)
