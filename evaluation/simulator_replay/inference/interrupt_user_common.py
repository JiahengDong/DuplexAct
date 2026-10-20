#!/usr/bin/env python3
"""
Shared helpers for the per-model `interrupt_user` simulator clients
(infer_interrupt_<model>.py), the counterpart of stop_common.py.

Roles are reversed relative to stop: the simulator speaks one continuous
utterance and the model may barge in; when the LLM decides the model has
taken the turn, the remaining user speech is truncated (see
interrupt_controller.py).

Data layout (evaluation/data/interrupt_user/<l3_condition>/<sample_id>/):
user_speech.wav + branch_acknowledged.wav + branch_ignored.wav +
metadata.json. metadata.json's `branch_label` (the source recording's
outcome) is not used; the outcome depends on the model under test.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Type

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stop_common import build_conversation_script  # noqa: E402

POST_BRANCH_OBSERVATION_S = 15.0  # same convention as stop_common.POST_CUE_OBSERVATION_S


def build_interrupt_simulator_speech_segments(
    *,
    monitor_result: dict,
    reference_context: dict,
    input_duration: float,
    branch_acknowledged_pcm: bytes,
    branch_ignored_pcm: bytes,
    pcm_sr: int,
) -> list[dict]:
    """
    Shared simulator-side timeline for interrupt_user clients. user_speech
    `end` is the REAL truncation point (t_model_onset_detected, else the
    full clip). There is no second simulator clip: the user either stops
    or plays through. `pcm_sr` is unused.
    """
    t_onset = monitor_result.get("t_model_onset_detected")
    t_user_speech_end = t_onset if t_onset is not None else input_duration
    user_text = (reference_context.get("transcript_user_full")
                 or reference_context.get("transcript_user") or "")
    return [{
        "start": 0.0,
        "end": round(t_user_speech_end, 4),
        "text": user_text,
        "source": "user_speech",
    }]


def build_conversation_wav_interrupt(
    output_wav: Path, conversation_wav: Path, api_sr: int,
    user_speech_pcm: bytes, branch_pcm: bytes,
    t_user_speech_end: float, t_branch_start: Optional[float],
) -> None:
    """
    Two-sided recording for interrupt_user (see
    stop_common.build_conversation_wav for why raw sample offsets line up
    with output.wav).

    ch0 (simulator) = the user utterance from t=0, truncated at
    `t_user_speech_end` (the full clip if ignored, the take-turn point if
    acknowledged).
    ch1 (model) = output.wav verbatim.
    """
    import soundfile as sf
    model_pcm, sr = sf.read(str(output_wav), dtype="int16")
    if sr != api_sr:
        raise ValueError(f"output.wav sample rate {sr} != expected {api_sr}")
    n_model = len(model_pcm)

    sim = np.zeros(n_model, dtype=np.int16)

    user_arr = np.frombuffer(user_speech_pcm, dtype=np.int16)
    n_user = min(len(user_arr), int(round(t_user_speech_end * api_sr)), n_model)
    if n_user > 0:
        sim[:n_user] = user_arr[:n_user]

    stereo = np.stack([sim, model_pcm], axis=1)
    conversation_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(conversation_wav), stereo, api_sr, subtype="PCM_16")


def build_interrupt_cli_parser(default_model: str) -> argparse.ArgumentParser:
    """Shared argparse setup (counterpart of stop_common.build_stop_cli_parser)."""
    parser = argparse.ArgumentParser(
        description="Simulator interrupt_user-action CLI (single sample).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sample_dir", type=Path, required=True,
                         help="e.g. evaluation/data/interrupt_user/"
                              "interrupt_for_help/synth_interrupt_for_help_0000")
    parser.add_argument("--output_wav", type=Path, required=True)
    parser.add_argument("--events_path", type=Path, required=True)
    parser.add_argument("--info_path", type=Path, default=None)
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--verbose", action="store_true")
    return parser


async def run_interrupt_sample_cli(
    client_cls: Type,
    model_slug: str,
    api_sr: int,
    args: argparse.Namespace,
    client_kwargs: Optional[dict] = None,
    api_key_env: Optional[str] = None,
    output_sr: Optional[int] = None,
) -> dict:
    """
    Generic single-sample driver (counterpart of stop_common.run_stop_sample_cli).

    `client_cls` must implement the same contract as
    infer_interrupt_moshi.py's InterruptMoshiClient:
    - constructor: (api_key, model_name, system_prompt, verbose,
      branch_acknowledged_pcm, branch_ignored_pcm, reference_context,
      llm_client=None, **client_kwargs). llm_client is the OpenAI client
      InterruptMonitorController uses to decide when the simulator stops
      talking.
    - after run_sample() resolves: `.monitor_result` (dict, from
      InterruptMonitorController.run()'s return value) and
      `.simulator_speech_segments` are both populated.
    - EITHER exposes `.text_tokens: list[tuple[float, str]]` OR overrides
      `compute_model_speech_transcripts(segments)` (same convention as
      stop_common's contract).
    """
    import soundfile as sf
    from dotenv import load_dotenv
    # Project-root .env (OPENAI_API_KEY for the interrupt_user stop LLM).
    # Explicit path so this still works when cwd is inference/.
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from realtime_common import load_and_resample  # noqa: E402
    from stop_common import default_model_speech_transcripts  # noqa: E402

    sample_dir: Path = args.sample_dir
    user_speech_wav = sample_dir / "user_speech.wav"
    branch_ack_wav  = sample_dir / "branch_acknowledged.wav"
    branch_ign_wav  = sample_dir / "branch_ignored.wav"
    meta_path       = sample_dir / "metadata.json"

    conv_sr = output_sr if output_sr is not None else api_sr

    reference_context = json.loads(meta_path.read_text())
    # Dataset `user_speech.wav` is the prefix up to t_target.
    # `branch_ignored.wav` is the same utterance in full (transcript_user_full).
    # Stream the full clip continuously; t_target is scoring-only.
    full_user_wav = branch_ign_wav if branch_ign_wav.exists() else user_speech_wav
    user_speech_duration = sf.info(str(full_user_wav)).duration
    user_speech_pcm = load_and_resample(full_user_wav, api_sr).tobytes()
    # Branch wavs are not played; clients still take them as (empty) arguments.
    branch_ack_pcm  = b""
    branch_ign_pcm  = b""
    user_speech_pcm_conv = user_speech_pcm if conv_sr == api_sr else load_and_resample(full_user_wav, conv_sr).tobytes()

    from trigger_controller import make_trigger_llm_client  # noqa: E402
    llm_client = make_trigger_llm_client()
    if llm_client is None:
        print("[WARN] OPENAI_API_KEY not set -- interrupt_user LLM stop "
              "controller cannot poll and will never truncate (ignored branch).",
              file=sys.stderr)

    client_api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key_env == "OPENAI_API_KEY":
        from openai_keys import current_key
        client_api_key = current_key() or client_api_key
    client = client_cls(
        api_key=client_api_key,
        model_name=args.model,
        system_prompt="",
        verbose=args.verbose,
        branch_acknowledged_pcm=branch_ack_pcm,
        branch_ignored_pcm=branch_ign_pcm,
        reference_context=reference_context,
        llm_client=llm_client,
        **(client_kwargs or {}),
    )
    try:
        result = await client.run_sample(
            input_wav=full_user_wav,
            output_wav=args.output_wav,
            events_path=args.events_path,
            input_duration=user_speech_duration,
        )
    finally:
        if llm_client is not None and hasattr(llm_client, "on_sample_finished"):
            llm_client.on_sample_finished()
    result["monitor"] = client.monitor_result
    result["simulator_speech_segments"] = client.simulator_speech_segments

    if hasattr(client, "recompute_model_speech_segments"):
        result["model_speech_segments"] = client.recompute_model_speech_segments(
            args.output_wav, result["model_speech_segments"],
        )

    segments = result["model_speech_segments"]
    if hasattr(client, "compute_model_speech_transcripts"):
        model_speech_transcripts = client.compute_model_speech_transcripts(segments)
    else:
        model_speech_transcripts = default_model_speech_transcripts(
            getattr(client, "text_tokens", []), segments,
        )
    result["model_speech_transcripts"] = model_speech_transcripts

    conversation_turns, conversation_text = build_conversation_script(
        segments, model_speech_transcripts, result["simulator_speech_segments"],
    )
    result["conversation"] = conversation_turns
    result["conversation_text"] = conversation_text

    # The user_speech segment's "end" is already the real truncation point.
    user_seg = next((s for s in result["simulator_speech_segments"] if s.get("source") == "user_speech"), None)
    t_user_speech_end = user_seg["end"] if user_seg else user_speech_duration

    conversation_wav = args.output_wav.with_name(args.output_wav.stem + ".conversation.wav")
    build_conversation_wav_interrupt(
        output_wav=args.output_wav, conversation_wav=conversation_wav, api_sr=conv_sr,
        user_speech_pcm=user_speech_pcm_conv, branch_pcm=b"",
        t_user_speech_end=t_user_speech_end, t_branch_start=None,
    )
    result["conversation_wav"] = str(conversation_wav)

    print(json.dumps({k: v for k, v in result.items() if k != "conversation_text"}, indent=2))
    print("\n--- conversation ---")
    print(conversation_text)

    # interrupt_user scoring reads instance.timestamps {"t_target", "t_user_start"}.
    timestamps = {
        "t_target": reference_context.get("t_target"),
        "t_user_start": reference_context.get("t_user_start", 0.0),
    }

    info = {
        "sample_id":    sample_dir.name,
        "action":       "interrupt_user",
        "l3_condition": reference_context.get("l3_condition"),
        "model":        model_slug,
        "instance":     {**reference_context, **client.monitor_result, "timestamps": timestamps},
        "inference":    result,
    }
    if args.info_path:
        args.info_path.write_text(json.dumps(info, indent=2))
    return info


def interrupt_cli_main(client_cls: Type, model_slug: str, api_sr: int, default_model: str,
                        client_kwargs: Optional[dict] = None, api_key_env: Optional[str] = None,
                        output_sr: Optional[int] = None) -> None:
    """Standalone single-sample entry point (counterpart of stop_common.stop_cli_main)."""
    parser = build_interrupt_cli_parser(default_model)
    args = parser.parse_args()
    asyncio.run(run_interrupt_sample_cli(client_cls, model_slug, api_sr, args, client_kwargs,
                                          api_key_env, output_sr))
