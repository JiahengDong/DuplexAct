#!/usr/bin/env python3
"""
Single-sample driver for the cue-injection actions (stop, hold_floor,
respond_and_recover), generalizing stop_common.py's `run_stop_sample_cli` /
`stop_cli_main`.

All three actions share one sample layout (lead_in.wav + cue.wav +
metadata.json, with the cue injected live while the model is speaking), so
every infer_stop_<model>.py client works for all of them. They differ in:
  1. metadata.json field names for the reference context. hold_floor uses
     `reference_before_text` / `cue_text` where trigger_controller.py expects
     `reference_original_response_text` / `reference_interruption_text`;
     see METADATA_FIELD_ALIASES.
  2. The timestamp keys each scorer reads (stop: `t_user_int_start`;
     hold_floor: `t_cue_start`/`t_cue_end`; respond_and_recover:
     `t_interruption_start`/`t_interruption_end`). All come from the cue
     segment's start and end; see derive_action_timestamps().

interrupt_user is not handled here: the user speaks continuously and the
driver must choose between two pre-recorded continuation branches depending
on whether the model barges in (see run_interrupt_batch.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Type

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stop_common import (  # noqa: E402
    build_conversation_wav, build_conversation_script,
    default_model_speech_transcripts, build_stop_cli_parser,
)


def _duplexact_root() -> Path:
    import os
    for key in ("DUPLEXACT_ROOT", "BENCH_ROOT"):
        v = os.environ.get(key)
        if v:
            return Path(v).resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "evaluation" / "paths.py").exists():
            return p
    return here.parents[min(3, len(here.parents)-1)]

REPO_ROOT = _duplexact_root()
DIRECT_REPLAY_DATA = REPO_ROOT / "evaluation" / "data"

# ── Action registry ──────────────────────────────────────────────────────
# Each action's data lives under evaluation/data/<action>/<l3_condition_dir>/<sample_id>/,
# and every sample dir has lead_in.wav + cue.wav + metadata.json.
ACTION_DATA_ROOTS: dict[str, Path] = {
    "stop": DIRECT_REPLAY_DATA / "stop",
    "hold_floor": DIRECT_REPLAY_DATA / "hold_floor",
    "respond_and_recover": DIRECT_REPLAY_DATA / "respond_and_recover",
}

# Per action: field name trigger_controller.py reads -> field name actually
# present in that action's metadata.json. stop and respond_and_recover already
# use the expected names.
METADATA_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "hold_floor": {
        "reference_original_response_text": "reference_before_text",
        "reference_interruption_text": "cue_text",
        # hold_floor has no interruption_type; l3_condition is the closest
        # description of the injection point for the trigger LLM.
        "interruption_type": "l3_condition",
    },
}


def normalize_reference_context(action: str, raw_metadata: dict) -> dict:
    """
    Return a copy of raw_metadata with the expected field names added from
    METADATA_FIELD_ALIASES. Original keys are kept, and an expected key is
    only filled if it is empty and its source field is non-empty.
    """
    ctx = dict(raw_metadata)
    aliases = METADATA_FIELD_ALIASES.get(action, {})
    for normalized_key, actual_key in aliases.items():
        if not ctx.get(normalized_key) and raw_metadata.get(actual_key):
            ctx[normalized_key] = raw_metadata[actual_key]
    return ctx


def derive_action_timestamps(action: str, simulator_speech_segments: list[dict],
                              trigger_result: dict) -> dict:
    """
    Map the injected cue's start and end (the trigger time and the
    `source="cue"` entry in `simulator_speech_segments`) to the
    EvalInstance.timestamps keys the given action's scorer reads.
    """
    t_start = trigger_result.get("t_user_int_start")
    cue_entries = [s for s in simulator_speech_segments if s.get("source") == "cue"]
    t_end = cue_entries[0]["end"] if cue_entries else None

    if action == "stop":
        return {"t_user_int_start": t_start}
    if action == "hold_floor":
        return {"t_cue_start": t_start, "t_cue_end": t_end}
    if action == "respond_and_recover":
        return {"t_interruption_start": t_start, "t_interruption_end": t_end}
    raise ValueError(f"derive_action_timestamps: unhandled action {action!r}")


async def run_cue_action_sample_cli(
    client_cls: Type,
    model_slug: str,
    api_sr: int,
    action: str,
    args,
    client_kwargs: Optional[dict] = None,
    api_key_env: Optional[str] = None,
    output_sr: Optional[int] = None,
) -> dict:
    """
    Run one cue-injection sample for `action`. The per-model client contract
    is the same as stop_common.run_stop_sample_cli. In addition:
      - `reference_context` goes through normalize_reference_context() so
        LiveTriggerController sees the expected field names.
      - `info["action"]` records the action.
      - `info["instance"]["timestamps"]` holds the keys from
        derive_action_timestamps() that the action's scorer reads.
    """
    import soundfile as sf
    from dotenv import load_dotenv
    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from realtime_common import load_and_resample  # noqa: E402

    sample_dir: Path = args.sample_dir
    lead_in_wav = sample_dir / "lead_in.wav"
    cue_wav     = sample_dir / "cue.wav"
    meta_path   = sample_dir / "metadata.json"

    conv_sr = output_sr if output_sr is not None else api_sr

    raw_metadata = json.loads(meta_path.read_text())
    reference_context = normalize_reference_context(action, raw_metadata)
    reference_context["action"] = action
    lead_in_duration = sf.info(str(lead_in_wav)).duration
    lead_in_pcm = load_and_resample(lead_in_wav, api_sr).tobytes()
    cue_pcm = load_and_resample(cue_wav, api_sr).tobytes()
    lead_in_pcm_conv = lead_in_pcm if conv_sr == api_sr else load_and_resample(lead_in_wav, conv_sr).tobytes()
    cue_pcm_conv = cue_pcm if conv_sr == api_sr else load_and_resample(cue_wav, conv_sr).tobytes()

    llm_client = None
    from trigger_controller import make_trigger_llm_client  # noqa: E402
    llm_client = make_trigger_llm_client()
    if llm_client is None:
        print("[WARN] OPENAI_API_KEY not set -- trigger controller will error on every "
              "poll and fall back to a forced trigger (onset/search timeout).", file=sys.stderr)

    client_api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key_env == "OPENAI_API_KEY":
        from openai_keys import current_key
        client_api_key = current_key() or client_api_key
    client = client_cls(
        api_key=client_api_key,
        model_name=args.model,
        system_prompt="",
        verbose=args.verbose,
        cue_pcm=cue_pcm,
        reference_context=reference_context,
        llm_client=llm_client,
        **(client_kwargs or {}),
    )
    try:
        result = await client.run_sample(
            input_wav=lead_in_wav,
            output_wav=args.output_wav,
            events_path=args.events_path,
            input_duration=lead_in_duration,
        )
    finally:
        if llm_client is not None and hasattr(llm_client, "on_sample_finished"):
            llm_client.on_sample_finished()
    result["trigger"] = client.trigger_result
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

    conversation_wav = args.output_wav.with_name(args.output_wav.stem + ".conversation.wav")
    build_conversation_wav(
        output_wav=args.output_wav, conversation_wav=conversation_wav, api_sr=conv_sr,
        lead_in_pcm=lead_in_pcm_conv, cue_pcm=cue_pcm_conv,
        t_user_int_start=client.trigger_result.get("t_user_int_start"),
    )
    result["conversation_wav"] = str(conversation_wav)

    timestamps = derive_action_timestamps(action, result["simulator_speech_segments"],
                                           client.trigger_result)

    print(json.dumps({k: v for k, v in result.items() if k != "conversation_text"}, indent=2))
    print("\n--- conversation ---")
    print(conversation_text)

    info = {
        "sample_id":    sample_dir.name,
        "action":       action,
        "l3_condition": reference_context.get("l3_condition"),
        "model":        model_slug,
        "instance":     {**reference_context, **client.trigger_result, "timestamps": timestamps},
        "inference":    result,
    }
    if args.info_path:
        args.info_path.write_text(json.dumps(info, indent=2))
    return info


def cue_action_cli_main(client_cls: Type, model_slug: str, api_sr: int, action: str,
                         default_model: str, client_kwargs: Optional[dict] = None,
                         api_key_env: Optional[str] = None, output_sr: Optional[int] = None) -> None:
    """Standalone single-sample entry point; the cue-action counterpart of
    stop_common.stop_cli_main."""
    import asyncio
    parser = build_stop_cli_parser(default_model)
    args = parser.parse_args()
    asyncio.run(run_cue_action_sample_cli(client_cls, model_slug, api_sr, action, args,
                                           client_kwargs, api_key_env, output_sr))
