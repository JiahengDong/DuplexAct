#!/usr/bin/env python3
"""
evaluation/simulator/inference/cue_action_common.py
=======================================================
Action-generalized sibling of stop_common.py's `run_stop_sample_cli` /
`stop_cli_main`. Deliberately a SEPARATE, NEW file rather than an edit to
stop_common.py (which stays exactly as-is, still used unmodified by every
infer_stop_<model>.py's own smoke-test `main()` -- same "always additive,
never touch the validated path" rule used throughout this project, e.g.
trigger_controller.py / realtime_api_stop_common.py's own module
docstrings).

Why this needed a new file instead of just pointing stop_common.py's
existing driver at a different `--sample_dir`: three of the four planned
Level-2 actions (stop, hold_floor, respond_and_recover -- NOT
interrupt_user, see below) share the exact same *physical* sample layout
(lead_in.wav + cue.wav + metadata.json, live-triggered cue injection while
the model is already speaking) and can reuse every existing per-model
infer_stop_<model>.py client UNCHANGED (they only implement the wire
protocol, they have no idea what "action" they're being used for) --
but they differ in two real ways stop_common.py hardcoded:
  1. `metadata.json`'s own field names for the reference-context prompt
     text aren't perfectly uniform across actions (hold_floor's data was
     produced with `reference_before_text` / `reference_after_text` /
     `cue_text` instead of stop/respond_and_recover's `reference_original_
     response_text` / `reference_interruption_text` -- see
     METADATA_FIELD_ALIASES below). trigger_controller.py's
     _build_user_message() reads the stop/respond_and_recover names
     directly via `ref.get(...)`, so hold_floor's raw metadata needs
     normalizing before being handed to LiveTriggerController, or the
     judge LLM would silently get empty strings for reference content.
  2. The DOWNSTREAM scorer (evaluation/actions/<action>.py) reads
     different `EvalInstance.timestamps` keys per action (stop wants
     `t_user_int_start`; hold_floor wants `t_cue_start`/`t_cue_end`;
     respond_and_recover wants `t_interruption_start`/`t_interruption_
     end`) -- all three are trivially the SAME underlying cue-segment
     start/end our trigger controller already produces (`client.
     simulator_speech_segments`'s `source="cue"` entry), just exposed
     under different key names. `derive_action_timestamps()` below is
     the one new piece of actual logic.

Everything else (build_conversation_wav, build_conversation_script,
default_model_speech_transcripts, DynamicTokenSource, the whole per-model
client contract) is reused directly, unmodified, by importing from
stop_common.py -- those were already fully action-agnostic (plain PCM /
segment lists in, no "stop" assumptions baked in).

interrupt_user is DELIBERATELY NOT covered by this file. Its data layout
(user_speech.wav + branch_acknowledged.wav + branch_ignored.wav, no lead_in/
cue) reflects a fundamentally different simulator shape: for stop/hold_floor/
respond_and_recover, WE inject a cue while the MODEL is speaking and watch
how it reacts; for interrupt_user, the USER speaks continuously and we watch
whether/when the MODEL barges in on the USER, then need to dynamically
select which of two pre-recorded user-continuation branches to play next
depending on that outcome. That needs a new live-branching driver, not a
data-path swap on this one -- see MODEL_COMPATIBILITY.md's own note once
that design is scoped.
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
DIRECT_REPLAY_DATA = REPO_ROOT / "evaluation" / "direct_replay" / "data"

# ── Action registry ──────────────────────────────────────────────────────
# `data_subdir`: matches each action's own prepare_<action>_samples.py output
# layout (direct_replay/data/<action>/<l3_condition_dir>/<sample_id>/). Every
# sample dir under here has lead_in.wav + cue.wav + metadata.json, confirmed
# identical to stop's own layout (see MODEL_COMPATIBILITY.md's production-run
# note for the audit that established this).
ACTION_DATA_ROOTS: dict[str, Path] = {
    "stop": DIRECT_REPLAY_DATA / "stop",
    "hold_floor": DIRECT_REPLAY_DATA / "hold_floor",
    "respond_and_recover": DIRECT_REPLAY_DATA / "respond_and_recover",
}

# Maps this file's own normalized field names (the ones trigger_controller.
# py's _build_user_message() already reads via ref.get(...)) to the ACTUAL
# key present in a given action's metadata.json, only where it differs from
# stop's own naming. stop and respond_and_recover need no entry here (their
# metadata.json was produced with the same field names already -- confirmed
# by direct inspection of prepare_stop_samples.py's and
# prepare_respond_recover_samples.py's own output). hold_floor's data used
# different field names (reference_before_text / cue_text instead of
# reference_original_response_text / reference_interruption_text, and has
# no interruption_type field at all -- confirmed by direct inspection of a
# real hold_floor metadata.json).
METADATA_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "hold_floor": {
        "reference_original_response_text": "reference_before_text",
        "reference_interruption_text": "cue_text",
        # hold_floor has no interruption_type field; falls back to
        # l3_condition itself (still informative for the trigger LLM's
        # "why is this a valid injection point" grounding) rather than an
        # empty string.
        "interruption_type": "l3_condition",
    },
}


def normalize_reference_context(action: str, raw_metadata: dict) -> dict:
    """
    Returns a NEW dict with normalized-name keys added (raw_metadata's own
    keys are preserved too, so nothing already-consumed elsewhere is lost --
    e.g. hold_floor's `reference_after_text` / `bc_type_original` stay
    available even though nothing currently reads them). Only adds a
    normalized key if the actual source field has a non-empty value AND the
    normalized key isn't already present verbatim in raw_metadata (so
    stop/respond_and_recover, which already use the normalized names
    directly, are completely unaffected -- this is purely additive for
    actions with an alias entry).
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
    Every action's scorer (evaluation/actions/<action>.py) reads different
    EvalInstance.timestamps key names, but they're all trivially derived
    from the SAME underlying fact our trigger controller already produces:
    when the injected cue started and ended on the shared sample timeline
    (the `source="cue"` entry in `simulator_speech_segments` -- see e.g.
    infer_stop_gemini_live.py's own construction of that entry). This is
    the one piece of real per-action logic in this file; everything else
    is just plumbing reuse.
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
    Action-generalized twin of stop_common.run_stop_sample_cli -- see that
    function's own docstring for the full per-model client contract (same
    contract here, unchanged). Differences from the `stop`-only original:
      - `reference_context` is passed through normalize_reference_context()
        so LiveTriggerController sees the right field names regardless of
        which action's metadata.json shape it's reading (no-op for stop/
        respond_and_recover, real remap for hold_floor -- see
        METADATA_FIELD_ALIASES above).
      - `info["action"]` records the real action instead of a hardcoded
        "stop" string.
      - `info["instance"]["timestamps"]` is populated via
        derive_action_timestamps() so each action's own
        evaluation/actions/<action>.py scorer has the keys it expects
        (stop's own downstream consumers, if any exist, still get exactly
        the same shape as before -- this key didn't exist in the original
        stop_common.py output, it's purely additive).
    """
    import soundfile as sf
    from dotenv import load_dotenv
    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "direct_replay" / "inference"))
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
    """Standalone single-sample smoke-test entry point, action-generalized
    twin of stop_common.stop_cli_main."""
    import asyncio
    parser = build_stop_cli_parser(default_model)
    args = parser.parse_args()
    asyncio.run(run_cue_action_sample_cli(client_cls, model_slug, api_sr, action, args,
                                           client_kwargs, api_key_env, output_sr))
