#!/usr/bin/env python3
"""
Shared, model-agnostic helpers for the per-model `stop`-action simulator
clients (infer_stop_<model>.py): two-channel conversation audio, merged
conversation script, transcript bucketing, and the single-sample CLI driver.

POST_CUE_OBSERVATION_S is the default time given to the model to respond
after the trigger point, matching direct_replay's PADDING_TIME_S=15.0.
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

POST_CUE_OBSERVATION_S = 15.0


class DynamicTokenSource:
    """
    Token-domain counterpart of realtime_common.DynamicPCMSource, for models
    whose streaming API consumes discrete audio tokens rather than PCM
    (e.g. bayling-duplex). pull_block(n) always returns exactly `n` tokens:
    queued content when available, else `silence_tokens` as filler.

    `silence_tokens` should be produced by the model's own tokenizer on a
    zeroed-PCM clip, since token values are model-specific. It must be
    non-empty and is cycled to fill any number of filler tokens.
    """

    def __init__(self, silence_tokens: list, initial_tokens: Optional[list] = None):
        if not silence_tokens:
            raise ValueError("silence_tokens must be non-empty")
        self._silence = list(silence_tokens)
        self._buffer: list = list(initial_tokens or [])
        self._finished = False

    def append(self, tokens: list) -> None:
        self._buffer.extend(tokens)

    def truncate_pending(self) -> int:
        """
        Token-domain counterpart of DynamicPCMSource.truncate_pending, used by
        interrupt_user (see interrupt_controller.py). Returns the number of
        tokens dropped. Does not set `_finished`.
        """
        n = len(self._buffer)
        self._buffer.clear()
        return n

    def finish(self) -> None:
        self._finished = True

    @property
    def is_done(self) -> bool:
        return self._finished and len(self._buffer) == 0

    def pull_block(self, n: int) -> list:
        if len(self._buffer) >= n:
            chunk = self._buffer[:n]
            del self._buffer[:n]
            return chunk
        chunk = self._buffer[:]
        self._buffer.clear()
        n_missing = n - len(chunk)
        filler = []
        i = 0
        while len(filler) < n_missing:
            filler.append(self._silence[i % len(self._silence)])
            i += 1
        return chunk + filler


def build_conversation_wav(output_wav: Path, conversation_wav: Path, api_sr: int,
                            lead_in_pcm: bytes, cue_pcm: bytes,
                            t_user_int_start: Optional[float]) -> None:
    """
    Build a two-channel conversation WAV from output.wav, which contains only
    the model's audio. ch0 (left) = what the simulator sent (lead_in.wav, then
    cue.wav at `t_user_int_start`, silence elsewhere); ch1 (right) =
    output.wav verbatim.

    Placing simulator audio at raw sample offsets (t * api_sr) aligns with
    output.wav because SynchronizedRecorder gap-fills with silence so its
    sample position tracks wall-clock time from the same origin that
    t_user_int_start is measured from.

    Raises ValueError if output.wav's sample rate is not `api_sr`.
    """
    import soundfile as sf
    model_pcm, sr = sf.read(str(output_wav), dtype="int16")
    if sr != api_sr:
        raise ValueError(f"output.wav sample rate {sr} != expected {api_sr}")
    n_model = len(model_pcm)

    sim = np.zeros(n_model, dtype=np.int16)
    lead_in_arr = np.frombuffer(lead_in_pcm, dtype=np.int16)
    n_lead = min(len(lead_in_arr), n_model)
    sim[:n_lead] = lead_in_arr[:n_lead]

    if t_user_int_start is not None:
        cue_arr = np.frombuffer(cue_pcm, dtype=np.int16)
        start_i = int(round(t_user_int_start * api_sr))
        if start_i < n_model:
            end_i = min(start_i + len(cue_arr), n_model)
            sim[start_i:end_i] = cue_arr[: end_i - start_i]

    stereo = np.stack([sim, model_pcm], axis=1)  # (n_model, 2) int16, ch0=sim ch1=model
    conversation_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(conversation_wav), stereo, api_sr, subtype="PCM_16")


def build_conversation_script(model_speech_segments: list, model_speech_transcripts: list,
                               simulator_speech_segments: list) -> tuple[list[dict], str]:
    """
    Merge the model's turns (model_speech_segments + per-segment transcripts)
    and the simulator's turns ({start, end, text, source} dicts) into one
    chronologically ordered script. Returns both a structured list of turns
    and a human-readable "[start-end s] Speaker: text" string.
    """
    turns = []
    for (on, off), text in zip(model_speech_segments, model_speech_transcripts):
        turns.append({"speaker": "model", "start": on, "end": off, "text": text})
    for seg in simulator_speech_segments:
        turns.append({
            "speaker": "simulator", "start": seg["start"], "end": seg["end"],
            "text": seg["text"], "source": seg.get("source"),
        })
    turns.sort(key=lambda t: t["start"])

    lines = []
    for t in turns:
        label = "Simulator" if t["speaker"] == "simulator" else "Model"
        tag = f" ({t['source']})" if t.get("source") else ""
        lines.append(f"[{t['start']:>6.2f}-{t['end']:>6.2f}s] {label}{tag}: {t['text']}")
    return turns, "\n".join(lines)


def default_model_speech_transcripts(text_tokens: list, segments: list) -> list[str]:
    """
    Default `model_speech_transcripts`: bucket a flat list of
    (t, token_text) tuples (e.g. a client's `.text_tokens`) into each
    `model_speech_segment` window. Tokens are joined with "" because the
    supported models' subword tokens already carry their own leading space
    at word boundaries (" day", ",", " how").

    Clients whose tokens do not self-space, or that have no token-level text
    (e.g. lychee-fd), should override `compute_model_speech_transcripts()`.
    """
    out = []
    for on, off in segments:
        words = [w for (t, w) in text_tokens if on <= t <= off]
        out.append("".join(words).strip())
    return out


def build_stop_cli_parser(default_model: str) -> argparse.ArgumentParser:
    """
    Shared argparse setup for every infer_stop_<model>.py single-sample CLI;
    only `default_model` (the HF repo id / model_name) differs per model.
    """
    parser = argparse.ArgumentParser(
        description="Simulator stop-action CLI (single sample).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sample_dir", type=Path, required=True,
                         help="e.g. evaluation/data/stop/direct_interruption/Candor_stop_0038")
    parser.add_argument("--output_wav", type=Path, required=True)
    parser.add_argument("--events_path", type=Path, required=True)
    parser.add_argument("--info_path", type=Path, default=None)
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--verbose", action="store_true")
    return parser


async def run_stop_sample_cli(
    client_cls: Type,
    model_slug: str,
    api_sr: int,
    args: argparse.Namespace,
    client_kwargs: Optional[dict] = None,
    api_key_env: Optional[str] = None,
    output_sr: Optional[int] = None,
) -> dict:
    """
    Generic single-sample driver shared by every infer_stop_<model>.py.

    `client_cls` must:
    - accept (api_key, model_name, system_prompt, verbose, cue_pcm,
      reference_context, llm_client, **client_kwargs);
    - populate `.trigger_result` (dict from LiveTriggerController.run()) and
      `.simulator_speech_segments` ({start,end,text,source} dicts) once
      run_sample() returns;
    - either expose `.text_tokens: list[tuple[float, str]]` or override
      `compute_model_speech_transcripts(segments) -> list[str]`.

    Loads lead_in.wav/cue.wav/metadata.json from `args.sample_dir`
    (evaluation/data/stop/direct_interruption/<sample_id>/), runs the sample
    with the trigger LLM, builds the transcripts, conversation script and
    two-channel conversation wav, prints the result, writes `args.info_path`
    if given, and returns the `info` dict.

    `api_sr` is the model's input rate. Pass `output_sr` when the output rate
    differs (e.g. bayling-duplex: 16kHz in / 22050Hz out); the conversation
    wav is then built from lead_in/cue resampled to `output_sr` so it aligns
    with output.wav.
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

    reference_context = json.loads(meta_path.read_text())
    reference_context["action"] = "stop"
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

    # `api_key_env` is None for locally hosted models; only closed-source API
    # clients (GPT-Realtime, Grok Voice, Gemini Live) need a key.
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

    # Optional hook for models whose native speaking-interval timestamps can
    # drift from the real audio placement (e.g. lychee-fd); re-derives
    # segments from the finished output.wav.
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

    print(json.dumps({k: v for k, v in result.items() if k != "conversation_text"}, indent=2))
    print("\n--- conversation ---")
    print(conversation_text)

    info = {
        "sample_id":    sample_dir.name,
        "action":       "stop",
        "l3_condition": reference_context.get("l3_condition"),
        "model":        model_slug,
        # reference_context carries the reference conversation: transcript_user=T1,
        # reference_original_response_text=T2, reference_interruption_text=T3,
        # reference_recovery_text=T4.
        "instance":     {**reference_context, **client.trigger_result},
        "inference":    result,
    }
    if args.info_path:
        args.info_path.write_text(json.dumps(info, indent=2))
    return info


def stop_cli_main(client_cls: Type, model_slug: str, api_sr: int, default_model: str,
                   client_kwargs: Optional[dict] = None, api_key_env: Optional[str] = None,
                   output_sr: Optional[int] = None) -> None:
    """
    Entry point for each infer_stop_<model>.py, e.g.:

        def main():
            stop_cli_main(FooStopClient, FOO_SLUG, API_SR, DEFAULT_MODEL)

        # ...or, for a closed-source client needing a real API key:
        def main():
            stop_cli_main(FooStopClient, FOO_SLUG, API_SR, DEFAULT_MODEL,
                          api_key_env="FOO_API_KEY")

        # ...or, when input/output rates differ:
        def main():
            stop_cli_main(FooStopClient, FOO_SLUG, INPUT_SR, DEFAULT_MODEL,
                          output_sr=OUTPUT_SR)

        if __name__ == "__main__":
            main()
    """
    parser = build_stop_cli_parser(default_model)
    args = parser.parse_args()
    asyncio.run(run_stop_sample_cli(client_cls, model_slug, api_sr, args, client_kwargs,
                                     api_key_env, output_sr))
