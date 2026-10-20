#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/stop_common.py
=================================================
Shared helpers for ALL per-model `stop`-action simulator clients
(infer_stop_moshi.py, infer_stop_flm_audio.py, infer_stop_gpt_realtime.py,
...). Extracted out of infer_stop_moshi.py (the first implementation) once
a second client needed the exact same conversation-audio-mixing and
conversation-script-merging logic -- both functions are already fully
model-agnostic (take plain PCM bytes / segment lists as arguments, no
Moshi-specific assumptions), so duplicating them per-model would just be
copy-paste drift waiting to happen.

Every per-model infer_stop_<model>.py file is a SEPARATE module from its
direct_replay/inference/infer_<model>.py counterpart (never a subclass
override, never an edit to the original) -- see trigger_controller.py's own
module docstring for the full architecture rationale. This file lives
next to them for the same reason: it is simulator-only, never imported by
anything under direct_replay/.

POST_CUE_OBSERVATION_S here is the shared default (see infer_stop_moshi.py's
own comment for the full derivation: matches direct_replay's own
PADDING_TIME_S=15.0 convention for "how long to give the model to respond
after the trigger point"). Individual clients may override it if a specific
model's own pacing genuinely needs a different value, but should default to
importing this rather than re-deriving their own number.
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
    Token-domain sibling of realtime_common.DynamicPCMSource, for models
    whose native streaming API consumes discrete audio TOKENS rather than
    raw PCM bytes (currently: bayling-duplex, whose BayLingDuplex.
    stream_audio_tokens() takes a list[int] of pre-tokenized user-audio
    tokens per call -- see infer_stop_bayling.py's module docstring for the
    full architecture). Same semantics as DynamicPCMSource: pull_block()
    ALWAYS returns exactly `n` tokens immediately, backed by real queued
    content when available, else repeating `silence_tokens` as filler (the
    token-domain equivalent of "silence" -- computed ONCE by the caller via
    the model's own tokenizer on a real zeroed-PCM clip, since token values
    are model-specific and can't be synthesized generically the way
    digital silence (`b"\\x00" * n`) can for raw PCM).

    `silence_tokens` must be at least 1 token long; it is cycled
    (repeated/truncated) to fill however many filler tokens are needed on
    any given pull, so its own length does not need to evenly divide `n`.
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
        Token-domain twin of DynamicPCMSource.truncate_pending, for
        interrupt_user only (see interrupt_controller.py). Stop / hold_floor /
        respond_and_recover never call this. Returns the number of tokens
        dropped. Does NOT set `_finished`.
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
        # Cycle through _silence to fill exactly n_missing tokens, however
        # short/long _silence itself is.
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
    output.wav (SynchronizedRecorder's own output -- see its class docstring
    in common/realtime_common.py) is, BY DESIGN, the
    model's audio ONLY: the recorder is never given what we SEND, only what
    the model produces. So listening to output.wav alone never lets you
    hear "the conversation" -- it's one side of a phone call with the other
    party muted. This builds the actual two-sided recording: a 2-channel
    WAV with ch0 (left) = what the simulator sent (lead_in.wav then
    cue.wav, placed at their real positions on the shared timeline, silence
    everywhere else) and ch1 (right) = output.wav verbatim.

    Why placing lead_in/cue at raw sample offsets (t * api_sr) on channel 0
    lines up correctly with output.wav's own samples on channel 1, i.e. why
    this is NOT mixing two independently-clocked recordings: output.wav's
    sample position IS wall-clock-accurate from the SAME t=0 origin --
    SynchronizedRecorder.run()'s expected_count()/gap-filling logic (F-05,
    see realtime_common.py) continuously inserts silence so that
    self.count/self.out_sr always equals real elapsed time since
    recorder.stream_origin, and stream_origin is set (in
    RealtimeClientBase.start_streaming()) from the exact same time.monotonic()
    reading that each client's own now_fn()/t_user_int_start are also
    anchored to (both originate within the same `await start_streaming();
    stream_t0 = time.monotonic()` pair in _connect_and_stream -- two
    independent reads a few microseconds apart, but that skew is negligible
    next to the ~100ms+ scale the `stop` metric (YIELD_WINDOWS,
    BOUNDARY_TOLERANCE=50ms) cares about). So "channel 0 sample i" and
    "channel 1 sample i" really did happen at the same instant -- this is a
    true synchronized 2-track recording, not a best-effort approximation.

    Requires output.wav's own sample rate to equal `api_sr` exactly (every
    client here uses one fixed API_SR throughout, same as direct_replay's
    clients) -- raises ValueError otherwise rather than silently resampling,
    since a mismatch would indicate a real bug in the caller, not something
    to paper over.
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
    Merges the model's turns (model_speech_segments + their per-segment
    transcripts) and the simulator's turns (simulator_speech_segments,
    already {start, end, text, source}) into ONE chronologically-ordered
    script, both as a structured list (for programmatic use -- e.g. a future
    scorer wanting `context_required: ["transcript_model", ...]` per
    actions/stop.py's JUDGE_PROMPTS) and as a human-readable
    "Simulator: ...\\nModel: ...\\n..." string (for a reviewer to just read
    top-to-bottom instead of cross-referencing two separate arrays by hand).
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
    Default `model_speech_transcripts` computation: bucket a flat list of
    (t, token_text) tuples -- as captured live by a client's own receive
    loop, e.g. MoshiStopClient.text_tokens / FLMAudioStopClient.text_tokens
    -- into each finalized `model_speech_segment` window. No separator is
    inserted between tokens on purpose: every model surveyed so far whose
    native text stream is word/subword-tokenized (Moshi, FLM-Audio, BayLing,
    MiniCPM-o, Raon) emits tokens that already carry their own leading space
    at real word boundaries (confirmed from raw events.jsonl on Moshi/FLM-
    Audio: " day", ",", " how" etc.) -- joining with "" reproduces normal
    spacing; joining with " " double-spaces every token and glues
    punctuation awkwardly ("day ,  how"). A client whose native tokens do
    NOT self-space would need its own override (see below) rather than
    changing this default.

    Models with no word-level text signal at all (e.g. lychee-fd, which
    only has native speaking-interval events, not tokens) should NOT call
    this -- leave `model_speech_transcripts` as a list of "" (or override
    `compute_model_speech_transcripts()` on the client class) rather than
    silently returning garbage.
    """
    out = []
    for on, off in segments:
        words = [w for (t, w) in text_tokens if on <= t <= off]
        out.append("".join(words).strip())
    return out


def build_stop_cli_parser(default_model: str) -> argparse.ArgumentParser:
    """
    Shared argparse setup for every infer_stop_<model>.py's standalone
    single-sample CLI (smoke-testing entry point) -- identical flags across
    every model, only `default_model` (the HF repo id / model_name) differs.
    """
    parser = argparse.ArgumentParser(
        description="Simulator stop-action smoke-test CLI (single sample).",
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
    Generic single-sample driver shared by every infer_stop_<model>.py's
    `_main_async` -- extracted out of infer_stop_moshi.py (the reference
    implementation) once a second client needed the exact same load-sample /
    run / bucket-transcripts / build-conversation / write-info.json sequence.
    Only the CLIENT CLASS itself (whatever _connect_and_stream does for that
    model's own protocol) differs per model; everything in THIS function is
    model-agnostic plumbing.

    `client_cls` must implement the same contract as MoshiStopClient:
    - constructor: (api_key, model_name, system_prompt, verbose, cue_pcm,
      reference_context, llm_client, **client_kwargs)
    - after run_sample() resolves: `.trigger_result` (dict, from
      LiveTriggerController.run()'s return value) and
      `.simulator_speech_segments` (list of {start,end,text,source} dicts)
      are both populated (mirrors MoshiStopClient._connect_and_stream's own
      `finally` block).
    - EITHER exposes `.text_tokens: list[tuple[float, str]]` (used by
      `default_model_speech_transcripts()` below) OR overrides
      `compute_model_speech_transcripts(segments) -> list[str]` itself for
      models with a different/no native text-capture granularity (e.g.
      lychee-fd's speaking-interval-only signal -- see MODEL_COMPATIBILITY.md).

    Loads lead_in.wav/cue.wav/metadata.json from `args.sample_dir` (the same
    `evaluation/data/stop/direct_interruption/<sample_id>/`
    layout every model shares -- these are direct_replay's own already-
    validated, WER/speaker-checked data files, read-only from here), builds
    the OpenAI client for the trigger LLM from OPENAI_API_KEY, runs the
    sample, assembles the final result/info dict (bucketed transcripts +
    merged conversation script + two-channel conversation wav), prints it,
    and writes `args.info_path` if given. Returns the `info` dict.

    `output_sr`: only needed when a model's INPUT rate differs from its
    OUTPUT rate (currently just bayling-duplex: 16kHz in / 22050Hz out --
    every other client so far uses one rate both ways). `api_sr` is always
    what `cue_pcm`/`lead_in_pcm` get resampled to for FEEDING the model
    (matching client_cls.INPUT_SR); when `output_sr` differs from `api_sr`,
    a SEPARATE resample of lead_in/cue at `output_sr` is used only for
    building the two-channel conversation wav (which must align with
    output.wav's own sample rate, i.e. client_cls.OUTPUT_SR) -- otherwise
    the sim-audio channel would play back at the wrong pitch/speed and be
    misaligned in time against the model's own channel. Defaults to
    `api_sr` (no extra resampling) when not given.
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
    # Only re-resampled when conv_sr != api_sr (bayling-duplex); every other
    # model reuses the exact same bytes with zero extra work.
    lead_in_pcm_conv = lead_in_pcm if conv_sr == api_sr else load_and_resample(lead_in_wav, conv_sr).tobytes()
    cue_pcm_conv = cue_pcm if conv_sr == api_sr else load_and_resample(cue_wav, conv_sr).tobytes()

    llm_client = None
    from trigger_controller import make_trigger_llm_client  # noqa: E402
    llm_client = make_trigger_llm_client()
    if llm_client is None:
        print("[WARN] OPENAI_API_KEY not set -- trigger controller will error on every "
              "poll and fall back to a forced trigger (onset/search timeout).", file=sys.stderr)

    # `api_key_env` is None for locally-hosted open-source clients
    # (LOCAL_NO_KEY convention, same as model_registry.py's own
    # MODEL_SPECS entries) -- only closed-source API clients (GPT-Realtime,
    # Grok Voice, Gemini Live) need a real key here.
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

    # Optional hook for models whose NATIVE speaking-interval signal is
    # known-unreliable (currently: lychee-fd -- see
    # MODEL_COMPATIBILITY.md's "SEVERE finding" section for the full
    # root-cause writeup on why its round_trace["events"] timestamps can
    # diverge from real audio placement once round dispatch falls behind
    # real-time). `args.output_wav` is already fully written and closed by
    # the time `run_sample()` above returns, so this can safely re-derive
    # segments directly from the finished audio file.
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
        # Full reference conversation (T1..T4) is already in reference_context
        # verbatim (transcript_user=T1, reference_original_response_text=T2,
        # reference_interruption_text=T3, reference_recovery_text=T4) --
        # see evaluation/data/prepare_stop_samples.py +
        # backfill_stop_recovery_text.py.
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
    Full `if __name__ == "__main__":`-ready entry point -- each
    infer_stop_<model>.py's own `main()` should just be:

        def main():
            stop_cli_main(FooStopClient, FOO_SLUG, API_SR, DEFAULT_MODEL)

        # ...or, for a closed-source client needing a real API key:
        def main():
            stop_cli_main(FooStopClient, FOO_SLUG, API_SR, DEFAULT_MODEL,
                          api_key_env="FOO_API_KEY")

        # ...or, when input/output rates differ (see run_stop_sample_cli's
        # own docstring -- currently only bayling-duplex):
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
