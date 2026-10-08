#!/usr/bin/env python3
"""
direct_replay/inference/infer_mini_omni2.py
==========================================
gpt-omni/mini-omni2 client. Category C (batch-replay, no real-time pacing
at all) -- but for a DIFFERENT reason than Voila-Autonomous's. Voila is a
genuine dual-stream architecture whose autoregressive step index IS a real
conversation-time clock (25Hz on both the user and assistant channel), just
never driven with live wall-clock pacing by its own authors. Mini-Omni2 has
no such structure at all: its only public entry point
(`inference.OmniInference.run_AT_batch_stream`) first runs a SINGLE Whisper
encoder pass over the *entire* input clip (a one-shot prefill, not a
per-frame streaming encode) and only then autoregressively generates a
response -- there is no interleaved "consume one more real-time tick of
user audio" step anywhere in the generation loop, so unlike Voila there is
no native per-step signal we could map back to conversation time even in
principle. See `local_common.BatchReplayClientBase` for why this whole
category exists and its segmentation caveat.

Architecture (arXiv:2410.11190, gpt-omni/mini-omni2): a single Qwen2-0.5B
backbone (via litgpt's own `GPT`/`Config` classes, NOT HF `transformers`)
consumes concatenated Whisper-small audio features + text embeddings, and
produces 8 PARALLEL output channels every step -- 7 SNAC-codec (24kHz,
`hubertsiuzdak/snac_24khz`) audio codebooks plus 1 text channel -- via a
text-guided delayed-parallel decoding scheme (each audio channel's tokens
are offset/layer-shifted so all 8 channels can share one vocabulary space
and be predicted from the same next-token head). `run_AT_batch_stream`
yields `(audio_bytes, text_str)` tuples incrementally as generation
proceeds (every `stream_stride=4` steps, decoding whatever SNAC codes have
accumulated since the last yield) -- so the OUTPUT side genuinely is
progressive/incremental, just with no relationship at all between step
count and real elapsed conversation time (steps advance at whatever speed
the GPU decodes them, independent of the user's own speech timing).

Because there's no live pacing signal to drive at all (not even Voila's
frame-locked one), we implement this the simplest possible way allowed by
`BatchReplayClientBase`: write the resampled input clip to a temp WAV,
drain the ENTIRE `run_AT_batch_stream` generator in one background thread
(concatenating every yielded audio-byte chunk in generation order -- which
is also strictly chronological within the response, since SNAC decoding is
itself causal/autoregressive), and hand the base class the complete
concatenated waveform. `model_speech_segments` therefore falls back to the
base class's default RMS-energy heuristic (`_native_segments()` returns
None) -- there's no better signal available here, unlike Voila's
`streamer=`-hook reconstruction of an exact internal boolean.

Turn-end gating (VAD, matching the model's OWN interactive demo): the
repo's only live-interaction reference, `webui/omni_streamlit.py`'s
`recording()`, never calls the model over a whole fixed-length clip at
all -- it accumulates mic audio in 0.5s strides (`VAD_STRIDE`), runs the
repo's bundled `utils/vad.py` Silero VAD on each new stride, starts
collecting frames once a stride has >0.2s of detected speech
(prepending the previous, pre-onset stride so the very start of speech
isn't clipped), and stops -- handing everything collected so far to
`run_AT_batch_stream` -- the moment a stride drops below 0.1s of
detected speech. `BatchReplayClientBase.run_sample()` only requires
`_generate_batch()`'s returned waveform to be frame-aligned with
`input_pcm`'s OWN timeline (silence anywhere the model hadn't been
called yet is fine, see its own docstring) -- it does NOT require
calling the model over the whole clip, so `_generate_batch()` below
replicates this exact stride/threshold state machine over the offline
`input_pcm` array (same thresholds, same pre-onset-stride prepend, same
INPUT_SR=16000 the VAD model itself expects -- no resampling needed,
unlike the demo's own 24kHz mic rate), calls `run_AT_batch_stream` only
on the audio collected up to the detected turn-end, and places the
result starting at that detected offset rather than at t=0. If VAD never
detects a clear speech onset, or detects onset but never a subsequent
end-of-turn before the clip (including its own padding) runs out, the
model is never called at all -- an honest reflection of this model's own
protocol (no live "forcibly cut off and generate now" case exists in
`recording()` either), not a synthetic override.

Whisper's fixed 30-second context window (openai-whisper's own
`whisper.pad_or_trim`, hardcoded to `N_SAMPLES = 30 * 16000`) is an
INHERENT limitation of this model's audio front-end, not something we work
around: `inference.load_audio()` calls `pad_or_trim()` unconditionally, so
any input clip longer than 30s is silently truncated to its first 30s
before the model ever sees the rest. Many direct_replay samples (e.g. Wait,
Ignore) run past 30s once trailing padding is included -- for those, this
model literally never "hears" the back half of the clip. We report this
plainly in the model doc rather than hand-chunking the Whisper encode
ourselves (that would require retraining-adjacent surgery on a frozen,
non-streaming pretrained encoder, not a client-side pacing choice like
every other adapter in this project makes).

No system-prompt support: `run_AT_batch_stream`'s prompt construction
(`get_input_ids_whisper_ATBatch`) has no slot for a system/instruction
string at all -- it hardcodes the same `_input_a`/`_answer_a` special-token
scaffolding around the raw Whisper audio features for every call. Same
situation as Voila-Autonomous and BayLing-Duplex; `self.system_prompt` is
accepted (for interface uniformity with every other client) but unused.

Practical setup notes:
  - `ffmpeg` must be on PATH (openai-whisper's `load_audio()` shells out to
    it) -- NOT bundled in this venv; on this cluster it's a module
    (`module load GCCcore/13.3.0 FFmpeg/7.1.1`), added to
    slurm_infer_opensource.sh for this model specifically.
  - Checkpoint: `gpt-omni/mini-omni2`'s own HF repo is a flat directory
    (`lit_model.pth`, `model_config.yaml`, `small.pt` Whisper weights,
    tokenizer files) consumed directly by `inference.load_model(ckpt_dir,
    device)` -- NOT the standard HF hub cache layout, so we
    `snapshot_download(..., local_dir=...)` into a plain local folder
    (skipping the bundled CLIP vision checkpoint and demo video/figures,
    unused by the audio-only path we drive here). Because `small.pt`
    already exists inside that folder, `inference.load_model()`'s own
    `whisper.load_model(whisper_model_path)` call loads it directly from
    disk and never touches Whisper's separate `~/.cache/whisper` download
    path at all -- so no extra env-var override is needed for Whisper
    specifically, `env_for_model()`'s existing HF_HOME covers the SNAC
    codec download (`SNAC.from_pretrained(...)`), which is the only other
    thing this model downloads from the Hub.
  - `litgpt==0.4.3` (NOT the HF `transformers` Qwen2 implementation) is a
    hard version pin from the repo's own `requirements.txt` -- newer litgpt
    releases have since changed `Config`/`GPT` internals in
    backwards-incompatible ways.

Standalone usage (single sample, for debugging):
    python infer_mini_omni2.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from local_common import BatchReplayClientBase, opensource_root_for

MINI_OMNI2_SLUG = "mini-omni2"
DEFAULT_MODEL = "gpt-omni/mini-omni2"

INPUT_SR = 16_000    # Whisper's own fixed input rate (inference.py's load_audio()) -- also utils/vad.py's Silero model's own native rate, so no resampling is needed for VAD either
OUTPUT_SR = 24_000   # SNAC codec output rate (snac_24khz)
STREAM_STRIDE = 4          # matches server.py's own request default
MAX_RETURNED_TOKENS = 2048  # matches server.py's own request default ("max_tokens": 2048)

# Turn-end VAD gating -- exact constants from webui/omni_streamlit.py's
# recording() loop (see module docstring). VAD_STRIDE is in seconds;
# START_SPEECH_S/END_SILENCE_S are the same dur_vad>0.2 / dur_vad<0.1
# thresholds against the speech-duration-within-this-stride measurement.
VAD_STRIDE_S = 0.5
VAD_START_SPEECH_S = 0.2
VAD_END_SILENCE_S = 0.1

REPO_DIR = opensource_root_for(MINI_OMNI2_SLUG) / "repo"
CKPT_DIR = opensource_root_for(MINI_OMNI2_SLUG) / "checkpoints" / "mini-omni2"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


class MiniOmni2Client(BatchReplayClientBase):
    MODEL_SLUG = MINI_OMNI2_SLUG
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = INPUT_SR

    _client = None
    _loaded_ckpt_dir: Optional[str] = None

    @classmethod
    def _load(cls, ckpt_dir: str):
        if cls._client is not None and cls._loaded_ckpt_dir == ckpt_dir:
            return cls._client

        import torch
        from inference import OmniInference

        client = OmniInference(ckpt_dir=ckpt_dir, device="cuda:0" if torch.cuda.is_available() else "cpu")
        # OmniInference.warm_up()'s own default argument is the
        # relative path './data/samples/output1.wav' (inference.py),
        # which only resolves correctly if the process's cwd happens
        # to be the repo root -- ours is the shared inference/
        # directory instead, since every model's client is invoked
        # from the same run_inference.py entrypoint. The file itself
        # is a real bundled repo asset, not missing; we just pass its
        # absolute path instead of relying on cwd. Not a vendored-code
        # change, just how we call an existing optional argument.
        client.warm_up(sample=str(REPO_DIR / "data" / "samples" / "output1.wav"))

        cls._client = client
        cls._loaded_ckpt_dir = ckpt_dir
        return client

    @staticmethod
    def _detect_turn_end(pcm_f32: np.ndarray) -> Optional[tuple]:
        """
        Replicates omni_streamlit.py's recording() state machine over an
        offline array instead of a live mic stream: ONE forward pass,
        one fresh (never re-examined) stride's VAD result per iteration
        -- exactly like the reference's `while recording: read new
        chunk; run_vad(new chunk only); ...`. Returns (feed_start_idx,
        feed_end_idx) into pcm_f32 (@ INPUT_SR) -- the exact span
        `recording()` would have handed to `speaking()` -- or None if a
        turn never starts (no stride ever exceeds VAD_START_SPEECH_S of
        detected speech) or never ends (speech detected but the array
        runs out before any stride drops below VAD_END_SILENCE_S). In
        either None case the caller must not call the model at all,
        matching recording()'s own behavior (it just keeps looping
        forever in both cases -- speaking() is never reached).
        """
        from utils.vad import get_speech_timestamps, VadOptions

        stride_samples = int(round(VAD_STRIDE_S * INPUT_SR))
        vad_options = VadOptions()
        n_strides = len(pcm_f32) // stride_samples

        start_talking = False
        feed_start_idx = 0
        for i in range(n_strides):
            s, e = i * stride_samples, (i + 1) * stride_samples
            speech_chunks = get_speech_timestamps(pcm_f32[s:e], vad_options)
            dur_vad = sum(c["end"] - c["start"] for c in speech_chunks) / INPUT_SR

            if dur_vad > VAD_START_SPEECH_S and not start_talking:
                start_talking = True
                # last_temp_audio prepend: include the one stride just
                # before onset was detected, same as the reference.
                feed_start_idx = max(0, s - stride_samples)
                continue   # this stride's own dur_vad > threshold, can't also be < VAD_END_SILENCE_S

            if dur_vad < VAD_END_SILENCE_S and start_talking:
                return feed_start_idx, e

        return None   # never started, or started but never ended within this clip

    async def _generate_batch(self, input_pcm: np.ndarray, input_duration_s: float) -> np.ndarray:
        pcm_f32 = input_pcm.astype(np.float32) / 32768.0

        turn_span = await asyncio.to_thread(self._detect_turn_end, pcm_f32)
        target_samples = int(round(input_duration_s * OUTPUT_SR))
        if turn_span is None:
            # No detectable turn-end within this clip (never started
            # talking, or started but never paused long enough) -- the
            # live demo would simply still be "recording" and never call
            # the model either. Reported as genuine silence, not an error.
            return np.zeros(target_samples, dtype=np.float32)

        feed_start_idx, feed_end_idx = turn_span
        client = await asyncio.to_thread(self._load, str(CKPT_DIR))
        trigger_s = feed_end_idx / INPUT_SR
        clip_f32 = pcm_f32[feed_start_idx:feed_end_idx]

        def _run() -> bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, clip_f32, INPUT_SR, subtype="PCM_16")
                chunks = bytearray()
                for audio_bytes, _text in client.run_AT_batch_stream(
                    tmp.name,
                    stream_stride=STREAM_STRIDE,
                    max_returned_tokens=MAX_RETURNED_TOKENS,
                ):
                    chunks.extend(audio_bytes)
                return bytes(chunks)

        pcm_bytes = await asyncio.to_thread(_run)
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        response_f32 = pcm_i16.astype(np.float32) / 32768.0

        # Place the response starting at the detected trigger offset,
        # not at t=0 -- see module docstring. Base class pads/never-
        # truncates against input_duration_s afterward, so we only need
        # to prepend the pre-trigger silence here.
        lead_silence = np.zeros(int(round(trigger_s * OUTPUT_SR)), dtype=np.float32)
        return np.concatenate([lead_silence, response_f32])


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = MiniOmni2Client(
        api_key="",
        model_name=args.model,
        system_prompt="",
        verbose=args.verbose,
    )
    result = await client.run_sample(
        input_wav=args.input_wav,
        output_wav=args.output_wav,
        events_path=args.events_path,
        input_duration=input_duration,
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_wav", type=Path, required=True)
    parser.add_argument("--output_wav", type=Path, required=True)
    parser.add_argument("--events_path", type=Path, required=True)
    parser.add_argument("--input_duration", type=float, default=None,
                         help="Defaults to the actual duration of input_wav")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
