#!/usr/bin/env python3
"""
gpt-omni/mini-omni2 client (batch replay, no real-time pacing).

Mini-Omni2 (arXiv:2410.11190) runs one Whisper-small encoder pass over the
whole input clip and then autoregressively generates 7 SNAC (24 kHz) audio
codebooks plus a text channel from a Qwen2-0.5B backbone (litgpt). There is no
per-step user-audio input, so generation steps carry no conversation-time
meaning. See `local_common.BatchReplayClientBase`; `model_speech_segments`
uses the base class's RMS-energy heuristic.

Turn-end gating follows the model's own demo (`webui/omni_streamlit.py`
`recording()`): Silero VAD over 0.5 s strides, a turn starts once a stride has
>0.2 s of speech (the preceding stride is prepended) and ends when a stride has
<0.1 s. Only the collected span is fed to `run_AT_batch_stream`, and the
response is placed at the detected turn-end offset. If no turn starts or ends
within the clip, the model is not called and the output is silence.

Limitations:
  - Whisper's `pad_or_trim` truncates input to 30 s, so the model never hears
    audio past that point.
  - The prompt format has no system-prompt slot; `self.system_prompt` is unused.

Setup:
  - `ffmpeg` must be on PATH (used by openai-whisper's `load_audio()`).
  - The checkpoint is the flat HF repo directory (`lit_model.pth`,
    `model_config.yaml`, `small.pt`, tokenizer) downloaded with
    `snapshot_download(..., local_dir=...)`; Whisper weights load from
    `small.pt` in that folder.
  - Requires `litgpt==0.4.3`; newer releases changed `Config`/`GPT` internals.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from local_common import BatchReplayClientBase, opensource_root_for

MINI_OMNI2_SLUG = "mini-omni2"
DEFAULT_MODEL = "gpt-omni/mini-omni2"

INPUT_SR = 16_000    # Whisper input rate, also the Silero VAD rate
OUTPUT_SR = 24_000   # SNAC codec output rate (snac_24khz)
STREAM_STRIDE = 4          # server.py request default
MAX_RETURNED_TOKENS = 2048  # server.py request default

# Turn-end VAD constants from webui/omni_streamlit.py recording(): stride
# length, and the per-stride speech duration that starts / ends a turn.
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
        # warm_up()'s default sample path is relative to the repo root,
        # so pass it as an absolute path.
        client.warm_up(sample=str(REPO_DIR / "data" / "samples" / "output1.wav"))

        cls._client = client
        cls._loaded_ckpt_dir = ckpt_dir
        return client

    @staticmethod
    def _detect_turn_end(pcm_f32: np.ndarray) -> Optional[tuple]:
        """
        Run omni_streamlit.py's recording() VAD state machine over an
        offline array, one stride at a time. Returns (feed_start_idx,
        feed_end_idx) into pcm_f32 (at INPUT_SR), the span recording()
        would hand to speaking(), or None if a turn never starts or never
        ends within the array. On None the model should not be called.
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
                # Include the stride just before onset, as the demo does.
                feed_start_idx = max(0, s - stride_samples)
                continue

            if dur_vad < VAD_END_SILENCE_S and start_talking:
                return feed_start_idx, e

        return None   # never started, or started but never ended within this clip

    async def _generate_batch(self, input_pcm: np.ndarray, input_duration_s: float) -> np.ndarray:
        pcm_f32 = input_pcm.astype(np.float32) / 32768.0

        turn_span = await asyncio.to_thread(self._detect_turn_end, pcm_f32)
        target_samples = int(round(input_duration_s * OUTPUT_SR))
        if turn_span is None:
            # No turn-end within the clip; the demo would never call the
            # model either, so the output is silence rather than an error.
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

        # Start the response at the detected turn-end offset. The base
        # class pads against input_duration_s, so only the leading
        # silence is added here.
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
