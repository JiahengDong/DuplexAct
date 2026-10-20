#!/usr/bin/env python3
"""
BayLing-Models/BayLing-Duplex client (in-process incremental streaming).
`BayLingDuplex.stream_audio_tokens` is a block-by-block generator, driven
here with wall-clock pacing.

Architecture: a GLM-4-Voice LLM consumes user-speech tokens (12.5Hz) and
assistant text/speech tokens as one interleaved sequence, blocked
x:y:z = 10:5:10 (one block = 10 user tokens = 0.8s of audio, matched to 5
text-token + 10-audio-token sub-steps). Turn-taking is next-token prediction
over [SILENCE], <|assistant|>, [PAD], [EPAD]; `assistant_started` /
`assistant_finished` events are the model's speaking signal.

Each loop iteration pulls `2 * y_ratio` events (one block). Audio tokens are
decoded per sub-step with `stream_tokens_to_audio` (flow-matching + HiFT).

The generator only runs over the user tokens it is given, so
INPUT_TRAILING_PAD_S of silence is appended to let the model finish its reply.

The vocoder is slower than real time; CONTIGUOUS_WHILE_SPEAKING writes one
turn's chunks back-to-back so decode delay does not appear as silence inside
the turn. Response onset is still the `<|assistant|>` time on the recorder
clock.

max_epad_count=1 (library default): generation stops after one assistant
utterance.

Standalone usage (single sample, for debugging):
    python infer_bayling.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger
from local_common import opensource_root_for

BAYLING_SLUG = "bayling-duplex"
DEFAULT_MODEL = "BayLing-Models/BayLing-Duplex"
TOKENIZER_MODEL = "zai-org/glm-4-voice-tokenizer"
DECODER_MODEL = "zai-org/glm-4-voice-decoder"
INTERLEAVE_RATIO = "10:5:10"

INPUT_SR = 16_000    # BayLingDuplex.sample_rate
OUTPUT_SR = 22_050   # BayLingDuplex.output_sample_rate
INPUT_TRAILING_PAD_S = 8.0   # see module docstring

REPO_DIR = opensource_root_for(BAYLING_SLUG) / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _pull_n(gen, n: int) -> list:
    out = []
    for _ in range(n):
        try:
            out.append(next(gen))
        except StopIteration:
            break
    return out


def _f32_to_i16_bytes(wav) -> bytes:
    arr = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class BayLingDuplexClient(RealtimeClientBase):
    MODEL_SLUG = BAYLING_SLUG
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = INPUT_SR
    CONTIGUOUS_WHILE_SPEAKING = True

    _bd = None   # BayLingDuplex singleton (loaded once per process)
    _model_name_loaded: Optional[str] = None

    @classmethod
    def _load(cls, model_name: str):
        if cls._bd is not None and cls._model_name_loaded == model_name:
            return cls._bd

        from huggingface_hub import snapshot_download
        from bayling_duplex.duplex import BayLingDuplex

        # AudioDecoder opens <decoder_path>/config.yaml, flow.pt, hift.pt
        # directly, so decoder_path must be a local directory, not an HF repo id.
        decoder_local_dir = snapshot_download(DECODER_MODEL)

        cls._bd = BayLingDuplex(
            model_path=model_name,
            speech_tokenizer_path=TOKENIZER_MODEL,
            decoder_path=decoder_local_dir,
            interleave_ratio=INTERLEAVE_RATIO,
            device="cuda",
        )
        cls._model_name_loaded = model_name
        return cls._bd

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        import torch

        bd = await asyncio.to_thread(self._load, self.model_name)

        pcm_f32 = input_pcm.astype(np.float32) / 32768.0
        pad_samples = int(round(INPUT_TRAILING_PAD_S * INPUT_SR))
        padded = np.concatenate([pcm_f32, np.zeros(pad_samples, dtype=np.float32)])
        wav_tensor = torch.from_numpy(padded).unsqueeze(0)

        def _tokenize():
            tokens, _ = bd.tokenize_audio((wav_tensor, INPUT_SR))
            return tokens

        user_audio_tokens = await asyncio.to_thread(_tokenize)
        event_logger.log("send", "tokenized_input", {"n_tokens": len(user_audio_tokens)})

        state = bd.create_stream_state()
        gen = bd.stream_audio_tokens(
            user_audio_tokens, state=state, temperature=0.8, top_p=0.8,
        )
        block_events_n = 2 * bd.y_ratio
        block_duration_s = bd.x_ratio / 12.5   # 0.8s for the default 10:5:10 ratio

        await start_streaming()
        # Segmentation follows <|assistant|>/[EPAD] events, not audio presence.
        recorder.set_speaking(False, at_s=0.0)

        stream_id = str(uuid.uuid4())
        t0 = time.monotonic()
        block_idx = 0

        while True:
            events = await asyncio.to_thread(_pull_n, gen, block_events_n)
            if not events:
                break

            finalize_next_audio = False
            for ev in events:
                if ev.kind == "text":
                    if ev.assistant_started:
                        recorder.set_speaking(True)
                        event_logger.log("recv", "assistant_started")
                    if ev.assistant_finished:
                        recorder.set_speaking(False)
                        finalize_next_audio = True
                        event_logger.log("recv", "assistant_finished")
                    if ev.text:
                        event_logger.log("recv", "text_token", {"text": ev.text})
                elif ev.kind == "audio":
                    if ev.audio_tokens:
                        fin = finalize_next_audio
                        wav_chunk = await asyncio.to_thread(
                            bd.stream_tokens_to_audio, ev.audio_tokens, stream_id, fin
                        )
                        if wav_chunk.numel() > 0:
                            pcm_bytes = _f32_to_i16_bytes(wav_chunk)
                            await recorder.add(pcm_bytes)
                            event_logger.log("recv", "audio", n_bytes=len(pcm_bytes))
                        if fin:
                            stream_id = str(uuid.uuid4())
                    finalize_next_audio = False

            block_idx += 1
            deadline = t0 + block_idx * block_duration_s
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = BayLingDuplexClient(
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
