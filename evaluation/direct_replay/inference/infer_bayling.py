#!/usr/bin/env python3
"""
direct_replay/inference/infer_bayling.py
==============================================
BayLing-Models/BayLing-Duplex client. Category A/B (genuine incremental
streaming, in-process, no network server) -- unlike Voila-Autonomous, this
model's own Python package already exposes a proper block-by-block
streaming primitive (`BayLingDuplex.stream_audio_tokens`, a real Python
generator that only computes block N+1 once we ask for it), so we drive it
with real wall-clock pacing exactly like infer_minicpm_o.py's
streaming_prefill/streaming_generate loop -- no batch call, no need to
reconstruct segmentation post-hoc.

Architecture (see BayLing-Models/BayLing-Duplex's duplex.py, GitHub): a
single autoregressive LLM (GLM-4-Voice backbone) consumes user-speech
tokens (Whisper-style VQ encoder, 12.5Hz) and assistant text/speech tokens
as one multi-channel interleaved sequence, blocked as
x_ratio:y_ratio:z_ratio = 10:5:10 by default (one block = 10 user-audio
tokens = 0.8s of user audio, matched to 5 (text-token, 10-audio-token)
generation sub-steps). Turn-taking is expressed purely via next-token
prediction over four added special tokens ([SILENCE], <|assistant|>,
[PAD], [EPAD]) -- no separate VAD/classifier -- so `<|assistant|>` /
`[EPAD]` text-token events ARE the model's own real speaking-state signal,
read directly off `DuplexStreamEvent.assistant_started` /
`.assistant_finished` (no reconstruction needed, unlike Voila's streamer
hack).

Per-block loop: `stream_audio_tokens()` is called ONCE per sample (it's a
generator covering the whole clip), and we pull exactly `2 * y_ratio`
events per iteration -- one block always yields exactly that many (a
(text, audio) pair per of the y_ratio sub-steps) unless the generator is
exhausted, so this needs no block-index bookkeeping. Audio tokens are
decoded to waveform immediately per sub-step via `stream_tokens_to_audio`
(CosyVoice-style streaming flow-matching + HiFT decoder) and fed straight
into the recorder -- genuinely incremental, not a big batch decode at the
end.

Trailing pad: `stream_audio_tokens`'s outer loop only iterates over
however many user-audio tokens we hand it, so if the model hasn't reached
[EPAD] by the time real user audio runs out, its reply would be cut off
mid-utterance. We append INPUT_TRAILING_PAD_S of silence to the input clip
before tokenizing (same fix, same rationale, as infer_voila.py) so the
model keeps "listening" (to silence) long enough to finish speaking.

Vocoder pacing: CosyVoice/HiFT decode is slower than realtime (~0.4 s of
audio every ~1.3 s). The default recorder would write that GPU delay as
silence inside one `<|assistant|>`…`[EPAD]` turn (F-05 + post-target pad),
so output.wav is chopped, Whisper ASR fragments or drops the sentence, and
listen Subsequent (Wait / Backchannel / Take turn / Ignore) judges
"no speech" or "incoherent". CONTIGUOUS_WHILE_SPEAKING writes one turn's
chunks back-to-back; Immediate onset is still `<|assistant|>` on the
recorder clock. Trigger text in the simulator is live tokens, not ASR.

max_epad_count=1 (the library's own default): generation stops as soon as
the model completes ONE assistant utterance -- matches this benchmark's
per-instance expectation of at most one model turn per sample, same as
every other direct-replay client.

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

    _bd = None   # BayLingDuplex singleton (loaded once per job)
    _model_name_loaded: Optional[str] = None

    @classmethod
    def _load(cls, model_name: str):
        if cls._bd is not None and cls._model_name_loaded == model_name:
            return cls._bd

        from huggingface_hub import snapshot_download
        from bayling_duplex.duplex import BayLingDuplex

        # decoder_path is NOT resolved through transformers' from_pretrained()
        # machinery like model_path/speech_tokenizer_path are -- AudioDecoder
        # (flow_inference.py) does a plain open(config_path)/torch.load(...)
        # on literal "<decoder_path>/config.yaml", "/flow.pt", "/hift.pt", so
        # it needs an actual local directory, not a bare HF repo id. Resolve
        # it to its on-disk snapshot dir ourselves first (same pattern as
        # infer_freeze_omni.py / infer_vita15.py).
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
        # Manual segmentation from t=0 -- <|assistant|>/[EPAD] text-token
        # events (not audio presence) are the model's own speaking signal,
        # same reasoning as Moshi's continuous-signal recorder.
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
