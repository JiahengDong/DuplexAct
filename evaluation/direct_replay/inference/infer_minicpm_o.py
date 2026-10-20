#!/usr/bin/env python3
"""
direct_replay/inference/infer_minicpm_o.py
==============================================
OpenBMB MiniCPM-o 4.5 client. Category B (in-process streaming, no network
server) -- unlike Moshi, MiniCPM-o 4.5's duplex mode is a plain Python
object (`model.as_duplex()`) exposing a chunk-in/chunk-out method pair
(`streaming_prefill` / `streaming_generate`) that we drive directly from
this async client, real-time-paced via `asyncio.sleep`, exactly mirroring
the closed-source clients' per-chunk send/receive loop -- just without a
websocket in between.

API reference (from the model's HF card, "Duplex Omni Mode" section --
https://huggingface.co/openbmb/MiniCPM-o-4_5): the only officially
documented duplex example is video+audio. We generalize it to audio-only
by passing `frame_list=[]` every chunk (no video), which the documented
loop already handles as a valid per-chunk state (`frame_list` is built up
conditionally and can legitimately be empty). This is our best-effort
reading of an under-documented audio-only path -- validate/adjust after
the first real smoke test.

Per-chunk cadence: the model's own examples consistently process audio in
1-second chunks ("For each 1s audio chunk, perform streaming_prefill once
..."), each followed by one `streaming_generate` call that returns:
  - "audio_waveform": the model's synthesized speech for this chunk, or
    None while it is only listening.
  - "is_listen": True while the model has decided not to speak this chunk.
  - "text": the text of any speech produced this chunk.
  - "end_of_turn": True when the model's utterance has just ended.
  - "current_time": the model's own notion of elapsed time (informational).

Because `audio_waveform` is None whenever the model is only listening (see
the reference example), we can rely on the SAME automatic, presence-based
segment tracking used by the closed-source clients (recorder.add() only
when audio is actually produced) -- no need for the explicit
`recorder.set_speaking()` override that Moshi requires.

The official reference clip named in the model card's own example
(assets/HT_ref_audio.wav) is NOT in the public GitHub repo, but IS present
as a file in the HF model repo itself -- we fetch it via
`hf_hub_download("openbmb/MiniCPM-o-4_5", "assets/HT_ref_audio.wav")` once
and cache a local copy at
opensource_models/minicpm-o-4.5/repo/assets/HT_ref_audio.wav, so we use
the exact same reference clip as the documented example (6s, 16kHz mono)
for `model.prepare(ref_audio=..., prompt_wav_path=...)`'s voice-cloning
input.

Standalone usage (single sample, for debugging):
    python infer_minicpm_o.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from local_common import opensource_root_for
from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger

MINICPM_SLUG = "minicpm-o-4.5"
DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"

INPUT_SR  = 16_000   # matches the model's own librosa.load(..., sr=16000) convention
OUTPUT_SR = 24_000   # matches generate_duplex_video's output_sample_rate=24000

CHUNK_SEC = 1.0                        # matches the model's documented 1s chunking
MAX_NEW_SPEAK_TOKENS_PER_CHUNK = 20     # from the reference duplex example

REF_AUDIO_PATH = opensource_root_for(MINICPM_SLUG) / "repo" / "assets" / "HT_ref_audio.wav"

DEFAULT_PREFIX_PROMPT = "Streaming Omni Conversation."


def _ensure_ref_audio(model_name: str) -> Path:
    """Fetches the model card's own demo reference clip from the HF repo
    (once; cached at REF_AUDIO_PATH thereafter) -- see module docstring."""
    if REF_AUDIO_PATH.exists():
        return REF_AUDIO_PATH
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(model_name, "assets/HT_ref_audio.wav")
    REF_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(src, REF_AUDIO_PATH)
    return REF_AUDIO_PATH


def _pcm_f32_to_i16_bytes(pcm_f32: np.ndarray) -> bytes:
    return (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class MiniCPMDuplexClient(RealtimeClientBase):
    MODEL_SLUG = MINICPM_SLUG
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = INPUT_SR

    # Class-level singleton: load weights once per batch job (~9B params),
    # reused across every sample run_inference.py feeds through this
    # process -- loading takes far longer than any one sample's duration.
    _model = None
    _model_name_loaded: str | None = None

    @classmethod
    def _load_model(cls, model_name: str):
        if cls._model is not None and cls._model_name_loaded == model_name:
            return cls._model

        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False,   # audio-only: no vision encoder needed
            init_audio=True,
            init_tts=True,
        )
        model.eval().cuda()
        model.init_tts()
        cls._model = model.as_duplex()
        cls._model_name_loaded = model_name
        return cls._model

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        import librosa

        model = await asyncio.to_thread(self._load_model, self.model_name)
        ref_audio_path = await asyncio.to_thread(_ensure_ref_audio, self.model_name)
        ref_audio, _ = librosa.load(str(ref_audio_path), sr=INPUT_SR, mono=True)

        def _prepare_session():
            if hasattr(model, "reset_session"):
                model.reset_session()
            model.prepare(
                prefix_system_prompt=self.system_prompt or DEFAULT_PREFIX_PROMPT,
                ref_audio=ref_audio,
                prompt_wav_path=str(ref_audio_path),
            )

        await asyncio.to_thread(_prepare_session)
        event_logger.log("send", "session_prepared")

        pcm_f32 = input_pcm.astype(np.float32) / 32768.0
        chunk_samples = int(INPUT_SR * CHUNK_SEC)
        pad = (-len(pcm_f32)) % chunk_samples
        if pad:
            pcm_f32 = np.concatenate([pcm_f32, np.zeros(pad, dtype=np.float32)])
        n_chunks = len(pcm_f32) // chunk_samples

        await start_streaming()
        t0 = time.monotonic()

        for i in range(n_chunks):
            chunk = pcm_f32[i * chunk_samples: (i + 1) * chunk_samples]

            def _step():
                model.streaming_prefill(
                    audio_waveform=chunk,
                    frame_list=[],            # audio-only: no video this chunk
                    max_slice_nums=1,
                    batch_vision_feed=False,
                )
                return model.streaming_generate(
                    prompt_wav_path=str(ref_audio_path),
                    max_new_speak_tokens_per_chunk=MAX_NEW_SPEAK_TOKENS_PER_CHUNK,
                    decode_mode="sampling",
                )

            result = await asyncio.to_thread(_step)
            now = time.monotonic() - t0

            audio_out = result.get("audio_waveform")
            has_audio = audio_out is not None and len(audio_out) > 0
            # Smoke-test finding: MiniCPM-o's audio-only duplex mode returns a
            # non-None (but near-silent) audio_waveform on almost every chunk,
            # even while listening -- "audio_waveform is not None" alone is
            # NOT a reliable speaking signal here (unlike the documented
            # video+audio example, where it appears to genuinely go None).
            # result["is_listen"] is the model's own explicit signal (used
            # exactly this way in the reference example's own print
            # statement: `"listen..." if result["is_listen"] else "speak> ..."`)
            # so it takes priority whenever present.
            is_listen = result.get("is_listen")
            if is_listen is None:
                is_listen = not has_audio

            if not is_listen and has_audio:
                await recorder.add(_pcm_f32_to_i16_bytes(np.asarray(audio_out)))
                event_logger.log("recv", "audio", {"text": result.get("text", "")},
                                  n_bytes=len(audio_out) * 2)
            else:
                event_logger.log("recv", "listen", {"t": round(now, 4)})

            if result.get("end_of_turn"):
                event_logger.log("recv", "end_of_turn", {"t": round(now, 4)})

            # Absolute-deadline pacing (not a flat post-hoc sleep): if a
            # chunk's GPU compute overruns CHUNK_SEC, we don't add extra
            # delay on top -- we just proceed immediately and let the next
            # iteration's deadline naturally reflect the real elapsed time,
            # so a model that can't keep up with real-time shows up as
            # genuinely late output rather than being artificially resynced.
            deadline = t0 + (i + 1) * CHUNK_SEC
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

    client = MiniCPMDuplexClient(
        api_key="",
        model_name=args.model,
        system_prompt=DEFAULT_PREFIX_PROMPT,
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
