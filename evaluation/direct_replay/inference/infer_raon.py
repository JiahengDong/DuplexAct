#!/usr/bin/env python3
"""
direct_replay/inference/infer_raon.py

KRAFTON/Raon-SpeechChat-9B client for direct replay.

Architecture: Qwen3 LLM + Voxtral-Mini audio encoder + Mimi codec, i.e. the
SAME codec family and frame geometry as Moshi (12.5 Hz, 80 ms). The model's own
per-frame API (`raon/utils/wrapper.py`'s `RaonInferenceModel.
duplex_decoding_step`) is an exact 1:1 frame-synchronous call: each invocation
consumes exactly one 80 ms frame of raw PCM and returns exactly one 80 ms frame
of output PCM, silence-coded when the model isn't speaking. Speaking/listening
comes off `state.machine_state.phase`, not audio presence, so segmentation goes
through `recorder.set_speaking()` the same way Moshi's does.

Live text signal: the model generates real text tokens internally every step on
`state.sequences`, a growing tensor of ALL token ids (input + generated),
frame-synchronized. KRAFTON's own demo backend (`demo/realtime/runtime/
session.py`'s `RealtimeDuplexSession._decode_step`) extracts it by diffing
`state.sequences` against the length recorded after the previous step and
running the new ids through `_decode_text_tokens()`. The four helpers that
needs (`_safe_int_attr`, `_encode_single_token`, `_resolve_ignored_token_ids`,
`_decode_text_tokens`) are copied verbatim rather than imported, because
session.py's module-level imports pull in the official demo's full serving
stack (FastAPI, sglang) which this HF-transformers-only environment does not
have and was never meant to.

Why the text matters here: Raon's `output.wav` has ~53% of the 20 ms frames
inside its speaking segments as literally all-zero samples, so Whisper returns
1.24 words/s against a 2.5-3.5 words/s conversational band. `run_metrics.
LIVE_TRANSCRIPT_MODELS` therefore scores this slug off the model's own text,
which means `model_speech_transcripts` has to exist -- the whole reason this
client records `self.text_tokens`.

KNOWN LIMITATION -- Raon runs at ~3.4x slower than realtime. Each 80 ms
`duplex_decoding_step` costs ~270 ms of wall clock on an A100, so streaming a
20 s input takes ~68 s. `SynchronizedRecorder`'s gap filling is wall-clock
driven (`expected_count()` reads `loop.time() - stream_t0`), so it pads ~190 ms
of silence per 80 ms of real audio -- which is exactly where the ~53% all-zero
frames come from, and it means output.wav's timeline is stretched ~3.4x
relative to input.wav's. Model text and model segments share that same
stretched recorder clock, so bucketing text into segments stays internally
consistent and Subsequent text is unaffected; but any comparison against
metadata.json's user-side timestamps (pause_start, bc_start,
side_speech_start_s), which live on input.wav's timeline, is misaligned by that
factor. Do not read Raon's timing-dependent metrics without accounting for it.

Standalone usage (single sample, debugging only):
    python infer_raon.py --input_wav in.wav --output_wav out.wav \\
        --events_path events.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger,
)
from local_common import opensource_root_for  # noqa: E402

RAON_SLUG = "raon-speechchat"
DEFAULT_MODEL = "KRAFTON/Raon-SpeechChat-9B"
API_SR = 24000   # RaonProcessor.sampling_rate (Mimi codec) -- same both directions

REPO_DIR = opensource_root_for(RAON_SLUG) / "repo"
REF_SPEAKER_AUDIO = REPO_DIR / "data" / "duplex" / "eval" / "audio" / "spk_ref.wav"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally."
)

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

# How long to wait for an in-flight decode thread to genuinely finish before
# touching the model's shared decoder state. See the incident note in
# `_connect_and_stream`'s finally block.
STEP_DRAIN_TIMEOUT_S = 30.0


# ── Text-decode helpers (copied verbatim from KRAFTON's own demo backend,
# demo/realtime/runtime/session.py -- see module docstring for why copied
# rather than imported) ──────────────────────────────────────────────────────

def _safe_int_attr(obj: Any, name: str, default: Optional[int] = None) -> Optional[int]:
    value = getattr(obj, name, None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _encode_single_token(tokenizer: Any, token_text: str) -> Optional[int]:
    try:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    except TypeError:
        try:
            token_ids = tokenizer.encode(token_text)
        except Exception:
            return None
    except Exception:
        return None
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _resolve_ignored_token_ids(model: Any, tokenizer: Any):
    epad_id = _encode_single_token(tokenizer, "<|fim_middle|>")
    if epad_id is None:
        epad_id = _safe_int_attr(model, "duplex_end_pad_token_id", 0) or 0

    dpad_id = _encode_single_token(tokenizer, "<|fim_prefix|>")
    if dpad_id is None:
        dpad_id = _safe_int_attr(model, "duplex_pad_token_id", 0) or 0

    sil_id = _encode_single_token(tokenizer, "<|audio_output_sil|>")
    if sil_id is None:
        sil_id = _safe_int_attr(model, "duplex_sil_token_id", None)

    ignored = {epad_id, dpad_id}
    for attr in ("audio_output_token_id", "audio_input_token_id", "im_start_token_id",
                 "audio_start_token_id", "speaker_token_id", "eos_token_id"):
        val = _safe_int_attr(model, attr, None)
        if val is not None:
            ignored.add(val)
    if sil_id is not None:
        ignored.add(sil_id)

    return ignored, sil_id, _safe_int_attr(model, "audio_start_token_id", None)


def _decode_text_tokens(token_ids, tokenizer, *, text_vocab_size, ignored_token_ids,
                         sil_token_id, audio_start_token_id) -> str:
    pieces = []
    text_buffer = []

    def _flush():
        if not text_buffer:
            return
        pieces.append(tokenizer.decode(text_buffer, skip_special_tokens=False))
        text_buffer.clear()

    for token_id in token_ids:
        if (sil_token_id is not None and token_id == sil_token_id) or \
           (audio_start_token_id is not None and token_id == audio_start_token_id):
            _flush()
            pieces.append("\n")
            continue
        if token_id in ignored_token_ids:
            _flush()
            continue
        if token_id < text_vocab_size:
            text_buffer.append(token_id)

    _flush()
    return "".join(pieces)


def _f32_frame_to_i16_bytes(frame) -> bytes:
    arr = frame.squeeze(0).detach().float().cpu().numpy()
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class RaonSpeechChatClient(RealtimeClientBase):
    MODEL_SLUG = RAON_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = API_SR
    # 3.4x slower than realtime (see module docstring), so the default 90 s on
    # top of input_duration cannot cover a sample. Being generous here is not
    # just about avoiding false timeouts: the 2026-09-06 incident traced
    # `Audio decoder worker error: <N>` to run_inference.py's outer
    # `asyncio.wait_for` cancelling a slow sample, which orphans the decode
    # thread onto the shared ConcurrentAudioDecoder state. Not timing out in
    # the first place is the cheapest defence.
    TIMEOUT_OVERHEAD_S = 600.0

    _model = None
    _processor = None
    _speaker_embeds = None
    _model_name_loaded: Optional[str] = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool = False):
        super().__init__(api_key, model_name, system_prompt or DEFAULT_SYSTEM_PROMPT, verbose)
        self.text_tokens: list[tuple[float, str]] = []

    @classmethod
    def _load(cls, model_name: str):
        if cls._model is not None and cls._model_name_loaded == model_name:
            return cls._model, cls._processor, cls._speaker_embeds

        import torch
        from transformers import AutoModel
        from raon.models.raon import RaonDuplexModel, RaonModel  # noqa: F401
        from raon.utils.processor import RaonProcessor

        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                          trust_remote_code=False)
        model = model.to("cuda").eval()
        processor = RaonProcessor.from_pretrained(model_name)

        speaker_embeds = None
        if hasattr(model, "speaker_encoder") and model.speaker_encoder is not None:
            from raon.utils.audio_io import load_audio as raon_load_audio
            spk_audio = raon_load_audio(
                str(REF_SPEAKER_AUDIO), processor.sampling_rate, mono=True,
                device="cuda", dtype=torch.bfloat16,
            )[0]
            speaker_embeds = model._compute_speaker_embeds(spk_audio, None)

        cls._model, cls._processor, cls._speaker_embeds = model, processor, speaker_embeds
        cls._model_name_loaded = model_name
        return model, processor, speaker_embeds

    @classmethod
    async def warmup(cls) -> None:
        # 2026-09-06 incident: 9B + Voxtral + Mimi load can alone exceed the
        # per-sample budget, so sample #1 times out mid-load and the retry
        # fires a second concurrent load into a still-initializing
        # ConcurrentAudioDecoder -- after which every later sample dies with
        # `Audio decoder worker error: <N>`. Load once, untimed, up front.
        await asyncio.to_thread(cls._load, DEFAULT_MODEL)

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        import torch
        from raon.utils.state_machine import DuplexPhase

        model, processor, speaker_embeds = await asyncio.to_thread(self._load, self.model_name)
        sr = processor.sampling_rate
        samples_per_frame = int(round(sr / processor.frame_rate))
        frame_duration_s = samples_per_frame / sr

        text_vocab_size = int(getattr(model, "text_vocab_size", None)
                              or getattr(model, "vocab_size", 0))
        ignored_token_ids, sil_token_id, audio_start_token_id = _resolve_ignored_token_ids(
            model, processor.tokenizer)

        system_messages = ([{"role": "system", "content": self.system_prompt}]
                           if self.system_prompt else [])

        def _init_state():
            if system_messages:
                inputs = processor(system_messages, add_generation_prompt=False,
                                    device="cuda", dtype=torch.bfloat16)
                system_tokens = inputs["input_ids"]
            else:
                system_tokens = torch.zeros((1, 0), dtype=torch.long, device="cuda")
            with torch.inference_mode():
                return model.init_duplex_decoding_state(
                    sequences=system_tokens,
                    attention_mask=torch.ones_like(system_tokens),
                    do_sample=True,
                    temperature=0.9,
                    top_k=66,
                    top_p=0.95,
                    speaker_embeds=speaker_embeds,
                    speak_first=False,
                )

        state = await asyncio.to_thread(_init_state)
        last_sequence_len = int(state.sequences.shape[1])
        event_logger.log("send", "duplex_state_initialized",
                          {"frame_duration_s": round(frame_duration_s, 4)})

        # Pad the tail so the final partial frame is a whole frame.
        pcm = input_pcm.astype(np.int16)
        n_frames = int(np.ceil(len(pcm) / samples_per_frame))
        pad = n_frames * samples_per_frame - len(pcm)
        if pad:
            pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])

        await start_streaming()
        # Manual segmentation from t=0 -- the model's own DuplexPhase, not
        # audio presence, is its speaking signal (it emits silence-coded
        # frames continuously), same reasoning as Moshi's recorder.
        recorder.set_speaking(False, at_s=0.0)

        t0 = time.monotonic()
        step_done = threading.Event()
        step_done.set()

        try:
            for i in range(n_frames):
                frame = pcm[i * samples_per_frame:(i + 1) * samples_per_frame]
                frame_f32 = frame.astype(np.float32) / 32768.0
                frame_tensor = torch.from_numpy(frame_f32).unsqueeze(0).to(
                    "cuda", dtype=torch.bfloat16)

                step_done = threading.Event()

                def _step(state=state, frame=frame_tensor, _done=step_done):
                    try:
                        with torch.inference_mode():
                            return model.duplex_decoding_step(state=state, audio_input=frame)
                    finally:
                        _done.set()

                state, decoded_audio = await asyncio.to_thread(_step)

                now = time.monotonic() - t0
                is_speech = (state.machine_state is not None
                             and state.machine_state.phase == DuplexPhase.SPEECH)
                recorder.set_speaking(is_speech, at_s=now)

                await recorder.add(_f32_frame_to_i16_bytes(decoded_audio))

                # Live text: diff state.sequences against the length recorded
                # after the previous step (KRAFTON's own demo backend does the
                # exact same diff every frame). `now` is the recorder's clock,
                # which is what model_speech_segments are on, so bucketing
                # these into those segments needs no further alignment.
                sequences = state.sequences
                new_token_ids = sequences[0, last_sequence_len:].tolist()
                last_sequence_len = int(sequences.shape[1])
                text_delta = _decode_text_tokens(
                    new_token_ids, processor.tokenizer,
                    text_vocab_size=text_vocab_size, ignored_token_ids=ignored_token_ids,
                    sil_token_id=sil_token_id, audio_start_token_id=audio_start_token_id,
                )
                if text_delta:
                    self.text_tokens.append((now, text_delta))
                    event_logger.log("recv", "text_token", {"text": text_delta})

                # Pace to the frame grid when the model runs faster than
                # realtime. It currently does not (~3.4x slower, see module
                # docstring), so this is a no-op in practice -- kept so the
                # client stays correct rather than accidentally correct.
                sleep_s = t0 + (i + 1) * frame_duration_s - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        finally:
            # NOT `future.done()`: an orphaned worker thread can still be
            # inside duplex_decoding_step (pushing/pulling on the old stream
            # id) well after the awaiting coroutine unwinds, and
            # free_duplex_decoding_state -> destroy_stream racing it corrupts
            # the decoder worker's `stream_states`, which then surfaces as a
            # bogus first-frame error on the NEXT sample. Wait for the thread
            # itself.
            if not step_done.is_set():
                await asyncio.to_thread(step_done.wait, STEP_DRAIN_TIMEOUT_S)
            await asyncio.to_thread(model.free_duplex_decoding_state, state)


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        input_duration = sf.info(str(args.input_wav)).duration

    client = RaonSpeechChatClient(
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
