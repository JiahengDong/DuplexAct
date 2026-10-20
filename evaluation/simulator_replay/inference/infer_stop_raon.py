#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/infer_stop_raon.py

KRAFTON/Raon-SpeechChat-9B client for the `stop` Level-2 action,
simulator-driven (live LLM-triggered injection -- see
trigger_controller.py's module docstring for the full architecture
rationale).

Raon's own per-frame API (raon/utils/wrapper.py's `RaonInferenceModel.
duplex_decoding_step`) is an exact 1:1 frame-synchronous call: every
invocation consumes exactly one 80ms frame of raw PCM and returns exactly
one 80ms frame of output PCM (silence-coded when the model isn't
speaking) -- see infer_raon.py's own module docstring for the full
architecture writeup (Qwen3 LLM + Voxtral-Mini audio encoder + Mimi codec,
SAME codec family/frame geometry as Moshi, 12.5Hz/80ms). This is simpler to
adapt than bayling-duplex: since the per-call unit is already raw PCM (not
pre-tokenized audio tokens), the existing byte-level `DynamicPCMSource` /
`LiveTriggerController` machinery (same as infer_stop_moshi.py) works
directly -- no token-domain source needed here.

Live text signal: this was the "MINOR ADJUST" flagged in
MODEL_COMPATIBILITY.md -- infer_raon.py's client never reads text at all
(only `state.machine_state.phase` for speaking/not-speaking), but the
model DOES generate real text tokens internally every step, on
`state.sequences` (a growing tensor of ALL token ids, input+generated,
frame-synchronized). KRAFTON's own official demo backend
(demo/realtime/runtime/session.py's `RealtimeDuplexSession._decode_step`)
already extracts exactly this signal every frame: diff `state.sequences`
against the length recorded after the previous step, then run the new
token ids through `_decode_text_tokens()` (skip SIL/PAD/EPAD-family
special tokens, decode the rest through the tokenizer). The two small
helpers this needs (`_resolve_ignored_token_ids`, `_decode_text_tokens`,
`_encode_single_token`, `_safe_int_attr`) are copied verbatim from that
file below (self-contained, no FastAPI/sglang dependency) rather than
imported, since session.py's own module-level imports pull in the
official demo's full serving stack (FastAPI, sglang backend) which our
much simpler `direct_replay`-style HF-transformers-only environment does
not have installed, and was never meant to.

This file is deliberately a SEPARATE module from
direct_replay/inference/infer_raon.py (never touches it, never imported by
it, loads its own independent copy of the model into this process) -- see
trigger_controller.py's own module docstring for the "always a separate
module" rule.

Standalone usage (single sample, for smoke testing):
    python infer_stop_raon.py --sample_dir <path to Candor_stop_0038 etc> \\
        --output_wav out.wav --events_path events.jsonl
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger, DynamicPCMSource,
)
from local_common import opensource_root_for  # noqa: E402
from trigger_controller import LiveTriggerController, poll_interval_for_native_chunk  # noqa: E402
from stop_common import stop_cli_main, POST_CUE_OBSERVATION_S  # noqa: E402

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

RAON_NATIVE_FRAME_SEC = 1.0 / 12.5   # 80ms, same Mimi frame geometry as Moshi
RAON_POLL_INTERVAL_S = poll_interval_for_native_chunk(RAON_NATIVE_FRAME_SEC)  # 1.0s unified user-side poll


# ── Text-decode helpers (copied verbatim from KRAFTON's own official demo
# backend, demo/realtime/runtime/session.py -- see module docstring for why
# these are copied rather than imported) ─────────────────────────────────

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


class RaonSpeechChatStopClient(RealtimeClientBase):
    MODEL_SLUG = RAON_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = API_SR

    _model = None
    _processor = None
    _speaker_embeds = None
    _model_name_loaded: Optional[str] = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, cue_pcm: bytes, reference_context: dict, llm_client):
        super().__init__(api_key, model_name, system_prompt or DEFAULT_SYSTEM_PROMPT, verbose)
        self.cue_pcm = cue_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result: dict = {}
        self.text_tokens: list[tuple[float, str]] = []
        self.simulator_speech_segments: list[dict] = []

    @classmethod
    def _load(cls, model_name: str):
        if cls._model is not None and cls._model_name_loaded == model_name:
            return cls._model, cls._processor, cls._speaker_embeds

        import torch
        from transformers import AutoModel
        from raon.models.raon import RaonDuplexModel, RaonModel  # noqa: F401
        from raon.utils.processor import RaonProcessor

        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=False)
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
        # Same 2026-09-06 incident as direct_replay/inference/infer_raon.py:
        # 9B + Voxtral + Mimi load can exceed the per-sample hard timeout.
        # Timeout then retries into a still-initializing ConcurrentAudioDecoder
        # and every later sample dies with `Audio decoder worker error: <N>`.
        # Load once, untimed, before the first sample. Do not edit infer_raon.py.
        await asyncio.to_thread(cls._load, DEFAULT_MODEL)

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        import torch
        from raon.utils.state_machine import DuplexPhase

        model, processor, speaker_embeds = await asyncio.to_thread(self._load, self.model_name)
        sr = processor.sampling_rate
        frame_rate = processor.frame_rate
        samples_per_frame = int(round(sr / frame_rate))
        frame_bytes = samples_per_frame * 2
        frame_duration_s = samples_per_frame / sr

        text_vocab_size = int(getattr(model, "text_vocab_size", None) or getattr(model, "vocab_size", 0))
        ignored_token_ids, sil_token_id, audio_start_token_id = _resolve_ignored_token_ids(
            model, processor.tokenizer)

        system_messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

        def _init_state():
            if system_messages:
                inputs = processor(system_messages, add_generation_prompt=False, device="cuda", dtype=torch.bfloat16)
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
        event_logger.log("send", "duplex_state_initialized", {"frame_duration_s": round(frame_duration_s, 4)})

        await start_streaming()
        recorder.set_speaking(False, at_s=0.0)

        t0 = time.monotonic()

        def now_fn() -> float:
            return time.monotonic() - t0

        source = DynamicPCMSource(initial_pcm=input_pcm.astype(np.int16).tobytes())

        controller = LiveTriggerController(
            recorder=recorder,
            source=source,
            cue_pcm=self.cue_pcm,
            now_fn=now_fn,
            reference_context=self.reference_context,
            client=self.llm_client,
            lead_in_duration_s=input_duration,
            event_logger=event_logger,
            poll_interval_s=RAON_POLL_INTERVAL_S,
        )
        trigger_task = asyncio.create_task(controller.run())

        def _on_trigger_done(t: asyncio.Task) -> None:
            source.finish()

        trigger_task.add_done_callback(_on_trigger_done)

        # Same threading.Event-based safe-cancellation pattern as
        # infer_raon.py -- see that file's own extensive 2026-09-06 incident
        # comment for why checking `future.done()` instead is NOT safe here
        # (an orphaned worker thread can still be touching the model's
        # shared ConcurrentAudioDecoder state well after the awaiting
        # coroutine appears "done").
        step_done = threading.Event()
        step_done.set()
        cue_drained_at: Optional[float] = None
        i = 0

        try:
            while True:
                frame_bytes_chunk = source.pull(frame_bytes)
                frame_f32 = np.frombuffer(frame_bytes_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                frame_tensor = torch.from_numpy(frame_f32).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

                step_done = threading.Event()

                def _step(state=state, frame=frame_tensor, _done=step_done):
                    try:
                        with torch.inference_mode():
                            return model.duplex_decoding_step(state=state, audio_input=frame)
                    finally:
                        _done.set()

                state, decoded_audio = await asyncio.to_thread(_step)

                now = now_fn()
                is_speech = state.machine_state is not None and state.machine_state.phase == DuplexPhase.SPEECH
                recorder.set_speaking(is_speech, at_s=now)

                pcm_bytes = _f32_frame_to_i16_bytes(decoded_audio)
                await recorder.add(pcm_bytes)

                # Live text signal -- see module docstring: diff
                # state.sequences against the length recorded after the
                # previous step (KRAFTON's own official demo backend does
                # the exact same diff every frame).
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
                    controller.on_text(text_delta, at_s=now)
                    event_logger.log("recv", "text_token", {"text": text_delta})

                if source.is_done and cue_drained_at is None:
                    cue_drained_at = now
                    event_logger.log("send", "cue_drained", {"t": cue_drained_at})
                if cue_drained_at is not None and now - cue_drained_at >= POST_CUE_OBSERVATION_S:
                    break

                i += 1
                deadline = t0 + i * frame_duration_s
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        finally:
            if not step_done.is_set():
                await asyncio.to_thread(step_done.wait, 30.0)
            await asyncio.to_thread(model.free_duplex_decoding_state, state)

            if not trigger_task.done():
                trigger_task.cancel()
                try:
                    await trigger_task
                except (asyncio.CancelledError, Exception):
                    pass
            else:
                try:
                    self.trigger_result = trigger_task.result()
                except Exception as e:
                    self.trigger_result = {"error": f"{type(e).__name__}: {e}"}

            self.simulator_speech_segments = [
                {
                    "start": 0.0,
                    "end": round(input_duration, 4),
                    "text": self.reference_context.get("transcript_user", ""),
                    "source": "lead_in",
                }
            ]
            t_trigger = self.trigger_result.get("t_user_int_start")
            if t_trigger is not None:
                cue_duration = len(self.cue_pcm) / 2 / API_SR
                self.simulator_speech_segments.append({
                    "start": round(t_trigger, 4),
                    "end": round(t_trigger + cue_duration, 4),
                    "text": self.reference_context.get("reference_interruption_text", ""),
                    "source": "cue",
                })


def main():
    stop_cli_main(RaonSpeechChatStopClient, RAON_SLUG, API_SR, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
