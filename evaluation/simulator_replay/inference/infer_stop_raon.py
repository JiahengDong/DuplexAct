#!/usr/bin/env python3
"""
KRAFTON/Raon-SpeechChat-9B client for the `stop` action, with the cue
injected live by an LLM trigger (see trigger_controller.py).

`duplex_decoding_step` consumes one 80ms frame of raw PCM and returns one
80ms frame of output PCM (silence when the model is not speaking); the
model uses a Mimi codec at 12.5Hz like Moshi (see infer_raon.py). Because
the per-step input is raw PCM, the byte-level `DynamicPCMSource` /
`LiveTriggerController` path works as in infer_stop_moshi.py.

Live text: the model's text tokens accumulate on `state.sequences`. As in
KRAFTON's demo backend (demo/realtime/runtime/session.py), each step decodes
the new token ids since the previous step, skipping SIL/PAD/EPAD special
tokens. The helper functions are copied from that file because importing it
pulls in the demo's FastAPI/sglang serving stack.

Standalone usage (single sample):
    python infer_stop_raon.py --sample_dir <stop sample dir> \\
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
API_SR = 24000   # RaonProcessor.sampling_rate (Mimi codec), both directions

REPO_DIR = opensource_root_for(RAON_SLUG) / "repo"
REF_SPEAKER_AUDIO = REPO_DIR / "data" / "duplex" / "eval" / "audio" / "spk_ref.wav"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and friendly conversational AI assistant. "
    "Listen carefully to the user and respond naturally."
)

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

RAON_NATIVE_FRAME_SEC = 1.0 / 12.5   # 80ms Mimi frame
RAON_POLL_INTERVAL_S = poll_interval_for_native_chunk(RAON_NATIVE_FRAME_SEC)


# ── Text-decode helpers (copied from KRAFTON's demo backend,
# demo/realtime/runtime/session.py) ──────────────────────────────────────

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
        # Loading 9B + Voxtral + Mimi can exceed the per-sample timeout, and a
        # retry into a still-initializing ConcurrentAudioDecoder breaks every
        # later sample. Load once, untimed, before the first sample.
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

        # Track step completion with a threading.Event rather than the
        # awaiting coroutine: after cancellation the worker thread can still
        # be using the model's shared ConcurrentAudioDecoder state, so cleanup
        # waits for the event (as in infer_raon.py).
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

                # Live text: decode the token ids appended to state.sequences
                # since the previous step.
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
