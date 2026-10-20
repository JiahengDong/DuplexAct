#!/usr/bin/env python3
"""
Raon-SpeechChat client for the simulator-driven `interrupt_user` action.
Uses the same duplex_decoding_step loop and sequences-diff text decoding as
infer_stop_raon.py (helpers imported from there), with an
InterruptMonitorController instead of the stop trigger.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger, DynamicPCMSource,
)
from local_common import opensource_root_for  # noqa: E402
from interrupt_controller import InterruptMonitorController  # noqa: E402
from interrupt_user_common import (  # noqa: E402
    interrupt_cli_main, POST_BRANCH_OBSERVATION_S, build_interrupt_simulator_speech_segments,
)
from infer_stop_raon import (  # noqa: E402
    _resolve_ignored_token_ids, _decode_text_tokens, _f32_frame_to_i16_bytes,
    RAON_SLUG, DEFAULT_MODEL, API_SR, REF_SPEAKER_AUDIO, DEFAULT_SYSTEM_PROMPT,
    RAON_POLL_INTERVAL_S, REPO_DIR,
)

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))


class InterruptRaonSpeechChatClient(RealtimeClientBase):
    MODEL_SLUG = RAON_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = API_SR

    _model = None
    _processor = None
    _speaker_embeds = None
    _model_name_loaded: Optional[str] = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, branch_acknowledged_pcm: bytes, branch_ignored_pcm: bytes,
                 reference_context: dict, llm_client=None):
        super().__init__(api_key, model_name, system_prompt or DEFAULT_SYSTEM_PROMPT, verbose)
        self.branch_acknowledged_pcm = branch_acknowledged_pcm
        self.branch_ignored_pcm = branch_ignored_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.monitor_result: dict = {}
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
                    do_sample=True, temperature=0.9, top_k=66, top_p=0.95,
                    speaker_embeds=speaker_embeds, speak_first=False,
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
        monitor = InterruptMonitorController(
            recorder=recorder, source=source,
            user_speech_duration=input_duration,
            branch_acknowledged_pcm=self.branch_acknowledged_pcm,
            branch_ignored_pcm=self.branch_ignored_pcm,
            now_fn=now_fn, reference_context=self.reference_context,
            llm_client=self.llm_client, event_logger=event_logger,
            poll_interval_s=RAON_POLL_INTERVAL_S,
        )
        monitor_task = asyncio.create_task(monitor.run())
        monitor_task.add_done_callback(lambda t: source.finish())

        step_done = threading.Event()
        step_done.set()
        branch_drained_at: Optional[float] = None
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
                await recorder.add(_f32_frame_to_i16_bytes(decoded_audio))

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
                    monitor.on_text(text_delta, at_s=now)
                    event_logger.log("recv", "text_token", {"text": text_delta})

                if source.is_done and branch_drained_at is None:
                    branch_drained_at = now
                    event_logger.log("send", "branch_drained", {"t": branch_drained_at})
                if branch_drained_at is not None and now - branch_drained_at >= POST_BRANCH_OBSERVATION_S:
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
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except (asyncio.CancelledError, Exception):
                    pass
            else:
                try:
                    self.monitor_result = monitor_task.result()
                except Exception as e:
                    self.monitor_result = {"error": f"{type(e).__name__}: {e}"}
            self.simulator_speech_segments = build_interrupt_simulator_speech_segments(
                monitor_result=self.monitor_result,
                reference_context=self.reference_context,
                input_duration=input_duration,
                branch_acknowledged_pcm=self.branch_acknowledged_pcm,
                branch_ignored_pcm=self.branch_ignored_pcm,
                pcm_sr=API_SR,
            )


def main():
    interrupt_cli_main(InterruptRaonSpeechChatClient, RAON_SLUG, API_SR, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
