#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_interrupt_minicpm.py

MiniCPM-o 4.5 interrupt_user client. Separate from infer_stop_minicpm.py.
Same as_duplex streaming_prefill/streaming_generate loop; InterruptMonitorController.
Poll interval remains 10.0s (native 1.0s chunk × tau2 10-tick ratio).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

DIRECT_REPLAY_INFERENCE = Path(__file__).resolve().parents[2] / "direct_replay" / "inference"
sys.path.insert(0, str(DIRECT_REPLAY_INFERENCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger, DynamicPCMSource,
)
from interrupt_controller import InterruptMonitorController  # noqa: E402
from interrupt_user_common import (  # noqa: E402
    interrupt_cli_main, POST_BRANCH_OBSERVATION_S, build_interrupt_simulator_speech_segments,
)
from infer_stop_minicpm import (  # noqa: E402  -- read-only constants/helpers
    MINICPM_SLUG, DEFAULT_MODEL, INPUT_SR, OUTPUT_SR, CHUNK_SEC, CHUNK_BYTES,
    MAX_NEW_SPEAK_TOKENS_PER_CHUNK, MINICPM_POLL_INTERVAL_S, DEFAULT_PREFIX_PROMPT,
    _ensure_ref_audio, _pcm_f32_to_i16_bytes,
)


class InterruptMiniCPMClient(RealtimeClientBase):
    MODEL_SLUG = MINICPM_SLUG
    OUTPUT_SR = OUTPUT_SR
    INPUT_SR = INPUT_SR

    _model = None
    _model_name_loaded: Optional[str] = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, branch_acknowledged_pcm: bytes, branch_ignored_pcm: bytes,
                 reference_context: dict, llm_client=None):
        super().__init__(api_key, model_name, system_prompt or DEFAULT_PREFIX_PROMPT, verbose)
        self.branch_acknowledged_pcm = branch_acknowledged_pcm
        self.branch_ignored_pcm = branch_ignored_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.monitor_result: dict = {}
        self.text_tokens: list[tuple[float, str]] = []
        self.simulator_speech_segments: list[dict] = []

    @classmethod
    def _load_model(cls, model_name: str):
        if cls._model is not None and cls._model_name_loaded == model_name:
            return cls._model
        import torch
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16, init_vision=False, init_audio=True, init_tts=True,
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
                ref_audio=ref_audio, prompt_wav_path=str(ref_audio_path),
            )

        await asyncio.to_thread(_prepare_session)
        event_logger.log("send", "session_prepared")
        await start_streaming()
        recorder.set_speaking(False, at_s=0.0)
        t0 = time.monotonic()

        def now_fn() -> float:
            return time.monotonic() - t0

        source = DynamicPCMSource(initial_pcm=input_pcm.tobytes())
        monitor = InterruptMonitorController(
            recorder=recorder, source=source,
            user_speech_duration=input_duration,
            branch_acknowledged_pcm=self.branch_acknowledged_pcm,
            branch_ignored_pcm=self.branch_ignored_pcm,
            now_fn=now_fn, reference_context=self.reference_context,
            llm_client=self.llm_client, event_logger=event_logger,
            poll_interval_s=MINICPM_POLL_INTERVAL_S,
        )
        monitor_task = asyncio.create_task(monitor.run())
        monitor_task.add_done_callback(lambda t: source.finish())

        branch_drained_at: Optional[float] = None
        i = 0
        try:
            while True:
                chunk_bytes = source.pull(CHUNK_BYTES)
                chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                def _step(chunk=chunk):
                    model.streaming_prefill(
                        audio_waveform=chunk, frame_list=[],
                        max_slice_nums=1, batch_vision_feed=False,
                    )
                    return model.streaming_generate(
                        prompt_wav_path=str(ref_audio_path),
                        max_new_speak_tokens_per_chunk=MAX_NEW_SPEAK_TOKENS_PER_CHUNK,
                        decode_mode="sampling",
                    )

                result = await asyncio.to_thread(_step)
                now = now_fn()
                audio_out = result.get("audio_waveform")
                has_audio = audio_out is not None and len(audio_out) > 0
                is_listen = result.get("is_listen")
                if is_listen is None:
                    is_listen = not has_audio
                speaking = (not is_listen) and has_audio
                recorder.set_speaking(speaking, at_s=now)
                if speaking:
                    await recorder.add(_pcm_f32_to_i16_bytes(np.asarray(audio_out)))
                text = result.get("text") or ""
                if text:
                    self.text_tokens.append((now, text))
                    monitor.on_text(text, at_s=now)
                    event_logger.log("recv", "text", {"text": text})
                else:
                    event_logger.log("recv", "listen" if is_listen else "audio", {"t": round(now, 4)})
                if result.get("end_of_turn"):
                    event_logger.log("recv", "end_of_turn", {"t": round(now, 4)})
                if source.is_done and branch_drained_at is None:
                    branch_drained_at = now
                    event_logger.log("send", "branch_drained", {"t": branch_drained_at})
                if branch_drained_at is not None and now - branch_drained_at >= POST_BRANCH_OBSERVATION_S:
                    break
                i += 1
                deadline = t0 + i * CHUNK_SEC
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        finally:
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
                pcm_sr=INPUT_SR,
            )


def main():
    interrupt_cli_main(InterruptMiniCPMClient, MINICPM_SLUG, INPUT_SR, DEFAULT_MODEL,
                       output_sr=OUTPUT_SR)


if __name__ == "__main__":
    main()
