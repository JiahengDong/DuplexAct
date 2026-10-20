#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/infer_stop_minicpm.py

OpenBMB MiniCPM-o 4.5 client for the `stop` Level-2 action,
simulator-driven (live LLM-triggered injection -- see
trigger_controller.py's module docstring for the full architecture
rationale).

Reuses direct_replay/inference/infer_minicpm_o.py's exact duplex-driving
pattern (model.as_duplex()'s `streaming_prefill`/`streaming_generate`
chunk-in/chunk-out pair, 1.0s chunks, `is_listen`/`text`/`end_of_turn`
fields -- see that file's own module docstring for the full API writeup
and its documented caveat that `audio_waveform is not None` alone is NOT a
reliable speaking signal for audio-only duplex, `is_listen` must be
checked instead). This is a DIRECT PORT (per MODEL_COMPATIBILITY.md): the
per-chunk `result["text"]` field is already a live signal the original
client receives every chunk (it just doesn't act on it beyond event
logging) -- so, unlike raon-speechchat, no new signal-sourcing logic is
needed, only wiring that already-present field into
`LiveTriggerController.on_text()` and swapping the fixed-length input loop
for `DynamicPCMSource`-driven injection (same restructuring as
infer_stop_moshi.py did to infer_moshi.py).

Native duplex granularity is the 1.0s chunk. Poll interval is the unified
1.0s simulated-user cadence (POLL_INTERVAL_S), not 10x that chunk.

This file is deliberately a SEPARATE module from
direct_replay/inference/infer_minicpm_o.py (never touches it, never
imported by it, loads its own independent copy of the model into this
process) -- see trigger_controller.py's own module docstring for the
"always a separate module" rule.

Standalone usage (single sample, for smoke testing):
    python infer_stop_minicpm.py --sample_dir <path to Candor_stop_0038 etc> \\
        --output_wav out.wav --events_path events.jsonl
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_common import opensource_root_for  # noqa: E402
from realtime_common import (  # noqa: E402
    RealtimeClientBase, SynchronizedRecorder, EventLogger, DynamicPCMSource,
)
from trigger_controller import LiveTriggerController, poll_interval_for_native_chunk  # noqa: E402
from stop_common import stop_cli_main, POST_CUE_OBSERVATION_S  # noqa: E402

MINICPM_SLUG = "minicpm-o-4.5"
DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"

INPUT_SR  = 16_000   # matches infer_minicpm_o.py's own librosa.load(..., sr=16000) convention
OUTPUT_SR = 24_000   # matches generate_duplex_video's output_sample_rate=24000

CHUNK_SEC = 1.0
MAX_NEW_SPEAK_TOKENS_PER_CHUNK = 20
CHUNK_SAMPLES = int(INPUT_SR * CHUNK_SEC)
CHUNK_BYTES = CHUNK_SAMPLES * 2   # int16

MINICPM_POLL_INTERVAL_S = poll_interval_for_native_chunk(CHUNK_SEC)  # 1.0s unified user-side poll

REF_AUDIO_PATH = opensource_root_for(MINICPM_SLUG) / "repo" / "assets" / "HT_ref_audio.wav"
DEFAULT_PREFIX_PROMPT = "Streaming Omni Conversation."


def _ensure_ref_audio(model_name: str) -> Path:
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


class MiniCPMStopClient(RealtimeClientBase):
    MODEL_SLUG = MINICPM_SLUG
    OUTPUT_SR = OUTPUT_SR
    INPUT_SR = INPUT_SR

    _model = None
    _model_name_loaded: Optional[str] = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, cue_pcm: bytes, reference_context: dict, llm_client):
        super().__init__(api_key, model_name, system_prompt or DEFAULT_PREFIX_PROMPT, verbose)
        self.cue_pcm = cue_pcm                   # int16 PCM bytes @ INPUT_SR
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result: dict = {}
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
                ref_audio=ref_audio,
                prompt_wav_path=str(ref_audio_path),
            )

        await asyncio.to_thread(_prepare_session)
        event_logger.log("send", "session_prepared")

        await start_streaming()
        recorder.set_speaking(False, at_s=0.0)
        t0 = time.monotonic()

        def now_fn() -> float:
            return time.monotonic() - t0

        source = DynamicPCMSource(initial_pcm=input_pcm.tobytes())

        controller = LiveTriggerController(
            recorder=recorder,
            source=source,
            cue_pcm=self.cue_pcm,
            now_fn=now_fn,
            reference_context=self.reference_context,
            client=self.llm_client,
            lead_in_duration_s=input_duration,
            event_logger=event_logger,
            poll_interval_s=MINICPM_POLL_INTERVAL_S,
        )
        trigger_task = asyncio.create_task(controller.run())

        def _on_trigger_done(t: asyncio.Task) -> None:
            source.finish()

        trigger_task.add_done_callback(_on_trigger_done)

        cue_drained_at: Optional[float] = None
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
                    controller.on_text(text, at_s=now)
                    event_logger.log("recv", "text", {"text": text})
                else:
                    event_logger.log("recv", "listen" if is_listen else "audio", {"t": round(now, 4)})

                if result.get("end_of_turn"):
                    event_logger.log("recv", "end_of_turn", {"t": round(now, 4)})

                if source.is_done and cue_drained_at is None:
                    cue_drained_at = now
                    event_logger.log("send", "cue_drained", {"t": cue_drained_at})
                if cue_drained_at is not None and now - cue_drained_at >= POST_CUE_OBSERVATION_S:
                    break

                i += 1
                deadline = t0 + i * CHUNK_SEC
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        finally:
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
                cue_duration = len(self.cue_pcm) / 2 / INPUT_SR
                self.simulator_speech_segments.append({
                    "start": round(t_trigger, 4),
                    "end": round(t_trigger + cue_duration, 4),
                    "text": self.reference_context.get("reference_interruption_text", ""),
                    "source": "cue",
                })


def main():
    stop_cli_main(MiniCPMStopClient, MINICPM_SLUG, INPUT_SR, DEFAULT_MODEL, output_sr=OUTPUT_SR)


if __name__ == "__main__":
    main()
