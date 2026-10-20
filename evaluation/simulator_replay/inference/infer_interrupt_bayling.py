#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/infer_interrupt_bayling.py

BayLing-Duplex interrupt_user client. Separate from infer_stop_bayling.py.
Same DynamicTokenSource + stream_audio_tokens(state=) driving; both branch
clips are tokenized up front and passed to InterruptMonitorController as
token lists (duck-typed append/truncate_pending). max_epad_count=3 so a
greeting + take-turn + post-branch reply can all land.
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger  # noqa: E402
from local_common import opensource_root_for  # noqa: E402
from interrupt_controller import InterruptMonitorController  # noqa: E402
from interrupt_user_common import (  # noqa: E402
    interrupt_cli_main, POST_BRANCH_OBSERVATION_S, build_interrupt_simulator_speech_segments,
)
from trigger_controller import poll_interval_for_native_chunk  # noqa: E402
from stop_common import DynamicTokenSource  # noqa: E402

BAYLING_SLUG = "bayling-duplex"
DEFAULT_MODEL = "BayLing-Models/BayLing-Duplex"
TOKENIZER_MODEL = "zai-org/glm-4-voice-tokenizer"
DECODER_MODEL = "zai-org/glm-4-voice-decoder"
INTERLEAVE_RATIO = "10:5:10"
INPUT_SR = 16000
API_SR = 22050
BAYLING_NATIVE_TOKEN_SEC = 1.0 / 12.5
BAYLING_POLL_INTERVAL_S = poll_interval_for_native_chunk(BAYLING_NATIVE_TOKEN_SEC)
MAX_EPAD_COUNT = 3

REPO_DIR = opensource_root_for(BAYLING_SLUG) / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _f32_to_i16_bytes(wav) -> bytes:
    arr = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class InterruptBayLingDuplexClient(RealtimeClientBase):
    MODEL_SLUG = BAYLING_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = INPUT_SR
    CONTIGUOUS_WHILE_SPEAKING = True

    _bd = None
    _model_name_loaded: Optional[str] = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, branch_acknowledged_pcm: bytes, branch_ignored_pcm: bytes,
                 reference_context: dict, llm_client=None):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.branch_acknowledged_pcm = branch_acknowledged_pcm
        self.branch_ignored_pcm = branch_ignored_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.monitor_result: dict = {}
        self.text_tokens: list[tuple[float, str]] = []
        self.simulator_speech_segments: list[dict] = []

    @classmethod
    def _load(cls, model_name: str):
        if cls._bd is not None and cls._model_name_loaded == model_name:
            return cls._bd
        from huggingface_hub import snapshot_download
        from bayling_duplex.duplex import BayLingDuplex
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
        x_ratio = bd.x_ratio

        def _tokenize(pcm_i16: np.ndarray) -> list:
            pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
            wav_tensor = torch.from_numpy(pcm_f32).unsqueeze(0)
            tokens, _ = bd.tokenize_audio((wav_tensor, INPUT_SR))
            return list(tokens)

        user_tokens, ack_tokens, ign_tokens, silence_tokens = await asyncio.to_thread(
            lambda: (
                _tokenize(input_pcm),
                _tokenize(np.frombuffer(self.branch_acknowledged_pcm, dtype=np.int16)),
                _tokenize(np.frombuffer(self.branch_ignored_pcm, dtype=np.int16)),
                _tokenize(np.zeros(x_ratio * int(round(INPUT_SR / 12.5)), dtype=np.int16)),
            )
        )
        event_logger.log("send", "tokenized_input", {
            "n_user_tokens": len(user_tokens),
            "n_ack_tokens": len(ack_tokens),
            "n_ign_tokens": len(ign_tokens),
            "n_silence_tokens": len(silence_tokens),
        })

        state = bd.create_stream_state()
        await start_streaming()
        recorder.set_speaking(False, at_s=0.0)
        t0 = time.monotonic()

        def now_fn() -> float:
            return time.monotonic() - t0

        source = DynamicTokenSource(silence_tokens=silence_tokens, initial_tokens=user_tokens)
        monitor = InterruptMonitorController(
            recorder=recorder, source=source,
            user_speech_duration=input_duration,
            branch_acknowledged_pcm=ack_tokens,
            branch_ignored_pcm=ign_tokens,
            now_fn=now_fn, reference_context=self.reference_context,
            llm_client=self.llm_client, event_logger=event_logger,
            poll_interval_s=BAYLING_POLL_INTERVAL_S,
        )
        monitor_task = asyncio.create_task(monitor.run())
        monitor_task.add_done_callback(lambda t: source.finish())

        stream_id = str(uuid.uuid4())
        block_idx = 0
        branch_drained_at: Optional[float] = None
        try:
            while True:
                block_tokens = source.pull_block(x_ratio)

                def _run_block():
                    events = []
                    for ev in bd.stream_audio_tokens(
                        block_tokens, state=state, temperature=0.8, top_p=0.8,
                        max_epad_count=MAX_EPAD_COUNT,
                    ):
                        events.append(ev)
                    return events

                events = await asyncio.to_thread(_run_block)
                finalize_next_audio = False
                for ev in events:
                    if ev.kind == "text":
                        if ev.assistant_started:
                            recorder.set_speaking(True, at_s=now_fn())
                            event_logger.log("recv", "assistant_started")
                        if ev.assistant_finished:
                            recorder.set_speaking(False, at_s=now_fn())
                            finalize_next_audio = True
                            event_logger.log("recv", "assistant_finished")
                        if ev.text:
                            now = now_fn()
                            self.text_tokens.append((now, ev.text))
                            monitor.on_text(ev.text, at_s=now)
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
                if source.is_done and branch_drained_at is None:
                    branch_drained_at = now_fn()
                    event_logger.log("send", "branch_drained", {"t": branch_drained_at})
                if (branch_drained_at is not None
                        and now_fn() - branch_drained_at >= POST_BRANCH_OBSERVATION_S):
                    break
                if state.stop_requested:
                    event_logger.log("recv", "state_stop_requested", {"t": now_fn()})
                    break
                deadline = t0 + block_idx * (x_ratio * BAYLING_NATIVE_TOKEN_SEC)
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
    interrupt_cli_main(InterruptBayLingDuplexClient, BAYLING_SLUG, INPUT_SR, DEFAULT_MODEL,
                       output_sr=API_SR)


if __name__ == "__main__":
    main()
