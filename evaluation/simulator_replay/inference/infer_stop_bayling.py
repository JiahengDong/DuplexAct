#!/usr/bin/env python3
"""
BayLing-Models/BayLing-Duplex client for simulator replay, with the user's
interruption injected live by the LLM-driven trigger (see trigger_controller.py).

BayLing-Duplex consumes user audio tokens rather than PCM.
`BayLingDuplex.stream_audio_tokens()` processes tokens block by block and keeps
its KV cache and counters on a `DuplexStreamState`, so calling it again with the
same `state` continues generation. This client feeds one block (x_ratio=10
tokens = 0.8 s of user audio) per call, pulled from a `DynamicTokenSource`
(stop_common.py). Silence filler is one block of zeroed PCM tokenized once
(silence tokens are model-specific) and cycled while no content is queued.

MAX_EPAD_COUNT is 2 because a sample can contain two model turns (the original
response and a post-interruption recovery turn); with 1, `state.stop_requested`
would fire after the first turn and later calls would return nothing.

Standalone usage (single sample):
    python infer_stop_bayling.py --sample_dir <sample dir> \\
        --output_wav out.wav --events_path events.jsonl
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
from trigger_controller import LiveTriggerController, poll_interval_for_native_chunk  # noqa: E402
from stop_common import stop_cli_main, POST_CUE_OBSERVATION_S, DynamicTokenSource  # noqa: E402

BAYLING_SLUG = "bayling-duplex"
DEFAULT_MODEL = "BayLing-Models/BayLing-Duplex"
TOKENIZER_MODEL = "zai-org/glm-4-voice-tokenizer"
DECODER_MODEL = "zai-org/glm-4-voice-decoder"
INTERLEAVE_RATIO = "10:5:10"

INPUT_SR = 16000    # BayLingDuplex.sample_rate
API_SR = 22050      # BayLingDuplex.output_sample_rate (used for OUTPUT_SR)

# BayLing's audio token rate is 12.5 Hz (80 ms per token, as for Moshi's
# Mimi frames); a block of x_ratio=10 tokens is 0.8 s, which is also the
# finest granularity at which new user content can be injected.
BAYLING_NATIVE_TOKEN_SEC = 1.0 / 12.5   # 0.08s per audio token
BAYLING_POLL_INTERVAL_S = poll_interval_for_native_chunk(BAYLING_NATIVE_TOKEN_SEC)  # 1.0s unified user-side poll

MAX_EPAD_COUNT = 2   # see module docstring: original response + recovery turn

REPO_DIR = opensource_root_for(BAYLING_SLUG) / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _f32_to_i16_bytes(wav) -> bytes:
    arr = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class BayLingDuplexStopClient(RealtimeClientBase):
    MODEL_SLUG = BAYLING_SLUG
    OUTPUT_SR = API_SR
    INPUT_SR = INPUT_SR
    # Vocoder output lags the text stream, as in infer_bayling.py. The trigger
    # still sees live text tokens; set_speaking(at_s=now_fn()) keeps speaking
    # onsets on the wall clock.
    CONTIGUOUS_WHILE_SPEAKING = True

    _bd = None
    _model_name_loaded: Optional[str] = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool, cue_pcm: bytes, reference_context: dict, llm_client):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self.cue_pcm = cue_pcm
        self.reference_context = reference_context
        self.llm_client = llm_client
        self.trigger_result: dict = {}
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

        # Tokenize the lead-in and cue up front (only when the cue is fed is
        # decided live), plus one block of silence tokens used as filler.
        lead_in_tokens, cue_tokens, silence_tokens = await asyncio.to_thread(
            lambda: (
                _tokenize(input_pcm),
                _tokenize(np.frombuffer(self.cue_pcm, dtype=np.int16)),
                _tokenize(np.zeros(x_ratio * int(round(INPUT_SR / 12.5)), dtype=np.int16)),
            )
        )
        event_logger.log("send", "tokenized_input", {
            "n_lead_in_tokens": len(lead_in_tokens), "n_cue_tokens": len(cue_tokens),
            "n_silence_tokens": len(silence_tokens),
        })

        state = bd.create_stream_state()

        await start_streaming()
        # Manual segmentation from t=0: <|assistant|>/[EPAD] text-token
        # events, not audio presence, are the model's speaking signal.
        recorder.set_speaking(False, at_s=0.0)

        def now_fn() -> float:
            return time.monotonic() - t0

        t0 = time.monotonic()

        source = DynamicTokenSource(silence_tokens=silence_tokens, initial_tokens=lead_in_tokens)

        controller = LiveTriggerController(
            recorder=recorder,
            source=source,
            cue_pcm=cue_tokens,   # token list, not bytes -- source.append() below is duck-typed to match
            now_fn=now_fn,
            reference_context=self.reference_context,
            client=self.llm_client,
            lead_in_duration_s=input_duration,
            event_logger=event_logger,
            poll_interval_s=BAYLING_POLL_INTERVAL_S,
        )
        trigger_task = asyncio.create_task(controller.run())

        def _on_trigger_done(t: asyncio.Task) -> None:
            source.finish()

        trigger_task.add_done_callback(_on_trigger_done)

        stream_id = str(uuid.uuid4())
        block_idx = 0
        cue_drained_at: Optional[float] = None

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
                            controller.on_text(ev.text, at_s=now)
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

                if source.is_done and cue_drained_at is None:
                    cue_drained_at = now_fn()
                    event_logger.log("send", "cue_drained", {"t": cue_drained_at})
                if cue_drained_at is not None and now_fn() - cue_drained_at >= POST_CUE_OBSERVATION_S:
                    break
                if state.stop_requested:
                    # MAX_EPAD_COUNT reached; further stream_audio_tokens()
                    # calls would return nothing.
                    event_logger.log("recv", "state_stop_requested", {"t": now_fn()})
                    break

                deadline = t0 + block_idx * (x_ratio * BAYLING_NATIVE_TOKEN_SEC)
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
                # self.cue_pcm is int16 at INPUT_SR (it is what gets tokenized;
                # the conversation wav uses a separate output_sr copy).
                cue_duration = len(self.cue_pcm) / 2 / INPUT_SR
                self.simulator_speech_segments.append({
                    "start": round(t_trigger, 4),
                    "end": round(t_trigger + cue_duration, 4),
                    "text": self.reference_context.get("reference_interruption_text", ""),
                    "source": "cue",
                })


async def await_sync_stream_tokens_to_audio(bd, audio_tokens, stream_id, finalize):
    # stream_tokens_to_audio() is synchronous CUDA work, so run it off the
    # event loop.
    return await asyncio.to_thread(bd.stream_tokens_to_audio, audio_tokens, stream_id, finalize)


def main():
    # api_sr=INPUT_SR: cue/lead-in PCM is tokenized at BayLing's input rate.
    # output_sr=API_SR is used only to build the two-channel conversation wav
    # at output.wav's sample rate.
    stop_cli_main(BayLingDuplexStopClient, BAYLING_SLUG, INPUT_SR, DEFAULT_MODEL,
                  output_sr=API_SR)


if __name__ == "__main__":
    main()
