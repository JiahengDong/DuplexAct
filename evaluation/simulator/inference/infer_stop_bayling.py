#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_stop_bayling.py

BayLing-Models/BayLing-Duplex client for the `stop` Level-2 action,
simulator-driven (live LLM-triggered injection -- see
trigger_controller.py's module docstring for the full architecture
rationale).

Unlike moshi/flm-audio (a WebSocket wire protocol we can feed byte-by-byte
via DynamicPCMSource) or the GA-realtime-schema clients (JSON+base64 over a
websocket, same DynamicPCMSource), BayLing-Duplex's own Python API
(BayLingDuplex.stream_audio_tokens(), see infer_bayling.py's own module
docstring) consumes pre-tokenized user-AUDIO TOKENS, not raw PCM, and its
direct_replay client tokenizes the ENTIRE (padded) input clip up front in
one `tokenize_audio()` call before streaming begins. That upfront-
tokenize-the-whole-clip step is just how infer_bayling.py happens to use
it, though -- NOT a hard requirement of the underlying API: reading
bayling_duplex/duplex.py directly confirms `stream_audio_tokens()` is a
真正 per-BLOCK-incremental generator (`for start in range(0, len(tokens),
x_ratio): ...`) that carries its KV cache / position / assistant-count /
epad-count forward on a mutable `DuplexStreamState` object passed in via
`state=`. Calling `stream_audio_tokens(new_tokens, state=state)` again
after a previous call already advanced that same `state` object simply
CONTINUES generation from exactly where it left off (confirmed at
duplex.py: the function's own first line is `if state.stop_requested:
return`, i.e. it explicitly supports being re-invoked on the same state --
this isn't a hack, it's the documented mechanism `generate_from_audio_
tokens()` itself uses internally, just with a single upfront token list
instead of several later-decided ones).

So this client drives the model ONE BLOCK (x_ratio=10 tokens = 0.8s of
user audio, BayLing's own native batching granularity -- see module
constants below) at a time, calling `stream_audio_tokens()` fresh each
time with exactly one block's worth of tokens and the SAME persistent
`state`, pulling those tokens from a `DynamicTokenSource` (stop_common.py)
instead of a single upfront array -- the token-domain equivalent of how
infer_stop_moshi.py pulls PCM bytes from a DynamicPCMSource. Silence
filler (for whenever the source has no real content queued yet -- waiting
for the trigger, or the post-cue observation window) is pre-tokenized
ONCE via `tokenize_audio()` on a real zeroed-PCM block (token values are
model-specific, unlike raw-PCM digital silence, so this can't be
synthesized generically) and cycled by DynamicTokenSource as needed --
this also means, unlike infer_bayling.py's own fixed INPUT_TRAILING_PAD_S
(8.0s, a bounded look-ahead sized to *usually* be enough), our driving
loop can extend the "listening to silence" window indefinitely for as
long as our own POST_CUE_OBSERVATION_S cutoff requires, not just a fixed
lookahead.

max_epad_count: infer_bayling.py uses 1 ("matches this benchmark's
per-instance expectation of at most one model turn per sample"). We use
2 here, since `stop` samples genuinely expect up to TWO model turns (the
original pre-interruption response, analogous to T2, and a post-
interruption recovery turn, analogous to T4) within one state -- 1 would
make `state.stop_requested` fire after the FIRST utterance's own EPAD,
silently dropping every subsequent call (`if state.stop_requested:
return`) before the interruption/recovery turn ever had a chance to run.

This file is deliberately a SEPARATE module from
direct_replay/inference/infer_bayling.py (never touches it, never imported
by it, loads its own independent copy of the model into this process) --
see trigger_controller.py's own module docstring for the "always a
separate module" rule.

Standalone usage (single sample, for smoke testing):
    python infer_stop_bayling.py --sample_dir <path to Candor_stop_0038 etc> \\
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

DIRECT_REPLAY_INFERENCE = Path(__file__).resolve().parents[2] / "direct_replay" / "inference"
sys.path.insert(0, str(DIRECT_REPLAY_INFERENCE))
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

# BayLing's underlying audio TOKEN rate is 12.5Hz (matches Moshi/FLM-
# Audio's own Mimi frame rate exactly -- see module docstring's "Whisper-
# style VQ encoder, 12.5Hz" and duplex.py's own `time_offset = x_ratio /
# 12.5`); x_ratio=10 tokens/block = 0.8s is just how many of those 80ms
# tokens BayLing's own generation loop happens to batch per Python-level
# step, not a different underlying granularity. So the SAME
# poll_interval_for_native_chunk(0.08) derivation used for Moshi/FLM-Audio
# applies here too -- and it also happens to equal the block size (0.8s),
# which is convenient since a block boundary is the finest granularity we
# can inject new content at anyway.
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
    # Same vocoder-lag fix as infer_bayling.py. Trigger still sees live
    # text tokens; set_speaking(at_s=now_fn()) keeps Immediate on wall-clock.
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

        # Tokenize the two known clips up front (their CONTENT is fully
        # known -- only WHEN to feed cue_tokens into the running generator
        # is decided live). Also precompute one block's worth of silence
        # tokens ONCE, to use as filler for DynamicTokenSource whenever
        # there's no real content queued yet.
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
        # Manual segmentation from t=0 -- <|assistant|>/[EPAD] text-token
        # events (not audio presence) are the model's own speaking signal,
        # same reasoning as infer_bayling.py / Moshi's continuous-signal
        # recorder.
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
                    # Model itself signaled it's done for good (hit
                    # MAX_EPAD_COUNT) -- natural end, no point calling
                    # stream_audio_tokens() again (it would just return
                    # immediately empty every time per its own top-of-
                    # function guard).
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
                # self.cue_pcm is raw int16 bytes resampled to INPUT_SR (16kHz)
                # by run_stop_sample_cli (it's what gets fed to bd.tokenize_
                # audio(), not what's mixed into the conversation wav -- that
                # uses a SEPARATE output_sr-resampled copy, see stop_common.py's
                # run_stop_sample_cli docstring) -- so duration must divide by
                # INPUT_SR here, not API_SR/output_sr.
                cue_duration = len(self.cue_pcm) / 2 / INPUT_SR
                self.simulator_speech_segments.append({
                    "start": round(t_trigger, 4),
                    "end": round(t_trigger + cue_duration, 4),
                    "text": self.reference_context.get("reference_interruption_text", ""),
                    "source": "cue",
                })


async def await_sync_stream_tokens_to_audio(bd, audio_tokens, stream_id, finalize):
    # stream_tokens_to_audio() is itself @torch.no_grad() synchronous CUDA
    # work (CosyVoice-style streaming flow-matching + HiFT decode, same as
    # infer_bayling.py's own usage) -- run off the event loop exactly like
    # the tokenize/generate calls above.
    return await asyncio.to_thread(bd.stream_tokens_to_audio, audio_tokens, stream_id, finalize)


def main():
    # api_sr=INPUT_SR (16kHz): cue_pcm/lead_in_pcm fed to the constructor
    # (and, inside _connect_and_stream, into bd.tokenize_audio()) must be at
    # BayLing's own INPUT rate. output_sr=API_SR (22050Hz, BayLing's OUTPUT
    # rate) is used separately just for building the two-channel
    # conversation wav so it aligns with output.wav's own sample rate --
    # see stop_common.py's run_stop_sample_cli docstring.
    stop_cli_main(BayLingDuplexStopClient, BAYLING_SLUG, INPUT_SR, DEFAULT_MODEL,
                  output_sr=API_SR)


if __name__ == "__main__":
    main()
