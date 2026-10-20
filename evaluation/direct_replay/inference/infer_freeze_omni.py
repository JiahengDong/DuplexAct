#!/usr/bin/env python3
"""
VITA-MLLM/Freeze-Omni client (in-process streaming, no network server).
Follows Full-Duplex-Bench v1.0's inference scripts: Freeze-Omni's own
`models.*` and `web.*` packages (`web.vad.VAD` etc.) are imported from the
cloned repo and driven in-process with `RealtimeClientBase`'s wall-clock
pacing.

Architecture (arXiv:2411.00774): frozen Qwen2-7B-Instruct backbone, a
streaming speech encoder, and an autoregressive TiCodec-based speech decoder.
A classification head predicts one of five states each chunk: 'sl' (start
listen), 'cl' (continue listen), 'el' (end listen, i.e. invalid break),
'ss' (start speak), 'cs' (continue speak).

Two signals share these labels and must not be conflated:
  - `vad_res["status"]`: the Silero-VAD wrapper's acoustic classification of
    the current 160ms chunk; gates whether audio features go to the LLM.
  - `outputs["stat"]`: the model's own duplex-head prediction from
    `speech_dialogue()`, i.e. the turn-taking decision.
`speech_dialogue()` must never be called with stat='el': `recognize()` has no
branch that builds `inputs_embeds` for it and raises `UnboundLocalError`.

VAD runs on every 160ms chunk, including while the model is speaking; a VAD
'sl' during a response is a barge-in, which stops the generation thread and
calls `recorder.interrupt()` (FDB's `force_tts_over` behaviour).

`speech_dialogue()` and `llm2TTS.run()` are blocking CUDA calls, so response
generation runs in a background `threading.Thread` concurrently with the
input loop, pushing PCM into the recorder via `loop.call_soon_threadsafe`.

TTS audio is only emitted during a response (digital silence in between), so
`SynchronizedRecorder`'s default audio-presence segmentation is used.

Weights: `VITA-MLLM/Freeze-Omni` (encoder, decoder, codec) and
`Qwen/Qwen2-7B-Instruct` (LLM backbone).

Standalone usage (single sample, for debugging):
    python infer_freeze_omni.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from realtime_common import RealtimeClientBase, SynchronizedRecorder, EventLogger
from local_common import opensource_root_for


FREEZE_OMNI_SLUG = "freeze-omni"
FREEZE_OMNI_REPO_ID = "VITA-MLLM/Freeze-Omni"
QWEN2_LLM_REPO_ID = "Qwen/Qwen2-7B-Instruct"

INPUT_SR = 16_000          # Freeze-Omni's own fbank front-end sample rate
OUTPUT_SR = 24_000         # TTS codec output sample rate
CHUNK_MS = 160             # VAD.get_chunk_size() = frame_shift(160) * chunk_size(16) = 2560 samples @16k
CHUNK_SAMPLES = int(INPUT_SR * CHUNK_MS / 1000)
CODEC_CHUNK_SIZE = 40
CODEC_PADDING_SIZE = 10
DECODER_TOPK = 2
MAX_PAST_TOKENS = 128      # turn-length cap from the official bin/inference.py
INPUT_TRAILING_PAD_S = 8.0  # trailing silence so the model has room to finish speaking

REPO_DIR = opensource_root_for(FREEZE_OMNI_SLUG) / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _f32_to_i16_bytes(arr: np.ndarray) -> bytes:
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class FreezeOmniClient(RealtimeClientBase):
    MODEL_SLUG = FREEZE_OMNI_SLUG
    OUTPUT_SR  = OUTPUT_SR
    INPUT_SR   = INPUT_SR

    _pipeline = None
    _tts = None
    _vad = None
    _system_role = None
    _loaded_key: Optional[tuple] = None

    @classmethod
    def _load(cls, top_p: float, top_k: int, temperature: float, system_prompt: str):
        key = (top_p, top_k, temperature, system_prompt)
        if cls._pipeline is not None and cls._loaded_key == key:
            return cls._pipeline, cls._tts, cls._vad, cls._system_role

        from huggingface_hub import snapshot_download
        from models.pipeline import inferencePipeline
        from models.decoder.llm2tts import llm2TTS
        from web.vad import VAD

        freeze_omni_dir = Path(snapshot_download(FREEZE_OMNI_REPO_ID)) / "checkpoints"
        llm_dir = Path(snapshot_download(QWEN2_LLM_REPO_ID))

        configs = argparse.Namespace(
            model_path=str(freeze_omni_dir), llm_path=str(llm_dir),
            top_p=top_p, top_k=top_k, temperature=temperature,
        )
        pipeline = inferencePipeline(configs)
        tts = llm2TTS(str(freeze_omni_dir))
        vad = VAD()
        system_role = pipeline.speech_dialogue(None, stat="pre", role=system_prompt)

        cls._pipeline, cls._tts, cls._vad, cls._system_role = pipeline, tts, vad, system_role
        cls._loaded_key = key
        return pipeline, tts, vad, system_role

    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder, event_logger: EventLogger,
                                   start_streaming) -> None:
        import torch

        pipeline, tts, vad, system_role = await asyncio.to_thread(
            self._load, 0.8, 20, 0.8, self.system_prompt or "You are a helpful voice assistant."
        )
        vad.reset_vad()
        outputs = deepcopy(system_role)

        loop = asyncio.get_running_loop()

        pcm_f32 = input_pcm.astype(np.float32) / 32768.0
        pad_samples = int(round(INPUT_TRAILING_PAD_S * INPUT_SR))
        padded = np.concatenate([pcm_f32, np.zeros(pad_samples, dtype=np.float32)])
        n_chunks = int(np.ceil(len(padded) / CHUNK_SAMPLES))
        padded = np.concatenate([padded, np.zeros(n_chunks * CHUNK_SAMPLES - len(padded), dtype=np.float32)])

        state = {
            "is_generate": False,
            "stop_generate": False,
            "gen_thread": None,
        }

        def _emit_pcm(arr_f32: np.ndarray) -> None:
            pcm_bytes = _f32_to_i16_bytes(arr_f32)
            loop.call_soon_threadsafe(recorder.queue.put_nowait, pcm_bytes)

        def _decode_and_emit(cur_hidden_state, cur_text) -> None:
            # Fixed codec chunk/padding (40/10, bin/inference.py's decoder()
            # defaults); server.json's smaller first-chunk size only affects
            # demo latency, not the waveform.
            hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
            cur_text_procced = pipeline.post_process(cur_text)
            token_ids = pipeline.model.tokenizer.encode(cur_text_procced)
            if not token_ids:
                # post_process can strip cur_text to nothing (e.g. a final
                # flush right after an incomplete-UTF-8 chunk). There is no
                # text to speak, so skip rather than embed an empty tensor.
                return
            embeddings = pipeline.model.llm_decoder.model.embed_tokens(
                torch.tensor(token_ids, dtype=torch.long).cuda()
            )
            for seg in tts.run(
                embeddings.reshape(-1, 896).unsqueeze(0), DECODER_TOPK,
                hidden_state_output.reshape(-1, 896).unsqueeze(0),
                CODEC_CHUNK_SIZE, CODEC_PADDING_SIZE,
            ):
                if state["stop_generate"]:
                    break
                _emit_pcm(seg.squeeze().float().cpu().numpy())

        def _generate_thread(initial_outputs: dict) -> None:
            """Mirrors FDB backchannel.py's generate(): loop speech_dialogue(None, ...)
            until stat leaves 'cs', decoding+emitting TTS audio incrementally."""
            state["is_generate"] = True
            try:
                cur_outputs = pipeline.speech_dialogue(None, **initial_outputs)
                cur_hidden_state = [cur_outputs["hidden_state"]]
                cur_text = ""
                last_text = ""
                suffix_full = ('。', '：', '？', '！', '.', ':', '?', '!', '\n')
                suffix_first = (',', '，') + suffix_full
                n_decoded_chunks = 0
                while True:
                    if state["stop_generate"]:
                        break
                    if len(cur_outputs.get("past_tokens", [])) > MAX_PAST_TOKENS:
                        break
                    del cur_outputs["text"]
                    del cur_outputs["hidden_state"]
                    cur_outputs = pipeline.speech_dialogue(None, **cur_outputs)
                    if cur_outputs["stat"] != "cs":
                        break
                    cur_hidden_state.append(cur_outputs["hidden_state"])
                    new_text = cur_outputs["text"][len(last_text):]
                    if "\ufffd" in new_text:
                        # Incomplete UTF-8 sequence: leave last_text unchanged so
                        # the next iteration re-slices once more tokens resolve it.
                        continue
                    cur_text += new_text
                    suffix_list = suffix_first if (n_decoded_chunks == 0 or len(cur_hidden_state) >= 20) else suffix_full
                    ends_ok = new_text.endswith(suffix_list) and len(cur_hidden_state) >= 4
                    if ends_ok and new_text.endswith('.') and last_text and last_text[-1].isdigit():
                        ends_ok = False
                    if ends_ok and cur_hidden_state:
                        _decode_and_emit(cur_hidden_state, cur_text)
                        n_decoded_chunks += 1
                        cur_hidden_state = []
                        cur_text = ""
                    last_text = cur_outputs["text"]
                if cur_hidden_state and not state["stop_generate"]:
                    _decode_and_emit(cur_hidden_state, cur_text)
            finally:
                state["is_generate"] = False
                state["stop_generate"] = False

        await start_streaming()
        t0 = time.monotonic()

        for i in range(n_chunks):
            chunk = padded[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]

            def _vad_step(chunk=chunk):
                return vad.predict(chunk)

            vad_res = await asyncio.to_thread(_vad_step)
            status = vad_res["status"]

            if status == "sl":
                if state["is_generate"]:
                    # Barge-in: user started talking while the model was still
                    # responding. Stop generation and drop queued audio.
                    state["stop_generate"] = True
                    recorder.interrupt("freeze_omni_vad_barge_in")
                    event_logger.log("recv", "barge_in_detected")

                fresh = deepcopy(outputs)
                fresh["adapter_cache"] = None
                fresh["encoder_cache"] = None
                fresh["pe_index"] = 0
                fresh["stat"] = "sl"
                fresh["last_id"] = None
                fresh.pop("text", None)
                fresh.pop("hidden_state", None)

                def _prefill_start(fresh=fresh, vad_res=vad_res):
                    # As in FDB's llm_prefill(is_first_pack=True): replaying
                    # buffered history chunks forces stat back to 'cl', so only
                    # the final live chunk can produce a real 'ss'.
                    out = fresh
                    for j, feat in enumerate(vad_res["feature_last_chunk"]):
                        if j == 0 or out["stat"] == "cl":
                            out = pipeline.speech_dialogue(torch.tensor(feat), **out)
                        out["stat"] = "cl"
                    if out["stat"] == "cl":
                        out = pipeline.speech_dialogue(torch.tensor(vad_res["feature"]), **out)
                    return out

                outputs = await asyncio.to_thread(_prefill_start)
                event_logger.log("send", "vad_speech_start")

            elif status == "cl":
                # Only feed the model if its own last `outputs["stat"]` is still
                # 'cl'. Do not copy the VAD status into `stat` (see module
                # docstring: stat='el' crashes recognize()).
                if outputs.get("stat") == "cl":
                    def _prefill_cont(outputs=outputs, vad_res=vad_res):
                        return pipeline.speech_dialogue(torch.tensor(vad_res["feature"]), **outputs)

                    outputs = await asyncio.to_thread(_prefill_cont)
                if outputs.get("stat") == "el":
                    # The model's duplex head predicted an invalid break: close
                    # the dialog without another model call this tick.
                    vad.in_dialog = False
                    event_logger.log("recv", "invalid_break")

            elif status == "el":
                # Acoustic VAD timeout: only reset in_dialog; the model is not
                # called with stat='el'.
                vad.in_dialog = False
                event_logger.log("recv", "vad_timeout")

            if outputs.get("stat") == "ss" and not state["is_generate"]:
                event_logger.log("recv", "response_start")
                gen_outputs = deepcopy(outputs)
                thread = threading.Thread(target=_generate_thread, args=(gen_outputs,), daemon=True)
                state["gen_thread"] = thread
                thread.start()
                # The generation thread now owns `gen_outputs`; return the main
                # loop to listening so the next tick does not re-trigger it.
                outputs["stat"] = "sl"

            deadline = t0 + (i + 1) * (CHUNK_MS / 1000.0)
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        # Input exhausted: give in-flight generation a bounded grace window.
        # The recorder still stops at run_sample's wall-clock deadline.
        grace_deadline = time.monotonic() + 20.0
        while state["is_generate"] and time.monotonic() < grace_deadline:
            await asyncio.sleep(0.2)


# ── Standalone single-sample CLI (debugging only) ────────────────────────────


async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = FreezeOmniClient(
        api_key="",
        model_name=args.model,
        system_prompt=args.system_prompt,
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
    parser.add_argument("--model", type=str, default=FREEZE_OMNI_REPO_ID)
    parser.add_argument("--system_prompt", type=str,
                         default="You are a helpful voice assistant. Your answer should be "
                                 "coherent, natural, simple, complete. Do not answer too long.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
