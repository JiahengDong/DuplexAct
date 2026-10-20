#!/usr/bin/env python3
"""
evaluation/simulator_replay/inference/infer_stop_freeze_omni.py

VITA-MLLM/Freeze-Omni client for the `stop` Level-2 action,
simulator-driven (live LLM-triggered injection -- see
trigger_controller.py's module docstring for the full architecture
rationale).

Reuses direct_replay/inference/infer_freeze_omni.py's exact architecture
(see that file's own extensive module docstring for the full writeup:
Freeze-Omni's native `web.vad.VAD` chunk classifier feeding a duplex-head
LLM whose OWN `outputs["stat"]` prediction -- 'sl'/'cl'/'el'/'ss'/'cs' --
IS its native turn-taking signal; response generation + TTS decode runs in
a background `threading.Thread` concurrently with the main 160ms
VAD-chunk-feeding loop, exactly like a real duplex server) almost
unchanged. Two real adaptations were needed (this is the "MINOR ADJUST"
flagged in MODEL_COMPATIBILITY.md):

1. Live text: `_generate_thread`'s `cur_text` (accumulated inside the
   background thread from `cur_outputs["text"]` deltas) was already being
   computed for TTS-decode chunking but never surfaced anywhere else. Added
   one `loop.call_soon_threadsafe(controller.on_text, ...)` call right next
   to where `cur_text` is updated (same `call_soon_threadsafe` pattern
   `_emit_pcm` already uses to push audio from that same background thread
   back onto the asyncio loop -- `LiveTriggerController.on_text()` is a
   plain synchronous method, safe to call this way).
2. Dynamic input: `infer_freeze_omni.py`'s sender loop iterates over a
   fixed pre-padded `input_pcm` array of known length; here it instead
   pulls 160ms chunks from a `DynamicPCMSource` until
   `POST_CUE_OBSERVATION_S` has elapsed after the cue drains (same pattern
   as every other client here), keeping `infer_freeze_omni.py`'s own
   post-loop "grace window for in-flight generation to finish" logic
   verbatim afterward.

Freeze-Omni's own native VAD-triggered barge-in handling
(`recorder.interrupt()` + `state["stop_generate"]` on a live 'sl' VAD
event while `state["is_generate"]`) is kept COMPLETELY UNCHANGED here --
when the cue plays while the model is mid-response, this is exactly the
`stop` action's real test: does the model's OWN native interruption
mechanism correctly react to real overlapping audio, not a simulated
approximation of it.

This file is deliberately a SEPARATE module from
direct_replay/inference/infer_freeze_omni.py (never touches it, never
imported by it, loads its own independent copy of the model into this
process) -- see trigger_controller.py's own module docstring for the
"always a separate module" rule.

Standalone usage (single sample, for smoke testing):
    python infer_stop_freeze_omni.py --sample_dir <path to Candor_stop_0038 etc> \\
        --output_wav out.wav --events_path events.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from copy import deepcopy
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
from trigger_controller import LiveTriggerController, poll_interval_for_native_chunk  # noqa: E402
from stop_common import stop_cli_main, POST_CUE_OBSERVATION_S  # noqa: E402

FREEZE_OMNI_SLUG = "freeze-omni"
FREEZE_OMNI_REPO_ID = "VITA-MLLM/Freeze-Omni"
QWEN2_LLM_REPO_ID = "Qwen/Qwen2-7B-Instruct"

INPUT_SR = 16_000
OUTPUT_SR = 24_000
CHUNK_MS = 160
CHUNK_SEC = CHUNK_MS / 1000.0
CHUNK_SAMPLES = int(INPUT_SR * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_SAMPLES * 2   # int16
CODEC_CHUNK_SIZE = 40
CODEC_PADDING_SIZE = 10
DECODER_TOPK = 2
MAX_PAST_TOKENS = 128

FREEZE_OMNI_POLL_INTERVAL_S = poll_interval_for_native_chunk(CHUNK_SEC)  # 1.0s unified user-side poll

DEFAULT_SYSTEM_PROMPT = ("You are a helpful voice assistant. Your answer should be "
                          "coherent, natural, simple, complete. Do not answer too long.")

REPO_DIR = opensource_root_for(FREEZE_OMNI_SLUG) / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _f32_to_i16_bytes(arr: np.ndarray) -> bytes:
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class FreezeOmniStopClient(RealtimeClientBase):
    MODEL_SLUG = FREEZE_OMNI_SLUG
    OUTPUT_SR = OUTPUT_SR
    INPUT_SR = INPUT_SR

    _pipeline = None
    _tts = None
    _vad = None
    _system_role = None
    _loaded_key: Optional[tuple] = None

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
            self._load, 0.8, 20, 0.8, self.system_prompt
        )
        vad.reset_vad()
        outputs = deepcopy(system_role)

        loop = asyncio.get_running_loop()

        await start_streaming()
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
            poll_interval_s=FREEZE_OMNI_POLL_INTERVAL_S,
        )
        trigger_task = asyncio.create_task(controller.run())

        def _on_trigger_done(t: asyncio.Task) -> None:
            source.finish()

        trigger_task.add_done_callback(_on_trigger_done)

        state = {
            "is_generate": False,
            "stop_generate": False,
            "gen_thread": None,
        }

        def _emit_pcm(arr_f32: np.ndarray) -> None:
            pcm_bytes = _f32_to_i16_bytes(arr_f32)
            loop.call_soon_threadsafe(recorder.queue.put_nowait, pcm_bytes)

        def _emit_text(text: str, at_s: float) -> None:
            self.text_tokens.append((at_s, text))
            loop.call_soon_threadsafe(controller.on_text, text, at_s)

        def _decode_and_emit(cur_hidden_state, cur_text) -> None:
            hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
            cur_text_procced = pipeline.post_process(cur_text)
            token_ids = pipeline.model.tokenizer.encode(cur_text_procced)
            if not token_ids:
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
                        continue
                    cur_text += new_text
                    if new_text:
                        _emit_text(new_text, now_fn())
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

        cue_drained_at: Optional[float] = None
        i = 0
        try:
            while True:
                chunk_bytes = source.pull(CHUNK_BYTES)
                chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                def _vad_step(chunk=chunk):
                    return vad.predict(chunk)

                vad_res = await asyncio.to_thread(_vad_step)
                status = vad_res["status"]

                if status == "sl":
                    if state["is_generate"]:
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
                    if outputs.get("stat") == "cl":
                        def _prefill_cont(outputs=outputs, vad_res=vad_res):
                            return pipeline.speech_dialogue(torch.tensor(vad_res["feature"]), **outputs)

                        outputs = await asyncio.to_thread(_prefill_cont)
                    if outputs.get("stat") == "el":
                        vad.in_dialog = False
                        event_logger.log("recv", "invalid_break")

                elif status == "el":
                    vad.in_dialog = False
                    event_logger.log("recv", "vad_timeout")

                if outputs.get("stat") == "ss" and not state["is_generate"]:
                    event_logger.log("recv", "response_start")
                    gen_outputs = deepcopy(outputs)
                    thread = threading.Thread(target=_generate_thread, args=(gen_outputs,), daemon=True)
                    state["gen_thread"] = thread
                    thread.start()
                    outputs["stat"] = "sl"

                now = now_fn()
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

            # Same bounded grace window as infer_freeze_omni.py: let any
            # in-flight generation finish naturally before ending the sample.
            grace_deadline = time.monotonic() + 20.0
            while state["is_generate"] and time.monotonic() < grace_deadline:
                await asyncio.sleep(0.2)
        finally:
            state["stop_generate"] = True
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
    stop_cli_main(FreezeOmniStopClient, FREEZE_OMNI_SLUG, INPUT_SR, FREEZE_OMNI_REPO_ID, output_sr=OUTPUT_SR)


if __name__ == "__main__":
    main()
