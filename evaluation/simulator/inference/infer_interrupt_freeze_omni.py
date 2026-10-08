#!/usr/bin/env python3
"""
evaluation/simulator/inference/infer_interrupt_freeze_omni.py

Freeze-Omni interrupt_user client. Separate from infer_stop_freeze_omni.py.
Native VAD barge-in (`state["stop_generate"]` on 'sl' while generating) is
kept unchanged. InterruptMonitorController decides when WE stop talking.
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
from infer_stop_freeze_omni import (  # noqa: E402  -- read-only constants/helpers
    FREEZE_OMNI_SLUG, FREEZE_OMNI_REPO_ID, QWEN2_LLM_REPO_ID, INPUT_SR, OUTPUT_SR,
    CHUNK_SEC, CHUNK_BYTES, CODEC_CHUNK_SIZE, CODEC_PADDING_SIZE, DECODER_TOPK,
    MAX_PAST_TOKENS, FREEZE_OMNI_POLL_INTERVAL_S, DEFAULT_SYSTEM_PROMPT, REPO_DIR,
    _f32_to_i16_bytes,
)

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


class InterruptFreezeOmniClient(RealtimeClientBase):
    MODEL_SLUG = FREEZE_OMNI_SLUG
    OUTPUT_SR = OUTPUT_SR
    INPUT_SR = INPUT_SR

    _pipeline = None
    _tts = None
    _vad = None
    _system_role = None
    _loaded_key: Optional[tuple] = None

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
        monitor = InterruptMonitorController(
            recorder=recorder, source=source,
            user_speech_duration=input_duration,
            branch_acknowledged_pcm=self.branch_acknowledged_pcm,
            branch_ignored_pcm=self.branch_ignored_pcm,
            now_fn=now_fn, reference_context=self.reference_context,
            llm_client=self.llm_client, event_logger=event_logger,
            poll_interval_s=FREEZE_OMNI_POLL_INTERVAL_S,
        )
        monitor_task = asyncio.create_task(monitor.run())
        monitor_task.add_done_callback(lambda t: source.finish())

        state = {"is_generate": False, "stop_generate": False, "gen_thread": None}

        def _emit_pcm(arr_f32: np.ndarray) -> None:
            pcm_bytes = _f32_to_i16_bytes(arr_f32)
            loop.call_soon_threadsafe(recorder.queue.put_nowait, pcm_bytes)

        def _emit_text(text: str, at_s: float) -> None:
            self.text_tokens.append((at_s, text))
            loop.call_soon_threadsafe(monitor.on_text, text, at_s)

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

        branch_drained_at: Optional[float] = None
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

            grace_deadline = time.monotonic() + 20.0
            while state["is_generate"] and time.monotonic() < grace_deadline:
                await asyncio.sleep(0.2)
        finally:
            state["stop_generate"] = True
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
    interrupt_cli_main(InterruptFreezeOmniClient, FREEZE_OMNI_SLUG, INPUT_SR, FREEZE_OMNI_REPO_ID,
                       output_sr=OUTPUT_SR)


if __name__ == "__main__":
    main()
