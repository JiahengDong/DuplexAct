#!/usr/bin/env python3
"""
direct_replay/inference/infer_freeze_omni.py
==========================================
VITA-MLLM/Freeze-Omni client. Category B (in-process streaming, no network
server) -- adapts Full-Duplex-Bench v1.0's own inference glue
(`Full-Duplex-Bench/v1_v1.5/model_inference/freeze-omni/backchannel.py`,
`pause_handling.py`, etc.) rather than the official repo's `bin/server.py`
(a Flask-SocketIO server) or `bin/inference.py` (a bare listen-then-speak
batch script with no VAD/duplex behavior at all). FDB's own scripts instead
call Freeze-Omni's official `web/` package (`web.vad.VAD`, `web.pool`,
`web.queue`) directly in-process -- the SAME modules `bin/server.py` uses
internally, just without the Flask-SocketIO transport layer wrapped around
them. We follow that same precedent: import Freeze-Omni's own `models.*` and
`web.*` packages (cloned verbatim, unmodified) and drive them ourselves with
`RealtimeClientBase`'s wall-clock pacing instead of FDB's raw
`threading`+`time.sleep` loop.

Architecture (arXiv:2411.00774): a frozen Qwen2-7B-Instruct LLM backbone with
a trainable streaming speech encoder (chunk-wise, 3-stage trained for
acoustic robustness) and an autoregressive single-codebook speech decoder
(prefix-tuned, TiCodec-based). A classification head on the LLM's last layer
predicts one of five states every chunk step -- this IS the model's native
turn-taking signal, no separate VAD-on-the-model-output needed:
  'sl' (start listen) / 'cl' (continue listen) / 'el' (end listen -- the
  model's OWN "invalid break" prediction) / 'ss' (start speak) / 'cs'
  (continue speak).

IMPORTANT: there are TWO different "state" signals sharing these same five
labels, and they must NOT be conflated:
  - `vad_res["status"]` -- Freeze-Omni's own Silero-VAD wrapper's (`web.vad.
    VAD`) purely ACOUSTIC classification of the current 160ms chunk (is
    there speech here, has it been silent long enough to time out). Gates
    whether/how real audio features get fed to the LLM at all.
  - `outputs["stat"]` -- the model's OWN duplex-head prediction returned
    from `speech_dialogue()`. This is the actual turn-taking decision and is
    NOT set from the VAD's status.
Forcing `outputs["stat"]` to mirror `vad_res["status"]` (as an earlier
version of this file did for the 'cl'/'el' case) will eventually feed
`stat='el'` into `speech_dialogue()`/`recognize()` -- and the vendored
`models/audioLLM.py`'s `recognize()` has no code path that assigns
`inputs_embeds` when `stat=='el'` (only 'sl'/'cl' build it via the encoder+
adapter, 'ss' via the chat-suffix branch), so that call raises
`UnboundLocalError`. FDB's own `backchannel.py` never makes this
substitution either -- see the long comments in the VAD-chunk loop below.

Genuine duplex, not turn-based: `web.vad.VAD.predict()` is called on every
160ms input chunk UNCONDITIONALLY, even while the model is mid-response --
so if the VAD fires 'sl' (user starts talking) while a response is still
being generated/spoken, that is real-time barge-in, exactly mirroring how a
closed-source API server signals mid-response cancellation. We reproduce
FDB's own `force_tts_over` handling: on such a barge-in, the in-flight
generation thread is signalled to stop and `recorder.interrupt()` is called
(clearing any queued-but-unwritten TTS audio) -- the same
`SynchronizedRecorder.interrupt()` used by every closed-source client for
server-side barge-in, so this model's interruption events are recorded in
an identical shape.

Threading model: `models.pipeline.inferencePipeline.speech_dialogue()` and
`models.decoder.llm2tts.llm2TTS.run()` are both synchronous, blocking CUDA
calls -- exactly like FDB's own code, response generation ('ss'/'cs' loop +
TTS decode) runs in a background `threading.Thread` (not `asyncio.to_thread`,
since we need it to run concurrently with -- not sequentially before -- the
main input-feeding loop) so that VAD keeps monitoring incoming audio in real
time while the model is still speaking. The thread pushes decoded PCM chunks
back into the event loop via `loop.call_soon_threadsafe(recorder.queue.put_nowait, ...)`
(the same underlying `asyncio.Queue` `SynchronizedRecorder.add()` uses,
just enqueued directly since `put_nowait` must run on the loop's own thread).

Segmentation: UNLIKE Moshi/BayLing/Voila/Raon, Freeze-Omni's decoded TTS
audio is only ever pushed to the recorder while a real response is being
synthesized -- there is genuine digital silence between utterances (no
continuously-emitted signal to disambiguate), so `SynchronizedRecorder`'s
DEFAULT automatic audio-presence segmentation is already correct here, same
as the closed-source clients. No `set_speaking()` override needed.

Weights: `VITA-MLLM/Freeze-Omni` (speech encoder + TTS decoder + codec) AND
`Qwen/Qwen2-7B-Instruct` (the frozen LLM backbone) -- two separate downloads,
matching the official repo's own two-checkpoint setup.

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
MAX_PAST_TOKENS = 128      # matches the official bin/inference.py's own hard turn-length cap
INPUT_TRAILING_PAD_S = 8.0  # see module docstring's "genuine duplex" note -- room to finish speaking

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
            # Fixed codec_chunk_size/padding for every call (40/10, matching the
            # official non-duplex bin/inference.py's own decoder() defaults) --
            # deliberately not replicating server.json's decoder_first_chunk_size
            # first-pack latency optimization, since that only reduces perceived
            # time-to-first-audio in the live interactive demo and has no effect
            # on the resulting waveform content this benchmark actually scores.
            hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
            cur_text_procced = pipeline.post_process(cur_text)
            token_ids = pipeline.model.tokenizer.encode(cur_text_procced)
            if not token_ids:
                # post_process can strip cur_text down to nothing -- e.g. the
                # final flush after the loop breaks right on top of an
                # incomplete-UTF-8 `continue` (cur_hidden_state already grew
                # via the unconditional append above the \ufffd check, but
                # cur_text never got that chunk's new_text). torch.tensor([])
                # has no elements to infer an int dtype from and defaults to
                # float32, which then fails embed_tokens's Long/Int check
                # (confirmed crash: "Expected tensor for argument #1
                # 'indices' to have one of the following scalar types: Long,
                # Int; but got torch.cuda.FloatTensor"). There is genuinely no
                # text to speak here, so skip this chunk instead of crashing
                # the whole generation thread and losing every already-queued
                # audio segment for the sample.
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
                        # Incomplete UTF-8 sequence -- do NOT advance last_text,
                        # so the next iteration re-slices from the same point
                        # once more tokens resolve it (matches FDB's bare
                        # `continue`, which likewise skips the trailing
                        # `last_text = outputs["text"]` update below).
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
                    # Real-time barge-in: user started talking while the model
                    # was still responding -- signal the generation thread to
                    # stop and discard whatever's queued-but-unwritten, exactly
                    # like a closed-source API's server-side cancel.
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
                    # Mirrors FDB backchannel.py's llm_prefill(is_first_pack=True)
                    # loop over cached history chunks: each call's resulting stat
                    # is forced back to 'cl' (masking any premature 'ss'/'el' the
                    # model might predict while merely replaying already-buffered
                    # history), so only the FINAL call below (the live edge chunk,
                    # is_first_pack=False) is allowed to propagate a real 'ss'.
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
                # Mirrors FDB's llm_prefill(): only re-poll the model if its OWN
                # last-returned `outputs["stat"]` is still 'cl' -- do NOT force
                # `stat` to the VAD's status here. `data["status"]` (this VAD
                # chunk classification) and `outputs["stat"]` (the model's own
                # duplex-head prediction) are two DIFFERENT signals that happen
                # to share label names; conflating them by overwriting one with
                # the other is exactly what used to feed `stat='el'` into
                # `speech_dialogue()`/`recognize()` below, which has no code
                # path that assigns `inputs_embeds` for `stat=='el'` (only
                # 'sl'/'cl' build it via the adapter, 'ss' via the chat-suffix
                # branch) -- an `UnboundLocalError`, confirmed on a live smoke
                # test. FDB never triggers this because it never makes that
                # substitution either.
                if outputs.get("stat") == "cl":
                    def _prefill_cont(outputs=outputs, vad_res=vad_res):
                        return pipeline.speech_dialogue(torch.tensor(vad_res["feature"]), **outputs)

                    outputs = await asyncio.to_thread(_prefill_cont)
                if outputs.get("stat") == "el":
                    # The model's OWN duplex head decided this was an invalid
                    # break (not the VAD) -- matches FDB's "Detect invalid
                    # break" branch: just stop treating this as an open dialog,
                    # take no further model call this tick.
                    vad.in_dialog = False
                    event_logger.log("recv", "invalid_break")

            elif status == "el":
                # Acoustic VAD-level timeout ("no speech long enough"), a
                # purely acoustic signal -- FDB's llm_prefill() never calls
                # speech_dialogue()/recognize() for this case either, it only
                # resets `wakeup_and_vad.in_dialog`. See the long comment in
                # the 'cl' branch above for why calling the model with
                # stat='el' crashes.
                vad.in_dialog = False
                event_logger.log("recv", "vad_timeout")

            if outputs.get("stat") == "ss" and not state["is_generate"]:
                event_logger.log("recv", "response_start")
                gen_outputs = deepcopy(outputs)
                thread = threading.Thread(target=_generate_thread, args=(gen_outputs,), daemon=True)
                state["gen_thread"] = thread
                thread.start()
                # New turn: the main loop's own `outputs` should go back to
                # listening state so the next VAD 'cl' tick doesn't re-trigger
                # generation -- the background thread owns `gen_outputs` from
                # here, feeding itself via its own speech_dialogue(None, ...) loop.
                outputs["stat"] = "sl"

            deadline = t0 + (i + 1) * (CHUNK_MS / 1000.0)
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        # Padded input exhausted -- give any in-flight generation a bounded
        # grace window to finish naturally (it still runs in its own thread
        # and keeps emitting into the recorder via call_soon_threadsafe; the
        # recorder itself is what actually stops writing once run_sample's
        # own wall-clock deadline elapses, same as every other client).
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
