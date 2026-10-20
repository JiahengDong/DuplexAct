#!/usr/bin/env python3
"""
direct_replay/inference/infer_voila.py
==========================================
maitrix-org Voila-Autonomous-preview client. Category C (batch-replay, no
incremental streaming API at all) -- see `BatchReplayClientBase` in
local_common.py for the design rationale and precedent (even Voila's own
authors only ever exercise this checkpoint via one batch `run_generate()`
call over a full pre-recorded clip; their official live demo,
gradio_demo.py, doesn't use the autonomous checkpoint at all).

Segmentation is NOT a crude energy-threshold fallback here, though -- it's
an EXACT reconstruction of the model's own internal `speaking` boolean
(model.py's `run_generate`, lines ~1290-1345), obtained by passing a custom
`streamer` object into the *public, already-documented* `streamer=` kwarg
of `run_generate()`. That parameter receives the model's per-step output
token (`next_tokens.cpu()`, shape [1, 1, num_codebooks]) once per
autoregressive step, before we ever see the final result. Codebook 0 of
that tensor equals `ASSISTANT_TOKEN_ID` on the exact step `speaking` flips
True, and `eos_token_id` on the exact step it flips back False (mirroring
model.py's own `if voila_head_pred == 1 and not speaking: ... speaking =
True` / `elif next_tokens[0] == eos_token_id: speaking = False`) -- so
reconstructing those two transitions from the streamer callback gives the
IDENTICAL boolean sequence the model computed internally, not an
approximation, and does not require touching model.py's generation loop
itself (streamer.put() is a supported extension point, we just have to
write our own thin eval_model()-equivalent glue -- infer.py's own
`eval_model()` is exactly this kind of 80-line glue layer already, just
without a streamer hook wired in).

Frame timeline: this model is a genuine dual-stream (user + assistant)
full-duplex architecture -- every autoregressive step where the model's
own text-level prediction resolves to the generic `<au_token>` audio
placeholder (`AUDIO_MIN_TOKEN_ID <= tok0 < AUDIO_MIN_TOKEN_ID +
codebook_size` after the placeholder is filled with a real code) both (a)
advances `input_generator` by exactly one real 40ms user-audio frame and
(b) counts as one 40ms tick on the shared conversation clock; the rare
turn-boundary marker steps (the literal ASSISTANT_TOKEN_ID / eos_token_id
insertions) do neither, so we track a separate `frame_idx` counter (audio
frames consumed) rather than raw step count for converting to seconds --
using raw step count would drift the reported segment times later than
reality by ~1 frame (40ms) per turn boundary, since marker steps don't
represent elapsed conversation time. frame_rate = 25 Hz is Voila-Tokenizer's
own EnCodec config (sampling_rate=16000, hop = 8*5*4*4 = 640 ->
16000/640 = 25).

Because generation is a single batch call with no live wall-clock pacing,
there's no way to know in advance how much *conversation time* the
returned audio spans until decoding finishes -- so we feed the model
padded input (original clip + INPUT_TRAILING_PAD_S of trailing silence,
via voila_input_format's `audio={"array":..., "sr":...}` in-memory path,
no temp file needed) so it has room to finish speaking after the user's
turn ends, exactly mirroring how the real-time clients let a model keep
talking past input_duration for as long as it needs (see
BatchReplayClientBase.run_sample's pad-but-never-truncate note).

Voice cloning reference: `examples/test1.mp3`, bundled in the model's own
GitHub repo and used as the CLI script's own `--ref-audio` default -- same
precedent-matching principle as MiniCPM-o 4.5's HT_ref_audio.wav.

Standalone usage (single sample, for debugging):
    python infer_voila.py --input_wav path/to/input.wav \\
        --output_wav path/to/output.wav --events_path path/to/events.jsonl \\
        --input_duration 33.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from local_common import BatchReplayClientBase, opensource_root_for

VOILA_SLUG = "voila-autonomous"
DEFAULT_MODEL = "maitrix-org/Voila-autonomous-preview"
TOKENIZER_MODEL = "maitrix-org/Voila-Tokenizer"

AUDIO_SR = 16_000          # Voila's own AUDIO_SR (tokenize_func.py) and Voila-Tokenizer's sampling_rate
FRAME_RATE_HZ = 25.0       # Voila-Tokenizer EnCodec: 16000 / (8*5*4*4) = 25
INPUT_TRAILING_PAD_S = 8.0 # silence appended after the real clip so the model has room to finish speaking

REPO_ROOT = Path(__file__).resolve().parents[3]
VOILA_REPO_DIR = opensource_root_for(VOILA_SLUG) / "repo"
REF_AUDIO_PATH = VOILA_REPO_DIR / "examples" / "test1.mp3"

if str(VOILA_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(VOILA_REPO_DIR))


class _SpeakingTracker:
    """
    Passed as `streamer=` into VoilaAutonomousModel.run_generate() -- see
    module docstring for exactly which two token-id transitions this
    reconstructs and why they exactly match the model's own internal
    `speaking` state.
    """

    def __init__(self, assistant_token_id: int, eos_token_id: int,
                 audio_min_token_id: int, codebook_size: int, frame_rate: float = FRAME_RATE_HZ):
        self.assistant_token_id = assistant_token_id
        self.eos_token_id = eos_token_id
        self.audio_min_token_id = audio_min_token_id
        self.audio_max_token_id = audio_min_token_id + codebook_size
        self.frame_rate = frame_rate

        self.speaking = False
        self.frame_idx = 0
        self.segments: list[tuple[float, float]] = []
        self._open_start: Optional[float] = None

    def put(self, value) -> None:
        tok0 = int(value[0, 0, 0].item())
        is_audio_frame = self.audio_min_token_id <= tok0 < self.audio_max_token_id
        t = self.frame_idx / self.frame_rate

        if tok0 == self.assistant_token_id and not self.speaking:
            self.speaking = True
            self._open_start = t
        elif tok0 == self.eos_token_id and self.speaking:
            self.speaking = False
            if self._open_start is not None and t > self._open_start:
                self.segments.append((round(self._open_start, 4), round(t, 4)))
            self._open_start = None

        if is_audio_frame:
            self.frame_idx += 1

    def end(self) -> None:
        if self.speaking and self._open_start is not None:
            t = self.frame_idx / self.frame_rate
            if t > self._open_start:
                self.segments.append((round(self._open_start, 4), round(t, 4)))
            self._open_start = None
            self.speaking = False


def _eval_model_autonomous_streaming(model, tokenizer, tokenizer_voila, history,
                                      ref_embs, ref_embs_mask, max_new_tokens: int,
                                      streamer: _SpeakingTracker):
    """
    Thin reimplementation of infer.py's `eval_model()`, specialized to
    model_type == "autonomous" / task_type == "chat_aiao_auto", with one
    addition: `streamer=streamer` wired into `model.run_generate(...)` (the
    upstream `eval_model()` doesn't expose this kwarg at all). Everything
    else -- tensor construction, decode -- is copied verbatim from
    infer.py so results match what the official script would produce.
    """
    import torch
    from tokenize_func import (
        voila_input_format,
        AUDIO_TOKEN_FORMAT,
        DEFAULT_AUDIO_TOKEN,
        DEFAULT_ASSISTANT_TOKEN,
    )

    num_codebooks = model.config.num_codebooks
    codebook_size = model.config.codebook_size

    AUDIO_MIN_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(0))
    AUDIO_MAX_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(codebook_size * num_codebooks - 1))
    AUDIO_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_TOKEN)
    ASSISTANT_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_ASSISTANT_TOKEN)

    data_cfg = {
        "input_type": "autonomous",
        "task_type": "chat_aiao_auto",
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
    }
    input_ids, _, _, streaming_user_input_audio_tokens = voila_input_format(
        history, tokenizer, tokenizer_voila, data_cfg
    )

    def get_input_generator(all_tokens):
        for i in range(len(all_tokens[0])):
            yield all_tokens[:, i]

    input_generator = get_input_generator(torch.as_tensor(streaming_user_input_audio_tokens).cuda())
    input_ids = [torch.as_tensor([inp]).transpose(1, 2).cuda() for inp in input_ids]
    input_ids = torch.cat(input_ids, dim=2)

    gen_params = {
        "input_ids": input_ids,
        "ref_embs": ref_embs,
        "ref_embs_mask": ref_embs_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "llm_audio_token_id": AUDIO_TOKEN_ID,
        "min_audio_token_id": AUDIO_MIN_TOKEN_ID,
        "temperature": 0.2,
        "top_k": 50,
        "audio_temperature": 0.8,
        "audio_top_k": 50,
        "input_generator": input_generator,
        "llm_assistant_token_id": ASSISTANT_TOKEN_ID,
        "streamer": streamer,
    }

    with torch.inference_mode():
        outputs = model.run_generate(**gen_params)
        outputs = outputs.chunk(2, dim=2)[1]   # assistant channel only
        outputs = outputs[0].cpu().tolist()

        predict_outputs = outputs[input_ids.shape[1]:]
        audio_outputs = [[] for _ in range(num_codebooks)]
        for item in predict_outputs:
            if AUDIO_MIN_TOKEN_ID <= item[0] <= AUDIO_MAX_TOKEN_ID:
                for n, at in enumerate(item):
                    audio_outputs[n].append((at - AUDIO_MIN_TOKEN_ID) % codebook_size)

        audio_values = tokenizer_voila.decode(torch.tensor(audio_outputs).cuda())
        return audio_values.detach().cpu().numpy()


class VoilaAutonomousClient(BatchReplayClientBase):
    MODEL_SLUG = VOILA_SLUG
    OUTPUT_SR  = AUDIO_SR
    INPUT_SR   = AUDIO_SR

    _model = None
    _tokenizer = None
    _tokenizer_voila = None
    _ref_embs = None
    _ref_embs_mask = None
    _model_name_loaded: str | None = None

    def __init__(self, api_key: str, model_name: str, system_prompt: str, verbose: bool = False):
        super().__init__(api_key, model_name, system_prompt, verbose)
        self._last_segments: Optional[list[tuple[float, float]]] = None

    @classmethod
    def _load(cls, model_name: str):
        if cls._model is not None and cls._model_name_loaded == model_name:
            return cls._model, cls._tokenizer, cls._tokenizer_voila, cls._ref_embs, cls._ref_embs_mask

        import torch
        import torchaudio
        from transformers import AutoTokenizer
        from model import VoilaAutonomousModel
        from voila_tokenizer import VoilaTokenizer
        from spkr import SpeakerEmbedding
        from infer import disable_torch_init

        # Reimplements infer.py's load_model() with attn_implementation="sdpa"
        # instead of use_flash_attention_2=True -- avoids a from-source
        # flash-attn compile in this isolated venv (same tradeoff already made
        # for MiniCPM-o 4.5, see infer_minicpm_o.py); sdpa is a standard,
        # always-available HF attention backend, just somewhat slower.
        disable_torch_init()
        model = VoilaAutonomousModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_cache=True,
        )
        model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_voila = VoilaTokenizer(model_path=TOKENIZER_MODEL, device="cuda")

        spkr_model = SpeakerEmbedding(device="cuda")
        wav, sr = torchaudio.load(str(REF_AUDIO_PATH))
        ref_embs = spkr_model(wav, sr)
        ref_embs_mask = torch.tensor([1]).cuda()

        cls._model, cls._tokenizer, cls._tokenizer_voila = model, tokenizer, tokenizer_voila
        cls._ref_embs, cls._ref_embs_mask = ref_embs, ref_embs_mask
        cls._model_name_loaded = model_name
        return model, tokenizer, tokenizer_voila, ref_embs, ref_embs_mask

    def _native_segments(self):
        return self._last_segments

    async def _generate_batch(self, input_pcm: np.ndarray, input_duration_s: float) -> np.ndarray:
        model, tokenizer, tokenizer_voila, ref_embs, ref_embs_mask = await asyncio.to_thread(
            self._load, self.model_name
        )

        audio_f32 = input_pcm.astype(np.float32) / 32768.0
        pad_samples = int(round(INPUT_TRAILING_PAD_S * AUDIO_SR))
        padded = np.concatenate([audio_f32, np.zeros(pad_samples, dtype=np.float32)])

        history = {
            "instruction": self.system_prompt or "",
            "conversations": [
                {"from": "user", "audio": {"array": padded, "sr": AUDIO_SR}},
                {"from": "assistant"},
            ],
        }

        padded_duration_s = len(padded) / AUDIO_SR
        # Generous margin: turn-boundary marker steps (start/eos) don't
        # advance the audio-frame clock but DO consume a generation step,
        # and we'd rather waste a few hundred harmless extra steps than
        # truncate a real reply early (see module docstring).
        max_new_tokens = int(padded_duration_s * FRAME_RATE_HZ * 1.3) + 200

        import torch
        num_codebooks = model.config.num_codebooks
        codebook_size = model.config.codebook_size
        from tokenize_func import AUDIO_TOKEN_FORMAT, DEFAULT_ASSISTANT_TOKEN
        audio_min_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(0))
        assistant_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_ASSISTANT_TOKEN)
        tracker = _SpeakingTracker(
            assistant_token_id=assistant_token_id,
            eos_token_id=tokenizer.eos_token_id,
            audio_min_token_id=audio_min_token_id,
            codebook_size=codebook_size,
        )

        def _run():
            return _eval_model_autonomous_streaming(
                model, tokenizer, tokenizer_voila, history,
                ref_embs, ref_embs_mask, max_new_tokens, tracker,
            )

        audio_values = await asyncio.to_thread(_run)
        self._last_segments = tracker.segments
        return np.asarray(audio_values, dtype=np.float32)


# ── Standalone single-sample CLI (debugging only) ────────────────────────────

async def _main_async(args) -> None:
    import soundfile as sf

    input_duration = args.input_duration
    if input_duration is None:
        info = sf.info(str(args.input_wav))
        input_duration = info.duration

    client = VoilaAutonomousClient(
        api_key="",
        model_name=args.model,
        system_prompt="",
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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
