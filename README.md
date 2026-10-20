# DuplexAct

Benchmark for **interaction action taking** in end-to-end full-duplex spoken language models (E2E FD-SLMs).

- **Code:** [github.com/JiahengDong/DuplexAct](https://github.com/JiahengDong/DuplexAct)
- **Data:** [huggingface.co/datasets/LincolnD/DuplexAct](https://huggingface.co/datasets/LincolnD/DuplexAct)

This repository contains:

- **Inference** for Direct Replay and Simulator Replay
- **Evaluation** (Immediate + Subsequent scoring, ASR, report writers)
- **Candor rebuild** (Candor audio cannot be redistributed; a fixed recipe rebuilds the exact Candor instances from your own copy)

Packaged evaluation audio for AMI / CORAAL / Loqui and the synthesized conditions is on [Hugging Face](https://huggingface.co/datasets/LincolnD/DuplexAct). Synthesis code is **not** included.

---

## What DuplexAct evaluates

E2E FD-SLMs listen and speak at the same time. That requires deciding **whether, when, and how** to participate as the conversation unfolds. Existing benchmarks usually measure whether a behavior *occurs* (takeover rate, latency, backchannel frequency), not whether the action *succeeds*, and they cover only a slice of the interaction space.

DuplexAct is grounded in human Interactional Competence and defines a **three-level action taxonomy**:

| Level-1 | Level-2 action | Conditions | Protocol |
|---|---|---:|---|
| Turn management | Take turn | 3 | Direct replay |
| | Stop | 1 | Simulator |
| | Hold floor | 5 | Simulator |
| | Interrupt | 2 | Simulator |
| Interactive listening | Wait | 3 | Direct replay |
| | Backchannel | 2 | Direct replay |
| | Ignore | 2 | Direct replay |
| Breakdown repair | Recover | 6 | Simulator |

**4,691** test instances, **24** conditions, four scenarios (casual conversation, meeting, customer service, interview). **2,531** instances come from real conversations; the rest are synthesized where natural data are scarce.

Success is two 0/1 scores, not a single occurrence rate:

- **Immediate** — did the model do the right thing in the annotated timing window?
- **Subsequent** — after that, did later behavior stay consistent with the action?

Action-level tables are a **micro-average** over the scored trials of that action (the test distribution as built). Condition-level tables report each condition on its own. Subsequent is gated on Immediate = 1.

Scoring definitions: [`evaluation/APPENDIX.md`](evaluation/APPENDIX.md).

---

## Repository layout

```
evaluation/
  data/                       # unpacked test set (created by the data steps below)
  common/                     # streaming clients shared by both protocols
  direct_replay/
    inference/                # stream input.wav to a model (14 models)
    metrics/                  # Wait, Ignore, Take turn, Backchannel
  simulator_replay/
    inference/                # live cue / barge-in drivers (10 models)
    metrics/                  # Stop, Hold floor, Recover, Interrupt
  scoring/                    # ASR, Immediate + Subsequent runner, report tables
  APPENDIX.md
scripts/Candor/               # rebuild the Candor instances from your own Candor copy
```

What is **not** here: `data_synthesize/`, AMI / CORAAL / Loqui extractors, model checkpoints, `output.wav` / score caches, API keys.

---

## Setup

```bash
git clone git@github.com:JiahengDong/DuplexAct.git
cd DuplexAct
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="$PWD"
export DUPLEXACT_ROOT="$PWD"
cp .env.example .env          # then fill API keys
```

Open-source local servers (Moshi, MiniCPM-o, …) need checkpoints. Point `OPENSOURCE_MODELS_ROOT` at that tree.

### Data

1. Download the Hugging Face evaluation set (AMI / CORAAL / Loqui + synthesized conditions) and unpack it into the inference layout:

   ```bash
   python scripts/unpack_hf_dataset.py --repo_id LincolnD/DuplexAct \
     --out_dir evaluation/data
   ```

   Or load a single action as a `datasets` object (audio columns are `{path, bytes}`):

   ```python
   from datasets import load_dataset
   ds = load_dataset("LincolnD/DuplexAct", "wait", split="test")
   ```

   Unpacked layout: `evaluation/data/<action>/<condition>/<sample_id>/`
   with the protocol audio (`input.wav`, or `lead_in.wav` + `cue.wav`, or Interrupt
   `user_speech.wav` / branch wavs) and `metadata.json`. Each `<action>/manifest.json` is generated from those `metadata.json` files ([`scripts/build_manifests.py`](scripts/build_manifests.py)); it only lists samples for Direct Replay inference.

   **Candor clips are not on the Hub.** The dump is 3,456 items.

2. **Candor** (1,235 items) is not on Hugging Face. Get the official corpus (see [Candor](#candor) below) and rebuild those instances into the same folders:

   ```bash
   python scripts/Candor/rebuild_candor.py --candor_root /path/to/Candor \
     --out_dir evaluation/data
   ```

---

## Inference

### Direct Replay (Wait, Ignore, Take turn, Backchannel)

The user clip is streamed in real time. The model may speak over it. Output is time-aligned `output.wav` + `inference_info.json`.

```bash
cd evaluation/direct_replay/inference

python run_inference.py \
  --action wait --l3_condition floor_holding \
  --model gpt-realtime-2.1 --workers 4

# all conditions of one action
python run_inference.py --action backchannel --model gpt-realtime-2.1

# smoke
python run_inference.py --action wait --l3_condition floor_holding \
  --model gpt-realtime-2.1 --limit 2 --verbose
```

`--model` slugs match `model_registry.py` (`gpt-realtime-2.1`, `grok-voice`, `qwen3.5-omni-flash-realtime`, `gemini-3.1-flash-live`, `moshi`, `minicpm-o-4.5`, `bayling-duplex`, `raon-speechchat`, …).

Results:

```
evaluation/direct_replay/results/inference_results/
  <action>/<condition>/<model>/<sample_id>/
    output.wav
    inference_info.json
    events.jsonl
```

### Simulator Replay (Stop, Hold floor, Recover, Interrupt)

The model first speaks from `lead_in.wav`. A cue (or, for Interrupt, ongoing user speech) is injected live. The same sample directories under `evaluation/data/` are used — there is no second test set.

```bash
cd evaluation/simulator_replay/inference

python run_action_batch.py --action stop --model gpt-realtime-2.1
python run_action_batch.py --action hold_floor --model gpt-realtime-2.1
python run_action_batch.py --action respond_and_recover --model gpt-realtime-2.1
python run_interrupt_batch.py --model gpt-realtime-2.1
```

Simulator results go to `evaluation/simulator_replay/results/<action>/<condition>/<model>/<sample_id>/`.

### Post-inference fixes

Two models need one extra pass before scoring:

```bash
# Bayling: write its native text stream into model_speech_transcripts
python evaluation/direct_replay/inference/backfill_bayling_transcripts.py

# Lychee-FD: rebuild model_speech_segments from output.wav
python evaluation/direct_replay/inference/recompute_lychee_fd_segments.py
```

---

## Evaluation

Most models are scored from Whisper `asr.json` on `output.wav`. Bayling and Raon use native `model_speech_transcripts` in `inference_info.json` (do not fall back to ASR).

```bash
# 1. ASR (skip for bayling-duplex / raon-speechchat)
python evaluation/scoring/asr_openai.py \
  --results_root evaluation/direct_replay/results/inference_results

# 2. Immediate + Subsequent
python evaluation/scoring/run_metrics.py --model gpt-realtime-2.1 --action wait
python evaluation/scoring/run_metrics.py --immediate_only
python evaluation/scoring/run_metrics.py --aggregate

# 3. Backchannel Immediate is two-stage (text screen + audio judge)
python evaluation/direct_replay/metrics/bc_two_stage_judge.py --models gpt-realtime-2.1

# 4. Tables (IMMEDIATE.md, SUBSEQUENT.md, TIMING.md)
bash evaluation/scoring/reports/make_reports.sh
```

`run_metrics.py` writes `evaluation/scoring/cache/<action>/<model>/<sample_id>.json`. Subsequent calls `gpt-5.4` (`temperature=0`, YES/NO JSON) except Stop Subsequent, which is automatic from segments.

---

## Candor

Candor is licensed such that we cannot host the audio, and we do not release our extraction pipeline. Instead, [`scripts/Candor/candor_instances.jsonl`](scripts/Candor/candor_instances.jsonl) fixes every Candor instance: conversation, stereo channel, exact sample range, silenced spans, trailing silence, its `metadata.json`, and its manifest position. [`rebuild_candor.py`](scripts/Candor/rebuild_candor.py) cuts those ranges out of your local copy, so the rebuilt instances are the ones we evaluated, checked by per-file SHA-256.

**Getting Candor.** Apply for access on the [BetterUp data request site](https://betterup-data-requests.herokuapp.com/). Once approved you receive download links (live for one week). Candor is also on [TalkBank](https://talkbank.org/ca/access/CANDOR.html): first get approval from the BetterUp site, then register at TalkBank (email address and password), then email macw@cmu.edu to request TalkBank CANDOR access. Cite Reece et al. (2023), *Science Advances* 9(13), eadf3197.

`rebuild_candor.py` reads the BetterUp release layout (`<conversation_id>/processed/<conversation_id>.mp3` and `processed/channel_map.json`) and was verified on it. The TalkBank copy is reorganised into 16 segments, so it may need re-arranging into that layout first.

```bash
python scripts/Candor/rebuild_candor.py --candor_root /path/to/Candor --out_dir evaluation/data
```

| Action | Wait | Backchannel | Take turn | Stop | Hold floor | Recover |
|---|---:|---:|---:|---:|---:|---:|
| Candor instances | 230 | 106 | 155 | 158 | 162 | 424 |

Run it after `unpack_hf_dataset.py`; it merges into the same `manifest.json` files. Details: [`scripts/Candor/README.md`](scripts/Candor/README.md).

---

## License and source data

This repository is **code only**. It does not host AMI, CORAAL, Loqui, or Candor audio.

The evaluation clips on [Hugging Face](https://huggingface.co/datasets/LincolnD/DuplexAct) mix AMI, CORAAL, Loqui, and synthesized conditions. Each source keeps its own terms; the Hub dump is **not** a single permissive license.

| Source | What we ship | Official terms | What that means here |
|---|---|---|---|
| **AMI Meeting Corpus** | clipped audio derived from AMI | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([AMI licence page](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)) | Redistribution is allowed if you keep attribution, name CC BY 4.0, and note that the clips are modified (segmented) relative to the original corpus. |
| **CORAAL** | clipped audio derived from CORAAL | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([User Guide](https://lingtools.uoregon.edu/coraal/userguide/CORAALUserGuide_current.pdf)) | Research/education only (non-commercial). Any redistribution of CORAAL-derived clips must stay **CC BY-NC-SA 4.0** and cite CORAAL. |
| **Synthesized** | clips we generated | original to this benchmark | Distributed with the Hub dump for research use of DuplexAct. |
| **Loqui** | clipped audio derived from Loqui (126 items) | Columbia Academic Commons deposit: [doi:10.7916/D82R3PW9](https://doi.org/10.7916/D82R3PW9). Redistributed with permission of the Loqui authors. | Attribute Passonneau & Sachar (2014). Copyright remains with the original authors. Selected evaluation clips only, not the full conversations. |
| **Candor** | not hosted; sample ranges only | original Candor license | Audio cannot be redistributed. Obtain Candor yourself ([BetterUp](https://betterup-data-requests.herokuapp.com/) or [TalkBank](https://talkbank.org/ca/access/CANDOR.html)) and run `scripts/Candor/rebuild_candor.py`. |

Because CORAAL is ShareAlike and NonCommercial, and Loqui is permission-based rather than Creative Commons, the **combined** Hub dataset cannot be treated as CC BY. Hugging Face is tagged `other` for that reason. Downstream use of a row must follow the `dataset` field on that row (`AMI` / `CORAAL` / `Loqui` / `synthesized`).

Suggested source citations (in addition to DuplexAct, when a citation exists):

- AMI: see [groups.inf.ed.ac.uk/ami/corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- CORAAL: Kendall, Tyler and Charlie Farrington. 2023. *The Corpus of Regional African American Language*. Version 2023.06. Eugene, OR: The Online Resources for African American Language Project. http://oraal.uoregon.edu/coraal — https://doi.org/10.7264/1ad5-6t35
- Loqui: Passonneau, Rebecca; Sachar, Evaneet. 2014. *Loqui Human-Human Dialogue Corpus (Transcriptions and Annotations)*. Columbia University Academic Commons. https://doi.org/10.7916/D82R3PW9

---

## Citation


