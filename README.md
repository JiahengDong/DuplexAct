# DuplexAct

Benchmark for **interaction action taking** in end-to-end full-duplex spoken language models (E2E FD-SLMs).

- **Code:** [github.com/JiahengDong/DuplexAct](https://github.com/JiahengDong/DuplexAct)
- **Data:** [huggingface.co/datasets/LincolnD/DuplexAct](https://huggingface.co/datasets/LincolnD/DuplexAct)

This repository contains:

- **Inference** for Direct Replay and Simulator Replay
- **Evaluation** (Immediate + Subsequent scoring, ASR, report writers)
- **Candor / Loqui processing** (those corpora cannot be redistributed; run the extractors locally after you download the official releases)

Packaged evaluation audio for AMI / CORAAL and the synthesized conditions is on [Hugging Face](https://huggingface.co/datasets/LincolnD/DuplexAct). Synthesis code is **not** included.

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

Scoring definitions: [`evaluation/APPENDIX.md`](evaluation/APPENDIX.md) and [`evaluation_rubric/`](evaluation_rubric/).

---

## Repository layout

```
evaluation/
  direct_replay/inference/    # stream input.wav to a model
  direct_replay/evaluation/   # Whisper ASR (asr.json)
  direct_replay/data/         # prepare_*.py (pack extracted / HF clips)
  simulator/inference/        # live cue / barge-in drivers
  metrics/                    # Immediate + Subsequent
  metrics/reports/            # table writers
  APPENDIX.md
evaluation_rubric/            # per-action rubrics
scripts/Candor/               # extract Candor clips you download yourself
scripts/Loqui/                # extract Loqui clips you download yourself
```

What is **not** here: `data_synthesize/`, AMI / CORAAL extractors, model checkpoints, `output.wav` / score caches, API keys.

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

1. Download the Hugging Face evaluation set (AMI / CORAAL + synthesized conditions) and unpack it into the inference layout:

   ```bash
   python scripts/unpack_hf_dataset.py --repo_id LincolnD/DuplexAct \
     --out_dir evaluation/direct_replay/data
   ```

   Or load a single action as a `datasets` object (audio columns are `{path, bytes}`):

   ```python
   from datasets import load_dataset
   ds = load_dataset("LincolnD/DuplexAct", "wait", split="test")
   ```

   Unpacked layout: `evaluation/direct_replay/data/<action>/<condition>/<sample_id>/`
   with the protocol audio (`input.wav`, or `lead_in.wav` + `cue.wav`, or Interrupt
   `user_speech.wav` / branch wavs), `metadata.json`, and per-action `manifest.json`.

   **Candor and Loqui clips are not on the Hub.** The dump is 3,408 items.

2. **Candor** and **Loqui** are not on Hugging Face. Download each corpus yourself and run [`scripts/Candor/`](scripts/Candor/) / [`scripts/Loqui/`](scripts/Loqui/) (below), then `prepare_*.py` to merge those clips into the same layout.

---

## Inference

### Direct Replay (Wait, Backchannel, Ignore, Take turn)

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

The model first speaks from `lead_in.wav`. A cue (or, for Interrupt, ongoing user speech) is injected live. The same sample directories under `evaluation/direct_replay/data/` are used — there is no second test set.

```bash
cd evaluation/simulator/inference

python run_action_batch.py --action stop --model gpt-realtime-2.1
python run_action_batch.py --action hold_floor --model gpt-realtime-2.1
python run_action_batch.py --action respond_and_recover --model gpt-realtime-2.1
python run_interrupt_batch.py --model gpt-realtime-2.1
```

Cluster launchers (`slurm_*.sh`) still contain the original HPC paths; treat them as templates and set `PROJECT_ROOT` / `DUPLEXACT_ROOT`.

---

## Evaluation

Most models are scored from Whisper `asr.json` on `output.wav`. Bayling and Raon use native `model_speech_transcripts` in `inference_info.json` (do not fall back to ASR).

```bash
# 1. ASR (skip for bayling-duplex / raon-speechchat)
python evaluation/direct_replay/evaluation/asr_openai.py \
  --results_root evaluation/direct_replay/results/inference_results

# 2. Immediate + Subsequent
python evaluation/metrics/run_metrics.py --model gpt-realtime-2.1 --action wait
python evaluation/metrics/run_metrics.py --immediate_only
python evaluation/metrics/run_metrics.py --aggregate

# 3. Backchannel Immediate is two-stage (text screen + audio judge)
python evaluation/metrics/bc_two_stage_judge.py --model gpt-realtime-2.1

# 4. Tables
bash evaluation/metrics/reports/make_reports.sh --all
```

`run_metrics.py` writes `evaluation/metrics/cache/<action>/<model>/<sample_id>.json`. Subsequent calls `gpt-5.4` (`temperature=0`, YES/NO JSON) except Stop Subsequent, which is automatic from segments.

---

## Candor processing

Candor is licensed such that we cannot host the audio. You need a local copy of the official release (per-conversation folders with audio + Backbiter transcripts).

These are the only Candor scripts in this repo. All four read Backbiter transcripts:

| Pipeline | Script | Used for |
|---|---|---|
| Pauses | `extract_pause_candidates.py` + `batch_pause_backbiter.py` | Wait |
| Floor + backchannel | `extract_floor_backchannel.py` + `batch_floor_backchannel_backbiter.py` | Backchannel, Hold floor (real) |
| Interruptions | `extract_interruptions.py` + `batch_interruptions.py` | Stop, Recover (real) |
| Turn-taking | `extract_turn_taking_backbiter.py` + `batch_turn_taking_backbiter.py` | Take turn (overlap) |

Example:

```bash
export CANDOR_ROOT=/path/to/Candor
export OPENAI_API_KEY=...   # GPT filter / recategorize

python scripts/Candor/batch_interruptions.py \
  --candor_root "$CANDOR_ROOT" \
  --results_dir results/Candor/interruptions
```

Defaults inside the scripts still point at an internal cluster; **always pass `--candor_root` and `--results_dir`**. Then pack into the eval layout:

```bash
python evaluation/direct_replay/data/prepare_wait_samples.py
python evaluation/direct_replay/data/prepare_backchannel_samples.py
python evaluation/direct_replay/data/prepare_stop_samples.py
# ...
```

Those packagers also look for AMI / CORAAL / Loqui extracts and Hugging Face synthesized clips. If you only have Candor or Loqui, they will take as many of those items as exist (some conditions are short of 200 by construction).

---

## Loqui processing

Loqui is not hosted: the Academic Commons record ([doi:10.7916/D82R3PW9](https://doi.org/10.7916/D82R3PW9)) does not attach a Creative Commons license, so we do not redistribute the audio. You need a local copy of `LoquiPackage` (`Transcriptions/*.final.trs` + `*.san.wav`).

These are the only Loqui scripts in this repo:

| Pipeline | Script | Used for |
|---|---|---|
| Pauses | `extract_pause_loqui.py` + `batch_pause_loqui.py` | Wait |
| Floor + backchannel | `extract_floor_backchannel_loqui.py` + `batch_floor_backchannel_loqui.py` | Backchannel, Hold floor (real) |
| Interruptions | `extract_interruptions_loqui.py` + `batch_interruptions_loqui.py` | Stop, Recover (real) |
| Turn-taking | `extract_turn_taking_loqui.py` + `batch_turn_taking_loqui.py` | Take turn |

Example:

```bash
export LOQUI_ROOT=/path/to/LoquiPackage
export OPENAI_API_KEY=...   # GPT filter / recategorize

python scripts/Loqui/batch_pause_loqui.py \
  --loqui_root "$LOQUI_ROOT" \
  --results_dir results/Loqui/pauses
```

`--loqui_root` may be `LoquiPackage`, its parent, or the `Transcriptions` folder. Defaults inside the scripts still point at an internal cluster; **always pass `--loqui_root` and `--results_dir`**. Then pack into the eval layout with the same `prepare_*.py` scripts as Candor.

---

## License and source data

This repository is **code only**. It does not host AMI, CORAAL, Loqui, or Candor audio.

The evaluation clips on [Hugging Face](https://huggingface.co/datasets/LincolnD/DuplexAct) mix AMI, CORAAL, and synthesized conditions. Each source keeps its own terms; the Hub dump is **not** a single permissive license.

| Source | What we ship | Official terms | What that means here |
|---|---|---|---|
| **AMI Meeting Corpus** | clipped audio derived from AMI | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([AMI licence page](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)) | Redistribution is allowed if you keep attribution, name CC BY 4.0, and note that the clips are modified (segmented) relative to the original corpus. |
| **CORAAL** | clipped audio derived from CORAAL | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([User Guide](https://lingtools.uoregon.edu/coraal/userguide/CORAALUserGuide_current.pdf)) | Research/education only (non-commercial). Any redistribution of CORAAL-derived clips must stay **CC BY-NC-SA 4.0** and cite CORAAL. |
| **Synthesized** | clips we generated | original to this benchmark | Distributed with the Hub dump for research use of DuplexAct. |
| **Loqui** | not hosted | Columbia Academic Commons deposit: [doi:10.7916/D82R3PW9](https://doi.org/10.7916/D82R3PW9). That record describes transcriptions/annotations and does **not** attach a Creative Commons license. | Copyright remains with the original authors. Obtain Loqui yourself and run `scripts/Loqui/`. |
| **Candor** | not hosted | original Candor license | Audio cannot be redistributed. Obtain Candor yourself and run `scripts/Candor/`. |

Because CORAAL is ShareAlike and NonCommercial, the **combined** Hub dataset cannot be treated as CC BY. Hugging Face is tagged `other` for that reason. Downstream use of a row must follow the `dataset` field on that row (`AMI` / `CORAAL` / `synthesized`).

Suggested source citations (in addition to DuplexAct, when a citation exists):

- AMI: see [groups.inf.ed.ac.uk/ami/corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- CORAAL: Kendall, Tyler and Charlie Farrington. 2023. *The Corpus of Regional African American Language*. Version 2023.06. Eugene, OR: The Online Resources for African American Language Project. http://oraal.uoregon.edu/coraal — https://doi.org/10.7264/1ad5-6t35
- Loqui: Passonneau, Rebecca; Sachar, Evaneet. 2014. *Loqui Human-Human Dialogue Corpus (Transcriptions and Annotations)*. Columbia University Academic Commons. https://doi.org/10.7916/D82R3PW9

---

## Citation


