# DuplexAct

Benchmark for **interaction action taking** in end-to-end full-duplex spoken language models (E2E FD-SLMs).

This repository contains:

- **Inference** for Direct Replay and Simulator Replay
- **Evaluation** (Immediate + Subsequent scoring, ASR, report writers)
- **Candor processing** (Candor cannot be redistributed; you run this locally)

Packaged evaluation audio for AMI / CORAAL / Loqui and the synthesized conditions is released separately on Hugging Face. Synthesis code is **not** included.

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

   **Candor clips are not on the Hub.** The dump is 3,539 items.

2. **Candor** is not on Hugging Face. Download Candor yourself and run [`scripts/Candor/`](scripts/Candor/) (below), then `prepare_*.py` to merge those clips into the same layout.

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

The **relaxed** extractors are what produced the evaluation set:

| Pipeline | Script | Used for |
|---|---|---|
| Pauses | `extract_pause_candidates.py` + `batch_pause_backbiter.py` | Wait |
| Floor + backchannel | `extract_floor_backchannel_relaxed.py` + `batch_floor_backchannel_backbiter_relaxed.py` | Backchannel, Hold floor (real) |
| Interruptions | `extract_interruptions_relaxed.py` + `batch_interruptions_relaxed.py` | Stop, Recover (real) |
| Turn-taking | `extract_turn_taking_backbiter.py` + `batch_turn_taking_backbiter.py` | Take turn (overlap) |

Example:

```bash
export CANDOR_ROOT=/path/to/Candor
export OPENAI_API_KEY=...   # GPT filter / recategorize

python scripts/Candor/batch_interruptions_relaxed.py \
  --candor_root "$CANDOR_ROOT" \
  --results_dir results/Candor/interruptions_relaxed
```

Defaults inside the scripts still point at an internal cluster; **always pass `--candor_root` and `--results_dir`**. Then pack into the eval layout:

```bash
python evaluation/direct_replay/data/prepare_wait_samples.py
python evaluation/direct_replay/data/prepare_backchannel_samples.py
python evaluation/direct_replay/data/prepare_stop_samples.py
# ...
```

Those packagers also look for AMI / CORAAL / Loqui extracts and Hugging Face synthesized clips. If you only have Candor, they will take as many Candor items as exist (some conditions are short of 200 by construction).

---

## Citation


