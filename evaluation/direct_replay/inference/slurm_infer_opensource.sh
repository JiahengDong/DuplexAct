#!/bin/bash
# ==============================================================================
# slurm_infer_opensource.sh -- direct-replay inference launcher for locally
# hosted open-source duplex models.
#
# Unlike slurm_infer.sh (closed-source, cloud APIs), each open-source model
# lives in its OWN isolated venv + HF cache under
#   /data/gpfs/projects/punim2960/jiahengd/opensource_models/<slug>/
# (never a shared env, never under $HOME -- see the per-model isolation
# policy). This script activates the right venv for the given model slug
# via the lookup table below, sets that model's HF cache env vars, requests
# a GPU, then runs the same generic run_inference.py used for every model
# (closed- or open-source).
#
# Local models also need --workers 1: the model's own local server
# typically serializes requests internally (e.g. Moshi's asyncio.Lock), and
# is started lazily on the first sample and reused for the rest of the job
# (see local_common.LocalServerManager) -- extra concurrent workers would
# just queue behind it for no benefit.
#
# Usage:
#   sbatch slurm_infer_opensource.sh <ACTION> <L3_CONDITION|all> <MODEL> [LIMIT] [--overwrite]
#
# Examples:
#   sbatch slurm_infer_opensource.sh wait inter_sentential moshi 2
#   sbatch slurm_infer_opensource.sh backchannel all moshi
#   sbatch slurm_infer_opensource.sh turn_taking take_turn_immediately moshi 4 --overwrite
# ==============================================================================

#SBATCH --job-name=direct_replay_infer_oss
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs/infer_oss_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs/infer_oss_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:1

set -euo pipefail

# vita-1.5 drives the model's own official 2-GPU-resident real-time server
# (web_demo/server.py -- two vLLM-loaded 8B copies on separate physical
# GPUs, gated by the model's own bundled VAD -- see infer_vita15.py's
# module docstring and model_docs/opensource/vita-1.5.md for why a single
# GPU can't replicate this). The #SBATCH --gres=gpu:1 default above must be
# overridden AT SUBMIT TIME for this model specifically (command-line
# sbatch flags take precedence over a script's own #SBATCH directives):
#   sbatch --gres=gpu:2 slurm_infer_opensource.sh <action> <l3> vita-1.5 ...
# This is just a runtime safety net catching a forgotten override, not the
# actual allocation mechanism.
if [[ "${3:-}" == "vita-1.5" ]]; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ "$N_GPUS" -lt 2 ]]; then
        echo "ERROR: vita-1.5 needs 2 GPUs, this job only has ${N_GPUS}." >&2
        echo "Resubmit with: sbatch --gres=gpu:2 slurm_infer_opensource.sh <action> <l3> vita-1.5 ..." >&2
        exit 1
    fi
fi

ACTION="${1:?Usage: sbatch slurm_infer_opensource.sh <ACTION> <L3_CONDITION|all> <MODEL> [LIMIT] [--overwrite]}"
L3_CONDITION="${2:?Missing L3_CONDITION (use 'all' to run every l3_condition for this action)}"
MODEL="${3:?Missing MODEL slug}"
LIMIT_ARG=()
OVERWRITE_FLAG=""
if [[ "${4:-}" =~ ^[0-9]+$ ]]; then
    LIMIT_ARG=(--limit "$4")
fi
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        OVERWRITE_FLAG="--overwrite"
    fi
done

PROJECT_ROOT=/data/gpfs/projects/punim2960/jiahengd
OSS_ROOT="${PROJECT_ROOT}/opensource_models"
INFER_DIR="${PROJECT_ROOT}/evaluation/direct_replay/inference"

# ── Per-model venv lookup ────────────────────────────────────────────────────
# Add one line per model as its adapter is built. Every entry points at that
# model's OWN venv under opensource_models/<slug>/venv -- never shared.
declare -A MODEL_VENV=(
    [moshi]="${OSS_ROOT}/moshi/venv"
    [minicpm-o-4.5]="${OSS_ROOT}/minicpm-o-4.5/venv"
    [voila-autonomous]="${OSS_ROOT}/voila-autonomous/venv"
    [bayling-duplex]="${OSS_ROOT}/bayling-duplex/venv"
    [raon-speechchat]="${OSS_ROOT}/raon-speechchat/venv"
    [freeze-omni]="${OSS_ROOT}/freeze-omni/venv"
    [mini-omni2]="${OSS_ROOT}/mini-omni2/venv"
    [vita-1.5]="${OSS_ROOT}/vita-1.5/venv"
    [flm-audio]="${OSS_ROOT}/flm-audio/venv"
    [lychee-fd]="${OSS_ROOT}/lychee-fd/venv"
)

# Conda-env models (as opposed to plain venvs above) -- currently just
# nemotron-voicechat, which needs conda specifically for bundled Python
# dev headers during its C-extension builds (mamba-ssm/causal-conv1d) --
# see model_docs/opensource/nemotron-voicechat.md.
declare -A MODEL_CONDA_ENV=(
    [nemotron-voicechat]="${OSS_ROOT}/nemotron-voicechat/conda_env"
)

CONDA_ENV_PATH="${MODEL_CONDA_ENV[$MODEL]:-}"
if [[ -n "$CONDA_ENV_PATH" ]]; then
    if [[ ! -d "$CONDA_ENV_PATH" ]]; then
        echo "ERROR: conda env not found at ${CONDA_ENV_PATH}." >&2
        exit 1
    fi
    source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_PATH"
    # nvcc (for mamba-ssm/causal-conv1d) + this env's own newer libstdc++ and
    # pinned FFmpeg 8.x (torchcodec only supports FFmpeg 4-8) ahead of the
    # system libstdc++ -- see model_docs/opensource/nemotron-voicechat.md
    # section 3 for the exact failures each of these fixes.
    module load CUDA/12.8.0
    export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
    # README: "Requires this repository to precede any installed nemo_toolkit
    # on PYTHONPATH" -- belt-and-suspenders on top of the editable install.
    export PYTHONPATH="${OSS_ROOT}/nemotron-voicechat/repo:${PYTHONPATH:-}"
    VENV_PATH="$CONDA_ENV_PATH"   # only used for the echo block below
else
    VENV_PATH="${MODEL_VENV[$MODEL]:-}"
    if [[ -z "$VENV_PATH" ]]; then
        echo "ERROR: no venv/conda env registered for model '${MODEL}' in slurm_infer_opensource.sh." >&2
        exit 1
    fi
    if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
        echo "ERROR: venv not found at ${VENV_PATH} (expected ${VENV_PATH}/bin/activate)." >&2
        exit 1
    fi
    source "${VENV_PATH}/bin/activate"
fi

# mini-omni2's own inference.py shells out to ffmpeg (via openai-whisper's
# load_audio()) -- not bundled in its venv, loaded as a cluster module instead.
if [[ "$MODEL" == "mini-omni2" ]]; then
    module load GCCcore/13.3.0 FFmpeg/7.1.1
fi

# Per-model HF cache isolation (mirrors local_common.env_for_model(), set
# here too so anything the model's server does at process-start time -- e.g.
# hf_hub_download for the static web assets -- also honors it).
# TRANSFORMERS_CACHE and HUGGINGFACE_HUB_CACHE point at the SAME "hub/"
# directory, not siblings -- see env_for_model()'s comment in
# local_common.py for why (this exact mismatch broke VITA-1.5's
# InternViT vision-tower download).
export HF_HOME="${OSS_ROOT}/${MODEL}/checkpoints"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"

# The project filesystem's inode quota (500k files) is the binding
# constraint, not disk space -- 10 fully-isolated per-model venvs (torch +
# transformers + friends) each regenerate tens of thousands of .pyc files
# under __pycache__/ on every single run (one compiled bytecode file per
# imported module across the ENTIRE dependency tree). Disabling bytecode
# caching entirely costs a bit of import-time CPU on each job start but
# keeps inode usage from silently creeping back up after every smoke test
# / batch run -- this bit us twice in one sitting already.
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME"

# Per-job unique local-server port for models that run their own network
# server on a fixed default port (Moshi, FLM-Audio -- see MOSHI_PORT/
# FLM_AUDIO_PORT in their respective infer_*.py, both already support this
# env var override). Without this, two same-model jobs landing on the SAME
# physical node concurrently (SLURM packs multiple single-GPU jobs onto one
# multi-GPU node all the time -- these jobs only request --gres=gpu:1, not
# a whole node) both try to bind the SAME port, and one of them gets
# `ConnectionClosedError: received 1002 (protocol error)` /
# `TimeoutError: timed out during opening handshake` for every sample for
# the rest of the job -- exactly what broke ~13% of FLM-Audio's first
# full-dataset run (2 of its 10 job chains landed on the same node at the
# exact same start second). $SLURM_JOB_ID is unique per job by
# construction, so deriving the port from it (offset into a high,
# unprivileged range) makes a same-node collision astronomically unlikely
# without needing --exclusive node allocation (which would waste the
# node's other GPUs).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MOSHI_PORT=$((20000 + SLURM_JOB_ID % 10000))
    export FLM_AUDIO_PORT=$((30000 + SLURM_JOB_ID % 10000))
    export VITA15_SERVER_PORT=$((28000 + SLURM_JOB_ID % 10000))
fi

cd "$INFER_DIR"
mkdir -p "${PROJECT_ROOT}/evaluation/direct_replay/logs"

L3_ARG=()
if [[ "$L3_CONDITION" != "all" ]]; then
    L3_ARG=(--l3_condition "$L3_CONDITION")
fi

echo "============================================================"
echo "  Direct-Replay Inference (open-source, local model)"
echo "  Action      : ${ACTION}"
echo "  L3 condition: ${L3_CONDITION}"
echo "  Model       : ${MODEL}"
echo "  Venv        : ${VENV_PATH}"
echo "  HF_HOME     : ${HF_HOME}"
echo "  Overwrite   : ${OVERWRITE_FLAG:-no}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi -L 2>&1 || echo 'nvidia-smi unavailable')"
echo ""

python3 run_inference.py \
    --action "${ACTION}" \
    "${L3_ARG[@]}" \
    --model "${MODEL}" \
    --workers 1 \
    "${LIMIT_ARG[@]}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
