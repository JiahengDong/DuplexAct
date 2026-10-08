#!/bin/bash
# ==============================================================================
# slurm_stop_opensource.sh -- `stop`-action simulator batch launcher for
# locally hosted open-source duplex models.
#
# Simulator-only sibling of direct_replay/inference/slurm_infer_opensource.sh
# -- same per-model venv/HF-cache/port-isolation setup, but runs
# run_stop_batch.py (this directory) against the 200-sample stop pool
# instead of direct_replay's run_inference.py. Deliberately a separate
# script (never edits/touches the direct_replay one).
#
# Usage:
#   sbatch slurm_stop_opensource.sh <MODEL> [LIMIT] [--overwrite]
#
# Examples:
#   sbatch slurm_stop_opensource.sh moshi
#   sbatch slurm_stop_opensource.sh raon-speechchat 5      # smoke test
#   sbatch slurm_stop_opensource.sh minicpm-o-4.5 "" --overwrite
# ==============================================================================

#SBATCH --job-name=stop_infer_oss
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/stop_oss_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/stop_oss_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:1

set -euo pipefail

MODEL="${1:?Usage: sbatch slurm_stop_opensource.sh <MODEL> [LIMIT] [--overwrite]}"
LIMIT_ARG=()
OVERWRITE_FLAG=""
if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
    LIMIT_ARG=(--limit "$2")
fi
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        OVERWRITE_FLAG="--overwrite"
    fi
done

PROJECT_ROOT=/data/gpfs/projects/punim2960/jiahengd
OSS_ROOT="${PROJECT_ROOT}/opensource_models"
INFER_DIR="${PROJECT_ROOT}/evaluation/simulator/inference"

# Same per-model venv lookup table as direct_replay/inference/
# slurm_infer_opensource.sh -- only the 6 models registered in
# run_stop_batch.py's MODEL_REGISTRY (real live text signal, not blocked).
declare -A MODEL_VENV=(
    [moshi]="${OSS_ROOT}/moshi/venv"
    [flm-audio]="${OSS_ROOT}/flm-audio/venv"
    [bayling-duplex]="${OSS_ROOT}/bayling-duplex/venv"
    [raon-speechchat]="${OSS_ROOT}/raon-speechchat/venv"
    [minicpm-o-4.5]="${OSS_ROOT}/minicpm-o-4.5/venv"
    [freeze-omni]="${OSS_ROOT}/freeze-omni/venv"
)

VENV_PATH="${MODEL_VENV[$MODEL]:-}"
if [[ -z "$VENV_PATH" ]]; then
    echo "ERROR: model '${MODEL}' is not registered for the stop-action batch (see run_stop_batch.py's MODEL_REGISTRY)." >&2
    exit 1
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV_PATH}." >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"

# Trigger LLM (when to inject the cue) needs OpenAI keys from .env.
set -a
[[ -f "${PROJECT_ROOT}/.env" ]] && source "${PROJECT_ROOT}/.env"
set +a

# Per-model HF cache isolation -- identical convention to
# slurm_infer_opensource.sh (see that file's own comment for the
# TRANSFORMERS_CACHE/HUGGINGFACE_HUB_CACHE rationale).
export HF_HOME="${OSS_ROOT}/${MODEL}/checkpoints"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME"

# Per-job unique local-server port (moshi/flm-audio) -- same rationale as
# slurm_infer_opensource.sh's own comment (avoids same-node port collision
# between two same-model jobs SLURM packs onto one multi-GPU node).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MOSHI_PORT=$((20000 + SLURM_JOB_ID % 10000))
    export FLM_AUDIO_PORT=$((30000 + SLURM_JOB_ID % 10000))
fi

cd "$INFER_DIR"
mkdir -p "${PROJECT_ROOT}/evaluation/simulator/logs"

echo "============================================================"
echo "  Simulator 'stop'-action Inference (open-source, local model)"
echo "  Model  : ${MODEL}"
echo "  Venv   : ${VENV_PATH}"
echo "  HF_HOME: ${HF_HOME}"
echo "  Overwrite: ${OVERWRITE_FLAG:-no}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi -L 2>&1 || echo 'nvidia-smi unavailable')"
echo ""

python3 run_stop_batch.py \
    --model "${MODEL}" \
    --workers 1 \
    "${LIMIT_ARG[@]}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
