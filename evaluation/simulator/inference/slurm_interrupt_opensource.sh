#!/bin/bash
# ==============================================================================
# slurm_interrupt_opensource.sh -- `interrupt_user` simulator batch launcher
# for locally hosted open-source duplex models that already have an
# infer_interrupt_<model>.py client.
#
# Sibling of slurm_stop_opensource.sh / slurm_action_opensource.sh.
# Does NOT call run_stop_batch.py / run_action_batch.py. Does not touch
# stop / hold_floor / respond_and_recover jobs or clients.
#
# Usage:
#   sbatch slurm_interrupt_opensource.sh <MODEL> [LIMIT] [--overwrite]
#
# Examples:
#   sbatch slurm_interrupt_opensource.sh moshi
#   sbatch slurm_interrupt_opensource.sh moshi 5      # smoke test
# ==============================================================================

#SBATCH --job-name=interrupt_infer_oss
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/interrupt_oss_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/interrupt_oss_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:1

set -euo pipefail

MODEL="${1:?Usage: sbatch slurm_interrupt_opensource.sh <MODEL> [LIMIT] [--overwrite]}"
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
    echo "ERROR: model '${MODEL}' is not registered for interrupt_user (see run_interrupt_batch.py)." >&2
    exit 1
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV_PATH}." >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"

# OPENAI_API_KEY for InterruptMonitorController (gpt-5.4 stop LLM).
set -a
[[ -f "${PROJECT_ROOT}/.env" ]] && source "${PROJECT_ROOT}/.env"
set +a

export HF_HOME="${OSS_ROOT}/${MODEL}/checkpoints"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MOSHI_PORT=$((20000 + SLURM_JOB_ID % 10000))
    export FLM_AUDIO_PORT=$((30000 + SLURM_JOB_ID % 10000))
fi

cd "$INFER_DIR"
mkdir -p "${PROJECT_ROOT}/evaluation/simulator/logs"

echo "============================================================"
echo "  Simulator 'interrupt_user' Inference (open-source, local model)"
echo "  Model  : ${MODEL}"
echo "  Venv   : ${VENV_PATH}"
echo "  HF_HOME: ${HF_HOME}"
echo "  MOSHI_PORT=${MOSHI_PORT:-}"
echo "  OPENAI_API_KEY set: $([[ -n "${OPENAI_API_KEY:-}" ]] && echo yes || echo NO)"
echo "  Overwrite: ${OVERWRITE_FLAG:-no}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi -L 2>&1 || echo 'nvidia-smi unavailable')"
echo ""

python3 run_interrupt_batch.py \
    --model "${MODEL}" \
    --workers 1 \
    "${LIMIT_ARG[@]}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
