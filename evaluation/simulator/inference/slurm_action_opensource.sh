#!/bin/bash
# ==============================================================================
# slurm_action_opensource.sh -- action-generalized sibling of
# slurm_stop_opensource.sh, for `hold_floor` / `respond_and_recover` (see
# cue_action_common.py's own module docstring for why interrupt_user isn't
# included here). Same per-model venv/HF-cache/port-isolation setup, runs
# run_action_batch.py instead of run_stop_batch.py. Never edits/touches
# slurm_stop_opensource.sh.
#
# Usage:
#   sbatch slurm_action_opensource.sh <ACTION> <MODEL> [LIMIT] [--overwrite] [--conditions=C1,C2]
#
# Examples:
#   sbatch slurm_action_opensource.sh respond_and_recover moshi 3      # smoke test
#   sbatch slurm_action_opensource.sh hold_floor raon-speechchat
# ==============================================================================

#SBATCH --job-name=action_infer_oss
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/action_oss_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/action_oss_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:1

set -euo pipefail

ACTION="${1:?Usage: sbatch slurm_action_opensource.sh <ACTION> <MODEL> [LIMIT] [--overwrite] [--conditions=C1,C2]}"
MODEL="${2:?Usage: sbatch slurm_action_opensource.sh <ACTION> <MODEL> [LIMIT] [--overwrite] [--conditions=C1,C2]}"
LIMIT_ARG=()
OVERWRITE_FLAG=""
CONDITIONS_ARG=()
if [[ "${3:-}" =~ ^[0-9]+$ ]]; then
    LIMIT_ARG=(--limit "$3")
fi
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        OVERWRITE_FLAG="--overwrite"
    fi
    if [[ "$arg" == --conditions=* ]]; then
        CONDITIONS_ARG=(--conditions "${arg#--conditions=}")
    fi
done

PROJECT_ROOT=/data/gpfs/projects/punim2960/jiahengd
OSS_ROOT="${PROJECT_ROOT}/opensource_models"
INFER_DIR="${PROJECT_ROOT}/evaluation/simulator/inference"

# Identical venv lookup table to slurm_stop_opensource.sh -- same 6 models
# (real live text signal, not blocked). Kept as a literal copy rather than
# sourcing the other script, since bash has no clean "import just this
# variable" primitive and duplicating one small table is simpler/safer than
# the two scripts secretly depending on each other's internals.
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
    echo "ERROR: model '${MODEL}' is not registered (see run_stop_batch.py's MODEL_REGISTRY)." >&2
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
if [[ "${USE_OPENAI_FALLBACK:-}" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY_FALLBACK:-}" ]]; then
    echo "ERROR: USE_OPENAI_FALLBACK=1 but OPENAI_API_KEY_FALLBACK is unset." >&2
    exit 1
  fi
  export OPENAI_API_KEY="${OPENAI_API_KEY_FALLBACK}"
  echo "Using OPENAI_API_KEY_FALLBACK for the trigger LLM (key value not printed)."
fi

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
echo "  Simulator action-generalized Inference (open-source, local model)"
echo "  Action : ${ACTION}"
echo "  Model  : ${MODEL}"
echo "  Venv   : ${VENV_PATH}"
echo "  HF_HOME: ${HF_HOME}"
echo "  Overwrite: ${OVERWRITE_FLAG:-no}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi -L 2>&1 || echo 'nvidia-smi unavailable')"
echo ""

python3 run_action_batch.py \
    --action "${ACTION}" \
    --model "${MODEL}" \
    --workers 1 \
    "${LIMIT_ARG[@]}" \
    "${CONDITIONS_ARG[@]}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
