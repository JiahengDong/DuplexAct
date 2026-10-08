#!/bin/bash
# ==============================================================================
# slurm_interrupt_closed.sh -- closed-source interrupt_user batch (GPT /
# Gemini / Grok). Sapphire CPU, no GPU. Sibling of slurm_action_closed.sh.
#
# Usage:
#   sbatch slurm_interrupt_closed.sh <MODEL> [LIMIT] [--overwrite] [--workers=N] [--conditions=C]
#
# Examples:
#   sbatch slurm_interrupt_closed.sh gpt-realtime-2.1
#   sbatch slurm_interrupt_closed.sh gemini-3.1-flash-live 2
#   sbatch slurm_interrupt_closed.sh grok-voice --workers=8
#   sbatch slurm_interrupt_closed.sh gpt-realtime-2.1 --overwrite --conditions=interrupt_for_help
# ==============================================================================

#SBATCH --job-name=interrupt_infer_api
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/interrupt_api_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/interrupt_api_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=sapphire

set -euo pipefail

MODEL="${1:?Usage: sbatch slurm_interrupt_closed.sh <MODEL> [LIMIT] [--overwrite] [--workers=N] [--conditions=C]}"
LIMIT_ARG=()
OVERWRITE_FLAG=""
WORKERS=8
SAMPLE_IDS_ARG=()
CONDITIONS_ARG=()
if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
    LIMIT_ARG=(--limit "$2")
fi
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        OVERWRITE_FLAG="--overwrite"
    fi
    if [[ "$arg" == --workers=* ]]; then
        WORKERS="${arg#--workers=}"
    fi
    if [[ "$arg" == --sample_ids=* ]]; then
        SAMPLE_IDS_ARG=(--sample_ids "${arg#--sample_ids=}")
    fi
    if [[ "$arg" == --conditions=* ]]; then
        CONDITIONS_ARG=(--conditions "${arg#--conditions=}")
    fi
done

PROJECT_ROOT=/data/gpfs/projects/punim2960/jiahengd
INFER_DIR="${PROJECT_ROOT}/evaluation/simulator/inference"
VENV_PATH="${PROJECT_ROOT}/venv"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "ERROR: base venv not found at ${VENV_PATH}." >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"

set -a
[[ -f "${PROJECT_ROOT}/.env" ]] && source "${PROJECT_ROOT}/.env"
set +a
if [[ "${USE_OPENAI_FALLBACK:-}" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY_FALLBACK:-}" ]]; then
    echo "ERROR: USE_OPENAI_FALLBACK=1 but OPENAI_API_KEY_FALLBACK is unset." >&2
    exit 1
  fi
  export OPENAI_API_KEY="${OPENAI_API_KEY_FALLBACK}"
  echo "Using OPENAI_API_KEY_FALLBACK for the model API (key value not printed)."
fi

cd "$INFER_DIR"
mkdir -p "${PROJECT_ROOT}/evaluation/simulator/logs"

echo "============================================================"
echo "  Simulator 'interrupt_user' Inference (closed-source, API model)"
echo "  Model  : ${MODEL}"
echo "  Workers: ${WORKERS}"
echo "  Venv   : ${VENV_PATH}"
echo "  OPENAI_API_KEY set: $([[ -n "${OPENAI_API_KEY:-}" ]] && echo yes || echo NO)"
echo "  GEMINI_API_KEY set: $([[ -n "${GEMINI_API_KEY:-}" ]] && echo yes || echo NO)"
echo "  XAI_API_KEY set: $([[ -n "${XAI_API_KEY:-}" ]] && echo yes || echo NO)"
echo "  DASHSCOPE_API_KEY set: $([[ -n "${DASHSCOPE_API_KEY:-}" ]] && echo yes || echo NO)"
echo "  DASHSCOPE_WS_HOST: ${DASHSCOPE_WS_HOST:-UNSET}"
echo "  Overwrite: ${OVERWRITE_FLAG:-no}"
echo "  Conditions: ${CONDITIONS_ARG[*]:-all}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo ""

python3 run_interrupt_batch.py \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    "${LIMIT_ARG[@]}" \
    "${SAMPLE_IDS_ARG[@]}" \
    "${CONDITIONS_ARG[@]}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
