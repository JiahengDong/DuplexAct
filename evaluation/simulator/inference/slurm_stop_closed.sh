#!/bin/bash
# ==============================================================================
# slurm_stop_closed.sh -- closed-source (API-backed) sibling of
# slurm_stop_opensource.sh, for gpt-realtime-2.1 / gemini-3.1-flash-live
# `stop` runs. Same sapphire CPU layout as slurm_action_closed.sh (no GPU;
# network I/O against the provider realtime API). Never edits/touches
# slurm_stop_opensource.sh or slurm_action_closed.sh.
#
# Usage:
#   sbatch slurm_stop_closed.sh <MODEL> [LIMIT] [--overwrite] [--workers=N]
#
# Examples:
#   sbatch slurm_stop_closed.sh gpt-realtime-2.1
#   sbatch slurm_stop_closed.sh gemini-3.1-flash-live 3   # smoke test
# ==============================================================================

#SBATCH --job-name=stop_infer_api
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/stop_api_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/stop_api_%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=sapphire

set -euo pipefail

MODEL="${1:?Usage: sbatch slurm_stop_closed.sh <MODEL> [LIMIT] [--overwrite] [--workers=N]}"
LIMIT_ARG=()
OVERWRITE_FLAG=""
WORKERS=8
SAMPLE_IDS_ARG=()
RESULTS_ROOT_ARG=()
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
    if [[ "$arg" == --results_root=* ]]; then
        RESULTS_ROOT_ARG=(--results_root "${arg#--results_root=}")
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
  echo "Using OPENAI_API_KEY_FALLBACK for the trigger LLM (key value not printed)."
fi

cd "$INFER_DIR"
mkdir -p "${PROJECT_ROOT}/evaluation/simulator/logs"

echo "============================================================"
echo "  Simulator 'stop'-action Inference (closed-source, API model)"
echo "  Model  : ${MODEL}"
echo "  Workers: ${WORKERS}"
echo "  Venv   : ${VENV_PATH}"
echo "  Overwrite: ${OVERWRITE_FLAG:-no}"
echo "  Results  : ${RESULTS_ROOT_ARG[*]:-default stop/}"
echo "  DASHSCOPE_API_KEY set: $([[ -n "${DASHSCOPE_API_KEY:-}" ]] && echo yes || echo NO)"
echo "  DASHSCOPE_WS_HOST: ${DASHSCOPE_WS_HOST:-UNSET}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo ""

python3 run_stop_batch.py \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    "${LIMIT_ARG[@]}" \
    "${SAMPLE_IDS_ARG[@]}" \
    "${RESULTS_ROOT_ARG[@]}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
