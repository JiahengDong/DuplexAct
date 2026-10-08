#!/bin/bash
# ==============================================================================
# slurm_action_closed.sh -- closed-source (API-backed) sibling of
# slurm_action_opensource.sh, for `hold_floor` / `respond_and_recover` runs of
# gpt-realtime-2.1 / gemini-3.1-flash-live (see cue_action_common.py's own
# module docstring for why interrupt_user isn't included here). No GPU
# needed (pure network I/O against the provider's realtime API), so this
# runs on the CPU `sapphire` partition instead of feit-gpu-a100 -- keeps it
# decoupled from any interactive GPU allocation's own walltime, and frees
# the a100 queue for the actual local-model jobs. Never edits/touches
# slurm_action_opensource.sh or slurm_stop_opensource.sh.
#
# Usage:
#   sbatch slurm_action_closed.sh <ACTION> <MODEL> [LIMIT] [--overwrite] [--conditions=C1,C2] [--workers=N]
#
# Examples:
#   sbatch slurm_action_closed.sh hold_floor gpt-realtime-2.1
#   sbatch slurm_action_closed.sh hold_floor gemini-3.1-flash-live 3   # smoke test
# ==============================================================================

#SBATCH --job-name=action_infer_api
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/action_api_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/simulator/logs/action_api_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=sapphire

set -euo pipefail

ACTION="${1:?Usage: sbatch slurm_action_closed.sh <ACTION> <MODEL> [LIMIT] [--overwrite] [--conditions=C1,C2] [--workers=N]}"
MODEL="${2:?Usage: sbatch slurm_action_closed.sh <ACTION> <MODEL> [LIMIT] [--overwrite] [--conditions=C1,C2] [--workers=N]}"
LIMIT_ARG=()
OVERWRITE_FLAG=""
CONDITIONS_ARG=()
WORKERS=8
SAMPLE_IDS_ARG=()
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
    if [[ "$arg" == --workers=* ]]; then
        WORKERS="${arg#--workers=}"
    fi
    if [[ "$arg" == --sample_ids=* ]]; then
        SAMPLE_IDS_ARG=(--sample_ids "${arg#--sample_ids=}")
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
echo "  Simulator action-generalized Inference (closed-source, API model)"
echo "  Action : ${ACTION}"
echo "  Model  : ${MODEL}"
echo "  Workers: ${WORKERS}"
echo "  Venv   : ${VENV_PATH}"
echo "  Overwrite: ${OVERWRITE_FLAG:-no}"
echo "  DASHSCOPE_API_KEY set: $([[ -n "${DASHSCOPE_API_KEY:-}" ]] && echo yes || echo NO)"
echo "  DASHSCOPE_WS_HOST: ${DASHSCOPE_WS_HOST:-UNSET}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo ""

python3 run_action_batch.py \
    --action "${ACTION}" \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    "${LIMIT_ARG[@]}" \
    "${CONDITIONS_ARG[@]}" \
    "${SAMPLE_IDS_ARG[@]}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
