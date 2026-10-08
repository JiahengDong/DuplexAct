#!/bin/bash
# ==============================================================================
# slurm_infer.sh -- parameterized direct-replay inference launcher
#
# Runs run_inference.py for one (action, l3_condition, model) combination.
# Inference is wall-clock bound (each sample takes ~its own audio duration to
# stream), so --time is generous; actual wall time scales with
# (n_samples / workers) * avg_sample_duration.
#
# Usage:
#   sbatch slurm_infer.sh <ACTION> <L3_CONDITION|all> <MODEL> [WORKERS] [--overwrite]
#
# Examples:
#   sbatch slurm_infer.sh wait floor_holding gpt-realtime-2.1
#   sbatch slurm_infer.sh backchannel all gpt-realtime-2.1 8
#   sbatch slurm_infer.sh turn_taking take_turn_with_overlap gpt-realtime-2.1 4 --overwrite
#
# Required environment:
#   API key for the chosen model must be set in
#   /data/gpfs/projects/punim2960/jiahengd/.env (OPENAI_API_KEY, GEMINI_API_KEY,
#   XAI_API_KEY, DASHSCOPE_API_KEY, or AWS credentials for Nova Sonic).
# ==============================================================================

#SBATCH --job-name=direct_replay_infer
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs/infer_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs/infer_%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=sapphire

set -euo pipefail

ACTION="${1:?Usage: sbatch slurm_infer.sh <ACTION> <L3_CONDITION|all> <MODEL> [WORKERS] [--overwrite]}"
L3_CONDITION="${2:?Missing L3_CONDITION (use 'all' to run every l3_condition for this action)}"
MODEL="${3:?Missing MODEL slug}"
WORKERS="${4:-4}"
OVERWRITE_FLAG=""
SAMPLE_IDS_ARG=()
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        OVERWRITE_FLAG="--overwrite"
    fi
    if [[ "$arg" == --sample_ids=* ]]; then
        SAMPLE_IDS_ARG=(--sample_ids "${arg#--sample_ids=}")
    fi
done

source /data/gpfs/projects/punim2960/jiahengd/venv/bin/activate

# Avoid __pycache__ .pyc regeneration -- the project filesystem's 500k-file
# inode quota, not disk space, is the binding constraint (see
# slurm_infer_opensource.sh's matching comment for the full story).
export PYTHONDONTWRITEBYTECODE=1

ENV_FILE=/data/gpfs/projects/punim2960/jiahengd/.env
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

cd /data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/inference
mkdir -p /data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs

L3_ARG=()
if [[ "$L3_CONDITION" != "all" ]]; then
    L3_ARG=(--l3_condition "$L3_CONDITION")
fi

echo "============================================================"
echo "  Direct-Replay Inference"
echo "  Action      : ${ACTION}"
echo "  L3 condition: ${L3_CONDITION}"
echo "  Model       : ${MODEL}"
echo "  Workers     : ${WORKERS}"
echo "  Overwrite   : ${OVERWRITE_FLAG:-no}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo ""

python3 run_inference.py \
    --action "${ACTION}" \
    "${L3_ARG[@]}" \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    "${SAMPLE_IDS_ARG[@]}" \
    ${OVERWRITE_FLAG}

echo ""
echo "Job finished: $(date)"
