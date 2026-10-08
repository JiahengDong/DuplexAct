#!/bin/bash
# ==============================================================================
# slurm_infer_raon_shard.sh -- sharded direct-replay launcher for
# raon-speechchat. Sibling of slurm_infer_opensource.sh; never edits or is
# imported by it.
#
# Why a separate script: raon runs ~3.4x slower than realtime (see
# infer_raon.py's module docstring), so one l3_condition's 200 samples is
# ~4.6 h -- long enough that a --time=06:00:00 job sits behind everything else
# on a contended feit-gpu-a100. slurm_infer_opensource.sh can only slice by
# l3_condition (it forwards --limit, not --sample_ids), so this script slices a
# condition into N contiguous shards by sample id and forwards --sample_ids,
# which turns the run into many short jobs that backfill instead of a few long
# ones that queue.
#
# Shards are contiguous blocks of the sorted sample-id list, so shard
# membership is deterministic and reproducible from the data directory alone --
# no manifest file to keep in sync.
#
# Usage:
#   sbatch slurm_infer_raon_shard.sh <ACTION> <L3_CONDITION> <SHARD_IDX> <N_SHARDS> [--overwrite]
#
# Examples:
#   sbatch slurm_infer_raon_shard.sh wait floor_holding 0 4 --overwrite
#   sbatch slurm_infer_raon_shard.sh wait floor_holding 3 4 --overwrite
#
# NOTE --overwrite is required to re-run samples that already have results.
# Back the old ones up first: this re-run exists to ADD
# model_speech_transcripts, and the pre-existing raon results are the only copy
# of the timeline the jitter analysis was done on.
# ==============================================================================

#SBATCH --job-name=raon_shard
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs/raon_shard_%x_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/evaluation/direct_replay/logs/raon_shard_%x_%j.err
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:1

set -euo pipefail

ACTION="${1:?Usage: sbatch slurm_infer_raon_shard.sh <ACTION> <L3_CONDITION> <SHARD_IDX> <N_SHARDS> [--overwrite]}"
L3_CONDITION="${2:?Missing L3_CONDITION}"
SHARD_IDX="${3:?Missing SHARD_IDX (0-based)}"
N_SHARDS="${4:?Missing N_SHARDS}"

OVERWRITE_FLAG=""
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        OVERWRITE_FLAG="--overwrite"
    fi
done

MODEL=raon-speechchat
PROJECT_ROOT=/data/gpfs/projects/punim2960/jiahengd
OSS_ROOT="${PROJECT_ROOT}/opensource_models"
INFER_DIR="${PROJECT_ROOT}/evaluation/direct_replay/inference"
DATA_DIR="${PROJECT_ROOT}/evaluation/direct_replay/data/${ACTION}/${L3_CONDITION}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: no data dir at ${DATA_DIR}" >&2
    exit 1
fi

# Contiguous shard of the sorted sample-id list.
mapfile -t ALL_IDS < <(cd "$DATA_DIR" && for d in */; do [[ -d "$d" ]] && echo "${d%/}"; done | sort)
TOTAL="${#ALL_IDS[@]}"
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: no samples under ${DATA_DIR}" >&2
    exit 1
fi
# Ceiling division so the last shard absorbs the remainder rather than a
# separate short shard being created for it.
PER=$(( (TOTAL + N_SHARDS - 1) / N_SHARDS ))
START=$(( SHARD_IDX * PER ))
if [[ "$START" -ge "$TOTAL" ]]; then
    echo "Shard ${SHARD_IDX}/${N_SHARDS} is empty for ${ACTION}/${L3_CONDITION} (${TOTAL} samples). Nothing to do."
    exit 0
fi
COUNT=$(( TOTAL - START ))
if [[ "$COUNT" -gt "$PER" ]]; then COUNT="$PER"; fi
SHARD_IDS=$(IFS=,; echo "${ALL_IDS[*]:$START:$COUNT}")

source "${OSS_ROOT}/${MODEL}/venv/bin/activate"

# Same per-model HF cache isolation and inode policy as
# slurm_infer_opensource.sh -- see that file's comments for why
# HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE point at the same hub/ dir and
# why bytecode caching is off.
export HF_HOME="${OSS_ROOT}/${MODEL}/checkpoints"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME"

cd "$INFER_DIR"
mkdir -p "${PROJECT_ROOT}/evaluation/direct_replay/logs"

echo "============================================================"
echo "  Direct-Replay Inference -- raon shard"
echo "  Action      : ${ACTION}"
echo "  L3 condition: ${L3_CONDITION}"
echo "  Shard       : ${SHARD_IDX} of ${N_SHARDS}  (${COUNT} of ${TOTAL} samples, from index ${START})"
echo "  First / last: ${ALL_IDS[$START]} .. ${ALL_IDS[$((START + COUNT - 1))]}"
echo "  Overwrite   : ${OVERWRITE_FLAG:-no}"
echo "============================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi -L 2>&1 || echo 'nvidia-smi unavailable')"
echo ""

python3 run_inference.py \
    --action "${ACTION}" \
    --l3_condition "${L3_CONDITION}" \
    --model "${MODEL}" \
    --workers 1 \
    --sample_ids "${SHARD_IDS}" \
    ${OVERWRITE_FLAG} \
    --verbose

echo ""
echo "Job finished: $(date)"
