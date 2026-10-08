#!/bin/bash
#SBATCH --job-name=candor_pause_no_gpt
#SBATCH --partition=sapphire
#SBATCH --account=punim2960
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/logs/candor_pause_no_gpt_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/logs/candor_pause_no_gpt_%j.err

module purge
source /data/gpfs/projects/punim2960/jiahengd/venv/bin/activate

SCRIPT_DIR="/data/gpfs/projects/punim2960/jiahengd/scripts/Candor"
RESULTS_DIR="/data/gpfs/projects/punim2960/jiahengd/results/pauses_no_gpt"
LOG_DIR="/data/gpfs/projects/punim2960/jiahengd/logs"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "CANDOR Pause — TIMING ONLY (no GPT)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Started  : $(date)"
echo "Results  : $RESULTS_DIR"
echo "=========================================="

python -u "$SCRIPT_DIR/batch_pause_backbiter.py" \
    --workers 8 \
    --skip_gpt \
    --force \
    --results_dir "$RESULTS_DIR"

echo ""
echo "Finished : $(date)"
