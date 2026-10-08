#!/bin/bash
#SBATCH --job-name=candor_pause
#SBATCH --partition=sapphire
#SBATCH --account=punim2960
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/logs/candor_pause_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/logs/candor_pause_%j.err

source /data/gpfs/projects/punim2960/jiahengd/venv/bin/activate

python /data/gpfs/projects/punim2960/jiahengd/scripts/Candor/batch_pause_backbiter.py \
    --workers 16 \
    --force \
    --cap_per_type 500 \
    --results_dir /data/gpfs/projects/punim2960/jiahengd/results/Candor/pause
