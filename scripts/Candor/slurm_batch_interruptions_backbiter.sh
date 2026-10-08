#!/bin/bash
#SBATCH --job-name=candor_intr_back
#SBATCH --partition=sapphire
#SBATCH --account=punim2960
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/logs/candor_intr_back_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/logs/candor_intr_back_%j.err

source /data/gpfs/projects/punim2960/jiahengd/venv/bin/activate

python /data/gpfs/projects/punim2960/jiahengd/scripts/Candor/batch_interruptions_backbiter.py \
    --workers 8 \
    --force \
    --results_dir /data/gpfs/projects/punim2960/jiahengd/results/interruptions_backbiter

