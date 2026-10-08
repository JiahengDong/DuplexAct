#!/bin/bash
#SBATCH --job-name=candor_bc_audio_relabel
#SBATCH --partition=sapphire
#SBATCH --account=punim2960
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/logs/candor_bc_audio_relabel_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/logs/candor_bc_audio_relabel_%j.err

source /data/gpfs/projects/punim2960/jiahengd/venv/bin/activate

python /data/gpfs/projects/punim2960/jiahengd/scripts/relabel_backchannel_audio.py \
    --skip_existing
