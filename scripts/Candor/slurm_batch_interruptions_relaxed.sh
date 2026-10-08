#!/bin/bash
#SBATCH --job-name=candor_intr_relaxed
#SBATCH --partition=sapphire
#SBATCH --account=punim2960
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/logs/Candor_relaxed/candor_intr_relaxed_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/logs/Candor_relaxed/candor_intr_relaxed_%j.err

module purge
source /data/gpfs/projects/punim2960/jiahengd/venv/bin/activate

SCRIPT_DIR="/data/gpfs/projects/punim2960/jiahengd/scripts/Candor"
RESULTS_DIR="/data/gpfs/projects/punim2960/jiahengd/results/Candor/interruptions_final_relaxed"
LOG_DIR="/data/gpfs/projects/punim2960/jiahengd/logs/Candor_relaxed"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "CANDOR Interruptions — RELAXED (final)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Started  : $(date)"
echo "Results  : $RESULTS_DIR"
echo "=========================================="

# ── Criteria to relax ─────────────────────────────────────────────────────────
# Add or remove --skip_* flags below to control which checks are active.
#
# Timing checks:
#   --skip_abab              skip A-B-A-B speaker alternation
#   --skip_t1_sent_final     skip T1 sentence-final punctuation
#   --skip_t1_pauses         skip T1 internal pause check
#   --skip_t1_words          skip T1 minimum word count
#   --skip_t1_backchannel    skip reject-if-B-speaks-during-T1
#   --skip_t2_overlap_t1     skip "T2 must not start before T1 ends"
#   --max_t2_t1_overlap N    max T2-before-T1 overlap allowed (default 1.5s; 0.05=strict)
#   --skip_t2_question       skip "T2 must not be a question"
#   --skip_t2_words          skip T2 minimum word count
#   --skip_t2_pre_words      skip T2 minimum pre-overlap word count
#   --skip_t2_backchannel    skip reject-if-A-speaks-during-T2-before-T3
#   --skip_t3_overlap        skip T3/T2 overlap requirement
#   --min_t3_overlap N       minimum T3/T2 overlap (default 0.3s; set via MIN_T3_OVERLAP above)
#   --skip_t3_backchannel    skip reject-if-B-speaks-during-T3
#   --skip_t3_pauses         skip T3 internal pause check
#   --skip_t3_duration       skip T3 max-duration cap
#   --skip_t4_overlap_t3     skip "T4 must not start before T3 ends"
#   --max_t4_t3_overlap N    max T4-before-T3 overlap allowed (default 1.5s; 0.05=strict)
#   --skip_t4_words          skip T4 minimum word count
#   --min_words_filter 1     lower global pre-filter (default 3; 0=off)
#
# GPT validity checks (only relevant when --no_gpt is NOT set):
#   --skip_gpt_t1_complete
#   --skip_gpt_t1_self_contained
#   --skip_gpt_t2_responds
#   --skip_gpt_t3_floor_intent       skip T3 floor-taking intent check
#   --skip_gpt_t3_pragmatic_complete skip T3 pragmatic completeness check
#   --skip_gpt_t3_mid_flow           skip T3 mid-flow onset check
#   --skip_gpt_t4_meaningful
# ─────────────────────────────────────────────────────────────────────────────

# ── Overlap / timing thresholds ───────────────────────────────────────────────
# T3–T2 minimum overlap: 300ms = ~1 syllable, well above the 50ms noise floor.
#   Suggested range: 0.15 (permissive) → 0.30 (default) → 0.50 (conservative)
MIN_T3_OVERLAP=0.05

# T2/T1 and T4/T3 maximum overlap: how much the "response" or "recovery" turn
# can start before the previous turn ends without being rejected.
# 1.5s matches the turn-taking extractor; use 0.05 to restore strict mode.
MAX_T2_T1_OVERLAP=1.5
MAX_T4_T3_OVERLAP=1.5
# ─────────────────────────────────────────────────────────────────────────────

python -u "$SCRIPT_DIR/batch_interruptions_relaxed.py" \
    --workers 8 \
    --force \
    --results_dir "$RESULTS_DIR" \
    --min_t3_overlap    ${MIN_T3_OVERLAP} \
    --max_t2_t1_overlap ${MAX_T2_T1_OVERLAP} \
    --max_t4_t3_overlap ${MAX_T4_T3_OVERLAP} \
    --skip_gpt_t1_self_contained \
    --skip_gpt_t2_responds \
    --skip_t1_backchannel \
    --skip_t2_backchannel \
    --skip_t3_backchannel

echo ""
echo "Finished : $(date)"
