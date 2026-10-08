#!/bin/bash
#SBATCH --job-name=candor_bc_relaxed
#SBATCH --partition=sapphire
#SBATCH --account=punim2960
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2960/jiahengd/logs/Candor_relaxed/candor_bc_relaxed_%j.out
#SBATCH --error=/data/gpfs/projects/punim2960/jiahengd/logs/Candor_relaxed/candor_bc_relaxed_%j.err

# ════════════════════════════════════════════════════════════════════════════════
# CANDOR Backchannel Extractor — Relaxed / Configurable Edition
# ════════════════════════════════════════════════════════════════════════════════
#
# ── TASK SELECTION ──────────────────────────────────────────────────────────────
#   Set TASK to one of:  backchannel
#
#   backchannel   : Use case — give T1 to a model, ask it to generate a BC reply.
#                   Validity requires T1 to be complete and self-contained.
#

# ════════════════════════════════════════════════════════════════════════════════
# CRITERIA REFERENCE
# ════════════════════════════════════════════════════════════════════════════════
#
# ┌─────────────────────────┬─────────────────┬─────────────────┬─────────────┐
# │ Criterion               │ backchannel     │ floor_holding   │ both        │
# ├─────────────────────────┼─────────────────┼─────────────────┼─────────────┤
# │ TIMING FILTER                                                              │
# ├─────────────────────────┼─────────────────┼─────────────────┼─────────────┤
# │ T0 same-speaker check   │ OFF (T0=context)│ ON              │ ON          │
# │ T0 min-words check      │ OFF             │ ON (≥ 4 words)  │ ON          │
# │ T0 no-internal-pause    │ OFF             │ ON              │ ON          │
# │ T0→T1 timing (T1 after) │ OFF             │ ON (no overlap) │ ON          │
# │ T1 min-words check      │ ON (≥ 4 words)  │ ON              │ ON          │
# │ T1 no-internal-pause    │ ON              │ ON              │ ON          │
# │ T1 not-interrupted (*)  │ ON (tol=0.05s)  │ ON (tol=0.05s)  │ ON          │
# │ After-BC min-words      │ ON (≥ 4 words)  │ ON              │ ON          │
# │ BC count == 1 (**)      │ ON (hardcoded)  │ ON (hardcoded)  │ ON          │
# ├─────────────────────────┼─────────────────┼─────────────────┼─────────────┤
# │ GPT FILTER                                                                 │
# ├─────────────────────────┼─────────────────┼─────────────────┼─────────────┤
# │ T0 complete             │ NOT ASKED       │ REQUIRED        │ REQUIRED    │
# │ T0 self-contained       │ NOT ASKED       │ REQUIRED        │ REQUIRED    │
# │ T1 complete             │ REQUIRED        │ REQUIRED        │ REQUIRED    │
# │ T1 self-contained       │ REQUIRED        │ NOT ASKED       │ REQUIRED    │
# └─────────────────────────┴─────────────────┴─────────────────┴─────────────┘
# (*) T1 not-interrupted: the next speaker must not start before T1 ends minus
#     max_t0_t1_overlap seconds. Default 0.05s absorbs annotation noise. Increase (e.g.
#     0.3s) to allow small natural overlaps without rejecting the candidate.
# (**) BC count == 1: T1 must contain exactly one distinct backchannel event from
#     B (as recorded by Backbiter). 0 = no BC present; 2+ = multiple BCs during
#     T1, making the episode ambiguous. Not currently skippable.
#
# ── TIMING SKIP FLAGS (set to empty "" to keep active, or the flag to skip) ───
#
# --skip_t0          : Disable ALL T0-side checks in one go (same-speaker, words,
#                      pause, timing).  Only relevant for floor_holding/both since
#                      backchannel already has T0 checks OFF by default.
# --skip_t0_words    : Allow T0 with fewer than min_t0_words words.
# --skip_t0_pauses   : Allow T0 with an internal pause (e.g. interviewer pausing
#                      to think — common in CORAAL / Loqui scenarios).
# --skip_t0_timing   : Allow T1 to start before T0 ends (T0↔T1 overlap OK).
# --skip_t1_words    : Allow T1 with fewer than min_t1_words words.
# --skip_t1_pauses   : Allow T1 with an internal pause.  Useful for corpora where
#                      speakers pause mid-thought (Loqui librarians, etc.).
# --skip_t1_interrupted : Allow T1 even if the next speaker cuts in before T1 ends.
# --skip_after_words : Allow fewer than min_after_words after the backchannel.
#
# ── THRESHOLD OVERRIDES (defaults shown) ────────────────────────────────────
# --min_t0_words 4   : T0 must have at least this many words (floor_holding/both).
# --min_t1_words 4   : T1 must have at least this many words.
# --min_after_words 4: A must say at least this many words after the backchannel.
#
# ── GPT SKIP FLAGS ──────────────────────────────────────────────────────────
# --skip_gpt_t0_complete       : Do not require GPT to confirm T0 is complete.
# --skip_gpt_t0_self_contained : Do not require GPT to confirm T0 is self-contained.
# --skip_gpt_t1_complete       : Do not require GPT to confirm T1 is complete.
# --skip_gpt_t1_self_contained : Do not require GPT to confirm T1 is self-contained.
#
# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURE YOUR RUN BELOW — uncomment / edit flags as needed
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════

source /data/gpfs/projects/punim2960/jiahengd/venv/bin/activate

SCRIPT_DIR="/data/gpfs/projects/punim2960/jiahengd/scripts/Candor"
TASK="backchannel"

echo "=========================================="
echo "CANDOR Backchannel (Relaxed)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Task     : $TASK"
echo "Started  : $(date)"
echo "=========================================="

RESULTS_DIR="/data/gpfs/projects/punim2960/jiahengd/results/Candor/Relaxed/backchannel_relaxed_t0_self_contained"

python -u "$SCRIPT_DIR/batch_floor_backchannel_backbiter_relaxed.py" \
    --workers 8 \
    --force \
    --task "$TASK" \
    --results_dir "$RESULTS_DIR" \
    --max_t0_t1_overlap 1.5 \
    --skip_gpt_t1_self_contained \

    # ── Threshold overrides (uncomment + set value to change from default) ────
    # --min_t0_words 3          \
    # --min_t1_words 3          \
    # --min_after_words 2       \
    # --max_t0_t1_overlap 0.3            \   # default 0.05s; increase to allow small overlaps
    # ── GPT skip flags (uncomment to relax) ─────────────────────────────────
    # --skip_gpt_t1_complete            \
    # --skip_gpt_t1_self_contained      \
    # --skip_gpt_t0_complete            \   # floor_holding/both only
    # --skip_gpt_t0_self_contained          # floor_holding/both only

echo ""
echo "Finished : $(date)"
