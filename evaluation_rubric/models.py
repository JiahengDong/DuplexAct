"""
Shared data models for the Full-Duplex Level-2 Rubric evaluation framework.

Hierarchy: Level-2 rubric → Level-3 condition → EvalInstance → InstanceRecord
           InstanceRecords → L3Result → L2Result
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─── Input ──────────────────────────────────────────────────────────────────

@dataclass
class EvalInstance:
    """
    One observed test case nested under a Level-3 execution condition.

    Timestamp conventions (all in seconds from interaction start):
        Each action module documents which keys it reads from `timestamps`.
        Common keys:
            t_user_end          – user turn completion (Smooth Turn Taking)
            t_user_int_start    – user interruption onset (Stop)
            t_wait_start        – start of the expected wait window (Wait)
            t_wait_end          – end of the expected wait window (Wait)
            t_release           – user's explicit release/completion cue (Wait)
            t_other_start       – irrelevant/background speech onset (Ignore, Hold Floor)
            t_other_end         – irrelevant/background speech offset (Ignore, Hold Floor)
            t_cue_start         – floor-retention cue onset (Hold Floor)
            t_cue_end           – floor-retention cue offset (Hold Floor)
            t_target            – target interruption/backchannel point (Interrupt, Backchannel)
            t_recovery_ref      – reference point for recovery window (Respond & Recover)
    """
    instance_id: str
    l2_action: str
    l3_condition: str
    rubric_version: str = "1.0"

    # Named event timestamps (seconds from interaction start)
    timestamps: dict[str, float] = field(default_factory=dict)

    # Model VAD output: list of (onset_s, offset_s) segments
    model_speech_segments: list[tuple[float, float]] = field(default_factory=list)

    # Text evidence
    transcript_user: Optional[str] = None          # user turn text
    transcript_model: Optional[str] = None         # model output text
    transcript_context: Optional[str] = None       # prior turns (context window)
    transcript_original_plan: Optional[str] = None # intended response before interruption

    # File-system references (label → path)
    audio_paths: dict[str, str] = field(default_factory=dict)

    # Condition-specific metadata (e.g. subcategory, scenario family)
    metadata: dict = field(default_factory=dict)


# ─── Per-instance scored record ──────────────────────────────────────────────

@dataclass
class InstanceRecord:
    """
    Complete scored record for one evaluation instance, per Section 5.3 of the rubric.

    Store raw B_i, T_i, evidence vector, and failure tags.
    Do not aggregate during marking; aggregation happens at Level-3 and Level-2.
    """
    instance_id: str
    l2_action: str
    l3_condition: str
    rubric_version: str

    # Observable (Section 1.1)
    B_i: int                                    # binary success: 0 or 1
    T_i_raw: Optional[float]                    # signed timing diagnostic (seconds); None = absent
    reference_time: Optional[float]             # timing anchor used for T_i (e.g. t_user_end)
    event_time: Optional[float]                 # observed event time (e.g. t_model_start)

    # Reliability (Section 1.1)
    e_i: list[int]                              # binary per criterion, same order as criterion_labels
    applicability_mask: list[bool]              # True = criterion applies to this instance
    R_i: float                                  # mean(e_i,k) over applicable items
    criterion_labels: list[str]                 # short descriptive label per criterion

    # Diagnostics (Section 5.3)
    evidence_spans: list[str]                   # supporting text/audio citations
    failure_tag: Optional[str]                  # e.g. 'window_miss', 'no_event', 'missing_timestamp'
    judge_source: str                           # 'auto', 'llm:gpt-4o', 'human'
    confidence_metadata: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def V_i(self) -> float:
        """Supplementary combined score B_i × R_i. Never replaces decomposed B_i / R_i."""
        return float(self.B_i) * self.R_i


# ─── Aggregated results ──────────────────────────────────────────────────────

@dataclass
class L3Result:
    """Aggregated result for one Level-3 execution condition (Section 4.2)."""
    l2_action: str
    l3_condition: str
    n_instances: int
    B_c: float                                  # mean B_i
    R_c: float                                  # mean R_i
    timing_median: Optional[float]              # median T_i (seconds)
    timing_iqr: Optional[tuple[float, float]]   # (Q1, Q3) of T_i
    criterion_means: list[float]                # mean e_i,k per criterion
    criterion_labels: list[str]
    coverage: float                             # fraction of instances with non-missing event_time


@dataclass
class L2Result:
    """
    Aggregated result for one Level-2 action (macro-average of Level-3 children).
    Each L3 child contributes equally by default (Section 4.2).
    """
    l2_action: str
    B_l2: float                                 # macro-average of B_c over conditions
    R_l2: float                                 # macro-average of R_c over conditions
    timing_summary: Optional[dict]              # combined timing diagnostics
    children: list[L3Result]                    # all child L3 results preserved for publication
