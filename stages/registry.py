"""
Stage metadata registry.

Aggregates TOOL_METADATA from every pipeline stage module and exposes helpers
used by the Level 2 planner to make informed decisions about which stages to
invoke, what resources they require, and what cost to expect.

Stage 5 sub-modules are stubs pending Phase 3 implementation.
"""

from __future__ import annotations

from stages.stage1_smact import TOOL_METADATA as STAGE1_META
from stages.stage2_radius import TOOL_METADATA as STAGE2_META
from stages.stage3_substitution import TOOL_METADATA as STAGE3_META
from stages.stage4_ml_prescreen import TOOL_METADATA as STAGE4_ML_META
from stages.stage4_viability import TOOL_METADATA as STAGE4V_META
from stages.stage5.sqs_generator import TOOL_METADATA as STAGE5A_META
from stages.stage5.mlip_relaxation import TOOL_METADATA as STAGE5B_META
from stages.stage5.property_calculator import TOOL_METADATA as STAGE5C_META

# ── Registry ──────────────────────────────────────────────────────────────────

ALL_STAGES: dict = {
    1: STAGE1_META,
    2: STAGE2_META,
    3: STAGE3_META,
    "4ml": STAGE4_ML_META,   # optional ML pre-screen (disabled by default)
    "4v": STAGE4V_META,      # element viability filter (always-on by default)
    "5a": STAGE5A_META,
    "5b": STAGE5B_META,
    "5c": STAGE5C_META,
}

_REQUIRED_KEYS = {"name", "stage", "cost", "requires_gpu"}


# ── Public helpers ────────────────────────────────────────────────────────────

def get_stage_metadata(stage_id: int | str) -> dict:
    """Return the TOOL_METADATA dict for the given stage ID."""
    return ALL_STAGES[stage_id]


def get_gpu_stages() -> list:
    """Return stage IDs that require a GPU."""
    return [k for k, v in ALL_STAGES.items() if v.get("requires_gpu")]


def get_structure_required_stages() -> list:
    """Return stage IDs that require a 3-D crystal structure."""
    return [k for k, v in ALL_STAGES.items() if v.get("requires_structure")]


def get_total_cost_estimate(
    n_candidates: int,
    n_concentrations: int,
    n_sqs: int,
) -> str:
    """Return a rough human-readable compute-time estimate for the full pipeline.

    Args:
        n_candidates:    Number of dopants entering Stage 5.
        n_concentrations: Concentration points per dopant.
        n_sqs:           SQS realisations per (dopant, concentration).

    Returns:
        A string like ``"Stages 1–3: <30s | Stage 5: ~150 min (50 relaxations × ~30 min each)"``.
    """
    stage5_runs = n_candidates * n_concentrations * n_sqs
    return (
        f"Stages 1–3: <30s | "
        f"Stage 5: ~{stage5_runs * 30} min ({stage5_runs} relaxations × ~30 min each)"
    )


def validate_registry() -> list[str]:
    """Check that every stage metadata dict contains the required keys.

    Returns:
        List of validation error strings (empty if all OK).
    """
    errors: list[str] = []
    for stage_id, meta in ALL_STAGES.items():
        missing = _REQUIRED_KEYS - set(meta.keys())
        if missing:
            errors.append(f"Stage {stage_id} missing keys: {missing}")
    return errors
