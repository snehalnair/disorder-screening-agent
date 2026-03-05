"""Tests for stages/registry.py — stage metadata registry."""

import pytest
from stages.registry import (
    ALL_STAGES,
    get_gpu_stages,
    get_stage_metadata,
    get_structure_required_stages,
    get_total_cost_estimate,
    validate_registry,
)

_REQUIRED_KEYS = {"name", "stage", "cost", "requires_gpu"}
_EXPECTED_STAGE_IDS = {1, 2, 3, "4ml", "4v", "5a", "5b", "5c"}


# ── Registry completeness ─────────────────────────────────────────────────────

def test_all_stages_present():
    """All 8 stages must be registered."""
    assert set(ALL_STAGES.keys()) == _EXPECTED_STAGE_IDS


def test_all_metadata_have_required_keys():
    """Every stage metadata dict must have name, stage, cost, requires_gpu."""
    errors = validate_registry()
    assert not errors, f"Registry validation errors: {errors}"


def test_get_stage_metadata_returns_correct_entry():
    meta = get_stage_metadata(1)
    assert meta["name"] == "smact_filter"
    assert meta["stage"] == 1

    meta4 = get_stage_metadata("4ml")
    assert meta4["name"] == "ml_prescreen"


def test_get_stage_metadata_invalid_key_raises():
    with pytest.raises(KeyError):
        get_stage_metadata(99)


# ── GPU stage detection ───────────────────────────────────────────────────────

def test_get_gpu_stages_returns_4ml_5b_5c():
    """Stage 4ml (ML inference), 5b (MLIP relaxation), and 5c (property calc) require GPU."""
    gpu_stages = get_gpu_stages()
    assert set(gpu_stages) == {"4ml", "5b", "5c"}, (
        f"Expected GPU stages {{4ml, 5b, 5c}}, got {set(gpu_stages)}"
    )


def test_stages_1_2_3_do_not_require_gpu():
    for sid in (1, 2, 3):
        assert not get_stage_metadata(sid)["requires_gpu"], (
            f"Stage {sid} should not require GPU"
        )


# ── Structure requirement detection ──────────────────────────────────────────

def test_stages_1_2_3_do_not_require_structure():
    for sid in (1, 2, 3):
        assert not get_stage_metadata(sid)["requires_structure"], (
            f"Stage {sid} should not require structure"
        )


def test_stages_5_require_structure():
    structure_stages = get_structure_required_stages()
    assert "5a" in structure_stages
    assert "5b" in structure_stages
    assert "5c" in structure_stages


# ── Cost estimate ─────────────────────────────────────────────────────────────

def test_cost_estimate_contains_relaxation_count():
    result = get_total_cost_estimate(n_candidates=5, n_concentrations=2, n_sqs=5)
    # 5 × 2 × 5 = 50 relaxations
    assert "50 relaxations" in result


def test_cost_estimate_is_string():
    result = get_total_cost_estimate(n_candidates=3, n_concentrations=2, n_sqs=3)
    assert isinstance(result, str)
    assert "Stage 5" in result


# ── Stage 4 metadata ──────────────────────────────────────────────────────────

def test_stage4ml_has_caveat():
    meta = get_stage_metadata("4ml")
    assert "caveat" in meta
    assert "ORDERED" in meta["caveat"]


# ── Stage 5 metadata ──────────────────────────────────────────────────────────

def test_stage5b_failure_modes():
    meta = get_stage_metadata("5b")
    assert "energy_divergence" in meta["failure_modes"]
    assert "volume_explosion" in meta["failure_modes"]
    assert "stagnation" in meta["failure_modes"]
    assert "force_spike" in meta["failure_modes"]
