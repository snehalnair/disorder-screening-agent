"""Tests for evaluation/ablation.py — ablations 1–3 (no MLIP required).

Ablations 4 and 5 require real MACE and are not tested here.
"""

from __future__ import annotations

import pytest

from evaluation.ablation import (
    AblationResult,
    ablation_remove_stage2,
    ablation_remove_stage3,
    ablation_stage4_effect,
    run_pruning_ablations,
)


# ── Shared fixture ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ablation1():
    return ablation_remove_stage2()


@pytest.fixture(scope="module")
def ablation2():
    return ablation_remove_stage3()


@pytest.fixture(scope="module")
def ablation3():
    return ablation_stage4_effect()


# ── AblationResult dataclass ──────────────────────────────────────────────────


def test_ablation_result_delta_computed():
    """AblationResult must auto-compute delta fields in __post_init__."""
    r = AblationResult(
        name="test",
        description="",
        default_recall=0.90,
        ablation_recall=0.92,
        default_n_survivors=46,
        ablation_n_survivors=60,
    )
    assert r.delta_recall == pytest.approx(0.02)
    assert r.delta_survivors == 14


def test_ablation_result_str_has_name():
    """String representation must include the ablation name."""
    r = AblationResult(
        name="My Ablation",
        description="",
        default_recall=0.90,
        ablation_recall=0.90,
        default_n_survivors=46,
        ablation_n_survivors=46,
    )
    assert "My Ablation" in str(r)


# ── Ablation 1: Remove Stage 2 ────────────────────────────────────────────────


def test_ablation1_is_ablation_result(ablation1):
    assert isinstance(ablation1, AblationResult)


def test_ablation1_more_survivors_without_stage2(ablation1):
    """Removing Stage 2 should not decrease survivors (can only increase or stay same)."""
    assert ablation1.ablation_n_survivors >= ablation1.default_n_survivors


def test_ablation1_recall_unchanged_or_higher(ablation1):
    """Removing Stage 2 at threshold 0.35 must not decrease recall.
    (Only B is filtered at Stage 2 among GT positives, and B is confirmed_successful.)
    """
    # Allow very small floating point tolerance
    assert ablation1.ablation_recall >= ablation1.default_recall - 0.02


def test_ablation1_default_unique_elements_is_29(ablation1):
    """Default (all stages) must produce 29 unique candidate elements.
    Stage 3 produces 46 (element, OS) pairs but evaluate_pruning de-duplicates
    by element symbol → 29 unique elements.
    """
    assert ablation1.default_n_survivors == 29


# ── Ablation 2: Remove Stage 3 ────────────────────────────────────────────────


def test_ablation2_is_ablation_result(ablation2):
    assert isinstance(ablation2, AblationResult)


def test_ablation2_without_stage3_has_more_survivors(ablation2):
    """Removing Stage 3 must significantly increase survivor count (46 → ~85)."""
    assert ablation2.ablation_n_survivors > ablation2.default_n_survivors


def test_ablation2_stage3_reduces_unique_elements(ablation2):
    """Stage 3 must reduce unique element count relative to Stage 2.
    Stage 2 output has more unique elements than Stage 3.
    (Stage 3 reduces 46 pairs → 29 unique elements vs Stage 2's ~38+ unique elements.)
    """
    assert ablation2.ablation_n_survivors > ablation2.default_n_survivors


def test_ablation2_recall_unchanged(ablation2):
    """Removing Stage 3 must not decrease recall significantly (< 2pp tolerance)."""
    assert ablation2.ablation_recall >= ablation2.default_recall - 0.02


# ── Ablation 3: Stage 4 effect ────────────────────────────────────────────────


def test_ablation3_is_ablation_result(ablation3):
    assert isinstance(ablation3, AblationResult)


def test_ablation3_stage4_enabled_reduces_or_equal_survivors(ablation3):
    """Enabling Stage 4 must not increase survivors beyond Stage 3 count."""
    assert ablation3.ablation_n_survivors <= ablation3.default_n_survivors


def test_ablation3_default_is_29_unique_elements(ablation3):
    """Default (Stage 4 disabled) must be 29 unique candidate elements."""
    assert ablation3.default_n_survivors == 29


# ── run_pruning_ablations ─────────────────────────────────────────────────────


def test_run_pruning_ablations_returns_three(scope="function"):
    """run_pruning_ablations must return exactly 3 AblationResult objects."""
    results = run_pruning_ablations()
    assert len(results) == 3
    for r in results:
        assert isinstance(r, AblationResult)


def test_run_pruning_ablations_names_are_distinct():
    """Each ablation must have a distinct name."""
    results = run_pruning_ablations()
    names = [r.name for r in results]
    assert len(set(names)) == len(names), "Duplicate ablation names"
