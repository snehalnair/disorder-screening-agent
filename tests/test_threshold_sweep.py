"""Tests for evaluation/threshold_sweep.py."""

import pytest
from evaluation.threshold_sweep import SweepPoint, sweep_stage3_threshold


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nmc_sweep():
    """Run the sweep once and reuse across tests (slow: runs pipeline 6 times)."""
    return sweep_stage3_threshold(
        parent_formula="LiNi0.8Mn0.1Co0.1O2",
        target_species="Co",
        target_os=3,
        target_cn=6,
        thresholds=[0.001, 0.005, 0.01],   # subset for speed
    )


# ── Output shape ──────────────────────────────────────────────────────────────

def test_returns_list_of_sweep_points(nmc_sweep):
    assert isinstance(nmc_sweep, list)
    assert all(isinstance(p, SweepPoint) for p in nmc_sweep)


def test_one_result_per_threshold(nmc_sweep):
    assert len(nmc_sweep) == 3


def test_sweep_point_fields(nmc_sweep):
    for pt in nmc_sweep:
        assert hasattr(pt, "threshold")
        assert hasattr(pt, "n_stage3_survivors")
        assert hasattr(pt, "recall")
        assert hasattr(pt, "missed_dopants")
        assert isinstance(pt.missed_dopants, list)


# ── Monotonicity ──────────────────────────────────────────────────────────────

def test_higher_threshold_fewer_survivors(nmc_sweep):
    """Higher threshold → fewer or equal candidates."""
    survivors = [p.n_stage3_survivors for p in nmc_sweep]
    for i in range(len(survivors) - 1):
        assert survivors[i] >= survivors[i + 1], (
            f"Non-monotone: threshold {nmc_sweep[i].threshold} gave {survivors[i]} "
            f"but {nmc_sweep[i+1].threshold} gave {survivors[i+1]}"
        )


def test_higher_threshold_lower_or_equal_recall(nmc_sweep):
    """Higher threshold → recall can only decrease or stay the same."""
    recalls = [p.recall for p in nmc_sweep]
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1] - 1e-6, (
            f"Recall increased from {recalls[i]:.1%} to {recalls[i+1]:.1%} "
            f"as threshold went from {nmc_sweep[i].threshold} to {nmc_sweep[i+1].threshold}"
        )


# ── Recall at default threshold ───────────────────────────────────────────────

def test_default_threshold_recall(nmc_sweep):
    """At threshold=0.001 (default), recall must be ≥90%."""
    default_pt = next(p for p in nmc_sweep if p.threshold == 0.001)
    assert default_pt.recall >= 0.90, (
        f"Recall at default threshold {default_pt.recall:.1%} < 90%"
    )


# ── Custom thresholds ─────────────────────────────────────────────────────────

def test_custom_thresholds():
    sweep = sweep_stage3_threshold(
        parent_formula="LiNi0.8Mn0.1Co0.1O2",
        target_species="Co",
        target_os=3,
        thresholds=[0.001, 0.01],
    )
    assert len(sweep) == 2
    thresholds = [p.threshold for p in sweep]
    assert 0.001 in thresholds
    assert 0.01 in thresholds
