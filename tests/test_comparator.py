"""
Tests for ranking/comparator.py.
Uses a mock LocalStore (in-memory SQLite) and mock SimulationResults.
"""

from __future__ import annotations

import pathlib
import tempfile
import uuid

import pytest

from db.local_store import LocalStore
from db.models import SimulationResult
from ranking.comparator import ComparisonReport, compare_runs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sim(dopant: str, voltage: float = None, formation_energy: float = None,
              converged: bool = True) -> SimulationResult:
    return SimulationResult(
        dopant_element=dopant,
        dopant_oxidation_state=3,
        concentration_pct=10.0,
        sqs_realisation_index=0,
        parent_formula="LiCoO2",
        target_site_species="Co",
        mlip_name="mock",
        mlip_version="test",
        relaxation_converged=converged,
        relaxation_steps=10,
        final_energy_per_atom=-5.0,
        voltage=voltage,
        formation_energy_above_hull=formation_energy,
    )


@pytest.fixture
def two_run_store(tmp_path):
    """Store with two runs containing Al, Ti, Mg."""
    db_path = str(tmp_path / "test.db")
    store = LocalStore(db_path)
    run1 = str(uuid.uuid4())
    run2 = str(uuid.uuid4())

    # Run 1: Al best, Mg worst
    for dopant, voltage in [("Al", 4.2), ("Ti", 4.0), ("Mg", 3.8)]:
        store.save_simulation(_make_sim(dopant, voltage=voltage), run1)

    # Run 2: Ti best, Al second
    for dopant, voltage in [("Ti", 4.3), ("Al", 4.1), ("Mg", 3.9)]:
        store.save_simulation(_make_sim(dopant, voltage=voltage), run2)

    yield store, run1, run2
    store.close()


# ── Basic comparison ──────────────────────────────────────────────────────────

def test_two_runs_produces_comparison_report(two_run_store):
    store, run1, run2 = two_run_store
    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    assert isinstance(report, ComparisonReport)


def test_common_dopants_identified(two_run_store):
    store, run1, run2 = two_run_store
    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    assert set(report.dopants_compared) == {"Al", "Ti", "Mg"}


def test_property_deltas_computed(two_run_store):
    store, run1, run2 = two_run_store
    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    assert "Al" in report.property_deltas
    # Run 2 Al = 4.1, Run 1 Al = 4.2 → delta = -0.1
    assert report.property_deltas["Al"]["voltage"] == pytest.approx(-0.1, abs=1e-6)


# ── Ranking changes ───────────────────────────────────────────────────────────

def test_ranking_change_detected(two_run_store):
    """Al was rank 1 in run1, rank 2 in run2 — should be reported."""
    store, run1, run2 = two_run_store
    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    assert len(report.ranking_changes) > 0
    # At least one change should mention Al or Ti
    assert any("Al" in c or "Ti" in c for c in report.ranking_changes)


def test_no_ranking_change_when_same_order(tmp_path):
    """If both runs have the same order, no ranking changes."""
    db_path = str(tmp_path / "test.db")
    store = LocalStore(db_path)
    run1, run2 = str(uuid.uuid4()), str(uuid.uuid4())

    for run in [run1, run2]:
        for dopant, v in [("Al", 4.2), ("Ti", 4.0), ("Mg", 3.8)]:
            store.save_simulation(_make_sim(dopant, voltage=v), run)

    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    assert report.ranking_changes == []
    store.close()


# ── Spearman ρ ────────────────────────────────────────────────────────────────

def test_spearman_rho_computed(two_run_store):
    store, run1, run2 = two_run_store
    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    # With 3 dopants, ρ should be computed
    assert "voltage" in report.spearman_rho
    rho = report.spearman_rho["voltage"]["rho"]
    assert -1.0 <= rho <= 1.0


# ── Missing dopant in one run ─────────────────────────────────────────────────

def test_missing_dopant_handled_gracefully(tmp_path):
    """Al in run1 but not run2 — should not crash."""
    db_path = str(tmp_path / "test.db")
    store = LocalStore(db_path)
    run1, run2 = str(uuid.uuid4()), str(uuid.uuid4())

    store.save_simulation(_make_sim("Al", voltage=4.0), run1)
    store.save_simulation(_make_sim("Ti", voltage=3.8), run1)

    store.save_simulation(_make_sim("Ti", voltage=3.9), run2)
    # No Al in run2

    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    # Should only compare Ti (common to both)
    assert "Al" not in report.dopants_compared
    assert "Ti" in report.dopants_compared
    store.close()


# ── Empty run ─────────────────────────────────────────────────────────────────

def test_empty_run_handled(tmp_path):
    """One run has no results — should not crash."""
    db_path = str(tmp_path / "test.db")
    store = LocalStore(db_path)
    run1, run2 = str(uuid.uuid4()), str(uuid.uuid4())

    store.save_simulation(_make_sim("Al", voltage=4.0), run1)
    # run2 has no results

    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    assert isinstance(report, ComparisonReport)
    store.close()


def test_no_run_ids_returns_empty_report(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = LocalStore(db_path)
    report = compare_runs([], store)
    assert report.dopants_compared == []
    assert report.property_deltas == {}
    store.close()


# ── Summary string ────────────────────────────────────────────────────────────

def test_summary_is_non_empty_string(two_run_store):
    store, run1, run2 = two_run_store
    report = compare_runs([run1, run2], store, target_properties=["voltage"])
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0
