"""
Tests for ranking/ranker.py.
Uses mock SimulationResult objects — no MLIP or structure required.
"""

from __future__ import annotations

import pytest

from db.models import SimulationResult
from ranking.ranker import DopantStats, RankedReport, rank_dopants


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sim(
    dopant: str,
    voltage: float = None,
    formation_energy: float = None,
    li_ni_exchange: float = None,
    volume_change: float = None,
    converged: bool = True,
    sqs_index: int = 0,
    concentration_pct: float = 10.0,
) -> SimulationResult:
    return SimulationResult(
        dopant_element=dopant,
        dopant_oxidation_state=3,
        concentration_pct=concentration_pct,
        sqs_realisation_index=sqs_index,
        parent_formula="LiNi0.8Mn0.1Co0.1O2",
        target_site_species="Co",
        mlip_name="mock",
        mlip_version="test",
        relaxation_converged=converged,
        relaxation_steps=10,
        final_energy_per_atom=-5.0,
        formation_energy_above_hull=formation_energy,
        li_ni_exchange_energy=li_ni_exchange,
        voltage=voltage,
        volume_change_pct=volume_change,
    )


# ── Basic ranking ─────────────────────────────────────────────────────────────

def test_basic_ranking_order():
    """Dopants are ranked by voltage descending (higher is better)."""
    results = [
        _make_sim("Al", voltage=4.0),
        _make_sim("Ti", voltage=3.5),
        _make_sim("Mg", voltage=4.2),
    ]
    report = rank_dopants(results, target_properties=["voltage"])
    voltage_ranking = report.all_rankings["voltage"]
    assert voltage_ranking[0] == "Mg"
    assert voltage_ranking[1] == "Al"
    assert voltage_ranking[2] == "Ti"


def test_basic_ranking_formation_energy():
    """Formation energy is ranked ascending (lower is better)."""
    results = [
        _make_sim("Al", formation_energy=0.05),
        _make_sim("Ti", formation_energy=0.01),
        _make_sim("Mg", formation_energy=0.15),
    ]
    report = rank_dopants(results, target_properties=["formation_energy"])
    fe_ranking = report.all_rankings["formation_energy"]
    assert fe_ranking[0] == "Ti"   # lowest
    assert fe_ranking[-1] == "Mg"  # highest


# ── SQS averaging ────────────────────────────────────────────────────────────

def test_sqs_averaging_mean():
    """Mean across 5 SQS realisations must be correct."""
    voltages = [3.8, 3.9, 4.0, 4.1, 4.2]
    results = [_make_sim("Al", voltage=v, sqs_index=i) for i, v in enumerate(voltages)]
    report = rank_dopants(results, target_properties=["voltage"])
    al_stats = next(d for d in report.rankings if d.dopant == "Al")
    assert al_stats.properties["voltage"]["mean"] == pytest.approx(4.0, abs=1e-6)


def test_sqs_averaging_std():
    """Std dev across SQS realisations must be correct."""
    import numpy as np
    voltages = [3.8, 3.9, 4.0, 4.1, 4.2]
    results = [_make_sim("Al", voltage=v, sqs_index=i) for i, v in enumerate(voltages)]
    report = rank_dopants(results, target_properties=["voltage"])
    al_stats = next(d for d in report.rankings if d.dopant == "Al")
    expected_std = float(np.std(voltages))
    assert al_stats.properties["voltage"]["std"] == pytest.approx(expected_std, rel=1e-5)


def test_sqs_n_count():
    """n field counts number of converged SQS values used."""
    results = [_make_sim("Al", voltage=v, sqs_index=i) for i, v in enumerate([3.8, 3.9, 4.0])]
    report = rank_dopants(results, target_properties=["voltage"])
    al_stats = next(d for d in report.rankings if d.dopant == "Al")
    assert al_stats.properties["voltage"]["n"] == 3


# ── Spearman ρ ────────────────────────────────────────────────────────────────

def test_spearman_rho_perfect_agreement():
    """Ordered and disordered give same ranking → ρ ≈ 1.0."""
    results = [
        _make_sim("Al", voltage=4.2),
        _make_sim("Ti", voltage=4.0),
        _make_sim("Mg", voltage=3.8),
    ]
    ordered = {
        "Al": {"voltage": 4.2},
        "Ti": {"voltage": 4.0},
        "Mg": {"voltage": 3.8},
    }
    report = rank_dopants(results, target_properties=["voltage"], ordered_results=ordered)
    rho_info = report.spearman_rho.get("voltage")
    assert rho_info is not None
    assert rho_info["rho"] == pytest.approx(1.0, abs=0.01)


def test_spearman_rho_reversed_ranking():
    """Ordered [Al, Ti, Mg], disordered [Mg, Ti, Al] → ρ ≈ -1.0."""
    results = [
        _make_sim("Al", voltage=3.8),   # disordered: Al last
        _make_sim("Ti", voltage=4.0),
        _make_sim("Mg", voltage=4.2),   # disordered: Mg first
    ]
    ordered = {
        "Al": {"voltage": 4.2},   # ordered: Al first
        "Ti": {"voltage": 4.0},
        "Mg": {"voltage": 3.8},   # ordered: Mg last
    }
    report = rank_dopants(results, target_properties=["voltage"], ordered_results=ordered)
    rho_info = report.spearman_rho.get("voltage")
    assert rho_info is not None
    assert rho_info["rho"] == pytest.approx(-1.0, abs=0.01)


def test_spearman_rho_too_few_dopants():
    """Only 2 dopants — Spearman ρ should not be computed (needs ≥ 3)."""
    results = [_make_sim("Al", voltage=4.0), _make_sim("Ti", voltage=3.8)]
    ordered = {"Al": {"voltage": 4.0}, "Ti": {"voltage": 3.8}}
    report = rank_dopants(results, target_properties=["voltage"], ordered_results=ordered)
    assert "voltage" not in report.spearman_rho


# ── High-variance flag ────────────────────────────────────────────────────────

def test_high_variance_flag():
    """σ > 20% of mean triggers warning."""
    # Values: 2.0, 4.0 → mean=3.0, std=1.0, CV=33%
    results = [
        _make_sim("Al", voltage=2.0, sqs_index=0),
        _make_sim("Al", voltage=4.0, sqs_index=1),
    ]
    report = rank_dopants(results, target_properties=["voltage"], variance_threshold=0.20)
    assert any("Al" in w and "variance" in w.lower() for w in report.warnings)


def test_low_variance_no_flag():
    """Small CV should not trigger warning."""
    results = [
        _make_sim("Al", voltage=4.00, sqs_index=0),
        _make_sim("Al", voltage=4.01, sqs_index=1),
        _make_sim("Al", voltage=4.02, sqs_index=2),
    ]
    report = rank_dopants(results, target_properties=["voltage"], variance_threshold=0.20)
    assert not any("Al" in w and "variance" in w.lower() for w in report.warnings)


# ── Sanity checks ────────────────────────────────────────────────────────────

def test_sanity_check_formation_energy():
    """formation_energy > threshold → warning."""
    results = [_make_sim("Al", formation_energy=0.15)]
    report = rank_dopants(
        results,
        target_properties=["formation_energy"],
        sanity_config={"max_formation_energy_above_hull": 0.10, "max_volume_change": 15.0},
    )
    assert any("formation energy" in w for w in report.warnings)


def test_sanity_check_volume_change():
    """volume_change > threshold → warning."""
    results = [_make_sim("Al", volume_change=20.0)]
    report = rank_dopants(
        results,
        target_properties=["volume_change"],
        sanity_config={"max_formation_energy_above_hull": 0.10, "max_volume_change": 15.0},
    )
    assert any("volume change" in w for w in report.warnings)


# ── Aborted results excluded ──────────────────────────────────────────────────

def test_aborted_results_excluded():
    """Results with relaxation_converged=False must not appear in rankings."""
    results = [
        _make_sim("Al", voltage=4.0, converged=True),
        _make_sim("Ti", voltage=4.5, converged=False),  # aborted
    ]
    report = rank_dopants(results, target_properties=["voltage"])
    ranked_dopants = [ds.dopant for ds in report.rankings]
    assert "Al" in ranked_dopants
    assert "Ti" not in ranked_dopants


def test_aborted_count_in_warnings():
    """Number of aborted runs should appear in warnings."""
    results = [
        _make_sim("Al", voltage=4.0, converged=True),
        _make_sim("Ti", voltage=3.5, converged=False),
    ]
    report = rank_dopants(results, target_properties=["voltage"])
    assert any("abort" in w.lower() for w in report.warnings)


# ── Disorder sensitivity ──────────────────────────────────────────────────────

def test_disorder_sensitivity_calculation():
    """|disordered - ordered| / |ordered| should equal 20% for 3.0 vs 2.4."""
    results = [_make_sim("Al", voltage=2.4)]
    ordered = {"Al": {"voltage": 3.0}}
    report = rank_dopants(results, target_properties=["voltage"], ordered_results=ordered)
    al_stats = next(d for d in report.rankings if d.dopant == "Al")
    sensitivity = al_stats.disorder_sensitivity.get("voltage")
    assert sensitivity == pytest.approx(0.20, rel=1e-5)


# ── RankedReport structure ────────────────────────────────────────────────────

def test_ranked_report_has_required_fields():
    results = [_make_sim("Al", voltage=4.0), _make_sim("Ti", voltage=3.8)]
    report = rank_dopants(results, target_properties=["voltage"])
    assert hasattr(report, "parent_formula")
    assert hasattr(report, "rankings")
    assert hasattr(report, "spearman_rho")
    assert hasattr(report, "recommended")
    assert hasattr(report, "warnings")
    assert hasattr(report, "candidates_simulated")


def test_recommended_list_length():
    """recommended should be at most top_n dopants."""
    results = [
        _make_sim("Al", voltage=4.2),
        _make_sim("Ti", voltage=4.0),
        _make_sim("Mg", voltage=3.8),
        _make_sim("Zr", voltage=3.6),
    ]
    report = rank_dopants(results, target_properties=["voltage"], top_n=3)
    assert len(report.recommended) <= 3


def test_empty_results_returns_empty_report():
    report = rank_dopants([], target_properties=["voltage"])
    assert report.candidates_simulated == 0
    assert report.rankings == []
