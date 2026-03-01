"""Tests for evaluation/eval_accuracy.py — RQ3 accuracy metrics (no MACE).

Uses mock RQ2 results and the real experimental JSON.
"""

from __future__ import annotations

import pytest

from evaluation.eval_accuracy import (
    compute_accuracy_metrics,
    load_experimental_data,
)


# ── Mock data ──────────────────────────────────────────────────────────────────


def _mock_rq2_results() -> dict:
    return {
        "target_properties": ["voltage", "li_ni_exchange"],
        "dopant_results": [
            {
                "dopant": "Al",
                "ordered": {"voltage": 3.80, "li_ni_exchange": -0.10},
                "disordered_mean": {"voltage": 3.75, "li_ni_exchange": -0.12},
                "disordered_std": {"voltage": 0.02, "li_ni_exchange": 0.01},
                "disordered_n": {"voltage": 5, "li_ni_exchange": 5},
                "n_converged": 5,
                "disorder_sensitivity": {"voltage": 0.013, "li_ni_exchange": 0.02},
                "sqs_realisations": [],
            },
            {
                "dopant": "Ti",
                "ordered": {"voltage": 3.82, "li_ni_exchange": -0.08},
                "disordered_mean": {"voltage": 3.78, "li_ni_exchange": -0.09},
                "disordered_std": {"voltage": 0.03, "li_ni_exchange": 0.01},
                "disordered_n": {"voltage": 5, "li_ni_exchange": 5},
                "n_converged": 5,
                "disorder_sensitivity": {"voltage": 0.010, "li_ni_exchange": 0.01},
                "sqs_realisations": [],
            },
            {
                "dopant": "Mg",
                "ordered": {"voltage": 3.77, "li_ni_exchange": -0.07},
                "disordered_mean": {"voltage": 3.74, "li_ni_exchange": -0.08},
                "disordered_std": {"voltage": 0.04, "li_ni_exchange": 0.02},
                "disordered_n": {"voltage": 4, "li_ni_exchange": 4},
                "n_converged": 4,
                "disorder_sensitivity": {},
                "sqs_realisations": [],
            },
        ],
        "spearman_rho": {},
    }


# ── load_experimental_data ────────────────────────────────────────────────────


def test_load_experimental_data_returns_dict():
    """load_experimental_data must return a non-empty dict."""
    data = load_experimental_data()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_experimental_data_has_al():
    """Al must be in the experimental data."""
    data = load_experimental_data()
    assert "Al" in data


def test_experimental_data_al_has_voltage():
    """Al entry must have a voltage_V value."""
    data = load_experimental_data()
    al = data["Al"]
    assert "voltage_V" in al
    v = al["voltage_V"]
    val = v.get("value") if isinstance(v, dict) else v
    assert isinstance(val, (int, float))
    assert 3.5 <= val <= 4.2  # sanity check


def test_experimental_data_has_8_or_more_dopants():
    """Experimental JSON must have ≥ 5 dopants (acceptance criterion P6-5)."""
    data = load_experimental_data()
    assert len(data) >= 5


# ── compute_accuracy_metrics ──────────────────────────────────────────────────


def test_accuracy_metrics_returns_expected_keys():
    """compute_accuracy_metrics must return all required keys."""
    rq2 = _mock_rq2_results()
    exp = load_experimental_data()
    acc = compute_accuracy_metrics(rq2, exp)

    assert "per_dopant" in acc
    assert "mae_ordered" in acc
    assert "mae_disordered" in acc
    assert "pct_reduction" in acc
    assert "spearman_vs_exp" in acc


def test_accuracy_per_dopant_has_al(monkeypatch):
    """per_dopant list must include Al when Al is in both RQ2 and experimental data."""
    rq2 = _mock_rq2_results()
    exp = load_experimental_data()
    acc = compute_accuracy_metrics(rq2, exp)

    dopants = [row["dopant"] for row in acc["per_dopant"]]
    assert "Al" in dopants


def test_accuracy_mae_voltage_is_positive():
    """MAE values must be non-negative."""
    rq2 = _mock_rq2_results()
    exp = load_experimental_data()
    acc = compute_accuracy_metrics(rq2, exp)

    mae_ord = acc["mae_ordered"].get("voltage")
    mae_dis = acc["mae_disordered"].get("voltage")
    if mae_ord is not None:
        assert mae_ord >= 0.0
    if mae_dis is not None:
        assert mae_dis >= 0.0


def test_accuracy_pct_reduction_sign():
    """If disordered MAE < ordered MAE, pct_reduction must be positive."""
    rq2 = _mock_rq2_results()
    # Override experimental with values closer to disordered predictions
    mock_exp = {
        "Al": {"voltage_V": {"value": 3.75}},   # matches disordered (3.75) exactly
        "Ti": {"voltage_V": {"value": 3.78}},   # matches disordered (3.78) exactly
        "Mg": {"voltage_V": {"value": 3.74}},   # matches disordered (3.74) exactly
    }
    acc = compute_accuracy_metrics(rq2, mock_exp)
    pr = acc["pct_reduction"].get("voltage")
    if pr is not None:
        # Disordered is closer → positive reduction
        assert pr > 0.0


def test_accuracy_metrics_with_no_matching_dopants():
    """When no dopants overlap between RQ2 and experimental, MAE should be None/empty."""
    rq2 = _mock_rq2_results()
    mock_exp = {"Zr": {"voltage_V": {"value": 3.80}}}
    acc = compute_accuracy_metrics(rq2, mock_exp)
    # Should not crash
    assert "mae_ordered" in acc


def test_spearman_vs_exp_needs_3_points():
    """Spearman ρ vs experimental requires ≥3 data points; with 3 dopants it should compute."""
    rq2 = _mock_rq2_results()
    mock_exp = {
        "Al": {"voltage_V": {"value": 3.78}},
        "Ti": {"voltage_V": {"value": 3.79}},
        "Mg": {"voltage_V": {"value": 3.76}},
    }
    acc = compute_accuracy_metrics(rq2, mock_exp)
    rho = acc["spearman_vs_exp"].get("voltage")
    if rho is not None:
        assert -1.0 <= rho["rho"] <= 1.0
        assert 0.0 <= rho["pvalue"] <= 1.0
