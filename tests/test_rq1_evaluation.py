"""Tests for the updated evaluation/eval_pruning.py — RQ1 full reporting.

All tests run without MLIP/GPU. Uses the real Stage 1-3 pruning pipeline
against the NMC811 ground truth.
"""

from __future__ import annotations

import pytest

from evaluation.eval_pruning import (
    PruningMetrics,
    evaluate_pruning,
    os_category_breakdown,
    per_dopant_breakdown,
    run_full_rq1,
)


# ── Shared fixture ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rq1_report():
    """Run the full RQ1 pipeline once for all tests in this module."""
    return run_full_rq1()


# ── PruningMetrics dataclass ───────────────────────────────────────────────────


def test_pruning_metrics_has_precision_known():
    """PruningMetrics must expose precision_known field."""
    m = PruningMetrics(
        stage="test",
        n_candidates=46,
        n_gt_positives=13,
        n_recalled=12,
        recall=12 / 13,
        precision=12 / 46,
        precision_known=12 / 15,  # 15 elements with known outcome
    )
    assert m.precision_known == pytest.approx(12 / 15)


def test_pruning_metrics_str_includes_precision_known():
    """String representation must include precision_known line."""
    m = PruningMetrics(
        stage="test",
        n_candidates=46,
        n_gt_positives=13,
        n_recalled=12,
        recall=12 / 13,
        precision=12 / 46,
        precision_known=0.80,
    )
    assert "Precision (known)" in str(m)


# ── run_full_rq1 ──────────────────────────────────────────────────────────────


def test_rq1_report_has_all_keys(rq1_report):
    """run_full_rq1 must return a dict with the expected keys."""
    assert "stage_metrics" in rq1_report
    assert "per_dopant" in rq1_report
    assert "os_breakdown" in rq1_report
    assert "funnel_counts" in rq1_report


def test_rq1_three_stage_metrics(rq1_report):
    """run_full_rq1 must return metrics for 3 stages."""
    assert len(rq1_report["stage_metrics"]) == 3


def test_rq1_funnel_monotonic(rq1_report):
    """Funnel must be monotonically non-increasing (each stage ≤ previous)."""
    fc = rq1_report["funnel_counts"]
    s1 = fc["stage1"]
    s2 = fc["stage2"]
    s3 = fc["stage3"]
    assert s1 >= s2 >= s3


def test_rq1_stage3_recall_above_90pct_confirmed_successful(rq1_report):
    """Stage 3 recall vs confirmed_successful-only must be ≥ 90%.

    Note: The 92.3% calibration target uses confirmed_successful (13 elements) only.
    When confirmed_limited is included (Sc, Hf, Y, Na), some are filtered at Stage 3,
    so overall recall drops. We test confirmed_successful recall here.
    """
    from evaluation.eval_pruning import evaluate_pruning

    state = rq1_report["state"]
    m = evaluate_pruning(
        state.get("stage3_candidates", []),
        gt_classes=["confirmed_successful"],
        stage_label="Stage 3 (confirmed_successful only)",
    )
    assert m.recall >= 0.90, f"Stage 3 recall (confirmed_successful) = {m.recall:.1%} < 90%"


def test_rq1_stage3_pair_count_is_46(rq1_report):
    """Stage 3 must produce exactly 46 (element, OS) pairs — the calibrated output."""
    state = rq1_report["state"]
    n_pairs = len(state.get("stage3_candidates", []))
    assert n_pairs == 46, f"Expected 46 (element, OS) pairs, got {n_pairs}"


def test_rq1_stage3_unique_elements_is_29(rq1_report):
    """Stage 3 produces 29 unique candidate elements (46 pairs with 14 duplicated OS)."""
    m = rq1_report["stage_metrics"][2]  # Stage 3 PruningMetrics
    assert m.n_candidates == 29, f"Expected 29 unique elements, got {m.n_candidates}"


# ── per_dopant_breakdown ──────────────────────────────────────────────────────


def test_per_dopant_returns_19_gt_rows(rq1_report):
    """per_dopant_breakdown must include all 19 GT dopants."""
    rows = rq1_report["per_dopant"]
    assert len(rows) == 19


def test_per_dopant_al_survives(rq1_report):
    """Al must survive all 3 stages (confirmed_successful)."""
    rows = {r.element: r for r in rq1_report["per_dopant"]}
    al = rows["Al"]
    assert al.survived_stage3 is True
    assert al.filtered_at_stage is None


def test_per_dopant_b_filtered_at_stage2(rq1_report):
    """B must be filtered at Stage 2 (50.5% mismatch > 35% threshold)."""
    rows = {r.element: r for r in rq1_report["per_dopant"]}
    b = rows["B"]
    assert b.survived_stage1 is True
    assert b.survived_stage2 is False
    assert b.filtered_at_stage == "Stage 2"


def test_per_dopant_all_have_gt_class(rq1_report):
    """All rows must have a non-empty gt_class."""
    for row in rq1_report["per_dopant"]:
        assert row.gt_class, f"Missing gt_class for {row.element}"


def test_per_dopant_survived_flags_consistent(rq1_report):
    """If survived_stage2 is False, survived_stage3 must also be False."""
    for row in rq1_report["per_dopant"]:
        if not row.survived_stage2:
            assert not row.survived_stage3, (
                f"{row.element}: survived_stage2=False but survived_stage3=True"
            )


# ── os_category_breakdown ─────────────────────────────────────────────────────


def test_os_breakdown_has_four_categories(rq1_report):
    """OS breakdown must have exactly 4 categories."""
    bd = rq1_report["os_breakdown"]
    assert set(bd.keys()) == {"isovalent_3+", "aliovalent_2+", "aliovalent_4+", "aliovalent_56+"}


def test_os_breakdown_isovalent_includes_al(rq1_report):
    """Al (3+) must appear in the isovalent_3+ category."""
    bd = rq1_report["os_breakdown"]
    assert "Al" in bd["isovalent_3+"]["elements"]


def test_os_breakdown_mg_in_aliovalent_2(rq1_report):
    """Mg (2+) must appear in aliovalent_2+ category."""
    bd = rq1_report["os_breakdown"]
    assert "Mg" in bd["aliovalent_2+"]["elements"]


def test_os_breakdown_recall_values_in_range(rq1_report):
    """Recall for every OS category must be in [0.0, 1.0]."""
    bd = rq1_report["os_breakdown"]
    for cat, d in bd.items():
        assert 0.0 <= d["recall"] <= 1.0, f"Category {cat} recall out of range"


def test_os_breakdown_nb_w_in_aliovalent_56(rq1_report):
    """Nb (5+) and W (6+) must appear in aliovalent_56+ category."""
    bd = rq1_report["os_breakdown"]
    assert "Nb" in bd["aliovalent_56+"]["elements"]
    assert "W" in bd["aliovalent_56+"]["elements"]


# ── evaluate_pruning precision_known ─────────────────────────────────────────


def test_precision_known_leq_1(rq1_report):
    """precision_known must be ≤ 1.0 for all stages."""
    for m in rq1_report["stage_metrics"]:
        if m.precision_known is not None:
            assert m.precision_known <= 1.0


def test_precision_known_geq_precision(rq1_report):
    """precision_known ≥ overall precision (fewer denominator elements)."""
    for m in rq1_report["stage_metrics"]:
        if m.precision is not None and m.precision_known is not None:
            assert m.precision_known >= m.precision - 1e-9
