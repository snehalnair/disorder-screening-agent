"""
Tests for io/generate_summary.py — Jinja2 report generation.

Uses mock RankedReport data (3 dopants, 2 properties); no MLIP required.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest

from pipeline_io.generate_summary import (
    _fmt_float,
    _fmt_pct,
    build_template_context,
    generate_report,
)


# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_ranked_report() -> dict:
    return {
        "parent_formula": "LiCoO2",
        "target_site": "Co",
        "candidates_simulated": 3,
        "primary_property": "voltage",
        "recommended": ["Al", "Mg"],
        "spearman_rho": {
            "voltage": {"rho": 0.95, "pvalue": 0.01, "n": 3},
        },
        "warnings": ["Al: high SQS variance for voltage"],
        "all_rankings": {"voltage": ["Al", "Mg", "Ti"]},
        "rankings": [
            {
                "dopant": "Al",
                "n_converged": 2,
                "properties": {
                    "voltage": {"mean": 4.2, "std": 0.05, "n": 2, "values": [4.15, 4.25]},
                    "formation_energy": {"mean": 0.04, "std": 0.01, "n": 2, "values": [0.03, 0.05]},
                },
                "ordered_properties": {"voltage": 4.0, "formation_energy": 0.03},
                "disorder_sensitivity": {"voltage": 0.05, "formation_energy": 0.33},
                "rank_by_property": {"voltage": 1, "formation_energy": 2},
            },
            {
                "dopant": "Mg",
                "n_converged": 2,
                "properties": {
                    "voltage": {"mean": 4.1, "std": 0.02, "n": 2, "values": [4.08, 4.12]},
                    "formation_energy": {"mean": 0.02, "std": 0.005, "n": 2, "values": [0.015, 0.025]},
                },
                "ordered_properties": {"voltage": 4.0, "formation_energy": 0.02},
                "disorder_sensitivity": {"voltage": 0.025, "formation_energy": 0.0},
                "rank_by_property": {"voltage": 2, "formation_energy": 1},
            },
            {
                "dopant": "Ti",
                "n_converged": 1,
                "properties": {
                    "voltage": {"mean": 3.8, "std": 0.0, "n": 1, "values": [3.8]},
                    "formation_energy": {"mean": 0.08, "std": 0.0, "n": 1, "values": [0.08]},
                },
                "ordered_properties": {},
                "disorder_sensitivity": {},
                "rank_by_property": {"voltage": 3, "formation_energy": 3},
            },
        ],
    }


def _mock_state(run_id: str = None) -> dict:
    return {
        "parent_formula": "LiCoO2",
        "target_site_species": "Co",
        "target_oxidation_state": 3,
        "target_coordination_number": 6,
        "run_id": run_id or str(uuid.uuid4()),
        "target_properties": ["voltage", "formation_energy"],
        "stage1_os_combinations": 500,
        "stage1_candidates": [{"element": e} for e in ["Al", "Mg", "Ti", "Zr", "Nb"]],
        "stage2_candidates": [{"element": e} for e in ["Al", "Mg", "Ti", "Zr"]],
        "stage3_candidates": [{"element": e} for e in ["Al", "Mg", "Ti"]],
        "config": {
            "pipeline": {
                "stage5_simulation": {
                    "potential": "mace-mp-0",
                    "device": "auto",
                    "concentrations": [0.05, 0.10],
                    "supercell": [2, 2, 2],
                    "n_sqs_realisations": 3,
                },
                "stage2_radius": {"mismatch_threshold": 0.35},
                "stage3_substitution": {"probability_threshold": 0.001},
                "property_weights": {"voltage": 0.35, "formation_energy": 0.25},
            }
        },
        "execution_log": ["Stage 1: 5 candidates", "Stage 5: completed"],
    }


# ── Template rendering ────────────────────────────────────────────────────────

def test_report_renders_without_error(tmp_path):
    """generate_report must not raise any Jinja2 errors on mock data."""
    report = _mock_ranked_report()
    state = _mock_state()
    out = generate_report(report, state, output_path=tmp_path / "test.md")
    assert out.exists()


def test_all_10_section_headers_present(tmp_path):
    """All 10 report section headings must appear in the rendered output."""
    report = _mock_ranked_report()
    state = _mock_state()
    out = generate_report(report, state, output_path=tmp_path / "test.md")
    text = out.read_text()

    expected_sections = [
        "1. Screening Summary",
        "2. Pruning Funnel",
        "3. Simulation Results",
        "4. Property Predictions",
        "5. Ordered vs Disordered",
        "6. Ranking",
        "7. Recommendations",
        "8. Warnings",
        "9. Configuration",
        "10. Metadata",
    ]
    for section in expected_sections:
        assert section in text, f"Missing section: {section!r}"


def test_none_values_render_as_na(tmp_path):
    """None property values must render as 'N/A', not 'None'."""
    report = _mock_ranked_report()
    # Force a None value in ordered_properties
    report["rankings"][2]["ordered_properties"]["voltage"] = None
    state = _mock_state()
    out = generate_report(report, state, output_path=tmp_path / "test.md")
    text = out.read_text()
    assert "None" not in text
    assert "N/A" in text


def test_empty_warnings_renders_gracefully(tmp_path):
    """When warnings list is empty, report should show 'No warnings.'"""
    report = _mock_ranked_report()
    report["warnings"] = []
    state = _mock_state()
    out = generate_report(report, state, output_path=tmp_path / "test.md")
    text = out.read_text()
    assert "No warnings." in text


def test_report_file_written_to_disk(tmp_path):
    """generate_report must create the output file."""
    out_path = tmp_path / "my_report.md"
    generate_report(_mock_ranked_report(), _mock_state(), output_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_report_default_path_uses_run_id(tmp_path, monkeypatch):
    """Without explicit output_path, file is named after run_id."""
    import pipeline_io.generate_summary as gs
    monkeypatch.setattr(gs, "_REPORTS_DIR", tmp_path)
    run_id = "test-run-123"
    state = _mock_state(run_id=run_id)
    out = generate_report(_mock_ranked_report(), state)
    assert out.name == f"{run_id}_screening_report.md"


# ── Funnel table ──────────────────────────────────────────────────────────────

def test_funnel_table_row_count_matches_stages(tmp_path):
    """Funnel table must have 3 rows for Stages 1–3 (no Stage 4 in mock)."""
    ctx = build_template_context(_mock_ranked_report(), _mock_state())
    assert len(ctx["funnel_rows"]) == 3


def test_funnel_table_has_stage4_when_candidates_present(tmp_path):
    """When stage4_candidates is in state, funnel must have 4 rows."""
    state = _mock_state()
    state["stage4_candidates"] = [{"element": "Al"}, {"element": "Mg"}]
    ctx = build_template_context(_mock_ranked_report(), state)
    assert len(ctx["funnel_rows"]) == 4


# ── Spearman table ────────────────────────────────────────────────────────────

def test_spearman_table_one_row_per_property():
    """spearman_rows must have one entry per property in spearman_rho."""
    report = _mock_ranked_report()
    report["spearman_rho"] = {
        "voltage": {"rho": 0.9, "pvalue": 0.02, "n": 3},
        "formation_energy": {"rho": 0.7, "pvalue": 0.15, "n": 3},
    }
    ctx = build_template_context(report, _mock_state())
    assert len(ctx["spearman_rows"]) == 2


def test_spearman_table_empty_when_no_rho():
    ctx = build_template_context(
        {**_mock_ranked_report(), "spearman_rho": {}}, _mock_state()
    )
    assert ctx["spearman_rows"] == []


# ── Jinja2 filters ────────────────────────────────────────────────────────────

def test_fmt_float_formats_to_3_decimal():
    assert _fmt_float(3.14159) == "3.142"


def test_fmt_float_none_returns_na():
    assert _fmt_float(None) == "N/A"


def test_fmt_pct_formats_fraction_as_percent():
    assert _fmt_pct(0.123) == "12.3%"


def test_fmt_pct_none_returns_na():
    assert _fmt_pct(None) == "N/A"


# ── build_template_context ────────────────────────────────────────────────────

def test_context_parent_formula_from_state():
    ctx = build_template_context(_mock_ranked_report(), _mock_state())
    assert ctx["parent_formula"] == "LiCoO2"


def test_context_target_properties_list():
    ctx = build_template_context(_mock_ranked_report(), _mock_state())
    assert "voltage" in ctx["target_properties"]
    assert "formation_energy" in ctx["target_properties"]


def test_context_recommended_list():
    ctx = build_template_context(_mock_ranked_report(), _mock_state())
    assert ctx["recommended"] == ["Al", "Mg"]


def test_context_comparison_rows_contain_ordered_values():
    ctx = build_template_context(_mock_ranked_report(), _mock_state())
    rows = ctx["comparison_rows"]
    al_voltage = next(
        (r for r in rows if r["dopant"] == "Al" and r["property"] == "voltage"), None
    )
    assert al_voltage is not None
    assert al_voltage["ordered"] == pytest.approx(4.0)
    assert al_voltage["disordered"] == pytest.approx(4.2)
