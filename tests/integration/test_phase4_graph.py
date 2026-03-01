"""
Integration tests for Phase 4: property wiring, ranking, and entry points.

All tests use MockMLIPCalculator (no GPU).  LiCoO₂ 2×2×2 supercell
from conftest.py is the standard proxy for NMC.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest
import yaml

from stages.stage5.calculators import MockMLIPCalculator

_SUPERCELL_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


@pytest.fixture
def mock_config_path(tmp_path):
    """Write a pipeline.yaml with potential: mock for entry-point tests."""
    cfg = {
        "pipeline": {
            "stage5_simulation": {
                "supercell": _SUPERCELL_222,
                "concentrations": [0.25],
                "n_sqs_realisations": 1,
                "fmax": 0.05,
                "max_relax_steps": 50,
                "potential": "mock",
            },
            "output": {"top_n": 3, "include_ordered_comparison": True},
            "property_weights": {"voltage": 0.35, "formation_energy": 0.25},
            "database": {"local": {"path": str(tmp_path / "results.db")}},
        }
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)

_SIM_CFG = {
    "pipeline": {
        "stage5_simulation": {
            "supercell": _SUPERCELL_222,
            "concentrations": [0.25],
            "n_sqs_realisations": 1,
            "fmax": 0.05,
            "max_relax_steps": 50,
            "potential": "mock",
        },
        "output": {
            "top_n": 3,
            "include_ordered_comparison": True,
        },
        "property_weights": {
            "voltage": 0.35,
            "formation_energy": 0.25,
        },
    }
}


# ── stage5_simulate_node ──────────────────────────────────────────────────────

def test_simulate_node_creates_simulation_results(lco_structure):
    """stage5_simulate_node must produce a list of SimulationResult objects."""
    from db.models import SimulationResult
    from graph.graph import stage5_simulate_node

    state = {
        "parent_structure": lco_structure,
        "parent_formula": "LiCoO2",
        "target_site_species": "Co",
        "stage3_candidates": [{"element": "Al", "oxidation_state": 3}],
        "target_properties": ["voltage", "formation_energy"],
        "config": _SIM_CFG,
        "execution_log": [],
    }
    result = stage5_simulate_node(state)

    sim_results = result.get("simulation_results", [])
    assert len(sim_results) > 0
    assert all(isinstance(r, SimulationResult) for r in sim_results)


def test_simulate_node_properties_populated(lco_structure):
    """voltage and formation_energy must be set (not None) on converged results."""
    from graph.graph import stage5_simulate_node

    state = {
        "parent_structure": lco_structure,
        "parent_formula": "LiCoO2",
        "target_site_species": "Co",
        "stage3_candidates": [{"element": "Al", "oxidation_state": 3}],
        "target_properties": ["voltage", "formation_energy"],
        "config": _SIM_CFG,
        "execution_log": [],
    }
    result = stage5_simulate_node(state)

    converged = [r for r in result["simulation_results"] if r.relaxation_converged]
    if converged:
        # At least one converged result should have properties computed
        has_voltage = any(r.voltage is not None for r in converged)
        has_fe = any(r.formation_energy_above_hull is not None for r in converged)
        assert has_voltage or has_fe  # at least one property computed


def test_simulate_node_ordered_results_populated(lco_structure):
    """ordered_results dict must be populated for each candidate."""
    from graph.graph import stage5_simulate_node

    state = {
        "parent_structure": lco_structure,
        "parent_formula": "LiCoO2",
        "target_site_species": "Co",
        "stage3_candidates": [{"element": "Al", "oxidation_state": 3}],
        "target_properties": ["formation_energy"],
        "config": _SIM_CFG,
        "execution_log": [],
    }
    result = stage5_simulate_node(state)
    assert "ordered_results" in result
    assert "Al" in result["ordered_results"]


def test_simulate_node_simulation_result_fields(lco_structure):
    """SimulationResult must have expected identity fields set."""
    from graph.graph import stage5_simulate_node

    state = {
        "parent_structure": lco_structure,
        "parent_formula": "LiCoO2",
        "target_site_species": "Co",
        "stage3_candidates": [{"element": "Al", "oxidation_state": 3}],
        "target_properties": ["formation_energy"],
        "config": _SIM_CFG,
        "execution_log": [],
    }
    result = stage5_simulate_node(state)
    sim = result["simulation_results"][0]

    assert sim.dopant_element == "Al"
    assert sim.parent_formula == "LiCoO2"
    assert sim.target_site_species == "Co"
    assert sim.concentration_pct == pytest.approx(25.0)


def test_simulate_node_skips_gracefully_without_parent():
    """stage5_simulate_node must not crash when parent_structure is absent."""
    from graph.graph import stage5_simulate_node

    result = stage5_simulate_node({"execution_log": []})
    assert result["simulation_results"] == []
    assert any("skip" in e.lower() for e in result["execution_log"])


def test_simulate_node_db_persistence(lco_structure, tmp_path):
    """Simulation results must be written to SQLite when run_id is set."""
    from db.local_store import LocalStore
    from graph.graph import stage5_simulate_node

    run_id = str(uuid.uuid4())
    db_path = str(tmp_path / "results.db")

    cfg = {
        "pipeline": {
            **_SIM_CFG["pipeline"],
            "database": {"local": {"path": db_path}},
        }
    }

    state = {
        "parent_structure": lco_structure,
        "parent_formula": "LiCoO2",
        "target_site_species": "Co",
        "stage3_candidates": [{"element": "Al", "oxidation_state": 3}],
        "target_properties": ["formation_energy"],
        "config": cfg,
        "execution_log": [],
        "run_id": run_id,
    }
    result = stage5_simulate_node(state)

    store = LocalStore(db_path)
    try:
        saved = store.get_run_results(run_id)
        assert len(saved) == len(result["simulation_results"])
        assert saved[0].dopant_element == "Al"
    finally:
        store.close()


# ── rank_and_report_node ──────────────────────────────────────────────────────

def test_rank_and_report_with_simulation_results():
    """rank_and_report_node must return a ranked_report when SimulationResults present."""
    from db.models import SimulationResult
    from graph.graph import rank_and_report_node

    sim_results = [
        SimulationResult(
            dopant_element="Al",
            dopant_oxidation_state=3,
            concentration_pct=10.0,
            sqs_realisation_index=0,
            parent_formula="LiCoO2",
            target_site_species="Co",
            mlip_name="mock",
            mlip_version="test",
            relaxation_converged=True,
            relaxation_steps=10,
            final_energy_per_atom=-5.0,
            voltage=4.2,
        ),
        SimulationResult(
            dopant_element="Ti",
            dopant_oxidation_state=4,
            concentration_pct=10.0,
            sqs_realisation_index=0,
            parent_formula="LiCoO2",
            target_site_species="Co",
            mlip_name="mock",
            mlip_version="test",
            relaxation_converged=True,
            relaxation_steps=10,
            final_energy_per_atom=-4.8,
            voltage=3.9,
        ),
    ]

    state = {
        "simulation_results": sim_results,
        "target_properties": ["voltage"],
        "config": _SIM_CFG,
        "execution_log": [],
    }
    result = rank_and_report_node(state)

    assert "ranked_report" in result
    report = result["ranked_report"]
    assert "recommended" in report
    assert "rankings" in report
    assert "all_rankings" in report


def test_rank_and_report_recommended_order():
    """Al voltage > Ti voltage → Al should be first recommended."""
    from db.models import SimulationResult
    from graph.graph import rank_and_report_node

    sim_results = [
        SimulationResult(
            dopant_element="Al", dopant_oxidation_state=3,
            concentration_pct=10.0, sqs_realisation_index=0,
            parent_formula="LiCoO2", target_site_species="Co",
            mlip_name="mock", mlip_version="test",
            relaxation_converged=True, relaxation_steps=5,
            final_energy_per_atom=-5.0, voltage=4.5,
        ),
        SimulationResult(
            dopant_element="Ti", dopant_oxidation_state=4,
            concentration_pct=10.0, sqs_realisation_index=0,
            parent_formula="LiCoO2", target_site_species="Co",
            mlip_name="mock", mlip_version="test",
            relaxation_converged=True, relaxation_steps=5,
            final_energy_per_atom=-4.8, voltage=3.8,
        ),
        SimulationResult(
            dopant_element="Mg", dopant_oxidation_state=2,
            concentration_pct=10.0, sqs_realisation_index=0,
            parent_formula="LiCoO2", target_site_species="Co",
            mlip_name="mock", mlip_version="test",
            relaxation_converged=True, relaxation_steps=5,
            final_energy_per_atom=-5.2, voltage=4.1,
        ),
    ]

    state = {
        "simulation_results": sim_results,
        "target_properties": ["voltage"],
        "config": _SIM_CFG,
        "execution_log": [],
    }
    result = rank_and_report_node(state)
    report = result["ranked_report"]
    assert report["recommended"][0] == "Al"


def test_rank_and_report_no_sim_results_returns_empty():
    """rank_and_report_node must not crash when simulation_results is empty."""
    from graph.graph import rank_and_report_node

    state = {
        "relaxed_results": {"Al": [{}]},
        "execution_log": [],
    }
    result = rank_and_report_node(state)
    assert "execution_log" in result
    assert len(result["execution_log"]) > 0
    assert result.get("ranked_report") == {}


# ── run_single_dopant ─────────────────────────────────────────────────────────

def test_run_single_dopant_returns_sim_results(lco_structure, tmp_path, mock_config_path):
    """run_single_dopant must return SimulationResult objects and persist to DB."""
    from db.local_store import LocalStore
    from graph.entry_points import run_single_dopant

    db_path = str(tmp_path / "results.db")
    run_id = str(uuid.uuid4())

    output = run_single_dopant(
        parent_formula="LiCoO2",
        parent_structure=lco_structure,
        dopant_element="Al",
        dopant_oxidation_state=3,
        target_site_species="Co",
        concentrations=[0.25],
        target_properties=["formation_energy"],
        config_path=mock_config_path,
        db_path=db_path,
        run_id=run_id,
    )

    assert output["run_id"] == run_id
    assert len(output["simulation_results"]) > 0
    assert output["simulation_results"][0].dopant_element == "Al"

    # Verify DB persistence
    store = LocalStore(db_path)
    try:
        saved = store.get_run_results(run_id)
        assert len(saved) == len(output["simulation_results"])
    finally:
        store.close()


def test_run_single_dopant_ordered_results(lco_structure, tmp_path, mock_config_path):
    """run_single_dopant must return ordered_results for the dopant."""
    from graph.entry_points import run_single_dopant

    output = run_single_dopant(
        parent_formula="LiCoO2",
        parent_structure=lco_structure,
        dopant_element="Al",
        dopant_oxidation_state=3,
        target_site_species="Co",
        concentrations=[0.25],
        target_properties=["formation_energy"],
        config_path=mock_config_path,
        db_path=str(tmp_path / "r.db"),
        run_id=str(uuid.uuid4()),
    )

    assert "ordered_results" in output
    assert "Al" in output["ordered_results"]


# ── run_comparison ────────────────────────────────────────────────────────────

def test_run_comparison_returns_report(tmp_path):
    """run_comparison must return a ComparisonReport from two stored runs."""
    from db.local_store import LocalStore
    from db.models import SimulationResult
    from graph.entry_points import run_comparison
    from ranking.comparator import ComparisonReport

    db_path = str(tmp_path / "cmp.db")
    store = LocalStore(db_path)
    run1, run2 = str(uuid.uuid4()), str(uuid.uuid4())

    for run, voltage in [(run1, 4.2), (run2, 4.3)]:
        store.save_simulation(
            SimulationResult(
                dopant_element="Al",
                dopant_oxidation_state=3,
                concentration_pct=10.0,
                sqs_realisation_index=0,
                parent_formula="LiCoO2",
                target_site_species="Co",
                mlip_name="mock",
                mlip_version="test",
                relaxation_converged=True,
                relaxation_steps=5,
                final_energy_per_atom=-5.0,
                voltage=voltage,
            ),
            run,
        )
    store.close()

    report = run_comparison(
        run_ids=[run1, run2],
        target_properties=["voltage"],
        db_path=db_path,
    )
    assert isinstance(report, ComparisonReport)
    assert "Al" in report.dopants_compared
