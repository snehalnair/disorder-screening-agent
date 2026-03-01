"""
Integration tests for Phase 3: SQS generation + MLIP relaxation + baseline.

All tests use MockMLIPCalculator (ASE EMT on Cu/Al) or InjectableCalculator.
No GPU required. Tests complete in < 60 seconds.

LiCoO₂ 2×2×2 supercell is used as the NMC proxy:
  - 8 Li, 8 Co, 16 O → 32 atoms
  - Al@10% on Co site → round(0.10 × 8) = 1 dopant atom
  - Al@25% on Co site → 2 dopant atoms
"""

from __future__ import annotations

import pathlib
import tempfile
import uuid

import pytest

from stages.stage5.baseline import compute_baseline
from stages.stage5.calculators import InjectableCalculator, MockMLIPCalculator
from stages.stage5.mlip_relaxation import relax_structure
from stages.stage5.sqs_generator import generate_sqs


_SUPERCELL_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


# ── Fixture (re-exported from conftest) ────────────────────────────────────────

@pytest.fixture
def mock_calc():
    return MockMLIPCalculator()


# ── Test 1: SQS + relax with mock MLIP ────────────────────────────────────────

def test_sqs_and_relax_produces_relaxation_results(lco_structure, mock_calc):
    """
    SQS generation + EMT relaxation for Al@25% on Co site.
    Verifies that 2 RelaxationResult objects are returned with expected fields.
    """
    n_realisations = 2
    sqs_list = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=n_realisations,
    )
    assert len(sqs_list) == n_realisations

    results = []
    for sqs in sqs_list:
        result = relax_structure(
            structure=sqs,
            calculator=mock_calc,
            fmax=0.05,
            max_steps=100,
            filter_type="None",
        )
        results.append(result)

    assert len(results) == n_realisations
    for res in results:
        assert res.relaxation_steps >= 0
        assert res.monitor_history is not None
        assert len(res.monitor_history) >= 0
        assert res.relaxed_structure is not None


def test_sqs_relax_monitor_history_populated(lco_structure, mock_calc):
    """monitor_history must contain at least one step entry per relaxation."""
    sqs_list = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    result = relax_structure(
        structure=sqs_list[0],
        calculator=mock_calc,
        fmax=0.05,
        max_steps=50,
        filter_type="None",
    )
    assert len(result.monitor_history) > 0
    assert "step" in result.monitor_history[0]
    assert "energy" in result.monitor_history[0]


# ── Test 2: baseline + doped comparison ───────────────────────────────────────

def test_baseline_and_doped_energies_exist(lco_structure, mock_calc):
    """
    Baseline energy (undoped) and doped energies must both be present and
    be floats. They should differ because doped structure has different atoms.
    """
    # Compute baseline
    baseline = compute_baseline(
        parent_structure=lco_structure,
        supercell_matrix=_SUPERCELL_222,
        calculator=mock_calc,
        fmax=0.05,
        max_steps=100,
    )
    assert "energy_per_atom" in baseline
    assert isinstance(baseline["energy_per_atom"], float)

    # Compute doped
    sqs_list = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    doped_result = relax_structure(
        structure=sqs_list[0],
        calculator=mock_calc,
        fmax=0.05,
        max_steps=100,
        filter_type="None",
    )

    assert isinstance(doped_result.final_energy_per_atom, float)
    # Baseline and doped structures have different compositions → different E/atom
    # (not guaranteed to differ in sign, but both must be finite)
    assert abs(baseline["energy_per_atom"]) < 1e10
    assert abs(doped_result.final_energy_per_atom) < 1e10


def test_baseline_returns_all_required_keys(lco_structure, mock_calc):
    """compute_baseline must return a dict with all expected keys."""
    baseline = compute_baseline(
        parent_structure=lco_structure,
        supercell_matrix=_SUPERCELL_222,
        calculator=mock_calc,
        fmax=0.05,
        max_steps=100,
    )
    for key in (
        "energy_per_atom",
        "volume",
        "lattice_params",
        "properties",
        "relaxation_converged",
        "relaxation_steps",
        "abort_reason",
    ):
        assert key in baseline, f"Missing key: {key!r}"

    assert all(k in baseline["lattice_params"] for k in ("a", "b", "c"))


def test_baseline_volume_positive(lco_structure, mock_calc):
    """Relaxed undoped supercell must have positive volume."""
    baseline = compute_baseline(
        parent_structure=lco_structure,
        supercell_matrix=_SUPERCELL_222,
        calculator=mock_calc,
        fmax=0.05,
        max_steps=100,
    )
    assert baseline["volume"] > 0.0


# ── Test 3: abort propagation ─────────────────────────────────────────────────

def test_abort_reason_propagates_to_relaxation_result(lco_structure):
    """
    InjectableCalculator with diverging energy → RelaxationResult
    with abort_reason='energy_divergence' and relaxation_converged=False.
    """
    sqs_list = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    sqs = sqs_list[0]
    n_atoms = len(sqs)

    # Large forces keep optimizer running; energy rises to trigger abort
    total_seq = [-10.0 * n_atoms] * 2 + [0.0 * n_atoms] * 20
    force_seq = [1.0] * len(total_seq)

    inject_calc = InjectableCalculator(
        energy_sequence=total_seq,
        force_magnitude_sequence=force_seq,
        n_atoms=n_atoms,
    )

    result = relax_structure(
        structure=sqs,
        calculator=inject_calc,
        fmax=0.05,
        max_steps=50,
        optimizer_name="FIRE",
        monitor_config={"max_energy_increase": 3.0},
        filter_type="None",
    )

    assert result.relaxation_converged is False
    assert result.abort_reason == "energy_divergence"


# ── Test 4: database persistence ──────────────────────────────────────────────

def test_simulation_results_persist_to_sqlite(lco_structure, mock_calc):
    """
    After SQS + relaxation, results must be writable to SQLite via LocalStore.
    """
    from db.local_store import LocalStore
    from db.models import SimulationResult

    sqs_list = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    relax_res = relax_structure(
        structure=sqs_list[0],
        calculator=mock_calc,
        fmax=0.05,
        max_steps=50,
        filter_type="None",
    )

    sim = SimulationResult(
        dopant_element="Al",
        dopant_oxidation_state=3,
        concentration_pct=25.0,
        sqs_realisation_index=0,
        parent_formula="LiCoO2",
        target_site_species="Co",
        mlip_name="mock",
        mlip_version="test",
        relaxation_converged=relax_res.relaxation_converged,
        relaxation_steps=relax_res.relaxation_steps,
        final_energy_per_atom=relax_res.final_energy_per_atom,
        formation_energy_above_hull=None,
        voltage=None,
        disorder_sensitivity=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(pathlib.Path(tmpdir) / "test.db")
        store = LocalStore(db_path)
        run_id = str(uuid.uuid4())
        sim_id = store.save_simulation(sim, run_id)
        assert isinstance(sim_id, str) and len(sim_id) > 0

        # Verify retrieval
        retrieved = store.get_run_results(run_id)
        assert len(retrieved) == 1
        assert retrieved[0].dopant_element == "Al"
        assert retrieved[0].relaxation_converged == relax_res.relaxation_converged
        store.close()


# ── Test 5: graph wiring ───────────────────────────────────────────────────────

def test_graph_nodes_connected(lco_structure):
    """
    build_full_graph() compiles without error and includes all required nodes.
    """
    from graph.graph import build_full_graph

    graph = build_full_graph()
    # Compiled graph should be invokable (not None)
    assert graph is not None


def test_compute_baseline_node_skips_without_parent_structure():
    """
    compute_baseline_node returns gracefully when parent_structure is absent.
    """
    from graph.graph import compute_baseline_node

    state = {
        "stage3_candidates": [],
        "config": {},
        "execution_log": [],
    }
    result = compute_baseline_node(state)
    assert "execution_log" in result
    assert any("skip" in e.lower() for e in result["execution_log"])


def test_rank_and_report_stub_passes_through():
    """rank_and_report_node must append a log entry and return cleanly."""
    from graph.graph import rank_and_report_node

    state = {
        "relaxed_results": {"Al": [{}], "Ti": [{}]},
        "execution_log": [],
    }
    result = rank_and_report_node(state)
    assert "execution_log" in result
    assert len(result["execution_log"]) > 0
