"""Tests for db/local_store.py — SQLite persistence layer."""

import pathlib
import tempfile
import uuid

import pytest

from db.local_store import LocalStore, _config_hash
from db.models import ExperimentalComparison, PruningRecord, SimulationResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Fresh LocalStore backed by a temp file for each test."""
    db_file = tmp_path / "test_results.db"
    s = LocalStore(str(db_file))
    yield s
    s.close()


def _mock_sim(**kwargs) -> SimulationResult:
    defaults = dict(
        dopant_element="Al",
        dopant_oxidation_state=3,
        concentration_pct=5.0,
        sqs_realisation_index=0,
        parent_formula="LiNi0.8Mn0.1Co0.1O2",
        target_site_species="Co",
        mlip_name="mattersim",
        mlip_version="1.0.0",
        relaxation_converged=True,
        relaxation_steps=120,
        final_energy_per_atom=-6.12,
        formation_energy_above_hull=0.03,
        voltage=3.95,
        disorder_sensitivity={"voltage": 0.02, "li_ni_exchange": 0.15},
    )
    defaults.update(kwargs)
    return SimulationResult(**defaults)


# ── Schema ────────────────────────────────────────────────────────────────────

def test_database_file_created(tmp_path):
    db_file = tmp_path / "subdir" / "results.db"
    s = LocalStore(str(db_file))
    assert db_file.exists()
    s.close()


def test_all_three_tables_exist(store):
    tables = store.list_tables()
    assert "simulations" in tables
    assert "pruning_records" in tables
    assert "experimental_comparisons" in tables


def test_simulations_table_has_key_columns(store):
    cols = store.table_columns("simulations")
    for col in ("id", "run_id", "dopant_element", "concentration_pct",
                "mlip_name", "relaxation_converged", "voltage",
                "disorder_sensitivity", "pipeline_config_hash"):
        assert col in cols, f"Column {col!r} missing from simulations"


def test_pruning_records_table_has_key_columns(store):
    cols = store.table_columns("pruning_records")
    for col in ("id", "run_id", "element", "stage1_passed", "stage2_passed",
                "stage3_passed", "stage4_passed"):
        assert col in cols, f"Column {col!r} missing from pruning_records"


# ── Simulation write + read roundtrip ─────────────────────────────────────────

def test_save_simulation_returns_id(store):
    run_id = str(uuid.uuid4())
    sim = _mock_sim()
    sim_id = store.save_simulation(sim, run_id)
    assert isinstance(sim_id, str) and len(sim_id) > 0


def test_get_run_results_returns_saved(store):
    run_id = str(uuid.uuid4())
    sim = _mock_sim(dopant_element="Ti", concentration_pct=10.0)
    store.save_simulation(sim, run_id)
    results = store.get_run_results(run_id)
    assert len(results) == 1
    assert results[0].dopant_element == "Ti"
    assert results[0].concentration_pct == 10.0


def test_get_run_results_empty_for_unknown_run(store):
    assert store.get_run_results("nonexistent-run-id") == []


def test_simulation_roundtrip_preserves_disorder_sensitivity(store):
    run_id = str(uuid.uuid4())
    ds = {"voltage": 0.05, "li_ni_exchange": 0.12}
    sim = _mock_sim(disorder_sensitivity=ds)
    store.save_simulation(sim, run_id)
    result = store.get_run_results(run_id)[0]
    assert result.disorder_sensitivity == ds


# ── Cache hit / miss ──────────────────────────────────────────────────────────

def test_find_simulation_cache_hit(store):
    run_id = str(uuid.uuid4())
    sim = _mock_sim()
    store.save_simulation(sim, run_id)
    found = store.find_simulation("LiNi0.8Mn0.1Co0.1O2", "Al", 5.0, "mattersim", "1.0.0")
    assert found is not None
    assert found.dopant_element == "Al"


def test_find_simulation_cache_miss(store):
    found = store.find_simulation("LiCoO2", "Ti", 5.0, "mattersim", "1.0.0")
    assert found is None


def test_find_simulation_wrong_mlip_version_is_miss(store):
    run_id = str(uuid.uuid4())
    store.save_simulation(_mock_sim(), run_id)
    found = store.find_simulation("LiNi0.8Mn0.1Co0.1O2", "Al", 5.0, "mattersim", "2.0.0")
    assert found is None


# ── get_all_for_parent ────────────────────────────────────────────────────────

def test_get_all_for_parent_returns_multiple_runs(store):
    for dopant in ("Al", "Ti", "Mg"):
        store.save_simulation(_mock_sim(dopant_element=dopant), str(uuid.uuid4()))

    all_results = store.get_all_for_parent("LiNi0.8Mn0.1Co0.1O2")
    assert len(all_results) == 3
    elements = {r.dopant_element for r in all_results}
    assert elements == {"Al", "Ti", "Mg"}


def test_get_all_for_parent_excludes_other_parents(store):
    store.save_simulation(_mock_sim(parent_formula="LiCoO2"), str(uuid.uuid4()))
    store.save_simulation(_mock_sim(parent_formula="LiNi0.8Mn0.1Co0.1O2"), str(uuid.uuid4()))
    nmc_results = store.get_all_for_parent("LiNi0.8Mn0.1Co0.1O2")
    assert all(r.parent_formula == "LiNi0.8Mn0.1Co0.1O2" for r in nmc_results)
    assert len(nmc_results) == 1


# ── Pruning records ───────────────────────────────────────────────────────────

def test_save_pruning_record_dataclass(store):
    run_id = str(uuid.uuid4())
    records = [
        PruningRecord(
            run_id=run_id,
            parent_formula="LiNi0.8Mn0.1Co0.1O2",
            target_site_species="Co",
            element="Al",
            stage1_passed=True,
            stage1_oxidation_state=3,
            stage2_passed=True,
            stage2_mismatch_pct=1.8,
            stage3_passed=True,
            stage3_sub_probability=0.042,
        )
    ]
    store.save_pruning_record(run_id, records)
    rows = store.get_pruning_records(run_id)
    assert len(rows) == 1
    assert rows[0]["element"] == "Al"
    assert rows[0]["stage3_sub_probability"] == pytest.approx(0.042)


def test_save_pruning_record_dict(store):
    """save_pruning_record also accepts raw dicts (from stage pipeline output)."""
    run_id = str(uuid.uuid4())
    records = [
        {
            "parent_formula": "LiCoO2",
            "target_site_species": "Co",
            "element": "Mg",
            "stage1_passed": True,
            "stage2_passed": False,
            "stage2_mismatch_pct": 32.1,
            "stage3_passed": False,
        }
    ]
    store.save_pruning_record(run_id, records)
    rows = store.get_pruning_records(run_id)
    assert len(rows) == 1
    assert rows[0]["element"] == "Mg"
    assert not rows[0]["stage2_passed"]


def test_save_multiple_pruning_records(store):
    run_id = str(uuid.uuid4())
    records = [{"element": el, "parent_formula": "X", "target_site_species": "Co"}
               for el in ("Al", "Ti", "Mg", "Fe")]
    store.save_pruning_record(run_id, records)
    rows = store.get_pruning_records(run_id)
    assert len(rows) == 4


# ── Experimental comparisons ──────────────────────────────────────────────────

def test_save_experimental_comparison_dict(store):
    run_id = str(uuid.uuid4())
    sim_id = store.save_simulation(_mock_sim(), run_id)
    store.save_experimental_comparison(sim_id, {
        "property_name": "voltage",
        "computed_value_ordered": 3.90,
        "computed_value_disordered": 3.85,
        "experimental_value": 3.88,
        "experimental_source": "10.1000/example",
        "mae_ordered": 0.02,
        "mae_disordered": 0.03,
    })
    # Verify row exists
    cur = store._con.execute(
        "SELECT * FROM experimental_comparisons WHERE simulation_id = ?", (sim_id,)
    )
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["property_name"] == "voltage"


# ── config_hash utility ───────────────────────────────────────────────────────

def test_config_hash_deterministic():
    cfg = {"pipeline": {"stage1": {"threshold": 0.15}}}
    h1 = _config_hash(cfg)
    h2 = _config_hash(cfg)
    assert h1 == h2


def test_config_hash_changes_with_config():
    cfg1 = {"threshold": 0.15}
    cfg2 = {"threshold": 0.35}
    assert _config_hash(cfg1) != _config_hash(cfg2)
