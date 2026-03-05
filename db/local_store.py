"""
LocalStore — SQLite persistence layer for the disorder-screening pipeline.

Serves three purposes:
  1. Cache: avoids re-running identical MLIP simulations.
  2. Session history: Level 2 planner queries past results for the same parent.
  3. Evaluation dataset: grows with every run; used to compute RQ2/RQ3 metrics.

Schema is defined by SQL DDL below and mirrors the PRD §7.1 specification.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import uuid
from typing import Optional

from db.models import ExperimentalComparison, PruningRecord, SimulationResult

_DDL = """
CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_formula TEXT NOT NULL,
    parent_mp_id TEXT,
    structure_type TEXT,
    target_site_species TEXT NOT NULL,
    target_oxidation_state INTEGER,
    dopant_element TEXT NOT NULL,
    dopant_oxidation_state INTEGER,
    concentration_pct REAL,
    supercell_size TEXT,
    n_atoms INTEGER,
    sqs_realisation_index INTEGER,
    mlip_name TEXT,
    mlip_version TEXT,
    relaxation_converged BOOLEAN,
    relaxation_steps INTEGER,
    abort_reason TEXT,
    initial_energy_per_atom REAL,
    final_energy_per_atom REAL,
    max_force_final REAL,
    formation_energy_above_hull REAL,
    li_ni_exchange_energy REAL,
    voltage REAL,
    volume_change_pct REAL,
    lattice_params TEXT,
    ordered_formation_energy REAL,
    ordered_voltage REAL,
    ordered_li_ni_exchange REAL,
    disorder_sensitivity TEXT,
    pipeline_config_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pruning_records (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_formula TEXT NOT NULL,
    target_site_species TEXT NOT NULL,
    element TEXT NOT NULL,
    stage1_passed BOOLEAN,
    stage1_oxidation_state INTEGER,
    stage2_passed BOOLEAN,
    stage2_mismatch_pct REAL,
    stage3_passed BOOLEAN,
    stage3_sub_probability REAL,
    stage4_passed BOOLEAN,
    stage4_predicted_property TEXT,
    stage4_viability_reason TEXT,
    thresholds_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experimental_comparisons (
    id TEXT PRIMARY KEY,
    simulation_id TEXT REFERENCES simulations(id),
    property_name TEXT NOT NULL,
    computed_value_ordered REAL,
    computed_value_disordered REAL,
    experimental_value REAL,
    experimental_source TEXT,
    mae_ordered REAL,
    mae_disordered REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sim_parent ON simulations(parent_formula, target_site_species);
CREATE INDEX IF NOT EXISTS idx_sim_dopant ON simulations(dopant_element, concentration_pct);
CREATE INDEX IF NOT EXISTS idx_sim_run   ON simulations(run_id);
CREATE INDEX IF NOT EXISTS idx_prune_run ON pruning_records(run_id);
"""


def _new_id() -> str:
    return str(uuid.uuid4())


def _config_hash(config: dict) -> str:
    """SHA-256 of the JSON-serialised pipeline config (deterministic)."""
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _j(obj) -> Optional[str]:
    """Serialise to JSON string, or None if obj is None."""
    return json.dumps(obj) if obj is not None else None


def _dj(s: Optional[str]) -> Optional[dict]:
    """Deserialise JSON string to dict, or None if s is None."""
    return json.loads(s) if s else None


class LocalStore:
    """SQLite-backed persistence for pipeline results.

    Usage:
        store = LocalStore("data/results.db")
        store.save_pruning_record(run_id, records)
        sim_id = store.save_simulation(result, run_id, config_hash)
    """

    def __init__(self, db_path: str | pathlib.Path = "data/results.db"):
        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._apply_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _apply_schema(self) -> None:
        self._con.executescript(_DDL)
        self._migrate_schema()
        self._con.commit()

    def _migrate_schema(self) -> None:
        """Add columns introduced after initial schema creation (idempotent)."""
        migrations = [
            "ALTER TABLE pruning_records ADD COLUMN stage4_viability_reason TEXT",
        ]
        for sql in migrations:
            try:
                self._con.execute(sql)
            except Exception:
                pass  # column already exists

    # ── Pruning records ───────────────────────────────────────────────────────

    def save_pruning_record(self, run_id: str, records: list[PruningRecord | dict]) -> None:
        """Insert pruning records for all elements in one pipeline run.

        ``records`` may be PruningRecord dataclass instances or raw dicts
        following the same field names.
        """
        rows = []
        for rec in records:
            if isinstance(rec, dict):
                r = rec
            else:
                r = rec.__dict__
            rows.append((
                _new_id(),
                run_id,
                r.get("parent_formula", ""),
                r.get("target_site_species", ""),
                r.get("element", ""),
                r.get("stage1_passed", False),
                r.get("stage1_oxidation_state"),
                r.get("stage2_passed", False),
                r.get("stage2_mismatch_pct"),
                r.get("stage3_passed", False),
                r.get("stage3_sub_probability"),
                r.get("stage4_passed", False),
                _j(r.get("stage4_predicted_property")),
                r.get("stage4_viability_reason"),
                _j(r.get("thresholds_used")),
            ))
        self._con.executemany(
            """
            INSERT INTO pruning_records
            (id, run_id, parent_formula, target_site_species, element,
             stage1_passed, stage1_oxidation_state,
             stage2_passed, stage2_mismatch_pct,
             stage3_passed, stage3_sub_probability,
             stage4_passed, stage4_predicted_property,
             stage4_viability_reason, thresholds_used)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self._con.commit()

    def get_pruning_records(self, run_id: str) -> list[dict]:
        """Return all pruning records for a pipeline run as plain dicts."""
        cur = self._con.execute(
            "SELECT * FROM pruning_records WHERE run_id = ?", (run_id,)
        )
        return [dict(row) for row in cur.fetchall()]

    # ── Simulation results ────────────────────────────────────────────────────

    def save_simulation(
        self,
        result: SimulationResult,
        run_id: str,
        config_hash: str = "",
    ) -> str:
        """Insert a simulation result and return its UUID."""
        sim_id = _new_id()
        r = result
        self._con.execute(
            """
            INSERT INTO simulations
            (id, run_id,
             parent_formula, parent_mp_id, structure_type,
             target_site_species, target_oxidation_state,
             dopant_element, dopant_oxidation_state,
             concentration_pct, supercell_size, n_atoms, sqs_realisation_index,
             mlip_name, mlip_version,
             relaxation_converged, relaxation_steps, abort_reason,
             initial_energy_per_atom, final_energy_per_atom, max_force_final,
             formation_energy_above_hull, li_ni_exchange_energy, voltage,
             volume_change_pct, lattice_params,
             ordered_formation_energy, ordered_voltage, ordered_li_ni_exchange,
             disorder_sensitivity, pipeline_config_hash)
            VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                sim_id, run_id,
                r.parent_formula, r.parent_mp_id, r.structure_type,
                r.target_site_species, r.target_oxidation_state,
                r.dopant_element, r.dopant_oxidation_state,
                r.concentration_pct, _j(r.supercell_size), r.n_atoms, r.sqs_realisation_index,
                r.mlip_name, r.mlip_version,
                r.relaxation_converged, r.relaxation_steps, r.abort_reason,
                r.initial_energy_per_atom, r.final_energy_per_atom, r.max_force_final,
                r.formation_energy_above_hull, r.li_ni_exchange_energy, r.voltage,
                r.volume_change_pct, _j(r.lattice_params),
                r.ordered_formation_energy, r.ordered_voltage, r.ordered_li_ni_exchange,
                _j(r.disorder_sensitivity), config_hash,
            ),
        )
        self._con.commit()
        return sim_id

    def find_simulation(
        self,
        parent: str,
        dopant: str,
        conc: float,
        mlip: str,
        mlip_version: str,
    ) -> Optional[SimulationResult]:
        """Cache lookup: return the first matching SimulationResult or None."""
        cur = self._con.execute(
            """
            SELECT * FROM simulations
            WHERE parent_formula = ?
              AND dopant_element = ?
              AND concentration_pct = ?
              AND mlip_name = ?
              AND mlip_version = ?
            LIMIT 1
            """,
            (parent, dopant, conc, mlip, mlip_version),
        )
        row = cur.fetchone()
        return self._row_to_simulation(row) if row else None

    def get_run_results(self, run_id: str) -> list[SimulationResult]:
        """All simulation results for a pipeline run."""
        cur = self._con.execute(
            "SELECT * FROM simulations WHERE run_id = ?", (run_id,)
        )
        return [self._row_to_simulation(r) for r in cur.fetchall()]

    def get_all_for_parent(self, parent_formula: str) -> list[SimulationResult]:
        """All simulations ever run for a given parent material."""
        cur = self._con.execute(
            "SELECT * FROM simulations WHERE parent_formula = ?", (parent_formula,)
        )
        return [self._row_to_simulation(r) for r in cur.fetchall()]

    # ── Experimental comparisons ──────────────────────────────────────────────

    def save_experimental_comparison(
        self, sim_id: str, comparison: ExperimentalComparison | dict
    ) -> None:
        """Link a simulation to experimental reference data."""
        if isinstance(comparison, dict):
            c = comparison
        else:
            c = comparison.__dict__
        self._con.execute(
            """
            INSERT INTO experimental_comparisons
            (id, simulation_id, property_name,
             computed_value_ordered, computed_value_disordered,
             experimental_value, experimental_source,
             mae_ordered, mae_disordered)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                _new_id(),
                sim_id,
                c.get("property_name", ""),
                c.get("computed_value_ordered"),
                c.get("computed_value_disordered"),
                c.get("experimental_value"),
                c.get("experimental_source"),
                c.get("mae_ordered"),
                c.get("mae_disordered"),
            ),
        )
        self._con.commit()

    # ── Schema introspection ──────────────────────────────────────────────────

    def list_tables(self) -> list[str]:
        """Return names of all tables in the database."""
        cur = self._con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cur.fetchall()]

    def table_columns(self, table: str) -> list[str]:
        """Return column names for a given table."""
        cur = self._con.execute(f"PRAGMA table_info({table})")  # nosec — table is internal
        return [row[1] for row in cur.fetchall()]

    # ── Helper ────────────────────────────────────────────────────────────────

    def _row_to_simulation(self, row: sqlite3.Row) -> SimulationResult:
        d = dict(row)
        return SimulationResult(
            dopant_element=d["dopant_element"],
            dopant_oxidation_state=d["dopant_oxidation_state"],
            concentration_pct=d["concentration_pct"],
            sqs_realisation_index=d["sqs_realisation_index"],
            parent_formula=d["parent_formula"],
            parent_mp_id=d.get("parent_mp_id"),
            structure_type=d.get("structure_type"),
            target_site_species=d.get("target_site_species", ""),
            target_oxidation_state=d.get("target_oxidation_state"),
            supercell_size=_dj(d.get("supercell_size")),
            n_atoms=d.get("n_atoms"),
            mlip_name=d.get("mlip_name", ""),
            mlip_version=d.get("mlip_version", ""),
            relaxation_converged=bool(d.get("relaxation_converged")),
            relaxation_steps=d.get("relaxation_steps", 0),
            abort_reason=d.get("abort_reason"),
            initial_energy_per_atom=d.get("initial_energy_per_atom"),
            final_energy_per_atom=d.get("final_energy_per_atom"),
            max_force_final=d.get("max_force_final"),
            formation_energy_above_hull=d.get("formation_energy_above_hull"),
            li_ni_exchange_energy=d.get("li_ni_exchange_energy"),
            voltage=d.get("voltage"),
            volume_change_pct=d.get("volume_change_pct"),
            lattice_params=_dj(d.get("lattice_params")),
            ordered_formation_energy=d.get("ordered_formation_energy"),
            ordered_voltage=d.get("ordered_voltage"),
            ordered_li_ni_exchange=d.get("ordered_li_ni_exchange"),
            disorder_sensitivity=_dj(d.get("disorder_sensitivity")),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()
