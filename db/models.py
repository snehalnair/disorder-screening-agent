"""
Database model dataclasses for the disorder-screening pipeline.

These dataclasses mirror the SQLite schema in local_store.py and are used
as the canonical in-memory representation of simulation and pruning results.
They are intentionally plain Python — no ORM dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationResult:
    """One MLIP relaxation result for a specific (dopant, concentration, SQS realisation)."""

    # Identity
    dopant_element: str
    dopant_oxidation_state: int
    concentration_pct: float
    sqs_realisation_index: int
    parent_formula: str
    parent_mp_id: Optional[str] = None
    structure_type: Optional[str] = None
    target_site_species: str = ""
    target_oxidation_state: Optional[int] = None

    # Simulation config
    supercell_size: Optional[list[int]] = None   # e.g. [2, 2, 2]
    n_atoms: Optional[int] = None
    mlip_name: str = "mattersim"
    mlip_version: str = ""

    # Relaxation outcome
    relaxation_converged: bool = False
    relaxation_steps: int = 0
    abort_reason: Optional[str] = None
    initial_energy_per_atom: Optional[float] = None
    final_energy_per_atom: Optional[float] = None
    max_force_final: Optional[float] = None

    # Properties (battery-specific)
    formation_energy_above_hull: Optional[float] = None
    li_ni_exchange_energy: Optional[float] = None
    voltage: Optional[float] = None
    volume_change_pct: Optional[float] = None
    lattice_params: Optional[dict] = None

    # Ordered baseline comparison
    ordered_formation_energy: Optional[float] = None
    ordered_voltage: Optional[float] = None
    ordered_li_ni_exchange: Optional[float] = None
    disorder_sensitivity: Optional[dict] = None  # {property: relative_change}


@dataclass
class PruningRecord:
    """Pass/fail record for one element across all pruning stages in one pipeline run."""

    run_id: str
    parent_formula: str
    target_site_species: str
    element: str

    # Stage 1
    stage1_passed: bool = False
    stage1_oxidation_state: Optional[int] = None

    # Stage 2
    stage2_passed: bool = False
    stage2_mismatch_pct: Optional[float] = None

    # Stage 3
    stage3_passed: bool = False
    stage3_sub_probability: Optional[float] = None

    # Stage 4 — ML pre-screen (optional)
    stage4_passed: bool = False
    stage4_predicted_property: Optional[dict] = None

    # Stage 4 — Viability filter (safety/regulatory)
    stage4_viability_reason: Optional[str] = None   # e.g. "radioactive", "toxicity (carcinogen)"

    # Thresholds used (serialised as JSON in DB)
    thresholds_used: Optional[dict] = None


@dataclass
class ExperimentalComparison:
    """Link between a simulation result and experimental reference data."""

    simulation_id: str
    property_name: str
    computed_value_ordered: Optional[float] = None
    computed_value_disordered: Optional[float] = None
    experimental_value: Optional[float] = None
    experimental_source: Optional[str] = None   # DOI
    mae_ordered: Optional[float] = None
    mae_disordered: Optional[float] = None
