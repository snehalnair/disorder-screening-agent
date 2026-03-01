"""
Entry points for running pipeline subsets without building the graph manually.

Provides:
    run_stages_1_3(...)    — run the chemical pruning funnel (Stages 1-3)
    run_full_pipeline(...) — run the complete pipeline end-to-end
    run_single_dopant(...) — run Stage 5 for one dopant (no pruning)
    run_comparison(...)    — compare results from two or more pipeline runs
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_config(config_path: str | pathlib.Path | None = None) -> dict:
    """Load pipeline.yaml. Defaults to config/pipeline.yaml relative to repo root."""
    if config_path is None:
        config_path = pathlib.Path(__file__).parent.parent / "config" / "pipeline.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _open_store(db_path: Optional[str | pathlib.Path], config: dict):
    """Open a LocalStore at db_path (falls back to config → 'data/results.db')."""
    from db.local_store import LocalStore

    if db_path is None:
        db_path = (
            config.get("pipeline", {})
            .get("database", {})
            .get("local", {})
            .get("path", "data/results.db")
        )
    return LocalStore(db_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 1: Stages 1-3 pruning
# ─────────────────────────────────────────────────────────────────────────────


def run_stages_1_3(
    parent_formula: str,
    target_site_species: str,
    target_oxidation_state: int,
    target_coordination_number: int = 6,
    target_properties: list[str] | None = None,
    constraints: dict[str, Any] | None = None,
    config_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Run the chemical pruning pipeline (Stages 1–3) and return the final state.

    Args:
        parent_formula:            Chemical formula of the host material
                                   (e.g. ``"LiNi0.8Mn0.1Co0.1O2"``).
        target_site_species:       Element symbol being substituted (e.g. ``"Co"``).
        target_oxidation_state:    Formal oxidation state of the target site
                                   (e.g. ``3``).
        target_coordination_number: CN of the target site (default ``6``).
        target_properties:         Properties to optimise downstream
                                   (e.g. ``["voltage", "li_ni_exchange"]``).
        constraints:               Optional user constraints dict; supports
                                   ``"exclude_elements"`` key.
        config_path:               Path to ``pipeline.yaml``; defaults to
                                   ``config/pipeline.yaml`` relative to project root.

    Returns:
        Final ``PipelineState`` dict containing:
        - ``stage1_candidates``  — output of SMACT filter
        - ``stage2_candidates``  — output of radius filter
        - ``stage3_candidates``  — output of substitution probability filter
        - ``execution_log``      — concatenated log entries from all three stages
    """
    from graph.graph import build_pruning_graph

    config = _load_config(config_path)

    initial_state: dict[str, Any] = {
        "parent_formula": parent_formula,
        "target_site_species": target_site_species,
        "target_oxidation_state": target_oxidation_state,
        "target_coordination_number": target_coordination_number,
        "target_properties": target_properties or [],
        "constraints": constraints or {},
        "config": config,
        "execution_log": [],
    }

    graph = build_pruning_graph()
    return graph.invoke(initial_state)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 2: Full pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_full_pipeline(
    parent_formula: str,
    parent_structure,
    target_site_species: str,
    target_oxidation_state: int,
    target_coordination_number: int = 6,
    target_properties: list[str] | None = None,
    constraints: dict[str, Any] | None = None,
    config_path: str | pathlib.Path | None = None,
    db_path: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the complete disorder-screening pipeline end-to-end.

    Executes Stages 1–3 (chemical pruning) followed by Stage 5 (SQS
    generation, MLIP relaxation, property calculation) and ranking.
    Results are saved to SQLite if ``run_id`` is provided (or auto-generated).

    Args:
        parent_formula:            Host material formula (e.g. ``"LiNi0.8Mn0.1Co0.1O2"``).
        parent_structure:          Pymatgen ``Structure`` of the host material.
        target_site_species:       Element being substituted (e.g. ``"Co"``).
        target_oxidation_state:    Formal OS of target site (e.g. ``3``).
        target_coordination_number: CN of target site (default ``6``).
        target_properties:         Properties to rank by (e.g. ``["voltage"]``).
                                   Defaults to keys in ``property_weights`` config.
        constraints:               Optional constraints dict.
        config_path:               Path to ``pipeline.yaml``.
        db_path:                   Path to SQLite results DB. Defaults to config value.
        run_id:                    UUID string for this run. Auto-generated if None.

    Returns:
        Final ``PipelineState`` dict with all stage outputs including
        ``ranked_report`` and ``simulation_results``.
    """
    from graph.graph import build_full_graph

    config = _load_config(config_path)
    run_id = run_id or str(uuid.uuid4())

    initial_state: dict[str, Any] = {
        "parent_formula": parent_formula,
        "parent_structure": parent_structure,
        "target_site_species": target_site_species,
        "target_oxidation_state": target_oxidation_state,
        "target_coordination_number": target_coordination_number,
        "target_properties": target_properties or [],
        "constraints": constraints or {},
        "config": config,
        "execution_log": [],
        "run_id": run_id,
    }

    graph = build_full_graph()
    final_state = graph.invoke(initial_state)

    logger.info(
        "run_full_pipeline: completed run_id=%s, %d SimulationResults.",
        run_id,
        len(final_state.get("simulation_results") or []),
    )
    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 3: Single-dopant Stage 5
# ─────────────────────────────────────────────────────────────────────────────


def run_single_dopant(
    parent_formula: str,
    parent_structure,
    dopant_element: str,
    dopant_oxidation_state: int,
    target_site_species: str,
    concentrations: list[float] | None = None,
    target_properties: list[str] | None = None,
    config_path: str | pathlib.Path | None = None,
    db_path: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run Stage 5 (SQS + MLIP + properties) for a single dopant.

    Bypasses Stages 1–3; useful for targeted re-runs or manual candidates.
    Saves results to SQLite if a ``run_id`` is given.

    Args:
        parent_formula:         Host material formula.
        parent_structure:       Pymatgen ``Structure`` of the host material.
        dopant_element:         Dopant element symbol (e.g. ``"Al"``).
        dopant_oxidation_state: Formal oxidation state of the dopant.
        target_site_species:    Element being substituted (e.g. ``"Co"``).
        concentrations:         Site fractions to evaluate (e.g. ``[0.05, 0.10]``).
                                Defaults to config values.
        target_properties:      Properties to compute (e.g. ``["voltage"]``).
        config_path:            Path to ``pipeline.yaml``.
        db_path:                SQLite results DB path.
        run_id:                 UUID for this run. Auto-generated if None.

    Returns:
        Dict with ``run_id``, ``simulation_results`` (list of SimulationResult),
        and ``ordered_results`` dict.
    """
    from db.models import SimulationResult
    from stages.stage5.calculators import get_calculator
    from stages.stage5.mlip_relaxation import relax_structure
    from stages.stage5.property_calculator import (
        compute_ordered_properties,
        compute_properties,
    )
    from stages.stage5.sqs_generator import generate_sqs

    config = _load_config(config_path)
    run_id = run_id or str(uuid.uuid4())

    sim_cfg = config.get("pipeline", {}).get("stage5_simulation", {})
    concentrations = concentrations or sim_cfg.get("concentrations", [0.05, 0.10])
    supercell_matrix = sim_cfg.get("supercell", [2, 2, 2])
    n_sqs: int = sim_cfg.get("n_sqs_realisations", 3)
    fmax: float = sim_cfg.get("fmax", 0.05)
    max_steps: int = sim_cfg.get("max_relax_steps", 500)
    mlip_name: str = sim_cfg.get("potential", "mock")
    device: str = sim_cfg.get("device", "auto")

    target_properties = target_properties or list(
        config.get("pipeline", {}).get("property_weights", {}).keys()
    ) or ["voltage", "formation_energy"]

    calculator = get_calculator(mlip_name, device=device)
    sim_results: list[SimulationResult] = []

    # ── Ordered-cell baseline (once, at first concentration) ──────────────
    ordered_props: dict = {}
    if concentrations:
        try:
            ordered_props = compute_ordered_properties(
                parent_structure=parent_structure,
                dopant_element=dopant_element,
                target_species=target_site_species,
                concentration=concentrations[0],
                supercell_matrix=supercell_matrix,
                calculator=calculator,
                target_properties=target_properties,
                max_steps=max_steps,
            )
        except Exception as exc:
            logger.warning(
                "run_single_dopant: ordered properties failed for %s: %s",
                dopant_element, exc,
            )

    # ── SQS + relax + properties ──────────────────────────────────────────
    for conc in concentrations:
        try:
            sqs_structures = generate_sqs(
                parent_structure=parent_structure,
                dopant_element=dopant_element,
                target_species=target_site_species,
                concentration=conc,
                supercell_matrix=supercell_matrix,
                n_realisations=n_sqs,
            )
        except ValueError as exc:
            logger.warning(
                "run_single_dopant: SQS failed for %s@%.0f%%: %s",
                dopant_element, conc * 100, exc,
            )
            continue

        for i, sqs in enumerate(sqs_structures):
            relax_res = relax_structure(
                structure=sqs,
                calculator=calculator,
                fmax=fmax,
                max_steps=max_steps,
            )

            props: dict = {}
            if relax_res.relaxation_converged:
                try:
                    props = compute_properties(
                        relaxed_structure=relax_res.relaxed_structure,
                        parent_structure=parent_structure,
                        calculator=calculator,
                        target_properties=target_properties,
                        final_energy_per_atom=relax_res.final_energy_per_atom,
                    )
                except Exception as exc:
                    logger.warning(
                        "run_single_dopant: property computation failed: %s", exc
                    )

            disorder_sensitivity: dict = {}
            for prop in target_properties:
                dis_val = props.get(prop)
                ord_val = ordered_props.get(prop)
                if (
                    dis_val is not None and ord_val is not None
                    and ord_val != 0
                    and isinstance(dis_val, (int, float))
                    and isinstance(ord_val, (int, float))
                ):
                    disorder_sensitivity[prop] = abs(dis_val - ord_val) / abs(ord_val)

            sim_results.append(
                SimulationResult(
                    dopant_element=dopant_element,
                    dopant_oxidation_state=dopant_oxidation_state,
                    concentration_pct=conc * 100.0,
                    sqs_realisation_index=i,
                    parent_formula=parent_formula,
                    target_site_species=target_site_species,
                    supercell_size=supercell_matrix,
                    mlip_name=mlip_name,
                    mlip_version=calculator.get_version(),
                    relaxation_converged=relax_res.relaxation_converged,
                    relaxation_steps=relax_res.relaxation_steps,
                    abort_reason=relax_res.abort_reason,
                    initial_energy_per_atom=relax_res.initial_energy_per_atom,
                    final_energy_per_atom=relax_res.final_energy_per_atom,
                    max_force_final=relax_res.max_force_final,
                    formation_energy_above_hull=props.get("formation_energy"),
                    li_ni_exchange_energy=props.get("li_ni_exchange"),
                    voltage=props.get("voltage"),
                    volume_change_pct=props.get("volume_change"),
                    lattice_params=props.get("lattice_params"),
                    ordered_formation_energy=ordered_props.get("formation_energy"),
                    ordered_voltage=ordered_props.get("voltage"),
                    ordered_li_ni_exchange=ordered_props.get("li_ni_exchange"),
                    disorder_sensitivity=disorder_sensitivity or None,
                )
            )

    # ── DB persistence ────────────────────────────────────────────────────
    if sim_results:
        store = _open_store(db_path, config)
        try:
            for sim in sim_results:
                store.save_simulation(sim, run_id)
            logger.info(
                "run_single_dopant: saved %d results (run_id=%s).",
                len(sim_results), run_id,
            )
        finally:
            store.close()

    return {
        "run_id": run_id,
        "simulation_results": sim_results,
        "ordered_results": {dopant_element: ordered_props} if ordered_props else {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 4: Cross-run comparison
# ─────────────────────────────────────────────────────────────────────────────


def run_comparison(
    run_ids: list[str],
    target_properties: list[str] | None = None,
    db_path: str | None = None,
    config_path: str | pathlib.Path | None = None,
):
    """Compare simulation results from two or more pipeline runs.

    Fetches results from the SQLite database and computes property deltas,
    ranking changes, and Spearman ρ between runs.

    Args:
        run_ids:           UUIDs of the runs to compare (≥ 2 recommended).
        target_properties: Properties to compare. Defaults to all available.
        db_path:           SQLite results DB path.
        config_path:       Path to ``pipeline.yaml``.

    Returns:
        ``ComparisonReport`` dataclass.
    """
    from ranking.comparator import compare_runs

    config = _load_config(config_path)
    store = _open_store(db_path, config)
    try:
        report = compare_runs(run_ids, store, target_properties=target_properties)
    finally:
        store.close()
    return report
