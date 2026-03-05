"""
LangGraph graph definitions for the disorder-screening pipeline.

Phase 1 exports ``build_pruning_graph()`` — a compiled Stages 1-3 subgraph.
Phase 4 completes ``build_full_graph()`` — the full pipeline with property
calculation, SimulationResult assembly, and dopant ranking.

Graph topologies
----------------
Pruning (Phase 1+):
    stage1_smact → stage2_radius → stage3_substitution → stage4_viability → END

Full pipeline (Phase 4+):
    stage1_smact → stage2_radius → stage3_substitution → stage4_viability
        → compute_baseline → stage5_simulate → rank_and_report → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from graph.state import PipelineState
from stages.stage1_smact import run_stage1_smact
from stages.stage2_radius import run_stage2_radius
from stages.stage3_substitution import run_stage3_substitution
from stages.stage4_viability import run_stage4_viability

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: pruning subgraph (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def build_pruning_graph():
    """Build and compile the Stages 1–3 pruning subgraph.

    Graph topology (linear):
        stage1_smact → stage2_radius → stage3_substitution → END

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to ``.invoke()``.
    """
    g = StateGraph(PipelineState)

    g.add_node("stage1_smact", run_stage1_smact)
    g.add_node("stage2_radius", run_stage2_radius)
    g.add_node("stage3_substitution", run_stage3_substitution)
    g.add_node("stage4_viability", run_stage4_viability)

    g.set_entry_point("stage1_smact")
    g.add_edge("stage1_smact", "stage2_radius")
    g.add_edge("stage2_radius", "stage3_substitution")
    g.add_edge("stage3_substitution", "stage4_viability")
    g.add_edge("stage4_viability", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Stage 5 node implementations
# ─────────────────────────────────────────────────────────────────────────────


def compute_baseline_node(state: dict) -> dict:
    """
    LangGraph node: relax the undoped parent supercell.

    Reads ``parent_structure`` and simulation config from state.
    If ``parent_structure`` is not set (pruning-only mode), returns a
    minimal baseline dict without running a relaxation.

    Returns:
        Updates to PipelineState: ``baseline_result`` key.
    """
    from stages.stage5.baseline import compute_baseline
    from stages.stage5.calculators import get_calculator

    parent_structure = state.get("parent_structure")
    if parent_structure is None:
        logger.warning(
            "compute_baseline_node: parent_structure not in state — skipping."
        )
        return {
            "execution_log": ["Stage 5 baseline: skipped (no parent_structure in state)"],
        }

    sim_cfg = _get_sim_config(state)
    supercell_matrix = sim_cfg.get("supercell", [2, 2, 2])
    fmax = sim_cfg.get("fmax", 0.05)
    max_steps = sim_cfg.get("max_relax_steps", 500)
    mlip_name = sim_cfg.get("potential", "mock")
    device = sim_cfg.get("device", "auto")

    calculator = get_calculator(mlip_name, device=device)

    baseline = compute_baseline(
        parent_structure=parent_structure,
        supercell_matrix=supercell_matrix,
        calculator=calculator,
        target_properties=state.get("target_properties", []),
        fmax=fmax,
        max_steps=max_steps,
    )

    converged_str = "converged" if baseline["relaxation_converged"] else "NOT converged"
    log_msg = (
        f"Stage 5 baseline: {converged_str} in {baseline['relaxation_steps']} steps, "
        f"E/atom = {baseline['energy_per_atom']:.4f} eV."
    )
    return {
        "baseline_result": baseline,
        "execution_log": [log_msg],
    }


def stage5_simulate_node(state: dict) -> dict:
    """
    LangGraph node: run SQS + MLIP relaxation + property calculation.

    For each candidate × concentration × SQS realisation:
      1. Generate SQS structure.
      2. Relax with MLIP + RelaxationMonitor.
      3. Compute battery properties (voltage, Li/Ni exchange, etc.).
      4. Build SimulationResult objects with all fields populated.

    Also computes ordered-cell properties (once per dopant at first concentration)
    for disorder-sensitivity analysis.

    Returns:
        Updates to PipelineState: ``relaxed_results``, ``ordered_results``,
        ``simulation_results`` keys.
    """
    from db.models import SimulationResult
    from stages.stage5.calculators import get_calculator
    from stages.stage5.mlip_relaxation import relax_structure
    from stages.stage5.property_calculator import (
        compute_ordered_properties,
        compute_properties,
    )
    from stages.stage5.sqs_generator import generate_sqs

    parent_structure = state.get("parent_structure")
    if parent_structure is None:
        logger.warning(
            "stage5_simulate_node: parent_structure not in state — skipping."
        )
        return {
            "execution_log": ["Stage 5 simulate: skipped (no parent_structure in state)"],
            "relaxed_results": {},
            "simulation_results": [],
            "ordered_results": {},
        }

    sim_cfg = _get_sim_config(state)
    concentrations: list[float] = sim_cfg.get("concentrations", [0.05, 0.10])
    supercell_matrix = sim_cfg.get("supercell", [2, 2, 2])
    n_sqs: int = sim_cfg.get("n_sqs_realisations", 3)
    fmax: float = sim_cfg.get("fmax", 0.05)
    max_steps: int = sim_cfg.get("max_relax_steps", 500)
    mlip_name: str = sim_cfg.get("potential", "mock")
    device: str = sim_cfg.get("device", "auto")

    pipeline_cfg = state.get("config", {}).get("pipeline", {})
    include_ordered: bool = (
        pipeline_cfg.get("output", {}).get("include_ordered_comparison", True)
    )
    target_properties: list[str] = state.get("target_properties") or (
        list(pipeline_cfg.get("property_weights", {}).keys())
        or ["voltage", "formation_energy"]
    )

    calculator = get_calculator(mlip_name, device=device)
    target_species: str = state.get("target_site_species", "")
    parent_formula: str = state.get("parent_formula", "")

    # Prefer viability-filtered list; fall back to ML pre-screen output, then Stage 3
    candidates: list[dict] = (
        state.get("stage4_viability_candidates")
        or state.get("stage4_candidates")
        or state.get("stage3_candidates", [])
    )

    all_sim_results: list[SimulationResult] = []
    ordered_results: dict[str, dict] = {}
    relaxed_results: dict[str, list[dict]] = {}
    errors: list[dict] = []

    for candidate in candidates:
        element: str = candidate["element"]
        dopant_os: int = candidate.get("oxidation_state", 0)
        relaxed_results[element] = []

        # ── Ordered-cell properties (once per dopant) ──────────────────────
        if include_ordered and concentrations:
            try:
                ordered_props = compute_ordered_properties(
                    parent_structure=parent_structure,
                    dopant_element=element,
                    target_species=target_species,
                    concentration=concentrations[0],
                    supercell_matrix=supercell_matrix,
                    calculator=calculator,
                    target_properties=target_properties,
                    max_steps=max_steps,
                )
                ordered_results[element] = ordered_props
            except Exception as exc:
                logger.warning(
                    "stage5_simulate: ordered properties failed for %s: %s", element, exc
                )
                ordered_results[element] = {}

        # ── SQS + relax + property calculation ────────────────────────────
        for conc in concentrations:
            try:
                sqs_structures = generate_sqs(
                    parent_structure=parent_structure,
                    dopant_element=element,
                    target_species=target_species,
                    concentration=conc,
                    supercell_matrix=supercell_matrix,
                    n_realisations=n_sqs,
                )
            except ValueError as exc:
                errors.append(
                    {"stage": "5a", "dopant": element, "concentration": conc, "error": str(exc)}
                )
                continue

            for i, sqs in enumerate(sqs_structures):
                relax_res = relax_structure(
                    structure=sqs,
                    calculator=calculator,
                    fmax=fmax,
                    max_steps=max_steps,
                )

                # Compute battery properties on relaxed structure
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
                            "stage5_simulate: property computation failed for %s@%.0f%%[%d]: %s",
                            element, conc * 100, i, exc,
                        )

                # Disorder sensitivity from ordered vs disordered means
                ord_props = ordered_results.get(element, {})
                disorder_sensitivity: dict = {}
                for prop in target_properties:
                    dis_val = props.get(prop)
                    ord_val = ord_props.get(prop)
                    if (
                        dis_val is not None and ord_val is not None
                        and ord_val != 0
                        and isinstance(dis_val, (int, float))
                        and isinstance(ord_val, (int, float))
                    ):
                        disorder_sensitivity[prop] = abs(dis_val - ord_val) / abs(ord_val)

                # Build SimulationResult with all fields
                sim_result = SimulationResult(
                    dopant_element=element,
                    dopant_oxidation_state=dopant_os,
                    concentration_pct=conc * 100.0,
                    sqs_realisation_index=i,
                    parent_formula=parent_formula,
                    target_site_species=target_species,
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
                    ordered_formation_energy=ord_props.get("formation_energy"),
                    ordered_voltage=ord_props.get("voltage"),
                    ordered_li_ni_exchange=ord_props.get("li_ni_exchange"),
                    disorder_sensitivity=disorder_sensitivity or None,
                )
                all_sim_results.append(sim_result)

                # Keep relaxed_results dict for backwards compatibility
                relaxed_results[element].append(
                    {
                        "concentration": conc,
                        "sqs_realisation_index": i,
                        "relaxed_structure": relax_res.relaxed_structure,
                        "relaxation_converged": relax_res.relaxation_converged,
                        "relaxation_steps": relax_res.relaxation_steps,
                        "initial_energy_per_atom": relax_res.initial_energy_per_atom,
                        "final_energy_per_atom": relax_res.final_energy_per_atom,
                        "max_force_final": relax_res.max_force_final,
                        "abort_reason": relax_res.abort_reason,
                        **{k: v for k, v in props.items() if not isinstance(v, dict)},
                    }
                )

    # ── DB persistence ────────────────────────────────────────────────────────
    run_id: str = state.get("run_id", "")
    if run_id and all_sim_results:
        try:
            from db.local_store import LocalStore
            db_path = (
                state.get("config", {})
                .get("pipeline", {})
                .get("database", {})
                .get("local", {})
                .get("path", "data/results.db")
            )
            store = LocalStore(db_path)
            for sim in all_sim_results:
                store.save_simulation(sim, run_id)
            store.close()
            logger.info("stage5_simulate: saved %d results to %s.", len(all_sim_results), db_path)
        except Exception as exc:
            logger.warning("stage5_simulate: DB persistence failed: %s", exc)

    n_total = sum(len(v) for v in relaxed_results.values())
    log_msg = (
        f"Stage 5 simulate: {len(candidates)} candidates × "
        f"{len(concentrations)} conc × {n_sqs} SQS = "
        f"{n_total} relaxations, {len(all_sim_results)} SimulationResults."
    )
    if errors:
        log_msg += f" {len(errors)} SQS errors."

    return {
        "relaxed_results": relaxed_results,
        "ordered_results": ordered_results,
        "simulation_results": all_sim_results,
        "execution_log": [log_msg] + [
            f"Stage 5 error: {e['dopant']}@{e['concentration']:.0%}: {e['error']}"
            for e in errors
        ],
    }


def rank_and_report_node(state: dict) -> dict:
    """
    LangGraph node: rank dopants and assemble the final report.

    Reads ``simulation_results`` (list[SimulationResult]) from state,
    calls ``rank_dopants()``, and stores the serialisable ranked report
    under ``ranked_report``.
    """
    from ranking.ranker import rank_dopants

    sim_results = state.get("simulation_results") or []
    if not sim_results:
        n_relaxed = len(state.get("relaxed_results") or {})
        return {
            "execution_log": [
                f"rank_and_report: no SimulationResults in state "
                f"({n_relaxed} relaxed_results keys present)."
            ],
            "ranked_report": {},
        }

    pipeline_cfg = state.get("config", {}).get("pipeline", {})
    target_properties: list[str] = state.get("target_properties") or (
        list(pipeline_cfg.get("property_weights", {}).keys()) or None
    )
    ordered_results = state.get("ordered_results") or None
    top_n: int = pipeline_cfg.get("output", {}).get("top_n", 3)
    sanity_cfg: dict = {}

    report = rank_dopants(
        simulation_results=sim_results,
        target_properties=target_properties or None,
        ordered_results=ordered_results,
        top_n=top_n,
        sanity_config=sanity_cfg,
    )

    # Serialise to plain dict for LangGraph state storage
    ranked_report = {
        "parent_formula": report.parent_formula,
        "target_site": report.target_site,
        "candidates_simulated": report.candidates_simulated,
        "primary_property": report.primary_property,
        "recommended": report.recommended,
        "spearman_rho": report.spearman_rho,
        "warnings": report.warnings,
        "all_rankings": report.all_rankings,
        "rankings": [
            {
                "dopant": ds.dopant,
                "n_converged": ds.n_converged,
                "properties": ds.properties,
                "ordered_properties": ds.ordered_properties,
                "disorder_sensitivity": ds.disorder_sensitivity,
                "rank_by_property": ds.rank_by_property,
            }
            for ds in report.rankings
        ],
    }

    rec_str = ", ".join(report.recommended[:3]) if report.recommended else "none"
    log_msg = (
        f"rank_and_report: {report.candidates_simulated} candidates ranked. "
        f"Recommended: {rec_str}. {len(report.warnings)} warnings."
    )
    return {
        "ranked_report": ranked_report,
        "execution_log": [log_msg],
    }


def generate_summary_node(state: dict) -> dict:
    """LangGraph node: render the Markdown screening report.

    Reads ``ranked_report`` from state and writes a Markdown file to
    ``reports/{run_id}_screening_report.md``.  Skips gracefully when
    ``ranked_report`` is absent or empty.
    """
    from pipeline_io.generate_summary import generate_report

    ranked_report = state.get("ranked_report") or {}
    if not ranked_report:
        return {
            "execution_log": ["generate_summary: no ranked_report in state — skipping."]
        }

    try:
        report_dir = (
            state.get("config", {})
            .get("pipeline", {})
            .get("output", {})
            .get("report_dir", "reports")
        )
        run_id = state.get("run_id", "unknown")
        import pathlib
        out_path = pathlib.Path(report_dir) / f"{run_id}_screening_report.md"
        written = generate_report(ranked_report, state, output_path=out_path)
        log_msg = f"generate_summary: report written to {written}."
        return {
            "report_path": str(written),
            "execution_log": [log_msg],
        }
    except Exception as exc:
        logger.warning("generate_summary_node failed: %s", exc)
        return {
            "execution_log": [f"generate_summary: failed — {exc}"]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4+: full pipeline graph
# ─────────────────────────────────────────────────────────────────────────────


def build_full_graph():
    """Build and compile the full pipeline graph (Stages 1–3 + Stage 5).

    Graph topology:
        stage1_smact → stage2_radius → stage3_substitution
            → compute_baseline → stage5_simulate
            → rank_and_report → generate_summary → END

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to ``.invoke()``.
    """
    g = StateGraph(PipelineState)

    # Pruning nodes
    g.add_node("stage1_smact", run_stage1_smact)
    g.add_node("stage2_radius", run_stage2_radius)
    g.add_node("stage3_substitution", run_stage3_substitution)
    g.add_node("stage4_viability", run_stage4_viability)

    # Stage 5 nodes
    g.add_node("compute_baseline", compute_baseline_node)
    g.add_node("stage5_simulate", stage5_simulate_node)
    g.add_node("rank_and_report", rank_and_report_node)
    g.add_node("generate_summary", generate_summary_node)

    # Edges
    g.set_entry_point("stage1_smact")
    g.add_edge("stage1_smact", "stage2_radius")
    g.add_edge("stage2_radius", "stage3_substitution")
    g.add_edge("stage3_substitution", "stage4_viability")
    g.add_edge("stage4_viability", "compute_baseline")
    g.add_edge("compute_baseline", "stage5_simulate")
    g.add_edge("stage5_simulate", "rank_and_report")
    g.add_edge("rank_and_report", "generate_summary")
    g.add_edge("generate_summary", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_sim_config(state: dict) -> dict:
    """Extract simulation config from state, returning defaults if missing."""
    return (
        state.get("config", {})
        .get("pipeline", {})
        .get("stage5_simulation", {})
    )
