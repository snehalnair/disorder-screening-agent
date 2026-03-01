"""
PipelineState — shared state schema for the disorder-screening LangGraph.

Design notes
------------
* All fields that are populated by later nodes are marked `NotRequired` so the
  graph can be initialised with just the input fields.
* `execution_log` uses an `Annotated` reducer (`operator.add`) so every node
  can append log entries without overwriting entries from earlier nodes.
* `parent_structure` and entries inside `sqs_structures` / `relaxed_results`
  are pymatgen `Structure` objects.  LangGraph's SqliteSaver serialises state
  via JSON; register a custom serialiser (see graph/checkpointer.py) if you
  need to persist these across process restarts.
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from pymatgen.core import Structure


class PipelineState(TypedDict, total=False):

    # ── Input (populated by parse_input) ─────────────────────────────────────
    parent_formula: str
    parent_structure: Structure         # pymatgen Structure of the host material
    target_site_species: str            # element being substituted, e.g. "Co"
    target_oxidation_state: int         # formal OS of the target site, e.g. 3
    target_coordination_number: int     # CN of the target site, e.g. 6
    target_properties: list[str]        # e.g. ["voltage", "li_ni_exchange"]
    constraints: dict                   # optional user constraints (cost, toxicity …)

    # ── Stage 1 output (smact_filter) ────────────────────────────────────────
    # Each entry: {element, oxidation_state, is_aliovalent}
    stage1_candidates: NotRequired[list[dict]]
    stage1_unique_elements: NotRequired[int]    # unique element count (paper metric)
    stage1_os_combinations: NotRequired[int]    # total element-OS pairs screened

    # ── Stage 2 output (radius_screen) ───────────────────────────────────────
    # Adds: shannon_radius, mismatch_pct, tolerance_factor
    stage2_candidates: NotRequired[list[dict]]

    # ── Stage 3 output (substitution_prob) ───────────────────────────────────
    # Adds: sub_probability
    stage3_candidates: NotRequired[list[dict]]

    # ── Stage 4 output (ml_prescreen — optional) ─────────────────────────────
    # Adds: ml_predicted_value
    stage4_candidates: NotRequired[list[dict]]

    # ── Stage 5a output (generate_sqs) ───────────────────────────────────────
    # {"Al": [Structure, Structure, ...], "Ti": [...], ...}
    sqs_structures: NotRequired[dict[str, list[Structure]]]

    # ── Stage 5 baseline (compute_baseline) ──────────────────────────────────
    # Undoped parent supercell reference properties
    baseline_result: NotRequired[dict]

    # ── Stage 5b output (mlip_relax) ─────────────────────────────────────────
    # {"Al": [{structure, energy, energy_per_atom, converged, n_steps}, ...], ...}
    relaxed_results: NotRequired[dict[str, list[dict]]]

    # ── Stage 5c output (compute_properties) ─────────────────────────────────
    # {"Al": {"voltage": 3.8, "li_ni_exchange": 0.42, ...}, ...}
    property_results: NotRequired[dict[str, dict]]

    # ── Stage 5c* output (compute_ordered_baseline) ──────────────────────────
    # Same schema as property_results but computed on an ordered supercell
    ordered_results: NotRequired[dict[str, dict]]

    # ── Stage 5 flat list of SimulationResult objects ─────────────────────────
    # One entry per (dopant, concentration, SQS realisation) — used by ranker
    simulation_results: NotRequired[list]

    # ── Run identity ──────────────────────────────────────────────────────────
    run_id: NotRequired[str]   # UUID for this pipeline run (used for DB persistence)

    # ── Final output ──────────────────────────────────────────────────────────
    # rank_and_report populates ranked_report; generate_summary populates summary
    ranked_report: NotRequired[dict]    # rankings, disorder_sensitivities, spearman_rho …
    summary: NotRequired[str]           # human-readable LLM summary
    report_path: NotRequired[str]       # path to the rendered Markdown report

    # ── Routing state (check_count) ───────────────────────────────────────────
    retry_count: NotRequired[int]       # how many retries have been attempted
    retry_stage: NotRequired[str]       # which stage is being retried

    # ── Execution log (appended by every node) ────────────────────────────────
    # Annotated with operator.add so LangGraph merges lists across node updates
    execution_log: Annotated[list[str], operator.add]

    # ── Pipeline config (loaded once by parse_input from pipeline.yaml) ───────
    config: NotRequired[dict]
