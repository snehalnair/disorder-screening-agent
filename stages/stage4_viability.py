"""
Stage 4 (Viability): element safety and regulatory metadata filter.

Performs a zero-cost metadata lookup against ``data/element_metadata.json``
to remove candidates that are radioactive or acutely toxic before committing
to expensive Stage 5 MLIP simulations.

Rationale
---------
Stages 1–3 are purely chemical (charge neutrality, radius compatibility,
substitution probability). They have no knowledge of real-world constraints.
This stage applies hard safety filters that any experimentalist or commercial
partner would independently enforce:

  * ``non_radioactive`` — removes elements where ``is_radioactive: true`` (e.g. U).
    These are regulated under nuclear-materials law in all jurisdictions.

  * ``non_toxic`` — removes elements where ``is_toxic: true``, which is set only
    for elements with serious synthesis-level hazards:
      - As  (IARC Group 1 carcinogen, RoHS restricted)
      - Cr  (Cr6+ formed during 700-900 °C O2 calcination; IARC Group 1, RoHS)
      - Os  (forms OsO4 in hot O2 atmosphere; TLV = 0.0002 mg m-3)
      - Sb  (IARC 2B, RoHS restricted; also confirmed_failed in NMC GT)

Filtered candidates are NOT discarded from the dataset. They are:
  1. Returned as ``stage4_viability_rejected`` in state (available for reporting).
  2. Logged to ``pruning_records`` with ``stage4_passed = False`` and
     ``stage4_viability_reason`` so the DB retains a complete audit trail.

Cost/scarcity (eu_crm_2023, usgs_criticality_2022, cost_annotation) are
annotated onto PASSING candidates as metadata but are NOT used as filters —
that trade-off belongs to the experimentalist.

Input state keys:
    stage3_candidates (list[dict])   — output of Stage 3
    config (dict)                    — loaded from pipeline.yaml
    run_id (str, optional)           — for DB persistence of pruning records
    parent_formula (str, optional)
    target_site_species (str, optional)

Output state keys:
    stage4_viability_candidates (list[dict])  — passed candidates, annotated with
                                                eu_crm_2023, toxicity_class,
                                                usgs_criticality, cost_annotation
    stage4_viability_rejected (list[dict])    — filtered candidates with
                                                viability_rejection_reason
    execution_log (list[str])
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

TOOL_METADATA = {
    "name": "element_viability_filter",
    "stage": "4v",
    "description": (
        "Zero-cost metadata lookup: removes radioactive and acutely toxic elements "
        "before Stage 5 simulations. Annotates survivors with criticality and cost data."
    ),
    "system_type": "periodic_crystal",
    "input_type": "list[CandidateDopant]",
    "output_type": "list[CandidateDopant] annotated with safety metadata",
    "cost": "milliseconds",
    "cost_per_candidate": "negligible",
    "typical_reduction": "29 → 24 candidates (removes U, As, Cr, Os, Sb)",
    "external_dependencies": ["data/element_metadata.json"],
    "requires_structure": False,
    "requires_network": False,
    "requires_gpu": False,
    "configurable_params": ["non_radioactive", "non_toxic"],
    "failure_modes": ["element not in metadata JSON → passed through with warning"],
}

_METADATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "element_metadata.json"


def _load_element_metadata(metadata_path: pathlib.Path | None = None) -> dict[str, dict]:
    """Load element_metadata.json and return the ``elements`` dict."""
    path = pathlib.Path(metadata_path) if metadata_path else _METADATA_PATH
    with open(path) as f:
        data = json.load(f)
    return data.get("elements", {})


def _rejection_reason(
    element: str,
    meta: dict,
    filter_radioactive: bool,
    filter_toxic: bool,
) -> str | None:
    """Return a rejection reason string, or None if the element passes."""
    if filter_radioactive and meta.get("is_radioactive", False):
        return "radioactive"
    if filter_toxic and meta.get("is_toxic", False):
        tclass = meta.get("toxicity_class", "toxic")
        return f"toxicity ({tclass})"
    return None


def run_stage4_viability(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: element safety and regulatory metadata filter."""
    stage3_candidates: list[dict] = state.get("stage3_candidates") or []

    cfg: dict = state.get("config") or {}
    stage_cfg: dict = (cfg.get("pipeline") or {}).get("stage4_viability") or {}
    constraints: dict = stage_cfg.get("constraints") or {}

    filter_radioactive: bool = constraints.get("non_radioactive", True)
    filter_toxic: bool = constraints.get("non_toxic", True)

    metadata_path = stage_cfg.get("metadata_path")
    try:
        metadata = _load_element_metadata(metadata_path)
    except FileNotFoundError as exc:
        logger.warning(
            "stage4_viability: metadata file not found (%s) — passing all candidates.", exc
        )
        return {
            "stage4_viability_candidates": stage3_candidates,
            "stage4_viability_rejected": [],
            "execution_log": [
                f"Stage 4 (Viability): metadata file missing — all {len(stage3_candidates)} "
                "candidates passed through."
            ],
        }

    passed: list[dict] = []
    rejected: list[dict] = []
    unknown_elements: list[str] = []

    for cand in stage3_candidates:
        element: str = cand["element"]
        meta = metadata.get(element)

        if meta is None:
            # Element not in metadata — pass through with warning
            unknown_elements.append(element)
            passed.append({
                **cand,
                "eu_crm_2023": None,
                "toxicity_class": "unknown",
                "usgs_criticality": "unknown",
                "cost_annotation": "unknown",
            })
            continue

        reason = _rejection_reason(element, meta, filter_radioactive, filter_toxic)
        if reason:
            rejected.append({**cand, "viability_rejection_reason": reason})
        else:
            passed.append({
                **cand,
                "eu_crm_2023": meta.get("eu_crm_2023", False),
                "toxicity_class": meta.get("toxicity_class", "unknown"),
                "usgs_criticality": meta.get("usgs_criticality_2022", "unknown"),
                "cost_annotation": meta.get("cost_annotation", ""),
            })

    # ── DB persistence: log pruning records for all elements ─────────────────
    run_id: str = state.get("run_id", "")
    if run_id and (passed or rejected):
        _persist_pruning_records(state, passed, rejected)

    # ── Build log messages ────────────────────────────────────────────────────
    log_parts = [
        f"Stage 4 (Viability): {len(passed)} passed, {len(rejected)} rejected "
        f"(from {len(stage3_candidates)} input)."
    ]
    if rejected:
        reasons = ", ".join(
            f"{c['element']} [{c['viability_rejection_reason']}]" for c in rejected
        )
        log_parts.append(f"  Rejected: {reasons}")
    if unknown_elements:
        log_parts.append(
            f"  Warning: {len(unknown_elements)} element(s) not in metadata, passed through: "
            + ", ".join(unknown_elements)
        )

    log_msg = " ".join(log_parts)
    logger.info(log_msg)

    return {
        "stage4_viability_candidates": passed,
        "stage4_viability_rejected": rejected,
        "execution_log": [log_msg],
    }


def _persist_pruning_records(
    state: dict,
    passed: list[dict],
    rejected: list[dict],
) -> None:
    """Save PruningRecords for viability-filtered elements to the DB."""
    try:
        from db.local_store import LocalStore
        from db.models import PruningRecord

        run_id: str = state.get("run_id", "")
        parent_formula: str = state.get("parent_formula", "")
        target_site: str = state.get("target_site_species", "")

        db_path = (
            state.get("config", {})
            .get("pipeline", {})
            .get("database", {})
            .get("local", {})
            .get("path", "data/results.db")
        )

        # Build records: passed elements carry all earlier stage data forward
        records: list[PruningRecord] = []
        for cand in passed:
            records.append(PruningRecord(
                run_id=run_id,
                parent_formula=parent_formula,
                target_site_species=target_site,
                element=cand["element"],
                stage1_passed=True,
                stage1_oxidation_state=cand.get("oxidation_state"),
                stage2_passed=True,
                stage2_mismatch_pct=cand.get("mismatch_pct"),
                stage3_passed=True,
                stage3_sub_probability=cand.get("sub_probability"),
                stage4_passed=True,
                stage4_viability_reason=None,
            ))
        for cand in rejected:
            records.append(PruningRecord(
                run_id=run_id,
                parent_formula=parent_formula,
                target_site_species=target_site,
                element=cand["element"],
                stage1_passed=True,
                stage1_oxidation_state=cand.get("oxidation_state"),
                stage2_passed=True,
                stage2_mismatch_pct=cand.get("mismatch_pct"),
                stage3_passed=True,
                stage3_sub_probability=cand.get("sub_probability"),
                stage4_passed=False,
                stage4_viability_reason=cand.get("viability_rejection_reason"),
            ))

        store = LocalStore(db_path)
        store.save_pruning_record(run_id, records)
        store.close()
        logger.debug(
            "stage4_viability: saved %d pruning records to %s.", len(records), db_path
        )
    except Exception as exc:
        logger.warning("stage4_viability: DB persistence failed: %s", exc)
