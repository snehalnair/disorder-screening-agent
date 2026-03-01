"""
Stage 3: Hautier-Ceder substitution probability filter.

Uses pymatgen's SubstitutionProbability (Hautier et al. 2011) to score the
chemical likelihood that each Stage 2 candidate can replace the target species.
The score is the conditional probability P(dopant_species | target_species)
derived from co-occurrence patterns in the ICSD.

Candidates below ``probability_threshold`` (default 0.001 from pipeline.yaml)
are pruned.  Results are sorted by probability descending.

Reference:
    Hautier, G. et al. "Data Mined Ionic Substitutions for the Discovery of
    New Compounds." Inorg. Chem. 50, 656–663 (2011).

Input state keys:
    target_site_species (str)        — e.g. "Co"
    target_oxidation_state (int)     — e.g. 3
    stage2_candidates (list[dict])   — output of Stage 2
    config (dict)                    — loaded from pipeline.yaml

Output state keys:
    stage3_candidates (list[dict])   — Stage 2 dict extended with:
                                        sub_probability (float)
    execution_log (list[str])        — appended with one summary line
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TOOL_METADATA = {
    "name": "substitution_probability",
    "stage": 3,
    "description": (
        "Hautier-Ceder substitution probability: filters by conditional probability "
        "P(dopant | target) from the ICSD ionic substitution database via pymatgen."
    ),
    "system_type": "periodic_crystal",
    "input_type": "list[CandidateDopant] + target species",
    "output_type": "list[CandidateDopant] sorted by probability",
    "cost": "seconds",
    "cost_per_candidate": "~0.5 s",
    "typical_reduction": "85 → 46 candidates (at threshold=0.001)",
    "external_dependencies": ["pymatgen SubstitutionProbability (ICSD lambda table)"],
    "requires_structure": False,
    "requires_network": False,
    "requires_gpu": False,
    "configurable_params": ["probability_threshold"],
    "failure_modes": ["ion pair not in ICSD training data → probability=0 → filtered"],
}


def run_stage3_substitution(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: Hautier-Ceder substitution probability filter."""
    from pymatgen.analysis.structure_prediction.substitution_probability import (
        SubstitutionProbability,
    )
    from pymatgen.core import Species

    target_species_str: str = state["target_site_species"]
    target_ox: int = state["target_oxidation_state"]
    stage2_candidates: list[dict] = state.get("stage2_candidates") or []

    cfg: dict = state.get("config") or {}
    stage_cfg: dict = (cfg.get("pipeline") or {}).get("stage3_substitution") or {}
    prob_threshold: float = stage_cfg.get("probability_threshold", 0.001)

    target_sp = Species(target_species_str, target_ox)
    sub_prob = SubstitutionProbability()

    passed: list[dict] = []
    n_zero = 0

    for cand in stage2_candidates:
        try:
            dopant_sp = Species(cand["element"], cand["oxidation_state"])
            prob: float = sub_prob.cond_prob(target_sp, dopant_sp)
        except Exception as exc:
            logger.debug(
                "  %s^%d+: probability lookup failed (%s)",
                cand["element"],
                cand["oxidation_state"],
                exc,
            )
            prob = 0.0

        if prob < prob_threshold:
            n_zero += 1
            continue

        passed.append({**cand, "sub_probability": round(prob, 6)})

    # Sort by probability descending so highest-confidence candidates come first
    passed.sort(key=lambda x: x["sub_probability"], reverse=True)

    log_msg = (
        f"Stage 3 (Substitution): {len(passed)} candidates "
        f"(threshold={prob_threshold}, pruned {n_zero} below threshold, "
        f"from {len(stage2_candidates)} input)"
    )
    logger.info(log_msg)

    return {
        "stage3_candidates": passed,
        "execution_log": [log_msg],
    }
