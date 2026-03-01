"""
Stage 2: Shannon ionic radius screening.

Filters Stage 1 candidates by size compatibility with the host site using
Shannon ionic radii (ML-extended table from SMACT).

For each candidate (element, oxidation_state) the radius at the target
coordination number is looked up in data/shannon_radii.json.  Candidates
with no radius entry are silently dropped.  The relative mismatch

    mismatch_pct = |r_dopant - r_host| / r_host × 100

is computed and candidates exceeding ``mismatch_threshold`` (from
pipeline.yaml, default 15%) are pruned.

Input state keys:
    target_site_species (str)        — e.g. "Co"
    target_oxidation_state (int)     — e.g. 3
    target_coordination_number (int) — e.g. 6
    stage1_candidates (list[dict])   — output of Stage 1
    config (dict)                    — loaded from pipeline.yaml

Output state keys:
    stage2_candidates (list[dict])   — Stage 1 dict extended with:
                                        shannon_radius (float, Å)
                                        mismatch_pct   (float, %)
    execution_log (list[str])        — appended with one summary line
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

TOOL_METADATA = {
    "name": "radius_screen",
    "stage": 2,
    "description": (
        "Shannon ionic radius screening: filters candidates by relative size mismatch "
        "with the host site using the ML-extended Shannon radii table."
    ),
    "system_type": "periodic_crystal",
    "input_type": "list[CandidateDopant] + target site species/CN",
    "output_type": "list[CandidateDopant]",
    "cost": "milliseconds",
    "cost_per_candidate": "~1 ms",
    "typical_reduction": "271 element-OS pairs → 85 candidates (at 35% threshold)",
    "external_dependencies": ["data/shannon_radii.json"],
    "requires_structure": False,
    "requires_network": False,
    "requires_gpu": False,
    "configurable_params": ["mismatch_threshold", "coordination_number", "use_bartel_tolerance"],
    "failure_modes": ["element not in Shannon table (logged, candidate dropped)"],
}

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

_CN_TO_ROMAN: dict[int, str] = {
    1: "I",   2: "II",   3: "III",  4: "IV",   5: "V",
    6: "VI",  7: "VII",  8: "VIII", 9: "IX",  10: "X",
    11: "XI", 12: "XII", 14: "XIV",
}


def _load_shannon_radii() -> dict:
    path = _DATA_DIR / "shannon_radii.json"
    with path.open() as f:
        return json.load(f)


def _lookup_radius(
    radii: dict,
    element: str,
    ox_state: int,
    cn: int,
) -> float | None:
    """Return r_ionic (Å) for the given (element, oxidation_state, CN) or None."""
    cn_str = _CN_TO_ROMAN.get(cn, "VI")
    try:
        return radii[element][str(ox_state)][cn_str]["r_ionic"]
    except KeyError:
        return None


def run_stage2_radius(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: Shannon ionic radius filter."""
    target_species: str = state["target_site_species"]
    target_ox: int = state["target_oxidation_state"]
    target_cn: int = state.get("target_coordination_number") or 6
    stage1_candidates: list[dict] = state.get("stage1_candidates") or []

    cfg: dict = state.get("config") or {}
    stage_cfg: dict = (cfg.get("pipeline") or {}).get("stage2_radius") or {}
    mismatch_threshold: float = stage_cfg.get("mismatch_threshold", 0.15)

    radii = _load_shannon_radii()

    # Host radius — must exist or we cannot screen
    host_radius = _lookup_radius(radii, target_species, target_ox, target_cn)
    if host_radius is None:
        raise ValueError(
            f"Shannon radius not found for {target_species}^{target_ox}+ CN={target_cn}. "
            "Check data/shannon_radii.json or target_coordination_number."
        )

    passed: list[dict] = []
    n_no_radius = 0

    for cand in stage1_candidates:
        r = _lookup_radius(radii, cand["element"], cand["oxidation_state"], target_cn)
        if r is None:
            n_no_radius += 1
            continue

        mismatch = abs(r - host_radius) / host_radius
        if mismatch <= mismatch_threshold:
            passed.append(
                {
                    **cand,
                    "shannon_radius": r,
                    "mismatch_pct": round(mismatch * 100, 2),
                }
            )

    log_msg = (
        f"Stage 2 (Radius): {len(passed)} candidates "
        f"(host {target_species}^{target_ox}+ r={host_radius:.3f} Å, "
        f"threshold={mismatch_threshold * 100:.0f}%, "
        f"{n_no_radius} dropped — no radius data)"
    )
    logger.info(log_msg)

    return {
        "stage2_candidates": passed,
        "execution_log": [log_msg],
    }
