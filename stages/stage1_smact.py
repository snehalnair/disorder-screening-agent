"""
Stage 1: SMACT composition filter.

Screens all elements Z=1–103 as potential dopant candidates at a specified
substitution site by applying:

  1. Element exclusion list (noble gases, radioactive elements — from pipeline.yaml).
  2. Pauling electronegativity < O (3.44) — dopant must form cations in an oxide.
  3. Has at least one positive oxidation state (1–7) in the SMACT database.

Aliovalent flag is set when the candidate's oxidation state differs from the
target site's oxidation state.

Input state keys:
    target_site_species (str)      — e.g. "Co"
    target_oxidation_state (int)   — e.g. 3
    constraints (dict, optional)   — may contain "exclude_elements" list
    config (dict)                  — loaded from pipeline.yaml

Output state keys:
    stage1_candidates (list[dict]) — [{element, oxidation_state, is_aliovalent,
                                        pauling_eneg}, ...]
    execution_log (list[str])      — appended with one summary line
"""

from __future__ import annotations

import logging
from typing import Any

import smact

logger = logging.getLogger(__name__)

TOOL_METADATA = {
    "name": "smact_filter",
    "stage": 1,
    "description": (
        "SMACT composition filtering: Pauling electronegativity ordering, "
        "oxidation state plausibility (OS=1–7), element exclusion list."
    ),
    "system_type": "periodic_crystal",
    "input_type": "formula + target site",
    "output_type": "list[CandidateDopant]",
    "cost": "microseconds",
    "cost_per_candidate": "~10 µs",
    "typical_reduction": "103 elements → ~80 unique elements (271 element-OS pairs)",
    "external_dependencies": ["smact"],
    "requires_structure": False,
    "requires_network": False,
    "requires_gpu": False,
    "configurable_params": ["excluded_elements", "max_oxidation_state"],
    "failure_modes": ["element not in SMACT database (silently skipped)"],
}

# Pauling electronegativity of oxygen — dopant must be less electronegative
_O_PAULING_ENEG: float = 3.44

# Maximum sensible cation charge for TM-site substitution in oxides
_MAX_OXIDATION_STATE: int = 7


def run_stage1_smact(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: SMACT composition filter.

    Returns a partial state update with ``stage1_candidates`` and one appended
    ``execution_log`` entry.
    """
    target_ox: int = state["target_oxidation_state"]
    cfg: dict = state.get("config") or {}
    stage_cfg: dict = (cfg.get("pipeline") or {}).get("stage1_smact") or {}

    # Build the full exclusion set from config + optional user constraints
    exclude: set[str] = set(stage_cfg.get("exclude_elements") or [])
    user_constraints: dict = state.get("constraints") or {}
    exclude.update(user_constraints.get("exclude_elements") or [])

    el_dict: dict = smact.element_dictionary()  # {symbol: smact.Element}

    candidates: list[dict] = []

    for symbol, element in el_dict.items():
        if symbol in exclude:
            continue

        # ── Electronegativity gate ────────────────────────────────────────────
        eneg = getattr(element, "pauling_eneg", None)
        if eneg is None or eneg >= _O_PAULING_ENEG:
            continue

        # ── Oxidation state gate ──────────────────────────────────────────────
        ox_states = [
            ox
            for ox in (element.oxidation_states or [])
            if 1 <= ox <= _MAX_OXIDATION_STATE
        ]
        if not ox_states:
            continue

        for ox in ox_states:
            candidates.append(
                {
                    "element": symbol,
                    "oxidation_state": ox,
                    "is_aliovalent": ox != target_ox,
                    "pauling_eneg": eneg,
                }
            )

    # Deduplicate on (element, oxidation_state) — smact may list duplicates
    seen: set[tuple] = set()
    unique: list[dict] = []
    for c in candidates:
        key = (c["element"], c["oxidation_state"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    unique_elements: int = len({c["element"] for c in unique})
    n_os_combinations: int = len(unique)

    log_msg = (
        f"Stage 1 (SMACT): {unique_elements} unique elements "
        f"({n_os_combinations} element-OS combinations) "
        f"(screened {len(el_dict)} elements, excluded {len(exclude)})"
    )
    logger.info(log_msg)

    return {
        "stage1_candidates": unique,
        "stage1_unique_elements": unique_elements,
        "stage1_os_combinations": n_os_combinations,
        "execution_log": [log_msg],
    }
