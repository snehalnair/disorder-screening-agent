"""
Cross-run comparison module.

Compares results from different pipeline runs. Used for:
- Comparing different concentrations of the same dopant
- Cross-validating against previous sessions
- Tracking property changes across MLIP versions
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_HIGHER_IS_BETTER = {"voltage", "li_ni_exchange"}


@dataclass
class ComparisonReport:
    """Output of cross-run comparison."""
    run_ids: list[str]
    dopants_compared: list[str]
    property_deltas: dict         # {dopant: {property: delta}}
    spearman_rho: dict            # {property: rho between run rankings}
    ranking_changes: list[str]    # Human-readable descriptions of ranking shifts
    summary: str


def compare_runs(
    run_ids: list[str],
    db,
    target_properties: Optional[list[str]] = None,
) -> ComparisonReport:
    """
    Compare results from two or more pipeline runs.

    Parameters
    ----------
    run_ids:
        List of run UUIDs to compare.
    db:
        LocalStore instance for fetching results.
    target_properties:
        Properties to compare. None = all available.

    Returns
    -------
    ComparisonReport
    """
    if not run_ids:
        return ComparisonReport(
            run_ids=run_ids,
            dopants_compared=[],
            property_deltas={},
            spearman_rho={},
            ranking_changes=[],
            summary="No run IDs provided.",
        )

    # ── Fetch results for each run ────────────────────────────────────
    run_results: dict[str, list] = {}
    for run_id in run_ids:
        run_results[run_id] = db.get_run_results(run_id)

    # ── Group each run's results by dopant (converged only) ───────────
    run_by_dopant: dict[str, dict[str, list]] = {}
    for run_id, results in run_results.items():
        by_dopant: dict[str, list] = defaultdict(list)
        for r in results:
            if r.relaxation_converged:
                by_dopant[r.dopant_element].append(r)
        run_by_dopant[run_id] = dict(by_dopant)

    # ── Infer properties if not specified ────────────────────────────
    if target_properties is None:
        target_properties = _infer_properties_from_runs(run_results)
    if not target_properties:
        target_properties = ["formation_energy"]

    # ── Compute per-dopant mean for each run ─────────────────────────
    run_means: dict[str, dict[str, dict[str, float]]] = {}
    for run_id, by_dopant in run_by_dopant.items():
        means: dict[str, dict[str, float]] = {}
        for dopant, results in by_dopant.items():
            means[dopant] = {}
            for prop in target_properties:
                vals = [_get_prop(r, prop) for r in results if _get_prop(r, prop) is not None]
                if vals:
                    means[dopant][prop] = float(np.mean(vals))
        run_means[run_id] = means

    # ── Find dopants common to all (or first two) runs ───────────────
    if len(run_ids) >= 2:
        all_sets = [set(run_by_dopant[rid].keys()) for rid in run_ids]
        common_dopants = sorted(set.intersection(*all_sets)) if all_sets else []
    else:
        common_dopants = sorted(run_by_dopant.get(run_ids[0], {}).keys())

    # ── Compute property deltas (run[1] - run[0]) ─────────────────────
    property_deltas: dict[str, dict[str, float]] = {}
    if len(run_ids) >= 2:
        means_0 = run_means.get(run_ids[0], {})
        means_1 = run_means.get(run_ids[1], {})
        for dopant in common_dopants:
            property_deltas[dopant] = {}
            for prop in target_properties:
                v0 = means_0.get(dopant, {}).get(prop)
                v1 = means_1.get(dopant, {}).get(prop)
                if v0 is not None and v1 is not None:
                    property_deltas[dopant][prop] = float(v1 - v0)

    # ── Rankings per run ──────────────────────────────────────────────
    run_rankings: dict[str, dict[str, list[str]]] = {}
    for run_id in run_ids:
        means = run_means.get(run_id, {})
        run_rankings[run_id] = {}
        for prop in target_properties:
            reverse = prop in _HIGHER_IS_BETTER
            eligible = [d for d in means if means[d].get(prop) is not None]
            ranked = sorted(
                eligible,
                key=lambda d: means[d][prop],
                reverse=reverse,
            )
            run_rankings[run_id][prop] = ranked

    # ── Spearman ρ between first two runs ────────────────────────────
    spearman_rho: dict[str, dict] = {}
    if len(run_ids) >= 2:
        r0_rankings = run_rankings.get(run_ids[0], {})
        r1_rankings = run_rankings.get(run_ids[1], {})
        for prop in target_properties:
            r0 = r0_rankings.get(prop, [])
            r1 = r1_rankings.get(prop, [])
            common = [d for d in r0 if d in r1]
            if len(common) >= 3:
                ranks0 = [r0.index(d) for d in common]
                ranks1 = [r1.index(d) for d in common]
                rho, pvalue = spearmanr(ranks0, ranks1)
                spearman_rho[prop] = {
                    "rho": float(rho),
                    "pvalue": float(pvalue),
                    "n": len(common),
                }

    # ── Ranking-change descriptions ────────────────────────────────────
    ranking_changes: list[str] = []
    if len(run_ids) >= 2:
        for prop in target_properties:
            r0 = run_rankings.get(run_ids[0], {}).get(prop, [])
            r1 = run_rankings.get(run_ids[1], {}).get(prop, [])
            for dopant in common_dopants:
                if dopant in r0 and dopant in r1:
                    pos0 = r0.index(dopant) + 1
                    pos1 = r1.index(dopant) + 1
                    if pos0 != pos1:
                        direction = "↑" if pos1 < pos0 else "↓"
                        ranking_changes.append(
                            f"{dopant} ({prop}): rank {pos0} → {pos1} {direction} "
                            f"({run_ids[0][:8]}… → {run_ids[1][:8]}…)"
                        )

    # ── Summary string ────────────────────────────────────────────────
    n_compared = len(common_dopants)
    rho_parts = []
    for prop, stats in spearman_rho.items():
        rho_parts.append(f"{prop}: ρ={stats['rho']:.3f}")

    summary_parts = [
        f"Compared {len(run_ids)} runs, {n_compared} common dopants, "
        f"{len(target_properties)} properties."
    ]
    if rho_parts:
        summary_parts.append("Spearman ρ: " + ", ".join(rho_parts))
    if ranking_changes:
        summary_parts.append(f"{len(ranking_changes)} ranking changes detected.")
    summary = " ".join(summary_parts)

    return ComparisonReport(
        run_ids=list(run_ids),
        dopants_compared=common_dopants,
        property_deltas=property_deltas,
        spearman_rho=spearman_rho,
        ranking_changes=ranking_changes,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_prop(result, prop: str):
    mapping = {
        "formation_energy": result.formation_energy_above_hull,
        "li_ni_exchange": result.li_ni_exchange_energy,
        "voltage": result.voltage,
        "volume_change": result.volume_change_pct,
    }
    return mapping.get(prop)


def _infer_properties_from_runs(run_results: dict) -> list[str]:
    all_results = [r for results in run_results.values() for r in results]
    prop_attrs = [
        ("formation_energy", "formation_energy_above_hull"),
        ("li_ni_exchange", "li_ni_exchange_energy"),
        ("voltage", "voltage"),
        ("volume_change", "volume_change_pct"),
    ]
    return [
        prop for prop, attr in prop_attrs
        if any(getattr(r, attr) is not None for r in all_results)
    ]
