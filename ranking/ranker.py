"""
Dopant ranking module.

Groups SimulationResult objects by dopant, averages properties across SQS
realisations (mean ± std), computes disorder sensitivity, ranks dopants by
each property, computes Spearman ρ between ordered and disordered rankings,
and flags high-variance or sanity-check violations.

This is the module that produces the paper's main results table.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from db.models import SimulationResult

logger = logging.getLogger(__name__)


# ── Ordering conventions ──────────────────────────────────────────────────────

# For these properties, HIGHER is better → sort descending
_HIGHER_IS_BETTER = {"voltage", "li_ni_exchange"}

# Default sanity-check thresholds (used when no config is passed)
_DEFAULT_MAX_FORMATION_ENERGY = 0.10    # eV/atom
_DEFAULT_MAX_VOLUME_CHANGE = 15.0       # %
_DEFAULT_VARIANCE_THRESHOLD = 0.20      # 20% CV


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class DopantStats:
    """Aggregated statistics for one dopant across all SQS realisations."""
    dopant: str
    n_converged: int
    properties: dict          # {prop: {"mean", "std", "n", "values"}}
    ordered_properties: dict  # {prop: value} from ordered-cell computation
    disorder_sensitivity: dict  # {prop: relative_change}
    rank_by_property: dict    # {prop: 1-based rank}


@dataclass
class RankedReport:
    """Full ranking output — the paper's main results table."""
    parent_formula: str
    target_site: str
    candidates_simulated: int
    rankings: list[DopantStats]
    spearman_rho: dict         # {prop: {"rho", "pvalue", "n"}}
    recommended: list[str]     # Top-N dopants by primary property
    warnings: list[str]
    primary_property: str      # property used for top-N recommendation
    all_rankings: dict         # {prop: [dopant, ...]} sorted list per property


# ── Main entry point ──────────────────────────────────────────────────────────

def rank_dopants(
    simulation_results: list[SimulationResult],
    baseline: Optional[dict] = None,
    target_properties: Optional[list[str]] = None,
    ordered_results: Optional[dict] = None,
    variance_threshold: float = _DEFAULT_VARIANCE_THRESHOLD,
    sanity_config: Optional[dict] = None,
    top_n: int = 3,
) -> RankedReport:
    """
    Rank dopants by target properties.

    Parameters
    ----------
    simulation_results:
        All SimulationResult objects from Stage 5 (multiple dopants ×
        concentrations × SQS realisations).
    baseline:
        Undoped reference properties dict from compute_baseline.
    target_properties:
        Which properties to rank by. Defaults to all available properties
        found in the results.
    ordered_results:
        {dopant: {property: value}} from ordered-cell predictions.
        If provided, computes Spearman ρ and disorder sensitivity.
    variance_threshold:
        Flag dopants where σ > this fraction of |mean|. Default 0.20 (20%).
    sanity_config:
        Optional dict with keys ``max_formation_energy_above_hull`` (eV/atom)
        and ``max_volume_change`` (%).
    top_n:
        How many top dopants to include in the ``recommended`` list.

    Returns
    -------
    RankedReport
    """
    if not simulation_results:
        logger.warning("rank_dopants: empty simulation_results — returning empty report.")
        return RankedReport(
            parent_formula="",
            target_site="",
            candidates_simulated=0,
            rankings=[],
            spearman_rho={},
            recommended=[],
            warnings=["No simulation results provided."],
            primary_property="",
            all_rankings={},
        )

    if target_properties is None:
        target_properties = _infer_properties(simulation_results)

    if not target_properties:
        target_properties = ["formation_energy"]

    sanity = sanity_config or {}
    max_fe = sanity.get("max_formation_energy_above_hull", _DEFAULT_MAX_FORMATION_ENERGY)
    max_vc = sanity.get("max_volume_change", _DEFAULT_MAX_VOLUME_CHANGE)

    # ── Step 1: Group by dopant (only converged runs) ─────────────────
    grouped: dict[str, list[SimulationResult]] = defaultdict(list)
    n_aborted = 0
    for result in simulation_results:
        if result.relaxation_converged:
            grouped[result.dopant_element].append(result)
        else:
            n_aborted += 1

    if n_aborted:
        logger.info("rank_dopants: %d aborted relaxations excluded.", n_aborted)

    # ── Step 2: Aggregate per dopant ──────────────────────────────────
    dopant_summary: dict[str, dict] = {}
    for dopant, results in grouped.items():
        props: dict[str, list] = defaultdict(list)
        for r in results:
            for prop in target_properties:
                val = _get_prop(r, prop)
                if val is not None:
                    props[prop].append(val)

        dopant_summary[dopant] = {
            prop: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
                "values": list(vals),
            }
            for prop, vals in props.items()
            if vals
        }

    # ── Step 3: Disorder sensitivity ──────────────────────────────────
    disorder_sensitivity: dict[str, dict] = {}
    if ordered_results:
        for dopant, summary in dopant_summary.items():
            disorder_sensitivity[dopant] = {}
            for prop in target_properties:
                disordered_mean = summary.get(prop, {}).get("mean")
                ordered_val = (ordered_results.get(dopant) or {}).get(prop)
                if (
                    disordered_mean is not None
                    and ordered_val is not None
                    and ordered_val != 0
                ):
                    sensitivity = abs(disordered_mean - ordered_val) / abs(ordered_val)
                    disorder_sensitivity[dopant][prop] = float(sensitivity)

    # ── Step 4: Rank per property ─────────────────────────────────────
    all_rankings: dict[str, list[str]] = {}
    for prop in target_properties:
        reverse = prop in _HIGHER_IS_BETTER
        eligible = [
            d for d in dopant_summary
            if dopant_summary[d].get(prop, {}).get("mean") is not None
        ]
        ranked = sorted(
            eligible,
            key=lambda d: dopant_summary[d][prop]["mean"],
            reverse=reverse,
        )
        all_rankings[prop] = ranked

    # ── Step 5: Spearman ρ between ordered and disordered ─────────────
    spearman_rho: dict[str, dict] = {}
    if ordered_results:
        for prop in target_properties:
            ordered_vals = {}
            for d, ov in ordered_results.items():
                if isinstance(ov, dict) and ov.get(prop) is not None:
                    ordered_vals[d] = ov[prop]

            dis_ranking = all_rankings.get(prop, [])
            common = [d for d in dis_ranking if d in ordered_vals]

            if len(common) >= 3:
                reverse = prop in _HIGHER_IS_BETTER
                ord_sorted = sorted(common, key=lambda d: ordered_vals[d], reverse=reverse)
                ord_ranks = [ord_sorted.index(d) for d in common]
                dis_ranks = [dis_ranking.index(d) for d in common]
                rho, pvalue = spearmanr(ord_ranks, dis_ranks)
                spearman_rho[prop] = {
                    "rho": float(rho),
                    "pvalue": float(pvalue),
                    "n": len(common),
                }

    # ── Step 6: Warnings — high variance ─────────────────────────────
    warnings: list[str] = []
    if n_aborted:
        warnings.append(f"{n_aborted} relaxation(s) aborted and excluded from ranking.")

    for dopant, summary in dopant_summary.items():
        for prop, stats in summary.items():
            mean_abs = abs(stats["mean"])
            if mean_abs > 0 and stats["std"] / mean_abs > variance_threshold:
                warnings.append(
                    f"{dopant}: high SQS variance for {prop} "
                    f"(σ={stats['std']:.4f}, mean={stats['mean']:.4f}, "
                    f"CV={stats['std']/mean_abs:.0%})"
                )

    # ── Step 7: Sanity check warnings ────────────────────────────────
    for dopant, summary in dopant_summary.items():
        fe_stats = summary.get("formation_energy", {})
        fe = fe_stats.get("mean")
        if fe is not None and fe > max_fe:
            warnings.append(
                f"{dopant}: formation energy {fe:.3f} eV/atom > threshold {max_fe}"
            )

        vc_stats = summary.get("volume_change", {})
        vc = vc_stats.get("mean")
        if vc is not None and vc > max_vc:
            warnings.append(
                f"{dopant}: volume change {vc:.1f}% > threshold {max_vc}%"
            )

    # ── Step 8: Compile per-dopant DopantStats ────────────────────────
    primary_property = target_properties[0] if target_properties else ""
    primary_ranking = all_rankings.get(primary_property, list(dopant_summary.keys()))

    ranking_objects: list[DopantStats] = []
    for dopant in primary_ranking:
        if dopant not in dopant_summary:
            continue
        rank_by_property = {}
        for prop, ranked_list in all_rankings.items():
            if dopant in ranked_list:
                rank_by_property[prop] = ranked_list.index(dopant) + 1  # 1-based

        ranking_objects.append(
            DopantStats(
                dopant=dopant,
                n_converged=len(grouped.get(dopant, [])),
                properties=dopant_summary[dopant],
                ordered_properties=(ordered_results or {}).get(dopant) or {},
                disorder_sensitivity=disorder_sensitivity.get(dopant, {}),
                rank_by_property=rank_by_property,
            )
        )

    # Add dopants that have no results for primary property
    ranked_set = {ds.dopant for ds in ranking_objects}
    for dopant in dopant_summary:
        if dopant not in ranked_set:
            ranking_objects.append(
                DopantStats(
                    dopant=dopant,
                    n_converged=len(grouped.get(dopant, [])),
                    properties=dopant_summary[dopant],
                    ordered_properties=(ordered_results or {}).get(dopant) or {},
                    disorder_sensitivity=disorder_sensitivity.get(dopant, {}),
                    rank_by_property={
                        prop: all_rankings[prop].index(dopant) + 1
                        for prop in all_rankings
                        if dopant in all_rankings[prop]
                    },
                )
            )

    recommended = primary_ranking[:top_n]

    parent_formula = simulation_results[0].parent_formula if simulation_results else ""
    target_site = simulation_results[0].target_site_species if simulation_results else ""

    return RankedReport(
        parent_formula=parent_formula,
        target_site=target_site,
        candidates_simulated=len(grouped),
        rankings=ranking_objects,
        spearman_rho=spearman_rho,
        recommended=recommended,
        warnings=warnings,
        primary_property=primary_property,
        all_rankings=all_rankings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_prop(result: SimulationResult, prop: str) -> Optional[float]:
    """Extract a property value from a SimulationResult."""
    mapping = {
        "formation_energy": result.formation_energy_above_hull,
        "li_ni_exchange": result.li_ni_exchange_energy,
        "voltage": result.voltage,
        "volume_change": result.volume_change_pct,
    }
    if prop in mapping:
        return mapping[prop]
    # Fall through to direct attribute access
    val = getattr(result, prop, None)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _infer_properties(results: list[SimulationResult]) -> list[str]:
    """Determine which properties are available in the result set."""
    available = []
    prop_attrs = [
        ("formation_energy", "formation_energy_above_hull"),
        ("li_ni_exchange", "li_ni_exchange_energy"),
        ("voltage", "voltage"),
        ("volume_change", "volume_change_pct"),
    ]
    for prop_name, attr in prop_attrs:
        if any(getattr(r, attr) is not None for r in results):
            available.append(prop_name)
    return available
