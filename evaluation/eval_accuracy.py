"""
RQ3: Accuracy of MACE predictions vs experiment.

Compares computed property values (ordered and disordered) against
published experimental measurements from data/experimental_measurements/.

Metrics:
- MAE(ordered vs experiment) per property
- MAE(disordered vs experiment) per property
- % MAE reduction: (MAE_ordered - MAE_disordered) / MAE_ordered * 100
- Spearman ρ between computed exchange energy ranking and experimental
  Li/Ni mixing ranking

Usage::

    # After running eval_disorder.py to get RQ2 results:
    python -m evaluation.eval_accuracy \\
        --results results/rq2_disorder.json \\
        --experimental data/experimental_measurements/nmc_dopants.json

    # Or specify both via config:
    python -m evaluation.eval_accuracy
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_EXP_PATH = (
    pathlib.Path(__file__).parent.parent
    / "data"
    / "experimental_measurements"
    / "nmc_dopants.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_experimental_data(path: str | pathlib.Path | None = None) -> dict:
    """Load experimental measurements JSON.

    Returns dict keyed by element symbol, each with property sub-dicts
    containing ``"value"`` and ``"source"`` keys.
    """
    p = pathlib.Path(path) if path else _DEFAULT_EXP_PATH
    with p.open() as f:
        raw = json.load(f)
    # Flatten: return dopants dict only
    return raw.get("dopants", raw)


# ─────────────────────────────────────────────────────────────────────────────
# Property name mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps internal property names → keys in the experimental JSON
_PROP_TO_EXP_KEY: dict[str, str] = {
    "voltage": "voltage_V",
    "li_ni_exchange": "li_ni_mixing_pct",
    "formation_energy": "formation_energy_ev_atom",
    "volume_change": "volume_change_pct",
}

# Units for display
_PROP_UNITS: dict[str, str] = {
    "voltage": "V",
    "li_ni_exchange": "%",
    "formation_energy": "eV/atom",
    "volume_change": "%",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core accuracy computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_accuracy_metrics(
    rq2_results: dict,
    experimental_data: dict,
) -> dict:
    """Compute MAE and Spearman ρ vs experiment for ordered and disordered predictions.

    Args:
        rq2_results:         Output of ``eval_disorder.run_disorder_evaluation()``.
        experimental_data:   Output of ``load_experimental_data()``.

    Returns:
        Dict with keys:
        - ``per_dopant``   : list of per-dopant comparison rows
        - ``mae_ordered``  : {property: float}
        - ``mae_disordered``: {property: float}
        - ``pct_reduction``: {property: float}  MAE reduction %
        - ``spearman_vs_exp``: {property: {"rho": float, "pvalue": float, "n": int}}
    """
    from scipy.stats import spearmanr

    target_properties = rq2_results.get("target_properties", [])
    dopant_rows = rq2_results.get("dopant_results", [])

    per_dopant = []
    # Collect errors per property
    errors_ordered: dict[str, list[float]] = {p: [] for p in target_properties}
    errors_disordered: dict[str, list[float]] = {p: [] for p in target_properties}

    for row in dopant_rows:
        dopant = row["dopant"]
        exp = experimental_data.get(dopant, {})
        if not exp:
            continue

        comparison = {"dopant": dopant, "properties": {}}

        for prop in target_properties:
            exp_key = _PROP_TO_EXP_KEY.get(prop)
            if not exp_key:
                continue
            exp_entry = exp.get(exp_key)
            if exp_entry is None:
                continue
            exp_val = exp_entry.get("value") if isinstance(exp_entry, dict) else exp_entry

            ord_v = row["ordered"].get(prop)
            dis_v = row["disordered_mean"].get(prop)

            err_ord = abs(ord_v - exp_val) if ord_v is not None and isinstance(ord_v, (int, float)) else None
            err_dis = abs(dis_v - exp_val) if dis_v is not None and isinstance(dis_v, (int, float)) else None

            if err_ord is not None:
                errors_ordered[prop].append(err_ord)
            if err_dis is not None:
                errors_disordered[prop].append(err_dis)

            comparison["properties"][prop] = {
                "experimental": exp_val,
                "ordered": ord_v,
                "disordered": dis_v,
                "error_ordered": err_ord,
                "error_disordered": err_dis,
                "units": _PROP_UNITS.get(prop, ""),
            }

        per_dopant.append(comparison)

    mae_ordered = {
        p: float(np.mean(v)) if v else None
        for p, v in errors_ordered.items()
    }
    mae_disordered = {
        p: float(np.mean(v)) if v else None
        for p, v in errors_disordered.items()
    }
    pct_reduction = {}
    for p in target_properties:
        mo = mae_ordered.get(p)
        md = mae_disordered.get(p)
        if mo is not None and md is not None and mo > 0:
            pct_reduction[p] = (mo - md) / mo * 100.0

    # Spearman ρ between computed exchange-energy ranking and experimental Li/Ni mixing
    spearman_vs_exp = {}
    for prop in target_properties:
        comp_vals = []
        exp_vals = []
        for row in dopant_rows:
            dopant = row["dopant"]
            exp = experimental_data.get(dopant, {})
            exp_key = _PROP_TO_EXP_KEY.get(prop)
            if not exp_key:
                continue
            exp_entry = exp.get(exp_key)
            if exp_entry is None:
                continue
            exp_val = exp_entry.get("value") if isinstance(exp_entry, dict) else exp_entry
            comp_val = row["disordered_mean"].get(prop)
            if comp_val is not None and exp_val is not None:
                comp_vals.append(comp_val)
                exp_vals.append(exp_val)
        if len(comp_vals) >= 3:
            rho, pvalue = spearmanr(comp_vals, exp_vals)
            spearman_vs_exp[prop] = {
                "rho": float(rho),
                "pvalue": float(pvalue),
                "n": len(comp_vals),
            }

    return {
        "per_dopant": per_dopant,
        "mae_ordered": mae_ordered,
        "mae_disordered": mae_disordered,
        "pct_reduction": pct_reduction,
        "spearman_vs_exp": spearman_vs_exp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing / formatting
# ─────────────────────────────────────────────────────────────────────────────


def print_table3(accuracy: dict) -> None:
    """Print Table 3: per-dopant comparison."""
    target_properties = list(
        {prop for row in accuracy["per_dopant"] for prop in row["properties"]}
    )

    print(f"\n{'Table 3: Computed vs Experimental Properties':=^100}")
    for prop in target_properties:
        units = _PROP_UNITS.get(prop, "")
        print(f"\n  Property: {prop} ({units})")
        header = f"  {'Dopant':<8} {'Experimental':>14} {'Ordered':>10} {'Err_ord':>9} {'Disordered':>12} {'Err_dis':>9}"
        print(header)
        print("  " + "-" * 70)
        for row in accuracy["per_dopant"]:
            p_data = row["properties"].get(prop)
            if not p_data:
                continue
            exp = f"{p_data['experimental']:.3f}" if p_data['experimental'] is not None else "N/A"
            ord_ = f"{p_data['ordered']:.3f}" if p_data['ordered'] is not None else "N/A"
            dis_ = f"{p_data['disordered']:.3f}" if p_data['disordered'] is not None else "N/A"
            eo = f"{p_data['error_ordered']:.3f}" if p_data['error_ordered'] is not None else "N/A"
            ed = f"{p_data['error_disordered']:.3f}" if p_data['error_disordered'] is not None else "N/A"
            print(f"  {row['dopant']:<8} {exp:>14} {ord_:>10} {eo:>9} {dis_:>12} {ed:>9}")


def print_mae_summary(accuracy: dict) -> None:
    """Print MAE summary: MAE(ordered) vs MAE(disordered) per property."""
    print(f"\n{'MAE Summary: Ordered vs Disordered':=^70}")
    header = f"{'Property':<16} {'MAE(ordered)':>14} {'MAE(disordered)':>16} {'% Reduction':>12}"
    print(header)
    print("-" * 62)
    for prop in accuracy["mae_ordered"]:
        mo = accuracy["mae_ordered"].get(prop)
        md = accuracy["mae_disordered"].get(prop)
        pr = accuracy["pct_reduction"].get(prop)
        mo_s = f"{mo:.4f}" if mo is not None else "N/A"
        md_s = f"{md:.4f}" if md is not None else "N/A"
        pr_s = f"{pr:+.1f}%" if pr is not None else "N/A"
        units = _PROP_UNITS.get(prop, "")
        print(f"{prop:<16} {mo_s:>14} ({units}) {md_s:>16} ({units}) {pr_s:>12}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="RQ3: Accuracy vs experiment.")
    parser.add_argument(
        "--results", metavar="FILE",
        help="RQ2 results JSON from eval_disorder.py.",
    )
    parser.add_argument(
        "--experimental", metavar="FILE",
        default=str(_DEFAULT_EXP_PATH),
        help="Experimental measurements JSON [default: nmc_dopants.json].",
    )
    parser.add_argument("--save", metavar="FILE", help="Save accuracy results JSON here.")
    args = parser.parse_args()

    if not args.results:
        print("Error: --results (RQ2 output JSON) is required.", file=sys.stderr)
        sys.exit(1)

    with open(args.results) as f:
        rq2 = json.load(f)

    exp_data = load_experimental_data(args.experimental)
    accuracy = compute_accuracy_metrics(rq2, exp_data)

    print_table3(accuracy)
    print_mae_summary(accuracy)

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(accuracy, f, indent=2, default=str)
        print(f"\nAccuracy results saved to {args.save}")
