#!/usr/bin/env python3
"""
SQS convergence analysis: how do rank correlations change with
the number of SQS realizations (3, 4, 5)?

Uses jackknife subsampling of existing 5-SQS data to estimate
convergence without new computation.
"""

import json
import pathlib
import itertools
import numpy as np
from scipy import stats

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
PROPERTIES = ["formation_energy", "voltage", "volume_change"]


def load_material(ckpt_dir):
    results = []
    if not ckpt_dir.exists():
        return results
    for f in sorted(ckpt_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        dopant = f.stem.split("_")[-1]
        ordered = data["ordered"]
        sqs = data.get("sqs_results", [])
        if len(sqs) < 3:
            continue
        results.append({"dopant": dopant, "ordered": ordered, "sqs_results": sqs})
    return results


def compute_rho_with_subset(results, prop, sqs_indices):
    """Compute Spearman ρ using only the specified SQS indices."""
    ord_vals, dis_vals = [], []
    for r in results:
        o = r["ordered"].get(prop)
        sqs = r["sqs_results"]
        # Take only the specified indices (skip if not enough SQS)
        subset = [sqs[i] for i in sqs_indices if i < len(sqs)]
        if len(subset) < 2:
            continue
        vals = [s.get(prop) for s in subset if prop in s]
        if not vals or o is None:
            continue
        d = np.mean(vals)
        if np.isnan(d):
            continue
        ord_vals.append(o)
        dis_vals.append(d)
    if len(ord_vals) < 4:
        return np.nan
    rho, _ = stats.spearmanr(ord_vals, dis_vals)
    return rho


def convergence_analysis(name, ckpt_dir):
    """Jackknife analysis: all C(5,k) subsets for k=3,4,5."""
    results = load_material(ckpt_dir)
    if not results:
        return

    # Check how many SQS each dopant has
    n_sqs = min(len(r["sqs_results"]) for r in results)
    print(f"\n{'='*60}")
    print(f"  {name}: {len(results)} dopants, {n_sqs} SQS each")
    print(f"{'='*60}")

    for prop in PROPERTIES:
        print(f"\n  {prop}:")
        # Full 5-SQS result
        rho_full = compute_rho_with_subset(results, prop, list(range(n_sqs)))
        print(f"    Full ({n_sqs} SQS): ρ = {rho_full:+.3f}")

        for k in [3, 4]:
            if k >= n_sqs:
                continue
            combos = list(itertools.combinations(range(n_sqs), k))
            rhos = [compute_rho_with_subset(results, prop, list(c)) for c in combos]
            rhos = [r for r in rhos if not np.isnan(r)]
            if rhos:
                mean_r = np.mean(rhos)
                std_r = np.std(rhos)
                min_r = np.min(rhos)
                max_r = np.max(rhos)
                print(f"    {k} SQS ({len(combos)} combos): "
                      f"ρ = {mean_r:+.3f} ± {std_r:.3f} "
                      f"[{min_r:+.3f}, {max_r:+.3f}]")

    # Leave-one-out jackknife for voltage specifically
    print(f"\n  Leave-one-out jackknife (voltage):")
    for i in range(n_sqs):
        idx = [j for j in range(n_sqs) if j != i]
        rho = compute_rho_with_subset(results, "voltage", idx)
        if not np.isnan(rho):
            print(f"    Drop SQS-{i}: ρ = {rho:+.3f}")


if __name__ == "__main__":
    materials = {
        "LiCoO₂ (layered)": PROJECT_DIR / "lco",
        "LiNiO₂ (layered)": PROJECT_DIR / "lno",
        "LiMn₂O₄ (spinel)": PROJECT_DIR / "lmo",
        "SrTiO₃ (perovskite)": PROJECT_DIR / "sto",
        "CeO₂ (fluorite)": PROJECT_DIR / "ceo2",
    }

    print("SQS Convergence Analysis")
    print("=" * 60)

    for name, path in materials.items():
        convergence_analysis(name, path)

    print("\n\nDone.")
