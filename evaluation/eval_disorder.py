"""
RQ2: Does disorder change dopant rankings?

Runs Stage 5 on a set of known NMC dopants with both ordered and
disordered (SQS) cells, computes Spearman ρ between the two rankings,
and identifies which dopants move the most between ordered and disordered.

Usage::

    # Run with real MACE (takes ~4 hours on M1 Max):
    python -m evaluation.eval_disorder --save results/rq2_disorder.json

    # Load pre-computed results and print tables:
    python -m evaluation.eval_disorder --results results/rq2_disorder.json

Results are saved as JSON for subsequent figure generation.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Default dopant set: 8 confirmed successful with experimental data (validation set)
_DEFAULT_DOPANTS = ["Al", "Ti", "Mg", "Ga", "Fe", "Zr", "Nb", "W"]

# Full Stage 3 survivor set — all 28 unique elements (Co self-substitution excluded).
# Best OS per element by Hautier-Ceder substitution probability:
#   Mn+3, Ni+3, Ru+3 > Fe+3, Al+3 > Cr+3, Nb+4 > ... (see paper_draft_notes.md)
# Use for RQ2 at n=28; no experimental data needed for computational comparison.
_ALL_STAGE3_DOPANTS = [
    # Known 8 (validation set)
    "Al", "Ti", "Mg", "Ga", "Fe", "Zr", "Nb", "W",
    # Novel 20 (synthesis targets — ranked by disorder-aware simulation)
    "Mn", "Ni", "Cr", "V",           # isovalent/mild aliovalent, low mismatch
    "Ge", "Sn", "Sb", "Ta",          # aliovalent, moderate mismatch
    "Se", "As",                       # chalcogen/metalloid
    "Ru", "Rh", "Ir", "Mo",          # platinum-group / refractory
    "Os", "Re", "Pt",                 # unusual OS, high-valence
    "Cu",                             # 33.9% mismatch, borderline
    "S",                              # non-metal at Co site — flag in paper
    "U",                              # radioactive — flag in paper
]

_DEFAULT_CONCENTRATION = 0.10  # 10% on Co site
_DEFAULT_N_SQS = 8  # increased from 5 for better statistics (buffer for convergence failures)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────


def run_disorder_evaluation(
    parent_structure,
    dopants: list[str] = _DEFAULT_DOPANTS,
    target_species: str = "Co",
    concentration: float = _DEFAULT_CONCENTRATION,
    n_sqs: int = _DEFAULT_N_SQS,
    config_path=None,
    save_path: Optional[str] = None,
) -> dict:
    """Run ordered + disordered Stage 5 for each dopant and compute Spearman ρ.

    For each dopant:
    - Ordered: compute_ordered_properties() at given concentration
    - Disordered: generate n_sqs SQS structures, relax each, compute properties

    Args:
        parent_structure: Pymatgen Structure of undoped NMC.
        dopants:          List of dopant element symbols.
        target_species:   Site being substituted (default "Co").
        concentration:    Dopant site fraction (default 0.10).
        n_sqs:            Number of SQS realisations per dopant.
        config_path:      Path to pipeline.yaml.
        save_path:        If given, save results JSON here.

    Returns:
        Results dict (see ``_build_results_dict``).
    """
    import yaml

    if config_path is None:
        config_path = pathlib.Path(__file__).parent.parent / "config" / "pipeline.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sim_cfg = config.get("pipeline", {}).get("stage5_simulation", {})
    mlip_name = sim_cfg.get("potential", "mace-mp-0")
    device = sim_cfg.get("device", "auto")
    supercell = sim_cfg.get("supercell", [2, 2, 2])
    fmax = sim_cfg.get("fmax", 0.05)
    fmax_sqs = sim_cfg.get("fmax_sqs", fmax)  # looser convergence for SQS (default: same as ordered)
    max_steps = sim_cfg.get("max_relax_steps", 500)
    target_properties = list(config.get("pipeline", {}).get("property_weights", {}).keys())

    from stages.stage5.calculators import get_calculator
    from stages.stage5.mlip_relaxation import relax_structure
    from stages.stage5.property_calculator import compute_ordered_properties, compute_properties
    from stages.stage5.sqs_generator import generate_sqs

    calculator = get_calculator(mlip_name, device=device)
    dopant_results = []

    for dopant in dopants:
        logger.info("Processing dopant %s …", dopant)

        # Ordered properties
        try:
            ordered_props = compute_ordered_properties(
                parent_structure=parent_structure,
                dopant_element=dopant,
                target_species=target_species,
                concentration=concentration,
                supercell_matrix=supercell,
                calculator=calculator,
                target_properties=target_properties,
            )
        except Exception as exc:
            logger.warning("Ordered properties failed for %s: %s", dopant, exc)
            ordered_props = {}

        # Disordered (SQS) properties
        sqs_props_list = []
        try:
            sqs_structures = generate_sqs(
                parent_structure=parent_structure,
                dopant_element=dopant,
                target_species=target_species,
                concentration=concentration,
                supercell_matrix=supercell,
                n_realisations=n_sqs,
            )
        except Exception as exc:
            logger.warning("SQS generation failed for %s: %s", dopant, exc)
            sqs_structures = []

        for sqs_idx, sqs in enumerate(sqs_structures):
            # Three-stage retry: BFGS → FIRE → FIRE with loose fmax
            # Uses fmax_sqs (default 0.15) for SQS — looser than ordered (0.10)
            # Rankings are robust to small force residuals since all dopants treated identically
            rr = None
            optimizer_used = "BFGS"
            fmax_used = fmax_sqs

            # Stage 1: BFGS (fast when it works)
            rr = relax_structure(
                sqs, calculator,
                fmax=fmax_sqs, max_steps=max_steps,
            )

            if not rr.relaxation_converged:
                # Stage 2: FIRE (robust for difficult energy landscapes)
                logger.info(
                    "Retrying %s SQS-%d with FIRE optimizer (stage 2)…",
                    dopant, sqs_idx,
                )
                optimizer_used = "FIRE"
                rr = relax_structure(
                    sqs, calculator,
                    fmax=fmax_sqs, max_steps=2000,
                    optimizer_name="FIRE",
                )

            if not rr.relaxation_converged:
                # Stage 3: FIRE with further loosened convergence
                logger.info(
                    "Retrying %s SQS-%d with FIRE + loose fmax=0.25 (stage 3)…",
                    dopant, sqs_idx,
                )
                fmax_used = 0.25
                rr = relax_structure(
                    sqs, calculator,
                    fmax=0.25, max_steps=2000,
                    optimizer_name="FIRE",
                )

            if rr.relaxation_converged:
                props = compute_properties(
                    relaxed_structure=rr.relaxed_structure,
                    parent_structure=parent_structure,
                    calculator=calculator,
                    target_properties=target_properties,
                    final_energy_per_atom=rr.final_energy_per_atom,
                )
                sqs_entry = {
                    k: v for k, v in props.items() if isinstance(v, (int, float))
                }
                # Convergence metadata
                sqs_entry["_convergence"] = {
                    "converged": True,
                    "optimizer_used": optimizer_used,
                    "fmax_used": fmax_used,
                    "relaxation_steps": rr.relaxation_steps,
                    "max_force_final": float(rr.max_force_final)
                        if rr.max_force_final is not None else None,
                }
                sqs_props_list.append(sqs_entry)
            else:
                logger.warning(
                    "SQS relaxation failed for %s SQS-%d after 3 retry stages: %s",
                    dopant, sqs_idx, rr.abort_reason,
                )

        # Aggregate disordered results
        disordered_mean = {}
        disordered_std = {}
        disordered_n = {}
        for prop in target_properties:
            vals = [r[prop] for r in sqs_props_list if prop in r and r[prop] is not None]
            if vals:
                disordered_mean[prop] = float(np.mean(vals))
                disordered_std[prop] = float(np.std(vals)) if len(vals) > 1 else 0.0
                disordered_n[prop] = len(vals)

        # Disorder sensitivity
        disorder_sensitivity = {}
        for prop in target_properties:
            ord_val = ordered_props.get(prop)
            dis_val = disordered_mean.get(prop)
            if (
                ord_val is not None and dis_val is not None
                and isinstance(ord_val, (int, float))
                and ord_val != 0
            ):
                disorder_sensitivity[prop] = abs(dis_val - ord_val) / abs(ord_val)

        dopant_results.append({
            "dopant": dopant,
            "ordered": {k: v for k, v in ordered_props.items() if isinstance(v, (int, float))},
            "disordered_mean": disordered_mean,
            "disordered_std": disordered_std,
            "disordered_n": disordered_n,
            "n_converged": len(sqs_props_list),
            "disorder_sensitivity": disorder_sensitivity,
            "sqs_realisations": sqs_props_list,
        })

    results = _build_results_dict(
        dopant_results=dopant_results,
        target_properties=target_properties,
        concentration=concentration,
        mlip_name=mlip_name,
        n_sqs=n_sqs,
    )

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("RQ2 results saved to %s", save_path)

    return results


def _build_results_dict(
    dopant_results: list[dict],
    target_properties: list[str],
    concentration: float,
    mlip_name: str,
    n_sqs: int,
) -> dict:
    """Compute Spearman ρ and assemble the full results dict."""
    from scipy.stats import spearmanr

    spearman_rho = {}
    for prop in target_properties:
        ordered_vals = []
        disordered_vals = []
        valid_dopants = []
        for row in dopant_results:
            ord_v = row["ordered"].get(prop)
            dis_v = row["disordered_mean"].get(prop)
            if ord_v is not None and dis_v is not None:
                ordered_vals.append(ord_v)
                disordered_vals.append(dis_v)
                valid_dopants.append(row["dopant"])
        if len(ordered_vals) >= 3:
            rho, pvalue = spearmanr(ordered_vals, disordered_vals)
            spearman_rho[prop] = {
                "rho": float(rho),
                "pvalue": float(pvalue),
                "n": len(ordered_vals),
                "dopants": valid_dopants,
                "interpretation": _interpret_rho(rho),
            }

    return {
        "concentration": concentration,
        "mlip": mlip_name,
        "n_sqs_realisations": n_sqs,
        "target_properties": target_properties,
        "dopant_results": dopant_results,
        "spearman_rho": spearman_rho,
    }


def _interpret_rho(rho: float) -> str:
    """Return a text interpretation of Spearman ρ."""
    if rho >= 0.9:
        return "Very high correlation — disorder does not change ranking"
    elif rho >= 0.8:
        return "High correlation — disorder has minor effect on ranking"
    elif rho >= 0.6:
        return "Moderate correlation — disorder meaningfully changes ranking"
    else:
        return "Low correlation — disorder strongly changes ranking"


# ─────────────────────────────────────────────────────────────────────────────
# Formatting / printing
# ─────────────────────────────────────────────────────────────────────────────


def print_table1(results: dict) -> None:
    """Print Table 1: dopant | property | ordered | disordered mean±std | sensitivity."""
    target_properties = results.get("target_properties", [])
    rows = results.get("dopant_results", [])

    print(f"\n{'Table 1: Ordered vs Disordered Property Predictions':=^90}")
    header = (
        f"{'Dopant':<7} {'Property':<16} {'Ordered':>10} {'Disordered (mean±std)':>22} "
        f"{'Sensitivity':>12} {'n':>4}"
    )
    print(header)
    print("-" * 80)
    for row in rows:
        dopant = row["dopant"]
        for prop in target_properties:
            ord_v = row["ordered"].get(prop)
            dis_m = row["disordered_mean"].get(prop)
            dis_s = row["disordered_std"].get(prop, 0.0)
            sens = row["disorder_sensitivity"].get(prop)
            n = row["disordered_n"].get(prop, 0)

            ord_str = f"{ord_v:.3f}" if ord_v is not None else "N/A"
            dis_str = f"{dis_m:.3f}±{dis_s:.3f}" if dis_m is not None else "N/A"
            sens_str = f"{sens:.1%}" if sens is not None else "N/A"
            print(f"{dopant:<7} {prop:<16} {ord_str:>10} {dis_str:>22} {sens_str:>12} {n:>4}")


def print_table2(results: dict) -> None:
    """Print Table 2: Spearman ρ per property with p-value and interpretation."""
    rho_data = results.get("spearman_rho", {})

    print(f"\n{'Table 2: Spearman ρ (Ordered vs Disordered Rankings)':=^90}")
    header = f"{'Property':<16} {'ρ':>8} {'p-value':>10} {'n':>4}  Interpretation"
    print(header)
    print("-" * 80)
    for prop, data in rho_data.items():
        print(
            f"{prop:<16} {data['rho']:>8.3f} {data['pvalue']:>10.3f} {data['n']:>4}  "
            f"{data['interpretation']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="RQ2: Disorder effect on dopant rankings.")
    parser.add_argument("--results", metavar="FILE", help="Load pre-computed results JSON.")
    parser.add_argument("--save", metavar="FILE", help="Save results JSON after running MACE.")
    parser.add_argument("--dopants", nargs="+", default=_DEFAULT_DOPANTS, metavar="EL")
    parser.add_argument("--target-species", default="Co", help="Host element being substituted (default: Co).")
    parser.add_argument("--conc", type=float, default=_DEFAULT_CONCENTRATION)
    parser.add_argument("--n-sqs", type=int, default=_DEFAULT_N_SQS)
    parser.add_argument("--structure", metavar="FILE", help="Path to parent structure CIF/POSCAR.")
    parser.add_argument("--config", metavar="FILE")
    args = parser.parse_args()

    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        print(f"Loaded pre-computed results from {args.results}")
    else:
        if not args.structure:
            print("Error: --structure is required when running MACE.", file=sys.stderr)
            sys.exit(1)
        from pymatgen.core import Structure
        parent_struct = Structure.from_file(args.structure)
        results = run_disorder_evaluation(
            parent_structure=parent_struct,
            dopants=args.dopants,
            target_species=args.target_species,
            concentration=args.conc,
            n_sqs=args.n_sqs,
            config_path=args.config,
            save_path=args.save,
        )

    print_table1(results)
    print_table2(results)
