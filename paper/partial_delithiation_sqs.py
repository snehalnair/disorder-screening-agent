#!/usr/bin/env python3
"""
Partial delithiation (x=0.5) with SQS ensembles for LiCoO₂.
=============================================================

Computes disordered voltage at x=0->0.5 for all LCO dopants using
5 SQS realisations per dopant. This tests whether the disorder gap
persists at partial state-of-charge (closer to real cycling conditions).

Per-dopant checkpointing: saves after each dopant, skips completed ones.

Usage (Colab):
    !pip install mace-torch ase pymatgen
    !python paper/partial_delithiation_sqs.py

Outputs:
    paper/partial_delithiation_sqs_results.json
"""

import json
import os
import pathlib
import random
import time

import numpy as np
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from scipy import stats

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LCO_DIR = PROJECT_DIR / "lco"
OUT_PATH = SCRIPT_DIR / "partial_delithiation_sqs_results.json"

DELITH_FRACTION = 0.5  # remove 50% of Li
N_LI_SEEDS = 3         # random Li-removal patterns per SQS
N_SQS = 5              # SQS realisations

# ── JSON encoder for numpy types ──
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.item()
        except ImportError:
            pass
        return super().default(obj)


def _save_checkpoint(results, path):
    """Save current results to JSON."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)


def load_checkpoint(path):
    """Load existing checkpoint if available."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_dopant_list():
    """Get list of dopants from LCO checkpoint directory."""
    dopants = []
    for d in sorted(LCO_DIR.iterdir()):
        if d.is_dir() and (d / "ordered_relaxed.json").exists():
            dopants.append(d.name)
    return dopants


def relax_structure(atoms, calc, fmax=0.05, steps=200):
    """Relax structure with FIRE optimizer."""
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    opt = FIRE(ecf, logfile=None)
    try:
        opt.run(fmax=fmax, steps=steps)
        return True
    except Exception as e:
        print(f"    Relaxation failed: {e}")
        return False


def remove_li_fraction(atoms, fraction, seed=42):
    """Remove a fraction of Li atoms randomly."""
    rng = random.Random(seed)
    li_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'Li']
    n_remove = max(1, int(len(li_indices) * fraction))
    to_remove = sorted(rng.sample(li_indices, n_remove), reverse=True)
    for idx in to_remove:
        del atoms[idx]
    return atoms, n_remove


def compute_partial_voltage(lith_atoms, calc, fraction=0.5, n_seeds=3):
    """
    Compute partial delithiation voltage for a given lithiated structure.

    V_partial = -(E_partial - E_lith) / (n_Li_removed * e)
    """
    from ase.io import read as ase_read
    import copy

    voltages = []

    for seed in range(n_seeds):
        delith = lith_atoms.copy()
        delith, n_removed = remove_li_fraction(delith, fraction, seed=seed + 42)

        converged = relax_structure(delith, calc)
        if not converged:
            continue

        e_delith = float(delith.get_potential_energy())
        e_lith = float(lith_atoms.get_potential_energy())
        v = -(e_delith - e_lith) / n_removed
        voltages.append(float(v))

    if not voltages:
        return None, None
    return float(np.mean(voltages)), float(np.std(voltages))


def main():
    print("=" * 60)
    print("PARTIAL DELITHIATION SQS STUDY")
    print(f"Material: LiCoO₂, x = 0 -> {DELITH_FRACTION}")
    print(f"SQS realisations: {N_SQS}, Li-removal seeds: {N_LI_SEEDS}")
    print("=" * 60)

    # Load MACE calculator
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    device = str(calc.device) if hasattr(calc, 'device') else "cpu"
    print(f"MACE device: {device}")

    dopants = get_dopant_list()
    print(f"Dopants found: {len(dopants)}")

    # Load or create checkpoint
    existing = load_checkpoint(OUT_PATH)
    if existing and "dopant_results" in existing:
        results = existing
        print(f"Resuming from checkpoint ({len(results['dopant_results'])} dopants done)")
    else:
        results = {
            "material": "LiCoO2",
            "delithiation_fraction": DELITH_FRACTION,
            "n_sqs": N_SQS,
            "n_li_seeds": N_LI_SEEDS,
            "device": device,
            "dopant_results": {},
        }

    from ase.io import read as ase_read

    for dopant in dopants:
        if dopant in results["dopant_results"]:
            print(f"\n[SKIP] {dopant} — already computed")
            continue

        print(f"\n{'='*40}")
        print(f"Dopant: {dopant}")
        print(f"{'='*40}")
        t0 = time.time()

        dopant_dir = LCO_DIR / dopant

        # --- Ordered partial delithiation ---
        ordered_path = dopant_dir / "ordered_relaxed.json"
        if not ordered_path.exists():
            print(f"  No ordered structure, skipping")
            continue

        ordered_lith = ase_read(str(ordered_path))
        ordered_lith.calc = calc

        # Get ordered lithiated energy
        e_lith_ordered = float(ordered_lith.get_potential_energy())

        # Partial delithiation of ordered structure
        v_ord_mean, v_ord_std = compute_partial_voltage(
            ordered_lith, calc, fraction=DELITH_FRACTION, n_seeds=N_LI_SEEDS
        )
        print(f"  Ordered V_partial: {v_ord_mean:.4f} +/- {v_ord_std:.4f} V")

        # --- SQS partial delithiation ---
        sqs_voltages = []
        for sqs_idx in range(1, N_SQS + 1):
            sqs_path = dopant_dir / f"sqs_{sqs_idx}_relaxed.json"
            if not sqs_path.exists():
                print(f"  SQS {sqs_idx} not found, skipping")
                continue

            sqs_lith = ase_read(str(sqs_path))
            sqs_lith.calc = calc

            v_sqs, v_sqs_std = compute_partial_voltage(
                sqs_lith, calc, fraction=DELITH_FRACTION, n_seeds=N_LI_SEEDS
            )
            if v_sqs is not None:
                sqs_voltages.append(float(v_sqs))
                print(f"  SQS {sqs_idx} V_partial: {v_sqs:.4f} V")

        if not sqs_voltages:
            print(f"  No SQS voltages computed, skipping")
            continue

        results["dopant_results"][dopant] = {
            "voltage_partial_ordered": float(v_ord_mean),
            "voltage_partial_ordered_std": float(v_ord_std) if v_ord_std else 0.0,
            "voltage_partial_disordered_mean": float(np.mean(sqs_voltages)),
            "voltage_partial_disordered_std": float(np.std(sqs_voltages)),
            "n_sqs_converged": len(sqs_voltages),
            "time_s": float(time.time() - t0),
        }

        _save_checkpoint(results, OUT_PATH)
        print(f"  Time: {time.time() - t0:.0f}s | Checkpoint saved")

    # --- Compute ranking correlations ---
    dopant_names = []
    v_ord_list = []
    v_dis_list = []

    for d, data in results["dopant_results"].items():
        if data["voltage_partial_ordered"] is not None and data["voltage_partial_disordered_mean"] is not None:
            dopant_names.append(d)
            v_ord_list.append(data["voltage_partial_ordered"])
            v_dis_list.append(data["voltage_partial_disordered_mean"])

    if len(v_ord_list) >= 4:
        rho, p = stats.spearmanr(v_ord_list, v_dis_list)
        results["voltage_partial_rho"] = float(rho)
        results["voltage_partial_p"] = float(p)
        results["n_dopants_ranked"] = len(v_ord_list)
        print(f"\n{'='*60}")
        print(f"PARTIAL DELITHIATION RANKING (x=0->{DELITH_FRACTION})")
        print(f"  Spearman ρ = {rho:.3f} (p = {p:.4f}, n = {len(v_ord_list)})")
        print(f"{'='*60}")
    else:
        print(f"\nNot enough dopants for correlation ({len(v_ord_list)})")

    _save_checkpoint(results, OUT_PATH)
    print(f"\nFinal results saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
