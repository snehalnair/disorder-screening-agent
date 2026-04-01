#!/usr/bin/env python3
"""
Partial delithiation — single-point version (fast).
Checks whether voltage RANKING at x=0.5 correlates with full delithiation ranking.
No relaxation after Li removal — compares single-point energy differences.
"""

import json
import pathlib
import time
import numpy as np
from scipy import stats

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LCO_DIR = PROJECT_DIR / "lco"
DATA_DIR = PROJECT_DIR / "data" / "structures"


def get_calc(device="cpu"):
    from mace.calculators import mace_mp
    print(f"Loading MACE-MP-0 on {device}...")
    calc = mace_mp(device=device, default_dtype="float64")
    print("Done.\n")
    return calc


def remove_fraction_li(atoms, fraction=0.5, seed=42):
    """Remove a fraction of Li atoms randomly."""
    rng = np.random.default_rng(seed)
    li_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
    n_remove = max(1, int(len(li_indices) * fraction))
    remove_idx = sorted(rng.choice(li_indices, size=n_remove, replace=False), reverse=True)
    new_atoms = atoms.copy()
    del new_atoms[remove_idx]
    return new_atoms, n_remove


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter

    calc = get_calc(args.device)
    adaptor = AseAtomsAdaptor()

    # Load parent and build supercell
    cif = DATA_DIR / "lco_parent.cif"
    parent = Structure.from_file(str(cif))
    parent.make_supercell([4, 4, 4])

    co_sites = [i for i, sp in enumerate(parent.species) if str(sp) == "Co"]
    print(f"Parent: {len(parent)} atoms, {len(co_sites)} Co sites")

    # Get all dopants from checkpoint directory
    dopants = []
    for f in sorted(LCO_DIR.glob("*.json")):
        dopants.append(f.stem.split("_")[-1])
    print(f"Dopants: {dopants} (n={len(dopants)})\n")

    results = {}
    for dop in dopants:
        t0 = time.time()

        # Load full-delithiation voltages from checkpoint
        with open(LCO_DIR / f"LiCoO2_layered_{dop}.json") as f:
            ckpt = json.load(f)
        v_full_ord = ckpt["ordered"]["voltage"]

        # Build ordered doped structure
        struct = parent.copy()
        struct.replace(co_sites[0], dop)
        atoms = adaptor.get_atoms(struct)
        atoms.calc = calc

        # Relax the lithiated doped structure
        try:
            ecf = FrechetCellFilter(atoms)
            opt = BFGS(ecf, logfile=None)
            opt.run(fmax=0.15, steps=300)
        except Exception:
            pass

        e_lith = atoms.get_potential_energy()

        # Partial delithiation: remove 50% Li, single-point energy (no relaxation)
        partial_voltages = []
        for seed in range(3):
            atoms_delith, n_removed = remove_fraction_li(atoms, 0.5, seed=seed)
            atoms_delith.calc = calc
            e_delith = atoms_delith.get_potential_energy()
            v = -(e_delith - e_lith) / n_removed
            partial_voltages.append(v)

        v_partial = np.mean(partial_voltages)
        v_partial_std = np.std(partial_voltages, ddof=1)
        dt = time.time() - t0

        results[dop] = {
            "v_full_ordered": v_full_ord,
            "v_partial_ordered": v_partial,
            "v_partial_std": v_partial_std,
        }
        print(f"  {dop:>3s}: full={v_full_ord:+.3f}  partial={v_partial:+.3f} ± {v_partial_std:.3f}  ({dt:.0f}s)")

    # Spearman ρ: full vs partial ordered rankings
    full_vals = [results[d]["v_full_ordered"] for d in results]
    partial_vals = [results[d]["v_partial_ordered"] for d in results]

    rho, p = stats.spearmanr(full_vals, partial_vals)
    print(f"\n{'='*60}")
    print(f"  Spearman ρ (full vs partial ordered voltage): {rho:+.3f} (p={p:.4f})")
    print(f"  n = {len(results)} dopants")
    print(f"{'='*60}")

    if rho > 0.7:
        print(f"\n  CONCLUSION: Full and partial delithiation rankings are strongly")
        print(f"  correlated. The ranking scrambling observed under disorder at")
        print(f"  full delithiation is expected to persist at x=0.5.")
    else:
        print(f"\n  CONCLUSION: Rankings differ between full and partial delithiation.")
        print(f"  Partial delithiation may show different disorder sensitivity.")

    output = {
        "fraction": 0.5,
        "n_samples": 3,
        "rho_full_vs_partial": rho,
        "p_value": p,
        "n_dopants": len(results),
        "dopants": results,
    }
    out_path = SCRIPT_DIR / "partial_delithiation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
