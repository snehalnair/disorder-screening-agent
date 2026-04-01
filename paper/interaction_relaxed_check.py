#!/usr/bin/env python3
"""
Relaxed vs unrelaxed interaction energy spot-check.

Computes E_int for Al in LiCoO₂ at 2-3 key distances WITH atomic
relaxation (fixed cell), comparing against the unrelaxed single-point
values to verify that the attractive NN sign is robust to relaxation.

Usage:
    python paper/interaction_relaxed_check.py --device cpu
"""

import argparse
import json
import pathlib
import time
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"


def get_calc(device="cpu"):
    from mace.calculators import mace_mp
    print(f"Loading MACE-MP-0 on {device}...")
    calc = mace_mp(device=device, default_dtype="float64")
    print("Done.")
    return calc


def relax_positions(atoms, calc, fmax=0.10, max_steps=200):
    """Relax atomic positions only (fixed cell)."""
    from ase.optimize import BFGS, FIRE
    atoms = atoms.copy()
    atoms.calc = calc
    try:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        return atoms, atoms.get_potential_energy(), opt.nsteps < max_steps
    except Exception:
        atoms2 = atoms.copy()
        atoms2.calc = calc
        opt = FIRE(atoms2, logfile=None)
        opt.run(fmax=fmax * 1.5, steps=max_steps)
        return atoms2, atoms2.get_potential_energy(), opt.nsteps < max_steps


def single_point(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy()


def compute_eint(struct_undoped, target_species, dopant, site_i, site_j, calc, relax=False):
    """Compute E_int = E(AB) - E(A) - E(B) + E(0)."""
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()

    fn = relax_positions if relax else lambda a, c, **kw: (a, single_point(a, c), True)

    s0 = struct_undoped.copy()
    _, E0, _ = fn(adaptor.get_atoms(s0), calc)

    sA = struct_undoped.copy()
    sA.replace(site_i, dopant)
    _, EA, _ = fn(adaptor.get_atoms(sA), calc)

    sB = struct_undoped.copy()
    sB.replace(site_j, dopant)
    _, EB, _ = fn(adaptor.get_atoms(sB), calc)

    sAB = struct_undoped.copy()
    sAB.replace(site_i, dopant)
    sAB.replace(site_j, dopant)
    _, EAB, _ = fn(adaptor.get_atoms(sAB), calc)

    E_int = EAB - EA - EB + E0
    return E_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from pymatgen.core import Structure

    calc = get_calc(args.device)

    # Load LCO
    cif = DATA_DIR / "lco_parent.cif"
    struct = Structure.from_file(str(cif))
    struct.make_supercell([3, 3, 2])  # 72 atoms, same as original scan

    co_sites = [i for i, sp in enumerate(struct.species) if str(sp) == "Co"]
    print(f"LiCoO₂: {len(struct)} atoms, {len(co_sites)} Co sites")

    # Pick 3 representative pairs: NN (~2.9Å), medium (~5.0Å), far (~14Å)
    # Using same sites as the original scan
    test_pairs = [
        (18, 20, "NN (~2.9 Å)"),
        (18, 28, "medium (~5.0 Å)"),
        (18, 19, "far (~14.2 Å)"),
    ]

    results = []
    print(f"\n{'Distance':<20} {'Unrelaxed (meV)':<18} {'Relaxed (meV)':<18} {'Δ (meV)':<12}")
    print("-" * 68)

    for si, sj, label in test_pairs:
        t0 = time.time()
        e_unrelax = compute_eint(struct, "Co", "Al", si, sj, calc, relax=False)
        t1 = time.time()
        e_relax = compute_eint(struct, "Co", "Al", si, sj, calc, relax=True)
        t2 = time.time()

        e_unrelax_meV = e_unrelax * 1000
        e_relax_meV = e_relax * 1000
        delta = e_relax_meV - e_unrelax_meV

        print(f"{label:<20} {e_unrelax_meV:>+12.1f}      {e_relax_meV:>+12.1f}      {delta:>+8.1f}")

        results.append({
            "label": label,
            "sites": [si, sj],
            "E_int_unrelaxed_meV": round(e_unrelax_meV, 1),
            "E_int_relaxed_meV": round(e_relax_meV, 1),
            "delta_meV": round(delta, 1),
            "time_unrelaxed_s": round(t1 - t0, 1),
            "time_relaxed_s": round(t2 - t1, 1),
        })

    # Summary
    print(f"\nKey question: does relaxation change the NN interaction sign?")
    nn = results[0]
    if nn["E_int_unrelaxed_meV"] < 0 and nn["E_int_relaxed_meV"] < 0:
        print(f"  NN interaction is ATTRACTIVE in both cases → sign is ROBUST")
    elif nn["E_int_unrelaxed_meV"] > 0 and nn["E_int_relaxed_meV"] > 0:
        print(f"  NN interaction is REPULSIVE in both cases → sign is ROBUST")
    else:
        print(f"  NN interaction CHANGES SIGN with relaxation → CAUTION")

    out = SCRIPT_DIR / "interaction_relaxed_check.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
