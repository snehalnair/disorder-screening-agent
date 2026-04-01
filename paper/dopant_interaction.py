#!/usr/bin/env python3
"""
Dopant-Dopant Interaction Energy vs Distance
=============================================
Computes E_interaction(r) = E(2 dopants at distance r) - 2*E(1 dopant) + E(undoped)
for layered LiCoO₂ and spinel LiMn₂O₄.

This reveals whether dopant-dopant interactions are long-ranged (layered)
or screened (spinel), explaining why disorder destroys voltage rankings
in layered but not spinel structures.

Usage:
    python dopant_interaction.py [--device cpu|cuda|mps] [--dopant Al]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# ASE imports
from ase.optimize import BFGS, FIRE
from ase.filters import FrechetCellFilter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"


def get_mace_calculator(device="cpu"):
    """Load MACE-MP-0 calculator (cached)."""
    from mace.calculators import mace_mp
    return mace_mp(default_dtype="float64", device=device)


def load_and_expand(cif_path, supercell_matrix):
    """Load CIF and create supercell."""
    struct = Structure.from_file(str(cif_path))
    struct.make_supercell(supercell_matrix)
    return struct


def get_target_site_indices(struct, target_species):
    """Get indices of all sites with target species."""
    return [i for i, site in enumerate(struct) if site.specie.symbol == target_species]


def compute_pair_distances(struct, indices):
    """Compute all pairwise distances between target sites, respecting PBC."""
    pairs = []
    for i, j in combinations(indices, 2):
        dist = struct.get_distance(i, j)
        pairs.append((i, j, dist))
    return sorted(pairs, key=lambda x: x[2])


def relax_structure(atoms, calc, fmax=0.15, max_steps=200):
    """Quick position-only relaxation (no cell relaxation for speed)."""
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    try:
        opt.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        print(f"    Relaxation failed: {e}", end="")
    return atoms, True


def compute_interaction_energy(struct_undoped, target_species, dopant, site_i, site_j, calc, relax=True):
    """
    Compute dopant-dopant interaction energy:
    E_int = E(AB) - E(A) - E(B) + E(0)

    where:
      E(0)  = energy of undoped supercell
      E(A)  = energy with dopant at site i only
      E(B)  = energy with dopant at site j only
      E(AB) = energy with dopant at both sites i and j
    """
    adaptor = AseAtomsAdaptor()

    # E(0): undoped
    s0 = struct_undoped.copy()
    a0 = adaptor.get_atoms(s0)
    if relax:
        a0, _ = relax_structure(a0, calc)
    else:
        a0.calc = calc
    E0 = a0.get_potential_energy()

    # E(A): dopant at site i
    sA = struct_undoped.copy()
    sA.replace(site_i, dopant)
    aA = adaptor.get_atoms(sA)
    if relax:
        aA, _ = relax_structure(aA, calc)
    else:
        aA.calc = calc
    EA = aA.get_potential_energy()

    # E(B): dopant at site j
    sB = struct_undoped.copy()
    sB.replace(site_j, dopant)
    aB = adaptor.get_atoms(sB)
    if relax:
        aB, _ = relax_structure(aB, calc)
    else:
        aB.calc = calc
    EB = aB.get_potential_energy()

    # E(AB): dopant at both sites
    sAB = struct_undoped.copy()
    sAB.replace(site_i, dopant)
    sAB.replace(site_j, dopant)
    aAB = adaptor.get_atoms(sAB)
    if relax:
        aAB, _ = relax_structure(aAB, calc)
    else:
        aAB.calc = calc
    EAB = aAB.get_potential_energy()

    E_int = EAB - EA - EB + E0
    return E_int, {"E0": E0, "EA": EA, "EB": EB, "EAB": EAB}


def run_material(name, cif_path, supercell, target_species, dopant, calc, max_pairs=15):
    """Run interaction energy scan for one material."""
    print(f"\n{'='*60}")
    print(f"{name}: {dopant} → {target_species} site")
    print(f"{'='*60}")

    struct = load_and_expand(cif_path, supercell)
    n_atoms = len(struct)
    print(f"  Supercell: {supercell}, {n_atoms} atoms")

    target_indices = get_target_site_indices(struct, target_species)
    print(f"  {target_species} sites: {len(target_indices)}")

    pairs = compute_pair_distances(struct, target_indices)
    print(f"  Unique pairs: {len(pairs)}")

    # Bin by distance (round to 0.1 Å) and pick one representative per bin
    bins = {}
    for i, j, dist in pairs:
        bin_key = round(dist, 1)
        if bin_key not in bins:
            bins[bin_key] = (i, j, dist)

    selected = sorted(bins.values(), key=lambda x: x[2])[:max_pairs]
    print(f"  Selected {len(selected)} distance bins from {selected[0][2]:.2f} to {selected[-1][2]:.2f} Å")

    results = []
    for idx, (site_i, site_j, dist) in enumerate(selected):
        t0 = time.time()
        print(f"  [{idx+1}/{len(selected)}] d={dist:.2f} Å (sites {site_i},{site_j})...", end="", flush=True)

        E_int, energies = compute_interaction_energy(
            struct, target_species, dopant, site_i, site_j, calc, relax=True
        )

        dt = time.time() - t0
        print(f" E_int = {E_int*1000:.1f} meV ({dt:.0f}s)")

        results.append({
            "distance": round(dist, 3),
            "site_i": site_i,
            "site_j": site_j,
            "E_interaction_eV": round(E_int, 6),
            "E_interaction_meV": round(E_int * 1000, 1),
            **{k: round(v, 6) for k, v in energies.items()},
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Dopant-dopant interaction energy scan")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dopant", default="Al", help="Dopant element")
    parser.add_argument("--no-relax", action="store_true", help="Skip relaxation (single-point only)")
    parser.add_argument("--max-pairs", type=int, default=12, help="Max distance bins per material")
    args = parser.parse_args()

    print(f"Loading MACE-MP-0 on {args.device}...")
    calc = get_mace_calculator(args.device)
    print("Done.\n")

    output = {"dopant": args.dopant, "device": args.device, "materials": {}}

    # ── LiCoO₂ (layered) ──
    # 3×3×2 supercell = 72 atoms, 18 Co sites — good for interaction scan
    lco_results = run_material(
        name="LiCoO₂ (layered R-3m)",
        cif_path=DATA_DIR / "lco_parent.cif",
        supercell=[3, 3, 2],
        target_species="Co",
        dopant=args.dopant,
        calc=calc,
        max_pairs=args.max_pairs,
    )
    output["materials"]["LiCoO2"] = {
        "structure": "layered",
        "supercell": [3, 3, 2],
        "results": lco_results,
    }

    # ── LiMn₂O₄ (spinel) ──
    # 1×1×1 = 56 atoms (already 8 formula units), 16 Mn sites
    lmo_results = run_material(
        name="LiMn₂O₄ (spinel Fd-3m)",
        cif_path=DATA_DIR / "lmo_spinel.cif",
        supercell=[1, 1, 1],  # already 56 atoms
        target_species="Mn",
        dopant=args.dopant,
        calc=calc,
        max_pairs=args.max_pairs,
    )
    output["materials"]["LiMn2O4"] = {
        "structure": "spinel",
        "supercell": [1, 1, 1],
        "results": lmo_results,
    }

    # ── Save results ──
    out_path = SCRIPT_DIR / f"interaction_{args.dopant}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for mat_name, mat_data in output["materials"].items():
        results = mat_data["results"]
        if not results:
            continue
        dists = [r["distance"] for r in results]
        e_ints = [r["E_interaction_meV"] for r in results]
        print(f"\n  {mat_name} ({mat_data['structure']}):")
        print(f"    Distance range: {min(dists):.1f} – {max(dists):.1f} Å")
        print(f"    E_int range: {min(e_ints):.0f} – {max(e_ints):.0f} meV")
        print(f"    |E_int| at max distance: {abs(e_ints[-1]):.0f} meV")

        # Decay: ratio of |E_int| at d_max vs d_min
        if abs(e_ints[0]) > 1e-3:
            decay = abs(e_ints[-1]) / abs(e_ints[0])
            print(f"    Decay ratio (far/near): {decay:.2f}")
            if decay > 0.3:
                print(f"    → LONG-RANGE interaction (slow decay)")
            else:
                print(f"    → SHORT-RANGE interaction (fast decay)")

    # ── Voltage sensitivity prediction ──
    print(f"\n  PREDICTION:")
    for mat_name, mat_data in output["materials"].items():
        results = mat_data["results"]
        if not results:
            continue
        e_ints = [abs(r["E_interaction_meV"]) for r in results]
        spread = max(e_ints) - min(e_ints)
        print(f"    {mat_name}: interaction spread = {spread:.0f} meV")
        print(f"      → {'HIGH' if spread > 20 else 'LOW'} sensitivity to dopant placement")
        print(f"      → Voltage rankings {'VULNERABLE' if spread > 20 else 'ROBUST'} to disorder")


if __name__ == "__main__":
    main()
