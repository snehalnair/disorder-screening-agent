#!/usr/bin/env python3
"""
NMC811 disorder screening: extend layered "danger zone" to mixed-TM cathode.

Creates an NMC811-like layered structure (LiNi0.8Mn0.1Co0.1O2) by
substituting the LiNiO2 parent, then screens dopants for disorder sensitivity.

This addresses the reviewer question: "Are the danger zones consistent
across other layered systems (e.g., NMC variants)?"

If NMC811 also shows voltage ranking destruction (rho ~ 0), it confirms
the layered topology — not specific chemistry — drives the effect.

Usage (Colab A100):
    !pip install mace-torch pymatgen ase scipy numpy
    !python paper/nmc_screening.py --device cuda

Usage (CPU, subset):
    !python paper/nmc_screening.py --device cpu --dopants Al,Cr,Ga,Ti,Fe,Mg
"""

import argparse
import json
import pathlib
import time
import numpy as np
from scipy import stats

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"
OUT_DIR = PROJECT_DIR / "paper"


def get_calc(device="cpu"):
    """Load MACE-MP-0."""
    from mace.calculators import mace_mp
    print(f"Loading MACE-MP-0 on {device}...")
    calc = mace_mp(device=device, default_dtype="float64")
    print("Done.\n")
    return calc


def relax_atoms(atoms, calc, fmax=0.15, max_steps=300):
    """Relax structure with BFGS, FIRE fallback."""
    from ase.optimize import BFGS, FIRE
    from ase.filters import FrechetCellFilter

    atoms = atoms.copy()
    atoms.calc = calc
    try:
        ecf = FrechetCellFilter(atoms)
        opt = BFGS(ecf, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        return atoms
    except Exception:
        pass
    atoms2 = atoms.copy()
    atoms2.calc = calc
    try:
        ecf = FrechetCellFilter(atoms2)
        opt = FIRE(ecf, logfile=None)
        opt.run(fmax=fmax * 1.5, steps=max_steps)
        return atoms2
    except Exception:
        return atoms2


def build_nmc811_supercell():
    """Build NMC811-like supercell from LiNiO2 parent.

    NMC811 = LiNi0.8Mn0.1Co0.1O2
    Strategy: start from LiNiO2 (R-3m), build 4x4x4 supercell (256 atoms, 64 Ni),
    then replace ~10% Ni with Mn and ~10% Ni with Co to make NMC811 baseline.
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = Structure.from_file(str(DATA_DIR / "lno_parent.cif"))
    struct.make_supercell([4, 4, 4])

    # Find all Ni sites
    ni_indices = [i for i, sp in enumerate(struct.species) if str(sp) == "Ni"]
    n_ni = len(ni_indices)
    print(f"Parent supercell: {len(struct)} atoms, {n_ni} Ni sites")

    # NMC811: 80% Ni, 10% Mn, 10% Co
    n_mn = max(1, round(n_ni * 0.1))  # ~6 Mn
    n_co = max(1, round(n_ni * 0.1))  # ~6 Co
    n_ni_final = n_ni - n_mn - n_co

    rng = np.random.default_rng(2024)
    shuffled = rng.permutation(ni_indices)
    mn_sites = sorted(shuffled[:n_mn])
    co_sites = sorted(shuffled[n_mn:n_mn + n_co])

    # Replace Ni with Mn and Co (do in reverse order of site index to avoid shifting)
    for site_idx in sorted(mn_sites + co_sites, reverse=True):
        if site_idx in mn_sites:
            struct.replace(site_idx, "Mn")
        else:
            struct.replace(site_idx, "Co")

    ni_remaining = [i for i, sp in enumerate(struct.species) if str(sp) == "Ni"]
    print(f"NMC811 supercell: {len(struct)} atoms")
    print(f"  Ni: {len(ni_remaining)}, Mn: {n_mn}, Co: {len(co_sites)}")
    print(f"  Composition: LiNi{len(ni_remaining)/n_ni:.2f}Mn{n_mn/n_ni:.2f}Co{len(co_sites)/n_ni:.2f}O2")

    return AseAtomsAdaptor.get_atoms(struct), struct


def substitute_ordered(struct, dopant, target="Ni"):
    """Farthest-first substitution at Ni site."""
    from pymatgen.io.ase import AseAtomsAdaptor
    s = struct.copy()
    target_indices = [i for i, sp in enumerate(s.species) if str(sp) == target]
    best_site = target_indices[0]
    best_min_dist = 0
    for idx in target_indices:
        dists = [s.get_distance(idx, j) for j in target_indices if j != idx]
        min_d = min(dists) if dists else 0
        if min_d > best_min_dist:
            best_min_dist = min_d
            best_site = idx
    s.replace(best_site, dopant)
    return AseAtomsAdaptor.get_atoms(s)


def generate_sqs(struct, dopant, target="Ni", n_sqs=5, seed=42):
    """Generate SQS realisations via random site selection."""
    from pymatgen.io.ase import AseAtomsAdaptor
    rng = np.random.default_rng(seed)
    target_indices = [i for i, sp in enumerate(struct.species) if str(sp) == target]
    results = []
    for i in range(n_sqs):
        s = struct.copy()
        site = rng.choice(target_indices)
        s.replace(site, dopant)
        results.append(AseAtomsAdaptor.get_atoms(s))
    return results


def remove_all_li(atoms):
    """Remove all Li atoms."""
    symbols = atoms.get_chemical_symbols()
    li_indices = sorted([i for i, s in enumerate(symbols) if s == "Li"], reverse=True)
    n_li = len(li_indices)
    new_atoms = atoms.copy()
    for idx in li_indices:
        del new_atoms[idx]
    return new_atoms, n_li


def run_nmc_screening(device="cpu", dopants=None, n_sqs=5):
    """Screen dopants in NMC811 for disorder sensitivity."""
    calc = get_calc(device)
    _, parent_struct = build_nmc811_supercell()

    # Default dopants: common NMC dopants from literature
    if dopants is None:
        dopants = ["Al", "Ti", "Mg", "Zr", "Nb", "Ta", "W", "Fe", "Cr", "Ga",
                    "V", "Mo", "Sn", "Sb", "Ge", "Cu"]

    print(f"\n{'=' * 70}")
    print(f"  NMC811 SCREENING: {len(dopants)} dopants, {n_sqs} SQS each")
    print(f"{'=' * 70}\n")

    results = {}
    for dopant in dopants:
        t0 = time.time()
        try:
            # --- Ordered ---
            ordered_atoms = substitute_ordered(parent_struct, dopant)
            ordered_lith = relax_atoms(ordered_atoms, calc)
            ordered_delith_raw, n_li = remove_all_li(ordered_lith)
            ordered_delith = relax_atoms(ordered_delith_raw, calc)
            v_ord = -(ordered_delith.get_potential_energy() -
                      ordered_lith.get_potential_energy()) / n_li
            ef_ord = ordered_lith.get_potential_energy() / len(ordered_lith)

            # --- SQS ---
            sqs_structs = generate_sqs(parent_struct, dopant, n_sqs=n_sqs)
            sqs_voltages, sqs_efs = [], []
            for sqs_atoms in sqs_structs:
                try:
                    sqs_lith = relax_atoms(sqs_atoms, calc)
                    sqs_delith_raw, n_li_sqs = remove_all_li(sqs_lith)
                    sqs_delith = relax_atoms(sqs_delith_raw, calc)
                    v_sqs = -(sqs_delith.get_potential_energy() -
                              sqs_lith.get_potential_energy()) / n_li_sqs
                    ef_sqs = sqs_lith.get_potential_energy() / len(sqs_lith)
                    sqs_voltages.append(v_sqs)
                    sqs_efs.append(ef_sqs)
                except Exception as e:
                    print(f"    SQS failed: {e}")

            dt = time.time() - t0
            results[dopant] = {
                "voltage_ordered": float(v_ord),
                "voltage_disordered_mean": float(np.mean(sqs_voltages)) if sqs_voltages else None,
                "voltage_disordered_std": float(np.std(sqs_voltages)) if sqs_voltages else None,
                "ef_ordered": float(ef_ord),
                "ef_disordered_mean": float(np.mean(sqs_efs)) if sqs_efs else None,
                "n_sqs_converged": len(sqs_voltages),
                "time_s": dt,
            }
            v_dis_str = f"{np.mean(sqs_voltages):.3f}+/-{np.std(sqs_voltages):.3f}" if sqs_voltages else "FAIL"
            print(f"  [{dopant:4s}] V_ord={v_ord:.3f}  V_dis={v_dis_str}  ({dt:.0f}s)")

        except Exception as e:
            dt = time.time() - t0
            print(f"  [{dopant:4s}] FAILED: {e}  ({dt:.0f}s)")
            results[dopant] = {"error": str(e), "time_s": dt}

    # --- Correlations ---
    print(f"\n{'=' * 70}")
    print(f"  NMC811 RESULTS")
    print(f"{'=' * 70}\n")

    v_ord_list = [r["voltage_ordered"] for r in results.values()
                  if "voltage_ordered" in r and r.get("voltage_disordered_mean")]
    v_dis_list = [r["voltage_disordered_mean"] for r in results.values()
                  if r.get("voltage_disordered_mean")]
    ef_ord_list = [r["ef_ordered"] for r in results.values()
                   if "ef_ordered" in r and r.get("ef_disordered_mean")]
    ef_dis_list = [r["ef_disordered_mean"] for r in results.values()
                   if r.get("ef_disordered_mean")]

    if len(v_ord_list) >= 4:
        rho_v, p_v = stats.spearmanr(v_ord_list, v_dis_list)
        print(f"  NMC811 voltage rho:          {rho_v:+.3f} (p={p_v:.4f}, n={len(v_ord_list)})")
    else:
        rho_v, p_v = None, None
        print(f"  NMC811 voltage: insufficient data (n={len(v_ord_list)})")

    if len(ef_ord_list) >= 4:
        rho_ef, p_ef = stats.spearmanr(ef_ord_list, ef_dis_list)
        print(f"  NMC811 formation energy rho: {rho_ef:+.3f} (p={p_ef:.4f}, n={len(ef_ord_list)})")

    # Compare with LCO/LNO
    print(f"\n  Comparison:")
    print(f"    LiCoO2  voltage rho: -0.25 (n=20)")
    print(f"    LiNiO2  voltage rho: -0.06 (n=14)")
    if rho_v is not None:
        print(f"    NMC811  voltage rho: {rho_v:+.2f} (n={len(v_ord_list)})")
        if rho_v < 0.3:
            print(f"\n  --> NMC811 CONFIRMS layered danger zone")
        else:
            print(f"\n  --> NMC811 shows partial ranking preservation")

    # Save
    output = {
        "material": "NMC811 (LiNi0.8Mn0.1Co0.1O2)",
        "structure": "layered R-3m",
        "mlip": "MACE-MP-0",
        "n_dopants": len(results),
        "n_sqs": n_sqs,
        "supercell": [4, 4, 4],
        "n_atoms": len(parent_struct),
        "dopant_results": results,
    }
    if rho_v is not None:
        output["voltage_rho"] = round(rho_v, 3)
        output["voltage_p"] = round(p_v, 4)
    if len(ef_ord_list) >= 4:
        output["ef_rho"] = round(rho_ef, 3)

    out_path = OUT_DIR / "nmc811_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dopants", default=None,
                        help="Comma-separated dopant list")
    parser.add_argument("--n-sqs", type=int, default=5)
    args = parser.parse_args()

    dopant_list = args.dopants.split(",") if args.dopants else None
    run_nmc_screening(device=args.device, dopants=dopant_list, n_sqs=args.n_sqs)
