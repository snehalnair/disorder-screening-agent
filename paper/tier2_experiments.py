#!/usr/bin/env python3
"""
Tier 2 Reviewer Response Experiments — GPU version
===================================================
Run on Colab/Lightning with GPU. Bundles all three experiments:

1. Partial delithiation (x=0.5) — does voltage ranking scrambling persist?
2. SQS convergence (already done from existing data, included for completeness)
3. Interaction energy with position relaxation — does NN sign hold?

Usage (Colab):
    !pip install mace-torch pymatgen ase scipy
    !python tier2_experiments.py --device cuda

Usage (CPU, slow):
    python paper/tier2_experiments.py --device cpu --dopants Al,Cr,Ga,Ge,Ni,Ti
"""

import argparse
import json
import pathlib
import time
import itertools
import numpy as np
from scipy import stats

# ── Paths ──
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


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Partial delithiation
# ══════════════════════════════════════════════════════════════════════

def remove_fraction_li(atoms, fraction=0.5, seed=42):
    rng = np.random.default_rng(seed)
    li_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
    n_remove = max(1, int(len(li_indices) * fraction))
    remove_idx = sorted(rng.choice(li_indices, size=n_remove, replace=False), reverse=True)
    new_atoms = atoms.copy()
    del new_atoms[remove_idx]
    return new_atoms, n_remove


def relax_atoms(atoms, calc, fmax=0.15, max_steps=300):
    """Relax with BFGS, FIRE fallback."""
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


def experiment1_partial_delithiation(calc, dopants, fraction=0.5, n_li_samples=3):
    """Compare full vs partial delithiation voltage rankings."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT 1: Partial delithiation (x = {fraction})")
    print("=" * 70)

    cif = DATA_DIR / "lco_parent.cif"
    parent = Structure.from_file(str(cif))
    parent.make_supercell([4, 4, 4])
    co_sites = [i for i, sp in enumerate(parent.species) if str(sp) == "Co"]
    print(f"  Parent: {len(parent)} atoms, {len(co_sites)} Co sites\n")

    results = {}
    for dop in dopants:
        ckpt_file = LCO_DIR / f"LiCoO2_layered_{dop}.json"
        if not ckpt_file.exists():
            print(f"  [{dop}] no checkpoint, skip")
            continue

        with open(ckpt_file) as f:
            ckpt = json.load(f)
        v_full_ord = ckpt["ordered"]["voltage"]
        sqs_voltages = [s["voltage"] for s in ckpt["sqs_results"] if "voltage" in s]
        v_full_dis = np.mean(sqs_voltages)

        t0 = time.time()

        # Build and relax ordered doped structure
        struct = parent.copy()
        struct.replace(co_sites[0], dop)
        atoms_lith = relax_atoms(adaptor.get_atoms(struct), calc)
        e_lith = atoms_lith.get_potential_energy()

        # Full delithiation (for consistency check)
        atoms_full = atoms_lith.copy()
        li_idx = [i for i, s in enumerate(atoms_full.get_chemical_symbols()) if s == "Li"]
        for idx in sorted(li_idx, reverse=True):
            del atoms_full[idx]
        atoms_full = relax_atoms(atoms_full, calc, fmax=0.20)
        e_full = atoms_full.get_potential_energy()
        v_full_recomputed = -(e_full - e_lith) / len(li_idx)

        # Partial delithiation
        partial_voltages = []
        for seed in range(n_li_samples):
            atoms_partial, n_removed = remove_fraction_li(atoms_lith, fraction, seed=seed)
            atoms_partial = relax_atoms(atoms_partial, calc, fmax=0.20)
            e_partial = atoms_partial.get_potential_energy()
            v = -(e_partial - e_lith) / n_removed
            partial_voltages.append(v)

        v_partial = np.mean(partial_voltages)
        v_partial_std = np.std(partial_voltages, ddof=1) if len(partial_voltages) > 1 else 0
        dt = time.time() - t0

        results[dop] = {
            "v_full_ordered": v_full_ord,
            "v_full_disordered": v_full_dis,
            "v_full_recomputed": v_full_recomputed,
            "v_partial_ordered": v_partial,
            "v_partial_std": v_partial_std,
        }
        print(f"  [{dop:>3s}] full={v_full_ord:+.3f}  partial={v_partial:+.3f} ± {v_partial_std:.3f}  ({dt:.0f}s)")

    # Correlations
    if len(results) >= 4:
        full = [results[d]["v_full_ordered"] for d in results]
        partial = [results[d]["v_partial_ordered"] for d in results]
        rho, p = stats.spearmanr(full, partial)
        print(f"\n  ρ(full vs partial ordered): {rho:+.3f} (p={p:.4f}, n={len(results)})")
        results["_rho_full_vs_partial"] = rho
        results["_p_value"] = p

    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: SQS convergence (reanalysis only)
# ══════════════════════════════════════════════════════════════════════

def experiment2_sqs_convergence():
    """Jackknife analysis of existing SQS data."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: SQS convergence (jackknife of existing data)")
    print("=" * 70)

    def load_material(ckpt_dir):
        results = []
        for f in sorted(ckpt_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            dopant = f.stem.split("_")[-1]
            sqs = data.get("sqs_results", [])
            if len(sqs) < 3:
                continue
            results.append({"dopant": dopant, "ordered": data["ordered"], "sqs": sqs})
        return results

    def rho_subset(results, prop, indices):
        o_vals, d_vals = [], []
        for r in results:
            o = r["ordered"].get(prop)
            subset = [r["sqs"][i] for i in indices if i < len(r["sqs"])]
            vals = [s.get(prop) for s in subset if prop in s]
            if o is None or not vals:
                continue
            d_vals.append(np.mean(vals))
            o_vals.append(o)
        if len(o_vals) < 4:
            return np.nan
        return stats.spearmanr(o_vals, d_vals)[0]

    summary = {}
    for name, path in [("LCO", LCO_DIR), ("LMO", PROJECT_DIR / "lmo")]:
        results = load_material(path)
        if not results:
            continue
        n_sqs = min(len(r["sqs"]) for r in results)
        print(f"\n  {name}: {len(results)} dopants, {n_sqs} SQS")

        for prop in ["voltage", "formation_energy"]:
            rho_full = rho_subset(results, prop, list(range(n_sqs)))
            if np.isnan(rho_full):
                continue
            row = {"full": rho_full}
            for k in [3, 4]:
                combos = list(itertools.combinations(range(n_sqs), k))
                rhos = [rho_subset(results, prop, list(c)) for c in combos]
                rhos = [r for r in rhos if not np.isnan(r)]
                row[f"k{k}_mean"] = np.mean(rhos)
                row[f"k{k}_std"] = np.std(rhos)
            print(f"    {prop}: full ρ={rho_full:+.3f}, "
                  f"k=3: {row.get('k3_mean', 0):+.3f}±{row.get('k3_std', 0):.3f}, "
                  f"k=4: {row.get('k4_mean', 0):+.3f}±{row.get('k4_std', 0):.3f}")
            summary[f"{name}_{prop}"] = row

    return summary


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Interaction energy — relaxed check
# ══════════════════════════════════════════════════════════════════════

def experiment3_interaction_relaxed(calc):
    """Compare relaxed vs unrelaxed E_int at NN, medium, far distances."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.optimize import BFGS, FIRE
    adaptor = AseAtomsAdaptor()

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Relaxed vs unrelaxed interaction energy")
    print("=" * 70)

    cif = DATA_DIR / "lco_parent.cif"
    struct = Structure.from_file(str(cif))
    struct.make_supercell([3, 3, 2])

    def sp_energy(struct_pm, calc):
        atoms = adaptor.get_atoms(struct_pm)
        atoms.calc = calc
        return atoms.get_potential_energy()

    def relaxed_energy(struct_pm, calc, fmax=0.05, steps=500):
        """Position-only relaxation (fixed cell) with tight convergence."""
        atoms = adaptor.get_atoms(struct_pm)
        atoms.calc = calc
        try:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        except Exception:
            try:
                opt = FIRE(atoms, logfile=None)
                opt.run(fmax=fmax * 2, steps=steps)
            except Exception:
                pass
        return atoms.get_potential_energy()

    def eint(struct, si, sj, dopant, energy_fn):
        s0 = struct.copy()
        E0 = energy_fn(s0, calc)

        sA = struct.copy()
        sA.replace(si, dopant)
        EA = energy_fn(sA, calc)

        sB = struct.copy()
        sB.replace(sj, dopant)
        EB = energy_fn(sB, calc)

        sAB = struct.copy()
        sAB.replace(si, dopant)
        sAB.replace(sj, dopant)
        EAB = energy_fn(sAB, calc)

        return (EAB - EA - EB + E0) * 1000  # meV

    pairs = [(18, 20, "NN ~2.9Å"), (18, 28, "med ~5.0Å"), (18, 19, "far ~14.2Å")]
    results = []

    print(f"\n  {'Pair':<15} {'SP (meV)':>12} {'Relaxed (meV)':>15} {'Δ (meV)':>10}")
    print("  " + "-" * 55)

    for si, sj, label in pairs:
        t0 = time.time()
        e_sp = eint(struct, si, sj, "Al", sp_energy)
        e_rx = eint(struct, si, sj, "Al", relaxed_energy)
        dt = time.time() - t0
        delta = e_rx - e_sp
        print(f"  {label:<15} {e_sp:>+12.1f} {e_rx:>+15.1f} {delta:>+10.1f}  ({dt:.0f}s)")
        results.append({
            "label": label, "sites": [si, sj],
            "sp_meV": round(e_sp, 1), "relaxed_meV": round(e_rx, 1),
            "delta_meV": round(delta, 1),
        })

    # Check sign
    if results[0]["sp_meV"] * results[0]["relaxed_meV"] > 0:
        print(f"\n  NN sign is CONSISTENT between SP and relaxed ✓")
    else:
        print(f"\n  NN sign CHANGES between SP and relaxed ✗")

    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dopants", default=None,
                        help="Comma-separated dopant list (default: all LCO)")
    args = parser.parse_args()

    calc = get_calc(args.device)

    # Determine dopants
    if args.dopants:
        dopants = args.dopants.split(",")
    else:
        dopants = [f.stem.split("_")[-1] for f in sorted(LCO_DIR.glob("*.json"))]

    print(f"Dopants for Exp 1: {dopants}")

    # Run experiments
    t_start = time.time()

    exp2_results = experiment2_sqs_convergence()
    exp1_results = experiment1_partial_delithiation(calc, dopants)
    exp3_results = experiment3_interaction_relaxed(calc)

    total_time = time.time() - t_start

    # Save all results
    output = {
        "experiment1_partial_delithiation": exp1_results,
        "experiment2_sqs_convergence": exp2_results,
        "experiment3_interaction_relaxed": exp3_results,
        "total_time_s": round(total_time),
        "device": args.device,
    }

    out_path = SCRIPT_DIR / "tier2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  All experiments complete in {total_time/60:.1f} min")
    print(f"  Results saved to {out_path}")
    print(f"{'='*70}")
