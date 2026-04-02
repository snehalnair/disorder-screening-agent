#!/usr/bin/env python3
"""
Partial delithiation (x=0.25) with SQS ensembles for LiCoO₂.
==============================================================

Same methodology as partial_delithiation_sqs.py but at x=0.25
(25% Li removal) to test ranking stability across delithiation depths.

Per-dopant checkpointing to Google Drive.

Usage (Colab):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install mace-torch ase pymatgen scipy
    !git clone https://github.com/snehalnair/disorder-screening-agent.git
    %cd disorder-screening-agent
    !python paper/partial_delithiation_x025.py --device cuda

Outputs:
    paper/partial_delithiation_x025_results.json
"""

import argparse
import json
import os
import pathlib
import time
import sys
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", message="logm result may be inaccurate")

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"
LOCAL_OUT = SCRIPT_DIR / "partial_delithiation_x025_results.json"

GDRIVE_DIR = pathlib.Path("/content/drive/MyDrive/disorder_results")
GDRIVE_OUT = GDRIVE_DIR / "partial_delithiation_x025_results.json"

DELITH_FRACTION = 0.25   # <-- only change vs x=0.5 script
N_LI_SEEDS = 3
N_SQS = 5

ALL_DOPANTS = [
    "Al", "Cr", "Cu", "Fe", "Ga", "Ge", "Mg", "Mn", "Mo",
    "Nb", "Ni", "Ru", "Sb", "Sc", "Sn", "Ta", "Ti", "V", "Zr"
]


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


def save_checkpoint(results):
    with open(LOCAL_OUT, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    if GDRIVE_DIR.exists():
        with open(GDRIVE_OUT, 'w') as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        print(f"    [checkpoint] saved to Drive + local")
    else:
        print(f"    [checkpoint] saved locally (Drive not mounted)")


def load_checkpoint():
    for path in [GDRIVE_OUT, LOCAL_OUT]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                print(f"  Resuming from {path} ({len(data.get('dopant_results', {}))} dopants done)")
                return data
            except (json.JSONDecodeError, KeyError):
                print(f"  Warning: corrupt checkpoint at {path}, starting fresh")
    return None


def get_calc(device="cpu"):
    from mace.calculators import mace_mp
    print(f"Loading MACE-MP-0 on {device}...")
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",
                   device=device)
    print("Done.")
    return calc


def build_ordered_structure(dopant, supercell=(3, 3, 2), n_dopant=2):
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    cif_path = DATA_DIR / "lco_parent.cif"
    struct = Structure.from_file(str(cif_path))
    struct.make_supercell(list(supercell))

    co_sites = [i for i, sp in enumerate(struct.species) if str(sp) == "Co"]
    chosen = [co_sites[0]]
    for _ in range(n_dopant - 1):
        best_site, best_min_dist = None, -1
        for s in co_sites:
            if s in chosen:
                continue
            min_d = min(struct.get_distance(s, c) for c in chosen)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_site = s
        chosen.append(best_site)

    for s in chosen:
        struct.replace(s, dopant)

    atoms = AseAtomsAdaptor.get_atoms(struct)
    return atoms, struct


def build_sqs_structures(dopant, n_realisations=5, supercell=(3, 3, 2), n_dopant=2):
    from pymatgen.core import Structure

    cif_path = DATA_DIR / "lco_parent.cif"
    struct = Structure.from_file(str(cif_path))
    struct.make_supercell(list(supercell))

    sys.path.insert(0, str(PROJECT_DIR))
    from stages.stage5.sqs_generator import generate_sqs

    parent_prim = Structure.from_file(str(cif_path))
    co_sites = [i for i, sp in enumerate(struct.species) if str(sp) == "Co"]
    n_co = len(co_sites)
    conc = n_dopant / n_co

    try:
        sqs_list = generate_sqs(
            parent_structure=parent_prim,
            dopant_element=dopant,
            target_species="Co",
            concentration=conc,
            supercell_matrix=list(supercell),
            n_realisations=n_realisations,
            correlation_cutoff=6.0,
        )
    except Exception as e:
        print(f"    SQS generation failed ({e}), using random placement fallback")
        sqs_list = _random_sqs_fallback(struct, dopant, co_sites, n_realisations, n_dopant)

    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_atoms(s) for s in sqs_list]


def _random_sqs_fallback(struct, dopant, co_sites, n_realisations, n_dopant=2):
    import random as rng
    from pymatgen.core import Structure

    structures = []
    for seed in range(n_realisations):
        s = struct.copy()
        rng.seed(seed + 42)
        sites = rng.sample(co_sites, n_dopant)
        for site in sites:
            s.replace(site, dopant)
        structures.append(s)
    return structures


def relax_structure(atoms, calc, fmax=0.15, max_steps=300):
    from ase.optimize import BFGS, FIRE
    from ase.filters import FrechetCellFilter

    atoms = atoms.copy()
    atoms.calc = calc
    try:
        ecf = FrechetCellFilter(atoms)
        opt = BFGS(ecf, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        if opt.nsteps < max_steps:
            return atoms, True
    except Exception:
        pass
    try:
        atoms2 = atoms.copy()
        atoms2.calc = calc
        ecf = FrechetCellFilter(atoms2)
        opt = FIRE(ecf, logfile=None)
        opt.run(fmax=fmax * 1.5, steps=max_steps)
        return atoms2, opt.nsteps < max_steps
    except Exception:
        return atoms, False


def remove_fraction_li(atoms, fraction=0.25, seed=42):
    rng = np.random.default_rng(seed)
    li_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
    n_remove = max(1, int(len(li_indices) * fraction))
    remove_idx = sorted(rng.choice(li_indices, size=n_remove, replace=False), reverse=True)
    new_atoms = atoms.copy()
    del new_atoms[list(remove_idx)]
    return new_atoms, n_remove


def relax_ionic_only(atoms, calc, fmax=0.10, max_steps=300):
    from ase.optimize import BFGS, FIRE

    atoms = atoms.copy()
    atoms.calc = calc
    try:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        return atoms, opt.nsteps < max_steps
    except Exception:
        pass
    try:
        atoms2 = atoms.copy()
        atoms2.calc = calc
        opt = FIRE(atoms2, logfile=None)
        opt.run(fmax=fmax * 1.5, steps=max_steps)
        return atoms2, opt.nsteps < max_steps
    except Exception:
        return atoms, False


def compute_partial_voltage(atoms_lith, calc, fraction=0.25, n_seeds=3):
    voltages = []
    for seed in range(n_seeds):
        atoms_delith, n_removed = remove_fraction_li(atoms_lith, fraction=fraction, seed=seed)
        atoms_delith, converged = relax_ionic_only(atoms_delith, calc)

        e_lith = atoms_lith.get_potential_energy()
        e_delith = atoms_delith.get_potential_energy()
        v = -(e_delith - e_lith) / n_removed

        if abs(v) < 20.0:
            voltages.append(v)
        else:
            print(f"      WARNING: unreasonable voltage {v:.1f} V (seed {seed}), skipping")

    if not voltages:
        return None, None
    return np.mean(voltages), np.std(voltages) if len(voltages) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dopants", default=None, help="Comma-separated dopant subset")
    args = parser.parse_args()

    dopants = args.dopants.split(",") if args.dopants else ALL_DOPANTS

    print("=" * 60)
    print("PARTIAL DELITHIATION SQS STUDY (x=0.25)")
    print(f"Material: LiCoO₂, x = 0 -> {DELITH_FRACTION}")
    print(f"SQS realisations: {N_SQS}, Li-removal seeds: {N_LI_SEEDS}")
    print("=" * 60)

    # Try to load checkpoint
    results = load_checkpoint()
    if results is None:
        results = {
            "material": "LiCoO2",
            "delithiation_fraction": DELITH_FRACTION,
            "n_sqs": N_SQS,
            "n_li_seeds": N_LI_SEEDS,
            "device": args.device,
            "dopant_results": {},
        }

    print(f"Google Drive checkpoint: {GDRIVE_OUT}")

    calc = get_calc(args.device)

    for dopant in dopants:
        if dopant in results["dopant_results"]:
            print(f"\n  Skipping {dopant} (already done)")
            continue

        print(f"\n{'='*50}")
        print(f"  Dopant: {dopant}")
        print(f"{'='*50}")

        t0 = time.time()

        # 1. Build and relax ordered structure
        print("  Building ordered structure...")
        atoms_ord, _ = build_ordered_structure(dopant, supercell=(3, 3, 2), n_dopant=2)
        print(f"  Relaxing ordered ({len(atoms_ord)} atoms)...")
        atoms_ord, _ = relax_structure(atoms_ord, calc)

        # 2. Compute ordered partial voltage
        print("  Computing ordered partial voltage...")
        v_ord, v_ord_std = compute_partial_voltage(atoms_ord, calc,
                                                    fraction=DELITH_FRACTION,
                                                    n_seeds=N_LI_SEEDS)
        if v_ord is None:
            print(f"    Ordered V_partial: ALL SEEDS DIVERGED — skipping dopant")
            results["dopant_results"][dopant] = {"error": "all voltage seeds diverged (ordered)"}
            save_checkpoint(results)
            continue

        print(f"    Ordered V_partial: {v_ord:.4f} +/- {v_ord_std:.4f}")

        # 3. Build and relax SQS structures
        print(f"  Generating {N_SQS} SQS structures...")
        sqs_atoms_list = build_sqs_structures(dopant, n_realisations=N_SQS,
                                               supercell=(3, 3, 2), n_dopant=2)
        print(f"    Generated {len(sqs_atoms_list)} SQS structures")

        sqs_voltages = []
        for j, sqs_atoms in enumerate(sqs_atoms_list):
            print(f"  Relaxing SQS {j+1}/{N_SQS}...")
            sqs_relaxed, _ = relax_structure(sqs_atoms, calc)

            v_sqs, _ = compute_partial_voltage(sqs_relaxed, calc,
                                                fraction=DELITH_FRACTION,
                                                n_seeds=N_LI_SEEDS)
            if v_sqs is not None:
                sqs_voltages.append(v_sqs)
                print(f"    SQS {j+1} V_partial: {v_sqs:.4f}")
            else:
                print(f"    SQS {j+1} V_partial: ALL SEEDS DIVERGED")

        if sqs_voltages:
            v_dis_mean = np.mean(sqs_voltages)
            v_dis_std = np.std(sqs_voltages, ddof=1) if len(sqs_voltages) > 1 else 0.0
            print(f"  Disordered V_partial: {v_dis_mean:.4f} +/- {v_dis_std:.4f}")
        else:
            v_dis_mean, v_dis_std = None, None
            print(f"  Disordered V_partial: ALL SQS DIVERGED")

        elapsed = time.time() - t0

        results["dopant_results"][dopant] = {
            "voltage_partial_ordered": v_ord,
            "voltage_partial_ordered_std": v_ord_std,
            "voltage_partial_disordered_mean": v_dis_mean,
            "voltage_partial_disordered_std": v_dis_std,
            "voltage_partial_sqs_values": sqs_voltages,
            "n_sqs_converged": len(sqs_voltages),
            "time_s": elapsed,
        }
        save_checkpoint(results)
        print(f"  Total time for {dopant}: {elapsed:.0f}s")

    # Compute Spearman ρ
    v_ords, v_diss = [], []
    for d, info in results["dopant_results"].items():
        if "error" in info or info.get("voltage_partial_disordered_mean") is None:
            continue
        v_ords.append(info["voltage_partial_ordered"])
        v_diss.append(info["voltage_partial_disordered_mean"])

    if len(v_ords) >= 4:
        rho, p = stats.spearmanr(v_ords, v_diss)
    else:
        rho, p = float('nan'), float('nan')

    results["voltage_partial_rho"] = rho
    results["voltage_partial_p"] = p
    results["n_dopants_ranked"] = len(v_ords)

    save_checkpoint(results)

    print(f"\n{'='*60}")
    print(f"  RESULTS (x=0 -> {DELITH_FRACTION})")
    print(f"  Spearman ρ (ordered vs disordered): {rho:.3f} (p={p:.4f})")
    print(f"  Dopants ranked: {len(v_ords)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
