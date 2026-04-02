#!/usr/bin/env python3
"""
Partial delithiation (x=0.5) with SQS ensembles for LiCoO₂.
=============================================================

Rebuilds ordered + SQS doped structures from parent CIF, relaxes them,
then computes voltage at x=0->0.5 (partial) for all LCO dopants.
Compares Spearman ρ(ordered vs disordered) at partial delithiation
to the full-delithiation result.

Per-dopant checkpointing to Google Drive (survives Colab disconnects).

Usage (Colab):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install mace-torch ase pymatgen
    !git clone https://github.com/snehalnair/disorder-screening-agent.git
    %cd disorder-screening-agent
    !python paper/partial_delithiation_sqs.py --device cuda

Outputs:
    Google Drive: /content/drive/MyDrive/disorder_results/partial_delithiation_sqs_results.json
    Local copy:   paper/partial_delithiation_sqs_results.json
"""

import argparse
import json
import os
import pathlib
import time
import sys

import numpy as np
from scipy import stats

# ── Paths ──
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"
LOCAL_OUT = SCRIPT_DIR / "partial_delithiation_sqs_results.json"

# Google Drive path (Colab)
GDRIVE_DIR = pathlib.Path("/content/drive/MyDrive/disorder_results")
GDRIVE_OUT = GDRIVE_DIR / "partial_delithiation_sqs_results.json"

DELITH_FRACTION = 0.5
N_LI_SEEDS = 3    # random Li-removal patterns per structure
N_SQS = 5         # SQS realisations per dopant

# Full dopant list for LiCoO₂ (from original screening)
ALL_DOPANTS = [
    "Al", "Cr", "Cu", "Fe", "Ga", "Ge", "Ir", "Mg", "Mn", "Mo",
    "Nb", "Ni", "Rh", "Ru", "Sb", "Sc", "Sn", "Ta", "Ti", "V", "Zr"
]


# ── JSON encoder ──
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
    """Save to both local and Google Drive."""
    # Local
    with open(LOCAL_OUT, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)

    # Google Drive (if mounted)
    if GDRIVE_DIR.exists():
        with open(GDRIVE_OUT, 'w') as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        print(f"    [checkpoint] saved to Drive + local")
    else:
        print(f"    [checkpoint] saved locally (Drive not mounted)")


def load_checkpoint():
    """Load from Google Drive first, then local."""
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
    """Load MACE-MP-0 calculator."""
    from mace.calculators import mace_mp
    print(f"Loading MACE-MP-0 on {device}...")
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",
                   device=device)
    print("Done.")
    return calc


def build_ordered_structure(dopant, supercell=(3, 3, 2), n_dopant=2):
    """Build ordered doped LiCoO₂ supercell from parent CIF.

    Uses farthest-first placement for n_dopant substitutions (maximises
    dopant–dopant distance = ordered limit).
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    cif_path = DATA_DIR / "lco_parent.cif"
    if not cif_path.exists():
        raise FileNotFoundError(f"Parent CIF not found: {cif_path}")

    struct = Structure.from_file(str(cif_path))
    struct.make_supercell(list(supercell))

    co_sites = [i for i, sp in enumerate(struct.species) if str(sp) == "Co"]
    if not co_sites:
        raise ValueError("No Co sites found in supercell")

    # Farthest-first placement
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
    """Generate SQS doped LiCoO₂ structures."""
    from pymatgen.core import Structure

    cif_path = DATA_DIR / "lco_parent.cif"
    struct = Structure.from_file(str(cif_path))
    struct.make_supercell(list(supercell))

    # Use the project's SQS generator
    sys.path.insert(0, str(PROJECT_DIR))
    from stages.stage5.sqs_generator import generate_sqs

    parent_prim = Structure.from_file(str(cif_path))
    co_sites = [i for i, sp in enumerate(struct.species) if str(sp) == "Co"]
    n_co = len(co_sites)
    conc = n_dopant / n_co  # ~11% for 2/18

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
    """Fallback: random dopant placement if SQS generator fails."""
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
    """Relax with BFGS, fallback to FIRE."""
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


def remove_fraction_li(atoms, fraction=0.5, seed=42):
    """Remove a fraction of Li atoms randomly."""
    rng = np.random.default_rng(seed)
    li_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
    n_remove = max(1, int(len(li_indices) * fraction))
    remove_idx = sorted(rng.choice(li_indices, size=n_remove, replace=False), reverse=True)
    new_atoms = atoms.copy()
    del new_atoms[list(remove_idx)]
    return new_atoms, n_remove


def relax_ionic_only(atoms, calc, fmax=0.10, max_steps=300):
    """Ionic-only relaxation (no cell changes). Safer for partially delithiated structures."""
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


def compute_voltage_partial(atoms_lith, calc, fraction=0.5, n_seeds=3):
    """Compute voltage from partial delithiation, averaged over random Li removals.

    Uses ionic-only relaxation for the partially delithiated structure
    (cell shape kept from lithiated relaxation) to avoid cell explosion
    when many Li vacancies are introduced.
    """
    voltages = []
    e_lith = float(atoms_lith.get_potential_energy())
    for seed in range(n_seeds):
        atoms_delith, n_removed = remove_fraction_li(atoms_lith, fraction, seed=seed)
        atoms_delith_relax, conv = relax_ionic_only(atoms_delith, calc, fmax=0.10, max_steps=300)
        e_delith = float(atoms_delith_relax.get_potential_energy())
        v = -(e_delith - e_lith) / n_removed
        # Sanity check: voltage should be in reasonable range
        if abs(v) < 20.0:
            voltages.append(float(v))
        else:
            print(f"      WARNING: unreasonable voltage {v:.1f} V (seed {seed}), skipping")
    if not voltages:
        return None, None
    return float(np.mean(voltages)), float(np.std(voltages, ddof=1)) if len(voltages) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dopants", default=",".join(ALL_DOPANTS))
    parser.add_argument("--n-sqs", type=int, default=N_SQS)
    parser.add_argument("--fraction", type=float, default=DELITH_FRACTION)
    args = parser.parse_args()

    print("=" * 60)
    print("PARTIAL DELITHIATION SQS STUDY")
    print(f"Material: LiCoO₂, x = 0 -> {args.fraction}")
    print(f"SQS realisations: {args.n_sqs}, Li-removal seeds: {N_LI_SEEDS}")
    print("=" * 60)

    # Setup Google Drive checkpoint dir
    if GDRIVE_DIR.parent.exists():
        GDRIVE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Google Drive checkpoint: {GDRIVE_OUT}")
    else:
        print("Google Drive not mounted — checkpointing locally only")
        print("  To mount: from google.colab import drive; drive.mount('/content/drive')")

    calc = get_calc(args.device)
    dopants = args.dopants.split(",")

    # Load or create checkpoint
    existing = load_checkpoint()
    if existing and "dopant_results" in existing:
        results = existing
    else:
        results = {
            "material": "LiCoO2",
            "delithiation_fraction": args.fraction,
            "n_sqs": args.n_sqs,
            "n_li_seeds": N_LI_SEEDS,
            "device": args.device,
            "dopant_results": {},
        }

    for dopant in dopants:
        if dopant in results["dopant_results"]:
            print(f"\n[SKIP] {dopant} — already in checkpoint")
            continue

        print(f"\n{'='*50}")
        print(f"  Dopant: {dopant}")
        print(f"{'='*50}")
        t0 = time.time()

        try:
            # ── Ordered structure ──
            print(f"  Building ordered structure...")
            atoms_ord, _ = build_ordered_structure(dopant)
            print(f"  Relaxing ordered ({len(atoms_ord)} atoms)...")
            atoms_ord_relax, conv_ord = relax_structure(atoms_ord, calc)
            atoms_ord_relax.calc = calc

            print(f"  Computing ordered partial voltage...")
            v_ord, v_ord_std = compute_voltage_partial(
                atoms_ord_relax, calc, args.fraction, N_LI_SEEDS
            )
            if v_ord is None:
                print(f"    Ordered V_partial: ALL SEEDS DIVERGED — skipping dopant")
                results["dopant_results"][dopant] = {"error": "all voltage seeds diverged (ordered)"}
                save_checkpoint(results)
                continue
            print(f"    Ordered V_partial: {v_ord:.4f} +/- {v_ord_std:.4f}")

            # ── SQS structures ──
            print(f"  Generating {args.n_sqs} SQS structures...")
            sqs_atoms_list = build_sqs_structures(dopant, args.n_sqs)
            print(f"    Generated {len(sqs_atoms_list)} SQS structures")

            sqs_voltages = []
            for idx, sqs_atoms in enumerate(sqs_atoms_list):
                print(f"  Relaxing SQS {idx+1}/{len(sqs_atoms_list)}...")
                sqs_relax, conv_sqs = relax_structure(sqs_atoms, calc)
                sqs_relax.calc = calc

                v_sqs, v_sqs_std = compute_voltage_partial(
                    sqs_relax, calc, args.fraction, N_LI_SEEDS
                )
                if v_sqs is not None:
                    sqs_voltages.append(float(v_sqs))
                    print(f"    SQS {idx+1} V_partial: {v_sqs:.4f}")
                else:
                    print(f"    SQS {idx+1} V_partial: ALL SEEDS DIVERGED")

            if not sqs_voltages:
                print(f"  No valid SQS voltages — skipping dopant")
                results["dopant_results"][dopant] = {"error": "all SQS voltage seeds diverged"}
                save_checkpoint(results)
                continue

            results["dopant_results"][dopant] = {
                "voltage_partial_ordered": float(v_ord),
                "voltage_partial_ordered_std": float(v_ord_std),
                "voltage_partial_disordered_mean": float(np.mean(sqs_voltages)),
                "voltage_partial_disordered_std": float(np.std(sqs_voltages, ddof=1)) if len(sqs_voltages) > 1 else 0.0,
                "voltage_partial_sqs_values": [float(v) for v in sqs_voltages],
                "n_sqs_converged": len(sqs_voltages),
                "time_s": float(time.time() - t0),
            }

            print(f"  Disordered V_partial: {np.mean(sqs_voltages):.4f} +/- {np.std(sqs_voltages):.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results["dopant_results"][dopant] = {"error": str(e)}

        # Save after every dopant
        save_checkpoint(results)
        print(f"  Total time for {dopant}: {time.time() - t0:.0f}s")

    # ── Compute ranking correlations ──
    v_ord_list, v_dis_list, dop_names = [], [], []
    for d, data in results["dopant_results"].items():
        if "error" in data:
            continue
        v_ord_list.append(data["voltage_partial_ordered"])
        v_dis_list.append(data["voltage_partial_disordered_mean"])
        dop_names.append(d)

    if len(v_ord_list) >= 4:
        rho, p = stats.spearmanr(v_ord_list, v_dis_list)
        results["voltage_partial_rho"] = float(rho)
        results["voltage_partial_p"] = float(p)
        results["n_dopants_ranked"] = len(v_ord_list)

        # Also load full-delithiation data for comparison
        lco_dir = PROJECT_DIR / "lco"
        v_full_ord, v_full_dis = [], []
        for d in dop_names:
            ckpt = lco_dir / f"LiCoO2_layered_{d}.json"
            if ckpt.exists():
                with open(ckpt) as f:
                    cd = json.load(f)
                v_full_ord.append(cd["ordered"]["voltage"])
                sqs_v = [s["voltage"] for s in cd["sqs_results"] if "voltage" in s]
                v_full_dis.append(np.mean(sqs_v))

        if len(v_full_ord) >= 4:
            rho_full, p_full = stats.spearmanr(v_full_ord, v_full_dis)
            results["voltage_full_rho"] = float(rho_full)
            results["voltage_full_p"] = float(p_full)

        print(f"\n{'='*60}")
        print(f"RESULTS: Partial delithiation (x=0->{args.fraction})")
        print(f"  Ordered vs Disordered voltage:")
        print(f"    Partial (x={args.fraction}): ρ = {rho:+.3f} (p = {p:.4f}, n = {len(v_ord_list)})")
        if "voltage_full_rho" in results:
            print(f"    Full (x=1):        ρ = {rho_full:+.3f} (p = {p_full:.4f})")
        print(f"{'='*60}")

    save_checkpoint(results)
    print(f"\nFinal results saved.")
    print(f"  Local:  {LOCAL_OUT}")
    if GDRIVE_DIR.exists():
        print(f"  Drive:  {GDRIVE_OUT}")


if __name__ == "__main__":
    main()
