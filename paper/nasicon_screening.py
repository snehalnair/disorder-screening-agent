#!/usr/bin/env python3
"""
Na3V2(PO4)3 NASICON dopant screening: out-of-sample test for disorder predictor.
=================================================================================

Na3V2(PO4)3 (R-3c NASICON) has a 3D framework with corner-sharing VO6 octahedra
and PO4 tetrahedra — a structure class not present in our training set.

V sublattice anisotropy = 1.15 (d2/d1 = 5.10/4.42 A)
Predictor says: R_voltage = 1.15 (UNSAFE), R_Ef = 0 (SAFE)

The NASICON framework's 3D connectivity is intermediate between perfectly
isotropic spinels/perovskites (anisotropy=1.0) and layered structures (~1.9).
This tests whether the predictor correctly flags marginal cases.

Per-dopant checkpointing to Google Drive.

Usage (Colab A100):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install mace-torch ase pymatgen scipy
    !git clone https://github.com/snehalnair/disorder-screening-agent.git
    %cd disorder-screening-agent
    !python paper/nasicon_screening.py --device cuda

Usage (CPU, quick test):
    !python paper/nasicon_screening.py --device cpu --dopants Al,Mg,Co,Ni,Mn,Ti
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

# Google Drive checkpoint
GDRIVE_DIR = pathlib.Path("/content/drive/MyDrive/disorder_results")
GDRIVE_OUT = GDRIVE_DIR / "nasicon_screening_results.json"
LOCAL_OUT = SCRIPT_DIR / "nasicon_screening_results.json"

# Na3V2(PO4)3 dopants: common V-site substituents from literature
# V3+ octahedral (CN=6, r=0.64 A) -> candidates within ~35% radius mismatch
# Fe replaces V in the dopant list (since V is the host metal)
DEFAULT_DOPANTS = [
    "Al", "Co", "Cr", "Cu", "Fe", "Ga", "Ge", "Mg", "Mn", "Mo",
    "Nb", "Ni", "Ru", "Sc", "Sn", "Ti", "Zn", "Zr",
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
    """Save to both local and Google Drive."""
    with open(LOCAL_OUT, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    if GDRIVE_DIR.exists():
        with open(GDRIVE_OUT, 'w') as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        print(f"    [checkpoint] Drive + local")
    else:
        print(f"    [checkpoint] local only")


def load_checkpoint():
    """Load from Drive first, then local."""
    for path in [GDRIVE_OUT, LOCAL_OUT]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                n = len(data.get("dopant_results", {}))
                print(f"  Resuming from {path} ({n} dopants done)")
                return data
            except (json.JSONDecodeError, KeyError):
                print(f"  Warning: corrupt checkpoint at {path}")
    return None


def get_calc(device="cpu"):
    from mace.calculators import mace_mp
    print(f"Loading MACE-MP-0 on {device}...")
    calc = mace_mp(device=device, default_dtype="float64")
    print("Done.\n")
    return calc


def relax_atoms(atoms, calc, fmax=0.15, max_steps=300):
    """Relax with BFGS, fallback to FIRE."""
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


def build_nasicon_cell(supercell_size=(1, 1, 1)):
    """Build Na3V2(PO4)3 NASICON cell from parent CIF.

    Returns (ase_atoms, pymatgen_struct).
    Conventional hexagonal cell: 126 atoms (12 V sites, 24 Na sites).
    With (1,1,1) no expansion is needed — 12 V sites is sufficient for
    n_dopant=2 SQS sampling.
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    cif = DATA_DIR / "nasicon_parent.cif"
    if not cif.exists():
        raise FileNotFoundError(f"NASICON CIF not found: {cif}")

    struct = Structure.from_file(str(cif))
    struct.make_supercell(list(supercell_size))

    v_indices = [i for i, sp in enumerate(struct.species) if str(sp) == "V"]
    na_indices = [i for i, sp in enumerate(struct.species) if str(sp) == "Na"]
    print(f"NASICON cell: {len(struct)} atoms, {len(v_indices)} V, {len(na_indices)} Na")

    return AseAtomsAdaptor.get_atoms(struct), struct


def substitute_ordered(struct, dopant, target="V", n_dopant=2):
    """Farthest-first substitution at V sites (ordered baseline).

    Places n_dopant atoms by iteratively choosing the site farthest from
    all previously chosen sites (maximises dopant-dopant distance).
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    s = struct.copy()
    target_indices = [i for i, sp in enumerate(s.species) if str(sp) == target]

    # Farthest-first placement
    chosen = [target_indices[0]]
    for _ in range(n_dopant - 1):
        best_site, best_min_dist = None, -1
        for idx in target_indices:
            if idx in chosen:
                continue
            min_d = min(s.get_distance(idx, c) for c in chosen)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_site = idx
        chosen.append(best_site)

    for site in chosen:
        s.replace(site, dopant)
    return AseAtomsAdaptor.get_atoms(s)


def generate_sqs(struct, dopant, target="V", n_sqs=5, n_dopant=2, seed=42):
    """Generate SQS realisations via random site selection.

    With n_dopant > 1, different random placements produce genuinely
    different spatial arrangements, enabling ordered-vs-disordered comparison.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    rng = np.random.default_rng(seed)
    target_indices = [i for i, sp in enumerate(struct.species) if str(sp) == target]
    results = []
    for i in range(n_sqs):
        s = struct.copy()
        sites = rng.choice(target_indices, size=n_dopant, replace=False)
        for site in sites:
            s.replace(site, dopant)
        results.append(AseAtomsAdaptor.get_atoms(s))
    return results


def remove_all_na(atoms):
    """Remove all Na atoms for full desodiation."""
    symbols = atoms.get_chemical_symbols()
    na_indices = sorted([i for i, s in enumerate(symbols) if s == "Na"],
                        reverse=True)
    n_na = len(na_indices)
    new_atoms = atoms.copy()
    for idx in na_indices:
        del new_atoms[idx]
    return new_atoms, n_na


def run_nasicon_screening(device="cpu", dopants=None, n_sqs=5):
    """Screen dopants in Na3V2(PO4)3 for disorder sensitivity."""
    calc = get_calc(device)
    _, parent_struct = build_nasicon_cell()

    if dopants is None:
        dopants = DEFAULT_DOPANTS

    # Setup Drive
    if GDRIVE_DIR.parent.exists():
        GDRIVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Na3V2(PO4)3 NASICON SCREENING: {len(dopants)} dopants, {n_sqs} SQS each")
    print(f"  Predictor says: R_voltage=1.15 (UNSAFE), R_Ef=0 (SAFE)")
    print(f"{'='*70}\n")

    # Load checkpoint
    checkpoint = load_checkpoint()
    results = checkpoint.get("dopant_results", {}) if checkpoint else {}

    for dopant in dopants:
        if dopant in results and "error" not in results[dopant]:
            print(f"  [{dopant:4s}] CACHED -- skipping")
            continue

        t0 = time.time()
        try:
            # --- Ordered ---
            ordered_atoms = substitute_ordered(parent_struct, dopant)
            ordered_sod = relax_atoms(ordered_atoms, calc)
            ordered_desod_raw, n_na = remove_all_na(ordered_sod)
            ordered_desod = relax_atoms(ordered_desod_raw, calc)

            v_ord = -(float(ordered_desod.get_potential_energy()) -
                      float(ordered_sod.get_potential_energy())) / n_na
            ef_ord = float(ordered_sod.get_potential_energy()) / len(ordered_sod)
            vol_sod = float(ordered_sod.get_volume())
            vol_desod = float(ordered_desod.get_volume())
            dv_ord = abs(vol_desod - vol_sod) / vol_sod * 100

            # --- SQS ---
            sqs_structs = generate_sqs(parent_struct, dopant, n_sqs=n_sqs)
            sqs_voltages, sqs_efs, sqs_dvs = [], [], []

            for idx, sqs_atoms in enumerate(sqs_structs):
                try:
                    sqs_sod = relax_atoms(sqs_atoms, calc)
                    sqs_desod_raw, n_na_sqs = remove_all_na(sqs_sod)
                    sqs_desod = relax_atoms(sqs_desod_raw, calc)

                    v_sqs = -(float(sqs_desod.get_potential_energy()) -
                              float(sqs_sod.get_potential_energy())) / n_na_sqs
                    ef_sqs = float(sqs_sod.get_potential_energy()) / len(sqs_sod)
                    vol_s = float(sqs_sod.get_volume())
                    vol_d = float(sqs_desod.get_volume())
                    dv_sqs = abs(vol_d - vol_s) / vol_s * 100

                    sqs_voltages.append(v_sqs)
                    sqs_efs.append(ef_sqs)
                    sqs_dvs.append(dv_sqs)
                except Exception as e:
                    print(f"    SQS {idx+1} failed: {e}")

            dt = time.time() - t0
            results[dopant] = {
                "voltage_ordered": float(v_ord),
                "voltage_disordered_mean": float(np.mean(sqs_voltages)) if sqs_voltages else None,
                "voltage_disordered_std": float(np.std(sqs_voltages)) if sqs_voltages else None,
                "ef_ordered": float(ef_ord),
                "ef_disordered_mean": float(np.mean(sqs_efs)) if sqs_efs else None,
                "ef_disordered_std": float(np.std(sqs_efs)) if sqs_efs else None,
                "volume_change_ordered": float(dv_ord),
                "volume_change_disordered_mean": float(np.mean(sqs_dvs)) if sqs_dvs else None,
                "volume_change_disordered_std": float(np.std(sqs_dvs)) if sqs_dvs else None,
                "n_sqs_converged": len(sqs_voltages),
                "time_s": dt,
            }
            v_dis = f"{np.mean(sqs_voltages):.3f}+/-{np.std(sqs_voltages):.3f}" if sqs_voltages else "FAIL"
            print(f"  [{dopant:4s}] V_ord={v_ord:.3f}  V_dis={v_dis}  Ef_ord={ef_ord:.4f}  ({dt:.0f}s)")

        except Exception as e:
            dt = time.time() - t0
            print(f"  [{dopant:4s}] FAILED: {e}  ({dt:.0f}s)")
            results[dopant] = {"error": str(e), "time_s": dt}

        # Checkpoint after every dopant
        output = {
            "material": "Na3V2(PO4)3 (NASICON)",
            "structure": "R-3c (rhombohedral)",
            "mlip": "MACE-MP-0",
            "n_dopants": len(results),
            "n_sqs": n_sqs,
            "supercell": [1, 1, 1],
            "n_atoms": len(parent_struct),
            "v_sublattice": {
                "nn_dist": 4.42,
                "next_shell": 5.10,
                "anisotropy": 1.15,
                "na_transport_dim": "3D",
            },
            "predictor": {
                "R_voltage": 1.15,
                "R_Ef": 0.0,
                "R_volume": 1.15,
                "prediction_voltage": "UNSAFE",
                "prediction_Ef": "SAFE",
            },
            "dopant_results": results,
        }
        save_checkpoint(output)

    # --- Compute correlations ---
    print(f"\n{'='*70}")
    print(f"  Na3V2(PO4)3 NASICON RESULTS")
    print(f"{'='*70}\n")

    valid = {d: r for d, r in results.items()
             if "error" not in r and r.get("voltage_disordered_mean") is not None}

    if len(valid) >= 4:
        v_ord = [r["voltage_ordered"] for r in valid.values()]
        v_dis = [r["voltage_disordered_mean"] for r in valid.values()]
        rho_v, p_v = stats.spearmanr(v_ord, v_dis)

        ef_ord = [r["ef_ordered"] for r in valid.values()]
        ef_dis = [r["ef_disordered_mean"] for r in valid.values()]
        rho_ef, p_ef = stats.spearmanr(ef_ord, ef_dis)

        dv_valid = {d: r for d, r in valid.items()
                    if r.get("volume_change_disordered_mean") is not None}
        if len(dv_valid) >= 4:
            dv_ord = [r["volume_change_ordered"] for r in dv_valid.values()]
            dv_dis = [r["volume_change_disordered_mean"] for r in dv_valid.values()]
            rho_dv, p_dv = stats.spearmanr(dv_ord, dv_dis)
        else:
            rho_dv, p_dv = None, None

        print(f"  Voltage rho:          {rho_v:+.3f} (p={p_v:.4f}, n={len(valid)})")
        print(f"  Formation energy rho: {rho_ef:+.3f} (p={p_ef:.4f}, n={len(valid)})")
        if rho_dv is not None:
            print(f"  Volume change rho:    {rho_dv:+.3f} (p={p_dv:.4f}, n={len(dv_valid)})")

        print(f"\n  --- PREDICTOR VALIDATION ---")
        print(f"  Predicted: voltage UNSAFE (R=1.15), Ef SAFE (R=0)")
        print(f"  Actual:    voltage rho={rho_v:+.3f}, Ef rho={rho_ef:+.3f}")

        if rho_v < 0.50:
            print(f"  [OK] Voltage prediction CORRECT (rho < 0.50, predictor said UNSAFE)")
        else:
            print(f"  [!!] Voltage prediction WRONG (rho >= 0.50, predictor said UNSAFE)")
            print(f"    -> Predictor is too conservative for NASICON")

        if rho_ef >= 0.50:
            print(f"  [OK] Ef prediction CORRECT (rho >= 0.50, predictor said SAFE)")
        else:
            print(f"  [!!] Ef prediction WRONG (rho < 0.50, predictor said SAFE)")
            print(f"    -> DANGER: predictor missed an unsafe case!")

        # Save final correlations
        output["voltage_rho"] = float(rho_v)
        output["voltage_p"] = float(p_v)
        output["ef_rho"] = float(rho_ef)
        output["ef_p"] = float(p_ef)
        if rho_dv is not None:
            output["volume_change_rho"] = float(rho_dv)
            output["volume_change_p"] = float(p_dv)
        save_checkpoint(output)

    else:
        print(f"  Insufficient data for correlation (n={len(valid)})")

    print(f"\n  Results: {LOCAL_OUT}")
    if GDRIVE_DIR.exists():
        print(f"  Drive:   {GDRIVE_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Na3V2(PO4)3 NASICON disorder screening")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dopants", default=None, help="Comma-separated dopant list")
    parser.add_argument("--n-sqs", type=int, default=5)
    args = parser.parse_args()

    dopants = args.dopants.split(",") if args.dopants else None
    run_nasicon_screening(device=args.device, dopants=dopants, n_sqs=args.n_sqs)
