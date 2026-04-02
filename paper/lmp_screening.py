#!/usr/bin/env python3
"""
LiMnPO4 olivine dopant screening: out-of-sample test for disorder predictor.
=============================================================================

LiMnPO4 (Pnma olivine) has 1D Li transport channels — a dimensionality class
not present in our training set (layered 2D, spinel/perovskite/fluorite 3D).

Mn sublattice anisotropy = 1.21 (d2/d1 = 4.74/3.92 A)
Predictor says: R_voltage = 1.21 (UNSAFE), R_Ef = 0 (SAFE)

This is a genuinely uncertain prediction — if voltage rankings are preserved
despite R > 1.0, the predictor needs recalibration; if destroyed, it's validated.

Per-dopant checkpointing to Google Drive.

Usage (Colab A100):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install mace-torch ase pymatgen scipy
    !git clone https://github.com/snehalnair/disorder-screening-agent.git
    %cd disorder-screening-agent
    !python paper/lmp_screening.py --device cuda

Usage (CPU, quick test):
    !python paper/lmp_screening.py --device cpu --dopants Al,Mg,Co,Ni,Fe,Ti
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
GDRIVE_OUT = GDRIVE_DIR / "lmp_screening_results.json"
LOCAL_OUT = SCRIPT_DIR / "lmp_screening_results.json"

# LiMnPO4 dopants: common Mn-site substituents from literature
# Filtered for elements likely to pass SMACT + Shannon radius checks
# Mn2+ octahedral (CN=6, r=0.83 A) -> candidates within ~35% radius mismatch
DEFAULT_DOPANTS = [
    "Al", "Co", "Cr", "Cu", "Ga", "Ge", "Mg", "Fe", "Mo", "Nb",
    "Ni", "Ru", "Sc", "Sn", "Ti", "V", "Zn", "Zr",
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


def build_lmp_supercell(supercell_size=(2, 2, 2)):
    """Build LiMnPO4 supercell from parent CIF.

    Returns (ase_atoms, pymatgen_struct).
    Unit cell: 28 atoms (4 LiMnPO4), supercell 2x2x2: 224 atoms (32 Mn sites).
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    cif = DATA_DIR / "limnpo4_parent.cif"
    if not cif.exists():
        raise FileNotFoundError(f"LiMnPO4 CIF not found: {cif}")

    struct = Structure.from_file(str(cif))
    struct.make_supercell(list(supercell_size))

    mn_indices = [i for i, sp in enumerate(struct.species) if str(sp) == "Mn"]
    li_indices = [i for i, sp in enumerate(struct.species) if str(sp) == "Li"]
    print(f"LiMnPO4 supercell: {len(struct)} atoms, {len(mn_indices)} Mn, {len(li_indices)} Li")

    return AseAtomsAdaptor.get_atoms(struct), struct


def substitute_ordered(struct, dopant, target="Mn", n_dopant=4):
    """Farthest-first substitution at Mn sites (ordered baseline).

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


def generate_sqs(struct, dopant, target="Mn", n_sqs=5, n_dopant=4, seed=42):
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


def remove_all_li(atoms):
    """Remove all Li atoms for full delithiation."""
    symbols = atoms.get_chemical_symbols()
    li_indices = sorted([i for i, s in enumerate(symbols) if s == "Li"],
                        reverse=True)
    n_li = len(li_indices)
    new_atoms = atoms.copy()
    for idx in li_indices:
        del new_atoms[idx]
    return new_atoms, n_li


def run_lmp_screening(device="cpu", dopants=None, n_sqs=5):
    """Screen dopants in LiMnPO4 for disorder sensitivity."""
    calc = get_calc(device)
    _, parent_struct = build_lmp_supercell()

    if dopants is None:
        dopants = DEFAULT_DOPANTS

    # Setup Drive
    if GDRIVE_DIR.parent.exists():
        GDRIVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  LiMnPO4 SCREENING: {len(dopants)} dopants, {n_sqs} SQS each")
    print(f"  Predictor says: R_voltage=1.21 (UNSAFE), R_Ef=0 (SAFE)")
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
            ordered_lith = relax_atoms(ordered_atoms, calc)
            ordered_delith_raw, n_li = remove_all_li(ordered_lith)
            ordered_delith = relax_atoms(ordered_delith_raw, calc)

            v_ord = -(float(ordered_delith.get_potential_energy()) -
                      float(ordered_lith.get_potential_energy())) / n_li
            ef_ord = float(ordered_lith.get_potential_energy()) / len(ordered_lith)
            vol_lith = float(ordered_lith.get_volume())
            vol_delith = float(ordered_delith.get_volume())
            dv_ord = abs(vol_delith - vol_lith) / vol_lith * 100

            # --- SQS ---
            sqs_structs = generate_sqs(parent_struct, dopant, n_sqs=n_sqs)
            sqs_voltages, sqs_efs, sqs_dvs = [], [], []

            for idx, sqs_atoms in enumerate(sqs_structs):
                try:
                    sqs_lith = relax_atoms(sqs_atoms, calc)
                    sqs_delith_raw, n_li_sqs = remove_all_li(sqs_lith)
                    sqs_delith = relax_atoms(sqs_delith_raw, calc)

                    v_sqs = -(float(sqs_delith.get_potential_energy()) -
                              float(sqs_lith.get_potential_energy())) / n_li_sqs
                    ef_sqs = float(sqs_lith.get_potential_energy()) / len(sqs_lith)
                    vol_l = float(sqs_lith.get_volume())
                    vol_d = float(sqs_delith.get_volume())
                    dv_sqs = abs(vol_d - vol_l) / vol_l * 100

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
            "material": "LiMnPO4 (olivine)",
            "structure": "Pnma (orthorhombic)",
            "mlip": "MACE-MP-0",
            "n_dopants": len(results),
            "n_sqs": n_sqs,
            "supercell": [2, 2, 2],
            "n_atoms": len(parent_struct),
            "mn_sublattice": {
                "nn_dist": 3.92,
                "next_shell": 4.74,
                "anisotropy": 1.21,
                "li_transport_dim": "1D",
            },
            "predictor": {
                "R_voltage": 1.21,
                "R_Ef": 0.0,
                "R_volume": 1.21,
                "prediction_voltage": "UNSAFE",
                "prediction_Ef": "SAFE",
            },
            "dopant_results": results,
        }
        save_checkpoint(output)

    # --- Compute correlations ---
    print(f"\n{'='*70}")
    print(f"  LiMnPO4 RESULTS")
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
        print(f"  Predicted: voltage UNSAFE (R=1.21), Ef SAFE (R=0)")
        print(f"  Actual:    voltage rho={rho_v:+.3f}, Ef rho={rho_ef:+.3f}")

        if rho_v < 0.50:
            print(f"  [PASS] Voltage prediction CORRECT (rho < 0.50, predictor said UNSAFE)")
        else:
            print(f"  [FAIL] Voltage prediction WRONG (rho >= 0.50, predictor said UNSAFE)")
            print(f"    -> Predictor is too conservative for olivine")

        if rho_ef >= 0.50:
            print(f"  [PASS] Ef prediction CORRECT (rho >= 0.50, predictor said SAFE)")
        else:
            print(f"  [FAIL] Ef prediction WRONG (rho < 0.50, predictor said SAFE)")
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
    parser = argparse.ArgumentParser(description="LiMnPO4 olivine disorder screening")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dopants", default=None, help="Comma-separated dopant list")
    parser.add_argument("--n-sqs", type=int, default=5)
    args = parser.parse_args()

    dopants = args.dopants.split(",") if args.dopants else None
    run_lmp_screening(device=args.device, dopants=dopants, n_sqs=args.n_sqs)
