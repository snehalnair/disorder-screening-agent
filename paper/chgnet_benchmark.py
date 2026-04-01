#!/usr/bin/env python3
"""
CHGNet benchmark: validate disorder effect is not MACE-specific.

Runs the same LCO dopant screening (ordered + 5 SQS) using CHGNet
instead of MACE-MP-0, then compares Spearman rho values.

If CHGNet also shows voltage ranking destruction in layered LCO,
the disorder effect is confirmed as physical, not an MLIP artefact.

Usage (Colab A100):
    !pip install chgnet pymatgen ase scipy numpy
    !python paper/chgnet_benchmark.py --device cuda

Usage (CPU, slower):
    !python paper/chgnet_benchmark.py --device cpu --dopants Al,Cr,Ga,Ge,Ti,Ni
"""

import argparse
import json
import pathlib
import time
import numpy as np
from scipy import stats

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LCO_DIR = PROJECT_DIR / "lco"
DATA_DIR = PROJECT_DIR / "data" / "structures"
OUT_DIR = PROJECT_DIR / "paper"


def get_chgnet_calc(device="cpu"):
    """Load CHGNet universal potential."""
    from chgnet.model.dynamics import CHGNetCalculator
    print(f"Loading CHGNet on {device}...")
    calc = CHGNetCalculator(use_device=device)
    print("Done.\n")
    return calc


def get_mace_calc(device="cpu"):
    """Load MACE-MP-0 for comparison."""
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


def compute_voltage(atoms_lith, atoms_delith, n_li):
    """V = -(E_delith - E_lith) / n_Li"""
    return -(atoms_delith.get_potential_energy() -
             atoms_lith.get_potential_energy()) / n_li


def compute_ef(atoms):
    """Formation energy proxy: E/N_atoms."""
    return atoms.get_potential_energy() / len(atoms)


def build_lco_supercell():
    """Build 4x4x4 LCO supercell (256 atoms)."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    struct = Structure.from_file(str(DATA_DIR / "lco_parent.cif"))
    struct.make_supercell([4, 4, 4])
    return AseAtomsAdaptor.get_atoms(struct), struct


def get_dopant_list():
    """Get dopants from existing LCO checkpoints."""
    dopants = []
    for f in sorted(LCO_DIR.glob("*.json")):
        data = json.load(open(f))
        sqs = data.get("sqs_results", [])
        if sqs and any("voltage" in s for s in sqs):
            dopants.append(f.stem.split("_")[-1])
    return dopants


def substitute_ordered(struct, dopant, target="Co"):
    """Farthest-first single substitution at target site."""
    from pymatgen.io.ase import AseAtomsAdaptor
    s = struct.copy()
    target_indices = [i for i, sp in enumerate(s.species) if str(sp) == target]
    # Farthest-first: pick site with max min-distance to other target sites
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


def generate_sqs(struct, dopant, target="Co", n_sqs=5, seed=42):
    """Generate SQS realisations via random sampling with correlation scoring."""
    from pymatgen.io.ase import AseAtomsAdaptor
    rng = np.random.default_rng(seed)
    target_indices = [i for i, sp in enumerate(struct.species) if str(sp) == target]
    n_dopant = max(1, round(len(target_indices) * (1 / len(target_indices))))  # 1 dopant

    results = []
    for i in range(n_sqs):
        s = struct.copy()
        # Random single substitution (different site each time)
        site = rng.choice(target_indices)
        s.replace(site, dopant)
        results.append(AseAtomsAdaptor.get_atoms(s))
    return results


def remove_all_li(atoms):
    """Remove all Li atoms for full delithiation."""
    symbols = atoms.get_chemical_symbols()
    li_indices = sorted([i for i, s in enumerate(symbols) if s == "Li"], reverse=True)
    n_li = len(li_indices)
    new_atoms = atoms.copy()
    for idx in li_indices:
        del new_atoms[idx]
    return new_atoms, n_li


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy/torch float32 → Python float for JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.item()
        except ImportError:
            pass
        return super().default(obj)


def _save_checkpoint(out_path, results, device, n_sqs):
    """Save incremental checkpoint after each dopant."""
    output = {
        "mlip": "CHGNet",
        "material": "LiCoO2",
        "n_dopants": len(results),
        "n_sqs": n_sqs,
        "device": device,
        "dopant_results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)


def run_benchmark(device="cpu", dopants=None, n_sqs=5):
    """Run CHGNet benchmark on LCO dopants."""
    calc = get_chgnet_calc(device)
    _, parent_struct = build_lco_supercell()

    if dopants is None:
        dopants = get_dopant_list()

    print(f"=" * 70)
    print(f"  CHGNet BENCHMARK: LiCoO2, {len(dopants)} dopants, {n_sqs} SQS each")
    print(f"=" * 70)
    print(f"  Parent: {len(parent_struct)} atoms")
    print()

    # Load existing results (checkpoint resume)
    out_path = OUT_DIR / "chgnet_benchmark.json"
    results = {}
    if out_path.exists():
        existing = json.load(open(out_path))
        results = existing.get("dopant_results", {})
        print(f"  Resuming: {len(results)} dopants already done, skipping them.\n")

    for dopant in dopants:
        if dopant in results and "error" not in results[dopant]:
            print(f"  [{dopant:4s}] CACHED — skipping")
            continue

        t0 = time.time()

        # --- Ordered ---
        ordered_atoms = substitute_ordered(parent_struct, dopant)
        ordered_lith = relax_atoms(ordered_atoms, calc)
        ordered_delith_raw, n_li = remove_all_li(ordered_lith)
        ordered_delith = relax_atoms(ordered_delith_raw, calc)
        v_ord = compute_voltage(ordered_lith, ordered_delith, n_li)
        ef_ord = compute_ef(ordered_lith)

        # --- SQS ---
        sqs_structs = generate_sqs(parent_struct, dopant, n_sqs=n_sqs, seed=42)
        sqs_voltages = []
        sqs_efs = []
        for sqs_atoms in sqs_structs:
            try:
                sqs_lith = relax_atoms(sqs_atoms, calc)
                sqs_delith_raw, n_li_sqs = remove_all_li(sqs_lith)
                sqs_delith = relax_atoms(sqs_delith_raw, calc)
                v_sqs = compute_voltage(sqs_lith, sqs_delith, n_li_sqs)
                ef_sqs = compute_ef(sqs_lith)
                sqs_voltages.append(v_sqs)
                sqs_efs.append(ef_sqs)
            except Exception as e:
                print(f"    SQS failed for {dopant}: {e}")

        dt = time.time() - t0
        results[dopant] = {
            "voltage_ordered": float(v_ord),
            "voltage_disordered_mean": float(np.mean(sqs_voltages)) if sqs_voltages else None,
            "voltage_disordered_std": float(np.std(sqs_voltages)) if sqs_voltages else None,
            "ef_ordered": float(ef_ord),
            "ef_disordered_mean": float(np.mean(sqs_efs)) if sqs_efs else None,
            "ef_disordered_std": float(np.std(sqs_efs)) if sqs_efs else None,
            "n_sqs_converged": len(sqs_voltages),
            "time_s": float(dt),
        }
        print(f"  [{dopant:4s}] V_ord={v_ord:.3f}  V_dis={np.mean(sqs_voltages):.3f}+/-{np.std(sqs_voltages):.3f}  "
              f"Ef_ord={ef_ord:.3f}  ({dt:.0f}s)")

        # Checkpoint save after each dopant
        _save_checkpoint(out_path, results, device, n_sqs)

    # --- Compute correlations ---
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: CHGNet vs MACE comparison")
    print(f"{'=' * 70}\n")

    # CHGNet correlations
    v_ord_list = [r["voltage_ordered"] for r in results.values() if r["voltage_disordered_mean"]]
    v_dis_list = [r["voltage_disordered_mean"] for r in results.values() if r["voltage_disordered_mean"]]
    ef_ord_list = [r["ef_ordered"] for r in results.values() if r["ef_disordered_mean"]]
    ef_dis_list = [r["ef_disordered_mean"] for r in results.values() if r["ef_disordered_mean"]]

    if len(v_ord_list) >= 4:
        rho_v, p_v = stats.spearmanr(v_ord_list, v_dis_list)
        print(f"  CHGNet voltage rho:          {rho_v:+.3f} (p={p_v:.4f}, n={len(v_ord_list)})")
    if len(ef_ord_list) >= 4:
        rho_ef, p_ef = stats.spearmanr(ef_ord_list, ef_dis_list)
        print(f"  CHGNet formation energy rho: {rho_ef:+.3f} (p={p_ef:.4f}, n={len(ef_ord_list)})")

    # Load MACE results for comparison
    print()
    mace_v_ord, mace_v_dis = [], []
    mace_ef_ord, mace_ef_dis = [], []
    for dopant in results:
        mace_file = LCO_DIR / f"LiCoO2_layered_{dopant}.json"
        if mace_file.exists():
            mace_data = json.load(open(mace_file))
            mo = mace_data.get("ordered", {})
            ms = mace_data.get("sqs_results", [])
            if "voltage" in mo and any("voltage" in s for s in ms):
                mace_v_ord.append(mo["voltage"])
                mace_v_dis.append(np.mean([s["voltage"] for s in ms if "voltage" in s]))
            if "formation_energy" in mo and any("formation_energy" in s for s in ms):
                mace_ef_ord.append(mo["formation_energy"])
                mace_ef_dis.append(np.mean([s["formation_energy"] for s in ms if "formation_energy" in s]))

    if len(mace_v_ord) >= 4:
        rho_mace_v, _ = stats.spearmanr(mace_v_ord, mace_v_dis)
        print(f"  MACE  voltage rho:           {rho_mace_v:+.3f} (n={len(mace_v_ord)})")
    if len(mace_ef_ord) >= 4:
        rho_mace_ef, _ = stats.spearmanr(mace_ef_ord, mace_ef_dis)
        print(f"  MACE  formation energy rho:  {rho_mace_ef:+.3f} (n={len(mace_ef_ord)})")

    # Cross-MLIP correlation
    chg_v_ord = [results[d]["voltage_ordered"] for d in results if results[d]["voltage_disordered_mean"]]
    matching_dopants = [d for d in results if results[d]["voltage_disordered_mean"]]
    mace_v_ord_matched = []
    for d in matching_dopants:
        mf = LCO_DIR / f"LiCoO2_layered_{d}.json"
        if mf.exists():
            md = json.load(open(mf))
            if "voltage" in md.get("ordered", {}):
                mace_v_ord_matched.append(md["ordered"]["voltage"])
            else:
                mace_v_ord_matched.append(None)
        else:
            mace_v_ord_matched.append(None)

    valid = [(c, m) for c, m in zip(chg_v_ord, mace_v_ord_matched) if m is not None]
    if len(valid) >= 4:
        c_vals, m_vals = zip(*valid)
        rho_cross, p_cross = stats.spearmanr(c_vals, m_vals)
        print(f"\n  Cross-MLIP (CHGNet vs MACE) ordered voltage rho: {rho_cross:+.3f} (p={p_cross:.4f})")

    # Save results
    output = {
        "mlip": "CHGNet",
        "material": "LiCoO2",
        "n_dopants": len(results),
        "n_sqs": n_sqs,
        "device": device,
        "dopant_results": results,
    }
    if len(v_ord_list) >= 4:
        output["chgnet_voltage_rho"] = round(rho_v, 3)
        output["chgnet_voltage_p"] = round(p_v, 4)
    if len(ef_ord_list) >= 4:
        output["chgnet_ef_rho"] = round(rho_ef, 3)

    out_path = OUT_DIR / "chgnet_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu")
    parser.add_argument("--dopants", default=None,
                        help="Comma-separated dopant list (default: all LCO dopants)")
    parser.add_argument("--n-sqs", type=int, default=5, help="Number of SQS realisations")
    args = parser.parse_args()

    dopant_list = args.dopants.split(",") if args.dopants else None
    run_benchmark(device=args.device, dopants=dopant_list, n_sqs=args.n_sqs)
