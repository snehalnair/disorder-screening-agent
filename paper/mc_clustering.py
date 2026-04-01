#!/usr/bin/env python3
"""
Monte Carlo dopant-clustering analysis at synthesis temperature.

Tests whether dopants remain dispersed (as assumed in our ordered screening)
or cluster at typical synthesis temperatures (800–1000 °C).

Physics: lattice MC on the TM sublattice with MACE pair energies.
  - E_pair(i,j) = energy penalty for dopant at site i, host at site j
  - Metropolis sampling at T_synth
  - Measure: Warren–Cowley short-range order parameter α₁
    α₁ < 0 → ordering (dopant avoids dopant neighbours)
    α₁ ~ 0 → random (SQS-like)
    α₁ > 0 → clustering (dopants aggregate)

If α₁ ~ 0 at synthesis T for most dopants, our random SQS assumption
is physically justified.  If α₁ ≠ 0, our SQS realisations don't capture
the thermodynamically preferred configurations.

Usage (Colab A100):
    !pip install mace-torch pymatgen ase scipy numpy
    !python paper/mc_clustering.py --device cuda

Usage (CPU, subset):
    !python paper/mc_clustering.py --device cpu --dopants Al,Ti,Mg --mc-steps 5000
"""

import argparse
import json
import pathlib
import time
import numpy as np

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


def build_supercell(material="lco"):
    """Build parent supercell."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    cif_map = {
        "lco": ("lco_parent.cif", [3, 3, 3]),   # 108 atoms, 27 Co sites
        "lno": ("lno_parent.cif", [3, 3, 3]),
        "lmo": ("lmo_spinel.cif", [2, 2, 2]),
    }
    cif_name, sc_size = cif_map[material]
    struct = Structure.from_file(str(DATA_DIR / cif_name))
    struct.make_supercell(sc_size)
    return struct


def get_tm_sublattice(struct, target="Co"):
    """Extract TM sublattice: site indices and neighbour list."""
    target_indices = [i for i, sp in enumerate(struct.species) if str(sp) == target]
    n_tm = len(target_indices)

    # Build neighbour list for TM sublattice (nearest TM-TM distance)
    # In layered R-3m, TM-TM nearest-neighbour ~ 2.8 Å
    dists = []
    for i, idx_i in enumerate(target_indices):
        for j, idx_j in enumerate(target_indices):
            if i < j:
                d = struct.get_distance(idx_i, idx_j)
                dists.append(d)

    dists = np.array(dists)
    # First shell: distances within 10% of minimum
    d_min = dists.min()
    cutoff = d_min * 1.15  # 15% tolerance for first shell
    print(f"  TM-TM nearest distance: {d_min:.3f} Å, cutoff: {cutoff:.3f} Å")

    # Build adjacency for TM sublattice
    neighbours = {i: [] for i in range(n_tm)}
    for i in range(n_tm):
        for j in range(n_tm):
            if i != j:
                d = struct.get_distance(target_indices[i], target_indices[j])
                if d < cutoff:
                    neighbours[i].append(j)

    n_nn = np.mean([len(neighbours[i]) for i in range(n_tm)])
    print(f"  TM sites: {n_tm}, avg nearest neighbours: {n_nn:.1f}")
    return target_indices, neighbours, n_nn


def compute_pair_energy(struct, calc, site_idx, dopant, target="Co"):
    """Compute single-site substitution energy using MACE.

    E_sub(site) = E(struct with dopant at site) - E(pristine struct)
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    # Pristine energy (cache this externally for efficiency)
    atoms_pristine = AseAtomsAdaptor.get_atoms(struct)
    atoms_pristine.calc = calc
    e_pristine = atoms_pristine.get_potential_energy()

    # Substituted energy
    s_doped = struct.copy()
    s_doped.replace(site_idx, dopant)
    atoms_doped = AseAtomsAdaptor.get_atoms(s_doped)
    atoms_doped.calc = calc
    e_doped = atoms_doped.get_potential_energy()

    return e_doped - e_pristine


def compute_pair_interaction(struct, calc, site_i, site_j, dopant, target="Co"):
    """Compute dopant-dopant pair interaction energy.

    E_int = E(both doped) - E(site_i doped) - E(site_j doped) + E(pristine)

    E_int < 0 → dopants attract (clustering tendency)
    E_int > 0 → dopants repel (ordering tendency)
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    # E(pristine)
    atoms_0 = AseAtomsAdaptor.get_atoms(struct)
    atoms_0.calc = calc
    e_0 = atoms_0.get_potential_energy()

    # E(site_i doped)
    s_i = struct.copy()
    s_i.replace(site_i, dopant)
    atoms_i = AseAtomsAdaptor.get_atoms(s_i)
    atoms_i.calc = calc
    e_i = atoms_i.get_potential_energy()

    # E(site_j doped)
    s_j = struct.copy()
    s_j.replace(site_j, dopant)
    atoms_j = AseAtomsAdaptor.get_atoms(s_j)
    atoms_j.calc = calc
    e_j = atoms_j.get_potential_energy()

    # E(both doped)
    s_ij = struct.copy()
    s_ij.replace(site_i, dopant)
    s_ij.replace(site_j, dopant)
    atoms_ij = AseAtomsAdaptor.get_atoms(s_ij)
    atoms_ij.calc = calc
    e_ij = atoms_ij.get_potential_energy()

    e_int = e_ij - e_i - e_j + e_0
    return e_int


def run_lattice_mc(e_nn, n_tm, n_dopant, neighbours, temperatures, mc_steps=20000, seed=42):
    """Run lattice Monte Carlo on the TM sublattice.

    Simple Ising-like model:
      - Binary occupation: σ_i = 1 (dopant) or 0 (host)
      - H = e_nn * Σ_{<i,j>} σ_i * σ_j
      - e_nn is the nearest-neighbour pair interaction energy

    Returns Warren-Cowley α₁ at each temperature.
    """
    rng = np.random.default_rng(seed)
    kB = 8.617333e-5  # eV/K

    results_by_T = {}

    for T in temperatures:
        beta = 1.0 / (kB * T) if T > 0 else float('inf')

        # Initialize: random placement of n_dopant atoms
        occupation = np.zeros(n_tm, dtype=int)
        dopant_sites = rng.choice(n_tm, size=n_dopant, replace=False)
        occupation[dopant_sites] = 1

        # MC loop: swap a random dopant with a random host
        n_accept = 0
        alpha_samples = []

        for step in range(mc_steps):
            # Pick a random dopant site and a random host site
            dop_sites = np.where(occupation == 1)[0]
            host_sites = np.where(occupation == 0)[0]
            if len(dop_sites) == 0 or len(host_sites) == 0:
                break

            i = rng.choice(dop_sites)   # dopant site
            j = rng.choice(host_sites)  # host site

            # Energy change from swapping i↔j
            # ΔE = e_nn * (change in dopant-dopant neighbour count)
            # Count dopant neighbours of i (excluding j) and j (excluding i)
            nn_i_dopant = sum(1 for k in neighbours[i] if occupation[k] == 1 and k != j)
            nn_j_dopant = sum(1 for k in neighbours[j] if occupation[k] == 1 and k != i)
            # After swap: i becomes host, j becomes dopant
            # Lost pairs: nn_i_dopant (i was dopant, had these dopant neighbours)
            # Gained pairs: nn_j_dopant (j becomes dopant, gains these dopant neighbours)
            delta_pairs = nn_j_dopant - nn_i_dopant
            delta_E = e_nn * delta_pairs

            # Metropolis criterion
            if delta_E <= 0 or rng.random() < np.exp(-beta * delta_E):
                occupation[i] = 0
                occupation[j] = 1
                n_accept += 1

            # Sample α₁ every 100 steps after burn-in
            if step > mc_steps // 4 and step % 100 == 0:
                alpha = compute_warren_cowley(occupation, neighbours, n_dopant, n_tm)
                alpha_samples.append(alpha)

        alpha_mean = np.mean(alpha_samples) if alpha_samples else 0.0
        alpha_std = np.std(alpha_samples) if alpha_samples else 0.0
        accept_rate = n_accept / mc_steps

        results_by_T[T] = {
            "alpha_mean": float(alpha_mean),
            "alpha_std": float(alpha_std),
            "accept_rate": float(accept_rate),
            "n_samples": len(alpha_samples),
        }

    return results_by_T


def compute_warren_cowley(occupation, neighbours, n_dopant, n_tm):
    """Compute Warren-Cowley short-range order parameter α₁.

    α₁ = 1 - P(AB) / x_B
    where P(AB) = fraction of A's neighbours that are B
    x_B = global concentration of B (dopant)

    α₁ = 0 → random
    α₁ < 0 → unlike neighbours preferred (ordering)
    α₁ > 0 → like neighbours preferred (clustering)
    """
    x_dop = n_dopant / n_tm
    if x_dop == 0 or x_dop == 1:
        return 0.0

    # For each dopant site, count fraction of neighbours that are host
    # P(dopant-host) = avg over dopant sites of (host neighbours / total neighbours)
    p_dh = 0.0
    n_counted = 0
    for i in range(n_tm):
        if occupation[i] == 1:  # dopant site
            nn = neighbours[i]
            if len(nn) > 0:
                n_host_nn = sum(1 for k in nn if occupation[k] == 0)
                p_dh += n_host_nn / len(nn)
                n_counted += 1

    if n_counted == 0:
        return 0.0

    p_dh /= n_counted  # average P(dopant has host neighbour)
    x_host = 1.0 - x_dop
    alpha = 1.0 - p_dh / x_host

    return alpha


def run_clustering_analysis(device="cpu", dopants=None, mc_steps=20000):
    """Run MC clustering analysis for LCO dopants."""
    calc = get_calc(device)
    struct = build_supercell("lco")
    target = "Co"
    tm_indices, neighbours, n_nn = get_tm_sublattice(struct, target)
    n_tm = len(tm_indices)

    # Default dopants (common LCO dopants)
    if dopants is None:
        dopants = ["Al", "Ti", "Mg", "Zr", "Nb", "Fe", "Cr", "Ga",
                    "Ni", "Mn", "V", "Cu", "W", "Mo", "Ge", "Sn"]

    # Temperatures: typical LCO synthesis range
    temperatures = [300, 600, 800, 900, 1000, 1100, 1200]

    # Number of dopants in MC: ~3-4% concentration (1 dopant in 27 TM sites)
    n_dopant = max(1, round(n_tm * 0.037))  # 1 in 27 ≈ 3.7%

    print(f"\n{'=' * 70}")
    print(f"  MONTE CARLO CLUSTERING ANALYSIS")
    print(f"  Material: LiCoO2, {n_tm} TM sites, {n_dopant} dopant(s)")
    print(f"  Temperatures: {temperatures} K")
    print(f"  MC steps: {mc_steps}")
    print(f"{'=' * 70}\n")

    # Load existing results (checkpoint resume)
    mc_out_path = OUT_DIR / "mc_clustering_results.json"
    all_results = {}
    if mc_out_path.exists():
        existing = json.load(open(mc_out_path))
        all_results = existing.get("dopant_results", {})
        print(f"  Resuming: {len(all_results)} dopants already done, skipping them.\n")

    for dopant in dopants:
        if dopant in all_results and "error" not in all_results[dopant]:
            print(f"  [{dopant:4s}] CACHED — skipping")
            continue

        t0 = time.time()
        print(f"  [{dopant:4s}] Computing pair interaction... ", end="", flush=True)

        try:
            # Find nearest-neighbour TM pair
            i0 = tm_indices[0]
            nn_of_0 = neighbours[0]
            if not nn_of_0:
                print("SKIP (no neighbours)")
                continue
            j0 = tm_indices[nn_of_0[0]]

            # Compute NN pair interaction energy
            e_nn = compute_pair_interaction(struct, calc, i0, j0, dopant, target)
            print(f"E_nn = {e_nn*1000:.1f} meV  ", end="", flush=True)

            # Run lattice MC at each temperature
            mc_results = run_lattice_mc(
                e_nn, n_tm, n_dopant, neighbours,
                temperatures, mc_steps=mc_steps, seed=42
            )

            dt = time.time() - t0

            # Summary
            alpha_800 = mc_results.get(800, {}).get("alpha_mean", None)
            alpha_1000 = mc_results.get(1000, {}).get("alpha_mean", None)
            print(f"α₁(800K)={alpha_800:+.3f}  α₁(1000K)={alpha_1000:+.3f}  ({dt:.0f}s)")

            all_results[dopant] = {
                "e_nn_eV": float(e_nn),
                "e_nn_meV": float(e_nn * 1000),
                "mc_results": mc_results,
                "time_s": dt,
            }

        except Exception as e:
            dt = time.time() - t0
            print(f"FAILED: {e}  ({dt:.0f}s)")
            all_results[dopant] = {"error": str(e), "time_s": dt}

        # Checkpoint save after each dopant
        _ckpt = {
            "material": "LiCoO2",
            "mlip": "MACE-MP-0",
            "n_tm_sites": n_tm,
            "mc_steps": mc_steps,
            "dopant_results": all_results,
        }
        with open(mc_out_path, "w") as f:
            json.dump(_ckpt, f, indent=2)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  CLUSTERING SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"  {'Dopant':>6s}  {'E_nn (meV)':>10s}  {'α₁(800K)':>10s}  {'α₁(1000K)':>10s}  {'Tendency':>12s}")
    print(f"  {'-'*6:>6s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*12:>12s}")

    clustering = []
    ordering = []
    random_like = []

    for dopant, res in all_results.items():
        if "error" in res:
            continue
        e_nn = res["e_nn_meV"]
        a800 = res["mc_results"].get(800, {}).get("alpha_mean", 0)
        a1000 = res["mc_results"].get(1000, {}).get("alpha_mean", 0)

        if a1000 > 0.05:
            tendency = "CLUSTERING"
            clustering.append(dopant)
        elif a1000 < -0.05:
            tendency = "ORDERING"
            ordering.append(dopant)
        else:
            tendency = "random"
            random_like.append(dopant)

        print(f"  {dopant:>6s}  {e_nn:>+10.1f}  {a800:>+10.3f}  {a1000:>+10.3f}  {tendency:>12s}")

    print(f"\n  Random-like (SQS valid):   {len(random_like)}/{len(all_results)} — {', '.join(random_like)}")
    print(f"  Ordering (α₁<-0.05):       {len(ordering)}/{len(all_results)} — {', '.join(ordering)}")
    print(f"  Clustering (α₁>+0.05):     {len(clustering)}/{len(all_results)} — {', '.join(clustering)}")

    frac_random = len(random_like) / max(1, len(random_like) + len(ordering) + len(clustering))
    print(f"\n  → {frac_random*100:.0f}% of dopants are SQS-compatible at synthesis T")

    # Save
    output = {
        "material": "LiCoO2",
        "structure": "layered R-3m",
        "mlip": "MACE-MP-0",
        "n_tm_sites": n_tm,
        "n_dopant_in_mc": n_dopant,
        "mc_steps": mc_steps,
        "temperatures_K": temperatures,
        "n_dopants_screened": len(all_results),
        "fraction_random_like": round(frac_random, 3),
        "clustering_dopants": clustering,
        "ordering_dopants": ordering,
        "random_dopants": random_like,
        "dopant_results": all_results,
    }

    out_path = OUT_DIR / "mc_clustering_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu")
    parser.add_argument("--dopants", default=None,
                        help="Comma-separated dopant list")
    parser.add_argument("--mc-steps", type=int, default=20000,
                        help="MC steps per temperature")
    args = parser.parse_args()

    dopant_list = args.dopants.split(",") if args.dopants else None
    run_clustering_analysis(device=args.device, dopants=dopant_list,
                           mc_steps=args.mc_steps)
