#!/usr/bin/env python3
"""
DFT (Quantum ESPRESSO) validation of dopant–dopant interaction energy sign.

Generates 4 QE input files for E_int at the NN distance in LiCoO₂:
  E_int = E(AB) - E(A) - E(B) + E(0)

Uses the same 3×3×2 supercell (72 atoms) as the MACE calculation,
with single-point SCF (no relaxation) to match protocol.

Optimized for Colab single-core: ultrasoft pseudopotentials,
ecutwfc=35 Ry, Gamma-only k-point.

Usage on Colab:
  # 1. Install QE + pseudopotentials
  !apt-get -qq install quantum-espresso
  !pip install -q pymatgen numpy scipy
  !mkdir -p pseudo
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Li.pbe-s-rrkjus_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Co.pbe-spn-rrkjus_psl.0.3.1.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-rrkjus_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Al.pbe-n-rrkjus_psl.1.0.0.UPF

  # 2. Generate inputs and run
  !python paper/dft_interaction_check.py --generate
  !bash run_qe.sh
"""

import argparse
import json
import pathlib
import re
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"

# ---------- QE parameters (optimized for speed on Colab) ----------
ECUTWFC = 35.0     # Ry — sufficient for USPP (vs 50 for PAW)
ECUTRHO = 280.0    # Ry — 8× ecutwfc for ultrasoft
CONV_THR = 1.0e-5  # Relaxed but fine for energy differences
MIXING_BETA = 0.3  # Conservative mixing for spin-polarized metals
ELECTRON_MAXSTEP = 200
PSEUDO_DIR = "./pseudo"

# 3×3×2 supercell = 72 atoms, 18 Co sites, ~11% doping with 2 dopants
SUPERCELL = [3, 3, 2]

# Hubbard U for Co 3d (Materials Project standard)
HUBBARD_U_CO = 3.32  # eV

# Ultrasoft pseudopotentials (PSlibrary PBE RRKJUS)
PSEUDOS = {
    "Li": "Li.pbe-s-rrkjus_psl.1.0.0.UPF",
    "Co": "Co.pbe-spn-rrkjus_psl.0.3.1.UPF",
    "O":  "O.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-rrkjus_psl.1.0.0.UPF",
}

MASSES = {"Li": 6.941, "Co": 58.933, "O": 15.999, "Al": 26.982}


def build_supercell():
    """Build 3×3×2 LCO supercell."""
    from pymatgen.core import Structure
    struct = Structure.from_file(str(DATA_DIR / "lco_parent.cif"))
    struct.make_supercell(SUPERCELL)
    print(f"Supercell: {SUPERCELL}, {len(struct)} atoms")
    return struct


def get_nn_pair(struct):
    """Find the NN Co–Co pair by scanning all Co-Co distances."""
    co_sites = [i for i, sp in enumerate(struct.species) if str(sp) == "Co"]
    print(f"Co sites: {len(co_sites)} total")

    min_dist = 999.0
    best_pair = (co_sites[0], co_sites[1])
    for i, si in enumerate(co_sites):
        for sj in co_sites[i + 1:]:
            d = struct.get_distance(si, sj)
            if d < min_dist:
                min_dist = d
                best_pair = (si, sj)

    si, sj = best_pair
    dist = struct.get_distance(si, sj)
    print(f"NN pair: sites {si}, {sj}, distance = {dist:.3f} Å")
    print(f"Species at {si}: {struct.species[si]}, at {sj}: {struct.species[sj]}")
    return si, sj, dist


def struct_to_qe_input(struct, prefix):
    """Convert pymatgen Structure to QE pw.x input string."""
    lattice = struct.lattice
    species_list = [str(sp) for sp in struct.species]
    unique_species = sorted(set(species_list))

    has_co = "Co" in unique_species
    nat = len(struct)
    ntyp = len(unique_species)

    lines = []

    # &CONTROL
    lines.append("&CONTROL")
    lines.append(f"  calculation = 'scf'")
    lines.append(f"  prefix = '{prefix}'")
    lines.append(f"  outdir = './tmp_{prefix}'")
    lines.append(f"  pseudo_dir = '{PSEUDO_DIR}'")
    lines.append(f"  tprnfor = .false.")
    lines.append(f"  tstress = .false.")
    lines.append(f"  verbosity = 'low'")
    lines.append(f"  disk_io = 'low'")
    lines.append("/")
    lines.append("")

    # &SYSTEM
    lines.append("&SYSTEM")
    lines.append(f"  ibrav = 0")
    lines.append(f"  nat = {nat}")
    lines.append(f"  ntyp = {ntyp}")
    lines.append(f"  ecutwfc = {ECUTWFC}")
    lines.append(f"  ecutrho = {ECUTRHO}")
    lines.append(f"  occupations = 'smearing'")
    lines.append(f"  smearing = 'mv'")
    lines.append(f"  degauss = 0.02")
    lines.append(f"  nspin = 2")
    # Set starting magnetization for Co
    for i, sp in enumerate(unique_species):
        if sp == "Co":
            lines.append(f"  starting_magnetization({i + 1}) = 0.5")
    if has_co:
        lines.append(f"  lda_plus_u = .true.")
        for i, sp in enumerate(unique_species):
            if sp == "Co":
                lines.append(f"  Hubbard_U({i + 1}) = {HUBBARD_U_CO}")
    lines.append("/")
    lines.append("")

    # &ELECTRONS
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {CONV_THR}")
    lines.append(f"  mixing_beta = {MIXING_BETA}")
    lines.append(f"  electron_maxstep = {ELECTRON_MAXSTEP}")
    lines.append(f"  diagonalization = 'david'")
    lines.append("/")
    lines.append("")

    # CELL_PARAMETERS
    lines.append("CELL_PARAMETERS angstrom")
    for vec in lattice.matrix:
        lines.append(f"  {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}")
    lines.append("")

    # ATOMIC_SPECIES
    lines.append("ATOMIC_SPECIES")
    for sp in unique_species:
        lines.append(f"  {sp:4s} {MASSES.get(sp, 1.0):10.4f} {PSEUDOS[sp]}")
    lines.append("")

    # ATOMIC_POSITIONS
    lines.append("ATOMIC_POSITIONS crystal")
    for sp, site in zip(species_list, struct.sites):
        fc = site.frac_coords
        lines.append(f"  {sp:4s} {fc[0]:16.10f} {fc[1]:16.10f} {fc[2]:16.10f}")
    lines.append("")

    # K_POINTS — Gamma only for 72-atom cell (big speedup)
    lines.append("K_POINTS gamma")
    lines.append("")

    return "\n".join(lines)


def generate_inputs():
    """Generate 4 QE input files for the interaction energy calculation."""
    struct = build_supercell()
    si, sj, dist = get_nn_pair(struct)

    configs = {}

    # E(0): undoped
    configs["e0_undoped"] = struct.copy()

    # E(A): dopant at site i only
    sA = struct.copy()
    sA.replace(si, "Al")
    configs["eA_site_i"] = sA

    # E(B): dopant at site j only
    sB = struct.copy()
    sB.replace(sj, "Al")
    configs["eB_site_j"] = sB

    # E(AB): dopant at both sites
    sAB = struct.copy()
    sAB.replace(si, "Al")
    sAB.replace(sj, "Al")
    configs["eAB_both"] = sAB

    out_dir = pathlib.Path("qe_inputs")
    out_dir.mkdir(exist_ok=True)

    run_lines = [
        "#!/bin/bash",
        "set -e",
        "START=$(date +%s)",
        "",
    ]

    for name, s in configs.items():
        inp = struct_to_qe_input(s, name)
        inp_file = out_dir / f"{name}.in"
        with open(inp_file, "w") as f:
            f.write(inp)
        print(f"  Written: {inp_file}")

        run_lines.append(f"echo '=== Running {name} ==='")
        run_lines.append(f"T0=$(date +%s)")
        run_lines.append(f"mkdir -p tmp_{name}")
        run_lines.append(f"pw.x < {out_dir}/{name}.in > {out_dir}/{name}.out 2>&1")
        run_lines.append(f"T1=$(date +%s)")
        run_lines.append(f"echo \"  Done: {name}  ($(($T1 - $T0))s)\"")
        run_lines.append("")

    run_lines.append("END=$(date +%s)")
    run_lines.append("echo \"\"")
    run_lines.append("echo \"All 4 calculations complete in $(($END - $START))s.\"")
    run_lines.append("echo \"\"")
    run_lines.append("python3 paper/dft_interaction_check.py --parse")

    with open("run_qe.sh", "w") as f:
        f.write("\n".join(run_lines))

    print(f"\n  Run script: run_qe.sh")
    print(f"  NN distance: {dist:.3f} Å")
    print(f"  Supercell: {SUPERCELL} = {len(struct)} atoms")
    print(f"  Pseudopotentials: ultrasoft (RRKJUS)")
    print(f"  ecutwfc = {ECUTWFC} Ry, ecutrho = {ECUTRHO} Ry")
    print(f"  K-points: Gamma only")
    print(f"  Expected time: ~30-60 min on Colab (serial pw.x)")


def parse_outputs():
    """Parse QE outputs and compute E_int."""
    out_dir = pathlib.Path("qe_inputs")
    energies = {}

    for name in ["e0_undoped", "eA_site_i", "eB_site_j", "eAB_both"]:
        out_file = out_dir / f"{name}.out"
        if not out_file.exists():
            print(f"  Missing: {out_file}")
            return

        text = out_file.read_text()

        if "convergence NOT achieved" in text:
            print(f"  WARNING: {name} did NOT converge!")

        matches = re.findall(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text)
        if not matches:
            print(f"  ERROR: No energy found in {out_file}")
            # Show last few lines for debugging
            tail = text.strip().split("\n")[-10:]
            for line in tail:
                print(f"    {line}")
            return

        e_ry = float(matches[-1])
        e_ev = e_ry * 13.605698  # Ry -> eV
        energies[name] = e_ev
        print(f"  {name:15s}: {e_ry:16.8f} Ry = {e_ev:16.6f} eV")

    if len(energies) < 4:
        print("  ERROR: Not all calculations converged.")
        return

    E0 = energies["e0_undoped"]
    EA = energies["eA_site_i"]
    EB = energies["eB_site_j"]
    EAB = energies["eAB_both"]

    E_int_eV = EAB - EA - EB + E0
    E_int_meV = E_int_eV * 1000

    print(f"\n{'=' * 60}")
    print(f"  E_int (DFT PBE+U)  = {E_int_meV:+.1f} meV")
    print(f"  E_int (MACE-MP-0)  = -128.4 meV")
    print(f"{'=' * 60}")

    if E_int_meV < 0:
        print(f"\n  NN interaction is ATTRACTIVE in DFT")
        print(f"  --> CONFIRMS MACE sign")
    else:
        print(f"\n  NN interaction is REPULSIVE in DFT")
        print(f"  --> CONTRADICTS MACE sign")

    sign_match = (E_int_meV < 0)
    ratio = E_int_meV / -128.4 if abs(E_int_meV) > 0.1 else 0.0

    result = {
        "method": "DFT PBE+U (QE, USPP, SCF only, Gamma-only)",
        "system": "LiCoO2 3x3x2 supercell",
        "n_atoms": 72,
        "n_co_sites": 18,
        "dopant_concentration_pct": 11.1,
        "dopant": "Al",
        "pair": "NN",
        "distance_ang": 2.875,
        "ecutwfc_Ry": ECUTWFC,
        "ecutrho_Ry": ECUTRHO,
        "kpoints": "gamma",
        "Hubbard_U_Co_eV": HUBBARD_U_CO,
        "E0_eV": E0,
        "EA_eV": EA,
        "EB_eV": EB,
        "EAB_eV": EAB,
        "E_int_eV": E_int_eV,
        "E_int_meV": round(E_int_meV, 1),
        "MACE_E_int_meV": -128.4,
        "sign_match": sign_match,
        "DFT_over_MACE_ratio": round(ratio, 2),
    }

    out_path = pathlib.Path("paper") / "dft_interaction_result.json"
    if not out_path.parent.exists():
        out_path = pathlib.Path("dft_interaction_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Generate QE input files")
    parser.add_argument("--parse", action="store_true",
                        help="Parse QE outputs and compute E_int")
    args = parser.parse_args()

    if args.generate:
        generate_inputs()
    elif args.parse:
        parse_outputs()
    else:
        print("Usage:")
        print("  --generate   Create QE input files + run script")
        print("  --parse      Extract energies and compute E_int")
