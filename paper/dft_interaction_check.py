#!/usr/bin/env python3
"""
DFT (Quantum ESPRESSO) validation of dopant–dopant interaction energy sign.

Generates 4 QE input files for E_int at the NN distance in LiCoO₂:
  E_int = E(AB) - E(A) - E(B) + E(0)

Uses the same 3×3×2 supercell (72 atoms) as the MACE calculation,
with single-point SCF (no relaxation) to match protocol.

Two-stage approach:
  Stage 1: Spin-polarized PBE+U (gold standard, may be slow/fragile)
  Stage 2: Non-spin-polarized PBE fallback (fast, robust convergence)
  The run script auto-detects convergence failure and falls back.

Usage on Colab:
  !apt-get -qq install quantum-espresso
  !pip install -q pymatgen numpy scipy
  !mkdir -p pseudo
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Li.pbe-s-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Co.pbe-spn-kjpaw_psl.0.3.1.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Al.pbe-n-kjpaw_psl.1.0.0.UPF
  !python paper/dft_interaction_check.py --generate
  !bash run_qe.sh
"""

import argparse
import json
import pathlib
import re

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"

# ---------- QE parameters ----------
ECUTWFC = 40.0      # Ry
ECUTRHO = 320.0     # Ry (8× ecutwfc for PAW)
CONV_THR = 1.0e-5   # ~0.14 meV — fine for ~100 meV signal
PSEUDO_DIR = "./pseudo"

SUPERCELL = [3, 3, 2]  # 72 atoms, 18 Co sites

HUBBARD_U_CO = 3.32  # eV (Materials Project standard)

PSEUDOS = {
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "Co": "Co.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
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


def struct_to_qe_input(struct, prefix, spin_polarized=True):
    """Convert pymatgen Structure to QE pw.x input string.

    Args:
        struct: pymatgen Structure
        prefix: calculation prefix
        spin_polarized: if True, nspin=2 + Hubbard U; if False, nspin=1, no U
    """
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

    if spin_polarized:
        lines.append(f"  nspin = 2")
        for i, sp in enumerate(unique_species):
            if sp == "Co":
                lines.append(f"  starting_magnetization({i + 1}) = 0.5")
        if has_co:
            lines.append(f"  lda_plus_u = .true.")
            for i, sp in enumerate(unique_species):
                if sp == "Co":
                    lines.append(f"  Hubbard_U({i + 1}) = {HUBBARD_U_CO}")
    # else: nspin defaults to 1, no U

    lines.append("/")
    lines.append("")

    # &ELECTRONS
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {CONV_THR}")
    if spin_polarized:
        lines.append(f"  mixing_beta = 0.3")
        lines.append(f"  electron_maxstep = 200")
    else:
        lines.append(f"  mixing_beta = 0.7")    # Aggressive — NM converges easily
        lines.append(f"  electron_maxstep = 100")
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

    # K_POINTS — Gamma only for 72-atom cell
    lines.append("K_POINTS gamma")
    lines.append("")

    return "\n".join(lines)


def generate_inputs():
    """Generate QE input files for both stages + auto-fallback run script."""
    struct = build_supercell()
    si, sj, dist = get_nn_pair(struct)

    configs = {}
    configs["e0_undoped"] = struct.copy()

    sA = struct.copy()
    sA.replace(si, "Al")
    configs["eA_site_i"] = sA

    sB = struct.copy()
    sB.replace(sj, "Al")
    configs["eB_site_j"] = sB

    sAB = struct.copy()
    sAB.replace(si, "Al")
    sAB.replace(sj, "Al")
    configs["eAB_both"] = sAB

    out_dir = pathlib.Path("qe_inputs")
    out_dir.mkdir(exist_ok=True)

    calc_names = list(configs.keys())

    # Write input files for both stages
    for name, s in configs.items():
        # Stage 1: spin-polarized PBE+U
        inp1 = struct_to_qe_input(s, name, spin_polarized=True)
        with open(out_dir / f"{name}.in", "w") as f:
            f.write(inp1)

        # Stage 2: non-magnetic PBE fallback
        inp2 = struct_to_qe_input(s, f"{name}_nm", spin_polarized=False)
        with open(out_dir / f"{name}_nm.in", "w") as f:
            f.write(inp2)

    print(f"  Written: 4 spin-polarized + 4 non-magnetic inputs to {out_dir}/")

    # --- Build run script with auto-fallback ---
    run = []
    run.append("#!/bin/bash")
    run.append("# Two-stage DFT interaction energy validation")
    run.append("# Stage 1: spin-polarized PBE+U")
    run.append("# Stage 2: non-magnetic PBE (auto-fallback if Stage 1 fails)")
    run.append("")
    run.append("START=$(date +%s)")
    run.append("STAGE1_OK=true")
    run.append("")

    # Stage 1
    run.append("echo '============================================'")
    run.append("echo '  STAGE 1: Spin-polarized PBE+U'")
    run.append("echo '============================================'")
    run.append("")

    for name in calc_names:
        run.append(f"echo '--- {name} (PBE+U, spin) ---'")
        run.append(f"T0=$(date +%s)")
        run.append(f"mkdir -p tmp_{name}")
        run.append(f"pw.x < qe_inputs/{name}.in > qe_inputs/{name}.out 2>&1")
        run.append(f"T1=$(date +%s)")
        run.append(f"echo \"  Elapsed: $(($T1 - $T0))s\"")
        # Check convergence
        run.append(f"if grep -q 'convergence NOT achieved' qe_inputs/{name}.out; then")
        run.append(f"  echo '  *** SCF DID NOT CONVERGE — will fallback ***'")
        run.append(f"  STAGE1_OK=false")
        run.append(f"  break")
        run.append(f"fi")
        run.append(f"if ! grep -q '!.*total energy' qe_inputs/{name}.out; then")
        run.append(f"  echo '  *** NO ENERGY FOUND — will fallback ***'")
        run.append(f"  STAGE1_OK=false")
        run.append(f"  break")
        run.append(f"fi")
        run.append(f"echo '  Converged.'")
        run.append("")

    # Check if Stage 1 succeeded
    run.append("if [ \"$STAGE1_OK\" = true ]; then")
    run.append("  MID=$(date +%s)")
    run.append("  echo ''")
    run.append("  echo \"Stage 1 complete in $(($MID - $START))s.\"")
    run.append("  echo ''")
    run.append("  python3 paper/dft_interaction_check.py --parse --stage pbe_u")
    run.append("  exit 0")
    run.append("fi")
    run.append("")

    # Stage 2: fallback
    run.append("echo ''")
    run.append("echo '============================================'")
    run.append("echo '  STAGE 2: Non-magnetic PBE (fallback)'")
    run.append("echo '============================================'")
    run.append("echo ''")
    run.append("")

    for name in calc_names:
        run.append(f"echo '--- {name} (PBE, non-magnetic) ---'")
        run.append(f"T0=$(date +%s)")
        run.append(f"mkdir -p tmp_{name}_nm")
        run.append(f"pw.x < qe_inputs/{name}_nm.in > qe_inputs/{name}_nm.out 2>&1")
        run.append(f"T1=$(date +%s)")
        run.append(f"echo \"  Elapsed: $(($T1 - $T0))s\"")
        run.append(f"if grep -q 'convergence NOT achieved' qe_inputs/{name}_nm.out; then")
        run.append(f"  echo '  *** STAGE 2 ALSO FAILED — check inputs ***'")
        run.append(f"fi")
        run.append(f"if grep -q '!.*total energy' qe_inputs/{name}_nm.out; then")
        run.append(f"  echo '  Converged.'")
        run.append(f"fi")
        run.append("")

    run.append("END=$(date +%s)")
    run.append("echo ''")
    run.append("echo \"Stage 2 complete in $(($END - $START))s.\"")
    run.append("echo ''")
    run.append("python3 paper/dft_interaction_check.py --parse --stage pbe")
    run.append("")

    with open("run_qe.sh", "w") as f:
        f.write("\n".join(run))

    print(f"\n  Run script: run_qe.sh")
    print(f"  NN distance: {dist:.3f} Å")
    print(f"  Supercell: {SUPERCELL} = {len(struct)} atoms")
    print(f"  ecutwfc = {ECUTWFC} Ry, K-points: Gamma only")
    print(f"  Stage 1: PBE+U, nspin=2, mixing_beta=0.3")
    print(f"  Stage 2: PBE, nspin=1, mixing_beta=0.7 (auto-fallback)")


def parse_outputs(stage="pbe_u"):
    """Parse QE outputs and compute E_int."""
    out_dir = pathlib.Path("qe_inputs")

    if stage == "pbe_u":
        suffix = ""
        method_label = "DFT PBE+U (spin-polarized)"
    else:
        suffix = "_nm"
        method_label = "DFT PBE (non-magnetic, fallback)"

    energies = {}
    names = ["e0_undoped", "eA_site_i", "eB_site_j", "eAB_both"]

    for name in names:
        out_file = out_dir / f"{name}{suffix}.out"
        if not out_file.exists():
            print(f"  Missing: {out_file}")
            return

        text = out_file.read_text()

        if "convergence NOT achieved" in text:
            print(f"  WARNING: {name} did NOT converge!")

        matches = re.findall(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text)
        if not matches:
            print(f"  ERROR: No energy found in {out_file}")
            tail = text.strip().split("\n")[-10:]
            for line in tail:
                print(f"    {line}")
            return

        e_ry = float(matches[-1])
        e_ev = e_ry * 13.605698  # Ry -> eV
        energies[name] = e_ev
        print(f"  {name:15s}: {e_ry:16.8f} Ry = {e_ev:16.6f} eV")

    if len(energies) < 4:
        print("  ERROR: Not all calculations produced energies.")
        return

    E0 = energies["e0_undoped"]
    EA = energies["eA_site_i"]
    EB = energies["eB_site_j"]
    EAB = energies["eAB_both"]

    E_int_eV = EAB - EA - EB + E0
    E_int_meV = E_int_eV * 1000

    print(f"\n{'=' * 60}")
    print(f"  Method:  {method_label}")
    print(f"  E_int (DFT)        = {E_int_meV:+.1f} meV")
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
        "method": f"DFT {'PBE+U' if stage == 'pbe_u' else 'PBE'} (QE, PAW, SCF, Gamma)",
        "spin_polarized": stage == "pbe_u",
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
        "Hubbard_U_Co_eV": HUBBARD_U_CO if stage == "pbe_u" else 0.0,
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
                        help="Generate QE input files + run script")
    parser.add_argument("--parse", action="store_true",
                        help="Parse QE outputs and compute E_int")
    parser.add_argument("--stage", default="pbe_u", choices=["pbe_u", "pbe"],
                        help="Which stage to parse: pbe_u (default) or pbe")
    args = parser.parse_args()

    if args.generate:
        generate_inputs()
    elif args.parse:
        parse_outputs(stage=args.stage)
    else:
        print("Usage:")
        print("  --generate          Create QE inputs + auto-fallback run script")
        print("  --parse             Extract energies and compute E_int")
        print("  --stage pbe_u|pbe   Which stage outputs to parse (default: pbe_u)")
