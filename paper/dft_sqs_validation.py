#!/usr/bin/env python3
"""
DFT (Quantum ESPRESSO) validation of MACE disorder predictions.

Compares MACE-predicted voltages for ordered vs SQS-disordered LiCoO₂
against single-point DFT energies on the same structures.

Protocol:
  1. Build ordered + SQS-disordered structures (regenerated with same
     code used in the screening pipeline).
  2. MACE-relax all structures (same as pipeline).
  3. Generate QE single-point SCF inputs on MACE-relaxed geometries.
  4. Parse QE outputs → compute voltages → Spearman ρ (MACE vs DFT).

This validates whether MACE preserves disorder-induced ranking changes
when compared against the DFT ground truth.

Selected dopants: Al, Ti, Mg, Ga, Fe (5 with experimental data).
Structures: 1 ordered + 2 SQS per dopant + 2 undoped refs = 17 calcs.

Usage on Colab:
  !apt-get -qq install quantum-espresso
  !pip install -q pymatgen mace-torch numpy scipy

  # Download pseudopotentials
  !mkdir -p pseudo
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Li.pbe-s-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Co.pbe-spn-kjpaw_psl.0.3.1.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Al.pbe-n-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Ga.pbe-dn-kjpaw_psl.1.0.0.UPF
  !wget -q -P pseudo https://pseudopotentials.quantum-espresso.org/upf_files/Fe.pbe-spn-kjpaw_psl.0.2.1.UPF

  # Step 1: Generate structures + MACE relax + QE inputs
  !python paper/dft_sqs_validation.py --generate

  # Step 2: Run all QE calculations
  !bash run_dft_sqs.sh

  # Step 3: Parse and compute Spearman ρ
  !python paper/dft_sqs_validation.py --parse
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "structures"

# ---------- QE parameters ----------
ECUTWFC = 40.0      # Ry
ECUTRHO = 320.0     # Ry (8× ecutwfc for PAW)
CONV_THR = 1.0e-5   # ~0.14 meV
PSEUDO_DIR = "./pseudo"

SUPERCELL = [3, 3, 2]  # 72 atoms, 18 Co sites
CONCENTRATION = 0.10    # 10% dopant on Co site → 2 dopants in 18 Co sites
N_SQS = 2              # 2 SQS per dopant (Colab time budget)

DOPANTS = ["Al", "Ti", "Mg", "Ga", "Fe"]

# Hubbard U values (eV) — Materials Project standard
# Non-TM dopants (Al, Mg, Ga) don't need U.
HUBBARD_U = {
    "Co": 3.32,
    "Ti": 0.0,   # Ti d-states, but U=0 is MP standard for Ti in oxides
    "Fe": 5.3,   # Materials Project standard
}

# Pseudopotentials — PSLibrary PAW-PBE
PSEUDOS = {
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "Co": "Co.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
}

MASSES = {
    "Li": 6.941, "Co": 58.933, "O": 15.999,
    "Al": 26.982, "Ti": 47.867, "Mg": 24.305,
    "Ga": 69.723, "Fe": 55.845,
}

# Reference energies for voltage calculation: E(Li metal) per atom
# From QE PBE (bcc Li, same pseudopotential) — will be computed as part of the run
# Fallback: use MACE Li metal energy for both, since we care about ranking not absolute V
LI_METAL_E_RY = None  # Set after parsing li_metal.out


# ─────────────────────────────────────────────────────────────────────
# Structure generation
# ─────────────────────────────────────────────────────────────────────

def build_parent_supercell():
    """Build 3×3×2 LCO supercell from parent CIF."""
    from pymatgen.core import Structure
    struct = Structure.from_file(str(DATA_DIR / "lco_parent.cif"))
    struct.make_supercell(SUPERCELL)
    print(f"Parent supercell: {SUPERCELL}, {len(struct)} atoms")
    return struct


def generate_sqs_structures(parent, dopant, n_sqs=N_SQS):
    """Generate SQS structures using the project's SQS generator."""
    sys.path.insert(0, str(PROJECT_DIR))
    from stages.stage5.sqs_generator import generate_sqs

    structures = generate_sqs(
        parent_structure=parent.copy(),
        dopant_element=dopant,
        target_species="Co",
        concentration=CONCENTRATION,
        supercell_matrix=SUPERCELL,
        n_realisations=n_sqs,
    )
    print(f"  {dopant}: generated {len(structures)} SQS structures")
    return structures


def build_ordered_structure(parent, dopant):
    """Build ordered doped structure (dopants at first n sites)."""
    struct = parent.copy()
    co_indices = [i for i, sp in enumerate(struct.species) if str(sp) == "Co"]
    n_dopant = round(CONCENTRATION * len(co_indices))
    for idx in co_indices[:n_dopant]:
        struct.replace(idx, dopant)
    print(f"  {dopant}: ordered structure with {n_dopant} dopants")
    return struct


def mace_relax(struct, fmax=0.15, steps=500):
    """Relax structure with MACE-MP-0 (same protocol as screening pipeline)."""
    from mace.calculators import mace_mp
    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(struct)
    calc = mace_mp(model="medium", default_dtype="float64")
    atoms.calc = calc

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    relaxed = AseAtomsAdaptor.get_structure(atoms)
    e_total = atoms.get_potential_energy()  # eV
    return relaxed, e_total


# ─────────────────────────────────────────────────────────────────────
# QE input generation
# ─────────────────────────────────────────────────────────────────────

def struct_to_qe_input(struct, prefix):
    """Convert pymatgen Structure to QE pw.x input (non-magnetic PBE+U)."""
    lattice = struct.lattice
    species_list = [str(sp) for sp in struct.species]
    unique_species = sorted(set(species_list))
    nat = len(struct)
    ntyp = len(unique_species)

    # Determine which species need Hubbard U
    u_species = {sp: HUBBARD_U[sp] for sp in unique_species if sp in HUBBARD_U and HUBBARD_U[sp] > 0}
    has_u = len(u_species) > 0

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

    # &SYSTEM — non-magnetic to ensure convergence on Colab
    lines.append("&SYSTEM")
    lines.append(f"  ibrav = 0")
    lines.append(f"  nat = {nat}")
    lines.append(f"  ntyp = {ntyp}")
    lines.append(f"  ecutwfc = {ECUTWFC}")
    lines.append(f"  ecutrho = {ECUTRHO}")
    lines.append(f"  occupations = 'smearing'")
    lines.append(f"  smearing = 'mv'")
    lines.append(f"  degauss = 0.02")
    # Use nspin=2 with starting mag for TM systems (helps convergence)
    lines.append(f"  nspin = 2")
    for i, sp in enumerate(unique_species):
        if sp in ("Co", "Fe", "Ti"):
            lines.append(f"  starting_magnetization({i + 1}) = 0.5")

    if has_u:
        lines.append(f"  lda_plus_u = .true.")
        for i, sp in enumerate(unique_species):
            if sp in u_species:
                lines.append(f"  Hubbard_U({i + 1}) = {u_species[sp]}")

    lines.append("/")
    lines.append("")

    # &ELECTRONS — improved convergence settings
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {CONV_THR}")
    lines.append(f"  mixing_mode = 'local-TF'")
    lines.append(f"  mixing_beta = 0.1")
    lines.append(f"  electron_maxstep = 500")
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


def generate_li_metal_input():
    """Generate QE input for Li metal reference (bcc, 2-atom cell)."""
    lines = []
    lines.append("&CONTROL")
    lines.append("  calculation = 'scf'")
    lines.append("  prefix = 'li_metal'")
    lines.append("  outdir = './tmp_li_metal'")
    lines.append(f"  pseudo_dir = '{PSEUDO_DIR}'")
    lines.append("  tprnfor = .false.")
    lines.append("  tstress = .false.")
    lines.append("  verbosity = 'low'")
    lines.append("  disk_io = 'low'")
    lines.append("/")
    lines.append("")
    lines.append("&SYSTEM")
    lines.append("  ibrav = 0")
    lines.append("  nat = 2")
    lines.append("  ntyp = 1")
    lines.append(f"  ecutwfc = {ECUTWFC}")
    lines.append(f"  ecutrho = {ECUTRHO}")
    lines.append("  occupations = 'smearing'")
    lines.append("  smearing = 'mv'")
    lines.append("  degauss = 0.02")
    lines.append("/")
    lines.append("")
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {CONV_THR}")
    lines.append("  mixing_beta = 0.7")
    lines.append("  electron_maxstep = 100")
    lines.append("  diagonalization = 'david'")
    lines.append("/")
    lines.append("")
    # BCC Li: a = 3.49 Å
    lines.append("CELL_PARAMETERS angstrom")
    lines.append("  3.4900000000   0.0000000000   0.0000000000")
    lines.append("  0.0000000000   3.4900000000   0.0000000000")
    lines.append("  0.0000000000   0.0000000000   3.4900000000")
    lines.append("")
    lines.append("ATOMIC_SPECIES")
    lines.append(f"  Li   {MASSES['Li']:10.4f} {PSEUDOS['Li']}")
    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    lines.append("  Li    0.0000000000   0.0000000000   0.0000000000")
    lines.append("  Li    0.5000000000   0.5000000000   0.5000000000")
    lines.append("")
    # Denser k-grid for metallic Li
    lines.append("K_POINTS automatic")
    lines.append("  8 8 8 0 0 0")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Main: --generate
# ─────────────────────────────────────────────────────────────────────

def generate_all():
    """Generate all structures, MACE-relax, write QE inputs + run script."""
    from pymatgen.core import Structure

    out_dir = pathlib.Path("qe_sqs_inputs")
    out_dir.mkdir(exist_ok=True)
    struct_dir = pathlib.Path("qe_sqs_structures")
    struct_dir.mkdir(exist_ok=True)

    parent = build_parent_supercell()

    # Track all calculations and MACE energies
    calc_manifest = {}  # name → {dopant, type, mace_energy, n_li, n_atoms}
    all_calc_names = []

    # ── Undoped references ──────────────────────────────────────────
    # Lithiated: LiCoO₂ supercell
    print("\n--- Undoped lithiated (LiCoO₂) ---")
    lith_relaxed, lith_mace_e = mace_relax(parent.copy())
    n_li_lith = sum(1 for sp in lith_relaxed.species if str(sp) == "Li")
    name = "undoped_lith"
    inp = struct_to_qe_input(lith_relaxed, name)
    (out_dir / f"{name}.in").write_text(inp)
    lith_relaxed.to(filename=str(struct_dir / f"{name}.cif"))
    calc_manifest[name] = {
        "dopant": "none", "type": "undoped_lith",
        "mace_energy_eV": lith_mace_e, "n_li": n_li_lith,
        "n_atoms": len(lith_relaxed),
    }
    all_calc_names.append(name)

    # Delithiated: CoO₂ supercell (remove all Li)
    print("--- Undoped delithiated (CoO₂) ---")
    delith = parent.copy()
    li_indices = sorted(
        [i for i, sp in enumerate(delith.species) if str(sp) == "Li"],
        reverse=True,
    )
    for idx in li_indices:
        delith.remove_sites([idx])
    delith_relaxed, delith_mace_e = mace_relax(delith)
    name = "undoped_delith"
    inp = struct_to_qe_input(delith_relaxed, name)
    (out_dir / f"{name}.in").write_text(inp)
    delith_relaxed.to(filename=str(struct_dir / f"{name}.cif"))
    calc_manifest[name] = {
        "dopant": "none", "type": "undoped_delith",
        "mace_energy_eV": delith_mace_e, "n_li": 0,
        "n_atoms": len(delith_relaxed),
    }
    all_calc_names.append(name)

    # Li metal reference
    print("--- Li metal (bcc) ---")
    name = "li_metal"
    inp = generate_li_metal_input()
    (out_dir / f"{name}.in").write_text(inp)
    all_calc_names.append(name)
    calc_manifest[name] = {
        "dopant": "none", "type": "li_metal",
        "mace_energy_eV": None, "n_li": 2, "n_atoms": 2,
    }

    # ── Per-dopant structures ───────────────────────────────────────
    for dopant in DOPANTS:
        print(f"\n=== Dopant: {dopant} ===")

        # Ordered (lithiated)
        ordered_lith = build_ordered_structure(parent.copy(), dopant)
        ordered_lith_r, ordered_lith_e = mace_relax(ordered_lith)
        n_li = sum(1 for sp in ordered_lith_r.species if str(sp) == "Li")
        name = f"{dopant}_ord_lith"
        inp = struct_to_qe_input(ordered_lith_r, name)
        (out_dir / f"{name}.in").write_text(inp)
        ordered_lith_r.to(filename=str(struct_dir / f"{name}.cif"))
        calc_manifest[name] = {
            "dopant": dopant, "type": "ordered_lith",
            "mace_energy_eV": ordered_lith_e, "n_li": n_li,
            "n_atoms": len(ordered_lith_r),
        }
        all_calc_names.append(name)

        # Ordered (delithiated)
        ordered_delith = build_ordered_structure(parent.copy(), dopant)
        li_idx = sorted(
            [i for i, sp in enumerate(ordered_delith.species) if str(sp) == "Li"],
            reverse=True,
        )
        for idx in li_idx:
            ordered_delith.remove_sites([idx])
        ordered_delith_r, ordered_delith_e = mace_relax(ordered_delith)
        name = f"{dopant}_ord_delith"
        inp = struct_to_qe_input(ordered_delith_r, name)
        (out_dir / f"{name}.in").write_text(inp)
        ordered_delith_r.to(filename=str(struct_dir / f"{name}.cif"))
        calc_manifest[name] = {
            "dopant": dopant, "type": "ordered_delith",
            "mace_energy_eV": ordered_delith_e, "n_li": 0,
            "n_atoms": len(ordered_delith_r),
        }
        all_calc_names.append(name)

        # SQS disordered (lithiated + delithiated)
        sqs_structs = generate_sqs_structures(parent, dopant, n_sqs=N_SQS)
        for i, sqs in enumerate(sqs_structs):
            # Lithiated SQS
            sqs_lith_r, sqs_lith_e = mace_relax(sqs.copy())
            n_li_sqs = sum(1 for sp in sqs_lith_r.species if str(sp) == "Li")
            name = f"{dopant}_sqs{i}_lith"
            inp = struct_to_qe_input(sqs_lith_r, name)
            (out_dir / f"{name}.in").write_text(inp)
            sqs_lith_r.to(filename=str(struct_dir / f"{name}.cif"))
            calc_manifest[name] = {
                "dopant": dopant, "type": f"sqs{i}_lith",
                "mace_energy_eV": sqs_lith_e, "n_li": n_li_sqs,
                "n_atoms": len(sqs_lith_r),
            }
            all_calc_names.append(name)

            # Delithiated SQS
            sqs_delith = sqs.copy()
            li_idx = sorted(
                [j for j, sp in enumerate(sqs_delith.species) if str(sp) == "Li"],
                reverse=True,
            )
            for idx in li_idx:
                sqs_delith.remove_sites([idx])
            sqs_delith_r, sqs_delith_e = mace_relax(sqs_delith)
            name = f"{dopant}_sqs{i}_delith"
            inp = struct_to_qe_input(sqs_delith_r, name)
            (out_dir / f"{name}.in").write_text(inp)
            sqs_delith_r.to(filename=str(struct_dir / f"{name}.cif"))
            calc_manifest[name] = {
                "dopant": dopant, "type": f"sqs{i}_delith",
                "mace_energy_eV": sqs_delith_e, "n_li": 0,
                "n_atoms": len(sqs_delith_r),
            }
            all_calc_names.append(name)

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(calc_manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    print(f"Total QE calculations: {len(all_calc_names)}")

    # ── Build run script ────────────────────────────────────────────
    run = []
    run.append("#!/bin/bash")
    run.append("# DFT SQS validation — single-point SCF on MACE-relaxed structures")
    run.append(f"# {len(all_calc_names)} calculations")
    run.append("")
    run.append("START=$(date +%s)")
    run.append("FAILED=0")
    run.append("")

    for name in all_calc_names:
        run.append(f"echo '--- {name} ---'")
        run.append(f"T0=$(date +%s)")
        run.append(f"mkdir -p tmp_{name}")
        run.append(f"pw.x < qe_sqs_inputs/{name}.in > qe_sqs_inputs/{name}.out 2>&1")
        run.append(f"T1=$(date +%s)")
        run.append(f"echo \"  Elapsed: $(($T1 - $T0))s\"")
        run.append(f"if grep -q '!.*total energy' qe_sqs_inputs/{name}.out; then")
        run.append(f"  echo '  Converged.'")
        run.append(f"else")
        run.append(f"  echo '  *** FAILED ***'")
        run.append(f"  FAILED=$((FAILED + 1))")
        run.append(f"fi")
        run.append("")

    run.append("END=$(date +%s)")
    run.append("echo ''")
    run.append(f"echo \"All {len(all_calc_names)} calculations done in $(($END - $START))s.\"")
    run.append("echo \"Failed: $FAILED\"")
    run.append("")

    with open("run_dft_sqs.sh", "w") as f:
        f.write("\n".join(run))
    print(f"Run script: run_dft_sqs.sh")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Dopants: {', '.join(DOPANTS)}")
    print(f"  Per dopant: 1 ordered (lith+delith) + {N_SQS} SQS (lith+delith)")
    print(f"  Plus: 2 undoped refs + 1 Li metal = {len(all_calc_names)} total")
    print(f"  Supercell: {SUPERCELL} = 72 atoms (54 atoms delithiated)")
    print(f"  ecutwfc = {ECUTWFC} Ry, K-points: Gamma (except Li metal: 8×8×8)")
    print(f"  Convergence: mixing_mode=local-TF, mixing_beta=0.1, maxstep=500")
    print(f"  Estimated Colab time: ~2-4 hours (T4 GPU not needed, CPU only)")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────
# Main: --parse
# ─────────────────────────────────────────────────────────────────────

def parse_qe_energy(out_file):
    """Extract total energy (Ry) from QE output. Returns None if not found."""
    if not out_file.exists():
        return None
    text = out_file.read_text()
    if "convergence NOT achieved" in text:
        print(f"  WARNING: {out_file.name} did NOT converge")
    matches = re.findall(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text)
    if not matches:
        return None
    return float(matches[-1])


def compute_voltage(e_lith_ry, e_delith_ry, n_li, e_li_metal_per_atom_ry):
    """Compute intercalation voltage from DFT total energies.

    V = -(E_lith - E_delith - n_Li * E_Li_metal) / n_Li
    All energies in Ry, result in eV.
    """
    RY_TO_EV = 13.605698
    e_diff = e_lith_ry - e_delith_ry - n_li * e_li_metal_per_atom_ry
    voltage = -e_diff * RY_TO_EV / n_li
    return voltage


def parse_all():
    """Parse all QE outputs, compute voltages, compare MACE vs DFT rankings."""
    from scipy.stats import spearmanr

    out_dir = pathlib.Path("qe_sqs_inputs")
    manifest_path = out_dir / "manifest.json"

    if not manifest_path.exists():
        print("ERROR: manifest.json not found. Run --generate first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Parse Li metal
    e_li_metal = parse_qe_energy(out_dir / "li_metal.out")
    if e_li_metal is None:
        print("ERROR: Li metal energy not found.")
        return
    e_li_per_atom = e_li_metal / 2.0  # 2-atom BCC cell
    print(f"Li metal: {e_li_metal:.8f} Ry ({e_li_per_atom:.8f} Ry/atom)")

    # Parse undoped reference
    e_undoped_lith = parse_qe_energy(out_dir / "undoped_lith.out")
    e_undoped_delith = parse_qe_energy(out_dir / "undoped_delith.out")
    if e_undoped_lith and e_undoped_delith:
        n_li_undoped = manifest["undoped_lith"]["n_li"]
        v_undoped = compute_voltage(e_undoped_lith, e_undoped_delith,
                                    n_li_undoped, e_li_per_atom)
        print(f"Undoped LiCoO₂ DFT voltage: {v_undoped:.4f} V")

    # Per-dopant results
    results = {}
    for dopant in DOPANTS:
        print(f"\n--- {dopant} ---")
        res = {"dopant": dopant}

        # Ordered voltage (DFT)
        e_ord_lith = parse_qe_energy(out_dir / f"{dopant}_ord_lith.out")
        e_ord_delith = parse_qe_energy(out_dir / f"{dopant}_ord_delith.out")
        if e_ord_lith and e_ord_delith:
            n_li = manifest[f"{dopant}_ord_lith"]["n_li"]
            v_ord_dft = compute_voltage(e_ord_lith, e_ord_delith, n_li, e_li_per_atom)
            res["voltage_ordered_dft"] = v_ord_dft
            print(f"  Ordered DFT voltage: {v_ord_dft:.4f} V")
        else:
            res["voltage_ordered_dft"] = None
            print(f"  Ordered DFT: MISSING")

        # Ordered voltage (MACE) — from manifest
        e_ord_lith_mace = manifest[f"{dopant}_ord_lith"]["mace_energy_eV"]
        e_ord_delith_mace = manifest[f"{dopant}_ord_delith"]["mace_energy_eV"]
        n_li = manifest[f"{dopant}_ord_lith"]["n_li"]
        # MACE voltage: same formula but energies already in eV
        v_ord_mace = -(e_ord_lith_mace - e_ord_delith_mace) / n_li
        # Note: no Li metal ref for MACE — we use the MACE undoped ref implicitly
        # For ranking comparison, we only need relative ordering, not absolute V
        res["voltage_ordered_mace_raw"] = v_ord_mace

        # SQS disordered voltages
        sqs_dft_voltages = []
        sqs_mace_voltages = []
        for i in range(N_SQS):
            e_sqs_lith = parse_qe_energy(out_dir / f"{dopant}_sqs{i}_lith.out")
            e_sqs_delith = parse_qe_energy(out_dir / f"{dopant}_sqs{i}_delith.out")
            if e_sqs_lith and e_sqs_delith:
                n_li_sqs = manifest[f"{dopant}_sqs{i}_lith"]["n_li"]
                v_sqs_dft = compute_voltage(e_sqs_lith, e_sqs_delith,
                                            n_li_sqs, e_li_per_atom)
                sqs_dft_voltages.append(v_sqs_dft)
                print(f"  SQS{i} DFT voltage: {v_sqs_dft:.4f} V")

            e_sqs_lith_mace = manifest[f"{dopant}_sqs{i}_lith"]["mace_energy_eV"]
            e_sqs_delith_mace = manifest[f"{dopant}_sqs{i}_delith"]["mace_energy_eV"]
            n_li_sqs = manifest[f"{dopant}_sqs{i}_lith"]["n_li"]
            v_sqs_mace = -(e_sqs_lith_mace - e_sqs_delith_mace) / n_li_sqs
            sqs_mace_voltages.append(v_sqs_mace)

        if sqs_dft_voltages:
            res["voltage_disordered_dft_mean"] = float(np.mean(sqs_dft_voltages))
            res["voltage_disordered_dft_std"] = float(np.std(sqs_dft_voltages))
        else:
            res["voltage_disordered_dft_mean"] = None
            res["voltage_disordered_dft_std"] = None

        res["voltage_disordered_mace_raw_mean"] = float(np.mean(sqs_mace_voltages))
        res["sqs_dft_voltages"] = sqs_dft_voltages
        res["sqs_mace_voltages"] = sqs_mace_voltages

        results[dopant] = res

    # ── Spearman correlations ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RANKING COMPARISON: MACE vs DFT")
    print(f"{'='*60}")

    # 1. Ordered voltages: MACE ranking vs DFT ranking
    ord_mace = []
    ord_dft = []
    ord_labels = []
    for d in DOPANTS:
        r = results[d]
        if r["voltage_ordered_dft"] is not None:
            ord_mace.append(r["voltage_ordered_mace_raw"])
            ord_dft.append(r["voltage_ordered_dft"])
            ord_labels.append(d)

    if len(ord_mace) >= 3:
        rho_ord, p_ord = spearmanr(ord_mace, ord_dft)
        print(f"\n  Ordered voltages (MACE vs DFT), n={len(ord_mace)}:")
        print(f"    Spearman ρ = {rho_ord:.3f}, p = {p_ord:.3f}")
        for i, d in enumerate(ord_labels):
            print(f"    {d:4s}: MACE={ord_mace[i]:.4f}  DFT={ord_dft[i]:.4f}")

    # 2. Disordered voltages: MACE ranking vs DFT ranking
    dis_mace = []
    dis_dft = []
    dis_labels = []
    for d in DOPANTS:
        r = results[d]
        if r["voltage_disordered_dft_mean"] is not None:
            dis_mace.append(r["voltage_disordered_mace_raw_mean"])
            dis_dft.append(r["voltage_disordered_dft_mean"])
            dis_labels.append(d)

    if len(dis_mace) >= 3:
        rho_dis, p_dis = spearmanr(dis_mace, dis_dft)
        print(f"\n  Disordered voltages (MACE vs DFT), n={len(dis_mace)}:")
        print(f"    Spearman ρ = {rho_dis:.3f}, p = {p_dis:.3f}")
        for i, d in enumerate(dis_labels):
            print(f"    {d:4s}: MACE={dis_mace[i]:.4f}  DFT={dis_dft[i]:.4f}")

    # 3. Key question: does MACE preserve the disorder-induced voltage shift?
    shift_mace = []
    shift_dft = []
    shift_labels = []
    for d in DOPANTS:
        r = results[d]
        if (r["voltage_ordered_dft"] is not None and
                r["voltage_disordered_dft_mean"] is not None):
            s_mace = r["voltage_disordered_mace_raw_mean"] - r["voltage_ordered_mace_raw"]
            s_dft = r["voltage_disordered_dft_mean"] - r["voltage_ordered_dft"]
            shift_mace.append(s_mace)
            shift_dft.append(s_dft)
            shift_labels.append(d)

    if len(shift_mace) >= 3:
        rho_shift, p_shift = spearmanr(shift_mace, shift_dft)
        print(f"\n  Disorder-induced voltage shift (MACE vs DFT), n={len(shift_mace)}:")
        print(f"    Spearman ρ = {rho_shift:.3f}, p = {p_shift:.3f}")
        print(f"    (This is the KEY validation metric)")
        for i, d in enumerate(shift_labels):
            print(f"    {d:4s}: MACE shift={shift_mace[i]:+.4f}  DFT shift={shift_dft[i]:+.4f}")

    # Save results
    output = {
        "dopants": DOPANTS,
        "supercell": SUPERCELL,
        "n_sqs": N_SQS,
        "concentration": CONCENTRATION,
        "ecutwfc_Ry": ECUTWFC,
        "li_metal_energy_Ry_per_atom": e_li_per_atom,
        "per_dopant": results,
        "spearman_ordered": {
            "rho": rho_ord if len(ord_mace) >= 3 else None,
            "p": p_ord if len(ord_mace) >= 3 else None,
            "n": len(ord_mace),
        },
        "spearman_disordered": {
            "rho": rho_dis if len(dis_mace) >= 3 else None,
            "p": p_dis if len(dis_mace) >= 3 else None,
            "n": len(dis_mace),
        },
        "spearman_disorder_shift": {
            "rho": rho_shift if len(shift_mace) >= 3 else None,
            "p": p_shift if len(shift_mace) >= 3 else None,
            "n": len(shift_mace),
        },
    }

    result_path = pathlib.Path("paper") / "dft_sqs_validation_results.json"
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {result_path}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DFT validation of MACE disorder predictions"
    )
    parser.add_argument("--generate", action="store_true",
                        help="Generate structures, MACE-relax, write QE inputs + run script")
    parser.add_argument("--parse", action="store_true",
                        help="Parse QE outputs and compute Spearman ρ")
    args = parser.parse_args()

    if args.generate:
        generate_all()
    elif args.parse:
        parse_all()
    else:
        print("Usage:")
        print("  --generate    Build structures + MACE relax + QE inputs")
        print("  --parse       Parse QE outputs + compute Spearman ρ")
