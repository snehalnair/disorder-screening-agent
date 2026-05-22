"""
Gate G0 Validation: Does MACE preserve DFT energy rankings for
ordered vs. disordered (SQS) structures?

Uses the Mathsphy HEA dataset (Zenodo 10854500):
  - 69,575 ordered structures (≤8 atoms)
  - 14,222 SQS structures (27, 64, 128 atoms)
  - Same compositions exist in both forms
  - DFT formation energies available

We run MACE-MP-0 on the same structures and compare:
  1. MACE vs DFT correlation for ordered structures (should be high)
  2. MACE vs DFT correlation for SQS structures (might be low — our hypothesis)
  3. Whether MACE preserves the DFT ranking ORDER when going ordered → SQS

If MACE gives high correlation for ordered but low for SQS,
that proves MACE is unreliable for disordered structures —
validating the need for our disorder-aware screening pipeline.

Usage:
  python paper/hea_mace_validation.py --download   # download dataset (450MB)
  python paper/hea_mace_validation.py --run         # run MACE on structures
  python paper/hea_mace_validation.py --analyze     # compute correlations
"""

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/hea_validation")
RESULTS_FILE = DATA_DIR / "mace_vs_dft_results.json"
ZENODO_URL = "https://zenodo.org/records/10854500/files/hea.2023-04-06.csv?download=1"


def download_dataset():
    """Download the HEA dataset from Zenodo."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "hea.2023-04-06.csv"

    if csv_path.exists():
        size_mb = csv_path.stat().st_size / 1e6
        print(f"Dataset already exists: {csv_path} ({size_mb:.0f} MB)")
        return csv_path

    print(f"Downloading HEA dataset (~450 MB)...")
    import urllib.request
    urllib.request.urlretrieve(ZENODO_URL, csv_path)
    print(f"Downloaded to {csv_path}")
    return csv_path


def find_matched_compositions(csv_path, max_per_system=5, max_systems=50):
    """Find compositions that exist in BOTH ordered and SQS forms.

    Returns a DataFrame with matched pairs.
    """
    print("Loading dataset (this takes ~30s for 450MB CSV)...")

    # Load only the columns we need for matching (skip structure dicts for now)
    df = pd.read_csv(
        csv_path,
        usecols=['fid', 'reduced_formula', 'chemical_system', 'nelements',
                 'NIONS', 'Ef_per_atom', 'e_per_atom', 'lattice'],
        on_bad_lines='skip',
    )

    print(f"Loaded {len(df)} entries")

    # Filter to alloys (2+ elements)
    df = df[df['nelements'] >= 2].copy()
    print(f"Alloys (2+ elements): {len(df)}")

    # Classify as ordered (≤8 atoms) or SQS (≥27 atoms)
    df['is_sqs'] = df['NIONS'] >= 27
    df['is_ordered'] = df['NIONS'] <= 8

    ordered = df[df['is_ordered']]
    sqs = df[df['is_sqs']]

    print(f"Ordered (≤8 atoms): {len(ordered)}")
    print(f"SQS (≥27 atoms): {len(sqs)}")

    # Find chemical systems with BOTH ordered and SQS entries
    ord_systems = set(ordered['chemical_system'].unique())
    sqs_systems = set(sqs['chemical_system'].unique())
    common_systems = sorted(ord_systems & sqs_systems)

    print(f"\nChemical systems with BOTH ordered and SQS: {len(common_systems)}")

    # For each common system, pick matched pairs
    matched = []
    for sys in common_systems[:max_systems]:
        ord_entries = ordered[ordered['chemical_system'] == sys].head(max_per_system)
        sqs_entries = sqs[sqs['chemical_system'] == sys].head(max_per_system)

        for _, row in ord_entries.iterrows():
            matched.append({
                'fid': row['fid'],
                'formula': row['reduced_formula'],
                'system': row['chemical_system'],
                'type': 'ordered',
                'n_atoms': row['NIONS'],
                'dft_Ef': row['Ef_per_atom'],
            })
        for _, row in sqs_entries.iterrows():
            matched.append({
                'fid': row['fid'],
                'formula': row['reduced_formula'],
                'system': row['chemical_system'],
                'type': 'SQS',
                'n_atoms': row['NIONS'],
                'dft_Ef': row['Ef_per_atom'],
            })

    matched_df = pd.DataFrame(matched)
    print(f"\nMatched entries: {len(matched_df)} "
          f"({len(matched_df[matched_df['type']=='ordered'])} ordered, "
          f"{len(matched_df[matched_df['type']=='SQS'])} SQS)")

    # Save matched FIDs for structure extraction
    matched_df.to_csv(DATA_DIR / "matched_pairs.csv", index=False)
    return matched_df


def extract_structures(csv_path, matched_df, batch_size=100):
    """Extract pymatgen structures for matched entries from the full CSV.

    Reads the CSV in chunks to avoid memory issues with the large
    structure_as_dict column.
    """
    target_fids = set(matched_df['fid'].tolist())
    structures = {}

    print(f"Extracting {len(target_fids)} structures from CSV...")

    # Read in chunks to handle the large CSV
    chunk_iter = pd.read_csv(
        csv_path,
        usecols=['fid', 'structure_as_dict'],
        chunksize=1000,
        on_bad_lines='skip',
    )

    found = 0
    for i, chunk in enumerate(chunk_iter):
        hits = chunk[chunk['fid'].isin(target_fids)]
        for _, row in hits.iterrows():
            try:
                struct_dict = ast.literal_eval(row['structure_as_dict'])
                structures[row['fid']] = struct_dict
                found += 1
            except (ValueError, SyntaxError):
                pass

        if found % 50 == 0 and found > 0:
            print(f"  Found {found}/{len(target_fids)} structures (chunk {i})...")

        if found >= len(target_fids):
            break

    print(f"Extracted {found}/{len(target_fids)} structures")

    # Save structures
    with open(DATA_DIR / "matched_structures.json", 'w') as f:
        json.dump(structures, f)

    return structures


def run_mace_on_structures(structures, matched_df):
    """Run MACE-MP-0 single-point on each structure, compare with DFT."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium", default_dtype="float64")
        print("Loaded MACE-MP-0 (medium)")
    except ImportError:
        print("MACE not installed. Install with: pip install mace-torch")
        sys.exit(1)

    adaptor = AseAtomsAdaptor()
    results = []

    total = len(matched_df)
    t_start = time.time()

    for i, (_, row) in enumerate(matched_df.iterrows()):
        fid = row['fid']

        if fid not in structures:
            continue

        try:
            # Build pymatgen Structure from dict
            struct = Structure.from_dict(structures[fid])

            # Convert to ASE atoms
            atoms = adaptor.get_atoms(struct)

            # Run MACE single-point
            atoms.calc = calc
            mace_energy = atoms.get_potential_energy()  # eV
            mace_e_per_atom = mace_energy / len(atoms)

            results.append({
                'fid': fid,
                'formula': row['formula'],
                'system': row['system'],
                'type': row['type'],
                'n_atoms': row['n_atoms'],
                'dft_Ef': row['dft_Ef'],
                'mace_e_per_atom': mace_e_per_atom,
            })

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate / 60
                print(f"  [{i+1}/{total}] {row['formula']} ({row['type']}): "
                      f"DFT={row['dft_Ef']:.4f}, MACE={mace_e_per_atom:.4f} eV/atom "
                      f"({remaining:.0f} min remaining)")

        except Exception as e:
            print(f"  FAILED {fid} ({row['formula']}): {e}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / "mace_vs_dft_results.csv", index=False)

    print(f"\nCompleted {len(results)}/{total} structures in {(time.time()-t_start)/60:.1f} min")
    return results_df


def analyze_results(results_df=None):
    """Compute Spearman rho and other correlation metrics."""
    from scipy.stats import spearmanr, pearsonr

    if results_df is None:
        results_df = pd.read_csv(DATA_DIR / "mace_vs_dft_results.csv")

    print("\n" + "="*70)
    print(" GATE G0 RESULTS: MACE vs DFT for Ordered vs SQS")
    print("="*70)

    ordered = results_df[results_df['type'] == 'ordered']
    sqs = results_df[results_df['type'] == 'SQS']

    print(f"\nDataset: {len(ordered)} ordered, {len(sqs)} SQS structures")

    # 1. MACE vs DFT correlation — ordered structures
    if len(ordered) >= 3:
        rho_ord, p_ord = spearmanr(ordered['dft_Ef'], ordered['mace_e_per_atom'])
        r_ord, _ = pearsonr(ordered['dft_Ef'], ordered['mace_e_per_atom'])
        mae_ord = np.mean(np.abs(ordered['dft_Ef'] - ordered['mace_e_per_atom']))
        print(f"\n1. MACE vs DFT (ORDERED structures):")
        print(f"   Spearman rho = {rho_ord:.4f} (p={p_ord:.2e})")
        print(f"   Pearson r    = {r_ord:.4f}")
        print(f"   MAE          = {mae_ord:.4f} eV/atom")

    # 2. MACE vs DFT correlation — SQS structures
    if len(sqs) >= 3:
        rho_sqs, p_sqs = spearmanr(sqs['dft_Ef'], sqs['mace_e_per_atom'])
        r_sqs, _ = pearsonr(sqs['dft_Ef'], sqs['mace_e_per_atom'])
        mae_sqs = np.mean(np.abs(sqs['dft_Ef'] - sqs['mace_e_per_atom']))
        print(f"\n2. MACE vs DFT (SQS/disordered structures):")
        print(f"   Spearman rho = {rho_sqs:.4f} (p={p_sqs:.2e})")
        print(f"   Pearson r    = {r_sqs:.4f}")
        print(f"   MAE          = {mae_sqs:.4f} eV/atom")

    # 3. Per-system analysis: does disorder change the MACE ranking?
    print(f"\n3. Per-system: MACE ranking preservation (ordered vs SQS)")
    print(f"   {'System':<20s} {'DFT_ord':>8s} {'DFT_sqs':>8s} {'MACE_ord':>9s} {'MACE_sqs':>9s} {'DFT_diff':>9s} {'MACE_diff':>10s} {'Same_sign':>10s}")
    print("   " + "-"*85)

    same_sign_count = 0
    total_compared = 0

    systems_with_both = set(ordered['system']) & set(sqs['system'])
    for sys in sorted(systems_with_both):
        ord_sys = ordered[ordered['system'] == sys]
        sqs_sys = sqs[sqs['system'] == sys]

        if len(ord_sys) == 0 or len(sqs_sys) == 0:
            continue

        dft_ord_mean = ord_sys['dft_Ef'].mean()
        dft_sqs_mean = sqs_sys['dft_Ef'].mean()
        mace_ord_mean = ord_sys['mace_e_per_atom'].mean()
        mace_sqs_mean = sqs_sys['mace_e_per_atom'].mean()

        dft_diff = dft_sqs_mean - dft_ord_mean
        mace_diff = mace_sqs_mean - mace_ord_mean

        same_sign = (dft_diff * mace_diff) > 0
        same_sign_count += int(same_sign)
        total_compared += 1

        sign_str = "✓" if same_sign else "✗ FLIPPED"
        print(f"   {sys:<20s} {dft_ord_mean:>+8.4f} {dft_sqs_mean:>+8.4f} "
              f"{mace_ord_mean:>+9.4f} {mace_sqs_mean:>+9.4f} "
              f"{dft_diff:>+9.4f} {mace_diff:>+10.4f} {sign_str:>10s}")

    if total_compared > 0:
        print(f"\n   Sign agreement: {same_sign_count}/{total_compared} "
              f"({100*same_sign_count/total_compared:.0f}%)")

    # 4. The key question: does MACE error increase for SQS vs ordered?
    if len(ordered) >= 3 and len(sqs) >= 3:
        print(f"\n4. KEY FINDING:")
        print(f"   MACE-DFT correlation for ORDERED:    rho = {rho_ord:.3f}")
        print(f"   MACE-DFT correlation for DISORDERED: rho = {rho_sqs:.3f}")
        print(f"   Correlation DROP:                    Δrho = {rho_sqs - rho_ord:+.3f}")
        print(f"   MAE increase:                        {mae_ord:.4f} → {mae_sqs:.4f} eV/atom")

        if rho_sqs < rho_ord - 0.1:
            print(f"\n   >>> MACE is LESS reliable for disordered structures.")
            print(f"   >>> This validates the need for disorder-aware screening.")
        elif rho_sqs >= rho_ord - 0.05:
            print(f"\n   >>> MACE correlation is similar for ordered and disordered.")
            print(f"   >>> MACE may be trustworthy for disorder screening.")

        if total_compared > 0 and same_sign_count / total_compared < 0.8:
            print(f"\n   >>> MACE FLIPS the ordered-vs-disordered preference for "
                  f"{total_compared - same_sign_count}/{total_compared} systems!")
            print(f"   >>> Rankings are NOT preserved — disorder-aware screening is essential.")


def main():
    parser = argparse.ArgumentParser(description="HEA MACE vs DFT validation")
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--run', action='store_true', help='Run MACE on structures')
    parser.add_argument('--analyze', action='store_true', help='Analyze results')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--max-systems', type=int, default=50,
                        help='Max chemical systems to compare')
    parser.add_argument('--max-per-system', type=int, default=3,
                        help='Max structures per system per type')
    args = parser.parse_args()

    if args.all:
        args.download = args.run = args.analyze = True

    if not (args.download or args.run or args.analyze):
        parser.print_help()
        return

    csv_path = DATA_DIR / "hea.2023-04-06.csv"

    if args.download:
        csv_path = download_dataset()

    if args.run:
        if not csv_path.exists():
            print(f"Dataset not found. Run with --download first.")
            return

        # Find matched compositions
        matched_df = find_matched_compositions(
            csv_path,
            max_per_system=args.max_per_system,
            max_systems=args.max_systems
        )

        # Extract structures
        structures = extract_structures(csv_path, matched_df)

        # Run MACE
        results_df = run_mace_on_structures(structures, matched_df)

        # Analyze
        analyze_results(results_df)

    elif args.analyze:
        analyze_results()


if __name__ == "__main__":
    main()
