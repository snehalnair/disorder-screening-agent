"""Merge NMC rerun batch files into a single all-22 JSON.

Usage (from project root):
    python scripts/merge_nmc_rerun_results.py

Expects:
    evaluation/results/rq2_nmc_rerun_batch1.json  — from Google Drive (11 dopants)
    evaluation/results/rq2_nmc_rerun_batch2.json  — from Google Drive (11 dopants)

Outputs:
    evaluation/results/rq2_disorder_all22_v2.json — all 22 dopants, fmax=0.10

This replaces rq2_disorder_all23.json (fmax=0.05, 78% convergence).
"""
import json, pathlib, sys

ROOT        = pathlib.Path(__file__).parent.parent
RESULTS     = ROOT / 'evaluation' / 'results'
BATCH_FILES = [
    RESULTS / 'rq2_nmc_rerun_batch1.json',
    RESULTS / 'rq2_nmc_rerun_batch2.json',
]
OUT_ALL = RESULTS / 'rq2_disorder_all22_v2.json'


def load(path: pathlib.Path) -> dict:
    if not path.exists():
        print(f'ERROR: file not found: {path}')
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ── Load and combine batches ───────────────────────────────────────────────────
all_results = []
for bf in BATCH_FILES:
    data = load(bf)
    all_results.extend(data['dopant_results'])
    print(f'{bf.name}: {len(data["dopant_results"])} dopants')

print(f'Total: {len(all_results)} dopants')

# ── Save merged ───────────────────────────────────────────────────────────────
merged = {
    'concentration': 0.10,
    'mlip': 'mace-mp-0',
    'n_sqs_realisations': 5,
    'supercell': [4, 4, 4],
    'host': 'LiCoO2_layered',
    'target_species': 'Co',
    'fmax': 0.10,
    'max_relax_steps': 1000,
    'notes': (
        '22 dopants at Co3+ site of LiNiMnCoO2 (LiCoO2 primitive-cell proxy). '
        '256-atom supercell (4×4×4 of 4-atom primitive cell), 64 Co sites, '
        '~6 substitutions at 10% doping. Colab A100, fmax=0.10 eV/Å, max_steps=1000. '
        'Standardised convergence criteria matching LNMO evaluation. '
        'Stage 4 viability filter applied: Cr/As/Sb/Os/U excluded (toxic/radioactive). '
        'Co excluded (self-substitution).'
    ),
    'dopant_results': all_results,
    'n_dopants': len(all_results),
}

RESULTS.mkdir(parents=True, exist_ok=True)
with open(OUT_ALL, 'w') as f:
    json.dump(merged, f, indent=2)
print(f'\n✓ Saved: {OUT_ALL}')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\nTotal dopants: {len(all_results)}')
n_converged = sum(r.get('n_converged', 0) for r in all_results)
n_total     = len(all_results) * 5
print(f'Total converged SQS: {n_converged}/{n_total} ({100*n_converged/n_total:.0f}%)')

print('\nPer-dopant convergence:')
for r in all_results:
    n_c = r.get('n_converged', '?')
    status = '✓' if n_c == 5 else ('⚠' if n_c >= 3 else '✗')
    print(f'  {status} {r["dopant"]:4s}  {n_c}/5')

print('\nComparison with original run (fmax=0.05):')
print('  Original: 86/110 (78%)   — rq2_disorder_all23.json')
print(f'  Rerun:    {n_converged}/{n_total} ({100*n_converged/n_total:.0f}%)   — rq2_disorder_all22_v2.json')

print('\nNext steps:')
print('  1. Update figures:')
print('       python -m evaluation.figures \\')
print('           --results evaluation/results/rq2_disorder_all22_v2.json \\')
print('           --lnmo evaluation/results/rq2_lnmo_all22.json \\')
print('           --format png')
print('  2. Recompute RQ3 accuracy:')
print('       python -m evaluation.eval_accuracy \\')
print('           --results evaluation/results/rq2_disorder_all22_v2.json \\')
print('           --save evaluation/results/rq3_accuracy_v2.json')
