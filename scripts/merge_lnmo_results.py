"""Merge LNMO batch files into a single all-22 JSON.

Usage (from project root):
    python scripts/merge_lnmo_results.py

Expects:
    evaluation/results/rq2_lnmo_batch1.json   — from Google Drive (8 dopants)
    evaluation/results/rq2_lnmo_batch2.json   — from Google Drive (7 dopants)
    evaluation/results/rq2_lnmo_batch3.json   — from Google Drive (7 dopants)

Outputs:
    evaluation/results/rq2_lnmo_all22.json    — all 22 dopants merged
"""
import json, pathlib, sys

ROOT        = pathlib.Path(__file__).parent.parent
RESULTS     = ROOT / 'evaluation' / 'results'
BATCH_FILES = [
    RESULTS / 'rq2_lnmo_batch1.json',
    RESULTS / 'rq2_lnmo_batch2.json',
    RESULTS / 'rq2_lnmo_batch3.json',
]
OUT_ALL = RESULTS / 'rq2_lnmo_all22.json'


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
    'supercell': [2, 2, 2],
    'host': 'LiMn2O4_spinel',
    'target_species': 'Mn',
    'notes': (
        '22 dopants at Mn4+ site of LiNi0.5Mn1.5O4 (LiMn2O4 proxy). '
        '448-atom supercell (2×2×2 of 56-atom conventional cell), 128 Mn sites, '
        '~13 substitutions at 10% doping. Colab A100, fmax=0.10 eV/Å. '
        'Stage 4 viability filter applied: Cr/As/Sb/Os/U excluded (toxic/radioactive). '
        'Mn excluded (self-substitution). Non-metals excluded (Br/Ge/I/P/S/Se/Si/Te).'
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
print('\nDopants:')
for r in all_results:
    print(f'  {r["dopant"]:4s}  n_converged={r.get("n_converged", "?")}')

print('\nNext step:')
print('  python -m evaluation.eval_accuracy \\')
print('      --results evaluation/results/rq2_lnmo_all22.json \\')
print('      --save evaluation/results/rq3_lnmo_accuracy.json')
