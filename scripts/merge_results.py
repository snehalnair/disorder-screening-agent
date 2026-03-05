"""Merge known-8 results with novel-14 batch files into a single all-22 JSON.

Usage (from project root):
    python scripts/merge_results.py

Expects:
    evaluation/results/rq2_disorder_444.json     — known-8 (already present)
    evaluation/results/rq2_novel_batch1.json     — from Google Drive
    evaluation/results/rq2_novel_batch2.json     — from Google Drive
    evaluation/results/rq2_novel_batch3.json     — from Google Drive

Outputs:
    evaluation/results/rq2_disorder_novel14.json — all 14 novel dopants combined
    evaluation/results/rq2_disorder_all22.json   — known-8 + novel-14 merged
"""
import json, pathlib, sys

ROOT       = pathlib.Path(__file__).parent.parent
RESULTS    = ROOT / 'evaluation' / 'results'
KNOWN8     = RESULTS / 'rq2_disorder_444.json'
BATCH_FILES = [
    RESULTS / 'rq2_novel_batch1.json',
    RESULTS / 'rq2_novel_batch2.json',
    RESULTS / 'rq2_novel_batch3.json',
]
OUT_NOVEL  = RESULTS / 'rq2_disorder_novel14.json'
OUT_ALL    = RESULTS / 'rq2_disorder_all22.json'


def load(path: pathlib.Path) -> dict:
    if not path.exists():
        print(f'ERROR: file not found: {path}')
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ── Load known-8 ──────────────────────────────────────────────────────────────
known8 = load(KNOWN8)
print(f'Known-8  : {len(known8["dopant_results"])} dopants  ({KNOWN8.name})')

# ── Load and combine novel batches ────────────────────────────────────────────
novel_results = []
for bf in BATCH_FILES:
    data = load(bf)
    novel_results.extend(data['dopant_results'])
    print(f'{bf.name}: {len(data["dopant_results"])} dopants')

print(f'Novel total: {len(novel_results)} dopants')

# ── Save novel-14 combined ────────────────────────────────────────────────────
novel14 = {
    'concentration': 0.10,
    'mlip': 'mace-mp-0',
    'n_sqs_realisations': 5,
    'supercell': [4, 4, 4],
    'notes': (
        '14 novel dopants (Stage 4 viability survivors, S excluded). '
        'Colab L4 GPU, 4x4x4 supercell, fmax=0.10 eV/A.'
    ),
    'dopant_results': novel_results,
    'n_dopants': len(novel_results),
}
RESULTS.mkdir(parents=True, exist_ok=True)
with open(OUT_NOVEL, 'w') as f:
    json.dump(novel14, f, indent=2)
print(f'\n✓ Saved novel-14: {OUT_NOVEL}')

# ── Merge known-8 + novel-14 → all-22 ────────────────────────────────────────
all_results = known8['dopant_results'] + novel_results
all22 = known8.copy()
all22['dopant_results'] = all_results
all22['n_dopants']      = len(all_results)
all22['notes'] = (
    'Merged: 8 known (Kaggle T4x2, 4x4x4) + 14 novel (Colab L4, 4x4x4). '
    'Stage 4 viability filter applied: Cr/Sb/As/Os/U excluded. S excluded (non-metal).'
)

with open(OUT_ALL, 'w') as f:
    json.dump(all22, f, indent=2)
print(f'✓ Saved all-22 : {OUT_ALL}')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\nTotal dopants in merged file: {len(all_results)}')
n_converged = sum(r.get('n_converged', 0) for r in all_results)
n_total     = len(all_results) * 5
print(f'Total converged SQS: {n_converged}/{n_total} ({100*n_converged/n_total:.0f}%)')
print('\nDopants:')
for r in all_results:
    print(f'  {r["dopant"]:4s}  n_converged={r.get("n_converged", "?")}')

print('\nNext step:')
print('  python -m evaluation.eval_accuracy \\')
print('      --results evaluation/results/rq2_disorder_all22.json \\')
print('      --save evaluation/results/rq3_accuracy_all22.json')
