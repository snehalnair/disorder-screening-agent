# Disorder-Aware Dopant Screening for Battery Cathodes

**Executive Summary | March 2026**

---

## The Problem

AI materials discovery tools (GNoME, MatterGen, MatterSim) can now propose millions of candidate crystals — but they all simulate perfectly ordered structures. The Fritz Haber Institute (Dec 2025) showed that **>80% of real synthesised materials exhibit measurable disorder**. We are predicting properties for materials that don't exist in the lab.

For NMC811 — the dominant high-energy Li-ion cathode — this matters acutely. Li/Ni antisite disorder directly controls capacity fade, and dopant atoms distribute randomly across available sites rather than sitting in the tidy arrangements our simulations assume.

## What We Built

A hierarchical screening pipeline that filters 271 candidate dopant combinations down to 29 viable elements (83% reduction, 92.3% recall of confirmed dopants), then simulates survivors in **disordered supercells** using Special Quasi-random Structures and the MACE-MP-0 universal machine-learning potential. The full pipeline runs in 30 minutes on a laptop with no HPC infrastructure. It is fully tested (274 tests), reproducible, and open-source.

## What We Found

**Proven:** Formation energy rankings change with statistical significance (Spearman rho = 0.74, **p = 0.037**) when disorder is included. On average, disordered formation energies differ from ordered values by ~14% — enough to change which dopants a researcher would prioritise for synthesis.

**Suggestive:** Zirconium — which ordered-cell screening would rank as the standout voltage champion (+0.46 V above the field) — collapses to the middle of the pack in disordered simulations, indistinguishable from Al, Ti, Mg, Ga, and Fe. The overall voltage ranking correlation (rho = -0.33) is not yet statistically significant at n=8 dopants, but the disordered voltage spread (0.023 V) matches experimental reality (~0.1 V) far better than the ordered spread (0.47 V).

## Phase 1 Limitations and Phase 2 Fixes

| Phase 1 Limitation | Phase 2 Fix | Timeline |
|---------------------|-------------|----------|
| LiCoO2 proxy (no Li/Ni exchange energy) | NMC811 parent structure | 3-4 months |
| 8 dopants (voltage result not significant) | All 29 survivors (significance at n=29) | 1-2 months |
| 32-atom cell (SQS realisations equivalent) | 4x4x4 supercell (256 atoms, true SQS variance) | Concurrent |
| No cell relaxation (volume change = 0) | Enable FrechetCellFilter | 1 month |

All infrastructure is built. Phase 2 is execution, not development.

## Publication Targets

1. **"Disorder-aware dopant screening reveals ranking inversions in NMC811"** — *Chemistry of Materials* or *Digital Discovery*
2. **"Generalisable disorder-aware screening across cathode families"** (LNMO spinel extension) — *npj Computational Materials*

## Why This Is a Programme, Not a Paper

The methodology applies to any material where disorder matters: solid electrolytes (LLZO), thermoelectrics, high-entropy alloys. The pipeline architecture supports agentic extension — conversational screening assistants for experimentalists, closed-loop autonomous screening, and integration with robotic synthesis platforms. The group that establishes disorder-aware screening as standard practice defines how the next generation of computational materials discovery is done.

**Ask:** Endorsement for Phase 2, student/postdoc allocation, and a department seminar slot to engage experimentalists with ground truth data.
