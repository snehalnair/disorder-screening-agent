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

**Suggestive but not yet conclusive:** Zirconium — which ordered-cell screening would rank as the standout voltage champion (+0.46 V above the field) — collapses to the middle of the pack in disordered simulations, indistinguishable from Al, Ti, Mg, Ga, and Fe. The overall voltage ranking correlation (rho = -0.33) is not yet statistically significant at n=8 dopants, but the disordered voltage spread (0.023 V) matches experimental reality (~0.1 V) far better than the ordered spread (0.47 V). **Important caveat:** we cannot yet distinguish whether the Zr anomaly is corrected by disorder specifically or simply by using a larger supercell — the ordered Zr voltage may itself be a small-cell artefact that would vanish in a 4x4x4 cell even without SQS. Phase 2 includes an ordered-cell size control to resolve this.

**Not yet computed:** Volume change on delithiation and Li/Ni exchange energy were held fixed in Phase 1. Position-only relaxation (no cell relaxation) means volume change is identically zero for all dopants. Li/Ni exchange energy — the most disorder-sensitive and commercially relevant property — requires Ni in the parent structure and could not be computed using the LiCoO2 proxy. These two properties carry 40% of the composite ranking weight (volume: 15%, Li/Ni exchange: 25%) and are expected to show the largest disorder sensitivity. They are the primary targets of Phase 2.

## Phase 1 Limitations and Phase 2 Fixes

| Phase 1 Limitation | Phase 2 Fix | Timeline |
|---------------------|-------------|----------|
| LiCoO2 proxy (no Li/Ni exchange energy) | NMC811 parent structure | 3-4 months |
| 8 dopants (voltage result not significant) | All 29 survivors (significance at n=29) | 1-2 months |
| 32-atom cell with 1 substituted site — all 5 SQS realisations are geometrically identical (zero inter-realisation variance), so we are comparing two specific site placements, not true ordered-vs-disordered statistics | 4x4x4 supercell (256 atoms, 6-7 substitutions at 10%) where SQS realisations are genuinely distinct | Concurrent |
| No cell relaxation (volume change = 0) | Enable FrechetCellFilter | 1 month |

All infrastructure is built. Phase 2 is execution, not development.

## Publication Targets

1. **"Disorder-aware dopant screening reveals ranking inversions in NMC811"** — *Chemistry of Materials* or *Digital Discovery*
2. **"Generalisable disorder-aware screening across cathode families"** (LNMO spinel extension) — *npj Computational Materials*

## Why This Is a Programme, Not a Paper

The methodology applies to any material where disorder matters: solid electrolytes (LLZO), thermoelectrics, high-entropy alloys. The pipeline architecture supports agentic extension — conversational screening assistants for experimentalists, closed-loop autonomous screening, and integration with robotic synthesis platforms. The group that establishes disorder-aware screening as standard practice defines how the next generation of computational materials discovery is done.

**Ask:** Endorsement for Phase 2, student/postdoc allocation, and a department seminar slot to engage experimentalists with ground truth data.
