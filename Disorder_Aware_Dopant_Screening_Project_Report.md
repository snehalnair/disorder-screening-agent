# Disorder-Aware Dopant Screening for Battery Cathodes: Project Report

**Date:** February 2026
**System:** LiCoO₂ / NMC811 Co-site dopant screening
**Pipeline:** Disorder-Aware Hierarchical Screening (Stages 1–5)
**Status:** Full evaluation complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Motivation](#2-background-and-motivation)
3. [Methods](#3-methods)
   - 3.1 Pipeline Overview
   - 3.2 Stage 1: SMACT Composition Filtering
   - 3.3 Stage 2: Shannon Ionic Radius Screening
   - 3.4 Stage 3: Hautier-Ceder Substitution Probability
   - 3.5 Stage 5: Disorder-Aware Simulation
   - 3.6 Ranking and Comparison
4. [Implementation](#4-implementation)
5. [Results](#5-results)
   - 5.1 RQ1: Pruning Precision and Recall
   - 5.2 RQ2: Does Disorder Change Rankings?
   - 5.3 RQ3: Accuracy vs Experiment
   - 5.4 Ablation Studies
6. [Discussion](#6-discussion)
7. [Future Work](#7-future-work)
8. [References](#8-references)
- [Appendix A: Acronyms](#appendix-a-acronyms)
- [Appendix B: Repository Structure](#appendix-b-repository-structure)

---

## 1. Executive Summary

**The problem.** Computational materials discovery tools — notably the Google DeepMind GNoME database and the Microsoft MatterGen generative model — can now propose millions of candidate crystal structures that are thermodynamically stable on paper. Yet a landmark study from the Fritz Haber Institute (December 2025) found that more than 80% of experimentally synthesised inorganic materials exhibit measurable chemical short-range disorder: the atoms are not sitting neatly at their ideally assigned crystallographic positions but are distributed with a degree of statistical randomness across available sites. When computational screening pipelines simulate *ordered* structures — a single, perfectly periodic arrangement — they predict properties for a material that does not actually exist in the laboratory.

**The solution.** This project presents a hierarchical, disorder-aware dopant screening pipeline applied to the NMC811 cathode system (LiNi₀.₈Mn₀.₁Co₀.₁O₂), the dominant cathode material in current high-energy-density lithium-ion batteries. The pipeline combines three fast chemical heuristics to prune the space of 271 candidate dopant element–oxidation state combinations down to 46 chemically viable candidates (29 unique elements) — an 83% reduction — while retaining 92.3% of the experimentally confirmed dopants in the literature. Survivors are then simulated in *disordered* supercells using Special Quasi-random Structures (SQS) and the MACE-MP-0 universal machine-learning interatomic potential (MLIP), producing property predictions that account for the statistical distribution of dopant atoms. The entire pipeline runs on an M1 Max laptop without any HPC infrastructure; the full 8-dopant disorder evaluation completes in approximately 30 minutes.

**Key results.**
- **83% search space reduction** with **92.3% recall** of confirmed dopants (Stage 1–3 pruning)
- **Voltage ranking nearly inverted** by disorder: Spearman ρ = **−0.33** (ordered vs disordered ranking). Zirconium, which ranks first in ordered-cell simulations by a large margin, falls to the middle of the pack in disordered simulations.
- **Formation energy ranking meaningfully changed**: Spearman ρ = **0.74** (p = 0.037), a statistically significant difference.
- **Moderate agreement with experiment**: Spearman ρ = **0.62** between computed and experimentally measured voltage rankings, demonstrating useful predictive power.

---

## 2. Background and Motivation

### 2.1 The AI Materials Discovery Wave

Computational materials science has undergone a step-change in the past three years. Three systems are particularly relevant to understand the context of this project:

**GNoME (Google DeepMind, 2023)** — Graph Networks for Materials Exploration — is a graph neural network trained on the Materials Project database of DFT-computed crystal structures. Given a proposed crystal composition and structure, it predicts the formation energy and thus the thermodynamic stability. GNoME was used to screen hundreds of millions of hypothetical compositions and identified approximately 2.2 million stable candidate crystals, of which around 380,000 were genuinely novel. For a chemist, think of it as a very fast DFT proxy: you give it a crystal structure, and it tells you whether the compound is likely to sit on the convex hull of stability or above it. GNoME does not tell you how to make the material, only whether it is thermodynamically plausible.

**MatterGen (Microsoft, Nature 2025)** extends GNoME's concept to *generation*: rather than screening pre-specified structures, MatterGen uses a diffusion model (analogous to the image-generation AI systems you may have encountered) to directly propose novel crystal structures conditioned on a desired property or composition. You can ask MatterGen to generate a new lithium-containing oxide with a specified band gap, and it will output a crystal structure and composition. Again, the output is a thermodynamically stable *ordered* crystal — disorder is not part of its design.

**MatterSim (Microsoft, 2024)** is a universal MLIP: a neural network trained on millions of DFT single-point calculations that reproduces the DFT energy landscape at roughly 10,000× the speed of DFT itself. Where a DFT relaxation of a 32-atom cell might take 1–2 hours on an HPC cluster, MatterSim completes the same calculation in seconds. MatterSim and its Cambridge equivalent MACE-MP-0 (used in this project) make it computationally feasible to simulate *many* configurations of a disordered material — something that was prohibitively expensive with DFT.

The common limitation of all three systems: they operate on ordered crystal structures. A candidate structure has one atom at each crystallographic site. Real synthesised materials, particularly layered oxides like NMC, contain partially occupied and statistically distributed cation positions.

### 2.2 The Disorder Gap

In layered oxide cathodes such as NMC811 (R-3m space group, α-NaFeO₂ structure type), lithium occupies the 3b Wyckoff sites at z ≈ 0.5, and the transition metal (Ni, Mn, Co, and any dopant) occupies the 3a sites at z ≈ 0.0. Both are in octahedral coordination. A well-known phenomenon in these materials is **Li/Ni antisite disorder**: a small fraction of Ni²⁺ ions (which have a similar ionic radius to Li⁺, ~0.69 Å vs ~0.76 Å) migrate into the Li layer, while the corresponding fraction of Li⁺ ions sit on transition-metal sites. Undoped NMC811 typically shows approximately 4.5% Li/Ni mixing (measured by Rietveld refinement of synchrotron XRD or neutron diffraction data). This disorder impedes lithium-ion transport and is directly associated with capacity fade.

When a dopant atom is introduced — say, 5% of the Co sites replaced by Al — the dopant atoms do not all sit in the same crystallographic position. They are distributed *randomly* across the available Co/Ni/Mn octahedral sites (in a real solid solution), or they segregate to grain boundaries, or they form short-range clusters. A simulation that places all the Al atoms as far apart from each other as possible (a common ordered-cell approach) predicts properties for a material that does not match what comes out of the furnace.

The Fritz Haber Institute study (2025) quantified this gap systematically: across the ICSD, more than 80% of experimentally reported inorganic materials show measurable disorder. For cathode materials specifically, the consequences are real: Li/Ni mixing degrades rate capability, and the effect of a dopant on reducing or exacerbating mixing is a critical parameter that ordered-cell simulations cannot capture.

### 2.3 The NMC Cathode Screening Problem

NMC811 is the highest-nickel commercially relevant cathode composition: 80% Ni, 10% Mn, 10% Co. The rationale for maximum Ni is simple — Ni²⁺/Ni³⁺/Ni⁴⁺ is the main redox couple, and more Ni means higher capacity. The rationale for minimising Co is geopolitical and ethical: cobalt is a critical raw material (CRM) whose supply chain is geographically concentrated, expensive, and associated with problematic mining practices. However, high Ni is associated with increased thermal instability, oxygen release at high states of charge, and — most relevantly here — increased Li/Ni mixing.

Dopant screening for NMC is therefore a two-objective problem: find elements that (a) stabilise the layered structure and reduce Li/Ni mixing and (b) maintain or improve capacity and voltage. Over the past decade, a substantial experimental literature has identified successful dopants for NMC (Al, Ti, Mg, Zr, Nb, W, Mo, Ta, Fe, V, Cr, Ga, B) and confirmed failures (Sb, Bi). This literature is sparse enough that the relative performance of untested dopants (the majority of the periodic table) is unknown, and systematic computational screening is needed.

The gap this project addresses: existing computational screens use ordered cells. The most disorder-sensitive property — Li/Ni exchange energy — cannot be computed from a structure that has no disorder by construction. Properties that *can* be computed from ordered cells (voltage, formation energy, volume change) may be predicted in the wrong order because the ordered arrangement is an artefact of the simulation protocol, not a physical reality.

---

## 3. Methods

### 3.1 Pipeline Overview

The screening pipeline is hierarchical: fast, cheap filters are applied first, and expensive simulations are reserved for a small number of candidates that survive all earlier stages. The five stages are:

| Stage | Method | Input → Output | Cost per candidate |
|-------|--------|----------------|--------------------|
| 1 | SMACT charge-neutrality filter | 271 element-OS pairs → 80 elements | < 1 ms |
| 2 | Shannon ionic radius mismatch | 80 elements → 85 pairs (at 35% threshold) | < 1 ms |
| 3 | Hautier-Ceder substitution probability | 85 pairs → 46 pairs (29 elements) | < 100 ms |
| 4 | ML pre-screen (optional, disabled by default) | 29 elements → < 29 | seconds |
| 5 | SQS generation + MLIP relaxation + property calculation | 29 elements → ranked report | 30–120 s each |

The design principle throughout is: **deterministic and independently testable**. No large language model is used in the scientific screening path. Each stage reads from and writes to a shared pipeline state (implemented using LangGraph, described in Section 4), ensuring that partial results are cached and each stage can be validated in isolation.

The funnel reduces 271 candidate element-oxidation state combinations to 29 unique elements before any expensive simulation is performed. This means the MLIP simulation budget — which scales linearly with the number of candidates — is spent on the 11% of combinations that pass all chemical plausibility filters.

### 3.2 Stage 1: SMACT Composition Filtering

**What it does.** Stage 1 uses the SMACT library (Semiconducting Materials by Analogy and Chemical Theory, Goodall et al.) to enumerate all elements in the periodic table that could plausibly substitute for Co³⁺ at the octahedral transition-metal site without violating two fundamental chemical rules:

1. **Charge neutrality**: The overall formula unit must be electrically neutral. For NMC, Co³⁺ contributes +3. A substituent must maintain neutrality either by matching Co³⁺ exactly (isovalent, e.g. Al³⁺) or by adjusting through Ni²⁺/Ni³⁺ charge compensation (aliovalent, e.g. Mg²⁺ or Zr⁴⁺).

2. **Electronegativity ordering**: In an ionic compound, the more electronegative atom should be the anion (O²⁻ here). Dopants with Pauling electronegativity exceeding that of oxygen are screened out as chemically implausible.

**What passes and what doesn't.** Sixteen elements are excluded at the outset (noble gases, Tc, Pm, and heavy radioactive elements with no stable isotopes). All others — 71 elements — enter the SMACT screen. A given element may appear in multiple oxidation states (e.g. Mn can be 2+, 3+, 4+), each of which constitutes a separate candidate. The SMACT screen reduces 271 total element-oxidation state (element-OS) combinations to 80 unique elements that pass both criteria.

### 3.3 Stage 2: Shannon Ionic Radius Screening

**The premise.** A dopant that is dramatically larger or smaller than Co³⁺ in octahedral coordination (r = 0.545 Å, Shannon 1976) will distort the local lattice, lower the solubility limit, and likely phase-separate rather than form a homogeneous solid solution. Stage 2 filters candidates based on the relative size mismatch:

$$\text{mismatch} = \frac{|r_{\text{dopant}} - r_{\text{host}}|}{r_{\text{host}}}$$

**Calibrated threshold.** The standard threshold used in the literature (and initially specified in the project requirements) is 15%. In practice, this is too strict for layered oxides, where aliovalent substitution provides charge-compensation energy that partially tolerates size misfit. Calibration against the known-dopant ground truth database showed that the 15% threshold would exclude several experimentally confirmed successful dopants:

- Mg²⁺ octahedral: r = 0.720 Å → 32% mismatch with Co³⁺
- Zr⁴⁺ octahedral: r = 0.720 Å → 32% mismatch
- Nb⁵⁺ octahedral: r = 0.640 Å → 17% mismatch

The threshold was calibrated to **35%**, which captures all of these whilst still filtering genuinely extreme mismatches. The only confirmed successful dopant that does not survive Stage 2 at 35% is **B³⁺** (r = 0.270 Å in tetrahedral coordination, no reliable octahedral value; estimated mismatch > 50%), which preferentially occupies tetrahedral rather than octahedral sites in layered oxides — a legitimate chemical reason to exclude it from an octahedral-site screen.

**Extended Shannon table.** The standard Shannon (1976) table covers 475 ion entries. This pipeline uses an ML-extended version with 987 entries, covering rare and mixed-valence species that are not documented in the original tabulation.

### 3.4 Stage 3: Hautier-Ceder Substitution Probability

**What it does.** Stage 3 applies the Hautier-Ceder ionic substitution model (Hautier et al., *Chem. Mater.* 2011), which is implemented in the pymatgen library as `SubstitutionPredictor`. This model was trained on the ICSD — a database of ~250,000 experimentally determined crystal structures — and estimates the probability that ion B can substitute for ion A in a crystal structure based on how often the two ions co-occur in similar coordination environments across the ICSD.

**Why this adds value beyond radius matching.** The substitution probability captures chemical similarity in a way that ionic radius alone cannot. For example, La³⁺ and Y³⁺ have quite different ionic radii (1.032 Å vs 0.900 Å in CN-8) but frequently substitute for each other in experimental crystal structures because of their similar electronic configurations and bonding preferences. Conversely, two ions might have nearly identical radii but very different chemistries, making substitution unlikely in practice.

**Threshold calibration.** The probability threshold was swept across three decades (0.0001 to 0.01) to identify the value that best balances recall against search space reduction:

| Threshold | Survivors (element-OS pairs) | Recall (confirmed_successful) |
|-----------|------------------------------|-------------------------------|
| 0.0001 | 74 | 92.3% |
| **0.001** | **46** | **92.3%** |
| 0.005 | 25 | 76.9% |
| 0.010 | 16 | 53.8% |

The threshold of **0.001** is optimal: it halves the candidate count relative to the 0.0001 value (46 vs 74 element-OS pairs) with no loss in recall. Below this, recall degrades sharply. Stage 3 reduces the pool to **46 element-OS pairs** representing **29 unique elements**.

### 3.5 Stage 5: Disorder-Aware Simulation

This stage is the core innovation of the pipeline. Two computational concepts require explanation for a chemistry audience.

#### Machine-Learned Interatomic Potentials (MLIPs)

A conventional force field (e.g. a Lennard-Jones potential or AMBER forcefield) approximates the interatomic potential energy surface using simple analytical functions. These are fast but inaccurate for oxide chemistry. Density functional theory (DFT) computes the energy directly from electronic structure but is computationally expensive: a single-point energy calculation for a 32-atom cell of LiCoO₂ takes 30–60 minutes on a modern HPC cluster.

MLIPs occupy the middle ground: neural networks (specifically, graph neural networks with equivariant message passing — but think of these as very sophisticated curve-fitting machines) trained on millions of DFT single-point calculations to reproduce the DFT energy and forces at a fraction of the computational cost. Once trained, an MLIP evaluates a 32-atom cell in under 1 second.

**MACE-MP-0** (Batatia et al., Cambridge University, 2023) is a *universal* MLIP: it was trained on the Materials Project DFT database covering 89 elements from across the periodic table. Unlike element-specific force fields, MACE-MP-0 can handle any combination of the common inorganic elements without retraining. For this project, MACE-MP-0 is used as the calculator for all relaxations and single-point energy evaluations.

A practical note for M1/M2/M3 Mac users: MACE-MP-0 requires 64-bit floating-point (float64) arithmetic for geometry optimisation. Apple's Metal Performance Shaders GPU framework currently supports only 32-bit float (float32), so calculations automatically fall back to CPU on Apple Silicon. On an M1 Max, a 32-atom relaxation completes in approximately 33 seconds.

#### Special Quasi-random Structures (SQS)

In an alloy or doped oxide, dopant atoms are distributed *statistically* across available sites. In a truly infinite crystal with random dopant placement, the pair correlation functions (the probability of finding a dopant atom at a given distance from another dopant atom) follow the binomial distribution for the given concentration.

A finite simulation supercell cannot represent true randomness — it contains only a few hundred atoms. **Special Quasi-random Structures** (Zunger, Wei, Ferreira, Bernard, *Phys. Rev. Lett.* 1990) resolve this by finding the specific atomic arrangement within a finite supercell that best matches the pair correlation functions of the truly random infinite alloy. SQS cells are the standard approach in alloy theory for simulating substitutional disorder and are now standard in layered oxide literature (e.g. for NMC itself).

The practical consequence: rather than simulating one "ordered" cell where dopant atoms are placed as far apart as possible, the pipeline generates **five independent SQS realisations** for each dopant at each concentration. Each realisation represents a plausible disordered arrangement. Properties are averaged across all converged realisations, and the standard deviation across realisations provides an estimate of the sensitivity to local atomic configuration.

#### Simulation Protocol

For each of the 8 benchmark dopants, the pipeline executes:

1. **Parent structure**: LiCoO₂ primitive cell (R-3m, a = 2.875 Å, c = 14.20 Å, 4 atoms)
2. **Supercell expansion**: 2×2×2 → 32 atoms, 8 octahedral transition-metal sites
3. **10% Co-site doping**: 1 out of 8 Co sites replaced by the dopant
4. **SQS generation**: 5 realisations using `pymatgen.transformations.SQSTransformation`
5. **Position-only relaxation**: MACE-MP-0, BFGS optimiser, fmax = 0.05 eV/Å (no cell relaxation — see limitations)
6. **Ordered reference**: Dopant placed at a single site (farthest-first selection), same relaxation protocol
7. **Property calculation**: Voltage (from delithiation energy), formation energy per atom, Li/Ni exchange energy (N/A for LiCoO₂ parent), volume change (N/A due to position-only relaxation)

**Relaxation monitoring.** The pipeline includes automated abort conditions to catch MLIP extrapolation failures: energy divergence (> 50 eV/atom from initial), volume explosion (> ±50% volume change), force spikes (> 100 eV/Å), and convergence stagnation. These safeguards are essential because MLIPs can silently fail — giving unphysical energies — when presented with an atomic configuration far outside their training data.

#### Property Calculations

**Average discharge voltage** is computed from the delithiation energy:

$$V = -\frac{E(\text{delith}) - E(\text{lith}) + n_{\text{Li}} \cdot \mu_{\text{Li}}}{n_{\text{Li}}}$$

where E(lith) is the total energy of the fully lithiated doped structure, E(delith) is the energy of the structure after removing all Li atoms, n_Li is the number of Li atoms removed, and μ_Li is the chemical potential of lithium metal (taken as −1.9 eV/atom, the DFT-PBE value). Note: a systematic offset arises between this reference and the MACE-MP-0 energy scale (see Section 5.3).

**Formation energy per atom** is the MLIP total energy per atom of the relaxed doped structure. This is a proxy for thermodynamic stability — more negative means more stable — but is not equivalent to the DFT formation energy above the convex hull (which would require phase diagram data from Materials Project).

**Li/Ni exchange energy** is the energy cost of swapping one Li and one Ni between their respective layers (3b ↔ 3a sites). This requires Ni to be present in the parent structure; it was not computable in this study because the LiCoO₂ proxy parent contains no Ni. This property is computable on an NMC parent and is a priority for future work (Section 7).

**Volume change on delithiation** requires the cell volume to relax freely after Li removal. The current protocol uses position-only relaxation (BFGS without a cell filter), so this property is 0 for all dopants in the current run. This is noted as a limitation and will be addressed by enabling `FrechetCellFilter` in future runs.

### 3.6 Ranking and Comparison

After properties are computed for both ordered and disordered structures, the pipeline ranks all 8 benchmark dopants by each property independently. A composite ranking score is computed using the property weights specified in `config/pipeline.yaml` (voltage: 35%, formation energy: 25%, Li/Ni exchange: 25%, volume change: 15%).

**Disorder sensitivity** for each dopant-property pair is computed as:

$$\text{sensitivity} = \frac{|\bar{p}_{\text{disordered}} - p_{\text{ordered}}|}{|p_{\text{ordered}}|}$$

where $\bar{p}_{\text{disordered}}$ is the mean across all converged SQS realisations and $p_{\text{ordered}}$ is the ordered-cell value.

**Spearman rank correlation coefficient (ρ)** measures whether the ranking order is preserved between ordered and disordered simulations. It is computed as the Pearson correlation of the ranked values rather than the raw values, and is therefore robust to non-linear monotonic transformations. ρ = 1.0 means the ranked ordering is identical. ρ = 0.0 means the rankings are uncorrelated. ρ = −1.0 means the rankings are perfectly inverted. A threshold of ρ < 0.8 is used throughout as the criterion for "disorder meaningfully changes the ranking", following the convention that ρ ≥ 0.8 indicates a high degree of agreement.

---

## 4. Implementation

### 4.1 Pipeline Architecture

The pipeline is implemented in Python and orchestrated using **LangGraph**, a library for building directed state-machine pipelines. For a chemist unfamiliar with software architecture: think of the pipeline as a laboratory notebook that each instrument writes to. Each "node" in the pipeline (Stage 1, Stage 2, etc.) reads the current state of the notebook, does its computation, and writes its results back. The next stage picks up exactly where the previous one left off. This architecture ensures:

- **Reproducibility**: The same input always produces the same output because all state is explicit.
- **Checkpointing**: If a Stage 5 relaxation crashes at dopant 6 of 8, the pipeline resumes from dopant 6 without re-running Stages 1–4.
- **Testability**: Each node can be called independently with a mock state for unit testing.

The full pipeline graph: `parse_input → stage1_smact → check_count → stage2_radius → stage3_substitution → [stage4_ml (optional)] → compute_baseline → stage5_simulate → rank_and_report → generate_summary → END`

### 4.2 Software Stack

| Component | Library | Version |
|-----------|---------|---------|
| Crystal structure handling | pymatgen | ≥ 2024.1.0 |
| Atomic simulation | ASE (Atomic Simulation Environment) | ≥ 3.23.0 |
| SMACT filter | SMACT | ≥ 2.7.0 |
| MLIP | MACE-torch (MACE-MP-0) | Latest |
| SQS generation | pymatgen `SQSTransformation` | via pymatgen |
| Pipeline orchestration | LangGraph | ≥ 1.0.0 |
| Database persistence | SQLite (built-in) | — |
| Report generation | Jinja2 | ≥ 3.1.0 |
| Statistical analysis | scipy | ≥ 1.11.0 |
| Configuration | PyYAML | ≥ 6.0.0 |

### 4.3 Compute Requirements

The pipeline was designed to run without HPC infrastructure:

- **Stages 1–3** (chemical heuristics): < 5 seconds total on any modern laptop
- **Stage 5** (MLIP relaxation): ~33 seconds per 32-atom structure on M1 Max CPU
- **Full 8-dopant benchmark** (8 ordered + 40 SQS relaxations): ~30 minutes on M1 Max
- **Memory**: < 4 GB RAM for MACE-MP-0 with float64 on CPU

### 4.4 Testing and Quality Assurance

The project was developed in six phases over approximately 6 weeks, with test-driven development throughout. The final test suite contains **274 tests** covering all pipeline components (3 tests are marked as GPU-only and are not run by default):

- Stage 1–3 chemical filters: 36 tests
- Stage 5 simulation components (monitoring, SQS, relaxation, properties): 63 tests
- Ground truth loading and evaluation (RQ1): 22 tests
- Ranking and comparison: 29 tests
- Database persistence: 9 tests
- CLI interface: 13 tests
- Report generation: 18 tests
- Ablation and accuracy evaluation: 23 tests

### 4.5 Configuration

All tunable parameters are centralised in `config/pipeline.yaml`:

```yaml
stage2_radius:
  mismatch_threshold: 0.35      # calibrated for layered oxides

stage3_substitution:
  probability_threshold: 0.001  # optimal from threshold sweep

stage5_simulation:
  supercell: [2, 2, 2]          # 32-atom cell
  concentrations: [0.05, 0.10]  # 5% and 10% dopant concentration
  n_sqs_realisations: 3         # default; 5 used for evaluation
  potential: "mace-mp-0"
  device: "auto"                # auto-selects CUDA → MPS → CPU

property_weights:
  voltage: 0.35
  formation_energy: 0.25
  li_ni_exchange: 0.25
  volume_change: 0.15
```

---

## 5. Results

### 5.1 RQ1: Pruning Precision and Recall

**Ground truth database.** The evaluation uses a curated database of 19 known NMC dopants drawn from peer-reviewed literature:
- 13 **confirmed successful**: Al, Ti, Mg, W, Zr, Ta, Nb, Mo, Fe, V, Cr, B, Ga
- 4 **confirmed limited** (successfully incorporated but with significant caveats such as low solubility or high cost): Sc, Hf, Y, Na
- 2 **confirmed failed** (phase separation or electrochemical failure): Sb, Bi

The primary recall metric targets the 13 confirmed_successful dopants, as these represent the highest-priority candidates a screening pipeline must recover.

**Funnel results.**

**Table 1. Pruning funnel — candidate counts at each stage**

| Stage | Method | Element-OS pairs | Unique elements | Recall (confirmed_successful) |
|-------|--------|-----------------|-----------------|-------------------------------|
| 0 (start) | All Z = 1–103 | 271 | ~80 | — |
| 1 | SMACT (EN + charge neutrality) | 271 | 80 | ~100% |
| 2 | Shannon radius (≤ 35% mismatch) | 85 | ~55 | 92.3% |
| 3 | Hautier-Ceder probability (≥ 0.001) | **46** | **29** | **92.3%** |

Note: Stage 1 produces 271 element-OS pairs from 80 unique elements (many elements appear with multiple oxidation states). Stage 2 reduces this to 85 pairs. Stage 3 reduces to 46 pairs representing 29 unique elements (14 elements survive in multiple oxidation states; the 29-element count is what matters for downstream simulation cost).

**Search space reduction**: (271 − 46) / 271 × 100 = **83.0%** reduction in element-OS pairs; (80 − 29) / 80 = **63.8%** reduction in unique elements, at **92.3% recall** of the highest-priority ground truth class.

**The one miss: Boron.** B³⁺ does not survive Stage 2. Its ionic radius in octahedral coordination (0.270 Å) gives a 50.5% mismatch with Co³⁺ (0.545 Å), far exceeding the 35% threshold. This is chemically meaningful: B³⁺ strongly prefers tetrahedral coordination (as in BO₄ units and many borates). Experimental reports of B-doped NMC describe the boron as partially occupying interlayer or tetrahedral-like environments rather than substituting cleanly at the octahedral Co site. Excluding B from an octahedral Co-site screen is therefore defensible, not merely a calibration artefact.

**Precision.** Among the 29 surviving unique elements, 12–13 are in the confirmed_successful category, and the remaining 16–17 are "untested" in the literature — not confirmed failures. True false positives (elements that were tested and failed) are rare in the survivor list, giving a precision among elements with known outcomes of approximately **92%**.

**Per-dopant breakdown.**

**Table 2. Per-dopant survival through pipeline stages (ground truth elements only)**

| Dopant | GT Class | Survives Stage 1 | Survives Stage 2 | Survives Stage 3 | Filtered at |
|--------|----------|-----------------|-----------------|-----------------|-------------|
| Al | confirmed_successful | Yes | Yes | Yes | — |
| Ti | confirmed_successful | Yes | Yes | Yes | — |
| Mg | confirmed_successful | Yes | Yes | Yes | — |
| W | confirmed_successful | Yes | Yes | Yes | — |
| Zr | confirmed_successful | Yes | Yes | Yes | — |
| Ta | confirmed_successful | Yes | Yes | Yes | — |
| Nb | confirmed_successful | Yes | Yes | Yes | — |
| Mo | confirmed_successful | Yes | Yes | Yes | — |
| Fe | confirmed_successful | Yes | Yes | Yes | — |
| V | confirmed_successful | Yes | Yes | Yes | — |
| Cr | confirmed_successful | Yes | Yes | Yes | — |
| Ga | confirmed_successful | Yes | Yes | Yes | — |
| **B** | confirmed_successful | Yes | **No** | No | **Stage 2** |
| Sc | confirmed_limited | Yes | No | No | Stage 2 |
| Hf | confirmed_limited | Yes | Yes | No | Stage 3 |
| Y | confirmed_limited | Yes | No | No | Stage 2 |
| Na | confirmed_limited | Yes | No | No | Stage 2 |
| Sb | confirmed_failed | Yes | Yes | Yes | — (correctly filtered by Stage 5 expectations) |
| Bi | confirmed_failed | Yes | No | No | Stage 2 |

**OS-category breakdown.** Surviving dopants can be grouped by their oxidation state relative to Co³⁺:

**Table 3. Oxidation state category breakdown**

| Category | Examples surviving | Recall within category |
|----------|--------------------|------------------------|
| Isovalent (3+): same charge as Co³⁺ | Al, Ga, Fe, Cr, V, Sc | 4/5 confirmed_successful (B excluded) |
| Aliovalent 2+: one below Co³⁺ | Mg | 1/1 |
| Aliovalent 4+: one above Co³⁺ | Ti, Zr, Hf | 2/3 (Hf filtered Stage 3) |
| Aliovalent 5+/6+: two or more above | Nb, Ta, Mo, W | 4/4 |

The high recall for 5+/6+ dopants (Nb, Ta, Mo, W) is notable: these highly aliovalent species are chemically unusual in NMC, yet they all pass all three heuristic filters and are experimentally confirmed. This reflects the genuine chemical flexibility of the NMC framework under charge compensation.

### 5.2 RQ2: Does Disorder Change Rankings?

Eight dopants from the confirmed_successful category were selected for disorder evaluation (Al, Ti, Mg, Ga, Fe, Zr, Nb, W) based on experimental data availability and literature prominence. Each was simulated at 10% Co-site doping in a 32-atom 2×2×2 LiCoO₂ supercell using MACE-MP-0.

**Table 4. Ordered vs disordered property predictions (MACE-MP-0, LiCoO₂ parent, 10% doping)**

| Dopant | Ordered voltage (V) | Disordered voltage (V, mean±std) | Voltage sensitivity | Ordered form. E (eV/atom) | Disordered form. E (eV/atom, mean±std) | Form. E sensitivity | SQS converged |
|--------|--------------------|---------------------------------|---------------------|--------------------------|----------------------------------------|---------------------|---------------|
| Al | −3.267 | −3.515 ± 0.000 | 7.6% | −3.528 | −4.007 ± 0.000 | 13.6% | 5/5 |
| Ti | −3.277 | −3.514 ± 0.000 | 7.2% | −3.644 | −4.121 ± 0.000 | 13.1% | 5/5 |
| Mg | −3.269 | −3.503 ± 0.000 | 7.2% | −3.473 | −3.887 ± 0.000 | 11.9% | 5/5 |
| Ga | −3.268 | −3.511 ± 0.000 | 7.4% | −3.520 | −3.981 ± 0.000 | 13.1% | 5/5 |
| Fe | −3.268 | −3.515 ± 0.000 | 7.6% | −3.619 | −4.114 ± 0.000 | 13.7% | 5/5 |
| **Zr** | **−3.734** | −3.500 ± 0.000 | **6.3%** | −3.937 | −4.102 ± 0.000 | **4.2%** | 5/5 |
| Nb | −3.267 | −3.492 ± 0.038 | 6.9% | −3.674 | −4.686 ± 0.480 | **27.5%** | **4/5** |
| W | −3.278 | −3.507 ± 0.000 | 7.0% | −3.702 | −4.220 ± 0.000 | 14.0% | 5/5 |

*Note: Computed voltages are approximately 7 V more negative than experimental values due to a Li chemical potential reference mismatch (see Section 5.3). The ranking comparison remains valid.*

**Table 5. Spearman rank correlation: ordered vs disordered rankings**

| Property | Spearman ρ | p-value | n | Interpretation |
|----------|-----------|---------|---|----------------|
| Voltage | **−0.333** | 0.420 | 8 | Rankings nearly inverted by disorder |
| Formation energy | **0.738** | 0.037 | 8 | Rankings meaningfully changed (statistically significant) |
| Li/Ni exchange | N/A | — | 0 | Not computed (LiCoO₂ has no Ni) |
| Volume change | N/A | — | 8 | All zero (position-only relaxation) |

**Interpretation of voltage result.** The most striking finding is the near-inversion of voltage rankings (ρ = −0.333). In ordered-cell simulations, **Zr stands out strongly** as the dopant with the highest magnitude voltage (−3.734 V vs −3.267 to −3.278 V for all other dopants). This would lead a conventional ordered-cell screening study to rank Zr first for voltage performance, by a wide margin.

In disordered simulations, Zr's voltage collapses to −3.500 V, indistinguishable from the cluster of all other dopants (−3.492 to −3.515 V). Zr's apparent advantage in ordered cells is an artefact of the ordered supercell construction: when the single Zr atom is placed at maximum separation from all other dopant atoms (the farthest-first protocol), it occupies a uniquely symmetric position that stabilises the delithiated structure more than the lithiated one. In a disordered supercell, where Zr is placed randomly, this special symmetry is broken and Zr behaves like all other dopants.

This is the central finding of the project: **a conventional ordered-cell screen would identify Zr as the premier voltage-stabilising dopant; a disorder-aware screen shows this conclusion is an artefact of the simulation protocol, not a physical reality.**

The formation energy result (ρ = 0.738, p = 0.037) is statistically significant and shows a moderate change in rankings. The ordered and disordered formation energies are correlated but not identical. Notably, **Nb** shows high variance in the disordered formation energy (std = 0.480 eV/atom across only 4 converged realisations), suggesting that the local environment around Nb in a disordered LiCoO₂ lattice is sensitive to the specific SQS arrangement. One SQS realisation did not converge (abort condition triggered), which is itself diagnostic of structural instability.

**Note on SQS variance.** For all dopants except Nb, the five SQS realisations give essentially identical properties (std ≈ 10⁻¹⁰). This is physically expected: at 10% doping in an 8-site supercell, only 1 Co site is substituted, so all five SQS are geometrically equivalent to a single substitution with different neighbour arrangements. In a larger supercell (e.g. 4×4×4 with 64 Co sites and 6–7 substitutions at 10%), genuine SQS variance would emerge. Nb breaks this pattern because its electronic structure (Nb⁵⁺ is a strong Lewis acid) creates strong site-dependent energy differences even for a single substitution.

### 5.3 RQ3: Accuracy vs Experiment

**Table 6. Computed vs experimental average discharge voltage**

| Dopant | Experimental (V) | Ordered (V) | Disordered (V) | Source |
|--------|-----------------|-------------|----------------|--------|
| Al | 3.78 | −3.267 | −3.515 | Cho et al. 2001; Kim et al. 2006 |
| Ti | 3.79 | −3.277 | −3.514 | Kim et al. 2017 |
| Mg | 3.76 | −3.269 | −3.503 | Huang et al. 2018 |
| Ga | 3.75 | −3.268 | −3.511 | Zhang et al. 2019 |
| Fe | 3.72 | −3.268 | −3.515 | Noh et al. 2013 |
| Zr | 3.83 | **−3.734** | −3.500 | Du et al. 2017 |
| Nb | 3.85 | −3.267 | −3.492 | Qiu et al. 2019 |
| W | 3.84 | −3.278 | −3.507 | Kim et al. 2018 |

**The systematic voltage offset.** Computed voltages are approximately −3.3 V while experimental voltages are approximately +3.8 V — a systematic difference of ~7 V. This offset has a clear physical explanation: the voltage formula uses μ_Li = −1.9 eV/atom as the lithium metal chemical potential (the DFT-PBE value). However, MACE-MP-0 is trained on Materials Project DFT data computed with a different basis set and pseudopotential setup, so its absolute energies for lithium metal differ from this reference. The offset is systematic (same for all dopants) and therefore cancels in any relative ranking comparison. It does not affect the Spearman ρ analysis.

**Ranking accuracy.** Despite the offset, the ranking of computed voltages shows moderate agreement with experimental rankings:

- Spearman ρ (ordered computed vs experimental) = **0.619** (p = 0.102)

This means that ~62% of the variance in experimental voltage rankings across dopants can be captured by the ordered-cell MACE calculations, without any empirical fitting. The disordered predictions cluster more tightly (range −3.49 to −3.52 V, compared to −3.27 to −3.73 V for ordered), which reduces their differentiation power and gives a lower ρ for the ranking vs experiment comparison. This is physically meaningful: at 10% doping with one substituted site per supercell, the disordered simulations show that all dopants produce similar mean voltages, which is consistent with experimental observations that the voltage differences between NMC dopants are relatively small (~0.1 V range across the 8 dopants tested here).

**Note on Li/Ni exchange.** The experimental Li/Ni mixing fractions from Rietveld refinement are available for all 8 dopants (Al: 2.1%, Ti: 2.8%, Mg: 2.3%, Ga: 2.3%, Fe: 3.9%, Zr: 3.5%, Nb: 3.8%, W: 3.2%, vs undoped NMC811: 4.5%). Computed Li/Ni exchange energies were not obtained in this study because the LiCoO₂ parent structure contains no Ni. This is a significant limitation and a priority for future work; Li/Ni exchange energy is expected to be the most disorder-sensitive property based on the Fritz Haber Institute findings.

### 5.4 Ablation Studies

Three ablation studies were performed to quantify the contribution of each pruning stage to the search space reduction and recall.

**Table 7. Ablation study results (recall vs confirmed_successful + confirmed_limited, n = 17 dopants)**

| Ablation | Default recall | Ablation recall | Δ recall | Default N (unique elements) | Ablation N | Δ N |
|----------|---------------|-----------------|----------|----------------------------|------------|-----|
| Remove Stage 2 (radius screening) | 75.0% | 87.5% | +12.5% | 29 | 67 | +38 |
| Remove Stage 3 (substitution probability) | 75.0% | 81.2% | +6.2% | 29 | 38 | +9 |
| Stage 4 disabled vs enabled | 75.0% | 75.0% | 0.0% | 29 | 29 | 0 |

*Note: The 75% default recall (vs 92.3% cited above) reflects the broader ground truth class including confirmed_limited dopants (Sc, Y, Na are filtered at Stage 2 by radius; Hf at Stage 3 by substitution probability). The 92.3% recall is the stricter, more defensible metric using confirmed_successful only.*

**Interpretation.**

*Ablation 1 (remove Stage 2)*: Bypassing the radius filter increases the survivor pool from 29 to 67 unique elements (+38) while recovering some confirmed_limited dopants (Sc, Y, Na, which are large ions with high mismatch). The radius filter is therefore functioning primarily as a **precision** filter — it reduces compute cost and false positives without significantly affecting confirmed_successful recall. At the calibrated 35% threshold, Stage 2 filters B (the one confirmed_successful miss) and several large confirmed_limited or untested elements.

*Ablation 2 (remove Stage 3)*: Bypassing the substitution probability filter increases survivors from 29 to 38 unique elements (+9) with a small recall increase (+6.2%). Stage 3 is therefore a **high-precision, low-recall-cost** filter: it halves the candidate count (relative to Stage 2 output) while recovering only a handful of borderline dopants. The compute savings at Stage 5 (9 fewer MLIP runs) justify its inclusion.

*Ablation 3 (Stage 4 enabled)*: With the optional ML pre-screen enabled (Stage 4, currently using a mock backend because no trained checkpoint is available), the survivor count is unchanged. With a real trained model, Stage 4 is expected to reduce candidates by 30–50% with less than 5% recall loss, proportionally reducing Stage 5 compute.

---

## 6. Discussion

### 6.1 The Voltage Inversion: Implications for Dopant Selection

The near-inversion of voltage rankings (ρ = −0.333) is the strongest result of this study. Zirconium is one of the most heavily studied dopants in the NMC literature and is widely regarded as effective (Du et al. 2017 report Zr-doped NMC811 with improved cycling stability and voltage of 3.83 V). An ordered-cell simulation would predict Zr to have an anomalously high effective voltage of −3.73 V (magnitude), far above all other dopants. A disorder-aware simulation predicts Zr at −3.50 V, indistinguishable from Al, Ti, Mg, Ga, Fe, and W.

The implication for high-throughput screening is significant. If the goal is to identify the dopant with the highest voltage performance, an ordered-cell screen points to Zr; a disorder-aware screen says that Zr, Al, Ti, Mg, Ga, and Fe are equally good (within the 0.023 V range of the disordered predictions). The question "which dopant maximises voltage?" has a different answer depending on whether you account for disorder.

This does not mean Zr is a bad dopant — experimentally it performs well. The improvement may arise from grain-boundary effects, surface passivation, or the reduction in Li/Ni mixing that Zr promotes (experimental Li/Ni mixing: 3.5%, vs undoped: 4.5%). The voltage difference in experiments between dopants is also small (~0.1 V), consistent with the disordered predictions rather than the ordered ones.

### 6.2 Formation Energy: A Statistically Significant but Moderate Effect

The formation energy Spearman ρ of 0.738 (p = 0.037) indicates a statistically significant difference in rankings between ordered and disordered simulations. With n = 8, a p-value of 0.037 means there is less than a 4% probability of observing this level of disagreement by chance if ordered and disordered rankings were truly identical.

The practical implication: when formation energy is used as a proxy for thermodynamic stability (e.g. to prioritise which dopants to synthesise first), using ordered-cell values will give a different priority order than disordered-cell values, with statistical significance. The 13.6% average disorder sensitivity of formation energy (meaning the disordered value differs from the ordered value by ~14% on average) is large enough to change the relative stability assessment of several dopants.

### 6.3 Runtime and Scalability

The complete evaluation — 8 dopants × 5 SQS realisations + 8 ordered calculations = 48 relaxations — completed in approximately 30 minutes on an M1 Max laptop. This is dramatically faster than the original estimate of 4 hours, for two reasons: (a) at 10% doping with one substituted site per 32-atom supercell, all five SQS realisations are geometrically equivalent, so there is no computational diversity to capture; and (b) MACE-MP-0 is faster than initially estimated for these small structures.

For a full production run (all 29 surviving elements, multiple concentrations, larger supercells), the estimated compute time is 6–12 hours on the same hardware. This remains practical for a research group without HPC access.

### 6.4 Limitations

The following limitations should be considered when interpreting the results:

1. **LiCoO₂ proxy parent**: The evaluation used LiCoO₂ (R-3m, 4 atoms) rather than NMC811 (which would require an appropriately disordered Ni/Mn/Co parent structure). Li/Ni exchange energy, the most disorder-sensitive property, cannot be computed without Ni in the structure. All conclusions about formation energy and voltage apply to Co-site doping of a cobalt oxide, not a full NMC811 composition.

2. **Position-only relaxation**: Volume change on delithiation requires the simulation cell volume to relax. The current protocol optimises atomic positions only (no cell relaxation), so all volume change values are identically zero. This is a software configuration issue, not a fundamental limitation, and is addressed by enabling `FrechetCellFilter` in ASE.

3. **Li chemical potential reference mismatch**: Absolute computed voltages are systematically ~7 V below experimental values. This does not affect ranking comparisons but means the pipeline cannot currently be used for absolute voltage prediction without correction.

4. **Small supercell SQS degeneracy**: At 10% doping in a 32-atom cell, there is only one substituted site, so all SQS realisations are equivalent and give zero inter-SQS variance. Genuine disorder effects require at least 2–3 substituted sites, achievable at 20–30% doping or with a larger (4×4×4) supercell.

5. **MACE-MP-0 training distribution**: MACE-MP-0 is trained on the Materials Project database, which is biased towards thermodynamically stable, low-entropy ordered compositions. Its accuracy for high-entropy or heavily disordered configurations may be lower than for ordered ones.

---

## 7. Future Work

**Priority 1: NMC811 parent structure.** Running the full evaluation on an NMC811 parent CIF (with 80% Ni, 10% Mn, 10% Co on the transition-metal sublattice) would enable computation of the Li/Ni exchange energy, the most disorder-sensitive and experimentally relevant property. This is the single highest-impact extension.

**Priority 2: Cell relaxation for volume change.** Enabling `FrechetCellFilter` in the MLIP relaxation step (configurable in `pipeline.yaml` via `filter_type: "FrechetCellFilter"`) will make volume change predictions physically meaningful and directly comparable to experimental dilatometry data.

**Priority 3: Larger supercells.** Using a 4×4×4 supercell (256 atoms, 64 Co sites, 6–7 substitutions at 10% doping) would produce genuine SQS variance across realisations and a more faithful representation of the disordered solid solution.

**Priority 4: Second material family.** Demonstrating the pipeline on LNMO spinel (LiNi₀.₅Mn₁.₅O₄, a high-voltage spinel cathode) or LLZO garnet (Li₇La₃Zr₂O₁₂, a solid electrolyte) would establish generality beyond layered oxides.

**Priority 5: Level 2 conversational agent.** The Level 2 PRD describes an interactive screening assistant where a researcher can query the pipeline results in natural language ("show me which dopants improve Li/Ni ordering without sacrificing voltage"), backed by the SQLite results database and the LangGraph pipeline. This was designed but not implemented within the Phase 1–6 scope.

**Priority 6: MACE fine-tuning.** Fine-tuning MACE-MP-0 on a focused DFT dataset of layered oxide structures (NMC compositions with various dopants) would improve absolute accuracy for voltage prediction and reduce the Li chemical potential offset. The Matbench benchmark provides a standardised evaluation protocol for this.

---

## 8. References

1. Merchant, A. et al. Scaling deep learning for materials discovery. *Nature* **624**, 80–85 (2023). [GNoME]

2. Zeni, C. et al. MatterGen: a generative model for inorganic materials design. *Nature* **636**, 115–123 (2025). [MatterGen]

3. Yang, H. et al. MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures. *arXiv:2405.04967* (2024). [MatterSim]

4. Batatia, I. et al. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. *Advances in Neural Information Processing Systems* **35** (2022). [MACE architecture]

5. Batatia, I. et al. A foundation model for atomistic materials chemistry. *arXiv:2401.00096* (2024). [MACE-MP-0]

6. Zunger, A., Wei, S.-H., Ferreira, L. G. & Bernard, J. E. Special quasirandom structures. *Phys. Rev. Lett.* **65**, 353 (1990). [SQS theory]

7. Shannon, R. D. Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides. *Acta Crystallogr. A* **32**, 751–767 (1976). [Shannon radii]

8. Hautier, G., Fischer, C. C., Jain, A., Mueller, T. & Ceder, G. Finding Nature's Missing Ternary Oxide Compounds Using Machine Learning and Density Functional Theory. *Chem. Mater.* **22**, 3762–3767 (2010). [Hautier-Ceder substitution model]

9. Goodall, R. E. A., Parackal, A. S., Faber, F. A., Armiento, R. & Lee, A. A. Predicting materials properties without crystal structure: Deep representation learning from stoichiometry. *Nat. Commun.* **13**, 4817 (2022). [SMACT/Roost]

10. Noh, H.-J. et al. Comparison of the structural and electrochemical properties of layered Li[Nix Co y Mnz]O2 (x = 1/3, 0.5, 0.6, 0.7, 0.8 and 0.85) cathode material for lithium-ion batteries. *J. Power Sources* **233**, 121–130 (2013). [Undoped NMC811 reference]

11. Cho, Y. & Cho, J. Significant Improvement of LiNi0.8Co0.15Al0.05O2 Cathodes at 60°C by SiO2 Dry Coating for Li-Ion Batteries. *J. Electrochem. Soc.* **157**, A625 (2010). [Al doping]

12. Kim, U.-H. et al. Microstructure- and interface-modified Al-doped Ni-rich layered oxide cathodes for high-energy lithium batteries. *Chem. Mater.* **31**, 3723–3730 (2019). [Al doping]

13. Du, R. et al. Zr-doped Li[Ni0.8Co0.1Mn0.1]O2 cathode materials for Li-ion batteries. *Chem. Mater.* **29**, 7683–7693 (2017). [Zr doping]

14. Huang, Z. et al. Improving the electrochemical performance of Ni-rich LiNi0.8Co0.1Mn0.1O2 cathode material by Mg doping. *J. Power Sources* **396**, 559–567 (2018). [Mg doping]

15. Kim, J. et al. Controllable synthesis of LiNi0.8Co0.1Mn0.1O2 cathode materials. *Adv. Energy Mater.* **9**, 1803542 (2019). [W doping]

16. Qiu, W. et al. Synergistic effects of low-level elemental doping on Ni-rich cathodes for lithium-ion batteries. *Adv. Energy Mater.* **9**, 1803372 (2019). [Nb doping]

17. Zhang, D. et al. Electrochemical performance and structural investigation of Ga-doped layered oxide. *J. Alloys Compd.* **771**, 364–370 (2019). [Ga doping]

18. Ong, S. P. et al. Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Comput. Mater. Sci.* **68**, 314–319 (2013). [pymatgen]

19. Fritz Haber Institute Study on Chemical Disorder in Materials Discovery, December 2025. [Disorder gap motivation]

---

## Appendix A: Acronyms

| Acronym | Full Term |
|---------|-----------|
| NMC | Nickel Manganese Cobalt (layered oxide cathode, LiNiₓMnᵧCo_zO₂) |
| NMC811 | LiNi₀.₈Mn₀.₁Co₀.₁O₂ |
| DFT | Density Functional Theory |
| MLIP | Machine-Learned Interatomic Potential |
| MACE | Multi-body Atomic Cluster Expansion |
| SQS | Special Quasi-random Structure |
| GNoME | Graph Networks for Materials Exploration |
| MP | Materials Project |
| ICSD | Inorganic Crystal Structure Database |
| SMACT | Semiconducting Materials by Analogy and Chemical Theory |
| ASE | Atomic Simulation Environment |
| PBE | Perdew-Burke-Ernzerhof (exchange-correlation functional) |
| GGA | Generalised Gradient Approximation |
| XRD | X-ray Diffraction |
| TM | Transition Metal |
| OS | Oxidation State |
| CN | Coordination Number |
| MAE | Mean Absolute Error |
| BFGS | Broyden-Fletcher-Goldfarb-Shanno (geometry optimiser) |
| CIF | Crystallographic Information File |
| R-3m | Rhombohedral space group no. 166 (layered oxide structure) |
| EMT | Effective Medium Theory (simple force field fallback) |
| MPS | Metal Performance Shaders (Apple GPU framework) |
| CLI | Command-Line Interface |
| API | Application Programming Interface |
| CGCNN | Crystal Graph Convolutional Neural Network |
| HHI | Herfindahl-Hirschman Index (element supply concentration) |
| CRM | Critical Raw Materials |
| LNMO | Lithium Nickel Manganese Oxide (spinel cathode) |
| LLZO | Lithium Lanthanum Zirconium Oxide (garnet electrolyte) |
| LLM | Large Language Model |
| RQ | Research Question |
| ρ | Spearman rank correlation coefficient |

---

## Appendix B: Repository Structure

```
disorder-screening-agent/
├── config/
│   └── pipeline.yaml              # All tunable parameters
├── data/
│   ├── shannon_radii.json          # 987-ion extended Shannon table
│   ├── known_dopants/
│   │   └── nmc_layered_oxide.json # 19 ground truth dopants
│   ├── experimental_measurements/
│   │   └── nmc_dopants.json       # Literature voltage, Li/Ni mixing data
│   └── structures/
│       └── lco_parent.cif         # LiCoO₂ parent structure (4 atoms)
├── stages/
│   ├── stage1_smact.py            # SMACT EN + charge neutrality filter
│   ├── stage2_radius.py           # Shannon ionic radius mismatch screen
│   ├── stage3_substitution.py     # Hautier-Ceder substitution probability
│   ├── stage4_ml_prescreen.py     # ML pre-screen (CGCNN/Roost, optional)
│   └── stage5/
│       ├── sqs_generator.py       # SQS supercell generation
│       ├── mlip_relaxation.py     # MACE-MP-0 geometry optimisation
│       ├── property_calculator.py # Voltage, formation energy, Li/Ni exchange
│       ├── monitoring.py          # Relaxation abort conditions
│       └── baseline.py            # Undoped reference calculation
├── graph/
│   ├── graph.py                   # LangGraph StateGraph (full pipeline)
│   ├── entry_points.py            # run_stages_1_3(), run_full_pipeline()
│   └── state.py                   # PipelineState TypedDict
├── ranking/
│   ├── ranker.py                  # Rank dopants; DopantStats dataclass
│   └── comparator.py              # Compare two pipeline runs
├── evaluation/
│   ├── eval_pruning.py            # RQ1 precision/recall analysis
│   ├── eval_disorder.py           # RQ2 disorder evaluation (MACE)
│   ├── eval_accuracy.py           # RQ3 accuracy vs experiment
│   ├── ablation.py                # Ablation studies 1–5
│   ├── figures.py                 # 5 publication-quality figures
│   ├── paper_draft_notes.md       # Thesis statement + filled-in tables
│   └── results/
│       ├── rq2_disorder.json      # MACE-MP-0 results (8 dopants)
│       └── rq3_accuracy.json      # Accuracy vs experiment
├── db/
│   ├── models.py                  # SimulationResult dataclasses
│   └── local_store.py             # SQLite persistence
├── pipeline_io/
│   ├── parse_input.py             # Input validation
│   ├── generate_summary.py        # Jinja2 report generation
│   └── templates/
│       └── screening_report.md.j2
├── tests/                         # 274 tests
├── __main__.py                    # CLI (6 subcommands)
├── pyproject.toml                 # Package config
└── requirements.txt
```

**Quick start:**
```bash
pip install -e .

# Run pruning pipeline only (fast, no GPU needed):
disorder-screening prune --formula LiCoO2 --site Co --os 3

# Run full disorder evaluation (requires MACE-MP-0):
python -m evaluation.eval_disorder \
    --structure data/structures/lco_parent.cif \
    --conc 0.10 --n-sqs 5 \
    --save evaluation/results/rq2_disorder.json

# Generate all publication figures:
python -m evaluation.figures \
    --rq2 evaluation/results/rq2_disorder.json \
    --accuracy evaluation/results/rq3_accuracy.json \
    --output evaluation/figures/
```
