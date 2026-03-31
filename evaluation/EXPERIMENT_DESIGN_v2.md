# Does Chemical Disorder Change Computational Dopant Rankings?
## A Systematic Study Across Three Crystal Structure Types

**Version:** 2.1 (31 March 2026)
**Target venues:** Nature Computational Science (primary), npj Computational Materials (alternative)

---

## Abstract

73% of experimentally characterised inorganic materials exhibit substitutional disorder (ICSD), and >80% of computationally predicted materials are likely to be disordered when synthesised (Jakob et al., Adv. Mater. 2025). Two-thirds of GNoME's "novel" materials were actually known disordered phases misidentified as ordered (Lotfi et al., PRX Energy 2024). Yet every major computational screening platform — GNoME (Merchant et al., Nature 2023), MatterGen (Zeni et al., Nature 2025), and hundreds of DFT screening studies — assumes perfectly ordered crystal structures. This assumption is untested: no systematic study has measured whether dopant property rankings change when disorder is properly accounted for.

We quantify this "disorder gap" across three crystal structure types: LiCoO2 (layered, R-3m), LiMn2O4 (spinel, Fd-3m), and SrTiO3 (perovskite, Pm-3m). For each material, we compare dopant rankings from conventional ordered-cell screening against disorder-averaged rankings from multiple SQS (Special Quasirandom Structure) realisations using the MACE-MP-0 universal machine learning interatomic potential.

**Preliminary results (position-only relaxation, to be updated with cell relaxation):**
- LiCoO2 voltage rankings are **destroyed** by disorder (Spearman rho = -0.07, n=22, p=0.76)
- LiMn2O4 voltage rankings are **perfectly preserved** (rho = +0.99, n=22, p<0.001)
- Formation energy rankings are robust in both systems (rho > 0.88)

This structure-dependent finding means that ordered-cell screening is reliable for some materials but systematically misleading for others — and we provide a diagnostic (the "arrangement sensitivity" metric, b_proxy) for predicting which category a new system falls into.

---

## 1. The Problem: An Untested Assumption at the Heart of Materials Screening

### 1.1 The Ordered-Cell Assumption

Computational dopant screening follows a standard protocol:
1. Start with a host crystal structure (e.g., LiCoO2)
2. Substitute one or more host atoms with a dopant (e.g., replace Co with Al)
3. Relax the structure using DFT or an MLIP
4. Compute target properties (voltage, formation energy, volume change)
5. Rank dopants by computed properties

**The critical assumption:** Step 2 places dopants in a single, usually maximally-dispersed arrangement. In reality, dopants occupy random positions at non-zero temperature, creating chemical disorder. The properties of the disordered material may differ from any single ordered arrangement.

### 1.2 Why This Hasn't Been Tested

SQS methods exist (Zunger et al. 1990) and are widely used for alloys. But applying them to dopant screening requires:
- Generating multiple SQS realisations per dopant (expensive)
- Relaxing each realisation (expensive x N)
- Computing properties for each (expensive x N x M)
- Repeating for every dopant candidate (expensive x N x M x D)

With DFT, this is prohibitively expensive. A single dopant with 5 SQS realisations requires ~10 DFT relaxations. For 20 dopants, that's 200 DFT relaxations — months of HPC time.

Universal MLIPs (MACE-MP-0, CHGNet, MatterSim) change the economics: 200 relaxations take hours on a single GPU. This makes systematic comparison feasible for the first time.

### 1.3 What Exists vs What's Missing

**What exists (qualitative recognition):**
- Lotfi et al. (PRX Energy 2024): Two-thirds of GNoME "novel" materials were known disordered phases → disorder causes *misidentification*
- Jakob et al. (Adv. Mater. 2025): >80% of GNoME predictions likely disordered → disorder is *pervasive*
- SQS methodology is mature (Zunger 1990, van de Walle 2013, Wong & Tan 2018)
- Individual SQS studies exist for specific materials, but none compare rankings systematically

**What is missing (our contributions):**
1. **Quantitative ranking impact** — no one has computed Spearman rho between ordered and disordered dopant rankings
2. **Decision metric (b_proxy)** — no diagnostic exists for *when* disorder matters enough to warrant SQS
3. **Practical threshold** — no "if radius mismatch > X%, use disorder-aware methods" rule
4. **Cross-structure comparison** — no study compares disorder sensitivity across structure types in a controlled setting
5. **Laptop-accessible pipeline** — existing disorder-aware approaches (cluster expansion, Monte Carlo) require HPC; MACE-MP-0 + SQS on a consumer GPU is new

**Note on competing pipelines:** BatteryFormer (JACS Au 2025), various ML screening tools, and synthesizability-guided pipelines all assume ordered structures. None include a disorder-aware pathway or SQS step.

### 1.4 What We Test

We run the most direct possible comparison:
- **Same dopants, same concentrations, same supercell, same MLIP**
- Only difference: ordered (single arrangement) vs disordered (8 SQS realisations, averaged)
- Measure: Spearman rank correlation (rho) between ordered and disordered rankings

If rho is near 1.0: ordered screening is fine, disorder can be ignored.
If rho is significantly below 1.0: ordered screening gives wrong rankings; disorder matters.

---

## 2. Hypotheses

### Primary Hypotheses

**H1 (Disorder changes rankings):** For at least one material and property, disorder-averaged rankings diverge significantly from ordered-cell rankings (Spearman rho < 0.8, p < 0.05).

**H2 (Structure-dependent):** The magnitude of divergence varies systematically across structure types. Specifically, layered structures (2D edge-sharing octahedra) are more disorder-sensitive than 3D frameworks (spinel, perovskite) because dopant-dopant interactions in 2D layers are geometrically confined.

**H3 (Predictable from descriptors):** The per-dopant arrangement sensitivity (b_proxy) correlates with a measurable descriptor — ionic radius mismatch, charge mismatch, or both — enabling a predictive rule for when disorder-aware methods are needed.

### Secondary Hypotheses

**H4 (Property-dependent):** Formation energy rankings are more robust to disorder than voltage rankings, because formation energy is a bulk thermodynamic average while voltage involves a phase transformation (delithiation) that amplifies local structural differences.

**H5 (Diagnostic metric):** The arrangement sensitivity metric b_proxy = |P_ordered - P_disordered| serves as a cheap diagnostic. If b_proxy < threshold for all dopants in a material, ordered-cell screening is sufficient for that system.

### Null Hypothesis

**H0:** Ordered and disordered rankings are statistically indistinguishable (rho > 0.9, p < 0.05) for all three materials and all properties. If H0 holds, the disorder-aware pipeline is unnecessary overhead, and the community can continue using ordered cells with confidence.

---

## 3. Materials

### Design Principles
1. **Real materials** — each is commercially or scientifically important, not a proxy
2. **All octahedral B-site doping** — apples-to-apples structural comparison
3. **Minimal spin risk** — Co3+ (low-spin d6), Mn4+ (d3, medium risk), Ti4+ (d0, zero risk)
4. **Existing DFT and experimental ground truth** — enables validation
5. **Three distinct structure types** — layered, spinel, perovskite
6. **Matched target site counts** — LiCoO2 and SrTiO3 both have 64 B-sites per supercell

### Material 1: LiCoO2 (Layered Oxide, R-3m)

| Property | Value |
|----------|-------|
| Space group | R-3m (No. 166) |
| Doping site | Co3+ octahedral (3a), low-spin d6, r = 0.545 A |
| Supercell | 4x4x4 = 256 atoms, 64 Co sites, 6 dopants at 10% |
| Connectivity | Edge-sharing octahedra within 2D TM layer |
| Commercial use | First commercial Li-ion cathode (Sony 1991); consumer electronics |
| Spin risk | **Low** — Co3+ is diamagnetic in low-spin state |
| Key properties | Intercalation voltage, formation energy, volume change |
| DFT ground truth | Yao et al. 2025 (63 dopants, Adv. Energy Mater.) |
| Experimental data | 8 dopants with measured voltages; 13 confirmed successful |

**Why LiCoO2:** The simplest layered cathode — single TM species, no intrinsic compositional disorder. Co3+ low-spin means MACE's lack of spin treatment is minimally problematic. Edge-sharing octahedra in the 2D TM layer create strong short-range dopant-dopant interactions that should amplify disorder sensitivity.

**Known limitations:** Not NMC811 (no Li/Ni antisite disorder, no Ni/Mn/Co mixing). Full delithiation voltage is an upper bound, not directly comparable to experimental operating voltage (3.0-4.3V range). Co3+/Co4+ redox may have d-electron effects not fully captured by MACE.

### Material 2: LiMn2O4 (Spinel, Fd-3m)

| Property | Value |
|----------|-------|
| Space group | Fd-3m (No. 227) |
| Doping site | Mn4+ octahedral (16d), d3, r = 0.530 A |
| Supercell | 2x2x2 = 448 atoms, 128 Mn sites, 13 dopants at 10% |
| Connectivity | Edge-sharing (pyrochlore sublattice) + corner-sharing with tetrahedral Li |
| Commercial use | Nissan Leaf, Chevy Volt; low-cost, high-safety cathode |
| Spin risk | **Medium** — Mn3+ (from aliovalent doping) is Jahn-Teller active |
| Key properties | Intercalation voltage, formation energy, volume change |
| DFT ground truth | J. Power Sources 2025 (12 dopants, DFT voltage + volume) |
| Experimental data | 7 confirmed successful dopants |

**Why LiMn2O4:** Fundamentally different site topology from layered (3D framework vs 2D layers). 128 Mn sites give 13 dopants at 10% — better configurational sampling than LiCoO2's 6. Tests whether 3D frameworks are inherently more disorder-tolerant.

**Known limitations:** Mn3+ (from aliovalent doping) is Jahn-Teller active; MACE has no explicit spin treatment. r2SCAN+U gives spurious phases for Mn oxides (arXiv:2412.16816). Mn dissolution (key failure mode) is a surface phenomenon not captured by bulk SQS.

### Material 3: SrTiO3 (Perovskite, Pm-3m)

| Property | Value |
|----------|-------|
| Space group | Pm-3m (No. 221) |
| Doping site | Ti4+ octahedral (1b), d0, r = 0.605 A |
| Supercell | 4x4x4 = 320 atoms, 64 Ti sites, 6 dopants at 10% |
| Connectivity | Corner-sharing octahedra (every octahedron shares all 6 corners) |
| Applications | Thin-film substrates, photocatalysis, thermoelectrics, memristors |
| Spin risk | **None** — Ti4+ is d0, Sr2+ is closed-shell |
| Key properties | Formation energy, doping volume change (no voltage — not a battery) |
| Validation | SimplySQS (Lebeda et al. 2025): MACE MATPES-r2SCAN-0 on (Pb,Sr)TiO3, lattice params <1% error (cubic), <4% (tetragonal), 320-atom supercell |
| DFT literature | Extensive — La, Nb, V, Fe, Cr, Al, Mg B-site dopants well-studied |

**Why SrTiO3:** The methodological control. Zero spin risk means any disorder effects we observe cannot be attributed to MACE's magnetic limitations. If the disorder effect is consistent between LiCoO2 (low spin risk) and SrTiO3 (zero spin risk), the finding is robust. Perovskite corner-sharing connectivity is fundamentally different from both layered edge-sharing (LiCoO2) and spinel mixed connectivity (LiMn2O4). Extends findings beyond batteries to general materials science.

**Known limitations:** No intercalation voltage (not a battery material) — compare on formation energy and volume change only. Octahedral tilting instabilities below ~105K, but we compute at 0K in the cubic phase. Perovskite tolerance factor may reject some dopants that pass for layered/spinel.

### Cross-Material Comparison

| Aspect | LiCoO2 | LiMn2O4 | SrTiO3 |
|--------|---------|----------|--------|
| Structure | Layered (2D) | Spinel (3D) | Perovskite (3D) |
| Octahedral connectivity | Edge-sharing | Edge+corner | Corner-sharing |
| Doping site | Co3+ (0.545 A) | Mn4+ (0.530 A) | Ti4+ (0.605 A) |
| Supercell atoms | 256 | 448 | 320 |
| B-sites / dopants at 10% | 64 / 6 | 128 / 13 | 64 / 6 |
| Spin risk | Low | Medium | **None** |
| Voltage measurable | Yes | Yes | No |
| MACE validated | This work | This work | SimplySQS 2025 |

---

## 4. Dopant Selection: Pipeline-Derived Common Set

### Methodology: No Cherry-Picking

The dopant set is determined entirely by the computational pipeline, not by the researcher. For each material independently:

1. **Stage 1 (SMACT filter):** Screen all elements Z=1-103 for charge neutrality and electronegativity ordering against the host structure (~80 survivors)
2. **Stage 2 (Shannon radius):** Filter by ionic radius mismatch to the host site (~50 survivors)
3. **Stage 3 (Hautier-Ceder):** Substitution probability from data-mined crystal structures (~40 survivors)
4. **Intersection:** The final dopant set is the intersection of all three materials' Stage 3 survivors

This is methodologically stronger than hand-selection because:
- The pipeline makes the selection, not the researcher
- Each material's chemical constraints are respected
- Reviewers cannot claim bias in dopant selection
- The pipeline code is open-source and the selection is reproducible

### Stage 1-3 Parameters

| Parameter | LiCoO2 | LiMn2O4 | SrTiO3 |
|-----------|---------|----------|--------|
| Host site | Co3+ (0.545 A) | Mn4+ (0.530 A) | Ti4+ (0.605 A) |
| Radius threshold | 35% | 40% | 35% |
| Substitution prob. threshold | 0.001 | 0.0001 | 0.001 |
| Expected survivors | ~46 | ~40 | ~35-45 |
| **Expected intersection** | **~12-20 dopants** |

The LiMn2O4 thresholds are deliberately looser (40% radius, 0.0001 probability) to recover all 7 experimentally confirmed dopants (Cu2+ at 37.7% mismatch, Mg2+ at 35.8%).

### Expected Common Survivors
Based on preliminary Stage 1-3 runs: Al, Ti, Fe, Cr, Mg, Zr, Nb, W, V, Ni, Cu, Zn, Ga, Sn, Ta.

The exact set will be determined by running the pipeline. We commit to using whatever the intersection produces.

---

## 5. Experimental Protocol

### 5.1 Simulation Setup

For each dopant in the common set, for each material:

**Ordered baseline (1 per dopant):**
- Farthest-first dopant placement (maximally dispersed — the implicit assumption in screening studies)
- Cell + ionic relaxation via FrechetCellFilter
- Compute all target properties

**Disordered / SQS (8 per dopant):**
- Generate 8 SQS realisations via pymatgen SQSTransformation (fallback to manual pair-correlation sampling)
- Cell + ionic relaxation with 3-stage retry:
  - Stage 1: BFGS, fmax=0.10 eV/A, max 1000 steps (fast)
  - Stage 2: FIRE, fmax=0.10 eV/A, max 2000 steps (robust)
  - Stage 3: FIRE, fmax=0.20 eV/A, max 2000 steps (loose convergence)
- Record convergence metadata: optimizer used, steps taken, final max force
- Compute properties for each converged realisation
- Report mean, standard deviation, and n_converged

### 5.2 Properties

| Property | LiCoO2 | LiMn2O4 | SrTiO3 | Definition |
|----------|---------|----------|--------|------------|
| Formation energy (eV/atom) | Yes | Yes | Yes | E_total / N_atoms from MACE |
| Intercalation voltage (V) | Yes | Yes | No | V = -(E_delith - E_lith + n_Li * E_Li_ref) / n_Li |
| Volume change (%) | Yes | Yes | Yes | Cathodes: |V_delith - V_lith|/V_lith. SrTiO3: |V_doped - V_parent|/V_parent |

**Removed:** li_ni_exchange (neither host contains Ni — always returns None).

**Property weights for composite ranking:**
- Cathodes: voltage 40%, formation_energy 35%, volume_change 25%
- SrTiO3: formation_energy 50%, volume_change 50%

### 5.3 Relaxation Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| MLIP | MACE-MP-0 | Universal, open-source, runs on consumer hardware |
| Cell filter | FrechetCellFilter | Full cell + ionic relaxation (not position-only) |
| fmax | 0.10 eV/A | Standard for 256+ atom cells |
| max_steps | 1000 (BFGS) / 2000 (FIRE) | Accommodates difficult dopants |
| SQS realisations | 8 | Target 5+ converged after retries |
| Concentration | 10% on host B-site | Standard screening concentration |
| Device | Colab A100 GPU | ~$1.50/hr, 80GB VRAM |

### 5.4 Analysis

**Per-material:**
1. Spearman rho (ordered vs disordered rankings) for each property
2. b_proxy per dopant = |P_ordered - P_disordered| (arrangement sensitivity)
3. Convergence statistics: n_converged per dopant, optimizer distribution, final force distribution
4. SQS convergence plot: running mean vs number of realisations (1, 2, ..., 8)

**Cross-material:**
5. rho comparison across structure types (tests H2)
6. b_proxy vs ionic radius mismatch scatter plot for each material (tests H3)
7. b_proxy vs charge mismatch |Z_dopant - Z_host| (disentangles size vs electronic effects)
8. Formation energy rho vs voltage rho within each material (tests H4)

---

## 6. Ground Truth and Validation

### 6.1 DFT Ground Truth

**LiCoO2 — Yao et al. (2025)**
- "Stepwise Screening of Doping Elements for High-Voltage LiCoO2 via Materials Genome Approach"
- *Adv. Energy Mater.* 2502026. DOI: 10.1002/aenm.202502026
- **63 dopant elements** screened through 5-step pruning: chemo-mechanical lattice strain (volume change), oxygen release tendency, cation mixing energy, kinetic stability (Li migration), electrochemical stability (voltage window)
- Final dopants passing all screens: **Sb and Ge** (experimentally validated — both showed reduced lattice contraction, fewer intergranular microcracks, less surface rock-salt formation vs undoped LCO)
- **Use:** Compare MACE formation energy and voltage rankings to DFT rankings for overlapping dopants. Largest single-host DFT screening study in the literature. Full paper contains intermediate screening data per dopant at each step.

**LiMn2O4 — J. Power Sources (2025)**
- "Computational design of multi-element-doped LiMn2O4 spinel cathodes for high-voltage and stable lithium-ion batteries"
- *J. Power Sources* 652, 237541. DOI: S0378775325013850
- **12 dopants** at Mn site: Mg, Al, Ti, V, Cr, Fe, Ni, Cu, Zn, Zr, Nb, W
- **148 configurations** evaluated via DFT + AIMD on 320-atom supercells (SQS-generated)
- Properties: average Li intercalation voltage, unit cell volume change (Delta-V), lattice distortion
- Threshold criteria: voltage > 4.0V AND volume change < 30.5 cubic Angstroms
- **117 multi-doped compositions** passed both criteria
- Optimal single dopants for voltage + stability: **Cr, Fe, Zn, Mg, W**
- Best co-doping: Cr/Fe/Ni preferentially stabilises Mn3+/Mn4+ redox couple
- **Use:** Direct comparison of MACE voltage rankings to DFT voltage rankings. The 12-dopant set has near-complete overlap with our expected common dopant set. **Strongest single validation source** — same host, same dopants, same property (voltage).

**SrTiO3 — Literature compilation (no single comprehensive source)**
- **3d TM dopants (V, Cr, Mn, Fe, Co, Ni, Cu at Ti site):** J. Magn. Magn. Mater. (2020). DFT+U study; formation energy increases monotonically with atomic number Z, V@Ti most favourable.
- **Nb, Zr, Mo, Hf, Ta, W, Re at Ti site:** Hybrid DFT band gap engineering study (arXiv:2105.14165).
- **Al at Ti site:** J. Am. Chem. Soc. (2025). Formation energies in oxygen-poor limit.
- **La, Nb in SrTiO3:** Materialia (2023). In-gap states and local structures; Nb prefers Ti site.
- **Mn at Ti site:** J. Phys. Chem. C (2014). GGA+U + interatomic potentials, defect formation energies.
- **Use:** Compilation needed. The 3d TM study covers 7 dopants; remaining require individual papers. **Key challenge acknowledged:** no single ground-truth ranking table exists for SrTiO3.

### 6.2 Experimental Validation

**LiCoO2:**
- 8 dopants with experimentally measured voltages: Al, Ti, Mg, Ga, Fe, Zr, Nb, W
- Source: compiled from peer-reviewed literature (data/known_dopants/nmc_layered_oxide.json)
- 13 confirmed successful, 3 limited-benefit, 2 failed dopants

**LiMn2O4:**
- 7 confirmed successful dopants: Al, Co, Cu, Fe, Mg, Ti, V
- Source: data/known_dopants/lnmo_spinel.json
- J. Power Sources 2025 experimental data for select dopants

**SrTiO3:**
- Experimental solubility and property data from ceramics literature
- Less directly comparable (no voltage), but formation energy trends can be validated

### 6.3 MACE-MP-0 Accuracy Context

| Metric | Value | Source |
|--------|-------|--------|
| Energy MAE (validation) | ~20 meV/atom | Batatia et al. 2024 |
| Force MAE (validation) | ~45 meV/A | Batatia et al. 2024 |
| Matbench Discovery hull distance | F1=0.59, MAE~60 meV/atom | matbench-discovery leaderboard |
| (Pb,Sr)TiO3 lattice params | Within 1-4% of experiment | SimplySQS (Lebeda et al. 2025), using MACE MATPES-r2SCAN-0 |
| Spin treatment | None | Known limitation |
| GGA+U mixing | Uncorrected in training data | Systematic error for TM oxides |

**Critical gap:** No published benchmark exists for MACE-MP-0 intercalation voltage prediction. This paper would be the first systematic assessment.

### 6.4 Validation Metrics

For each material, we compute 4 Spearman rho values:

| Comparison | What it tests |
|-----------|--------------|
| rho(MACE-ordered vs DFT) | Is MACE accurate for conventional ordered-cell screening? |
| rho(MACE-disordered vs DFT) | Does disorder-averaging improve MACE predictions? |
| rho(MACE-ordered vs experiment) | Does conventional screening predict experiments? |
| rho(MACE-disordered vs experiment) | Does disorder-aware screening predict experiments better? |

**The key test:** If rho(MACE-disordered vs experiment) > rho(MACE-ordered vs experiment), disorder-aware screening is closer to experimental reality.

---

## 7. Preliminary Results (Position-Only Relaxation)

These results use the original position-only relaxation (FrechetCellFilter was inadvertently disabled). They will be superseded by cell-relaxed results but illustrate the core finding.

### 7.1 Spearman Rank Correlations

| Material | Property | rho | p-value | n | Interpretation |
|----------|----------|-----|---------|---|----------------|
| LiCoO2 | Formation energy | +0.956 | <0.001 | 22 | Rankings preserved |
| LiCoO2 | **Voltage** | **-0.069** | **0.759** | **22** | **Rankings destroyed** |
| LiCoO2 | Volume change | N/A (all zero) | — | 22 | Artefact of position-only relaxation |
| LiMn2O4 | Formation energy | +1.000 | <0.001 | 22 | Rankings perfectly preserved |
| LiMn2O4 | **Voltage** | **+0.988** | **<0.001** | **22** | **Rankings perfectly preserved** |
| LiMn2O4 | Volume change | N/A (all zero) | — | 22 | Artefact of position-only relaxation |

### 7.2 The Core Finding

**Disorder destroys voltage rankings in layered LiCoO2 (rho = -0.07) but preserves them perfectly in spinel LiMn2O4 (rho = +0.99).** Formation energy rankings are robust in both systems.

This is exactly the structure-dependent effect predicted by H2. The layered structure, with its 2D TM layer and edge-sharing octahedra, creates strong local dopant-dopant interactions that are arrangement-sensitive. The spinel 3D framework distributes these interactions more uniformly.

### 7.3 Arrangement Sensitivity (b_proxy)

Preliminary b_proxy values for LiCoO2 (8 dopants, voltage):

| Dopant | b_proxy (V) | Radius mismatch | Charge mismatch |
|--------|-------------|-----------------|-----------------|
| Fe3+ | 0.070 | ~0% | 0 |
| Al3+ | 0.123 | ~2% | 0 |
| Ga3+ | 0.165 | ~14% | 0 |
| Nb5+ | 0.175 | ~17% | +2 |
| Ti4+ | 0.189 | ~11% | +1 |
| Mg2+ | 0.215 | ~32% | -1 |
| W6+ | 0.300 | ~10% | +3 |
| Zr4+ | 0.313 | ~32% | +1 |

**Key observation:** W6+ has small radius mismatch (~10%) but large b_proxy (0.300). If this persists after cell relaxation, it indicates electronic/charge effects beyond simple size mismatch. This is the test case for disentangling size vs electronic disorder sensitivity (H3).

### 7.4 Caveats on Preliminary Data

1. **Position-only relaxation** — Volume was fixed, creating trapped strain energy proportional to size mismatch. Cell relaxation may reduce b_proxy for large-ion dopants (Zr, Mg) and clarify the W anomaly.
2. **Poor convergence for difficult dopants** — Zr (2/5), Mg (2/5), Ta (2/5) converged. The 3-stage retry and 8 SQS realisations should fix this.
3. **Volume change = 0** for all dopants — artefact, not a real finding. Cell relaxation will produce meaningful volume change values.

---

## 8. Self-Critique and Known Limitations

### Fundamental Limitations

**L1: 0K static energies, no entropy.**
Free energy G = E + PV - TS. We compute E and PV (with cell relaxation) but not TS. Configurational entropy S_config = -k_B * sum(x_i * ln(x_i)) favours disordered arrangements at operating temperature. At 300K with 10% doping: TS ~ 25 meV/atom — small vs formation energy (100s of meV) but potentially significant for voltage (10s of meV differences between dopants).
**Mitigation:** Entropy would strengthen our case (it stabilises disorder). This is a conservative bias.

**L2: MACE-MP-0 has no spin/magnetic treatment.**
Transition metal d-electrons create magnetic ordering that MACE cannot model. Affects: Mn3+ (JT distortion), Cr3+ (half-filled t2g), some dopants in non-ground-state configurations.
**Mitigation:** LiCoO2 (low-spin Co3+) and SrTiO3 (d0 Ti4+) have minimal spin risk. If the disorder effect is consistent between these two systems, it cannot be a spin artefact. LiMn2O4 results are reported with explicit magnetic caveats.

**L3: GGA+U mixing in MACE training data.**
MACE-MP-0 trained on MPtrj with mixed GGA/GGA+U without MP2020Compatibility corrections. Formation energies are not on the same scale as corrected Materials Project values.
**Mitigation:** We compare RANKINGS (Spearman rho), not absolute values. Rank correlations are robust to monotonic systematic offsets.

**L4: No explicit charge compensation.**
Aliovalent doping creates charge imbalance. MACE has no explicit charge states, though it was trained on DFT data where compensation occurred implicitly.
**Mitigation:** Compare dopants of similar charge mismatch to control for this. The b_proxy vs charge mismatch analysis directly addresses whether charge effects drive disorder sensitivity.

**L5: Full delithiation for voltage.**
V = -(E_delith - E_lith + n_Li * E_Li_ref) / n_Li removes ALL Li. Real cathodes cycle between partial states. Full delithiation may involve phase transformations.
**Mitigation:** Standard in computational screening. We compare rankings, not absolute voltages.

### Methodological Limitations

**L6: Single concentration (10%).**
Rankings could change at 5% or 15%. The pipeline supports multiple concentrations but we report only 10%.
**Mitigation:** 10% is a standard screening concentration. Concentration dependence is explicitly noted as future work.

**L7: 8 SQS realisations sample a fraction of configuration space.**
With 6 dopant atoms (LiCoO2), the number of distinct configurations is C(64,6) ~ 7.4 million. 8 SQS realisations is a tiny sample.
**Mitigation:** Report SQS convergence plot (running mean vs n). Standard practice: 6-8x primitive cell is sufficient for property convergence in metallic alloys (Wong & Tan 2018). For ionic systems, convergence may be slower (Jiang 2014) — we report this honestly.

**L8: Heterovalent ionic SQS convergence.**
Jiang (PRB 91, 024106, 2015; arXiv:1408.6875) showed that SQS properties do not converge with supercell size in heterovalent ionic systems (tested on MgAl2O4 spinel, ZnSnP2 chalcopyrite). Root cause: correlation functions of long-range clusters larger than the supercell period are unoptimised, and electrostatic interactions in charge-mismatched systems are long-range. They proposed 1/N extrapolation to estimate the infinite-size disordered limit.
**Mitigation:** Our systems ARE heterovalent (aliovalent doping). 256-448 atoms is large by literature standards. Multiple realisations partially compensate. We report SQS convergence plots and acknowledge this limitation explicitly. The 1/N extrapolation scheme could be applied as a post-hoc correction if convergence is poor.

**L9: SQS generation quality.**
pymatgen SQSTransformation often fails, falling back to manual pair-correlation sampling. The quality of this fallback is not reported.
**Mitigation:** Report pair correlation function match for generated SQS. Add to evaluation output.

**L10: Farthest-first ordered baseline is a specific choice.**
The "ordered" reference maximally disperses dopants. Other orderings (clustered, periodic superlattice) could give different properties.
**Mitigation:** Farthest-first is the most common choice in screening studies because it minimises dopant-dopant interactions — the implicit assumption in ordered screening. Our comparison tests this specific assumption.

---

## 9. Why This Paper Will Be Highly Cited

### 9.1 It Fills a Known but Unquantified Gap

Two recent high-profile papers established that disorder is pervasive and ignored: Lotfi et al. (PRX Energy 2024) showed two-thirds of GNoME's "novel" materials were actually known disordered phases misidentified as ordered; Jakob et al. (Adv. Mater. 2025) showed >80% of GNoME predictions are likely disordered when synthesised. Both papers demonstrate the *existence* of the problem. **Neither quantifies the *consequence* for property-based rankings.** Our paper provides the first systematic numbers — Spearman rho values showing exactly how much rankings shift. That is immediately citable by anyone publishing a computational screening study.

### 9.2 It Provides a Decision Tool

The b_proxy metric gives a practical rule: "If b_proxy > X for your dopant, use disorder-aware methods." This is analogous to the Goldschmidt tolerance factor for perovskite stability — a simple metric that everyone uses because it saves time. Every screening paper published after ours must either:
- (a) Use disorder-aware methods, or
- (b) Cite our paper to justify why they didn't need to

Either way, it gets cited.

### 9.3 Three Structure Types = Broad Applicability

Covering layered + spinel + perovskite makes the paper relevant to:
- **Battery researchers** (cathodes, solid electrolytes)
- **Ceramics researchers** (dielectrics, piezoelectrics, ferroelectrics)
- **Catalyst researchers** (photocatalysis, SOFC, oxygen evolution)
- **Semiconductor researchers** (transparent conductors, thermoelectrics)

A battery-only paper is cited by battery researchers. A cross-structure paper is cited by the entire computational materials community.

### 9.4 First MLIP Voltage Benchmark

No published work benchmarks a universal MLIP for intercalation voltage prediction. By validating MACE-MP-0 against DFT and experiment for two cathode systems, this becomes the reference for anyone using MLIPs for battery screening — a rapidly growing field.

### 9.5 Reproducible on a Laptop

Total cost: ~$8-10 on Colab A100 (~6 hours). No HPC allocation, no VASP license, no months of queue time. This removes the adoption barrier entirely. The pipeline is open-source and the CIF files, configs, and notebooks are included. Any researcher can reproduce or extend the results.

### 9.6 The Methodology Is Immediately Reusable

1. Download the pipeline (open-source on GitHub)
2. Swap in a new host material CIF
3. Run Stages 1-3 to get dopant candidates
4. Run Stage 5 to get disorder-aware rankings
5. Compute b_proxy to check if disorder matters for the new system

This plug-and-play utility drives citations from application papers across materials subfields.

---

## 10. How Researchers Will Use These Results

### Immediate Impact

**Experimentalists (battery, ceramics, catalysis):**
- Before synthesising dopant candidates, check b_proxy table or run the pipeline
- If b_proxy is high for their system type: use SQS-averaged properties instead of single-point ordered cells
- Save months of wasted effort on dopants that were incorrectly ranked

**Computational screeners:**
- Integrate SQS module into existing screening pipelines when b_proxy > threshold
- Use our rho values as baseline for their own system validations
- Cite the paper to justify their methodological choices

**MLIP developers:**
- Use our voltage/formation energy benchmarks to test their models
- Identify where MLIPs fail (voltage prediction is harder than formation energy)

### Foundation for an Agentic Materials Platform

This work establishes three capabilities that are prerequisites for autonomous materials discovery:

1. **Disorder-awareness as a module:** The pipeline's SQS pathway can be activated when b_proxy exceeds a threshold, integrated into any agent framework (LangGraph, AutoGen, etc.)

2. **Capability gating:** Our companion paper (DisorderBench, targeting NeurIPS 2026) shows that LLMs score 100% on multiple-choice but 43-73% on execution tasks. This means agents need capability-validated tools, not just LLM reasoning.

3. **Multi-fidelity screening funnel:** MLIP (fast, cheap) → DFT (accurate, expensive) → experiment (ground truth, very expensive). The pipeline implements the first tier; the b_proxy metric determines which candidates need the second tier.

---

## 11. Computational Budget

### Production Runs (Colab A100, ~$1.50/hr)

| Material | Supercell | Dopants | Relaxations | Est. Time | Est. Cost |
|----------|-----------|---------|-------------|-----------|-----------|
| LiCoO2 (256 atoms) | 4x4x4 | ~15 | 15 * (1 + 8) * 2 = ~270 | ~1.5 hr | ~$2.25 |
| LiMn2O4 (448 atoms) | 2x2x2 | ~15 | 15 * (1 + 8) * 2 = ~270 | ~3 hr | ~$4.50 |
| SrTiO3 (320 atoms) | 4x4x4 | ~15 | 15 * (1 + 8) = ~135 | ~1 hr | ~$1.50 |
| **Total** | | | **~675 + retries** | **~6 hr** | **~$8-10** |

SrTiO3 has fewer relaxations: no delithiation step needed (no voltage calculation).
Retries (FIRE stages) may add 20-30% to total time.

### Smoke Test (Colab T4, free)

Fe + Zr on LiCoO2 only: ~45 min. Validates:
- FrechetCellFilter is active (volume_change != 0)
- 3-stage retry works (Zr was 2/5 before; should improve)
- Cell-relaxed b_proxy values are meaningful

---

## 12. Timeline

| Week | Task |
|------|------|
| Week 1 | Smoke test (Fe+Zr on Colab T4). Run Stages 1-3 for all 3 materials. Determine common dopant set. |
| Week 2 | Full production runs: LiCoO2 + LiMn2O4 + SrTiO3 on Colab A100 (~6 hours total). |
| Week 3 | Extract DFT ground truth from Yao 2025, JPS 2025, SrTiO3 literature. Compute 4 rho values per material. |
| Week 4 | Post-processing: b_proxy analysis, convergence plots, cross-system comparison figures. |
| Week 5-6 | Write paper draft. |

---

## 13. Success Criteria

### Outcome A (strongest): Structure-dependent disorder sensitivity
- At least one material: rho < 0.8 for voltage
- At least one material: rho > 0.9 for voltage
- b_proxy correlates with identifiable descriptor(s)
- Narrative: "Disorder matters for X but not Y, and here's how to predict which."

### Outcome B (strong): Disorder matters everywhere
- All three materials: rho < 0.8 for voltage
- Narrative: "Ordered screening is never safe — always use SQS."

### Outcome C (still publishable): Disorder doesn't matter
- All three materials: rho > 0.9 for all properties
- Null hypothesis confirmed
- Narrative: "We tested across 3 structure types and found disorder doesn't change rankings. The community can stop worrying."
- This is a valuable result: it gives permission to keep using simpler methods.

### Failure modes (detectable in smoke test)
- MACE produces unphysical results (negative volumes, divergent energies)
- Convergence rates too low (<50%) even with retries
- SQS quality too poor to approximate random alloys

---

## 14. Publication Strategy

### Paper 1: DisorderBench (NeurIPS 2026 Datasets & Benchmarks Track)
- **Deadline:** ~June 2026
- **Headline:** "The explain-execute gap, not a disorder-knowledge gap, is the primary barrier to AI-assisted materials design."
- **Status:** Draft complete; needs human validation of LLM-as-judge and final polish.
- **This experiment's role:** Infrastructure/ground truth only (2-3 paragraphs in methods). The disorder-changes-rankings story is saved for Paper 2.

### Paper 2: The Disorder Gap (Nature Computational Science)
- **Deadline:** Rolling submission, target August 2026
- **Headline:** "Computational dopant screening gives systematically wrong rankings when it ignores chemical disorder — but only for some structure types. We quantify this across three systems and provide a diagnostic."
- **Status:** Needs cell-relaxed results from this experiment.
- **This experiment IS the paper.**

### Paper 3: Platform Vision (Perspective/Commentary)
- **Deadline:** After Papers 1 & 2 accepted
- **Headline:** "Autonomous materials agents assume ordered structures and trust LLM recommendations. Both assumptions are wrong."
- **No new experiments** — synthesises findings from Papers 1 & 2.

---

## 15. Files and Repository Structure

| File | Status | Purpose |
|------|--------|---------|
| `data/structures/lco_parent.cif` | Exists | LiCoO2 parent structure |
| `data/structures/lmo_spinel.cif` | Exists | LiMn2O4 parent structure |
| `data/structures/srtio3_parent.cif` | **Created** | SrTiO3 parent (mp-5229) |
| `config/pipeline_444.yaml` | **Updated** | LiCoO2 config: 8 SQS, li_ni_exchange=0 |
| `config/pipeline_lnmo.yaml` | **Updated** | LiMn2O4 config: 8 SQS, li_ni_exchange=0 |
| `config/pipeline_srtio3.yaml` | **Created** | SrTiO3 config: 8 SQS, formation_energy+volume |
| `config/targets/srtio3_perovskite.yaml` | **Created** | SrTiO3 target definition |
| `evaluation/eval_disorder.py` | **Updated** | Retry logic, FrechetCellFilter, --target-species |
| `stages/stage5/property_calculator.py` | **Updated** | FrechetCellFilter, doping volume change fallback |
| `colab_nmc_smoketest.ipynb` | **Created** | Fe+Zr smoke test on Colab T4 |
| `colab_lnmo_eval.ipynb` | **Updated** | LNMO production with retry + cell relaxation |
| `colab_nmc_full.ipynb` | To create | Full LiCoO2 production run |
| `colab_srtio3_eval.ipynb` | To create | SrTiO3 production run |

---

## 16. Key References

### Core SQS and Methodology
1. **Zunger, A. et al.** (1990). "Special quasirandom structures." *Phys. Rev. Lett.* 65, 353. — The foundational SQS method.
2. **Wong, J.J. & Tan, T.L.** (2018). "SQS convergence in 2D MXene alloys." *J. Phys.: Condens. Matter*. — 6-8x primitive cell sufficient for property convergence.
3. **Jiang, C.** (2015). "Special quasirandom structure in heterovalent ionic systems." *Phys. Rev. B* 91, 024106. arXiv:1408.6875. — SQS properties don't converge for heterovalent systems; proposes 1/N extrapolation.

### The Disorder Problem (Our Paper Addresses These)
4. **Lotfi, S. et al.** (2024). "Challenges in High-Throughput Inorganic Materials Prediction and Autonomous Synthesis." *PRX Energy* 3, 011002. — Two-thirds of GNoME "novel" materials were known disordered phases misidentified as ordered.
5. **Jakob, J., Walsh, A., Reuter, K. & Margraf, J.T.** (2025). "Learning Crystallographic Disorder: Bridging Prediction and Experiment in Materials Discovery." *Adv. Mater.* e2514226. DOI: 10.1002/adma.202514226. — ICSD: 73% substitutional disorder, 46% positional; ML classifiers predict >80% of GNoME materials would be disordered experimentally. 90% accuracy for disorder classification from composition alone.

### Screening Platforms That Assume Order
6. **Merchant, A. et al.** (2023). "Scaling deep learning for materials discovery." *Nature* 624, 80-85. — GNoME; explicitly 0K ordered structures.
7. **Zeni, C. et al.** (2025). "MatterGen: a generative model for inorganic materials design." *Nature*. — Post-hoc disorder deduplication only; no disorder-aware property prediction.

### MLIP
8. **Batatia, I. et al.** (2024). "MACE-MP-0: A Universal Machine Learning Interatomic Potential." — The MLIP we use.
9. **Lebeda, M. et al.** (2025). "SimplySQS: An Automated and Reproducible Workflow for SQS Generation with ATAT." arXiv:2510.18020. — MACE MATPES-r2SCAN-0 validated on (Pb,Sr)TiO3; lattice params <1% error (cubic), <4% (tetragonal), 320-atom supercell. Correctly reproduced cubic-to-tetragonal phase transition.

### DFT Ground Truth
10. **Yao, X. et al.** (2025). "Stepwise Screening of Doping Elements for High-Voltage LiCoO2." *Adv. Energy Mater.* 2502026. — 63 dopants, DFT ground truth.
11. **J. Power Sources** 652, 237541 (2025). "Computational design of multi-element-doped LiMn2O4 spinel cathodes." — 12 dopants, 148 configs, DFT+AIMD voltage + volume on 320-atom SQS supercells.

### SrTiO3 DFT Ground Truth
12. **J. Magn. Magn. Mater.** (2020). 3d TM-doped SrTiO3 (V, Cr, Mn, Fe, Co, Ni, Cu at Ti site). DFT+U; formation energy increases with Z.
13. **arXiv:2105.14165**. Hybrid DFT: Nb, Zr, Mo, Hf, Ta, W, Re at Ti site — band gap engineering.
14. **J. Am. Chem. Soc.** (2025). Al at Ti and Sr sites in SrTiO3 — formation energies.
15. **Materialia** (2023). La, Nb in SrTiO3 — in-gap states, local structures.

### Other
16. **Bartel, C.J. et al.** (2019). "New tolerance factor for perovskite stability." *Sci. Adv.* 5, eaav0693. — Bartel tolerance factor.
17. **Lotfi, S. et al.** (2024). "Challenges in High-Throughput Inorganic Materials Prediction." *PRX Energy* 3, 011002. — 2/3 of GNoME "novel" materials were known disordered phases.
