# Experiment Design v2: Disorder-Aware Dopant Screening Across Structure Types

## Document History
- v1: LiCoO2 "proxy for NMC" + LiMn2O4 "proxy for LNMO" with position-only relaxation
- v2 (this): LiCoO2 + LiMn2O4 + SrTiO3 as real materials, cell relaxation, common dopant set

---

## 1. Hypothesis

### Primary Hypothesis
**The error introduced by ignoring dopant disorder in computational screening is structure-dependent: it is large enough to invert property rankings in some crystal structures but negligible in others.**

Specifically, we hypothesise that:
- H1: Ordered-cell dopant rankings diverge from disorder-averaged rankings (Spearman rho < 0.8) for at least one property in at least one structure type.
- H2: The magnitude of divergence varies systematically across structure types (layered vs spinel vs perovskite).
- H3: The divergence for a given dopant correlates with a measurable structural/chemical descriptor (ionic radius mismatch, charge mismatch, or both), enabling a predictive rule for when disorder-aware methods are needed.

### Secondary Hypotheses
- H4: Formation energy rankings are more robust to disorder than voltage rankings (because formation energy is a bulk average while voltage involves a phase transformation).
- H5: The arrangement sensitivity metric (b_proxy) can serve as a cheap diagnostic: if b_proxy < threshold for a given dopant, ordered-cell screening is sufficient.

### Null Hypothesis
Ordered and disordered rankings are statistically indistinguishable (rho > 0.9, p < 0.05) across all materials and properties. If true, the entire disorder-aware pipeline is unnecessary overhead.

---

## 2. Materials Selection

### Design Principles
1. **Real materials, not proxies** — each system is commercially or scientifically important in its own right
2. **All octahedral doping sites** — enables apples-to-apples comparison of disorder sensitivity across structures
3. **Non-magnetic or low-spin host sites** — minimises MACE-MP-0 spin limitations
4. **Existing DFT/experimental validation data** — enables MLIP accuracy assessment
5. **Three distinct structure types** — layered, spinel, perovskite

### Material 1: LiCoO2 (Layered, R-3m)

| Property | Value |
|----------|-------|
| Space group | R-3m (No. 166) |
| Structure type | Layered oxide (alpha-NaFeO2) |
| Doping site | Co3+ octahedral (3a) |
| Host ion | Co3+ (low-spin d6, t2g6 eg0) |
| Shannon radius | 0.545 A |
| Spin risk | **Low** — Co3+ is non-magnetic in low-spin state |
| Parent CIF | data/structures/lco_parent.cif |
| Supercell | 4x4x4 = 256 atoms, 64 TM sites, 6-7 dopants at 10% |
| Octahedral connectivity | Edge-sharing within TM layer |
| Commercial relevance | First commercial Li-ion cathode (Sony 1991), still used in consumer electronics |
| Key property | Intercalation voltage, formation energy |

**Why this material:**
- Simplest layered cathode — single TM species, no composition disorder in host
- Co3+ low-spin means minimal magnetic complications for MACE
- Enormous literature: 63 dopants screened by DFT in Yao et al. 2025
- Edge-sharing octahedra create strong dopant-dopant interactions within TM layer

**Known limitations:**
- Not NMC811 — no Li/Ni antisite disorder, no Ni/Mn/Co mixing
- Full delithiation voltage is not directly comparable to experimental operating voltage
- Co3+/Co4+ redox may not be perfectly captured by MACE (d-electron effects)

### Material 2: LiMn2O4 (Spinel, Fd-3m)

| Property | Value |
|----------|-------|
| Space group | Fd-3m (No. 227) |
| Structure type | Normal spinel |
| Doping site | Mn4+ octahedral (16d) |
| Host ion | Mn4+ (d3, t2g3 eg0) |
| Shannon radius | 0.530 A |
| Spin risk | **Medium** — Mn3+ (from doping-induced reduction) is JT-active |
| Parent CIF | data/structures/lmo_spinel.cif |
| Supercell | 2x2x2 = 448 atoms, 128 Mn sites, ~13 dopants at 10% |
| Octahedral connectivity | Edge-sharing (pyrochlore sublattice) + corner-sharing with tetrahedral Li |
| Commercial relevance | Used in Nissan Leaf, Chevy Volt; low-cost, high-safety cathode |
| Key property | Intercalation voltage, formation energy |

**Why this material:**
- Spinel has fundamentally different site topology from layered (3D framework vs 2D layers)
- 128 Mn sites at 2x2x2 gives 13 dopants at 10% — better configurational sampling than LiCoO2
- J. Power Sources 2025 provides DFT voltage + volume data for 12 dopants
- Tests whether the 3D spinel framework is inherently more/less disorder-tolerant than 2D layers

**Known limitations:**
- Mn3+ (created when aliovalent dopants substitute Mn4+) is Jahn-Teller active
- MACE-MP-0 has no explicit spin treatment — Mn magnetic ordering may be incorrect
- Mn dissolution (a key failure mode) cannot be captured by bulk SQS simulations
- r2SCAN+U itself gives spurious phases for Mn oxides (arXiv:2412.16816)

### Material 3: SrTiO3 (Perovskite, Pm-3m)

| Property | Value |
|----------|-------|
| Space group | Pm-3m (No. 221) |
| Structure type | Cubic perovskite |
| Doping site | Ti4+ octahedral (1b) |
| Host ion | Ti4+ (d0) |
| Shannon radius | 0.605 A |
| Spin risk | **None** — Ti4+ is d0, Sr2+ is closed-shell |
| Parent CIF | Obtain from Materials Project (mp-5229) |
| Supercell | 4x4x4 = 320 atoms, 64 Ti sites, 6-7 dopants at 10% |
| Octahedral connectivity | Corner-sharing (every octahedron shares all 6 corners) |
| Commercial relevance | Substrate for thin films, photocatalysis, thermoelectrics, memristors |
| Key property | Formation energy, volume change |

**Why this material:**
- **Zero spin risk** — both cation sites are non-magnetic
- MACE already validated on (Pb,Sr)TiO3 by SimplySQS (Lebeda et al. 2025): lattice parameters within 1-4% of experiment
- Perovskite = completely different connectivity (corner-sharing vs edge-sharing)
- Extends findings beyond battery materials → general materials science
- Massive DFT literature on doped SrTiO3 (thermoelectrics, photocatalysis)
- Same number of B-sites per supercell as LiCoO2 (64) → matched comparison

**Known limitations:**
- No voltage property (not an intercalation material) — compare on formation energy and volume change only
- Perovskite tolerance factor may reject some dopants that pass for layered/spinel
- Octahedral tilting instabilities at low temperature (but we compute at 0K in cubic phase)

### Cross-Material Comparison Summary

| Aspect | LiCoO2 | LiMn2O4 | SrTiO3 |
|--------|---------|----------|--------|
| Structure | Layered | Spinel | Perovskite |
| Doping site | Co3+ oct | Mn4+ oct | Ti4+ oct |
| Spin risk | Low | Medium | **None** |
| Supercell atoms | 256 | 448 | 320 |
| TM/B-sites | 64 | 128 | 64 |
| Dopants at 10% | 6-7 | 13 | 6-7 |
| Octahedral connectivity | Edge-sharing (2D) | Edge-sharing (3D) | Corner-sharing (3D) |
| Voltage measurable | Yes | Yes | No |
| Formation energy | Yes | Yes | Yes |
| Volume change | Yes | Yes | Yes |
| MACE validation | Not published | Not published | SimplySQS 2025 |
| DFT ground truth | Yao et al. 2025 | JPS 2025 | Literature |

---

## 3. Dopant Selection: Pipeline-Derived Common Set

### Method
**No cherry-picking.** Run Stages 1-3 independently on all three host materials. The dopant set for the disorder comparison is the **intersection** of all three survivor sets.

This is methodologically stronger than hand-selecting dopants because:
- The pipeline makes the selection, not the researcher
- Each material's chemical constraints are respected (different radius thresholds, substitution probabilities)
- Reviewers cannot claim bias in dopant selection

### Stage 1-3 Parameters

| Parameter | LiCoO2 | LiMn2O4 | SrTiO3 |
|-----------|---------|----------|--------|
| Host site | Co3+ (0.545 A) | Mn4+ (0.530 A) | Ti4+ (0.605 A) |
| Radius threshold | 35% | 40% | 35% |
| Substitution prob. threshold | 0.001 | 0.0001 | 0.001 |
| Expected survivors | ~46 | ~40 | ~35-45 |
| Expected intersection | ~12-20 dopants |

### Expected Common Survivors
Based on preliminary Stage 1-3 runs for LiCoO2 and LiMn2O4, the intersection likely includes:
Al3+, Ti4+, Fe3+, Cr3+, Mg2+, Zr4+, Nb5+, W6+, V3+/5+, Ni2+/3+, Cu2+, Zn2+, Ga3+, Sn4+, Ta5+

The exact set will be determined by running the pipeline. We commit to using whatever the intersection produces, even if it's smaller or larger than expected.

---

## 4. Experimental Protocol

### 4.1 Pruning (Stages 1-3)
For each material independently:
1. **Stage 1 (SMACT):** Charge neutrality + electronegativity screen → ~80 survivors
2. **Stage 2 (Shannon radius):** Ionic radius mismatch filter → ~50 survivors
3. **Stage 3 (Hautier-Ceder):** Substitution probability → ~40 survivors
4. **Intersection:** Common survivors across all 3 materials → ~12-20 dopants

Report per-material: initial candidates, survivors per stage, recall against known dopants.

### 4.2 Simulation (Stage 5)
For each dopant in the common set, for each material:

**Ordered baseline:**
- Farthest-first dopant placement (maximally dispersed)
- Cell + ionic relaxation (FrechetCellFilter)
- Compute: formation energy, volume change, voltage (cathodes only)

**Disordered (SQS):**
- 8 SQS realisations per dopant per material
- Cell + ionic relaxation with 3-stage retry (BFGS → FIRE → loose FIRE)
- Same properties computed for each realisation
- Report: mean, std, convergence metadata

### 4.3 Properties Computed

| Property | LiCoO2 | LiMn2O4 | SrTiO3 | Method |
|----------|---------|----------|--------|--------|
| Formation energy (eV/atom) | Yes | Yes | Yes | E_total / N_atoms from MACE |
| Volume change (%) | Yes | Yes | Yes | |V_transformed - V_host| / V_host |
| Intercalation voltage (V) | Yes | Yes | No | V = -(E_delith - E_lith + n_Li * E_Li_ref) / n_Li |

**li_ni_exchange is REMOVED** — neither LiCoO2 nor LiMn2O4 host contains Ni. This was previously weighted at 25% but returned None for all dopants. Removed to eliminate dead weight in composite ranking.

**Property weights (cathodes):** voltage 50%, formation_energy 30%, volume_change 20%
**Property weights (SrTiO3):** formation_energy 60%, volume_change 40%

### 4.4 Relaxation Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| MLIP | MACE-MP-0 (or MACE-MPA-0 if available) | Universal, no HPC required |
| Cell filter | FrechetCellFilter | Cell + ionic relaxation (NOT position-only) |
| fmax | 0.10 eV/A | Standard for 256+ atom cells |
| max_steps | 1000 (BFGS), 2000 (FIRE retries) | Accommodates difficult dopants |
| SQS realisations | 8 | Buffer for convergence failures; target 5+ converged |
| Concentration | 10% on host site | Standard screening concentration |

### 4.5 Analysis Protocol

For each material, compute:
1. **Spearman rho (ordered vs disordered)** for each property — tests H1
2. **b_proxy per dopant** = |P_ordered - P_disordered| — arrangement sensitivity
3. **Convergence statistics** — n_converged, optimizer used, final forces
4. **SQS convergence plot** — running mean of property vs number of realisations (1,2,...8)

Cross-material analysis:
5. **rho comparison** across structures — tests H2
6. **b_proxy vs ionic radius mismatch** for each material — tests H3
7. **b_proxy vs charge mismatch** (|Z_dopant - Z_host|) — disentangles size vs electronic effects
8. **Formation energy rho vs voltage rho** within each material — tests H4

---

## 5. Ground Truth and Validation Sources

### 5.1 DFT Ground Truth

**LiCoO2:**
- **Yao et al. (2025)**, "Stepwise Screening of Doping Elements for High-Voltage LiCoO2 via Materials Genome Approach," *Advanced Energy Materials*, 2502026.
  - 63 dopant elements screened through 4 DFT criteria
  - Reports: lattice strain, oxygen release energy, cation mixing energy
  - Experimental validation: Sb and Ge synthesised and tested
  - **Use:** Compare MACE formation energy rankings to DFT rankings for overlapping dopants

- **Phys. Rev. Materials (2017)**, dopant solubility in LiCoO2.
  - 14 dopants (Na, K, Mg, Ca, Zn, Al, Ga, Sc, Y, Zr, Nb, etc.)
  - Reports: defect formation energies, solubility limits
  - **Use:** Validate MACE defect energetics

**LiMn2O4:**
- **J. Power Sources (2025)**, "Computational design of multi-element-doped LiMn2O4."
  - 12 dopants (Mg, Al, Ti, V, Cr, Fe, Ni, Cu, Zn, Zr, Nb, W)
  - Reports: average intercalation voltage, volume change per configuration
  - 148 configurations evaluated
  - **Use:** Direct comparison of MACE voltage rankings to DFT voltage rankings

**SrTiO3:**
- Extensive DFT literature on doped SrTiO3 for thermoelectrics and photocatalysis.
  - La, Nb, V, Fe, Cr, Al, Mg dopants at Ti site well-studied
  - Formation energies and defect levels available
  - **Use:** Validate MACE formation energies for non-battery system
  - **Key papers to extract data from:** (to be identified during literature search)

### 5.2 Experimental Validation

**LiCoO2:**
- 8 dopants with experimentally measured voltages (Al, Ti, Mg, Ga, Fe, Zr, Nb, W)
- Source: compiled from peer-reviewed literature in data/known_dopants/nmc_layered_oxide.json
- 13 confirmed successful dopants, 3 limited-benefit, 2 failed

**LiMn2O4:**
- 7 confirmed successful dopants (Al, Co, Cu, Fe, Mg, Ti, V)
- Source: data/known_dopants/lnmo_spinel.json
- J. Power Sources 2025 experimental data for select dopants

**SrTiO3:**
- Experimental dopant solubility and property data available from ceramics literature
- Less directly comparable (no single "voltage" metric), but formation energy trends can be validated

### 5.3 MACE-MP-0 Accuracy Context

| Metric | Value | Source |
|--------|-------|--------|
| Validation MAE (energy) | ~20 meV/atom | Batatia et al. 2024 |
| Validation MAE (forces) | ~45 meV/A | Batatia et al. 2024 |
| Matbench Discovery hull MAE | ~60 meV/atom | matbench-discovery.materialsproject.org |
| Spin treatment | None | Acknowledged limitation |
| GGA+U mixing | Uncorrected in training data | Systematic error for TM oxides |

**Critical gap:** No published benchmark for MACE-MP-0 intercalation voltage prediction. This paper would be the first.

### 5.4 Validation Metrics (4 rho values per material)

| Comparison | What it tests | Expected |
|-----------|--------------|----------|
| rho(MACE-ordered vs DFT) | Is MACE accurate for ordered cells? | rho > 0.7 |
| rho(MACE-disordered vs DFT) | Does disorder-averaging improve/worsen MACE? | Compare to above |
| rho(MACE-ordered vs experiment) | Does ordered screening predict experiments? | rho ~ 0.5-0.7 |
| rho(MACE-disordered vs experiment) | Does disorder improve experimental prediction? | rho > ordered |

**The key test:** If rho(MACE-disordered vs experiment) > rho(MACE-ordered vs experiment), disorder-aware screening is closer to reality.

---

## 6. Self-Critique and Known Limitations

### Fundamental Limitations (cannot be fixed without different methods)

**L1: 0K static energies, no entropy.**
Free energy = E + PV - TS. We compute E and PV (with cell relaxation). Configurational entropy
S_config = -k_B * sum(x_i * ln(x_i)) favours disordered arrangements at operating temperature.
At 300K with 10% doping: TS ~ 25 meV/atom. This is small compared to formation energy
differences (100s of meV) but could be significant for voltage (10s of meV differences).
**Mitigation:** Acknowledge explicitly. Note that entropy would strengthen the case for disorder
mattering (it stabilises disordered configurations).

**L2: MACE-MP-0 has no spin/magnetic treatment.**
Transition metal d-electrons create magnetic ordering (AFM, FM) that MACE cannot model.
This affects: Mn3+ (JT distortion), Ni2+/3+ (magnetic moments), Cr3+ (half-filled t2g).
LiMn2O4 is the most affected system; LiCoO2 (low-spin Co3+) and SrTiO3 (d0 Ti4+) are safer.
**Mitigation:** Report LiMn2O4 results with explicit caveat. Use SrTiO3 (zero spin risk) as
the methodological control. If the disorder effect is consistent between LiCoO2 (low spin risk)
and SrTiO3 (no spin risk), the finding is robust regardless of LiMn2O4.

**L3: GGA+U mixing in MACE training data.**
MACE-MP-0 trained on MPtrj (mixed GGA/GGA+U) without MP2020Compatibility corrections.
Formation energies are not on the same scale as corrected Materials Project values.
**Mitigation:** We compare RANKINGS (Spearman rho), not absolute values. Ranking comparisons
are robust to systematic energy offsets. Explicitly state that absolute values should not be
compared to MP formation energies.

**L4: No charge compensation model.**
Aliovalent doping (e.g., Zr4+ for Co3+) creates charge imbalance. In DFT, this manifests as
electron localisation (Co3+ → Co2+) or oxygen holes. MACE has no explicit charge states.
**Mitigation:** Acknowledge. Note that MACE is trained on DFT data where charge compensation
DID occur, so it implicitly captures the energetics to the extent that GGA/GGA+U does.
Formation energy rankings may be more reliable than absolute values because the compensation
mechanism is similar across dopants of the same charge.

**L5: Full delithiation for voltage.**
V = -(E_delith - E_lith + n_Li * E_Li_ref) / n_Li removes ALL Li. Real cathodes cycle between
partial states (e.g., x = 0.5 to 1.0 in LixCoO2). Full delithiation may involve phase
transformations not captured by this protocol.
**Mitigation:** Frame as "average intercalation voltage" (standard in computational screening).
Compare rankings, not absolute voltage values. Note that partial delithiation protocols exist
but require additional computational cost (multiple delithiation levels per dopant).

### Methodological Limitations (could be improved with more compute)

**L6: Single concentration (10%).**
Rankings could change at 5% or 15%. Configuration space differs at different concentrations.
**Mitigation:** The pipeline config generates both 5% and 10%. Report both if feasible.
If only 10%, acknowledge and note that 10% is a standard screening concentration.

**L7: 8 SQS realisations may not fully converge properties.**
With 6-7 dopant atoms per SQS (LiCoO2, SrTiO3), configurational space is large.
8 realisations sample a small fraction.
**Mitigation:** Report convergence plot (running mean vs n_realisations). If the mean
stabilises by realisation 5-6, 8 is adequate. If it's still drifting, acknowledge.
Also cite literature: 6-8x primitive cell is standard for SQS convergence (Wong & Tan 2018).

**L8: SQS periodicity and heterovalent ionic convergence.**
Jiang (2014, arXiv:1408.6875) showed properties may not converge with SQS size for
heterovalent ionic systems because long-range electrostatic interactions extend beyond
the SQS periodicity.
**Mitigation:** Our systems ARE heterovalent (aliovalent doping). This is a fundamental
SQS limitation. 256-448 atoms is large by literature standards. Multiple realisations
partially compensate. Acknowledge and note that cluster expansion or Monte Carlo approaches
would be the next methodological step.

**L9: SQS generation quality.**
Logs show pymatgen SQSTransformation fails and falls back to manual pair-correlation sampling.
The quality of this fallback (how well it matches random alloy correlation functions) is not
reported.
**Mitigation:** Report SQS quality metrics (pair correlation function match to random alloy
target) for each generated structure. Add to evaluation output.

**L10: Farthest-first ordered baseline is one specific ordering.**
The "ordered" reference places dopants maximally apart. Other orderings (clustered, periodic
superlattice) could give different properties. The comparison is really "maximally-dispersed
vs random," not "ordered vs disordered."
**Mitigation:** Acknowledge. Note that farthest-first is the most common choice in screening
studies (it minimises dopant-dopant interactions, which is the implicit assumption in ordered
screening). Consider adding a "clustered" ordered baseline as a control.

### What Would Make The Experiment Stronger (but is out of scope)

- DFT spot-checks: Run 2-3 dopants through VASP for each material to bound MLIP error
- MACE-MPA-0: Newer model with better accuracy; drop-in replacement
- Partial delithiation: Compute voltage at x=0.5 instead of x=0
- Spin-polarised MLIP: Use a spin-aware model (e.g., SpinMACE when available)
- Cluster expansion: Replace SQS with CE for exhaustive configurational sampling

---

## 7. Why This Paper Will Be Highly Cited

### 7.1 It Fills a Gap Everyone Knows Exists But No One Has Quantified

The Fritz Haber Institute (December 2025) documented that >80% of synthesised inorganic
materials exhibit chemical disorder. Yet every major computational screening study
(GNoME, MatterGen, MatterSim, and hundreds of DFT screening papers) assumes ordered
structures. Everyone in the field knows this is an approximation. No one has systematically
measured how much it matters.

This paper answers: "How wrong are you if you ignore disorder?" with specific numbers
(rho values) across three structure types. That's immediately citable by anyone doing
computational screening.

### 7.2 It Provides a Practical Decision Tool

The b_proxy (arrangement sensitivity) metric gives experimentalists a simple rule:
"If b_proxy > X for your dopant, use disorder-aware methods. If b_proxy < X, ordered
screening is fine." This is analogous to the Goldschmidt tolerance factor — a simple
metric that everyone uses because it saves time.

Every screening paper published after this one will need to either:
(a) Use disorder-aware methods, or
(b) Cite this paper to justify why they didn't need to

Either way, it gets cited.

### 7.3 It's Reproducible on a Laptop

The pipeline uses MACE-MP-0 (free, open-source) and runs on consumer hardware.
No HPC allocation needed. This removes the biggest barrier to adoption.
Compare to: DFT screening requires VASP license ($$$) + HPC time (months of allocation).
MLIP screening requires a GPU laptop + a few hours.

### 7.4 Three Structure Types = Broad Applicability

By covering layered, spinel, AND perovskite, the paper is relevant to:
- Battery researchers (cathodes, solid electrolytes)
- Ceramics researchers (dielectrics, piezoelectrics)
- Catalyst researchers (photocatalysis, SOFC)
- Semiconductor researchers (transparent conductors, thermoelectrics)

A two-material battery-only paper is citable by battery researchers.
A three-material cross-structure paper is citable by the entire computational materials community.

### 7.5 It Creates a Benchmark for MLIP Voltage Prediction

No published work benchmarks MACE-MP-0 (or any universal MLIP) for intercalation voltage
prediction. By validating against DFT and experiment for two cathode systems, this paper
becomes the reference point for anyone using MLIPs for battery screening.

### 7.6 The Methodology Is Immediately Reusable

Other researchers can:
1. Download the pipeline (open-source)
2. Swap in their host material CIF
3. Run Stages 1-3 to get their dopant candidates
4. Run Stage 5 to get disorder-aware rankings
5. Compute b_proxy to check if disorder matters for their system

This "plug and play" utility drives citations from application papers.

---

## 8. How Researchers Will Use The Results

### Battery Materials Community
- **Cathode designers:** Check b_proxy before trusting ordered-cell DFT screening
- **Solid electrolyte designers:** Apply the same methodology to LLZO, NASICON, etc.
- **Anode designers:** Apply to silicon alloys, titanate anodes

### Computational Materials Science
- **MLIP developers:** Use our voltage/formation energy benchmarks to test their models
- **Screening pipeline developers:** Integrate SQS module when b_proxy > threshold
- **Materials databases:** Flag entries where disorder effects may invalidate reported properties

### Machine Learning for Materials
- **Active learning:** Use SQS variance as an uncertainty signal for acquisition functions
- **Generative models:** Test whether GNoME/MatterGen structures are disorder-stable
- **LLM agents:** Use DisorderBench (Paper 1) to gate LLM execution capabilities

---

## 9. Computational Budget

### Per-Material Cost (Colab A100)

| Material | Dopants | Relaxations | Est. Time | Est. Cost |
|----------|---------|-------------|-----------|-----------|
| LiCoO2 (256 atoms) | ~15 common | 15 * (1 ord + 8 SQS) * 2 states = ~270 | ~1.5 hours | ~$2.25 |
| LiMn2O4 (448 atoms) | ~15 common | 15 * (1 ord + 8 SQS) * 2 states = ~270 | ~3 hours | ~$4.50 |
| SrTiO3 (320 atoms) | ~15 common | 15 * (1 ord + 8 SQS) = ~135 | ~1 hour | ~$1.50 |
| **Total** | | **~675 + retries** | **~6 hours** | **~$8-10** |

Note: SrTiO3 has fewer relaxations because no delithiation step is needed (no voltage calculation).
Retries (FIRE stages) may add 20-30% to total time.

### Smoke Test First
- Fe + Zr on LiCoO2 only: ~45 min on T4 (free)
- Validates: FrechetCellFilter active, retry logic works, volume_change non-zero

---

## 10. Timeline

| Week | Task |
|------|------|
| Week 1 | Smoke test (Fe+Zr on Colab T4). Obtain SrTiO3 CIF. Run Stages 1-3 for all 3 materials. Determine common dopant set. |
| Week 2 | Full production runs: LiCoO2 + LiMn2O4 + SrTiO3 on Colab A100 (~6 hours total). |
| Week 3 | Extract DFT ground truth from Yao 2025, JPS 2025, SrTiO3 literature. Compute all rho values. |
| Week 4 | Post-processing: b_proxy analysis, convergence plots, cross-system comparison figures. |
| Week 5-6 | Write Disorder Gap paper draft. |

---

## 11. Success Criteria

The experiment succeeds if ANY of these outcomes occurs:

**Outcome A (strongest): Structure-dependent disorder sensitivity confirmed.**
- At least one material shows rho < 0.8 for voltage or formation energy
- At least one material shows rho > 0.9 (disorder doesn't matter)
- b_proxy correlates with an identifiable descriptor
- Clean narrative: "Disorder matters for X but not Y, and here's how to predict which"

**Outcome B (strong): Disorder matters everywhere.**
- All three materials show rho < 0.8
- b_proxy varies by dopant but not by structure
- Narrative: "Ordered screening is never safe — always use SQS"

**Outcome C (still publishable): Disorder doesn't matter.**
- All three materials show rho > 0.9
- Null hypothesis confirmed
- Narrative: "We tested whether disorder matters across 3 structure types and found it doesn't.
  Ordered screening is sufficient. The community can stop worrying."
- This is actually a valuable result — it gives permission to keep using simpler methods.

**The experiment FAILS only if:**
- MACE-MP-0 produces unphysical results (negative volumes, divergent energies)
- Convergence rates are too low (<50%) even with retries
- SQS quality is too poor to approximate random alloys

These failure modes can be detected in the smoke test before committing to the full campaign.

---

## 12. Files To Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| config/targets/srtio3_perovskite.yaml | Create | SrTiO3 target configuration |
| data/structures/srtio3_parent.cif | Create | Download from MP (mp-5229) |
| config/pipeline_srtio3.yaml | Create | Pipeline config for SrTiO3 |
| colab_nmc_smoketest.ipynb | Already created | Fe+Zr smoke test |
| colab_nmc_full.ipynb | Create | Full NMC production run |
| colab_srtio3_eval.ipynb | Create | SrTiO3 production run |
| colab_lnmo_eval.ipynb | Already updated | LNMO production run |
| evaluation/eval_disorder.py | Already updated | Retry + FrechetCellFilter + metadata |
| stages/stage5/property_calculator.py | Already updated | FrechetCellFilter enabled |
| config/pipeline_444.yaml | Already updated | 8 SQS realisations |
| evaluation/PLANNED_CHANGES.md | Superseded by this document | Historical record |
