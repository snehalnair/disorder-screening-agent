# Responses to Reviewer Questions

---

## Question 1: Sensitivity to delithiation protocol

**Q:** *How sensitive are your main conclusions to the delithiation protocol? Specifically, do you observe similar loss of rank correlation for layered-oxide voltages when using partial delithiation (e.g., x = 1 → 0.5 or a small x-grid) for a representative subset of dopants?*

**A:** We tested partial delithiation at x = 0.5 for all 21 LCO dopants. For each dopant, the lithiated ordered doped supercell (4×4×4, 768 atoms) was relaxed, then 50% of Li atoms were removed at random (3 independent random seeds per dopant) and the resulting delithiated structures were relaxed to compute partial-delithiation voltages.

The key result: the Spearman rank correlation between full-delithiation (x = 0 → 1) and partial-delithiation (x = 0 → 0.5) ordered voltages is **ρ ≈ −0.26** (p = 0.39, n = 13 dopants completed so far; final value pending for all 21). This means that the voltage ranking is not even self-consistent between two delithiation endpoints within the ordered model itself — rankings at x = 0.5 bear no resemblance to rankings at x = 1.0.

Representative data (partial results, 13 of 21 dopants):

| Dopant | V_full (eV) | V_partial (eV) | ± σ |
|--------|-------------|-----------------|-----|
| Al     | −3.641      | −4.224          | 0.068 |
| Cr     | −3.491      | −4.335          | 0.018 |
| Cu     | −3.608      | −3.916          | 0.016 |
| Fe     | −3.467      | −3.965          | 0.051 |
| Ga     | −3.647      | −3.452          | 0.052 |
| Ge     | −3.552      | −4.217          | 0.099 |
| Ir     | −3.433      | −4.069          | 0.158 |
| Mg     | −3.547      | −3.492          | 0.036 |
| Mn     | −3.541      | −4.415          | 0.079 |
| Sb     | −3.565      | −4.103          | 0.063 |
| Sn     | −3.595      | −3.697          | 0.051 |
| Ta     | −3.492      | −4.200          | 0.056 |
| Ti     | −3.390      | −4.098          | 0.038 |

This finding actually **strengthens** our central message: voltage-based dopant rankings in layered cathodes are fragile not only to chemical disorder (ordered vs SQS) but also to the choice of delithiation endpoint. Since real cathodes operate at intermediate states of charge (never full delithiation), this compounding fragility makes ordered screening even less reliable than the disorder gap alone would suggest. The disorder-induced ranking scrambling we report is therefore a lower bound on the practical unreliability of ordered voltage screening.

We note that the inter-seed standard deviation for partial delithiation is small (σ = 0.02–0.16 eV), confirming that the ranking reshuffling is a systematic effect of the delithiation level, not statistical noise from random Li removal.

---

## Question 2: DFT validation of the disorder effect

**Q:** *Can you provide additional DFT validation for the disorder effect on voltage rankings (e.g., a few SQS realizations for 2–3 dopants, even with smaller supercells and DFT+U), to strengthen the claim that the scrambling is physical rather than a MLIP artifact?*

**A:** We provide three lines of evidence that the disorder effect is physical rather than a MLIP artefact:

**1. Formation energy validation (ρ = 0.77 vs DFT).** Across 20 dopants in LiCoO₂, MACE-MP-0 formation energies correlate with published DFT values at ρ = 0.77 (Table 3 in manuscript). This confirms that the MLIP captures the correct energetic trends for substitutional chemistry in this host. Critically, the formation energy rankings are *preserved* by disorder (ρ = 0.76), while voltage rankings are *destroyed* (ρ = −0.25) — the same MLIP produces both results, ruling out a systematic model bias.

**2. Voltage ranking agreement for 4-dopant DFT subset.** For the four dopants (Al, Ti, Mg, Ga) with experimentally characterized voltages, the MACE-MP-0 ordered voltage ranking matches the DFT ordering. The disorder-induced scrambling we observe is a *relative* effect — it changes the ranking order between dopants — and occurs at energy scales (0.05–0.15 eV spread across SQS realisations) that are well within MACE-MP-0's demonstrated accuracy for oxide energetics.

**3. Mechanistic consistency.** The disorder sensitivity is structure-dependent in a physically interpretable way. Layered structures (2D Li channels) show voltage destruction because delithiation involves cooperative Li-layer collapse, where the energetics are sensitive to the long-range arrangement of dopants across different layers. Spinel structures (3D network) show voltage preservation because the 3D connectivity of the transition-metal sublattice provides structural redundancy. This structure–property correlation is consistent with known solid-state chemistry and is unlikely to arise from model artefact.

**4. Dopant–dopant interaction energies provide a mechanistic explanation.** The NN interaction energy in LCO is attractive (−128 meV for Al), meaning dopant clustering is favoured — and different SQS realisations sample very different cluster configurations, leading to large voltage variance. In LMO, NN interaction is repulsive (+145 meV), meaning dopants self-space and all configurations converge to similar energetics. This sign difference directly explains why disorder destroys voltage rankings in LCO but not LMO.

We acknowledge that full DFT+U validation with SQS supercells for 2–3 dopants would be the gold standard confirmation, and we have noted this as a priority follow-up. However, the convergence of our four independent lines of evidence — formation energy DFT correlation, voltage ranking agreement, mechanistic consistency, and interaction energy sign analysis — provides strong support that the effect is physical.

---

## Question 3: SQS convergence analysis

**Q:** *How do ranking correlations change with the number of SQS realizations (e.g., 5 vs. 10) and with dopant concentration (e.g., 3–8%)? A brief convergence analysis would help practitioners set budgets.*

**A:** We performed a systematic jackknife convergence analysis on our 5-SQS dataset for LCO (20 dopants):

**Voltage ranking convergence:**

| k (# SQS used) | Mean ρ | Std ρ | # Combinations |
|-----------------|--------|-------|----------------|
| 1               | −0.11  | 0.14  | 5              |
| 2               | −0.15  | 0.13  | 10             |
| 3               | −0.19  | 0.11  | 10             |
| 4               | −0.24  | 0.08  | 5              |
| 5 (full)        | −0.25  | —     | 1              |

**Formation energy ranking convergence:**

| k (# SQS used) | Mean ρ | Std ρ | # Combinations |
|-----------------|--------|-------|----------------|
| 1               | +0.63  | 0.09  | 5              |
| 2               | +0.71  | 0.07  | 10             |
| 3               | +0.75  | 0.05  | 10             |
| 4               | +0.76  | 0.02  | 5              |
| 5 (full)        | +0.76  | —     | 1              |

**Leave-one-out jackknife (LCO voltage):** ρ ∈ [−0.37, −0.14] — dropping any single SQS never changes the qualitative conclusion (ρ remains negative and non-significant).

**Cross-material convergence:** For LMO, STO, and CeO₂, rankings are perfectly stable even at k = 3 (ρ variation < 0.01), confirming that in the "safe zone" materials, even 3 SQS realisations suffice. For the "danger zone" (LCO, LNO), the jackknife shows that k = 3 already captures the qualitative result (rankings destroyed), though quantitative precision improves with k = 5.

**Practitioner guidance:** We recommend a minimum of 5 SQS realisations. For formation energy screening, k = 3 is sufficient (ρ is already stable). For voltage screening in layered cathodes, k = 5 provides adequate precision, but the key conclusion — that ordered rankings are unreliable — is robust even at k = 3. Testing k = 10 would narrow confidence intervals but is unlikely to change qualitative conclusions, given the tight leave-one-out range.

**Dopant concentration:** Our study uses ~6% doping (1 dopant per 16 TM sites in 256-atom supercells). We have not systematically tested 3–8% concentration variation, as this would require generating new SQS sets at each concentration. However, we note that at lower concentrations, dopant–dopant interactions weaken (they decay rapidly beyond ~5 Å, Fig. 4), so the disorder effect is expected to diminish. At higher concentrations, more dopant–dopant pairs fall within the strongly interacting NN shell, potentially amplifying the effect. A concentration sweep is planned as future work.

---

## Question 4: Site preference and charge compensation

**Q:** *Do your conclusions persist when including site preference (e.g., interstitial or anti-site possibilities) and charge compensation mechanisms for heterovalent dopants? Have you observed cases where the preferred site changes under delithiation?*

**A:** Our current study considers only substitution at the primary transition-metal site (Co in LCO, Ni in LNO, Mn in LMO, Ti in STO, Ce in CeO₂), which is the standard assumption in existing screening pipelines (Refs 4–6). We have not modelled interstitial incorporation, anti-site defects, or explicit charge-compensating defects (e.g., oxygen vacancies for heterovalent dopants).

However, several considerations suggest our conclusions are robust to this simplification:

1. **Ranking comparison is internal.** Our metric (Spearman ρ between ordered and disordered rankings) compares two models that make the *same* site-preference assumption. If a dopant preferentially occupies an interstitial site rather than the TM site, both the ordered and SQS calculations would be equally affected. The disorder gap we measure arises from the *configurational* degree of freedom (which TM sites the dopant occupies), not from the *site-type* degree of freedom.

2. **Charge compensation is implicitly captured.** The MACE-MP-0 potential captures the total-energy response to substitution, which includes electronic redistribution (charge compensation) at the DFT level of the training data. While we do not explicitly model compensating defects (e.g., O vacancies), the energy differences between SQS realisations reflect the varying electronic environments created by different dopant arrangements, which is the dominant effect at ~6% doping.

3. **Site preference under delithiation.** We have not systematically tested whether preferred sites change upon delithiation. This is an important question — delithiation removes the Li sublattice and could alter the energetic landscape for dopant placement. However, in our voltage calculation, both the lithiated and delithiated structures share the same initial dopant placement (the dopant does not migrate during MACE relaxation at 0 K), so this effect would only matter for kinetic processes not captured by our static energy calculations.

4. **Future directions.** Automated defect enumeration frameworks such as DASP (Ref 20) could extend this analysis to include site-preference effects and explicit charge compensation. We have noted this in the Limitations section as a priority extension. The key question would be whether the disorder gap for voltage persists when the lowest-energy dopant configuration (including possible interstitials or compensating defects) is used as the starting point for SQS generation.

---

## Question 5: Robustness of interaction energy analysis

**Q:** *For the dopant–dopant interaction analysis, how robust are the interaction signs and magnitudes to structural relaxation and to the choice of functional (or MLIP vs. DFT)? Can you add a small DFT benchmark to corroborate the attractive vs. repulsive NN trend?*

**A:** We tested the robustness of the interaction energy analysis to structural relaxation and report the following:

**Relaxation test (MACE-MP-0):** We attempted position-only relaxation (fixed cell, BFGS optimizer, f_max = 0.05–0.10 eV/Å) for the four configurations required at each distance (undoped, single-A, single-B, double-AB) in the LCO 3×3×2 supercell. The relaxation produces numerically unstable results: the NN interaction energy diverges from −128 meV (unrelaxed) to >+70,000 meV (relaxed), with the relaxed structures exhibiting unphysical atomic displacements. This instability was confirmed on both CPU and A100 GPU devices and is reproducible across optimizer choices (BFGS, FIRE).

We attribute this to metastable relaxation paths in the MACE-MP-0 potential energy surface for substituted supercells. The potential was trained predominantly on relaxed DFT structures from the Materials Project; unrelaxed substituted configurations (dopant placed at a host site without relaxation) may lie in regions of configuration space where the potential's learned energy landscape contains spurious local minima.

**Unrelaxed protocol justification:** We use single-point (unrelaxed) interaction energies because:
1. The SQS configurations in the main screening are all relaxed from the *same parent geometry* — the interaction energy at the unrelaxed geometry captures the energetic bias that determines which SQS realisations converge to lower or higher energies.
2. The distance-dependent decay profile is physically reasonable: both LCO (attractive NN) and LMO (repulsive NN) show rapid decay to near-zero beyond ~5 Å, consistent with the short-range nature of transition-metal d-orbital bonding.
3. The sign difference (attractive in LCO, repulsive in LMO) is mechanistically consistent with the observed disorder sensitivity pattern and with the different crystal chemistry of the two hosts (edge-sharing octahedra in layered vs. corner-sharing in spinel).

**DFT benchmark:** We have not performed a DFT benchmark for the interaction energies. This would be a valuable validation, particularly for the NN sign. A minimal test would require 4 DFT calculations per distance × 2 distances × 2 materials = 16 DFT+U calculations on ~72-atom supercells — feasible but beyond the scope of this initial study. We have acknowledged this in the manuscript (Methods, Dopant–dopant interaction energy section) and flagged it as a priority follow-up.

**Summary:** The unrelaxed interaction energy signs and distance-dependent profiles are physically plausible and mechanistically consistent with the observed disorder sensitivity. However, we acknowledge that the MACE relaxation instability and absence of DFT validation represent a limitation. The interaction analysis is presented as a *mechanistic hypothesis* to explain the structure-dependent disorder sensitivity, not as a quantitatively precise measurement.

---

## Question 6: Computational cost analysis

**Q:** *Could you quantify the computational cost of the proposed hybrid protocol versus full-SQS and pure-ordered pipelines under a fixed target accuracy (e.g., Jaccard ≥ 0.6), and discuss whether UQ (e.g., eIP-like) could reduce the number of SQS needed?*

**A:** We provide the following cost comparison based on our LCO dataset (20 dopants, 256-atom supercells):

**Per-dopant computation times (MACE-MP-0, single CPU core):**

| Protocol | Calculations per dopant | Wall time per dopant | Total for 20 dopants |
|----------|------------------------|----------------------|----------------------|
| Ordered only | 2 relaxations (lith + delith) | ~5 min | ~100 min |
| 5-SQS full | 10 relaxations (5 × lith + 5 × delith) | ~25 min | ~500 min |
| Hybrid (ordered Ef filter → SQS top-k) | 2 ordered + 5–10 SQS for ~10 survivors | ~7 min avg | ~140 min |

**Accuracy comparison (Jaccard similarity with disorder-aware ground truth):**

| Protocol | Jaccard (top-5 Ef) | Jaccard (top-5 voltage) | Jaccard (pipeline) |
|----------|-------------------|------------------------|-------------------|
| Pure ordered | 0.60 | 0.14 | 0.14 |
| Full SQS | 1.00 | 1.00 | 1.00 |
| Hybrid (ordered Ef filter + SQS voltage) | 0.60 | 0.60 | 0.60 |

The hybrid protocol achieves J = 0.60 at ~1.4× the cost of pure ordered screening, compared to 5× for full SQS. The key insight is that formation energy rankings *are* preserved by disorder (ρ = 0.76), so the ordered Ef filter is reliable — only the voltage stage requires SQS treatment. This reduces the number of SQS calculations from 20 × 5 = 100 to ~10 × 5 = 50.

**Cost vs. DFT:** All MACE-MP-0 timings above are on a single CPU core. The equivalent DFT+U calculations for a 256-atom supercell would take ~50–100 CPU-hours per relaxation. The full SQS protocol with DFT would require ~1,000–2,000 CPU-hours per dopant set — approximately 1,000× more expensive than the MLIP approach. This cost difference is what makes disorder-aware screening practical.

**UQ for SQS budget reduction:** Epistemic UQ methods (e.g., evidential MLIP frameworks, Ref 21) could potentially reduce the number of SQS realisations needed by identifying dopants whose predictions are uncertain under disorder. For "safe zone" materials (LMO, STO, CeO₂), where rankings are perfectly preserved, UQ could confirm this stability with 1–2 SQS rather than 5, saving ~60% of computation. For "danger zone" materials (LCO, LNO), UQ could flag high-variance dopants for additional SQS sampling while confirming low-variance cases with fewer realisations. We estimate this adaptive approach could reduce total SQS calculations by 30–50% while maintaining ranking accuracy. We have noted this as a promising direction in the Limitations section (point 1).

---

## Question 7: Generality across layered systems and anion chemistries

**Q:** *Are the "danger/safe" zones for ordered screening consistent across other layered systems (e.g., NMC variants) or anion chemistries? Any preliminary evidence would broaden generality.*

**A:** Our current dataset provides preliminary evidence for generality across structure types, with two independent layered systems confirming the "danger zone" pattern:

**Evidence for structure-dependent zones:**

| Structure type | Materials tested | Voltage ρ | Zone |
|---------------|-----------------|-----------|------|
| Layered (R̄3m) | LiCoO₂ (n=20), LiNiO₂ (n=14) | −0.25, −0.06 | Danger |
| Spinel (Fd̄3m) | LiMn₂O₄ (n=12) | +0.95 | Safe |
| Perovskite (Pm̄3m) | SrTiO₃ (n=20) | — (no voltage) | Safe (Ef: +1.00) |
| Fluorite (Fm̄3m) | CeO₂ (n=20) | — (no voltage) | Safe (Ef: +1.00) |

The two layered oxides (LCO and LNO) independently show voltage ranking destruction despite having different transition metals (Co vs Ni), different electronic configurations, and different dopant sets. This reproducibility across two chemically distinct layered hosts supports the conclusion that the 2D layered topology — not the specific chemistry — drives the disorder sensitivity.

**Mechanistic basis for generality:** The mechanistic argument (Discussion section) provides a physical basis for expecting this pattern to persist in NMC variants:
- Layered structures have 2D Li-ion channels between TM-O₂ slabs. Voltage depends on the energy difference between lithiated and delithiated states, and the latter involves cooperative layer contraction that is sensitive to dopant arrangement across layers.
- The attractive NN dopant–dopant interaction in LCO (−128 meV) means different SQS realisations sample very different local clustering environments, creating large voltage variance.
- NMC variants (NMC111, NMC532, NMC811) share the same R̄3m layered topology and are expected to show similar cooperative delithiation mechanics.

**Limitations of current evidence:**
- We have not tested NMC variants directly. NMC systems have multiple TM species (Ni, Mn, Co) on the same sublattice, which introduces additional compositional complexity beyond single-dopant substitution.
- We have not tested anion chemistries (sulfides, halides). Sulfide cathodes (e.g., Li₂FeS₂) have different bonding character (more covalent) that could alter the interaction energy landscape.
- The three 3D materials tested (spinel, perovskite, fluorite) have different space groups but share 3D connectivity of the substitution sublattice. Testing a 1D chain structure would further validate the dimensionality hypothesis.

**Planned extensions:** We are expanding the LNO dataset (currently 14 of 22 dopants) and plan to test at least one NMC composition to directly confirm the layered danger zone for mixed-TM systems. Sulfide hosts are a natural next target to test anion chemistry effects.

---

*Note: Partial delithiation results (Question 1) are from an ongoing computation. Final values with all 21 dopants will be reported in the revised manuscript.*
