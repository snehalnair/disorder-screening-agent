# Planned Changes & Reviewer Mitigations

## Code Changes (Priority Order)

### 1. Retry Logic for Failed Relaxations
**File:** `evaluation/eval_disorder.py` (~line 145)
**Change:** Three-stage retry chain:
- Stage 1: BFGS, fmax=0.10, max_steps=1000 (current)
- Stage 2 (on failure): FIRE, fmax=0.10, max_steps=2000
- Stage 3 (on failure): FIRE, fmax=0.20, max_steps=2000
**Reason:** Zr and Mg only 2/5 converged. Fe and W only 3/5. Current code discards failures entirely.

### 2. Enable FrechetCellFilter (Cell + Volume Relaxation)
**Files to change:**
- `evaluation/eval_disorder.py` line ~145: remove `filter_type="None"`
- `stages/stage5/property_calculator.py` lines ~103, ~430: remove `filter_type="None"`
- `colab_lnmo_eval.ipynb` cells 7-9: remove `filter_type='None'`
**Reason:** All production runs used position-only relaxation. Volume was fixed → volume_change = 0.0 for all dopants. Voltage sensitivity correlated with ionic radius mismatch (Zr, Mg highest) — cannot distinguish disorder physics from trapped strain artefact.

### 3. Save Convergence Metadata to Results JSON
**File:** `evaluation/eval_disorder.py`
**Change:** For each SQS realisation, save:
- `relaxation_steps`: number of optimizer steps
- `max_force_final`: final max force (eV/Å)
- `optimizer_used`: "BFGS" or "FIRE" (from retry)
- `fmax_used`: actual convergence criterion used
- `converged`: boolean
**Reason:** Currently discarded after convergence check. Needed to report convergence quality per dopant.

### 4. Increase SQS Realisations from 5 to 8
**File:** `config/pipeline_444.yaml` line 61
**Change:** `n_sqs_realisations: 8`
**Reason:** With retries, most should converge. 8 gives buffer to still get 5+ converged per dopant.

---

## Re-Run Campaigns

### Campaign 1: NMC (LiCoO2 proxy) — M1 Max
- 23 dopants × 8 SQS × 2 states = 368 relaxations (plus retries)
- Estimate: 12-16 hours with cell relaxation + retries

### Campaign 2: LNMO — Colab A100
- 22 dopants × 8 SQS × 2 states = 352 relaxations
- Estimate: 6-8 hours on A100

### Sanity Check First
- Run Fe (smallest radius mismatch) + Zr (largest) before full campaign (~1 hour)
- If Zr variance drops dramatically → volume was the story, not disorder
- If Zr variance persists → finding is real

---

## Reviewer Limitations & Mitigations

### L1: LiCoO2 ≠ NMC811 (CRITICAL)
**Issue:** Proxy host has no Ni → li_ni_exchange = 0 for all dopants, but weighted 25% in composite ranking. Li/Ni antisite mixing is the dominant disorder mechanism in real NMC and is completely absent.
**Mitigation:**
- Drop li_ni_exchange weight to 0 or redistribute weights
- Explicitly acknowledge LCO proxy limitation
- Note that LNMO (second system) partially addresses generalisability
- Phase 2: use actual NMC811 parent structure

### L2: 0K Static Energies, No Entropy
**Issue:** Free energy = E + PV − TS. Missing configurational entropy (S_config = −kB Σ x_i ln x_i) which favours disordered arrangements at operating temperature (~300K).
**Mitigation:** Acknowledge as standard limitation of SQS approaches. Note that entropy contribution is ~25 meV/atom at 300K for 10% doping — small compared to formation energy differences but potentially significant for voltage.

### L3: MACE-MP-0 Accuracy vs DFT
**Issue:** RQ3 shows ρ = +0.62 (p = 0.10) against experiment — not statistically significant. Unknown error bars for doped transition metal oxides.
**Mitigation:**
- Run 2-3 dopants (Al, Zr, Fe) through DFT as spot-check
- Report MACE-vs-DFT energy differences
- Strongest single addition for reviewer confidence

### L4: Single Concentration (10%)
**Issue:** Rankings could invert at 5% vs 15%. Config generates both 5% and 10% but results only report 10%.
**Mitigation:** Report both concentrations and show whether rankings are concentration-dependent. Data may already exist.

### L5: Delithiation Model
**Issue:** How is voltage computed? Full delithiation vs partial? Real NMC operates at partial delithiation (3.0-4.3V range).
**Mitigation:** Document the delithiation protocol clearly. If full delithiation, acknowledge this represents upper-bound voltage.

### L6: SQS Periodicity
**Issue:** SQS are still periodic. 5-8 realisations at 4×4×4 samples tiny fraction of configurational space.
**Mitigation:** Standard practice — acknowledge. Report SQS variance to show sampling adequacy.

### L7: Volume Constraint (RESOLVED by Change #2)
**Issue:** Position-only relaxation trapped strain energy proportional to radius mismatch.
**Mitigation:** Enable FrechetCellFilter. Compare position-only vs cell-relaxed results.

---

## NOT Adding Now

### LLZO
- Wait until cell-relaxed LCO + LNMO results are stable
- If voltage ρ still ~0 after cell relaxation → story confirmed, LLZO adds novelty but isn't needed
- If voltage ρ jumps to ~0.7+ → need to rethink narrative before adding systems
- Better as follow-up paper

### DFT Spot-Checks
- Highest-impact addition if VASP/CP2K access available
- 2-3 dopants × ordered + 1 SQS = 6-8 DFT relaxations
- Compare MACE vs DFT energies to bound MLIP error

---

## Post-Run Analysis: Arrangement Sensitivity & Bowing

### Vegard's Law / Interpolation Analysis
**Motivation:** Researcher question — can disordered properties be predicted by interpolation between end-members (e.g., NaCl → KCl)? If yes, disorder is irrelevant for that dopant. If not, SQS methods are essential.

### Proxy Bowing Parameter
For each dopant M at concentration x = 0.10:
```
b_proxy(M) = V_ordered(M, x) − V_disordered(M, x)
```
This measures "arrangement sensitivity" — how much the property depends on atomic configuration vs just composition. Small b_proxy ≈ Vegard-like (ordered cells sufficient). Large b_proxy ≈ strong bowing (SQS essential).

### Current Data (position-only, to be updated with cell-relaxed results)
| Dopant | b_proxy (V) | Radius Mismatch | Interpretation |
|--------|-------------|-----------------|----------------|
| Fe     | 0.070       | ~0%             | Near-linear, ordered cells sufficient |
| Al     | 0.123       | ~2%             | Small bowing |
| Ga     | 0.165       | ~14%            | Moderate bowing |
| Nb     | 0.175       | ~17%            | Moderate bowing |
| Ti     | 0.189       | ~11%            | Moderate bowing |
| Mg     | 0.215       | ~32%            | Strong bowing |
| W      | 0.300       | ~10%            | Strong bowing (charge mismatch?) |
| Zr     | 0.313       | ~32%            | Strongest bowing |

### Key Analysis After Cell-Relaxed Re-Run
1. **Figure: b_proxy vs ionic radius mismatch** for all 23 dopants
   - If linear correlation persists → bowing is dominated by size effect (strain)
   - If correlation breaks → genuine chemical/electronic disorder sensitivity
   - W is the test case: small radius mismatch (~10%) but large b_proxy (0.300) — if this persists after cell relaxation, it's electronic, not strain
2. **Practical rule for experimentalists:** "If radius mismatch > X%, use disorder-aware screening" — derive X from the inflection point in the b_proxy vs mismatch curve
3. **Per-system comparison:** Does LNMO show smaller b_proxy across the board? If yes, spinel structure is inherently more Vegard-like → ordered screening is valid for spinels
4. **Extend to formation energy:** Compute b_proxy for formation energy too. Current ρ = +0.956 suggests small bowing for formation energy — but worth quantifying per-dopant

### No New Runs Needed
This is purely a post-processing analysis on existing (and upcoming cell-relaxed) data. Add to `evaluation/figures.py` as a new figure.

---

## Publication Strategy

### Paper 1: DisorderBench → NeurIPS 2026 Datasets & Benchmarks Track
**Deadline:** ~June 2026
**Status:** Closest to ready — benchmark exists, experiments run, findings clear

**Headline:** "The explain-execute gap, not a disorder-knowledge gap, is the primary barrier to AI-assisted materials design."

**Core findings (keep):**
- MCQ overestimates capability by 27-57 pp
- Explain-execute gap persists across model sizes (8B matches 200B on hardest tasks)
- Prompt framing effect (critique vs recommend)
- Cross-system validation (4 material systems)

**Include as infrastructure only (2-3 paragraphs in methods):**
- Pipeline as ground truth source
- SQS methodology (brief)
- Cell-relaxed ρ values (one sentence)

**Cut entirely:**
- Pipeline methodology detail (supercell sizes, SQS generation, MLIP choices)
- Disorder-changes-rankings story (it's ground truth, not the finding)
- Bowing/arrangement sensitivity analysis (save for Paper 2)

**Before submission — must complete:**
- [ ] Human validation of LLM-as-judge scores (Cohen's kappa) — mentioned in limitations
- [ ] Finalise variance analysis (3 runs × 10 hardest × 5 models — done)
- [ ] Update EX-05 ground truth with cell-relaxed Zr data
- [ ] Clean up question numbering (61 in abstract, 46 in contributions, 36+10+25 in body)

**Reviewer risks:**
- "Is 61 questions enough?" → Mitigate with variance analysis + per-category breakdown
- "Is LLM-as-judge validated?" → Human validation subset + cross-model judging
- "Is this disorder-specific or general?" → Section 5.4 shows it's general (execution gap, not disorder gap)

---

### Paper 2: Disorder Gap → Nature Computational Science (rolling submission)
**Timeline:** After cell-relaxed re-runs complete (~2-3 weeks)
**Status:** Needs clean results + bowing analysis

**Headline:** "Computational dopant screening gives systematically wrong voltage rankings because it ignores chemical disorder. We quantify this across two material systems and provide both a diagnostic and a fix."

**Core findings:**
- Voltage rankings destroyed by disorder in layered oxide (ρ ≈ ? after cell relaxation)
- Voltage rankings preserved in spinel (ρ ≈ 1.0) → structure-dependence
- Arrangement sensitivity (b_proxy) as a novel diagnostic metric
- Practical threshold: "if radius mismatch > X%, use SQS"
- W anomaly: electronic vs size-driven disorder sensitivity
- Volume change now meaningful (non-zero, differentiates dopants)
- Pipeline runs on laptop (MACE-MP-0, no HPC) — democratises access

**What makes it Nature-level:**
- 80% disorder prevalence (Fritz Haber, Dec 2025) gives urgency
- Practical rule for experimentalists is immediately actionable
- Two systems showing opposite behaviour = clean comparative story
- Laptop-accessible pipeline lowers barrier to adoption

**Before submission — must complete:**
- [ ] Cell-relaxed NMC re-run (23 dopants × 8 SQS) — overnight on M1 Max
- [ ] Cell-relaxed LNMO re-run (22 dopants × 8 SQS) — Colab A100
- [ ] Bowing analysis figure (b_proxy vs radius mismatch)
- [ ] SQS convergence plot (property vs number of realisations)
- [ ] Reconcile Zr narrative with cell-relaxed data
- [ ] Fix li_ni_exchange weight (drop to 0 for LCO proxy)
- [ ] DFT spot-checks on 2-3 dopants (if VASP access available — strongest addition)

**Reviewer risks:**
- "LiCoO₂ is not NMC811" → Frame as proxy with explicit limitations, note LNMO as second system
- "MACE-MP-0 accuracy?" → RQ3 experimental validation + DFT spot-checks if available
- "Heterovalent SQS convergence?" → 256-atom cell is standard (cite literature), 8 realisations, convergence plot
- "No entropy?" → Acknowledge; configurational entropy ~25 meV/atom at 300K, small vs formation energy differences

---

### Paper 3 (Future): Platform Vision → Nature Comp. Sci. Perspective/Commentary
**Timeline:** After Papers 1 & 2 accepted
**Status:** No new experiments — writeup only

**Headline:** "Autonomous materials discovery agents assume ordered structures and trust LLM recommendations. Both assumptions are wrong. Here's the architecture for disorder-aware, capability-validated materials agents."

**Key arguments:**
1. **Disorder-awareness as an agent module** — SQS pathway activated when b_proxy > threshold
2. **Capability gating** — DisorderBench-style tests before trusting LLM for execution tasks
3. **Hierarchical funnel pattern** — reusable across electrolytes, catalysts, alloys
4. **SQS variance as epistemic uncertainty** — natural acquisition function for active learning
5. **Multi-fidelity screening** — MLIP → DFT → experiment, with disorder at each tier

**Cites:** Paper 1 (DisorderBench) + Paper 2 (Disorder Gap) + Fritz Haber 80% + GNoME/MatterGen as examples of order-assuming platforms

---

### Summary Timeline

| Month | Milestone |
|-------|-----------|
| April 2026 | Cell-relaxed re-runs complete (NMC + LNMO) |
| April 2026 | Bowing analysis + convergence plots |
| May 2026 | Human validation of LLM-as-judge (Cohen's kappa) |
| May 2026 | DisorderBench draft finalised |
| June 2026 | **Submit DisorderBench → NeurIPS** |
| June-July 2026 | Disorder Gap paper draft with cell-relaxed results |
| July 2026 | DFT spot-checks (if VASP access) |
| Aug 2026 | **Submit Disorder Gap → Nature Comp. Sci.** |
| Late 2026 | Platform Vision perspective (after acceptances) |
