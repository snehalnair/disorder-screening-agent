# Supplementary Information

## Chemical Disorder Invalidates Voltage Rankings in Computational Dopant Screening for Layered Battery Cathodes

Snehal Nair

---

## Contents

- [S1. Dopant pre-screening pipeline](#s1-dopant-pre-screening-pipeline)
- [S2. Per-dopant data tables for all nine materials](#s2-per-dopant-data-tables)
- [S3. SQS convergence: jackknife subsampling](#s3-sqs-convergence)
- [S4. Predictor threshold sensitivity sweep](#s4-threshold-sweep)
- [S5. Monte Carlo clustering analysis](#s5-mc-clustering)
- [S6. Partial delithiation detailed results](#s6-partial-delithiation)
- [S7. Interaction energy details](#s7-interaction-energy)
- [S8. Disorder-risk predictor: complete observation table](#s8-predictor-table)

---

## S1. Dopant pre-screening pipeline

Candidate dopants for each host material were selected by a three-stage automated pre-screen:

1. **SMACT charge neutrality.** The dopant must be able to adopt an oxidation state that maintains overall charge neutrality when substituted for the host TM ion. We use the SMACT library to enumerate allowed oxidation states.

2. **Shannon ionic radius mismatch.** The dopant's Shannon ionic radius (at the appropriate coordination number and oxidation state) must be within 35% of the host ion's radius: |r_dopant - r_host| / r_host <= 0.35.

3. **Hautier-Ceder substitution probability.** The ICSD-mined substitution probability (Hautier et al., *Chem. Mater.* 2011) for the dopant-host pair must exceed 0.001.

Because these filters depend on the host ion identity (e.g., Co^3+ in LiCoO2 vs Fe^2+ in LiFePO4), the dopant sets are material-specific. Approximately 80% of dopants overlap across materials, but the exact sets differ. Supplementary Table S2.1-S2.9 list the dopants screened for each material.

---

## S2. Per-dopant data tables for all nine materials

All energies in eV/atom (formation energy) or V (voltage). Volume change in %. "Ord" = ordered supercell; "Dis" = SQS-ensemble mean; "Dis std" = inter-realisation standard deviation.

### Table S2.1. LiCoO2 (layered R-3m, 4x4x4, 256 atoms, 21 dopants)

| Dopant | Ef Ord | Ef Dis | V Ord | V Dis | V Dis std | dV Ord (%) | dV Dis (%) | n_SQS |
|--------|--------|--------|-------|-------|-----------|------------|------------|-------|
| Al | -4.788 | -4.944 | -3.641 | -3.480 | — | 0.10 | 4.11 | 5 |
| Cr | -4.856 | -5.056 | -3.491 | -3.495 | — | 0.44 | 0.33 | 5 |
| Cu | -4.737 | -4.894 | -3.608 | -3.453 | — | 0.19 | 4.19 | 5 |
| Fe | -4.826 | -5.067 | -3.467 | -3.531 | — | 0.53 | 4.83 | 5 |
| Ga | -4.760 | -4.899 | -3.647 | -3.465 | — | 7.17 | 4.29 | 5 |
| Ge | -4.817 | -4.982 | -3.552 | -3.511 | — | 0.10 | 1.44 | 5 |
| Ir | -4.883 | -5.060 | -3.433 | -3.452 | — | 5.23 | 2.01 | 5 |
| Mg | -4.712 | -4.932 | -3.547 | -3.470 | — | 13.35 | 0.98 | 5 |
| Mn | -4.861 | -5.051 | -3.541 | -3.414 | — | 10.52 | 2.91 | 5 |
| Mo | -4.900 | -5.092 | -3.390 | -3.519 | — | 8.95 | 5.84 | 5 |
| Nb | -4.901 | -5.096 | -3.536 | -3.442 | — | 1.39 | 1.66 | 5 |
| Ni | -4.784 | -4.998 | -3.458 | -3.542 | — | 6.86 | 0.69 | 5 |
| Rh | -4.840 | -4.964 | -3.452 | -3.496 | — | 0.34 | 2.33 | 5 |
| Ru | -4.885 | -5.044 | -3.402 | -3.462 | — | 7.22 | 2.11 | 5 |
| Sb | -4.785 | -4.892 | -3.565 | -3.471 | — | 0.20 | 4.00 | 5 |
| Sc | — | — | — | — | — | — | — | — |
| Sn | -4.779 | -4.946 | -3.595 | -3.486 | — | 0.27 | 1.72 | 5 |
| Ta | -4.943 | — | -3.492 | — | — | 0.70 | — | 0 |
| Ti | -4.881 | -5.082 | -3.390 | -3.500 | — | 7.63 | 2.41 | 5 |
| V | -4.878 | -5.057 | -3.548 | -3.485 | — | 0.45 | 4.83 | 5 |
| W | -4.900 | -5.096 | -3.420 | -3.468 | — | 9.62 | 7.09 | 5 |
| Zr | -4.918 | -4.980 | -3.519 | -3.523 | — | 1.17 | 4.77 | 5 |

**Spearman rho:** Ef = +0.76, Voltage = -0.25, Volume = +0.09

### Table S2.2. LiNiO2 (layered R-3m, 4x4x4, 256 atoms, 14 dopants)

| Dopant | Ef Ord | Ef Dis | V Ord | V Dis | dV Ord (%) | dV Dis (%) | n_SQS |
|--------|--------|--------|-------|-------|------------|------------|-------|
| Al | -4.337 | -4.661 | -3.461 | -3.439 | 11.83 | 4.08 | 5 |
| Co | -4.499 | -4.720 | -3.257 | -3.499 | 7.60 | 1.89 | 5 |
| Cr | -4.493 | -4.766 | -3.503 | -3.480 | 7.57 | 2.40 | 5 |
| Fe | -4.412 | -4.729 | -3.279 | -3.446 | 9.92 | 2.83 | 5 |
| Ga | -4.351 | -4.654 | -3.618 | -3.452 | 1.25 | 0.80 | 5 |
| Ge | -4.536 | -4.684 | -3.364 | -3.513 | 0.52 | 2.71 | 5 |
| Mg | -4.399 | -4.543 | -3.412 | -3.499 | 10.85 | 5.76 | 5 |
| Mn | -4.573 | -4.807 | -3.304 | -3.485 | 11.52 | 5.07 | 5 |
| Nb | -4.589 | -4.852 | -3.452 | -3.517 | 1.06 | 2.31 | 5 |
| Sn | -4.330 | -4.651 | -3.440 | -3.535 | 0.50 | 3.61 | 5 |
| Ti | -4.532 | -4.774 | -3.424 | -3.529 | 1.32 | 1.83 | 5 |
| V | -4.425 | -4.715 | -3.456 | -3.449 | 3.10 | 2.68 | 5 |
| W | -4.589 | -4.730 | -3.584 | -3.461 | 9.71 | 3.31 | 5 |
| Zr | -4.572 | -4.812 | -3.847 | -3.533 | 1.34 | 1.45 | 5 |

**Spearman rho:** Ef = +0.82, Voltage = -0.06, Volume = +0.54

### Table S2.3. NMC811 (layered R-3m, 4x4x4, 256 atoms, 16 dopants)

| Dopant | Ef Ord | Ef Dis | V Ord | V Dis | V Dis std | n_SQS |
|--------|--------|--------|-------|-------|-----------|-------|
| Al | -4.720 | -4.713 | -4.240 | -4.198 | 0.120 | 5 |
| Cr | -4.734 | -4.728 | -4.345 | -4.234 | 0.040 | 5 |
| Cu | -4.720 | -4.710 | -4.275 | -4.231 | 0.050 | 5 |
| Fe | -4.732 | -4.734 | -4.190 | -4.262 | 0.070 | 5 |
| Ga | -4.720 | -4.719 | -4.305 | -4.292 | 0.015 | 5 |
| Ge | -4.728 | -4.718 | -4.377 | -4.230 | 0.026 | 5 |
| Mg | -4.708 | -4.714 | -4.325 | -4.313 | 0.038 | 5 |
| Mo | -4.756 | -4.734 | -4.263 | -4.162 | 0.079 | 5 |
| Nb | -4.738 | -4.753 | -4.295 | -4.273 | 0.042 | 5 |
| Sb | -4.726 | -4.731 | -4.357 | -4.318 | 0.052 | 5 |
| Sn | -4.716 | -4.722 | -4.188 | -4.259 | 0.030 | 5 |
| Ta | -4.719 | -4.754 | -4.135 | -4.315 | 0.025 | 5 |
| Ti | -4.741 | -4.741 | -4.305 | -4.256 | 0.033 | 5 |
| V | -4.741 | -4.733 | -4.341 | -4.256 | 0.044 | 5 |
| W | -4.752 | -4.747 | -4.289 | -4.217 | 0.027 | 5 |
| Zr | -4.755 | -4.747 | -4.333 | -4.272 | 0.068 | 5 |

**Spearman rho:** Ef = +0.52, Voltage = +0.09

### Table S2.4. NaCoO2 (layered R-3m, O3-type, 19 candidates, 18 converged)

Data from out-of-sample validation run. NaCoO2 is isostructural with LiCoO2 (same R-3m layered topology) but with Na+ instead of Li+ as the alkali ion.

| Dopant | V Ord | V Dis | V Dis std | Ef Ord | Ef Dis | dV Ord (%) | dV Dis (%) | n_SQS |
|--------|-------|-------|-----------|--------|--------|------------|------------|-------|
| Al | — | — | — | — | — | — | — | 5 |
| Co | — | — | — | — | — | — | — | 5 |
| Cr | — | — | — | — | — | — | — | 5 |
| Cu | — | — | — | — | — | — | — | 5 |
| Fe | — | — | — | — | — | — | — | 5 |
| Ga | — | — | — | — | — | — | — | 5 |
| Ge | — | — | — | — | — | — | — | 5 |
| Mg | — | — | — | — | — | — | — | 5 |
| Mn | — | — | — | — | — | — | — | 5 |
| Mo | — | — | — | — | — | — | — | 5 |
| Nb | — | — | — | — | — | — | — | 5 |
| Ni | — | — | — | — | — | — | — | 5 |
| Ru | — | — | — | — | — | — | — | 5 |
| Sc | — | — | — | — | — | — | — | 5 |
| Sn | — | — | — | — | — | — | — | 5 |
| Ti | — | — | — | — | — | — | — | 5 |
| V | — | — | — | — | — | — | — | 5 |
| Zn | — | — | — | — | — | — | — | 5 |
| Zr | — | — | — | — | — | — | — | 5 |

**Spearman rho:** Ef = +0.79, Voltage = +0.23, Volume = -0.01

*Note: Per-dopant numerical values to be populated from nco_screening_results.json (Colab output pending transfer).*

### Table S2.5. LiMn2O4 (spinel Fd-3m, 4x4x4, 256 atoms, 12 dopants)

| Dopant | Ef Ord | Ef Dis | V Ord | V Dis | dV Ord (%) | dV Dis (%) | n_SQS |
|--------|--------|--------|-------|-------|------------|------------|-------|
| Al | -6.902 | -6.901 | -4.409 | -4.430 | 5.04 | 4.85 | 8 |
| Co | -6.787 | -6.785 | -4.329 | -4.309 | 5.16 | 5.03 | 8 |
| Cr | -6.935 | -6.934 | -4.348 | -4.374 | 4.91 | 5.07 | 5 |
| Cu | -6.716 | -6.714 | -4.359 | -4.363 | 4.84 | 4.86 | 5 |
| Fe | -6.835 | -6.834 | -4.400 | -4.410 | 4.80 | 4.87 | 5 |
| Ga | -6.794 | -6.792 | -4.399 | -4.421 | 5.25 | 5.14 | 5 |
| Mg | -6.786 | -6.781 | -4.527 | -4.541 | 3.59 | 3.19 | 5 |
| Ni | -6.721 | -6.717 | -4.438 | -4.462 | 4.76 | 4.94 | 5 |
| Ti | -7.046 | -7.044 | -4.378 | -4.369 | 5.48 | 5.47 | 5 |
| V | -6.946 | -6.944 | -4.272 | -4.258 | 4.57 | 4.65 | 5 |
| Zn | -6.691 | -6.685 | -4.500 | -4.514 | 3.73 | 3.79 | 5 |
| Zr | -7.072 | -7.072 | -4.352 | -4.340 | 5.14 | 5.53 | 5 |

**Spearman rho:** Ef = +1.00, Voltage = +0.95, Volume = +0.84

### Table S2.6. SrTiO3 (perovskite Pm-3m, 4x4x4, 320 atoms, 20 dopants)

| Dopant | Ef Ord | Ef Dis | dV Ord (%) | dV Dis (%) | n_SQS |
|--------|--------|--------|------------|------------|-------|
| Al | -7.910 | -7.909 | 0.81 | 0.02 | 8 |
| Co | -7.833 | -7.833 | 0.55 | 0.01 | 8 |
| Cr | -7.936 | -7.936 | 0.01 | 0.00 | 8 |
| Cu | -7.782 | -7.783 | 0.01 | 0.01 | 8 |
| Fe | -7.873 | -7.873 | 0.00 | 0.00 | 8 |
| Ga | -7.840 | -7.840 | 0.01 | 0.00 | 8 |
| La | -7.910 | -7.911 | 5.43 | 5.11 | 8 |
| Mg | -7.815 | -7.813 | 0.80 | 0.03 | 8 |
| Mn | -7.917 | -7.917 | 0.54 | 0.01 | 8 |
| Mo | -7.931 | -7.930 | 0.71 | 0.03 | 8 |
| Nb | -8.047 | -8.046 | 0.97 | 0.03 | 8 |
| Ni | -7.776 | -7.776 | 0.00 | 0.01 | 8 |
| Sc | -7.980 | -7.978 | 1.17 | 0.05 | 8 |
| Sn | -7.860 | -7.859 | 0.97 | 0.04 | 8 |
| Ta | -8.102 | -8.101 | 0.85 | 0.03 | 8 |
| V | -7.956 | -7.956 | 0.00 | 0.00 | 8 |
| W | -7.957 | -7.955 | 0.86 | 0.03 | 8 |
| Y | -7.964 | -7.967 | 3.51 | 3.37 | 8 |
| Zn | -7.756 | -7.755 | 0.80 | 0.02 | 8 |
| Zr | -8.039 | -8.038 | 1.99 | 1.72 | 8 |

**Spearman rho:** Ef = +1.00, Volume = +0.94 (no voltage computed for perovskite)

### Table S2.7. CeO2 (fluorite Fm-3m, 3x3x3, 324 atoms, 20 dopants)

| Dopant | Ef Ord | Ef Dis | dV Ord (%) | dV Dis (%) | E_Ovac Ord | E_Ovac Dis | n_SQS |
|--------|--------|--------|------------|------------|------------|------------|-------|
| Al | -8.471 | -8.478 | 2.33 | 1.29 | -0.04 | -1.27 | 5 |
| Ba | -8.366 | -8.374 | 6.83 | 7.07 | -1.28 | -1.36 | 5 |
| Ca | -8.414 | -8.415 | 3.46 | 3.43 | -1.25 | -1.30 | 5 |
| Co | -8.348 | -8.361 | 3.61 | 2.17 | -0.69 | -0.84 | 5 |
| Cr | -8.518 | -8.528 | 1.75 | 1.07 | 0.51 | 0.19 | 5 |
| Cu | -8.285 | -8.288 | 4.22 | 3.57 | 0.12 | -0.67 | 5 |
| Fe | -8.424 | -8.429 | 1.97 | 1.50 | -0.18 | -0.91 | 5 |
| Gd | -8.952 | -8.951 | 3.01 | 3.10 | -0.03 | -0.42 | 5 |
| Hf | -8.857 | -8.856 | 0.76 | 0.03 | 0.95 | 0.27 | 5 |
| La | -8.639 | -8.638 | 4.26 | 4.39 | 0.04 | -0.22 | 5 |
| Mg | -8.331 | -8.353 | 2.30 | 1.63 | -1.62 | -0.46 | 5 |
| Mn | -8.489 | -8.505 | 1.87 | 1.53 | -1.27 | -1.31 | 5 |
| Nd | -8.629 | -8.627 | 3.67 | 3.78 | 0.06 | -0.28 | 5 |
| Ni | -8.243 | -8.259 | 2.90 | 1.85 | -1.18 | -1.07 | 5 |
| Pr | -8.625 | -8.623 | 3.95 | 4.04 | 0.08 | -0.11 | 5 |
| Sm | -8.632 | -8.631 | 3.28 | 3.38 | 0.03 | -0.41 | 5 |
| Sr | -8.393 | -8.393 | 4.88 | 5.05 | -1.15 | -1.20 | 5 |
| Ti | -8.714 | -8.712 | 2.02 | 1.12 | 1.42 | 0.34 | 5 |
| Y | -8.695 | -8.694 | 2.66 | 2.69 | -0.16 | -0.60 | 5 |
| Zr | -8.798 | -8.797 | 1.05 | 0.08 | 1.10 | 0.62 | 5 |

**Spearman rho:** Ef = +1.00, Volume = +0.96, O-vacancy energy = +0.85

### Table S2.8. LiFePO4 (olivine Pnma, 2x2x2, 224 atoms, 18 dopants)

| Dopant | Ef Ord | Ef Dis | V Ord | V Dis | V Dis std | dV Ord (%) | dV Dis (%) | n_SQS |
|--------|--------|--------|-------|-------|-----------|------------|------------|-------|
| Al | -6.835 | -6.833 | -5.188 | -5.175 | 0.008 | 5.28 | 5.52 | 5 |
| Co | -6.806 | -6.806 | -5.486 | -5.487 | 0.003 | 5.57 | 5.31 | 5 |
| Cr | -6.870 | -6.870 | -5.307 | -5.288 | 0.002 | 5.01 | 6.43 | 5 |
| Cu | -6.760 | -6.760 | -5.473 | -5.478 | 0.001 | 4.37 | 4.59 | 5 |
| Ga | -6.772 | -6.771 | -5.226 | -5.201 | 0.006 | 4.63 | 5.66 | 5 |
| Ge | -6.782 | -6.782 | -5.136 | -5.150 | 0.007 | 8.29 | 7.25 | 5 |
| Mg | -6.809 | -6.809 | -5.497 | -5.505 | 0.006 | 4.86 | 4.76 | 5 |
| Mn | -6.874 | -6.874 | -5.424 | -5.424 | 0.001 | 5.77 | 6.04 | 5 |
| Mo | -6.844 | -6.844 | -5.161 | -5.146 | 0.011 | 3.40 | 4.81 | 5 |
| Nb | -6.900 | -6.898 | -4.823 | -4.819 | 0.009 | 4.29 | 4.08 | 5 |
| Ni | -6.763 | -6.763 | -5.531 | -5.537 | 0.006 | 4.84 | 5.24 | 5 |
| Ru | -6.827 | -6.827 | -5.160 | -5.172 | 0.007 | 5.82 | 5.54 | 5 |
| Sc | -6.917 | -6.916 | -5.195 | -5.176 | 0.005 | 4.32 | 4.88 | 5 |
| Sn | -6.776 | -6.775 | -5.098 | -5.102 | 0.011 | 7.66 | 7.19 | 5 |
| Ti | -6.903 | -6.901 | -4.959 | -4.959 | 0.004 | 5.49 | 5.19 | 5 |
| V | -6.875 | -6.875 | -5.223 | -5.224 | 0.002 | 6.10 | 5.68 | 5 |
| Zn | -6.746 | -6.746 | -5.496 | -5.503 | 0.006 | 5.21 | 4.91 | 5 |
| Zr | -6.928 | -6.925 | -4.918 | -4.899 | 0.006 | 4.57 | 5.18 | 5 |

**Spearman rho:** Ef = +1.00, Voltage = +0.99, Volume = +0.79

### Table S2.9. Na3V2(PO4)3 NASICON (R-3c, 1x1x1, 126 atoms, 18 dopants)

| Dopant | Ef Ord | Ef Dis | V Ord | V Dis | V Dis std | dV Ord (%) | dV Dis (%) | n_SQS |
|--------|--------|--------|-------|-------|-----------|------------|------------|-------|
| Al | -6.564 | -6.568 | -4.304 | -4.174 | 0.286 | 2.90 | 3.37 | 5 |
| Co | -6.607 | -6.560 | -4.713 | -4.552 | 0.163 | 8.39 | 2.67 | 5 |
| Cr | -6.677 | -6.603 | -4.103 | -4.173 | 0.330 | 9.99 | 3.73 | 5 |
| Cu | -6.581 | -6.531 | -4.822 | -4.632 | 0.066 | 11.61 | 2.99 | 5 |
| Fe | -6.585 | -6.588 | -4.431 | -4.425 | 0.119 | 3.71 | 4.04 | 5 |
| Ga | -6.546 | -6.535 | -4.194 | -4.192 | 0.493 | 5.90 | 4.82 | 5 |
| Ge | -6.541 | -6.504 | -4.119 | -4.033 | 0.350 | 6.60 | 5.68 | 5 |
| Mg | -6.527 | -6.504 | -4.470 | -4.026 | 0.241 | 4.14 | 3.90 | 5 |
| Mn | -6.583 | -6.559 | -4.480 | -4.211 | 0.292 | 0.56 | 5.35 | 5 |
| Mo | -6.436 | -6.430 | -3.646 | -3.599 | 0.545 | 1.04 | 1.28 | 5 |
| Nb | -6.627 | -6.507 | -4.132 | -3.453 | 0.556 | 2.42 | 5.98 | 5 |
| Ni | -6.603 | -6.534 | -4.511 | -4.321 | 0.260 | 8.43 | 4.74 | 5 |
| Ru | -6.549 | -6.454 | -4.406 | -3.663 | 0.376 | 0.60 | 1.88 | 5 |
| Sc | -6.832 | -6.680 | -4.270 | -4.404 | 0.035 | 10.07 | 3.25 | 5 |
| Sn | -6.529 | -6.505 | -4.172 | -4.071 | 0.409 | 2.94 | 6.77 | 5 |
| Ti | -6.536 | -6.532 | -2.957 | -3.642 | 0.568 | 2.23 | 3.45 | 5 |
| Zn | -6.672 | -6.516 | -4.504 | -4.428 | 0.311 | 9.88 | 5.14 | 5 |
| Zr | -6.681 | -6.607 | -4.104 | -4.003 | 0.408 | 1.66 | 4.63 | 5 |

**Spearman rho:** Ef = +0.72, Voltage = +0.77, Volume = -0.04

---

## S3. SQS convergence: jackknife subsampling

To verify that 5 SQS realisations are sufficient for stable rankings, we performed jackknife subsampling for each material. For each k in {3, 4, 5}, we computed the Spearman rho using all (n choose k) subsets and report the mean and range.

**Table S3.1. Jackknife stability of voltage rho (layered materials)**

| Material | k=3 | k=4 | k=5 |
|----------|-----|-----|-----|
| LiCoO2 | -0.22 [-0.41, -0.05] | -0.24 [-0.30, -0.18] | -0.25 |
| LiNiO2 | -0.04 [-0.22, +0.14] | -0.05 [-0.10, +0.01] | -0.06 |
| NMC811 | +0.07 [-0.10, +0.21] | +0.08 [+0.04, +0.12] | +0.09 |

Rankings are stable within +/- 0.05 from k=4 to k=5, confirming convergence.

**Table S3.2. Jackknife stability of Ef rho (all materials)**

All materials show Ef rho stable to within +/- 0.03 between k=4 and k=5, consistent with the low inter-realisation variance in formation energy.

---

## S4. Predictor threshold sensitivity sweep

We swept the risk threshold from R = 0.5 to R = 2.0 and computed predictor accuracy, false-safe rate, and false-unsafe rate for each threshold.

**Table S4.1. Threshold sweep (27 observations)**

| R threshold | Accuracy | False-safe | False-unsafe |
|-------------|----------|------------|--------------|
| 0.50 | 66.7% | 0 | 9 |
| 0.80 | 70.4% | 0 | 8 |
| 1.00 | **85.2%** | **0** | 4 |
| 1.20 | 85.2% | 1 | 3 |
| 1.50 | 92.6% | 1 | 1 |
| 1.80 | 92.6% | 1 | 1 |
| 1.93 | 81.5% | 3 | 2 |
| 2.00 | 77.8% | 5 | 1 |

The R = 1.0 threshold is the highest threshold that maintains zero false-safe predictions. Raising the threshold above 1.15 introduces false-safe predictions (NASICON volume at R = 1.15 is the first to flip). The choice of R = 1.0 represents the Pareto-optimal point for the safety-first objective.

---

## S5. Monte Carlo clustering analysis

We performed lattice Monte Carlo simulations to investigate whether dopant atoms in LiCoO2 cluster, order, or distribute randomly. This provides a thermodynamic check on the SQS assumption: if dopants strongly cluster or order at synthesis temperatures, the SQS (random solid solution) model may not be the most appropriate representation.

### Method

A 4x4x4 supercell (64 TM sites) with 4 dopant atoms (~6.25% concentration) was used. At each MC step, a randomly chosen dopant-host pair was proposed for swap. The energy change was computed using MACE-MP-0, and the swap was accepted or rejected by the Metropolis criterion. 20,000 MC steps were performed at temperatures from 300 K to 1200 K.

The Warren-Cowley short-range order parameter alpha was computed:
- alpha < 0: ordering tendency (dopants prefer unlike neighbours)
- alpha = 0: random distribution
- alpha > 0: clustering tendency (dopants prefer like neighbours)

### Results

**Table S5.1. MC clustering classification for LiCoO2 (16 dopants)**

| Dopant | E_nn (meV) | alpha (300 K) | alpha (1200 K) | Classification |
|--------|-----------|---------------|----------------|----------------|
| Al | +61 | -0.059 | -0.038 | Random |
| Cr | +418 | -0.067 | -0.065 | Ordering |
| Cu | -162 | +0.377 | +0.128 | Clustering |
| Fe | -83 | +0.350 | +0.035 | Random (high T) |
| Ga | -209 | +0.377 | +0.239 | Clustering |
| Ge | +12 | -0.034 | -0.016 | Random |
| Mg | -37 | +0.114 | -0.003 | Random |
| Mn | +238 | -0.066 | -0.062 | Ordering |
| Mo | +354 | -0.067 | -0.065 | Ordering |
| Nb | -235 | +0.378 | +0.292 | Clustering |
| Ni | -146 | +0.376 | +0.117 | Clustering |
| Sn | +384 | -0.067 | -0.065 | Ordering |
| Ti | +32 | -0.050 | -0.035 | Random |
| V | +392 | -0.067 | -0.066 | Ordering |
| W | +94 | -0.066 | -0.042 | Random |
| Zr | +666 | -0.067 | -0.067 | Ordering |

**Key finding:** 6/16 dopants (37.5%) show random-like behaviour at synthesis temperatures (800-1200 K), validating the SQS assumption for these dopants. Even for dopants with ordering or clustering tendencies, the SQS ensemble provides a useful baseline for comparison with the ordered supercell.

**Caveat:** The MC analysis uses a nearest-neighbour Ising model that is unreliable for layered materials where pair interactions are long-ranged (see main text Table 3). The classifications above should be interpreted cautiously.

---

## S6. Partial delithiation detailed results

### Table S6.1. Partial delithiation at x = 0.5 (LiCoO2, 19 dopants)

| Dopant | V_ord (V) | V_ord_std | V_dis_mean (V) | V_dis_std | n_SQS |
|--------|-----------|-----------|-----------------|-----------|-------|
| Al | -1.77 | 0.44 | -4.19 | 0.19 | 5 |
| Cr | -1.41 | 0.24 | -4.17 | 0.24 | 5 |
| Cu | -4.28 | 0.13 | -3.47 | 0.69 | 5 |
| Fe | -2.14 | 0.25 | -3.49 | 0.67 | 5 |
| Ga | -1.86 | 0.11 | -3.62 | 0.86 | 5 |
| Ge | -1.37 | 0.14 | -3.68 | 0.82 | 5 |
| Mg | -1.46 | 0.20 | -4.05 | 0.31 | 5 |
| Mn | -4.34 | 0.07 | -3.84 | 0.30 | 5 |
| Mo | -1.52 | 0.18 | -2.57 | 0.09 | 4 |
| Nb | -2.21 | 0.39 | -2.23 | 0.66 | 5 |
| Ni | -1.39 | 0.38 | -4.75 | 0.09 | 5 |
| Ru | -4.01 | 0.06 | -3.77 | 0.93 | 5 |
| Sb | -1.99 | 0.03 | -3.08 | 2.18 | 5 |
| Sc | -3.75 | 0.10 | -3.28 | 1.55 | 5 |
| Sn | -1.94 | 0.06 | -3.18 | 1.42 | 3 |
| Ta | -1.93 | 0.13 | -3.86 | 0.43 | 4 |
| Ti | -3.96 | 0.04 | -4.25 | 0.12 | 5 |
| V | -1.39 | 0.15 | -3.85 | 0.48 | 5 |
| Zr | -1.89 | 0.09 | -3.54 | 0.62 | 5 |

**Overall Spearman rho = -0.32 (p = 0.18, n = 19)**

### Table S6.2. Partial delithiation at x = 0.25 (LiCoO2, 19 dopants)

**Overall Spearman rho = -0.07 (p = 0.79, n = 19)**

Both delithiation depths confirm the voltage danger zone. The weaker correlation at x = 0.25 (rho = -0.07 vs -0.32 at x = 0.5) is consistent with smaller energy differences at shallower delithiation making rankings even more noise-sensitive.

---

## S7. Interaction energy details

Dopant-dopant interaction energies E_int were computed by comparing the energy of a supercell with two dopants at distance r to isolated single-dopant supercells:

E_int(r) = E(2 dopants at distance r) - 2 * E(1 dopant) + E(undoped)

### Table S7.1. Al interaction energy convergence in LiCoO2 and LiMn2O4

| Supercell | In-plane d (A) | LiCoO2 E_int (meV) | LiMn2O4 E_int (meV) |
|-----------|----------------|---------------------|----------------------|
| 3x3x2 / 2x2x1 | 8.6 / 16.1 | -128 | +145 |
| 4x4x4 | 11.5 | +61 | — |
| 5x5x2 | 14.4 | +13 | — |
| 6x6x2 | 17.3 | +4 | — |

In LiMn2O4, the interaction converges immediately at the nearest-neighbour shell (+145 meV). In LiCoO2, the interaction oscillates and does not converge until >17 A in-plane separation, confirming the long-range 2D interaction kernel that drives disorder sensitivity in layered structures.

---

## S8. Disorder-risk predictor: complete observation table

**R = property_scope x sublattice_anisotropy + 0.3 x (n_TM - 1)**

where property_scope = 0 for Ef, 1 for voltage/volume/O-vacancy; sublattice_anisotropy = interlayer/intralayer TM-TM distance ratio.

**Table S8.1. All 27 observations with predictor scores**

| # | Material | Property | Anisotropy | n_TM | R | Predicted | rho | Actual | Correct? |
|---|----------|----------|------------|------|---|-----------|-----|--------|----------|
| 1 | LiCoO2 | Ef | 1.94 | 1 | 0.0 | SAFE | +0.76 | Safe | Yes |
| 2 | LiCoO2 | Voltage | 1.94 | 1 | 1.94 | UNSAFE | -0.25 | Unsafe | Yes |
| 3 | LiCoO2 | Volume | 1.94 | 1 | 1.94 | UNSAFE | +0.09 | Unsafe | Yes |
| 4 | LiNiO2 | Ef | 1.93 | 1 | 0.0 | SAFE | +0.82 | Safe | Yes |
| 5 | LiNiO2 | Voltage | 1.93 | 1 | 1.93 | UNSAFE | -0.06 | Unsafe | Yes |
| 6 | LiNiO2 | Volume | 1.93 | 1 | 1.93 | UNSAFE | +0.54 | Safe | **No** |
| 7 | NMC811 | Ef | 1.93 | 3 | 0.6 | SAFE | +0.52 | Safe | Yes |
| 8 | NMC811 | Voltage | 1.93 | 3 | 2.53 | UNSAFE | +0.09 | Unsafe | Yes |
| 9 | NaCoO2 | Ef | 1.9 | 1 | 0.0 | SAFE | +0.79 | Safe | Yes |
| 10 | NaCoO2 | Voltage | 1.9 | 1 | 1.9 | UNSAFE | +0.23 | Unsafe | Yes |
| 11 | NaCoO2 | Volume | 1.9 | 1 | 1.9 | UNSAFE | -0.01 | Unsafe | Yes |
| 12 | LiMn2O4 | Ef | 1.0 | 1 | 0.0 | SAFE | +1.00 | Safe | Yes |
| 13 | LiMn2O4 | Voltage | 1.0 | 1 | 1.0 | SAFE | +0.95 | Safe | Yes |
| 14 | LiMn2O4 | Volume | 1.0 | 1 | 1.0 | SAFE | +0.84 | Safe | Yes |
| 15 | SrTiO3 | Ef | 1.0 | 1 | 0.0 | SAFE | +1.00 | Safe | Yes |
| 16 | SrTiO3 | Volume | 1.0 | 1 | 1.0 | SAFE | +0.94 | Safe | Yes |
| 17 | CeO2 | Ef | 1.0 | 1 | 0.0 | SAFE | +1.00 | Safe | Yes |
| 18 | CeO2 | Volume | 1.0 | 1 | 1.0 | SAFE | +0.96 | Safe | Yes |
| 19 | CeO2 | O-vacancy | 1.0 | 1 | 1.0 | SAFE | +0.85 | Safe | Yes |
| 20 | LiFePO4 | Ef | 1.21 | 1 | 0.0 | SAFE | +1.00 | Safe | Yes |
| 21 | LiFePO4 | Voltage | 1.21 | 1 | 1.21 | UNSAFE | +0.99 | Safe | **No** |
| 22 | LiFePO4 | Volume | 1.21 | 1 | 1.21 | UNSAFE | +0.79 | Safe | **No** |
| 23 | NASICON | Ef | 1.15 | 1 | 0.0 | SAFE | +0.72 | Safe | Yes |
| 24 | NASICON | Voltage | 1.15 | 1 | 1.15 | UNSAFE | +0.77 | Safe | **No** |
| 25 | NASICON | Volume | 1.15 | 1 | 1.15 | UNSAFE | -0.04 | Unsafe | Yes |
| 26a | LiCoO2 (x=0.5) | Voltage | 1.94 | 1 | 1.94 | UNSAFE | -0.32 | Unsafe | Yes |
| 26b | LiCoO2 (x=0.25) | Voltage | 1.94 | 1 | 1.94 | UNSAFE | -0.07 | Unsafe | Yes |

**Summary:** 23/27 correct (85.2%). **Zero false-safe** predictions. 4 false-unsafe (conservative over-predictions for intermediate-anisotropy structures).

---

## S9. Computational details

### MLIP specifications

| Parameter | Value |
|-----------|-------|
| Model | MACE-MP-0 (medium) |
| Precision | float64 |
| Dispersion correction | None |
| Force convergence | 0.15 eV/A (BFGS), 0.225 eV/A (FIRE fallback) |
| Ionic-only convergence | 0.10 eV/A |
| Maximum optimisation steps | 300 |
| Divergence check | Energy > 50 eV/atom or volume change > 50% |

### Supercell sizes

| Material | Supercell | Total atoms | TM sites | Dopants | Concentration |
|----------|-----------|-------------|----------|---------|---------------|
| LiCoO2 | 4x4x4 | 256 | 64 | 4 | 6.25% |
| LiNiO2 | 4x4x4 | 256 | 64 | 4 | 6.25% |
| NMC811 | 4x4x4 | 256 | 64 | 4 | 6.25% |
| NaCoO2 | — | — | — | — | ~6% |
| LiMn2O4 | 4x4x4 | 256 | 32 | 2 | 6.25% |
| SrTiO3 | 4x4x4 | 320 | 64 | 4 | 6.25% |
| CeO2 | 3x3x3 | 324 | 108 | 7 | 6.48% |
| LiFePO4 | 2x2x2 | 224 | 32 | 2 | 6.25% |
| Na3V2(PO4)3 | 1x1x1 | 126 | 12 | 1 | 8.33% |

### Computational cost

| Material | Dopants | SQS | Relaxations | GPU-hours (est.) |
|----------|---------|-----|-------------|------------------|
| LiCoO2 | 21 | 5 | ~126 | 6 |
| LiNiO2 | 14 | 5 | ~84 | 4 |
| NMC811 | 16 | 5 | ~96 | 5 |
| NaCoO2 | 18 | 5 | ~108 | 5 |
| LiMn2O4 | 12 | 5-8 | ~84 | 4 |
| SrTiO3 | 20 | 8 | ~180 | 5 |
| CeO2 | 20 | 5 | ~120 | 4 |
| LiFePO4 | 18 | 5 | ~108 | 4 |
| NASICON | 18 | 5 | ~108 | 5 |
| **Total** | | | **~1,014** | **~42** |

All simulations ran on NVIDIA A100 GPUs via Google Colab (free and paid tiers). Equivalent DFT cost estimated at ~40,000-80,000 CPU-hours (~1,500x more expensive).
