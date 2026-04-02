# Supplementary Information

**Chemical Disorder Amplifies Screening Errors in Computational Dopant Selection for Layered Cathodes**

Snehal Nair

---

## Table S1. Pipeline threshold sensitivity analysis

To test whether the Jaccard divergence between ordered and disordered pipelines is robust to the choice of gate thresholds, we swept all three gate sizes across +/-20% of the Yao-equivalent proportions (Gate 1: 60-82%, Gate 2: 42-64%, Gate 3: 40-60%), yielding 54 threshold combinations.

| Statistic | Jaccard (final gate) |
|-----------|---------------------|
| Mean      | 0.17                |
| Median    | 0.14                |
| Min       | 0.00                |
| Max       | 0.33                |
| % below 0.33 | 96%             |
| % below 0.20 | 78%             |

The final-gate Jaccard similarity remains below 0.33 in 96% of threshold combinations, confirming that the pipeline divergence is not an artefact of a specific threshold choice.

---

## Table S2. SQS convergence jackknife analysis

Spearman rho was recomputed using all C(5,k) subsets of SQS realisations for k = 1 through 5. Values shown as mean +/- standard deviation across all subsets.

### LiCoO2 voltage ranking (n = 20 dopants)

| k (# SQS) | Mean rho | Std  | # Subsets |
|------------|----------|------|-----------|
| 1          | -0.11    | 0.14 | 5         |
| 2          | -0.15    | 0.13 | 10        |
| 3          | -0.19    | 0.11 | 10        |
| 4          | -0.24    | 0.08 | 5         |
| 5 (full)   | -0.25    | --   | 1         |

Leave-one-out jackknife: rho in [-0.37, -0.14]. No single SQS realisation drives the result.

### LiCoO2 formation energy ranking (n = 20 dopants)

| k (# SQS) | Mean rho | Std  | # Subsets |
|------------|----------|------|-----------|
| 1          | +0.63    | 0.09 | 5         |
| 2          | +0.71    | 0.07 | 10        |
| 3          | +0.75    | 0.05 | 10        |
| 4          | +0.76    | 0.02 | 5         |
| 5 (full)   | +0.76    | --   | 1         |

### Cross-material convergence

| Material   | Property          | rho (k=3)     | rho (k=5)  | LOO range         |
|------------|-------------------|---------------|------------|-------------------|
| LiCoO2     | Voltage           | -0.19 +/- 0.11 | -0.25    | [-0.37, -0.14]    |
| LiCoO2     | Formation energy  | +0.75 +/- 0.05 | +0.76    | [+0.74, +0.80]    |
| LiMn2O4    | Voltage           | +0.95 +/- 0.01 | +0.95    | [+0.95, +0.95]    |
| LiMn2O4    | Formation energy  | +1.00 +/- 0.00 | +1.00    | [+1.00, +1.00]    |
| LiNiO2     | Voltage           | -0.13 +/- 0.18 | -0.06    | [-0.35, +0.08]    |
| LiNiO2     | Formation energy  | +0.76 +/- 0.06 | +0.82    | [+0.70, +0.84]    |
| CeO2       | Formation energy  | +1.00 +/- 0.00 | +1.00    | [+1.00, +1.00]    |
| SrTiO3     | Formation energy  | +1.00 +/- 0.00 | +1.00    | [+1.00, +1.00]    |

For "safe zone" materials (LMO, STO, CeO2), rankings are perfectly stable even at k = 3. For "danger zone" materials (LCO, LNO), the qualitative result (rankings destroyed) is robust at k = 3, though quantitative precision improves at k = 5.

---

## Table S3. Partial delithiation test (x = 0.5)

Ordered voltages for full delithiation (x = 0 -> 1) vs partial delithiation (x = 0 -> 0.5) for all 21 LiCoO2 dopants. Partial delithiation removes 50% of Li atoms using 3 independent random seeds; mean and standard deviation are reported.

| Dopant | V_full (eV) | V_partial (eV) | +/- std  |
|--------|-------------|-----------------|----------|
| Al     | -3.641      | -4.224          | 0.068    |
| Cr     | -3.491      | -4.335          | 0.018    |
| Cu     | -3.608      | -3.916          | 0.016    |
| Fe     | -3.467      | -3.965          | 0.051    |
| Ga     | -3.647      | -3.452          | 0.052    |
| Ge     | -3.552      | -4.217          | 0.099    |
| Ir     | -3.433      | -4.069          | 0.158    |
| Mg     | -3.547      | -3.492          | 0.036    |
| Mn     | -3.541      | -4.415          | 0.079    |
| Mo     | -3.390      | -3.614          | 0.093    |
| Nb     | -3.536      | -3.879          | 0.107    |
| Ni     | -3.458      | -4.047          | 0.048    |
| Rh     | -3.452      | -4.142          | 0.040    |
| Ru     | -3.402      | -3.801          | 0.067    |
| Sb     | -3.565      | -4.321          | 0.046    |
| Sn     | -3.595      | -3.697          | 0.051    |
| Ta     | -3.492      | -4.200          | 0.056    |
| Ti     | -3.390      | -4.098          | 0.038    |
| V      | -3.548      | -3.978          | 0.066    |
| W      | -3.420      | -3.812          | 0.033    |
| Zr     | -3.519      | -4.085          | 0.113    |

**Spearman rho (V_full vs V_partial):** +0.05 (p = 0.82, n = 21)
**Kendall tau:** +0.06 (p = 0.69)

Rankings are completely reshuffled between delithiation endpoints:
- Ga: rank #1 (full) -> #21 (partial)
- Ti: rank #21 (full) -> #8 (partial)
- Mn: rank #9 (full) -> #1 (partial)

The inter-seed standard deviation is small (0.02-0.16 eV), confirming that reshuffling is a systematic effect of the delithiation endpoint, not noise from random Li removal.

---

## Tables S4-S8. Per-dopant property data

### Table S4. LiCoO2 (n = 21 dopants, layered R-3m)

| Dopant | Ef_ord (eV/at) | Ef_dis (eV/at) | V_ord (V) | V_dis (V) | dV_ord (%) | dV_dis (%) |
|--------|----------------|----------------|-----------|-----------|------------|------------|
| Al     | -4.788         | -4.944 +/- 0.066 | -3.641  | -3.480 +/- 0.023 | 0.10    | 4.11 +/- 3.58 |
| Cr     | -4.856         | -5.056 +/- 0.064 | -3.491  | -3.495 +/- 0.056 | 0.44    | 0.34 +/- 0.33 |
| Cu     | -4.737         | -4.894 +/- 0.078 | -3.608  | -3.453 +/- 0.032 | 0.19    | 4.19 +/- 3.93 |
| Fe     | -4.826         | -5.067 +/- 0.017 | -3.467  | -3.531 +/- 0.069 | 0.53    | 4.83 +/- 4.30 |
| Ga     | -4.760         | -4.899 +/- 0.070 | -3.647  | -3.465 +/- 0.043 | 7.17    | 4.29 +/- 2.88 |
| Ge     | -4.817         | -4.982 +/- 0.058 | -3.552  | -3.511 +/- 0.042 | 0.10    | 1.44 +/- 2.66 |
| Ir     | -4.883         | -5.060 +/- 0.016 | -3.433  | -3.452 +/- 0.049 | 5.23    | 2.01 +/- 1.51 |
| Mg     | -4.712         | -4.932 +/- 0.013 | -3.547  | -3.470 +/- 0.095 | 13.35   | 0.98 +/- 0.89 |
| Mn     | -4.861         | -5.051 +/- 0.021 | -3.541  | -3.414 +/- 0.036 | 10.52   | 2.91 +/- 1.88 |
| Mo     | -4.900         | -5.092 +/- 0.027 | -3.390  | -3.519 +/- 0.087 | 8.95    | 5.85 +/- 5.85 |
| Nb     | -4.901         | -5.096 +/- 0.024 | -3.536  | -3.442 +/- 0.074 | 1.39    | 1.66 +/- 2.36 |
| Ni     | -4.784         | -4.998 +/- 0.004 | -3.458  | -3.542 +/- 0.109 | 6.86    | 0.69 +/- 0.71 |
| Rh     | -4.840         | -4.964 +/- 0.069 | -3.452  | -3.496 +/- 0.036 | 0.34    | 2.33 +/- 1.69 |
| Ru     | -4.885         | -5.044 +/- 0.048 | -3.402  | -3.462 +/- 0.047 | 7.22    | 2.11 +/- 2.66 |
| Sb     | -4.785         | -4.892 +/- 0.084 | -3.565  | -3.471 +/- 0.049 | 0.20    | 4.00 +/- 2.85 |
| Sn     | -4.779         | -4.946 +/- 0.067 | -3.595  | -3.486 +/- 0.048 | 0.27    | 1.72 +/- 0.93 |
| Ta     | -4.943         | --               | -3.492  | --               | 0.70    | --            |
| Ti     | -4.881         | -5.082 +/- 0.013 | -3.390  | -3.500 +/- 0.042 | 7.63    | 2.41 +/- 1.77 |
| V      | -4.878         | -5.057 +/- 0.060 | -3.548  | -3.485 +/- 0.062 | 0.46    | 4.83 +/- 3.30 |
| W      | -4.900         | -5.096 +/- 0.094 | -3.420  | -3.468 +/- 0.044 | 9.62    | 7.09 +/- 4.68 |
| Zr     | -4.918         | -4.980 +/- 0.074 | -3.519  | -3.523 +/- 0.046 | 1.17    | 4.77 +/- 5.63 |

Note: Ta has no converged SQS realisations.

### Table S5. LiMn2O4 (n = 12 dopants, spinel Fd-3m)

| Dopant | Ef_ord (eV/at) | Ef_dis (eV/at) | V_ord (V) | V_dis (V) | dV_ord (%) | dV_dis (%) |
|--------|----------------|----------------|-----------|-----------|------------|------------|
| Al     | -6.902         | -6.901 +/- 0.001 | -4.409  | -4.430 +/- 0.010 | 5.04    | 4.85 +/- 0.16 |
| Co     | -6.787         | -6.785 +/- 0.001 | -4.329  | -4.309 +/- 0.006 | 5.16    | 5.03 +/- 0.18 |
| Cr     | -6.935         | -6.934 +/- 0.001 | -4.348  | -4.374 +/- 0.006 | 4.91    | 5.07 +/- 0.25 |
| Cu     | -6.716         | -6.714 +/- 0.000 | -4.359  | -4.363 +/- 0.009 | 4.84    | 4.86 +/- 0.26 |
| Fe     | -6.835         | -6.834 +/- 0.000 | -4.400  | -4.410 +/- 0.003 | 4.80    | 4.87 +/- 0.20 |
| Ga     | -6.794         | -6.792 +/- 0.001 | -4.399  | -4.421 +/- 0.012 | 5.25    | 5.14 +/- 0.23 |
| Mg     | -6.786         | -6.781 +/- 0.002 | -4.527  | -4.541 +/- 0.011 | 3.59    | 3.19 +/- 0.17 |
| Ni     | -6.721         | -6.717 +/- 0.001 | -4.438  | -4.462 +/- 0.007 | 4.76    | 4.94 +/- 0.24 |
| Ti     | -7.046         | -7.044 +/- 0.001 | -4.378  | -4.369 +/- 0.009 | 5.48    | 5.47 +/- 0.25 |
| V      | -6.946         | -6.944 +/- 0.001 | -4.272  | -4.258 +/- 0.005 | 4.57    | 4.65 +/- 0.10 |
| Zn     | -6.691         | -6.685 +/- 0.001 | -4.500  | -4.514 +/- 0.006 | 3.73    | 3.79 +/- 0.07 |
| Zr     | -7.072         | -7.072 +/- 0.001 | -4.352  | -4.340 +/- 0.010 | 5.14    | 5.53 +/- 0.26 |

Note: LMO inter-realisation standard deviations are 10-100x smaller than LCO, consistent with the repulsive NN interaction that drives dopant self-spacing.

### Table S6. LiNiO2 (n = 14 dopants, layered R-3m)

| Dopant | Ef_ord (eV/at) | Ef_dis (eV/at) | V_ord (V) | V_dis (V) | dV_ord (%) | dV_dis (%) |
|--------|----------------|----------------|-----------|-----------|------------|------------|
| Al     | -4.337         | -4.661 +/- 0.045 | -3.461  | -3.439 +/- 0.022 | 11.83   | 4.08 +/- 3.22 |
| Co     | -4.499         | -4.720 +/- 0.047 | -3.257  | -3.499 +/- 0.047 | 7.60    | 1.89 +/- 1.66 |
| Cr     | -4.493         | -4.766 +/- 0.052 | -3.503  | -3.480 +/- 0.085 | 7.57    | 2.40 +/- 2.46 |
| Fe     | -4.412         | -4.729 +/- 0.071 | -3.279  | -3.446 +/- 0.048 | 9.92    | 2.84 +/- 2.44 |
| Ga     | -4.351         | -4.654 +/- 0.026 | -3.618  | -3.452 +/- 0.035 | 1.25    | 0.80 +/- 1.00 |
| Ge     | -4.536         | -4.684 +/- 0.053 | -3.363  | -3.513 +/- 0.077 | 0.52    | 2.71 +/- 2.90 |
| Mg     | -4.399         | -4.543 +/- 0.064 | -3.412  | -3.499 +/- 0.024 | 10.85   | 5.76 +/- 5.97 |
| Mn     | -4.573         | -4.807 +/- 0.018 | -3.304  | -3.485 +/- 0.044 | 11.52   | 5.07 +/- 3.06 |
| Nb     | -4.589         | -4.852 +/- 0.059 | -3.452  | -3.517 +/- 0.077 | 1.06    | 2.31 +/- 2.58 |
| Sn     | -4.330         | -4.651 +/- 0.066 | -3.440  | -3.535 +/- 0.077 | 0.50    | 3.61 +/- 4.73 |
| Ti     | -4.532         | -4.774 +/- 0.050 | -3.424  | -3.529 +/- 0.032 | 1.32    | 1.83 +/- 2.21 |
| V      | -4.425         | -4.715 +/- 0.040 | -3.456  | -3.449 +/- 0.063 | 3.10    | 2.68 +/- 1.90 |
| W      | -4.588         | -4.730 +/- 0.013 | -3.584  | -3.461 +/- 0.037 | 9.71    | 3.31 +/- 3.15 |
| Zr     | -4.572         | -4.812 +/- 0.096 | -3.847  | -3.533 +/- 0.082 | 1.34    | 1.45 +/- 0.80 |

### Table S7. CeO2 (n = 20 dopants, fluorite Fm-3m)

| Dopant | Ef_ord (eV/at) | Ef_dis (eV/at) | dV_ord (%) | dV_dis (%) |
|--------|----------------|----------------|------------|------------|
| Al     | -8.471         | -8.478 +/- 0.002 | 2.33    | 1.29 +/- 0.22 |
| Ba     | -8.366         | -8.374 +/- 0.003 | 6.83    | 7.07 +/- 0.16 |
| Ca     | -8.414         | -8.415 +/- 0.003 | 3.46    | 3.43 +/- 0.08 |
| Co     | -8.348         | -8.361 +/- 0.002 | 3.61    | 2.17 +/- 0.41 |
| Cr     | -8.518         | -8.528 +/- 0.002 | 1.75    | 1.07 +/- 0.11 |
| Cu     | -8.285         | -8.288 +/- 0.003 | 4.22    | 3.57 +/- 0.16 |
| Fe     | -8.424         | -8.429 +/- 0.002 | 1.97    | 1.50 +/- 0.19 |
| Gd     | -8.952         | -8.951 +/- 0.000 | 3.01    | 3.10 +/- 0.00 |
| Hf     | -8.857         | -8.856 +/- 0.000 | 0.76    | 0.03 +/- 0.00 |
| La     | -8.639         | -8.638 +/- 0.000 | 4.26    | 4.40 +/- 0.01 |
| Mg     | -8.331         | -8.353 +/- 0.003 | 2.30    | 1.63 +/- 0.39 |
| Mn     | -8.489         | -8.505 +/- 0.002 | 1.87    | 1.53 +/- 0.18 |
| Nd     | -8.628         | -8.627 +/- 0.000 | 3.67    | 3.78 +/- 0.03 |
| Ni     | -8.243         | -8.259 +/- 0.002 | 2.90    | 1.85 +/- 0.22 |
| Pr     | -8.625         | -8.623 +/- 0.000 | 3.95    | 4.04 +/- 0.09 |
| Sm     | -8.632         | -8.631 +/- 0.000 | 3.28    | 3.38 +/- 0.00 |
| Sr     | -8.393         | -8.393 +/- 0.002 | 4.88    | 5.05 +/- 0.12 |
| Ti     | -8.713         | -8.712 +/- 0.001 | 2.02    | 1.12 +/- 0.27 |
| Y      | -8.695         | -8.694 +/- 0.000 | 2.66    | 2.69 +/- 0.05 |
| Zr     | -8.798         | -8.797 +/- 0.000 | 1.05    | 0.08 +/- 0.01 |

### Table S8. SrTiO3 (n = 20 dopants, perovskite Pm-3m)

| Dopant | Ef_ord (eV/at) | Ef_dis (eV/at) | dV_ord (%) | dV_dis (%) |
|--------|----------------|----------------|------------|------------|
| Al     | -7.910         | -7.909 +/- 0.000 | 0.81    | 0.02 +/- 0.01 |
| Co     | -7.833         | -7.833 +/- 0.000 | 0.55    | 0.01 +/- 0.00 |
| Cr     | -7.936         | -7.936 +/- 0.000 | 0.01    | 0.01 +/- 0.00 |
| Cu     | -7.782         | -7.783 +/- 0.000 | 0.01    | 0.01 +/- 0.00 |
| Fe     | -7.873         | -7.873 +/- 0.000 | 0.00    | 0.00 +/- 0.00 |
| Ga     | -7.840         | -7.840 +/- 0.000 | 0.01    | 0.00 +/- 0.00 |
| La     | -7.910         | -7.911 +/- 0.006 | 5.43    | 5.11 +/- 0.24 |
| Mg     | -7.815         | -7.813 +/- 0.000 | 0.80    | 0.03 +/- 0.00 |
| Mn     | -7.917         | -7.917 +/- 0.000 | 0.54    | 0.01 +/- 0.00 |
| Mo     | -7.931         | -7.930 +/- 0.000 | 0.71    | 0.03 +/- 0.00 |
| Nb     | -8.047         | -8.046 +/- 0.000 | 0.97    | 0.03 +/- 0.00 |
| Ni     | -7.776         | -7.776 +/- 0.000 | 0.00    | 0.01 +/- 0.00 |
| Sc     | -7.980         | -7.978 +/- 0.000 | 1.17    | 0.05 +/- 0.01 |
| Sn     | -7.860         | -7.859 +/- 0.000 | 0.97    | 0.04 +/- 0.00 |
| Ta     | -8.102         | -8.101 +/- 0.000 | 0.85    | 0.03 +/- 0.00 |
| V      | -7.956         | -7.956 +/- 0.000 | 0.00    | 0.00 +/- 0.00 |
| W      | -7.957         | -7.955 +/- 0.000 | 0.86    | 0.03 +/- 0.00 |
| Y      | -7.964         | -7.967 +/- 0.001 | 3.51    | 3.37 +/- 0.16 |
| Zn     | -7.756         | -7.755 +/- 0.000 | 0.81    | 0.02 +/- 0.00 |
| Zr     | -8.039         | -8.038 +/- 0.000 | 1.99    | 1.72 +/- 0.17 |

---

## Table S9. NMC811 dopant screening (n = 16, layered R-3m)

LiNi₀.₈Mn₀.₁Co₀.₁O₂ (NMC811) screened with MACE-MP-0 using 4×4×4 supercell (256 atoms). Dopants substituted at Ni sites (~2% concentration). 5 SQS realisations per dopant.

**Voltage ρ (ordered vs disordered) = +0.09 (p = 0.75)**
**Formation energy ρ = +0.52**

| Dopant | V_ord (V) | V_dis (V) | V_dis_std (V) | Ef_ord (eV/at) | Ef_dis (eV/at) |
|--------|-----------|-----------|---------------|----------------|----------------|
| Al     | -4.240    | -4.197    | 0.120         | -4.720         | -4.713         |
| Ti     | -4.305    | -4.256    | 0.033         | -4.741         | -4.741         |
| Mg     | -4.325    | -4.313    | 0.038         | -4.708         | -4.714         |
| Fe     | -4.190    | -4.262    | 0.070         | -4.732         | -4.734         |
| Cr     | -4.345    | -4.234    | 0.040         | -4.734         | -4.728         |
| Ga     | -4.305    | -4.292    | 0.015         | -4.720         | -4.719         |
| Zr     | -4.333    | -4.272    | 0.068         | -4.755         | -4.747         |
| Nb     | -4.295    | -4.273    | 0.042         | -4.738         | -4.753         |
| Ta     | -4.135    | -4.315    | 0.025         | -4.719         | -4.754         |
| W      | -4.289    | -4.217    | 0.027         | -4.752         | -4.747         |
| V      | -4.341    | -4.256    | 0.044         | -4.741         | -4.733         |
| Mo     | -4.263    | -4.162    | 0.079         | -4.756         | -4.734         |
| Sn     | -4.188    | -4.259    | 0.030         | -4.716         | -4.722         |
| Sb     | -4.357    | -4.318    | 0.051         | -4.726         | -4.731         |
| Ge     | -4.377    | -4.230    | 0.026         | -4.728         | -4.718         |
| Cu     | -4.274    | -4.231    | 0.050         | -4.720         | -4.710         |

Notable rank-swaps: Ta (ordered rank 16 → disordered rank 2, +180 mV), Ge (rank 1 → rank 13, -147 mV), Cr (rank 3 → rank 12, -111 mV).

---

## Table S10. CHGNet cross-validation (LiCoO2, n = 6)

CHGNet v0.3.0 applied to 6 LiCoO₂ dopants to test whether disorder effect is MLIP-specific.

**CHGNet voltage ρ (ordered vs disordered) = -0.26 (p = 0.62)**
**CHGNet formation energy ρ = +0.66 (p = 0.16)**
**Cross-MLIP ordered voltage ρ (CHGNet vs MACE) = -0.71 (p = 0.11)**

| Dopant | CHGNet V_ord (V) | CHGNet V_dis (V) | CHGNet V_dis_std (V) | CHGNet Ef_ord (eV/at) | MACE V_ord (V) |
|--------|-----------------|-----------------|---------------------|----------------------|----------------|
| Al     | -2.736          | -2.848          | 0.242               | -5.177               | -3.641         |
| Cr     | -3.728          | -2.897          | 0.297               | -5.371               | -3.491         |
| Ga     | -2.314          | -2.940          | 0.181               | -5.181               | -3.647         |
| Ge     | -2.543          | -3.167          | 0.306               | -5.254               | -3.552         |
| Ni     | -3.195          | -2.948          | 0.164               | -5.242               | -3.458         |
| Ti     | -2.837          | -2.672          | 0.212               | -5.277               | -3.390         |

Both MLIPs independently show near-zero or negative voltage ρ, confirming that the disorder effect is structural (R-3m layered geometry) rather than MLIP-specific. The cross-MLIP disagreement on ordered voltage rankings (ρ = -0.71) further cautions against over-interpreting absolute voltage values from any single potential.

---

## Table S11. Monte Carlo clustering analysis (LiCoO2, 16 dopants)

Lattice Monte Carlo on the LiCoO₂ TM sublattice (4×4×4 supercell, 64 Co sites, 4 dopants at 6.25% concentration, 20,000 MC steps per temperature). Pair interaction energy E_nn computed from MACE-MP-0 single-point evaluations. Warren-Cowley α₁ < 0 indicates ordering tendency; α₁ > 0 indicates clustering tendency.

| Dopant | E_nn (meV) | α₁ (800 K) | α₁ (1000 K) | Tendency |
|--------|-----------|------------|-------------|----------|
| Al     | +60.9     | -0.044     | -0.042      | random   |
| Ti     | +32.4     | -0.033     | -0.021      | random   |
| Mg     | -36.5     | +0.016     | +0.003      | random   |
| Zr     | +665.6    | -0.067     | -0.067      | ORDERING |
| Nb     | -234.9    | +0.363     | +0.343      | CLUSTERING |
| Fe     | -82.7     | +0.079     | +0.047      | random   |
| Cr     | +418.0    | -0.067     | -0.067      | ORDERING |
| Ga     | -208.8    | +0.339     | +0.301      | CLUSTERING |
| Ni     | -146.4    | +0.233     | +0.165      | CLUSTERING |
| Mn     | +238.2    | -0.066     | -0.062      | ORDERING |
| V      | +392.1    | -0.067     | -0.067      | ORDERING |
| Cu     | -162.5    | +0.295     | +0.187      | CLUSTERING |
| W      | +94.0     | -0.054     | -0.045      | random   |
| Mo     | +354.3    | -0.067     | -0.065      | ORDERING |
| Ge     | +12.0     | -0.027     | -0.026      | random   |
| Sn     | +383.7    | -0.067     | -0.066      | ORDERING |

**Summary at 1000 K:** 6/16 random (38%), 6/16 ordering (38%), 4/16 clustering (25%).

E_nn sign determines behaviour: positive E_nn → repulsive → self-ordering; negative E_nn → attractive → clustering. The α₁ magnitude increases at lower temperatures as thermal fluctuations decrease.

**Important caveat:** The NN Ising model is a simplification. As shown in Table 5 of the main text, the NN interaction energy for Al in LiCoO₂ varies from −128 meV (3×3×2) to +4 meV (6×6×2) depending on supercell size, indicating that interactions in layered structures are not NN-dominated. The NN E_nn values used here (computed on a 4×4×4 supercell) are therefore not fully converged, and the ordering/clustering classifications should be treated as exploratory rather than definitive for the layered system. The key finding is the contrast with spinel LiMn₂O₄, where NN interactions are well-converged (+145 meV across all tested supercell sizes) and an NN Ising model is reliable.
