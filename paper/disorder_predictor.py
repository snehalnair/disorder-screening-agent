#!/usr/bin/env python3
"""
Disorder-risk predictor: when is ordered screening safe?
=========================================================

Builds a quantitative disorder-risk score from cheap structural descriptors
that predicts whether ordered-cell screening preserves dopant rankings (safe)
or destroys them (unsafe).

Risk score:
    R = property_scope × sublattice_anisotropy + 0.3 × (n_TM_species − 1)

Descriptors (all computable from crystal structure alone, no simulation):
  1. sublattice_anisotropy  – ratio of interlayer to intralayer TM-TM distance
                              (~1.9 for layered R-3m, 1.0 for 3D structures)
  2. property_scope         – 0 for local properties (Ef), 1 for global (V, ΔV)
  3. n_TM_species           – number of distinct TM species on the dopant sublattice

Decision rule: R > 1.0 → unsafe (disorder-aware screening required)

Usage:
    python paper/disorder_predictor.py

Outputs:
    paper/disorder_predictor_results.json
    paper/figures/fig7_predictor.png / .pdf
"""

import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Crystal structure descriptors ────────────────────────────────────────
# Computed from standard crystallographic data (Materials Project / ICSD)
STRUCTURE_DATA = {
    "LiCoO2": {
        "structure": "layered R-3m",
        "a": 2.816, "c": 14.054,
        "nn_tm_dist": 2.816,       # in-plane Co-Co
        "interlayer_dist": 5.458,   # sqrt(a² + (c/3)²)
        "nn_count": 6,              # triangular lattice
        "anisotropy": 1.94,         # interlayer/intralayer
        "n_tm": 1,
    },
    "LiNiO2": {
        "structure": "layered R-3m",
        "a": 2.878, "c": 14.19,
        "nn_tm_dist": 2.878,
        "interlayer_dist": 5.554,
        "nn_count": 6,
        "anisotropy": 1.93,
        "n_tm": 1,
    },
    "NMC811": {
        "structure": "layered R-3m",
        "a": 2.870, "c": 14.23,
        "nn_tm_dist": 2.870,
        "interlayer_dist": 5.537,
        "nn_count": 6,
        "anisotropy": 1.93,
        "n_tm": 3,  # Ni, Mn, Co
    },
    "LiMn2O4": {
        "structure": "spinel Fd-3m",
        "a": 8.248,
        "nn_tm_dist": 2.916,       # Mn-Mn on pyrochlore sublattice
        "interlayer_dist": 2.916,   # isotropic 3D
        "nn_count": 6,
        "anisotropy": 1.00,
        "n_tm": 1,
    },
    "SrTiO3": {
        "structure": "perovskite Pm-3m",
        "a": 3.905,
        "nn_tm_dist": 3.905,       # Ti-Ti on simple cubic
        "interlayer_dist": 3.905,
        "nn_count": 6,
        "anisotropy": 1.00,
        "n_tm": 1,
    },
    "CeO2": {
        "structure": "fluorite Fm-3m",
        "a": 5.411,
        "nn_tm_dist": 3.826,       # Ce-Ce on FCC
        "interlayer_dist": 3.826,
        "nn_count": 12,
        "anisotropy": 1.00,
        "n_tm": 1,
    },
    "LiFePO4": {
        "structure": "olivine Pnma",
        "a": 10.33, "b": 6.01, "c": 4.69,
        "nn_tm_dist": 3.87,        # Fe-Fe NN
        "interlayer_dist": 4.69,    # next shell
        "nn_count": 4,
        "anisotropy": 1.21,
        "n_tm": 1,
    },
    "NASICON": {
        "structure": "NASICON R-3c",
        "a": 8.73, "c": 21.85,
        "nn_tm_dist": 4.42,        # V-V NN
        "interlayer_dist": 5.10,    # next shell
        "nn_count": 6,
        "anisotropy": 1.15,
        "n_tm": 1,
    },
    "NaCoO2": {
        "structure": "layered R-3m",
        "a": 2.89, "c": 15.61,
        "nn_tm_dist": 2.85,        # Co-Co in-plane
        "interlayer_dist": 5.42,
        "nn_count": 6,
        "anisotropy": 1.90,
        "n_tm": 1,
    },
}

# ── Dataset ──────────────────────────────────────────────────────────────
# (material, property, property_scope, rho)
OBSERVATIONS = [
    # Original 5 materials
    ("LiCoO2",  "Ef",      0,  0.76),
    ("LiCoO2",  "voltage", 1, -0.25),
    ("LiCoO2",  "dV",      1,  0.09),
    ("LiNiO2",  "Ef",      0,  0.82),
    ("LiNiO2",  "voltage", 1, -0.06),
    ("LiNiO2",  "dV",      1,  0.54),
    ("NMC811",  "Ef",      0,  0.52),
    ("NMC811",  "voltage", 1,  0.09),
    ("LiMn2O4", "Ef",      0,  1.00),
    ("LiMn2O4", "voltage", 1,  0.95),
    ("LiMn2O4", "dV",      1,  0.84),
    ("SrTiO3",  "Ef",      0,  1.00),
    ("SrTiO3",  "dV",      1,  0.94),
    ("CeO2",    "Ef",      0,  1.00),
    ("CeO2",    "dV",      1,  0.96),
    ("CeO2",    "E_Ovac",  0,  0.85),
    # Out-of-sample: LiFePO4 (olivine)
    ("LiFePO4", "Ef",      0,  1.00),
    ("LiFePO4", "voltage", 1,  0.99),
    ("LiFePO4", "dV",      1,  0.79),
    # Out-of-sample: NASICON Na3V2(PO4)3
    ("NASICON",  "Ef",      0,  0.72),
    ("NASICON",  "voltage", 1,  0.77),
    ("NASICON",  "dV",      1, -0.04),
    # Out-of-sample: NaCoO2
    ("NaCoO2",  "Ef",      0,  0.79),
    ("NaCoO2",  "voltage", 1,  0.23),
    ("NaCoO2",  "dV",      1, -0.01),
    # Partial delithiation (LiCoO2)
    ("LiCoO2",  "V_x05",   1, -0.32),
    ("LiCoO2",  "V_x025",  1, -0.07),
]

RHO_THRESHOLD = 0.50
RISK_THRESHOLD = 1.0

PROP_LABELS = {
    "Ef": "E$_f$",
    "voltage": "V",
    "dV": "$\\Delta$V",
    "E_Ovac": "E$_{\\mathrm{Ovac}}$",
    "V_x05": "V(x=0.5)",
    "V_x025": "V(x=0.25)",
}

MAT_LABELS = {
    "LiCoO2": "LiCoO$_2$",
    "LiNiO2": "LiNiO$_2$",
    "NMC811": "NMC811",
    "LiMn2O4": "LiMn$_2$O$_4$",
    "SrTiO3": "SrTiO$_3$",
    "CeO2": "CeO$_2$",
    "LiFePO4": "LiFePO$_4$",
    "NASICON": "NASICON",
    "NaCoO2": "NaCoO$_2$",
}


def compute_risk(material, property_scope):
    """Compute disorder risk score from structural descriptors."""
    m = STRUCTURE_DATA[material]
    aniso = m["anisotropy"]
    n_tm = m["n_tm"]
    return property_scope * aniso + 0.3 * (n_tm - 1)


def build_dataset():
    """Build feature matrix and labels."""
    names, risks, rhos, labels, scopes, anisos = [], [], [], [], [], []
    for mat, prop, scope, rho in OBSERVATIONS:
        risk = compute_risk(mat, scope)
        names.append(f"{MAT_LABELS[mat]} {PROP_LABELS[prop]}")
        risks.append(risk)
        rhos.append(rho)
        labels.append(1 if rho >= RHO_THRESHOLD else 0)
        scopes.append(scope)
        anisos.append(STRUCTURE_DATA[mat]["anisotropy"])
    return (np.array(risks), np.array(rhos), np.array(labels),
            names, np.array(scopes), np.array(anisos))


def evaluate(risks, labels, threshold=RISK_THRESHOLD):
    """Evaluate decision rule: risk > threshold → unsafe."""
    preds = (risks <= threshold).astype(int)  # 1=safe, 0=unsafe
    correct = (preds == labels)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "accuracy": float(correct.mean()),
        "n_correct": int(correct.sum()),
        "n_total": len(labels),
        "predictions": preds.tolist(),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision_safe": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall_safe": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


def plot_predictor(risks, rhos, labels, names, scopes, anisos, metrics, savepath):
    """
    Figure 7: Two-panel predictor visualisation.
    (a) Risk score vs actual ρ — shows clear separation.
    (b) Decision boundary with structural interpretation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [3, 2], "wspace": 0.3})

    # ── Panel (a): Risk score vs actual ρ ──
    ax = axes[0]

    preds = np.array(metrics["predictions"])

    # Pre-compute label offsets to avoid overlap
    # Group by (R_rounded, rho_rounded) and stagger
    label_offsets = {}
    for i, name in enumerate(names):
        label_offsets[name] = None  # default: auto-place

    # Manual offsets for known clusters (dx, dy in fontsize units)
    # Only label a curated subset — skip Ef points at R=0 to avoid clutter
    labeled_set = {
        # Unsafe region (right side, R≈1.9–2.5) — spread vertically
        "LiCoO$_2$ V":            (0.8,  1.5),    # ρ=-0.25
        "LiCoO$_2$ V(x=0.5)":    (0.8, -1.5),    # ρ=-0.32
        "LiCoO$_2$ V(x=0.25)":   (-6.0,  1.2),   # ρ=-0.07
        "NaCoO$_2$ V":            (-6.0, -0.5),   # ρ=0.23
        "NMC811 V":                (0.8,  0.8),    # ρ=0.09, R=2.53
        "NASICON $\\Delta$V":     (0.8, -1.2),    # ρ=-0.04, R=1.15
        # Safe region — only label distinctive points
        "NASICON V":               (0.8,  0.8),    # ρ=0.77, R=1.15
        "LiFePO$_4$ V":           (0.8,  0.8),    # ρ=0.99, R=1.21
        "LiMn$_2$O$_4$ V":       (-6.0, -0.5),   # ρ=0.95, R=1.0
    }

    for i in range(len(risks)):
        correct = preds[i] == labels[i]
        if labels[i] == 0:
            color = '#E74C3C'  # actually unsafe
            marker = 's'
        else:
            color = '#27AE60'  # actually safe
            marker = 'o'

        edgecolor = 'black' if not correct else (color if correct else 'black')
        lw = 2.5 if not correct else 1.0

        ax.scatter(risks[i], rhos[i], c=color, marker=marker, s=110,
                  edgecolors=edgecolor, linewidths=lw, zorder=5)

        # Only label curated points to avoid clutter
        if names[i] in labeled_set:
            dx, dy = labeled_set[names[i]]
            ax.annotate(names[i], (risks[i], rhos[i]),
                       xytext=(dx, dy), textcoords='offset fontsize',
                       fontsize=6.5, color='#444444', ha='left', va='center',
                       arrowprops=dict(arrowstyle='-', color='#AAAAAA',
                                       lw=0.4, shrinkB=3))

    # Decision boundary
    ax.axvline(RISK_THRESHOLD, color='#7F8C8D', ls='--', lw=2, alpha=0.8)
    ax.axhline(RHO_THRESHOLD, color='#7F8C8D', ls=':', lw=1.5, alpha=0.5)

    # Shading
    ax.axvspan(RISK_THRESHOLD, 3.0, alpha=0.06, color='red')
    ax.axvspan(-0.1, RISK_THRESHOLD, alpha=0.06, color='green')

    # Zone labels — placed at bottom corners, clear of data
    ax.text(0.15, -0.45, 'PREDICTED\nSAFE', fontsize=9, color='#27AE60',
            ha='center', fontweight='bold', alpha=0.6)
    ax.text(2.4, -0.45, 'PREDICTED\nUNSAFE', fontsize=9, color='#E74C3C',
            ha='center', fontweight='bold', alpha=0.6)

    ax.set_xlabel('Disorder risk score $R$', fontsize=12)
    ax.set_ylabel('Actual Spearman $\\rho$ (ordered vs disordered)', fontsize=11)
    ax.set_xlim(-0.15, 2.8)
    ax.set_ylim(-0.50, 1.15)
    ax.set_title('(a) Risk score vs ranking preservation', fontsize=12)

    # Accuracy annotation — bottom-left, clear of data points
    acc_text = (f"Accuracy: {metrics['accuracy']:.1%}\n"
                f"False safe: {metrics['FP']}\n"
                f"False unsafe: {metrics['FN']}")
    ax.text(0.02, 0.02, acc_text, transform=ax.transAxes,
           fontsize=8, va='bottom', ha='left',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='#CCCCCC', alpha=0.9))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
               markersize=10, label='Actually safe ($\\rho \\geq 0.50$)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='Actually unsafe ($\\rho < 0.50$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
               markeredgecolor='black', markeredgewidth=2, markersize=10,
               label='Misclassified'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8.5,
             framealpha=0.9)

    # ── Panel (b): Risk score formula and interpretation ──
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) Risk score decomposition', fontsize=12)

    # Formula box
    formula_box = FancyBboxPatch((0.5, 8.2), 9, 1.3, boxstyle="round,pad=0.3",
                                 facecolor='#EBF5FB', edgecolor='#2980B9', linewidth=2)
    ax2.add_patch(formula_box)
    ax2.text(5, 8.85,
            '$R = \\mathrm{scope} \\times \\mathrm{anisotropy} + 0.3 \\times (n_{\\mathrm{TM}} - 1)$',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='#2C3E50')

    # Descriptor explanations
    y_pos = 7.2
    descriptors = [
        ("Property scope", "0 = local (E$_f$)\n1 = global (V, $\\Delta$V)",
         "Formation energy depends on\nlocal bonding; voltage depends\non full delithiation response"),
        ("Sublattice\nanisotropy",
         "Layered R$\\bar{3}$m: ~1.9\nSpinel/perov./fluorite: 1.0",
         "Interlayer/intralayer TM-TM\ndistance ratio; captures 2D\ninteraction propagation"),
        ("TM species", "1 = single-TM\n3 = mixed (NMC811)",
         "Mixed-TM creates heterogeneous\nlocal environments that amplify\ndisorder sensitivity"),
    ]

    for name, values, explanation in descriptors:
        # Name
        ax2.text(0.5, y_pos, name, fontsize=10, fontweight='bold',
                color='#2C3E50', va='top')
        # Values
        ax2.text(3.5, y_pos, values, fontsize=8.5, color='#555555', va='top')
        # Explanation
        ax2.text(6.5, y_pos, explanation, fontsize=7.5, color='#777777',
                va='top', style='italic')
        y_pos -= 2.0

    # Threshold rule
    rule_box = FancyBboxPatch((0.5, 0.5), 9, 1.2, boxstyle="round,pad=0.2",
                              facecolor='#FDEBD0', edgecolor='#E67E22', linewidth=1.5)
    ax2.add_patch(rule_box)
    ax2.text(5, 1.1,
            'Decision: $R > 1.0$ $\\Rightarrow$ disorder-aware screening required',
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color='#D35400')

    plt.tight_layout()
    plt.savefig(savepath.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(savepath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath.with_suffix('.png')}")


def main():
    print("=" * 60)
    print("DISORDER-RISK PREDICTOR (continuous risk score)")
    print("=" * 60)

    risks, rhos, labels, names, scopes, anisos = build_dataset()

    n_safe = labels.sum()
    n_unsafe = len(labels) - n_safe
    print(f"\nDataset: {len(labels)} observations")
    print(f"  Safe (ρ >= {RHO_THRESHOLD}): {n_safe}")
    print(f"  Unsafe (ρ < {RHO_THRESHOLD}): {n_unsafe}")

    print(f"\n--- Risk scores ---")
    for i, (name, risk, rho) in enumerate(zip(names, risks, rhos)):
        safe = "SAFE" if labels[i] else "UNSAFE"
        print(f"  {name:25s}  R = {risk:.2f}  ρ = {rho:+.2f}  [{safe}]")

    print(f"\n--- Evaluation (threshold R > {RISK_THRESHOLD}) ---")
    metrics = evaluate(risks, labels, RISK_THRESHOLD)
    print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['n_correct']}/{metrics['n_total']})")
    print(f"  True safe:    {metrics['TP']}")
    print(f"  True unsafe:  {metrics['TN']}")
    print(f"  False safe:   {metrics['FP']}  (dangerous)")
    print(f"  False unsafe: {metrics['FN']}  (conservative)")
    print(f"  Precision:    {metrics['precision_safe']:.3f}")
    print(f"  Recall:       {metrics['recall_safe']:.3f}")
    print(f"  Specificity:  {metrics['specificity']:.3f}")

    # Show misclassifications
    preds = np.array(metrics["predictions"])
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            direction = "SAFE→actually unsafe" if preds[i] == 1 else "UNSAFE→actually safe"
            print(f"  MISS: {names[i]} (R={risks[i]:.2f}, ρ={rhos[i]:.2f}) — {direction}")

    # Correlation between risk and rho
    r_corr, r_p = stats.spearmanr(risks, rhos)
    print(f"\n  Risk-ρ Spearman correlation: {r_corr:.3f} (p = {r_p:.4f})")

    # Threshold sensitivity
    print(f"\n--- Threshold sensitivity ---")
    for t in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8]:
        m = evaluate(risks, labels, t)
        print(f"  R > {t:.1f}: acc={m['accuracy']:.3f}, "
              f"FP={m['FP']}, FN={m['FN']}")

    # Save results
    results = {
        "risk_formula": "R = property_scope × sublattice_anisotropy + 0.3 × (n_TM - 1)",
        "risk_threshold": RISK_THRESHOLD,
        "rho_threshold": RHO_THRESHOLD,
        "n_observations": len(labels),
        "accuracy": metrics["accuracy"],
        "specificity": metrics["specificity"],
        "precision_safe": metrics["precision_safe"],
        "recall_safe": metrics["recall_safe"],
        "false_safe": metrics["FP"],
        "false_unsafe": metrics["FN"],
        "risk_rho_correlation": float(r_corr),
        "risk_rho_p_value": float(r_p),
        "confusion": {
            "TP": metrics["TP"], "TN": metrics["TN"],
            "FP": metrics["FP"], "FN": metrics["FN"],
        },
        "structural_descriptors": {
            mat: {
                "anisotropy": data["anisotropy"],
                "nn_tm_dist": data["nn_tm_dist"],
                "nn_count": data["nn_count"],
                "n_tm": data["n_tm"],
            } for mat, data in STRUCTURE_DATA.items()
        },
        "observations": [],
    }
    for i in range(len(labels)):
        mat = OBSERVATIONS[i][0]
        results["observations"].append({
            "name": names[i].replace("$", ""),
            "material": mat,
            "property": OBSERVATIONS[i][1],
            "risk_score": float(risks[i]),
            "rho": float(rhos[i]),
            "actual": "safe" if labels[i] else "unsafe",
            "predicted": "safe" if preds[i] else "unsafe",
            "correct": bool(preds[i] == labels[i]),
        })

    out_path = SCRIPT_DIR / "disorder_predictor_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Generate figure
    print("\nGenerating Figure 7...")
    plot_predictor(risks, rhos, labels, names, scopes, anisos, metrics,
                  FIG_DIR / "fig7_predictor")

    print("\nDone.")


if __name__ == "__main__":
    main()
