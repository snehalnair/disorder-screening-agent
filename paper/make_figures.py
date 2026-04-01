#!/usr/bin/env python3
"""
Publication figures for the Disorder Gap paper.
================================================
Generates 5 figures from checkpoint data:

  Fig 1: Spearman ρ heatmap (material × property)
  Fig 2: Pipeline divergence funnel (ordered vs disordered Jaccard)
  Fig 3: Voltage distributions — ordered value vs SQS spread (LCO)
  Fig 4: Dopant–dopant interaction energy decay (LCO vs LMO)
  Fig 5: Phase-space map — when is ordered screening safe?

Usage:
    python paper/make_figures.py
"""

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIRS = {
    "LiCoO₂\n(layered)":  PROJECT_DIR / "lco",
    "LiNiO₂\n(layered)":  PROJECT_DIR / "lno",
    "LiMn₂O₄\n(spinel)":  PROJECT_DIR / "lmo",
    "SrTiO₃\n(perovskite)": PROJECT_DIR / "sto",
    "CeO₂\n(fluorite)":   PROJECT_DIR / "ceo2",
}

PROPERTIES = ["formation_energy", "voltage", "volume_change"]
PROP_LABELS = {
    "formation_energy": "Formation\nenergy",
    "voltage": "Voltage",
    "volume_change": "Volume\nchange",
}

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

CB_BLUE = "#0072B2"
CB_AMBER = "#E69F00"
CB_GREEN = "#009E73"
CB_PINK = "#CC79A7"
CB_SKY = "#56B4E9"
CB_VERMILION = "#D55E00"
CB_GREY = "#999999"


# ── Data loading ───────────────────────────────────────────────────────────

def load_material(ckpt_dir):
    """Load all dopant checkpoints from a directory.
    Returns list of dicts with keys: dopant, ordered, sqs_values, sqs_mean, sqs_std
    """
    results = []
    if not ckpt_dir.exists():
        return results
    for f in sorted(ckpt_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        dopant = f.stem.split("_")[-1]
        ordered = data["ordered"]
        sqs = data.get("sqs_results", [])
        if len(sqs) < 2:
            continue
        sqs_vals = {prop: [s[prop] for s in sqs if prop in s] for prop in PROPERTIES}
        sqs_mean = {prop: np.mean(v) if v else np.nan for prop, v in sqs_vals.items()}
        sqs_std = {prop: np.std(v, ddof=1) if len(v) > 1 else np.nan for prop, v in sqs_vals.items()}
        results.append({
            "dopant": dopant,
            "ordered": ordered,
            "sqs_values": sqs_vals,
            "sqs_mean": sqs_mean,
            "sqs_std": sqs_std,
        })
    return results


def compute_spearman(results, prop):
    """Compute Spearman ρ between ordered and SQS-mean for a property."""
    ord_vals, dis_vals = [], []
    for r in results:
        o = r["ordered"].get(prop)
        d = r["sqs_mean"].get(prop)
        if o is not None and d is not None and not np.isnan(d):
            ord_vals.append(o)
            dis_vals.append(d)
    if len(ord_vals) < 4:
        return np.nan, np.nan, len(ord_vals)
    rho, p = stats.spearmanr(ord_vals, dis_vals)
    return rho, p, len(ord_vals)


def bootstrap_spearman(results, prop, n_boot=10000):
    """Bootstrap 95% CI for Spearman ρ."""
    ord_vals, dis_vals = [], []
    for r in results:
        o = r["ordered"].get(prop)
        d = r["sqs_mean"].get(prop)
        if o is not None and d is not None and not np.isnan(d):
            ord_vals.append(o)
            dis_vals.append(d)
    if len(ord_vals) < 4:
        return np.nan, np.nan
    ord_arr, dis_arr = np.array(ord_vals), np.array(dis_vals)
    rng = np.random.default_rng(42)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ord_arr), size=len(ord_arr))
        r, _ = stats.spearmanr(ord_arr[idx], dis_arr[idx])
        rhos.append(r)
    return np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)


# ── Figure 1: Spearman ρ heatmap ──────────────────────────────────────────

def fig1_heatmap():
    """Material × property Spearman ρ heatmap."""
    mat_names = list(CHECKPOINT_DIRS.keys())
    prop_names = PROPERTIES

    rho_matrix = np.full((len(mat_names), len(prop_names)), np.nan)
    ci_matrix = {}
    n_matrix = np.zeros((len(mat_names), len(prop_names)), dtype=int)

    for i, (mat_label, ckpt_dir) in enumerate(CHECKPOINT_DIRS.items()):
        results = load_material(ckpt_dir)
        for j, prop in enumerate(prop_names):
            rho, p, n = compute_spearman(results, prop)
            rho_matrix[i, j] = rho
            n_matrix[i, j] = n
            lo, hi = bootstrap_spearman(results, prop)
            ci_matrix[(i, j)] = (lo, hi)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(rho_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(len(mat_names)):
        for j in range(len(prop_names)):
            val = rho_matrix[i, j]
            n = n_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="grey")
                continue
            lo, hi = ci_matrix.get((i, j), (np.nan, np.nan))
            # Bold if CI excludes zero
            if not np.isnan(lo) and not np.isnan(hi) and (lo > 0 or hi < 0):
                weight = "bold"
            else:
                weight = "normal"
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=10, fontweight=weight, color=color)
            # CI below
            if not np.isnan(lo):
                ax.text(j, i + 0.30, f"[{lo:+.2f}, {hi:+.2f}]",
                        ha="center", va="center", fontsize=6, color=color, alpha=0.8)

    ax.set_xticks(range(len(prop_names)))
    ax.set_xticklabels([PROP_LABELS[p] for p in prop_names], fontsize=9)
    ax.set_yticks(range(len(mat_names)))
    ax.set_yticklabels(mat_names, fontsize=8)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Spearman ρ (ordered vs disordered)", fontsize=9)

    ax.set_title("Disorder sensitivity by material and property", fontsize=11, pad=20)

    out = FIG_DIR / "fig1_heatmap"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 1 saved: {out}.pdf")
    return rho_matrix, n_matrix


# ── Figure 2: Pipeline divergence funnel ──────────────────────────────────

def fig2_pipeline_funnel():
    """Horizontal funnel showing Jaccard drop through 3-gate pipeline."""
    lco_results = load_material(PROJECT_DIR / "lco")
    if not lco_results:
        print("  Fig 2 skipped: no LCO data")
        return

    # Extract ordered and disordered rankings for each property
    dopants = [r["dopant"] for r in lco_results]
    n = len(dopants)

    def rank_by(results, prop, use_disorder=False):
        """Return dopant list sorted by property value."""
        if use_disorder:
            vals = [(r["dopant"], r["sqs_mean"].get(prop, np.nan)) for r in results]
        else:
            vals = [(r["dopant"], r["ordered"].get(prop, np.nan)) for r in results]
        vals = [(d, v) for d, v in vals if not np.isnan(v)]
        # formation_energy: more negative = more stable → sort ascending
        # voltage: more negative MACE = higher physical voltage → sort ascending
        # volume_change: smaller |change| = better → sort by abs
        if prop == "volume_change":
            vals.sort(key=lambda x: abs(x[1]))
        else:
            vals.sort(key=lambda x: x[1])
        return [d for d, v in vals]

    # Gate 1: Formation energy — keep top 71% (most stable)
    gate1_frac = 0.71
    # Gate 2: Volume change — keep top 53% of G1 survivors (smallest |ΔV|)
    gate2_frac = 0.53
    # Gate 3: Voltage — keep top 50% of G2 survivors (highest voltage)
    gate3_frac = 0.50

    def run_pipeline(use_disorder):
        ranked_ef = rank_by(lco_results, "formation_energy", use_disorder)
        g1_n = max(1, int(len(ranked_ef) * gate1_frac))
        g1_survivors = set(ranked_ef[:g1_n])

        # Filter results to G1 survivors for volume ranking
        g1_results = [r for r in lco_results if r["dopant"] in g1_survivors]
        ranked_vol = rank_by(g1_results, "volume_change", use_disorder)
        g2_n = max(1, int(len(ranked_vol) * gate2_frac))
        g2_survivors = set(ranked_vol[:g2_n])

        # Filter to G2 survivors for voltage ranking
        g2_results = [r for r in lco_results if r["dopant"] in g2_survivors]
        ranked_v = rank_by(g2_results, "voltage", use_disorder)
        g3_n = max(1, int(len(ranked_v) * gate3_frac))
        g3_survivors = set(ranked_v[:g3_n])

        return g1_survivors, g2_survivors, g3_survivors

    ord_g1, ord_g2, ord_g3 = run_pipeline(False)
    dis_g1, dis_g2, dis_g3 = run_pipeline(True)

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    stages = ["All\ndopants", "Gate 1\nEf", "Gate 2\nVolume", "Gate 3\nVoltage"]
    ord_counts = [n, len(ord_g1), len(ord_g2), len(ord_g3)]
    dis_counts = [n, len(dis_g1), len(dis_g2), len(dis_g3)]
    jaccards = [1.0, jaccard(ord_g1, dis_g1), jaccard(ord_g2, dis_g2), jaccard(ord_g3, dis_g3)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.5), height_ratios=[3, 1.2],
                                     gridspec_kw={"hspace": 0.4})

    # Top: funnel bars
    x = np.arange(len(stages))
    w = 0.35
    ax1.bar(x - w/2, ord_counts, w, color=CB_BLUE, alpha=0.85, label="Ordered", edgecolor="white")
    ax1.bar(x + w/2, dis_counts, w, color=CB_AMBER, alpha=0.85, label="Disordered (SQS)", edgecolor="white")

    for i, (oc, dc) in enumerate(zip(ord_counts, dis_counts)):
        ax1.text(i - w/2, oc + 0.3, str(oc), ha="center", va="bottom", fontsize=8, color=CB_BLUE, fontweight="bold")
        ax1.text(i + w/2, dc + 0.3, str(dc), ha="center", va="bottom", fontsize=8, color=CB_AMBER, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylabel("Candidates remaining")
    ax1.set_title("Sequential pruning pipeline — LiCoO₂", fontsize=11)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: Jaccard similarity
    colors = [CB_GREEN if j > 0.5 else CB_VERMILION for j in jaccards]
    ax2.bar(x, jaccards, 0.5, color=colors, alpha=0.85, edgecolor="white")
    for i, j in enumerate(jaccards):
        ax2.text(i, j + 0.03, f"J={j:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.set_ylabel("Jaccard\nsimilarity")
    ax2.set_ylim(0, 1.15)
    ax2.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    out = FIG_DIR / "fig2_pipeline_funnel"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 2 saved: {out}.pdf")
    return jaccards


# ── Figure 3: Voltage distributions (LCO) ────────────────────────────────

def fig3_voltage_distributions():
    """Ordered voltage vs SQS distribution for each LCO dopant, sorted by z-score."""
    lco_results = load_material(PROJECT_DIR / "lco")
    if not lco_results:
        print("  Fig 3 skipped: no LCO data")
        return

    # Compute z-scores
    zdata = []
    for r in lco_results:
        v_ord = r["ordered"].get("voltage")
        v_sqs = r["sqs_values"].get("voltage", [])
        if v_ord is None or len(v_sqs) < 2:
            continue
        mean = np.mean(v_sqs)
        std = np.std(v_sqs, ddof=1)
        z = (v_ord - mean) / std if std > 1e-6 else 0.0
        zdata.append({
            "dopant": r["dopant"],
            "v_ord": v_ord,
            "v_sqs": v_sqs,
            "mean": mean,
            "std": std,
            "z": z,
        })

    # Sort by z-score
    zdata.sort(key=lambda x: x["z"])

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    y_positions = np.arange(len(zdata))

    # First pass: draw data
    for i, d in enumerate(zdata):
        sqs = np.array(d["v_sqs"])
        is_tail = abs(d["z"]) > 1.5
        color = CB_VERMILION if is_tail else CB_GREY
        # SQS individual points
        ax.scatter(sqs, [i] * len(sqs), color=CB_SKY, s=18, alpha=0.5, zorder=2)
        # Mean ± std bar
        ax.errorbar(d["mean"], i, xerr=d["std"], fmt="none", color=CB_BLUE,
                    capsize=2.5, capthick=1, linewidth=1, zorder=3)
        # Ordered value as diamond
        ax.scatter(d["v_ord"], i, marker="D", s=45, color=color,
                   edgecolor="black", linewidth=0.5, zorder=4)

    # Set axis limits, then add z-score annotations at right edge
    ax.set_xlim(auto=True)
    xlim = ax.get_xlim()
    x_annot = xlim[1] + 0.005  # just past right edge

    for i, d in enumerate(zdata):
        is_tail = abs(d["z"]) > 1.5
        color = CB_VERMILION if is_tail else CB_GREY
        weight = "bold" if is_tail else "normal"
        ax.text(x_annot, i, f"z={d['z']:+.1f}", fontsize=6.5, va="center", ha="left",
                color=color, fontweight=weight, clip_on=False)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([d["dopant"] for d in zdata], fontsize=8)
    ax.set_xlabel("MACE voltage (eV, more negative = higher physical voltage)")
    ax.set_title("LiCoO₂: Ordered voltage vs SQS distribution", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=CB_VERMILION,
                   markeredgecolor="black", markersize=8, label="Ordered (|z|>1.5)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=CB_GREY,
                   markeredgecolor="black", markersize=8, label="Ordered (|z|≤1.5)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CB_SKY,
                   markersize=8, label="SQS realisations"),
        plt.Line2D([0], [0], color=CB_BLUE, linewidth=1.5, label="SQS mean ± σ"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)

    out = FIG_DIR / "fig3_voltage_distributions"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 3 saved: {out}.pdf")


# ── Figure 4: Interaction energy decay ────────────────────────────────────

def fig4_interaction_energy():
    """E_int vs distance for LCO and LMO from interaction_Al.json."""
    json_path = SCRIPT_DIR / "interaction_Al.json"
    if not json_path.exists():
        print("  Fig 4 skipped: interaction_Al.json not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    for mat_key, style in [("LiCoO2", {"color": CB_BLUE, "marker": "o", "label": "LiCoO₂ (layered)"}),
                            ("LiMn2O4", {"color": CB_AMBER, "marker": "s", "label": "LiMn₂O₄ (spinel)"})]:
        mat = data["materials"].get(mat_key, {})
        results = mat.get("results", [])
        if not results:
            continue
        dists = [r["distance"] for r in results]
        e_ints = [r["E_interaction_meV"] for r in results]
        ax.plot(dists, e_ints, style["marker"] + "-", color=style["color"],
                label=style["label"], markersize=6, linewidth=1.5, alpha=0.85)

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.axhspan(-20, 20, color="grey", alpha=0.08, label="±20 meV (noise floor)")

    ax.set_xlabel("Dopant–dopant distance (Å)")
    ax.set_ylabel("Interaction energy E_int (meV)")
    ax.set_title("Al dopant–dopant interaction energy", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate key values
    ax.annotate(f"−128 meV\n(attractive)", xy=(2.87, -128), xytext=(5, -100),
                fontsize=7, color=CB_BLUE, ha="left",
                arrowprops=dict(arrowstyle="->", color=CB_BLUE, lw=0.8))
    ax.annotate(f"+145 meV\n(repulsive)", xy=(2.85, 145), xytext=(5, 120),
                fontsize=7, color=CB_AMBER, ha="left",
                arrowprops=dict(arrowstyle="->", color=CB_AMBER, lw=0.8))

    out = FIG_DIR / "fig4_interaction_energy"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 4 saved: {out}.pdf")


# ── Figure 5: Phase-space safe/danger zone ────────────────────────────────

def fig5_phase_space():
    """2D map: voltage ρ (x) vs structural dimensionality (y) with jitter.
    Materials placed in safe vs danger zones.
    """
    materials = {
        "LiCoO₂": {"x_rho": None, "y_dim": 2, "color": CB_VERMILION, "structure": "layered"},
        "LiNiO₂": {"x_rho": None, "y_dim": 2, "color": CB_VERMILION, "structure": "layered"},
        "LiMn₂O₄": {"x_rho": None, "y_dim": 3, "color": CB_GREEN, "structure": "spinel"},
        "SrTiO₃": {"x_rho": None, "y_dim": 3, "color": CB_GREEN, "structure": "perovskite"},
        "CeO₂": {"x_rho": None, "y_dim": 3, "color": CB_GREEN, "structure": "fluorite"},
    }

    # Compute voltage ρ from data
    rho_map = {}
    for mat_label, ckpt_dir in CHECKPOINT_DIRS.items():
        results = load_material(ckpt_dir)
        rho, _, n = compute_spearman(results, "voltage")
        clean = mat_label.split("\n")[0]
        rho_map[clean] = rho

    for name, info in materials.items():
        info["x_rho"] = rho_map.get(name, np.nan)

    # STO and CeO2 don't have voltage — use formation energy ρ as proxy
    for mat_label, ckpt_dir in CHECKPOINT_DIRS.items():
        results = load_material(ckpt_dir)
        clean = mat_label.split("\n")[0]
        if np.isnan(materials.get(clean, {}).get("x_rho", np.nan)):
            rho_ef, _, _ = compute_spearman(results, "formation_energy")
            if clean in materials:
                materials[clean]["x_rho"] = rho_ef

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Background zones
    ax.axvspan(-1, 0.5, color=CB_VERMILION, alpha=0.06)
    ax.axvspan(0.5, 1.15, color=CB_GREEN, alpha=0.06)
    ax.text(-0.05, 3.65, "DANGER ZONE\n(disorder-aware screening required)",
            fontsize=8, color=CB_VERMILION, ha="center", va="top", style="italic")
    ax.text(0.85, 3.65, "SAFE ZONE\n(ordered screening sufficient)",
            fontsize=8, color=CB_GREEN, ha="center", va="top", style="italic")

    # Jitter 3D materials vertically to avoid overlap
    y_offsets = {
        "LiCoO₂": -0.08, "LiNiO₂": 0.08,
        "LiMn₂O₄": -0.15, "SrTiO₃": 0.0, "CeO₂": 0.15,
    }

    # Label positions: (dx, dy) in points, ha
    label_pos = {
        "LiCoO₂": (-12, -14, "center"),
        "LiNiO₂": (12, -14, "center"),
        "LiMn₂O₄": (-12, -14, "center"),
        "SrTiO₃": (0, 14, "center"),
        "CeO₂": (12, -14, "center"),
    }

    for name, info in materials.items():
        x = info["x_rho"]
        y = info["y_dim"] + y_offsets.get(name, 0)
        if np.isnan(x):
            continue
        ax.scatter(x, y, s=140, color=info["color"], edgecolor="black",
                   linewidth=0.8, zorder=5)
        dx, dy, ha = label_pos.get(name, (0, 12, "center"))
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, ha=ha, va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5) if abs(dy) > 10 else None)

    ax.axvline(0.5, color="grey", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Ranking preservation (Spearman ρ)\n← disrupted          preserved →", fontsize=9)
    ax.set_ylabel("Li transport dimensionality")
    ax.set_xlim(-0.5, 1.15)
    ax.set_ylim(1.5, 3.8)
    ax.set_yticks([2, 3])
    ax.set_yticklabels(["2D\n(layered)", "3D\n(spinel / perovskite / fluorite)"])
    ax.set_title("When is ordered dopant screening safe?", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIG_DIR / "fig5_phase_space"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 5 saved: {out}.pdf")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating publication figures...\n")

    rho_matrix, n_matrix = fig1_heatmap()
    print(f"    ρ matrix:\n{rho_matrix}\n    n matrix:\n{n_matrix}\n")

    jaccards = fig2_pipeline_funnel()
    if jaccards:
        print(f"    Jaccards: {jaccards}\n")

    fig3_voltage_distributions()
    fig4_interaction_energy()
    fig5_phase_space()

    print(f"\nAll figures saved to {FIG_DIR}/")
