#!/usr/bin/env python3
"""
Publication figures for the Disorder Gap paper (v2).
====================================================
Generates 6 main figures + Extended Data figures:

  Fig 1: Workflow schematic (process/steps diagram)
  Fig 2: Spearman ρ heatmap (8 materials × 3 properties)
  Fig 3: Pipeline divergence funnel (LCO)
  Fig 4: Partial delithiation voltage comparison
  Fig 5: Dopant–dopant interaction energy convergence
  Fig 6: Disorder-risk predictor phase space (23 observations)

  ED Fig 1: LCO voltage distributions with z-scores
  ED Fig 2: NMC811 voltage comparison
  ED Fig 3: LFP and NASICON detailed results

Usage:
    python paper/make_figures_v2.py
"""

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Original 5-material checkpoint directories
CHECKPOINT_DIRS = {
    "LiCoO₂":  PROJECT_DIR / "lco",
    "LiNiO₂":  PROJECT_DIR / "lno",
    "LiMn₂O₄": PROJECT_DIR / "lmo",
    "SrTiO₃":  PROJECT_DIR / "sto",
    "CeO₂":    PROJECT_DIR / "ceo2",
}

PROPERTIES = ["formation_energy", "voltage", "volume_change"]
PROP_LABELS = {
    "formation_energy": "Formation\nenergy",
    "voltage": "Voltage",
    "volume_change": "Volume\nchange",
}

# ── Nature-quality style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
})

# Colour-blind friendly palette (Wong 2011)
C_BLUE = "#0072B2"
C_AMBER = "#E69F00"
C_GREEN = "#009E73"
C_PINK = "#CC79A7"
C_SKY = "#56B4E9"
C_RED = "#D55E00"
C_GREY = "#999999"
C_BLACK = "#000000"

SAFE_COL = C_GREEN
UNSAFE_COL = C_RED
ORD_COL = C_BLUE
DIS_COL = C_AMBER


# ── Data loading ───────────────────────────────────────────────────────────

def load_material(ckpt_dir):
    """Load all dopant checkpoints from a directory."""
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 1: Workflow / Process Diagram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_box(ax, xy, w, h, text, facecolor, edgecolor="0.3", fontsize=7.5,
              fontweight="normal", alpha=0.9, text_color="black", zorder=3):
    """Draw a rounded rectangle with centred text."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=0.8, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            zorder=zorder+1, linespacing=1.3)
    return box


def _draw_arrow(ax, start, end, color="0.4", style="-|>", lw=1.0):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                            color=color, linewidth=lw,
                            connectionstyle="arc3,rad=0.0",
                            zorder=2)
    ax.add_patch(arrow)


def fig1_workflow():
    """Process/steps schematic showing conventional vs disorder-aware screening."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # ── Column headers ──
    ax.text(0.22, 0.98, "Conventional\n(Ordered)", ha="center", va="top",
            fontsize=9, fontweight="bold", color=ORD_COL)
    ax.text(0.78, 0.98, "Disorder-Aware\n(SQS Ensemble)", ha="center", va="top",
            fontsize=9, fontweight="bold", color=DIS_COL)
    ax.text(0.50, 0.98, "Decision\nRule", ha="center", va="top",
            fontsize=8, fontweight="bold", color="0.3")

    bw, bh = 0.28, 0.10  # box width, height

    # ── Step 1: Input (shared) ──
    _draw_box(ax, (0.36, 0.83), 0.28, 0.08,
              "Crystal structure\n+ dopant candidates",
              facecolor="#E8E8E8", fontsize=7.5, fontweight="bold")

    # ── Left track: Ordered ──
    y_left = [0.70, 0.55, 0.40, 0.25]
    left_labels = [
        "1. Build ordered\nsupercell (1 config)",
        "2. MACE-MP-0\nrelaxation",
        "3. Compute Ef, V, ΔV\n(single value per dopant)",
        "4. Rank & prune\n(sequential gates)",
    ]
    left_colors = ["#DAEAF6", "#DAEAF6", "#DAEAF6", "#DAEAF6"]

    for i, (y, label, col) in enumerate(zip(y_left, left_labels, left_colors)):
        _draw_box(ax, (0.08, y), bw, bh, label, facecolor=col,
                  edgecolor=C_BLUE, fontsize=6.5)
        if i > 0:
            _draw_arrow(ax, (0.22, y_left[i-1]), (0.22, y + bh), color=C_BLUE)

    # Arrow from input to left
    _draw_arrow(ax, (0.42, 0.83), (0.22, 0.70 + bh), color="0.5")

    # ── Right track: SQS ──
    y_right = [0.70, 0.55, 0.40, 0.25]
    right_labels = [
        "1. Generate 5 SQS\nrealisations per dopant",
        "2. MACE-MP-0\nrelaxation (×5)",
        "3. Compute Ef, V, ΔV\n(mean ± σ per dopant)",
        "4. Rank with\nuncertainty propagation",
    ]
    right_colors = ["#FFF2CC", "#FFF2CC", "#FFF2CC", "#FFF2CC"]

    for i, (y, label, col) in enumerate(zip(y_right, right_labels, right_colors)):
        _draw_box(ax, (0.64, y), bw, bh, label, facecolor=col,
                  edgecolor=C_AMBER, fontsize=6.5)
        if i > 0:
            _draw_arrow(ax, (0.78, y_right[i-1]), (0.78, y + bh), color=C_AMBER)

    # Arrow from input to right
    _draw_arrow(ax, (0.58, 0.83), (0.78, 0.70 + bh), color="0.5")

    # ── Centre: Decision rule ──
    _draw_box(ax, (0.39, 0.58), 0.22, 0.12,
              "Risk Score\nR = scope × aniso\n+ 0.3(n_TM − 1)\n\nR > 1.0 → right track",
              facecolor="#E8F5E9", edgecolor=C_GREEN, fontsize=6,
              fontweight="bold")

    # ── Bottom: Comparison ──
    _draw_box(ax, (0.08, 0.10), bw, 0.09,
              "Finalist set A\n{Al, Ge, V}",
              facecolor="#BBDEFB", edgecolor=C_BLUE, fontsize=6.5, fontweight="bold")
    _draw_arrow(ax, (0.22, 0.25), (0.22, 0.10 + 0.09), color=C_BLUE)

    _draw_box(ax, (0.64, 0.10), bw, 0.09,
              "Finalist set B\n{Ge, Ni, Rh}",
              facecolor="#FFF9C4", edgecolor=C_AMBER, fontsize=6.5, fontweight="bold")
    _draw_arrow(ax, (0.78, 0.25), (0.78, 0.10 + 0.09), color=C_AMBER)

    # Disorder gap label
    _draw_box(ax, (0.38, 0.02), 0.24, 0.07,
              "DISORDER GAP\nJaccard = 0.20\n(1 of 3 shared)",
              facecolor="#FFCDD2", edgecolor=C_RED, fontsize=6.5,
              fontweight="bold", text_color=C_RED)
    _draw_arrow(ax, (0.36, 0.145), (0.38, 0.07), color=C_RED, lw=0.8)
    _draw_arrow(ax, (0.64, 0.145), (0.62, 0.07), color=C_RED, lw=0.8)

    # Panel label
    ax.text(-0.02, 1.02, "a", fontsize=12, fontweight="bold", transform=ax.transAxes)

    out = FIG_DIR / "fig1_workflow"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 1 saved: {out}.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 2: Expanded Spearman ρ Heatmap (8 materials)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fig2_heatmap():
    """8-material × 3-property Spearman ρ heatmap."""

    # Load original 5 materials from checkpoints
    orig_data = {}
    for mat_label, ckpt_dir in CHECKPOINT_DIRS.items():
        results = load_material(ckpt_dir)
        rhos = {}
        for prop in PROPERTIES:
            rho, p, n = compute_spearman(results, prop)
            lo, hi = bootstrap_spearman(results, prop)
            rhos[prop] = {"rho": rho, "lo": lo, "hi": hi, "n": n}
        orig_data[mat_label] = rhos

    # Additional materials from JSON results (with bootstrap CIs)
    extra = {
        "NMC811": {"formation_energy": 0.52, "voltage": 0.09, "volume_change": np.nan},
        "NaCoO₂": {"formation_energy": 0.79, "voltage": 0.23, "volume_change": -0.01},
        "LiFePO₄": {"formation_energy": 1.00, "voltage": 0.99, "volume_change": 0.79},
        "Na₃V₂(PO₄)₃": {"formation_energy": 0.72, "voltage": 0.77, "volume_change": -0.04},
    }
    extra_ci = {
        "NMC811": {
            "formation_energy": (-0.02, 0.85), "voltage": (-0.51, 0.63),
            "volume_change": (np.nan, np.nan),
        },
        "NaCoO₂": {
            "formation_energy": (0.46, 0.94), "voltage": (-0.26, 0.64),
            "volume_change": (-0.50, 0.49),
        },
        "LiFePO₄": {
            "formation_energy": (1.00, 1.00), "voltage": (0.94, 1.00),
            "volume_change": (0.49, 0.94),
        },
        "Na₃V₂(PO₄)₃": {
            "formation_energy": (0.31, 0.93), "voltage": (0.44, 0.93),
            "volume_change": (-0.57, 0.50),
        },
    }

    # Build the full matrix — all 9 materials
    mat_names = [
        "LiCoO$_2$", "LiNiO$_2$", "NMC811", "NaCoO$_2$",  # layered
        "LiMn$_2$O$_4$",                                     # spinel
        "LiFePO$_4$",                                         # olivine
        "Na$_3$V$_2$(PO$_4$)$_3$",                           # NASICON
        "SrTiO$_3$", "CeO$_2$",                              # non-cathode 3D
    ]
    # Keys for data lookup (plain unicode)
    mat_keys = [
        "LiCoO₂", "LiNiO₂", "NMC811", "NaCoO₂",
        "LiMn₂O₄", "LiFePO₄", "Na₃V₂(PO₄)₃",
        "SrTiO₃", "CeO₂",
    ]
    structures = [
        "Layered", "Layered", "Layered", "Layered",
        "Spinel", "Olivine", "NASICON",
        "Perovskite", "Fluorite",
    ]

    # Build unsorted matrix: rows = materials, cols = properties
    n_mat = len(mat_names)
    rho_raw = np.full((n_mat, 3), np.nan)
    ci_raw = {}

    for i, mat_key in enumerate(mat_keys):
        for j, prop in enumerate(PROPERTIES):
            mat = mat_key
            if mat in orig_data:
                d = orig_data[mat][prop]
                rho_raw[i, j] = d["rho"]
                ci_raw[(i, j)] = (d["lo"], d["hi"])
            elif mat in extra:
                rho_raw[i, j] = extra[mat][prop]
                ci_raw[(i, j)] = extra_ci.get(mat, {}).get(prop, (np.nan, np.nan))

    # Sort materials by volume_change ρ (column index 2), NaN sorts last
    vol_rhos = rho_raw[:, 2].copy()
    vol_rhos_sortable = np.where(np.isnan(vol_rhos), -999, vol_rhos)
    sort_idx = np.argsort(vol_rhos_sortable)  # ascending: red→green

    mat_names_sorted = [mat_names[i] for i in sort_idx]
    structures_sorted = [structures[i] for i in sort_idx]
    mat_keys_sorted = [mat_keys[i] for i in sort_idx]

    # Build sorted matrix and CI lookup
    rho_matrix = rho_raw[sort_idx, :]
    ci_matrix = {}
    for new_i, old_i in enumerate(sort_idx):
        for j in range(3):
            if (old_i, j) in ci_raw:
                ci_matrix[(new_i, j)] = ci_raw[(old_i, j)]

    # ── Horizontal layout: transpose so rows=properties, cols=materials ──
    rho_T = rho_matrix.T  # shape (3, n_mat)

    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(rho_T, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for j_prop in range(3):
        for i_mat in range(n_mat):
            val = rho_T[j_prop, i_mat]
            if np.isnan(val):
                ax.text(i_mat, j_prop, "—", ha="center", va="center",
                        fontsize=8, color="grey")
                continue
            lo, hi = ci_matrix.get((i_mat, j_prop), (np.nan, np.nan))
            bold = not np.isnan(lo) and not np.isnan(hi) and (lo > 0 or hi < 0)
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(i_mat, j_prop, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if bold else "normal", color=color)
            if not np.isnan(lo):
                ax.text(i_mat, j_prop + 0.32, f"[{lo:+.2f},{hi:+.2f}]",
                        ha="center", va="center", fontsize=4.5, color=color, alpha=0.7)

    # X-axis: material names with structure type (top)
    xlabels = [f"{m}\n({s})" for m, s in zip(mat_names_sorted, structures_sorted)]
    ax.set_xticks(range(n_mat))
    ax.set_xticklabels(xlabels, fontsize=6.5, ha="center")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Y-axis: property names (left)
    prop_labels_list = [PROP_LABELS[p] for p in PROPERTIES]
    ax.set_yticks(range(3))
    ax.set_yticklabels(prop_labels_list, fontsize=8)

    # Vertical divider lines between structure groups (find boundaries in sorted order)
    struct_boundaries = []
    for k in range(1, n_mat):
        if structures_sorted[k] != structures_sorted[k - 1]:
            struct_boundaries.append(k - 0.5)
    for x_pos in struct_boundaries:
        ax.axvline(x_pos, color="white", lw=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=15,
                         orientation="vertical")
    cbar.set_label("Spearman $\\rho$", fontsize=8)

    ax.text(-0.06, 1.25, "b", fontsize=12, fontweight="bold", transform=ax.transAxes)

    out = FIG_DIR / "fig2_heatmap"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 2 saved: {out}.pdf")
    return rho_matrix


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 3: Pipeline Divergence Funnel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fig3_pipeline_funnel():
    """Horizontal funnel showing Jaccard drop through 3-gate pipeline."""
    lco_results = load_material(PROJECT_DIR / "lco")
    if not lco_results:
        print("  Fig 3 skipped: no LCO data")
        return

    dopants = [r["dopant"] for r in lco_results]
    n = len(dopants)

    def rank_by(results, prop, use_disorder=False):
        if use_disorder:
            vals = [(r["dopant"], r["sqs_mean"].get(prop, np.nan)) for r in results]
        else:
            vals = [(r["dopant"], r["ordered"].get(prop, np.nan)) for r in results]
        vals = [(d, v) for d, v in vals if not np.isnan(v)]
        if prop == "volume_change":
            vals.sort(key=lambda x: abs(x[1]))
        else:
            vals.sort(key=lambda x: x[1])
        return [d for d, v in vals]

    def run_pipeline(use_disorder, data=None):
        data = data or lco_results
        ranked_ef = rank_by(data, "formation_energy", use_disorder)
        g1_n = max(1, int(len(ranked_ef) * 0.71))
        g1 = set(ranked_ef[:g1_n])
        g1r = [r for r in data if r["dopant"] in g1]
        ranked_vol = rank_by(g1r, "volume_change", use_disorder)
        g2_n = max(1, int(len(ranked_vol) * 0.53))
        g2 = set(ranked_vol[:g2_n])
        g2r = [r for r in data if r["dopant"] in g2]
        ranked_v = rank_by(g2r, "voltage", use_disorder)
        g3_n = max(1, int(len(ranked_v) * 0.50))
        g3 = set(ranked_v[:g3_n])
        return g1, g2, g3

    def run_pipeline_from(data, use_disorder):
        return run_pipeline(use_disorder, data)

    ord_g1, ord_g2, ord_g3 = run_pipeline(False)
    dis_g1, dis_g2, dis_g3 = run_pipeline(True)

    jaccard = lambda a, b: len(a & b) / len(a | b) if (a or b) else 1.0

    stages = ["All\ndopants", "Gate 1\n$E_f$", "Gate 2\n$\\Delta$V", "Gate 3\nVoltage"]
    ord_counts = [n, len(ord_g1), len(ord_g2), len(ord_g3)]
    dis_counts = [n, len(dis_g1), len(dis_g2), len(dis_g3)]
    jaccards = [1.0, jaccard(ord_g1, dis_g1), jaccard(ord_g2, dis_g2), jaccard(ord_g3, dis_g3)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 3.8), height_ratios=[3, 1.2],
                                   gridspec_kw={"hspace": 0.45})

    x = np.arange(len(stages))
    w = 0.32
    ax1.bar(x - w/2, ord_counts, w, color=ORD_COL, alpha=0.85, label="Ordered", edgecolor="white")
    ax1.bar(x + w/2, dis_counts, w, color=DIS_COL, alpha=0.85, label="Disordered (SQS)", edgecolor="white")
    for i, (oc, dc) in enumerate(zip(ord_counts, dis_counts)):
        ax1.text(i - w/2, oc + 0.3, str(oc), ha="center", va="bottom", fontsize=7, color=ORD_COL, fontweight="bold")
        ax1.text(i + w/2, dc + 0.3, str(dc), ha="center", va="bottom", fontsize=7, color=DIS_COL, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylabel("Candidates remaining")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # MC bootstrap for Jaccard CIs
    rng = np.random.default_rng(42)
    n_mc = 1000
    jaccard_samples = {g: [] for g in range(4)}
    for _ in range(n_mc):
        # Resample SQS values for each dopant, recompute means
        resampled = []
        for r in lco_results:
            rd = dict(r)
            new_sqs_mean = {}
            for prop in PROPERTIES:
                vals = r["sqs_values"].get(prop, [])
                if vals:
                    boot_vals = [vals[rng.integers(0, len(vals))] for _ in range(len(vals))]
                    new_sqs_mean[prop] = np.mean(boot_vals)
                else:
                    new_sqs_mean[prop] = r["sqs_mean"].get(prop, np.nan)
            rd = {**r, "sqs_mean": new_sqs_mean}
            resampled.append(rd)
        dg1, dg2, dg3 = run_pipeline_from(resampled, True)
        jaccard_samples[0].append(1.0)
        jaccard_samples[1].append(jaccard(ord_g1, dg1))
        jaccard_samples[2].append(jaccard(ord_g2, dg2))
        jaccard_samples[3].append(jaccard(ord_g3, dg3))

    j_lo = [np.percentile(jaccard_samples[g], 2.5) for g in range(4)]
    j_hi = [np.percentile(jaccard_samples[g], 97.5) for g in range(4)]
    j_err_lo = [jaccards[g] - j_lo[g] for g in range(4)]
    j_err_hi = [j_hi[g] - jaccards[g] for g in range(4)]

    colors = [SAFE_COL if j > 0.5 else UNSAFE_COL for j in jaccards]
    ax2.bar(x, jaccards, 0.45, color=colors, alpha=0.85, edgecolor="white")
    ax2.errorbar(x, jaccards, yerr=[j_err_lo, j_err_hi], fmt="none",
                 color="black", capsize=3, capthick=0.8, linewidth=0.8, zorder=5)
    for i, j in enumerate(jaccards):
        ax2.text(i, j_hi[i] + 0.04, f"J={j:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.set_ylabel("Jaccard")
    ax2.set_ylim(0, 1.25)
    ax2.axhline(0.5, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax1.text(-0.12, 1.05, "c", fontsize=12, fontweight="bold", transform=ax1.transAxes)

    out = FIG_DIR / "fig3_pipeline_funnel"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 3 saved: {out}.pdf")
    return jaccards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 4: Partial Delithiation Voltage Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fig4_partial_delithiation():
    """Ordered vs disordered partial delithiation voltages for LCO."""
    json_path = SCRIPT_DIR / "partial_delithiation_sqs_results.json"
    if not json_path.exists():
        print("  Fig 4 skipped: partial_delithiation_sqs_results.json not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    dopant_results = data["dopant_results"]

    # Collect valid dopants
    dopants, v_ord, v_dis_mean, v_dis_std = [], [], [], []
    for d, info in sorted(dopant_results.items()):
        if "error" in info:
            continue
        dopants.append(d)
        v_ord.append(info["voltage_partial_ordered"])
        v_dis_mean.append(info["voltage_partial_disordered_mean"])
        v_dis_std.append(info["voltage_partial_disordered_std"])

    # Sort by ordered voltage
    idx = np.argsort(v_ord)
    dopants = [dopants[i] for i in idx]
    v_ord = [v_ord[i] for i in idx]
    v_dis_mean = [v_dis_mean[i] for i in idx]
    v_dis_std = [v_dis_std[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = np.arange(len(dopants))

    # SQS mean ± std
    ax.errorbar(v_dis_mean, y, xerr=v_dis_std, fmt="o", color=DIS_COL,
                markersize=5, capsize=2.5, capthick=0.8, linewidth=0.8,
                label="SQS mean ± σ", zorder=3, alpha=0.85)
    # Ordered values
    ax.scatter(v_ord, y, marker="D", s=35, color=ORD_COL,
               edgecolor="black", linewidth=0.4, zorder=4, label="Ordered")

    # Connecting lines
    for i in range(len(dopants)):
        ax.plot([v_ord[i], v_dis_mean[i]], [y[i], y[i]], color="0.8", lw=0.5, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(dopants, fontsize=7)
    ax.set_xlabel("Partial delithiation voltage (eV, x = 0 → 0.5)")
    ax.set_title(f"LiCoO$_2$ partial delithiation — $\\rho$ = {data['voltage_partial_rho']:.2f}", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ρ annotation
    ax.text(0.03, 0.97, f"Spearman ρ = {data['voltage_partial_rho']:.2f}\np = {data['voltage_partial_p']:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", edgecolor=UNSAFE_COL, alpha=0.8))

    ax.text(-0.12, 1.05, "d", fontsize=12, fontweight="bold", transform=ax.transAxes)

    out = FIG_DIR / "fig4_partial_delithiation"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 4 saved: {out}.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 5: Interaction Energy Convergence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fig5_interaction_energy():
    """E_int vs supercell size for LCO and LMO."""
    json_path = SCRIPT_DIR / "interaction_Al.json"
    if not json_path.exists():
        print("  Fig 5 skipped: interaction_Al.json not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for mat_key, style in [
        ("LiCoO2", {"color": ORD_COL, "marker": "o", "label": "LiCoO$_2$ (layered)"}),
        ("LiMn2O4", {"color": DIS_COL, "marker": "s", "label": "LiMn$_2$O$_4$ (spinel)"}),
    ]:
        mat = data["materials"].get(mat_key, {})
        results = mat.get("results", [])
        if not results:
            continue
        dists = [r["distance"] for r in results]
        e_ints = [r["E_interaction_meV"] for r in results]
        ax.plot(dists, e_ints, style["marker"] + "-", color=style["color"],
                label=style["label"], markersize=5, linewidth=1.2, alpha=0.85)

    ax.axhline(0, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.axhspan(-20, 20, color="grey", alpha=0.06, label="±20 meV noise floor")

    ax.set_xlabel("Dopant–dopant distance (Å)")
    ax.set_ylabel("$E_{int}$ (meV)")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(-0.12, 1.05, "e", fontsize=12, fontweight="bold", transform=ax.transAxes)

    out = FIG_DIR / "fig5_interaction_energy"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 5 saved: {out}.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 6: Disorder-Risk Predictor (updated, 23 observations)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fig6_predictor():
    """Risk score R vs actual Spearman ρ for all 26 observations."""

    # All 26 observations: (label, R, rho, is_safe)
    # safe threshold: rho >= 0.50
    observations = [
        # Original 5 materials (16 obs)
        ("LCO Ef", 0.0, 0.76, True),
        ("LCO V", 1.94, -0.25, False),
        ("LCO dV", 1.94, 0.09, False),
        ("LNO Ef", 0.0, 0.82, True),
        ("LNO V", 1.93, -0.06, False),
        ("LNO dV", 1.93, 0.54, True),
        ("LMO Ef", 0.0, 1.00, True),
        ("LMO V", 1.0, 0.95, True),
        ("LMO dV", 1.0, 0.84, True),
        ("STO Ef", 0.0, 1.00, True),
        ("STO dV", 1.0, 0.94, True),
        ("CeO2 Ef", 0.0, 1.00, True),
        ("CeO2 dV", 1.0, 0.96, True),
        ("CeO2 Ovac", 1.0, 0.85, True),
        # NMC811
        ("NMC Ef", 0.6, 0.52, True),
        ("NMC V", 2.53, 0.09, False),
        # Out-of-sample: LiFePO4
        ("LFP Ef", 0.0, 1.00, True),
        ("LFP V", 1.21, 0.99, True),
        ("LFP dV", 1.21, 0.79, True),
        # Out-of-sample: NASICON
        ("NASICON Ef", 0.0, 0.72, True),
        ("NASICON V", 1.15, 0.77, True),
        ("NASICON dV", 1.15, -0.04, False),
        # Out-of-sample: NaCoO2
        ("NCO Ef", 0.0, 0.79, True),
        ("NCO V", 1.90, 0.23, False),
        ("NCO dV", 1.90, -0.01, False),
        # Partial delithiation
        ("LCO x=0.5", 1.94, -0.32, False),
        ("LCO x=0.25", 1.94, -0.07, False),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    # Background zones
    ax.axvspan(-0.3, 1.0, color=SAFE_COL, alpha=0.05, zorder=0)
    ax.axvspan(1.0, 3.0, color=UNSAFE_COL, alpha=0.05, zorder=0)
    ax.axvline(1.0, color="0.5", ls="--", lw=0.8, zorder=1)
    ax.axhline(0.5, color="0.5", ls=":", lw=0.6, alpha=0.5, zorder=1)

    # Zone labels — positioned to avoid the annotation box
    ax.text(0.5, 1.05, "Predicted SAFE", ha="center", fontsize=7,
            color=SAFE_COL, fontweight="bold")
    ax.text(2.0, 1.05, "Predicted UNSAFE", ha="center", fontsize=7,
            color=UNSAFE_COL, fontweight="bold")

    # Jitter overlapping R=0 Ef points vertically for visibility
    # Collect R=0 points and add small horizontal jitter
    r0_count = 0
    r0_jitter = {
        "LCO Ef": -0.06, "LNO Ef": -0.02, "LMO Ef": 0.02,
        "STO Ef": 0.06, "CeO2 Ef": -0.06, "LFP Ef": 0.02,
        "NASICON Ef": 0.06, "NCO Ef": -0.02,
        "NMC Ef": 0.0,  # already at R=0.6, no jitter needed
    }

    for label, R, rho, safe in observations:
        marker = "o" if safe else "s"
        color = SAFE_COL if safe else UNSAFE_COL
        edgecolor = "black"
        s = 40
        # Highlight out-of-sample
        if any(tag in label for tag in ["LFP", "NASICON", "NCO", "x=0.5", "x=0.25"]):
            edgecolor = C_PINK
            s = 55
        # Apply jitter for R=0 cluster
        R_plot = R + r0_jitter.get(label, 0.0)
        ax.scatter(R_plot, rho, marker=marker, s=s, color=color,
                   edgecolor=edgecolor, linewidth=0.6, zorder=4)

    # Label key points with carefully tuned offsets to avoid overlap
    label_points = {
        # Unsafe region — spread vertically
        "LCO V":     (0.12, 0.06),    # ρ=-0.25
        "LCO x=0.5": (0.12, -0.08),   # ρ=-0.32
        "LCO x=0.25":(-0.30, 0.07),   # ρ=-0.07
        "NCO V":     (-0.20, -0.08),   # ρ=0.23
        "NMC V":     (0.10, 0.06),     # ρ=0.09, R=2.53
        "NASICON dV":(-0.20, -0.06),   # ρ=-0.04
        # Safe region
        "LFP V":     (0.12, -0.06),    # ρ=0.99
        "NASICON V": (-0.18, -0.06),   # ρ=0.77
        "LMO V":     (-0.15, -0.06),   # ρ=0.95
    }
    for label, R, rho, safe in observations:
        if label in label_points:
            dx, dy = label_points[label]
            R_plot = R + r0_jitter.get(label, 0.0)
            ax.annotate(label, (R_plot, rho), xytext=(R_plot + dx, rho + dy),
                        fontsize=5.5, color="0.3", ha="center",
                        arrowprops=dict(arrowstyle="-", color="0.7", lw=0.4))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SAFE_COL,
               markeredgecolor="black", markersize=7,
               label="Actually safe ($\\rho$ $\\geq$ 0.50)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=UNSAFE_COL,
               markeredgecolor="black", markersize=7,
               label="Actually unsafe ($\\rho$ < 0.50)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.8",
               markeredgecolor=C_PINK, markersize=7, markeredgewidth=1.5,
               label="Out-of-sample validation"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=6.5,
              bbox_to_anchor=(0.99, 0.55))

    ax.set_xlabel("Risk score R", fontsize=9)
    ax.set_ylabel("Actual Spearman $\\rho$", fontsize=9)
    ax.set_xlim(-0.3, 2.8)
    ax.set_ylim(-0.55, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Accuracy annotation — bottom-left, compact
    ax.text(0.03, 0.03,
            "Zero false-safe predictions\n"
            "Accuracy: 23/27 (85.2%)\n"
            "False-unsafe: 4 (conservative)",
            transform=ax.transAxes, fontsize=6.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    ax.text(-0.12, 1.05, "f", fontsize=12, fontweight="bold", transform=ax.transAxes)

    out = FIG_DIR / "fig6_predictor"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  Fig 6 saved: {out}.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extended Data Figure 1: LCO Voltage Distributions with z-scores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ed_fig1_voltage_distributions():
    """Ordered voltage vs SQS distribution for each LCO dopant."""
    lco_results = load_material(PROJECT_DIR / "lco")
    if not lco_results:
        print("  ED Fig 1 skipped: no LCO data")
        return

    zdata = []
    for r in lco_results:
        v_ord = r["ordered"].get("voltage")
        v_sqs = r["sqs_values"].get("voltage", [])
        if v_ord is None or len(v_sqs) < 2:
            continue
        mean = np.mean(v_sqs)
        std = np.std(v_sqs, ddof=1)
        z = (v_ord - mean) / std if std > 1e-6 else 0.0
        zdata.append({"dopant": r["dopant"], "v_ord": v_ord, "v_sqs": v_sqs,
                       "mean": mean, "std": std, "z": z})
    zdata.sort(key=lambda x: x["z"])

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    y_pos = np.arange(len(zdata))

    for i, d in enumerate(zdata):
        sqs = np.array(d["v_sqs"])
        is_tail = abs(d["z"]) > 1.5
        color = UNSAFE_COL if is_tail else C_GREY
        ax.scatter(sqs, [i]*len(sqs), color=C_SKY, s=14, alpha=0.5, zorder=2)
        ax.errorbar(d["mean"], i, xerr=d["std"], fmt="none", color=ORD_COL,
                    capsize=2, capthick=0.8, linewidth=0.8, zorder=3)
        ax.scatter(d["v_ord"], i, marker="D", s=35, color=color,
                   edgecolor="black", linewidth=0.4, zorder=4)

    ax.set_xlim(auto=True)
    xlim = ax.get_xlim()
    for i, d in enumerate(zdata):
        is_tail = abs(d["z"]) > 1.5
        color = UNSAFE_COL if is_tail else C_GREY
        weight = "bold" if is_tail else "normal"
        ax.text(xlim[1] + 0.005, i, f"z={d['z']:+.1f}", fontsize=6, va="center",
                color=color, fontweight=weight, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([d["dopant"] for d in zdata], fontsize=7)
    ax.set_xlabel("MACE voltage (eV)")
    ax.set_title("LiCoO$_2$: Ordered voltage vs SQS distribution", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=UNSAFE_COL,
               markeredgecolor="black", markersize=7, label="Ordered (|z|>1.5)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_GREY,
               markeredgecolor="black", markersize=7, label="Ordered (|z|≤1.5)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SKY,
               markersize=7, label="SQS realisations"),
        Line2D([0], [0], color=ORD_COL, linewidth=1.2, label="SQS mean ± σ"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5)

    out = FIG_DIR / "ed_fig1_voltage_distributions"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png")
    plt.close(fig)
    print(f"  ED Fig 1 saved: {out}.pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("Generating publication figures (v2)...\n")

    fig1_workflow()
    rho_matrix = fig2_heatmap()
    jaccards = fig3_pipeline_funnel()
    fig4_partial_delithiation()
    fig5_interaction_energy()
    fig6_predictor()

    print("\n--- Extended Data ---")
    ed_fig1_voltage_distributions()

    print(f"\nAll figures saved to {FIG_DIR}/")
