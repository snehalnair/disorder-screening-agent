"""
Publication-quality figure generation for the disorder-screening paper.

Generates 5 figures from evaluation results. Saves as PDF (vector) to
``evaluation/figures/``.

Usage::

    # Generate all figures from saved RQ2/RQ3 results:
    python -m evaluation.figures \\
        --rq1 evaluation/results/rq1_report.json \\
        --rq2 evaluation/results/rq2_disorder.json \\
        --accuracy evaluation/results/rq3_accuracy.json \\
        --output evaluation/figures/

    # Or import individual functions:
    from evaluation.figures import plot_funnel_diagram, plot_ordered_vs_disordered
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)

_FIGURES_DIR = pathlib.Path(__file__).parent / "figures"

# Style constants
_SERIF = "DejaVu Serif"
_COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # amber
    "#009E73",  # green
    "#CC79A7",  # pink/purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#D55E00",  # vermilion
    "#999999",  # grey
]
_SINGLE_COL_W = 3.5  # inches
_DOUBLE_COL_W = 7.0  # inches


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Pipeline funnel diagram
# ─────────────────────────────────────────────────────────────────────────────


def plot_funnel_diagram(
    funnel_counts: dict,
    output_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> pathlib.Path:
    """Horizontal bar chart showing candidate narrowing through pipeline stages.

    Args:
        funnel_counts: Dict with keys ``stage0`` (OS combos), ``stage1``,
                       ``stage2``, ``stage3``.
        output_path:   Where to save the PDF. Defaults to figures/fig1_funnel.pdf.
        show:          If True, call plt.show().

    Returns:
        Path to saved figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({"font.family": _SERIF})

    stage_labels = [
        f"Stage 0\n(All element-OS pairs)\n{funnel_counts.get('stage0', '?')} combinations",
        f"Stage 1 — SMACT\n(EN + charge neutrality)\n{funnel_counts.get('stage1', '?')} candidates",
        f"Stage 2 — Radius\n(Shannon mismatch ≤ 35%)\n{funnel_counts.get('stage2', '?')} candidates",
        f"Stage 3 — Substitution\n(Hautier-Ceder probability)\n{funnel_counts.get('stage3', '?')} candidates",
    ]
    counts = [
        funnel_counts.get("stage0", 0) if isinstance(funnel_counts.get("stage0"), int) else 500,
        funnel_counts.get("stage1", 0),
        funnel_counts.get("stage2", 0),
        funnel_counts.get("stage3", 0),
    ]
    max_count = max(c for c in counts if isinstance(c, (int, float)) and c > 0)

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL_W, 3.0))

    colors = [_COLORBLIND_PALETTE[0], _COLORBLIND_PALETTE[1],
              _COLORBLIND_PALETTE[2], _COLORBLIND_PALETTE[3]]

    for i, (count, label, color) in enumerate(zip(counts, stage_labels, colors)):
        bar_width = count / max_count
        ax.barh(
            y=i,
            width=bar_width,
            height=0.6,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.text(
            bar_width + 0.01, i,
            f"{count}",
            va="center", ha="left",
            fontsize=9, fontweight="bold",
        )
        ax.text(
            -0.01, i,
            label.split("\n")[0],
            va="center", ha="right",
            fontsize=8,
        )

    ax.set_xlim(-0.35, 1.15)
    ax.set_ylim(-0.6, len(counts) - 0.4)
    ax.set_xlabel("Fraction of initial candidates", fontsize=9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Hierarchical Pruning Funnel — NMC811 Co³⁺ dopant screening",
        fontsize=10, pad=8,
    )

    plt.tight_layout()
    out = _save_fig(fig, output_path or _FIGURES_DIR / "fig1_funnel.pdf", show)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Ordered vs disordered bar chart
# ─────────────────────────────────────────────────────────────────────────────


def plot_ordered_vs_disordered(
    rq2_results: dict,
    target_property: str | None = None,
    output_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> pathlib.Path:
    """Grouped bar chart: ordered vs disordered predictions per dopant.

    Uses the property with the lowest Spearman ρ (most affected by disorder).
    Annotates rank changes between ordered and disordered.

    Args:
        rq2_results:    Output of eval_disorder.run_disorder_evaluation().
        target_property: Override property to plot. Defaults to lowest-ρ property.
        output_path:    Where to save PDF.
        show:           If True, call plt.show().
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": _SERIF})

    # Select property with lowest rho if not specified
    rho_data = rq2_results.get("spearman_rho", {})
    if target_property is None and rho_data:
        target_property = min(rho_data, key=lambda p: rho_data[p].get("rho", 1.0))
    if target_property is None:
        target_property = rq2_results.get("target_properties", ["voltage"])[0]

    rows = rq2_results.get("dopant_results", [])
    dopants, ordered_vals, dis_means, dis_stds = [], [], [], []

    for row in rows:
        ord_v = row["ordered"].get(target_property)
        dis_m = row["disordered_mean"].get(target_property)
        dis_s = row["disordered_std"].get(target_property, 0.0)
        if ord_v is not None and dis_m is not None:
            dopants.append(row["dopant"])
            ordered_vals.append(ord_v)
            dis_means.append(dis_m)
            dis_stds.append(dis_s)

    if not dopants:
        logger.warning("No data available for property '%s' — skipping Figure 2.", target_property)
        return _FIGURES_DIR / "fig2_ordered_vs_disordered.pdf"

    x = np.arange(len(dopants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL_W, 3.5))

    bars1 = ax.bar(
        x - width / 2, ordered_vals, width,
        label="Ordered (single cell)", color=_COLORBLIND_PALETTE[0], alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, dis_means, width,
        label="Disordered (SQS mean)", color=_COLORBLIND_PALETTE[1], alpha=0.85,
        yerr=dis_stds, capsize=3, error_kw={"elinewidth": 1},
    )

    # Annotate rank changes
    ord_ranks = _rank_list(ordered_vals, reverse=True)
    dis_ranks = _rank_list(dis_means, reverse=True)
    for i, (or_, dr) in enumerate(zip(ord_ranks, dis_ranks)):
        delta = dr - or_
        if delta != 0:
            arrow = f"↑{abs(delta)}" if delta < 0 else f"↓{abs(delta)}"
            ax.annotate(
                arrow,
                xy=(x[i], max(ordered_vals[i], dis_means[i]) + max(dis_stds) * 0.1),
                ha="center", va="bottom",
                fontsize=7, color="#D55E00", fontweight="bold",
            )

    units = {"voltage": "V", "li_ni_exchange": "eV", "formation_energy": "eV/atom",
             "volume_change": "%"}.get(target_property, "")
    rho_str = ""
    if target_property in rho_data:
        r = rho_data[target_property]
        rho_str = f"  (Spearman ρ = {r['rho']:.2f}, p = {r['pvalue']:.3f})"

    ax.set_xlabel("Dopant", fontsize=9)
    ax.set_ylabel(f"{target_property} ({units})", fontsize=9)
    ax.set_title(f"Ordered vs Disordered: {target_property}{rho_str}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(dopants, fontsize=9)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = _save_fig(fig, output_path or _FIGURES_DIR / "fig2_ordered_vs_disordered.pdf", show)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Parity plot (computed vs experimental)
# ─────────────────────────────────────────────────────────────────────────────


def plot_parity(
    accuracy: dict,
    target_property: str = "voltage",
    output_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> pathlib.Path:
    """Parity plot: computed vs experimental with two series (ordered, disordered).

    Args:
        accuracy:        Output of eval_accuracy.compute_accuracy_metrics().
        target_property: Which property to plot.
        output_path:     Where to save PDF.
        show:            If True, call plt.show().
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": _SERIF})

    exp_vals, ord_vals, dis_vals, dopant_labels = [], [], [], []
    for row in accuracy.get("per_dopant", []):
        p_data = row["properties"].get(target_property)
        if not p_data:
            continue
        ev = p_data.get("experimental")
        ov = p_data.get("ordered")
        dv = p_data.get("disordered")
        if ev is not None:
            exp_vals.append(ev)
            ord_vals.append(ov)
            dis_vals.append(dv)
            dopant_labels.append(row["dopant"])

    if not exp_vals:
        logger.warning("No data available for parity plot ('%s').", target_property)
        return _FIGURES_DIR / "fig3_parity.pdf"

    all_vals = [v for v in exp_vals + ord_vals + dis_vals if v is not None]
    vmin, vmax = min(all_vals) * 0.97, max(all_vals) * 1.03
    ref = np.linspace(vmin, vmax, 100)

    fig, ax = plt.subplots(figsize=(_SINGLE_COL_W, _SINGLE_COL_W))

    ax.plot(ref, ref, "k--", linewidth=0.8, label="y = x", zorder=0)

    for ev, ov, dv, label in zip(exp_vals, ord_vals, dis_vals, dopant_labels):
        if ov is not None:
            ax.scatter(ev, ov, marker="x", s=50, color=_COLORBLIND_PALETTE[0],
                       linewidths=1.5, zorder=3)
        if dv is not None:
            ax.scatter(ev, dv, marker="o", s=40, color=_COLORBLIND_PALETTE[1],
                       zorder=3, alpha=0.85)
        # Label points
        ref_v = dv if dv is not None else ov
        if ref_v is not None:
            ax.annotate(label, (ev, ref_v), textcoords="offset points",
                        xytext=(4, 2), fontsize=6.5, color="#444444")

    mae_ord = accuracy.get("mae_ordered", {}).get(target_property)
    mae_dis = accuracy.get("mae_disordered", {}).get(target_property)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="x", color=_COLORBLIND_PALETTE[0], linewidth=0,
               markersize=7, label=f"Ordered  (MAE={mae_ord:.3f})" if mae_ord else "Ordered"),
        Line2D([0], [0], marker="o", color=_COLORBLIND_PALETTE[1], linewidth=0,
               markersize=6, label=f"Disordered (MAE={mae_dis:.3f})" if mae_dis else "Disordered"),
        Line2D([0], [0], linestyle="--", color="k", linewidth=0.8, label="y = x"),
    ]

    units = {"voltage": "V", "li_ni_exchange": "eV", "formation_energy": "eV/atom"}.get(
        target_property, ""
    )
    ax.set_xlabel(f"Experimental {target_property} ({units})", fontsize=9)
    ax.set_ylabel(f"Computed {target_property} ({units})", fontsize=9)
    ax.set_title(f"Parity Plot: {target_property}", fontsize=10)
    ax.legend(handles=legend_elements, fontsize=7.5, loc="upper left")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = _save_fig(fig, output_path or _FIGURES_DIR / "fig3_parity.pdf", show)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Disorder sensitivity heatmap
# ─────────────────────────────────────────────────────────────────────────────


def plot_disorder_heatmap(
    rq2_results: dict,
    output_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> pathlib.Path:
    """Heatmap of disorder sensitivity %: dopants (rows) × properties (cols).

    Args:
        rq2_results: Output of eval_disorder.run_disorder_evaluation().
        output_path: Where to save PDF.
        show:        If True, call plt.show().
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    plt.rcParams.update({"font.family": _SERIF})

    rows = rq2_results.get("dopant_results", [])
    target_properties = rq2_results.get("target_properties", [])

    dopants = [r["dopant"] for r in rows]
    data = np.full((len(dopants), len(target_properties)), np.nan)

    for i, row in enumerate(rows):
        for j, prop in enumerate(target_properties):
            s = row.get("disorder_sensitivity", {}).get(prop)
            if s is not None:
                data[i, j] = s * 100.0  # convert to %

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL_W * 0.7, max(2.5, len(dopants) * 0.45 + 0.5)))

    cmap = plt.cm.RdYlGn_r
    cmap.set_bad("whitesmoke")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=30)

    ax.set_xticks(range(len(target_properties)))
    ax.set_xticklabels(target_properties, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(dopants)))
    ax.set_yticklabels(dopants, fontsize=9)

    # Annotate cells
    for i in range(len(dopants)):
        for j in range(len(target_properties)):
            v = data[i, j]
            if not np.isnan(v):
                text_color = "white" if v > 20 else "black"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label="Disorder sensitivity (%)", shrink=0.8)
    ax.set_title("Disorder Sensitivity: |disordered − ordered| / |ordered|", fontsize=10, pad=8)

    plt.tight_layout()
    out = _save_fig(fig, output_path or _FIGURES_DIR / "fig4_disorder_heatmap.pdf", show)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: SQS variance box/strip plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_sqs_variance(
    rq2_results: dict,
    target_property: str | None = None,
    output_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> pathlib.Path:
    """Box plot showing spread of property values across SQS realisations per dopant.

    Args:
        rq2_results:    Output of eval_disorder.run_disorder_evaluation().
        target_property: Which property to show. Defaults to primary property.
        output_path:    Where to save PDF.
        show:           If True, call plt.show().
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": _SERIF})

    target_properties = rq2_results.get("target_properties", [])
    if target_property is None:
        target_property = target_properties[0] if target_properties else "voltage"

    rows = rq2_results.get("dopant_results", [])
    data_by_dopant = {}
    for row in rows:
        vals = [r.get(target_property) for r in row.get("sqs_realisations", [])
                if r.get(target_property) is not None]
        if vals:
            data_by_dopant[row["dopant"]] = vals

    if not data_by_dopant:
        logger.warning("No SQS realisation data found for '%s' — skipping Figure 5.", target_property)
        return _FIGURES_DIR / "fig5_sqs_variance.pdf"

    dopants = list(data_by_dopant.keys())
    values_list = [data_by_dopant[d] for d in dopants]

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL_W * 0.8, 3.5))

    bp = ax.boxplot(
        values_list,
        labels=dopants,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#333333", "linewidth": 1.5},
        boxprops={"facecolor": _COLORBLIND_PALETTE[0], "alpha": 0.6},
        whiskerprops={"linewidth": 1},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )

    # Overlay individual points
    import numpy as np
    for i, vals in enumerate(values_list, start=1):
        ax.scatter(
            [i + np.random.uniform(-0.15, 0.15) for _ in vals],
            vals,
            s=18, alpha=0.7, color=_COLORBLIND_PALETTE[1], zorder=5,
        )

    units = {"voltage": "V", "li_ni_exchange": "eV", "formation_energy": "eV/atom"}.get(
        target_property, ""
    )
    ax.set_xlabel("Dopant", fontsize=9)
    ax.set_ylabel(f"{target_property} ({units})", fontsize=9)
    ax.set_title(f"SQS Realisation Variance: {target_property}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = _save_fig(fig, output_path or _FIGURES_DIR / "fig5_sqs_variance.pdf", show)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all figures
# ─────────────────────────────────────────────────────────────────────────────


def plot_sqs_reliability(
    rq2_results: dict,
    output_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> pathlib.Path:
    """Fig 6: SQS realisation spread vs dopant-to-dopant resolution.

    Shows every individual SQS voltage as a dot, sorted by disordered mean.
    Colour-codes by convergence (n=5 full / n=3-4 partial / n=2 unreliable).
    Adds a shaded band equal to ±1 SQS std per dopant to make clear that
    adjacent ranks overlap — i.e. single-structure disorder calculations
    cannot resolve the ordering within the band.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    plt.rcParams.update({"font.family": _SERIF})

    rows = rq2_results.get("dopant_results", [])

    data = []
    for r in rows:
        vs = [s.get("voltage") for s in r.get("sqs_realisations", [])
              if s.get("voltage") is not None]
        mean_v = r["disordered_mean"].get("voltage")
        if vs and mean_v is not None:
            data.append({
                "dopant": r["dopant"],
                "mean": mean_v,
                "vals": vs,
                "n": r["n_converged"],
                "std": r["disordered_std"].get("voltage", 0.0),
            })

    if not data:
        logger.warning("No SQS realisation voltage data — skipping Figure 6.")
        return _FIGURES_DIR / "fig6_sqs_reliability.pdf"

    data.sort(key=lambda x: -x["mean"])  # highest voltage first

    dopants      = [d["dopant"] for d in data]
    x_pos        = list(range(len(data)))
    means        = [d["mean"] for d in data]
    total_spread = max(means) - min(means)
    mean_std     = float(np.mean([d["std"] for d in data]))

    def _colour(n):
        if n >= 5: return _COLORBLIND_PALETTE[0]
        if n >= 3: return _COLORBLIND_PALETTE[4]
        return _COLORBLIND_PALETTE[3]

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL_W, 3.8))

    # Shaded ±1 std band per dopant
    for xi, d in zip(x_pos, data):
        ax.fill_between(
            [xi - 0.4, xi + 0.4],
            d["mean"] - d["std"], d["mean"] + d["std"],
            alpha=0.18, color=_colour(d["n"]), linewidth=0,
        )

    # Individual SQS realisations
    rng = np.random.default_rng(42)
    for xi, d in zip(x_pos, data):
        jitter = rng.uniform(-0.18, 0.18, len(d["vals"]))
        ax.scatter(
            [xi + j for j in jitter], d["vals"],
            s=22, alpha=0.8, color=_colour(d["n"]), zorder=4,
        )

    # Mean markers
    ax.scatter(x_pos, means, s=45, marker="D", color="#222222",
               zorder=5, label="Disordered mean")

    ax.annotate(
        f"Mean within-dopant σ = {mean_std:.3f} V\n"
        f"Total dopant spread   = {total_spread:.3f} V\n"
        f"σ / spread = {mean_std/total_spread:.0%}",
        xy=(0.02, 0.04), xycoords="axes fraction",
        fontsize=7.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dopants, fontsize=7.5, rotation=45, ha="right")
    ax.set_ylabel("Voltage (V)", fontsize=9)
    ax.set_title(
        "SQS Realisation Spread vs Dopant-to-Dopant Resolution\n"
        "(shaded = ±1σ; overlapping bands indicate indistinguishable ranks)",
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=_COLORBLIND_PALETTE[0], alpha=0.7, label="n = 5 (full)"),
        mpatches.Patch(color=_COLORBLIND_PALETTE[4], alpha=0.7, label="n = 3–4 (partial)"),
        mpatches.Patch(color=_COLORBLIND_PALETTE[3], alpha=0.7, label="n = 2 (unreliable)"),
        plt.scatter([], [], s=45, marker="D", color="#222222", label="Mean"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper right")

    plt.tight_layout()
    out = _save_fig(fig, output_path or _FIGURES_DIR / "fig6_sqs_reliability.pdf", show)
    plt.close(fig)
    return out


def save_all_figures(
    rq1_data: dict | None = None,
    rq2_data: dict | None = None,
    accuracy_data: dict | None = None,
    output_dir: str | pathlib.Path = _FIGURES_DIR,
    show: bool = False,
) -> list[pathlib.Path]:
    """Generate and save all 5 publication figures.

    Args:
        rq1_data:      RQ1 report dict (for funnel diagram).
        rq2_data:      RQ2 results dict (for Figs 2, 4, 5).
        accuracy_data: RQ3 accuracy dict (for Fig 3).
        output_dir:    Directory to write PDFs.
        show:          Display each figure interactively.

    Returns:
        List of saved figure paths.
    """
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if rq1_data:
        fc = rq1_data.get("funnel_counts", {})
        p = plot_funnel_diagram(fc, output_path=out_dir / "fig1_funnel.pdf", show=show)
        saved.append(p)
        logger.info("Figure 1 saved: %s", p)

    if rq2_data:
        p = plot_ordered_vs_disordered(rq2_data, output_path=out_dir / "fig2_ordered_vs_disordered.pdf", show=show)
        saved.append(p)
        logger.info("Figure 2 saved: %s", p)

        p = plot_disorder_heatmap(rq2_data, output_path=out_dir / "fig4_disorder_heatmap.pdf", show=show)
        saved.append(p)
        logger.info("Figure 4 saved: %s", p)

        p = plot_sqs_variance(rq2_data, output_path=out_dir / "fig5_sqs_variance.pdf", show=show)
        saved.append(p)
        logger.info("Figure 5 saved: %s", p)

        p = plot_sqs_reliability(rq2_data, output_path=out_dir / "fig6_sqs_reliability.pdf", show=show)
        saved.append(p)
        logger.info("Figure 6 saved: %s", p)

    if accuracy_data and rq2_data:
        # Use property with best experimental coverage
        props = list(accuracy_data.get("mae_ordered", {}).keys())
        best_prop = props[0] if props else "voltage"
        p = plot_parity(accuracy_data, target_property=best_prop,
                        output_path=out_dir / "fig3_parity.pdf", show=show)
        saved.append(p)
        logger.info("Figure 3 saved: %s", p)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_fig(fig, path: pathlib.Path | str, show: bool = False) -> pathlib.Path:
    """Save figure to path (creates parent dirs). Returns absolute path."""
    import matplotlib.pyplot as plt

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return path


def _rank_list(values: list[float], reverse: bool = False) -> list[int]:
    """Return 1-based ranks for a list of values (1 = best)."""
    import numpy as np

    arr = np.array(values)
    if reverse:
        order = np.argsort(-arr)
    else:
        order = np.argsort(arr)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--rq1", metavar="FILE", help="RQ1 report JSON.")
    parser.add_argument("--rq2", metavar="FILE", help="RQ2 disorder results JSON.")
    parser.add_argument("--accuracy", metavar="FILE", help="RQ3 accuracy results JSON.")
    parser.add_argument("--output", default=str(_FIGURES_DIR), metavar="DIR")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = parser.parse_args()

    rq1_data = json.load(open(args.rq1)) if args.rq1 else None
    rq2_data = json.load(open(args.rq2)) if args.rq2 else None
    accuracy_data = json.load(open(args.accuracy)) if args.accuracy else None

    saved = save_all_figures(
        rq1_data=rq1_data,
        rq2_data=rq2_data,
        accuracy_data=accuracy_data,
        output_dir=args.output,
        show=args.show,
    )

    print(f"Saved {len(saved)} figures to {args.output}:")
    for p in saved:
        print(f"  {p}")
