"""
Report generation for the disorder-screening pipeline.

Takes a ``ranked_report`` dict (from PipelineState) and the full
``PipelineState`` dict, renders a Markdown report via Jinja2, and writes
it to ``reports/{run_id}_screening_report.md``.

Entry points
------------
generate_report(ranked_report, state, output_path=None) → pathlib.Path
build_template_context(ranked_report, state) → dict
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from datetime import datetime
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_TEMPLATE_NAME = "screening_report.md.j2"
_REPORTS_DIR = pathlib.Path(__file__).parent.parent / "reports"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(
    ranked_report: dict,
    state: dict,
    output_path: Optional[str | pathlib.Path] = None,
) -> pathlib.Path:
    """Render the screening report to a Markdown file.

    Parameters
    ----------
    ranked_report:
        The dict stored in ``PipelineState["ranked_report"]`` by
        ``rank_and_report_node``.
    state:
        Full ``PipelineState`` dict.
    output_path:
        Where to write the report. Defaults to
        ``reports/{run_id}_screening_report.md``.

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        raise ImportError(
            "Jinja2 is required for report generation. "
            "Install with: pip install jinja2"
        )

    ctx = build_template_context(ranked_report, state)

    template_dir = pathlib.Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["fmt_float"] = _fmt_float
    env.filters["fmt_pct"] = _fmt_pct

    template = env.get_template(_TEMPLATE_NAME)
    rendered = template.render(**ctx)

    if output_path is None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        run_id = state.get("run_id") or ranked_report.get("run_id", "unknown")
        output_path = _REPORTS_DIR / f"{run_id}_screening_report.md"

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")

    logger.info("Report written to %s", output_path)
    return output_path


def build_template_context(ranked_report: dict, state: dict) -> dict:
    """Build the flat context dict used to render the Jinja2 template.

    Parameters
    ----------
    ranked_report:
        Dict from ``PipelineState["ranked_report"]``.
    state:
        Full ``PipelineState`` dict.

    Returns
    -------
    dict
        All variables referenced in the template.
    """
    cfg = state.get("config", {}).get("pipeline", {})
    sim_cfg = cfg.get("stage5_simulation", {})
    target_properties: list[str] = (
        state.get("target_properties")
        or list(cfg.get("property_weights", {}).keys())
        or []
    )

    ctx: dict = {}

    # ── 1. Screening summary ──────────────────────────────────────────────
    ctx["parent_formula"] = (
        state.get("parent_formula")
        or ranked_report.get("parent_formula", "N/A")
    )
    ctx["target_site"] = (
        state.get("target_site_species")
        or ranked_report.get("target_site", "N/A")
    )
    ctx["target_oxidation_state"] = state.get("target_oxidation_state", "N/A")
    ctx["target_coordination_number"] = state.get("target_coordination_number", "N/A")
    ctx["run_id"] = state.get("run_id", "N/A")
    ctx["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctx["mlip_name"] = sim_cfg.get("potential", "N/A")
    ctx["device"] = sim_cfg.get("device", "auto")
    ctx["concentrations"] = sim_cfg.get("concentrations", [])
    ctx["supercell"] = sim_cfg.get("supercell", [])
    ctx["n_sqs_realisations"] = sim_cfg.get("n_sqs_realisations", "N/A")
    ctx["primary_property"] = ranked_report.get("primary_property", "N/A")
    ctx["candidates_simulated"] = ranked_report.get("candidates_simulated", 0)
    ctx["target_properties"] = target_properties

    # ── 2. Pruning funnel ─────────────────────────────────────────────────
    ctx["funnel_rows"] = _build_funnel_rows(state, cfg)

    # ── 3. Simulation results ─────────────────────────────────────────────
    ctx["sim_rows"] = _build_simulation_summary(ranked_report)

    # ── 4. Property predictions (disordered) ──────────────────────────────
    ctx["prop_rows"] = _build_property_table(ranked_report, target_properties)

    # ── 5. Ordered vs disordered comparison ──────────────────────────────
    ctx["comparison_rows"] = _build_comparison_table(ranked_report, target_properties)

    # ── 6. Ranking ────────────────────────────────────────────────────────
    ctx["ranking_rows"] = _build_ranking_table(ranked_report, target_properties)
    ctx["spearman_rows"] = [
        {
            "property": prop,
            "rho": stats.get("rho"),
            "pvalue": stats.get("pvalue"),
            "n": stats.get("n"),
        }
        for prop, stats in (ranked_report.get("spearman_rho") or {}).items()
    ]

    # ── 7. Recommendations ────────────────────────────────────────────────
    recommended = ranked_report.get("recommended") or []
    ctx["recommended"] = recommended
    rankings = ranked_report.get("rankings") or []
    ctx["top_dopant_details"] = [
        r for r in rankings if r.get("dopant") in recommended
    ]

    # ── 8. Warnings ───────────────────────────────────────────────────────
    ctx["warnings"] = ranked_report.get("warnings") or []

    # ── 9. Configuration YAML ─────────────────────────────────────────────
    raw_cfg = state.get("config", {})
    try:
        ctx["config_yaml"] = yaml.dump(raw_cfg, default_flow_style=False).strip()
    except Exception:
        ctx["config_yaml"] = str(raw_cfg)

    # ── 10. Metadata ──────────────────────────────────────────────────────
    ctx["mlip_version"] = _get_mlip_version(ctx["mlip_name"])
    raw_bytes = json.dumps(raw_cfg, sort_keys=True, default=str).encode()
    ctx["config_hash"] = hashlib.sha256(raw_bytes).hexdigest()[:12]
    ctx["stage_timings"] = {}  # populated by graph nodes when available

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Section helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_funnel_rows(state: dict, cfg: dict) -> list[dict]:
    stage1 = state.get("stage1_candidates") or []
    stage2 = state.get("stage2_candidates") or []
    stage3 = state.get("stage3_candidates") or []
    stage4 = state.get("stage4_candidates")

    mismatch = cfg.get("stage2_radius", {}).get("mismatch_threshold", 0.35)
    prob = cfg.get("stage3_substitution", {}).get("probability_threshold", 0.001)

    rows = [
        {
            "stage": "Stage 1 (SMACT)",
            "in": state.get("stage1_os_combinations", "?"),
            "out": len(stage1),
            "threshold": "Charge neutrality + EN ordering",
        },
        {
            "stage": "Stage 2 (Radius)",
            "in": len(stage1),
            "out": len(stage2),
            "threshold": f"Radius mismatch ≤ {mismatch:.0%}",
        },
        {
            "stage": "Stage 3 (Sub. Prob.)",
            "in": len(stage2),
            "out": len(stage3),
            "threshold": f"Hautier–Ceder P ≥ {prob}",
        },
    ]
    if stage4 is not None:
        rows.append(
            {
                "stage": "Stage 4 (ML pre-screen)",
                "in": len(stage3),
                "out": len(stage4),
                "threshold": "ML formation energy filter",
            }
        )
    return rows


def _build_simulation_summary(ranked_report: dict) -> list[dict]:
    """One row per dopant: n_converged + n_aborted."""
    rows = []
    for r in ranked_report.get("rankings") or []:
        props = r.get("properties") or {}
        # n from any property gives SQS count; n_converged is on DopantStats
        n_sqs = max(
            (v.get("n", 0) for v in props.values() if isinstance(v, dict)),
            default=0,
        )
        n_converged = r.get("n_converged", 0)
        rows.append(
            {
                "dopant": r.get("dopant", "?"),
                "n_converged": n_converged,
                "n_aborted": max(0, n_sqs - n_converged) if n_sqs else "?",
            }
        )
    return rows


def _build_property_table(ranked_report: dict, target_properties: list[str]) -> list[dict]:
    rows = []
    for r in ranked_report.get("rankings") or []:
        props = r.get("properties") or {}
        row: dict = {"dopant": r.get("dopant", "?")}
        for prop in target_properties:
            row[prop] = props.get(prop) or {}
        rows.append(row)
    return rows


def _build_comparison_table(
    ranked_report: dict, target_properties: list[str]
) -> list[dict]:
    rows = []
    for r in ranked_report.get("rankings") or []:
        dopant = r.get("dopant", "?")
        props = r.get("properties") or {}
        ord_props = r.get("ordered_properties") or {}
        sensitivity = r.get("disorder_sensitivity") or {}
        for prop in target_properties:
            dis_stats = props.get(prop) or {}
            dis_mean = dis_stats.get("mean")
            ord_val = ord_props.get(prop)
            if dis_mean is not None or ord_val is not None:
                rows.append(
                    {
                        "dopant": dopant,
                        "property": prop,
                        "ordered": ord_val,
                        "disordered": dis_mean,
                        "sensitivity": sensitivity.get(prop),
                    }
                )
    return rows


def _build_ranking_table(ranked_report: dict, target_properties: list[str]) -> list[dict]:
    rows = []
    for r in ranked_report.get("rankings") or []:
        rows.append(
            {
                "dopant": r.get("dopant", "?"),
                "ranks": r.get("rank_by_property") or {},
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Jinja2 custom filters
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_float(value) -> str:
    """Format a float to 3 decimal places; None → "N/A"."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(value) -> str:
    """Format a fractional value as percentage with 1 decimal; None → "N/A"."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _get_mlip_version(mlip_name: str) -> str:
    try:
        if "mace" in mlip_name.lower():
            import mace
            return mace.__version__
        if "mattersim" in mlip_name.lower():
            import mattersim
            return mattersim.__version__
    except Exception:
        pass
    return "unknown"
