"""disorder-screening CLI.

Usage::

    python -m disorder_screening <command> [args]
    disorder-screening <command> [args]   # after pip install

Commands:
    run         Run the full pipeline end-to-end
    prune       Run the chemical pruning funnel (Stages 1–3) and output JSON
    single      Run Stage 5 for one dopant, bypassing pruning
    compare     Compare simulation results from multiple runs
    evaluate    Evaluate pruning recall/precision against ground truth
    list-runs   List past pipeline runs from the local database
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger("disorder_screening")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline, reading material config from pipeline.yaml or CLI."""
    import pathlib

    import yaml

    try:
        config_path = args.config or None
        # Load config to get material defaults
        if config_path is None:
            config_path = pathlib.Path(__file__).parent / "config" / "pipeline.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as exc:
        print(f"Error: config file not found: {exc}", file=sys.stderr)
        return 1

    material = config.get("material", {})
    formula = args.formula or material.get("formula")
    site = args.site or material.get("target_site")
    os_val = args.os or material.get("target_os")
    cn = args.cn or material.get("target_cn", 6)

    if not formula or not site or os_val is None:
        print(
            "Error: --formula, --site, and --os are required (or set material: section in pipeline.yaml).",
            file=sys.stderr,
        )
        return 1

    report_dir = args.report_dir
    if report_dir:
        if "output" not in config.get("pipeline", {}):
            config.setdefault("pipeline", {})["output"] = {}
        config["pipeline"]["output"]["report_dir"] = report_dir

    try:
        from graph.entry_points import run_full_pipeline

        # Structure must be loaded separately; for CLI use we attempt to load
        # from a CIF/POSCAR file if --structure is given, else pass None
        # (stage5_simulate_node handles missing parent_structure gracefully).
        parent_structure = None
        if getattr(args, "structure", None):
            try:
                from pymatgen.core import Structure
                parent_structure = Structure.from_file(args.structure)
                logger.info("Loaded parent structure from %s.", args.structure)
            except Exception as exc:
                print(f"Error loading structure file: {exc}", file=sys.stderr)
                return 1

        state = run_full_pipeline(
            parent_formula=formula,
            parent_structure=parent_structure,
            target_site_species=site,
            target_oxidation_state=int(os_val),
            target_coordination_number=int(cn),
            config_path=str(config_path),
        )
        report_path = state.get("report_path")
        if report_path:
            print(f"Report written to: {report_path}")
        else:
            print("Pipeline completed. No report path in state.")
        return 0

    except Exception as exc:
        logger.error("run failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_prune(args: argparse.Namespace) -> int:
    """Run Stages 1–3 (chemical pruning) and output JSON."""
    try:
        from graph.entry_points import run_stages_1_3

        state = run_stages_1_3(
            parent_formula=args.formula,
            target_site_species=args.site,
            target_oxidation_state=int(args.os),
            target_coordination_number=int(args.cn),
            config_path=args.config or None,
        )

        output = {
            "parent_formula": args.formula,
            "target_site": args.site,
            "stage1_candidates": state.get("stage1_candidates", []),
            "stage2_candidates": state.get("stage2_candidates", []),
            "stage3_candidates": state.get("stage3_candidates", []),
            "execution_log": state.get("execution_log", []),
        }
        serialised = json.dumps(output, indent=2, default=str)

        if args.output:
            import pathlib
            out = pathlib.Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(serialised)
            print(f"Pruning results written to: {args.output}")
        else:
            print(serialised)
        return 0

    except Exception as exc:
        logger.error("prune failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_single(args: argparse.Namespace) -> int:
    """Run Stage 5 for a single dopant, bypassing Stages 1–3."""
    try:
        from graph.entry_points import run_single_dopant

        concentrations = None
        if args.conc:
            concentrations = [float(c) for c in args.conc]

        # Structure is optional; single runs without it still compute mock properties
        parent_structure = None
        if getattr(args, "structure", None):
            try:
                from pymatgen.core import Structure
                parent_structure = Structure.from_file(args.structure)
            except Exception as exc:
                print(f"Error loading structure file: {exc}", file=sys.stderr)
                return 1

        result = run_single_dopant(
            parent_formula=args.formula,
            parent_structure=parent_structure,
            dopant_element=args.dopant,
            dopant_oxidation_state=int(args.dopant_os) if args.dopant_os else 0,
            target_site_species=args.site,
            concentrations=concentrations,
            config_path=args.config or None,
        )

        n = len(result.get("simulation_results", []))
        print(f"Completed {n} relaxation(s) for dopant {args.dopant!r}.")
        print(f"Run ID: {result.get('run_id')}")
        return 0

    except Exception as exc:
        logger.error("single failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare simulation results from multiple runs."""
    try:
        from graph.entry_points import run_comparison

        report = run_comparison(
            run_ids=args.runs,
            db_path=args.db or None,
            config_path=args.config or None,
        )

        # Serialise ComparisonReport to JSON
        if hasattr(report, "__dict__"):
            data = report.__dict__
        else:
            data = dict(report)

        serialised = json.dumps(data, indent=2, default=str)

        if args.output:
            import pathlib
            out = pathlib.Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(serialised)
            print(f"Comparison report written to: {args.output}")
        else:
            print(serialised)
        return 0

    except Exception as exc:
        logger.error("compare failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate pruning pipeline recall and precision against ground truth."""
    try:
        import pathlib

        import yaml

        config_path = args.config or None
        if config_path is None:
            default_cfg = pathlib.Path(__file__).parent / "config" / "pipeline.yaml"
            if default_cfg.exists():
                config_path = str(default_cfg)

        # Read material defaults from config
        formula = site = os_val = None
        if config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                material = cfg.get("material", {})
                formula = material.get("formula", "LiNi0.8Mn0.1Co0.1O2")
                site = material.get("target_site", "Co")
                os_val = material.get("target_os", 3)
            except Exception:
                formula, site, os_val = "LiNi0.8Mn0.1Co0.1O2", "Co", 3

        formula = formula or "LiNi0.8Mn0.1Co0.1O2"
        site = site or "Co"
        os_val = int(os_val or 3)

        from evaluation.eval_pruning import evaluate_pruning, print_metrics
        from graph.entry_points import run_stages_1_3

        logger.info("Running stages 1–3 for %s (site=%s, OS=%d)…", formula, site, os_val)
        state = run_stages_1_3(
            parent_formula=formula,
            target_site_species=site,
            target_oxidation_state=os_val,
            config_path=config_path,
        )

        ground_truth_path = args.ground_truth or None
        for stage_key, label in [
            ("stage1_candidates", "Stage 1 — SMACT"),
            ("stage2_candidates", "Stage 2 — Radius"),
            ("stage3_candidates", "Stage 3 — Substitution"),
        ]:
            candidates = state.get(stage_key, [])
            m = evaluate_pruning(candidates, ground_truth_path=ground_truth_path, stage_label=label)
            print_metrics(m)

        return 0

    except Exception as exc:
        logger.error("evaluate failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_list_runs(args: argparse.Namespace) -> int:
    """List past pipeline runs from the local SQLite database."""
    try:
        import pathlib

        import yaml

        config_path = args.config or None
        if config_path is None:
            default_cfg = pathlib.Path(__file__).parent / "config" / "pipeline.yaml"
            if default_cfg.exists():
                config_path = str(default_cfg)

        db_path = args.db
        if not db_path and config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                db_path = (
                    cfg.get("pipeline", {})
                    .get("database", {})
                    .get("local", {})
                    .get("path", "data/results.db")
                )
            except Exception:
                db_path = "data/results.db"

        db_path = db_path or "data/results.db"

        from db.local_store import LocalStore

        store = LocalStore(db_path)
        try:
            # Query distinct run summaries
            cur = store._con.execute(
                """
                SELECT run_id,
                       parent_formula,
                       target_site_species,
                       COUNT(*) AS n_simulations,
                       MIN(created_at) AS first_created
                FROM simulations
                {where}
                GROUP BY run_id, parent_formula, target_site_species
                ORDER BY first_created DESC
                """.format(
                    where="WHERE parent_formula = ?" if args.parent else ""
                ),
                (args.parent,) if args.parent else (),
            )
            rows = cur.fetchall()
        finally:
            store.close()

        if not rows:
            print("No runs found.")
            return 0

        # Print table
        header = f"{'RUN ID':<38} {'PARENT':<24} {'SITE':<6} {'N SIMS':>6}  {'DATE':<20}"
        print(header)
        print("-" * len(header))
        for row in rows:
            run_id = row[0] or ""
            parent = row[1] or ""
            site = row[2] or ""
            n = row[3] or 0
            date = (row[4] or "")[:19]
            print(f"{run_id:<38} {parent:<24} {site:<6} {n:>6}  {date:<20}")

        return 0

    except Exception as exc:
        logger.error("list-runs failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="disorder-screening",
        description="Disorder-aware substitutional dopant screening for battery cathodes.",
    )
    parser.add_argument(
        "--version", action="version", version="disorder-screening 0.1.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run the full pipeline end-to-end.")
    p_run.add_argument("--formula", metavar="FORMULA", help="Parent material formula (e.g. LiCoO2).")
    p_run.add_argument("--site", metavar="ELEMENT", help="Target substitution site (e.g. Co).")
    p_run.add_argument("--os", metavar="INT", type=int, help="Target oxidation state (e.g. 3).")
    p_run.add_argument("--cn", metavar="INT", type=int, default=6, help="Target coordination number [default: 6].")
    p_run.add_argument("--structure", metavar="FILE", help="Path to CIF/POSCAR parent structure file.")
    p_run.add_argument("--config", metavar="FILE", help="Path to pipeline.yaml [default: config/pipeline.yaml].")
    p_run.add_argument("--report-dir", metavar="DIR", help="Directory for the output report.")

    # ── prune ─────────────────────────────────────────────────────────────────
    p_prune = sub.add_parser("prune", help="Run pruning pipeline (Stages 1–3) and output JSON.")
    p_prune.add_argument("--formula", required=True, metavar="FORMULA", help="Parent material formula.")
    p_prune.add_argument("--site", required=True, metavar="ELEMENT", help="Target substitution site.")
    p_prune.add_argument("--os", required=True, type=int, metavar="INT", help="Target oxidation state.")
    p_prune.add_argument("--cn", type=int, default=6, metavar="INT", help="Coordination number [default: 6].")
    p_prune.add_argument("--config", metavar="FILE", help="Path to pipeline.yaml.")
    p_prune.add_argument("--output", "-o", metavar="FILE", help="Write JSON to FILE instead of stdout.")

    # ── single ────────────────────────────────────────────────────────────────
    p_single = sub.add_parser("single", help="Run Stage 5 for one dopant (skip pruning).")
    p_single.add_argument("--formula", required=True, metavar="FORMULA", help="Parent material formula.")
    p_single.add_argument("--site", required=True, metavar="ELEMENT", help="Target substitution site.")
    p_single.add_argument("--os", required=True, type=int, metavar="INT", help="Target oxidation state.")
    p_single.add_argument("--dopant", required=True, metavar="ELEMENT", help="Dopant element symbol.")
    p_single.add_argument("--dopant-os", type=int, metavar="INT", help="Dopant oxidation state.")
    p_single.add_argument("--conc", nargs="+", metavar="FLOAT", help="Concentrations to evaluate (e.g. 0.05 0.10).")
    p_single.add_argument("--structure", metavar="FILE", help="Path to parent structure CIF/POSCAR.")
    p_single.add_argument("--config", metavar="FILE", help="Path to pipeline.yaml.")

    # ── compare ───────────────────────────────────────────────────────────────
    p_compare = sub.add_parser("compare", help="Compare results from multiple pipeline runs.")
    p_compare.add_argument("--runs", required=True, nargs="+", metavar="RUN_ID", help="Run IDs to compare.")
    p_compare.add_argument("--db", metavar="FILE", help="Path to SQLite results database.")
    p_compare.add_argument("--output", "-o", metavar="FILE", help="Write JSON output to FILE.")
    p_compare.add_argument("--config", metavar="FILE", help="Path to pipeline.yaml.")

    # ── evaluate ──────────────────────────────────────────────────────────────
    p_evaluate = sub.add_parser("evaluate", help="Evaluate pruning recall/precision vs ground truth.")
    p_evaluate.add_argument("--ground-truth", metavar="FILE", help="Path to ground truth JSON.")
    p_evaluate.add_argument("--config", metavar="FILE", help="Path to pipeline.yaml.")

    # ── list-runs ─────────────────────────────────────────────────────────────
    p_list = sub.add_parser("list-runs", help="List past pipeline runs from the database.")
    p_list.add_argument("--db", metavar="FILE", help="Path to SQLite results database.")
    p_list.add_argument("--parent", metavar="FORMULA", help="Filter by parent formula.")
    p_list.add_argument("--config", metavar="FILE", help="Path to pipeline.yaml.")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


_HANDLERS = {
    "run": _cmd_run,
    "prune": _cmd_prune,
    "single": _cmd_single,
    "compare": _cmd_compare,
    "evaluate": _cmd_evaluate,
    "list-runs": _cmd_list_runs,
}


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
