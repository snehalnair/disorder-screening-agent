"""
Stage 3 threshold sweep for calibrating the Hautier-Ceder probability cutoff.

Runs Stages 1–3 at a range of Stage 3 ``probability_threshold`` values and
reports recall against the ground truth at each level.  The goal is to find
the highest threshold that still maintains ≥90% recall, minimising the number
of candidates passed to expensive Stage 5 simulations.

Usage (standalone):
    python -m evaluation.threshold_sweep

Or import ``sweep_stage3_threshold`` directly.
"""

from __future__ import annotations

import copy
import sys
import pathlib
from dataclasses import dataclass

from evaluation.eval_pruning import evaluate_pruning


@dataclass
class SweepPoint:
    """Result for one threshold value."""
    threshold: float
    n_stage3_survivors: int
    recall: float
    precision: float | None
    missed_dopants: list[str]

    def __str__(self) -> str:
        missed = ", ".join(self.missed_dopants) if self.missed_dopants else "—"
        prec = f"{self.precision:.1%}" if self.precision is not None else "N/A"
        return (
            f"  {self.threshold:<10.5f}  {self.n_stage3_survivors:<12}  "
            f"{self.recall:<8.1%}  {prec:<10}  {missed}"
        )


def sweep_stage3_threshold(
    parent_formula: str,
    target_species: str,
    target_os: int,
    target_cn: int = 6,
    ground_truth_path=None,
    gt_classes: list[str] | None = None,
    thresholds: list[float] | None = None,
    config_path=None,
) -> list[SweepPoint]:
    """Run Stages 1–3 at each Stage 3 threshold value and report recall.

    Args:
        parent_formula:    Host material formula (e.g. ``"LiNi0.8Mn0.1Co0.1O2"``).
        target_species:    Element being substituted (e.g. ``"Co"``).
        target_os:         Oxidation state of target site (e.g. ``3``).
        target_cn:         Coordination number (default ``6``).
        ground_truth_path: Path to ground truth JSON; defaults to NMC file.
        gt_classes:        Ground truth classes counted as positives.
        thresholds:        List of Stage 3 probability thresholds to sweep.
        config_path:       Path to pipeline.yaml; defaults to config/pipeline.yaml.

    Returns:
        List of ``SweepPoint`` dataclasses, one per threshold value.
    """
    from graph.entry_points import _load_config, run_stages_1_3

    if thresholds is None:
        thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

    if gt_classes is None:
        gt_classes = ["confirmed_successful"]

    base_config = _load_config(config_path)
    results: list[SweepPoint] = []

    for thresh in thresholds:
        # Deep-copy config so each run is independent
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("pipeline", {}).setdefault("stage3_substitution", {})
        cfg["pipeline"]["stage3_substitution"]["probability_threshold"] = thresh

        result = run_stages_1_3(
            parent_formula=parent_formula,
            target_site_species=target_species,
            target_oxidation_state=target_os,
            target_coordination_number=target_cn,
            config_path=None,   # We inject config directly below
        )
        # Override config used in the run — re-run with injected config
        from graph.graph import build_pruning_graph
        from stages.stage1_smact import run_stage1_smact
        from stages.stage2_radius import run_stage2_radius
        from stages.stage3_substitution import run_stage3_substitution

        # Re-run with patched threshold
        state = {
            "parent_formula": parent_formula,
            "target_site_species": target_species,
            "target_oxidation_state": target_os,
            "target_coordination_number": target_cn,
            "target_properties": [],
            "constraints": {},
            "config": cfg,
            "execution_log": [],
        }
        s1 = run_stage1_smact(state)
        state = {**state, **s1, "execution_log": []}
        s2 = run_stage2_radius(state)
        state = {**state, **s2, "execution_log": []}
        s3 = run_stage3_substitution(state)
        stage3_candidates = s3.get("stage3_candidates", [])

        metrics = evaluate_pruning(
            stage3_candidates,
            ground_truth_path=ground_truth_path,
            gt_classes=gt_classes,
            stage_label=f"thresh={thresh}",
        )

        results.append(
            SweepPoint(
                threshold=thresh,
                n_stage3_survivors=len(stage3_candidates),
                recall=metrics.recall,
                precision=metrics.precision,
                missed_dopants=metrics.missed,
            )
        )

    return results


def print_sweep_table(sweep: list[SweepPoint]) -> None:
    """Print a formatted sweep results table."""
    header = (
        f"  {'threshold':<10}  {'survivors':<12}  {'recall':<8}  "
        f"{'precision':<10}  missed"
    )
    print(f"\n{'='*70}")
    print("  Stage 3 Probability Threshold Sweep (NMC Co³⁺, confirmed_successful GT)")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*66}")
    for pt in sweep:
        print(pt)
    print()

    # Recommend the highest threshold maintaining ≥90% recall
    qualifying = [p for p in sweep if p.recall >= 0.90]
    if qualifying:
        best = max(qualifying, key=lambda p: p.threshold)
        print(
            f"  Recommendation: threshold={best.threshold}  "
            f"({best.n_stage3_survivors} candidates, recall={best.recall:.1%})"
        )
    else:
        print("  No threshold achieves ≥90% recall — lower thresholds required.")
    print()


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    sweep = sweep_stage3_threshold(
        parent_formula="LiNi0.8Mn0.1Co0.1O2",
        target_species="Co",
        target_os=3,
        target_cn=6,
    )
    print_sweep_table(sweep)
