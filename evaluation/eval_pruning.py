"""
RQ1: Evaluate pruning pipeline recall and precision against the ground truth.

Provides:
- ``evaluate_pruning()``     — recall/precision for one stage vs GT
- ``per_dopant_breakdown()`` — per-element table showing which stage filtered each GT dopant
- ``os_category_breakdown()``— recall split by oxidation-state category
- ``run_full_rq1()``         — comprehensive report dict for paper

Usage (standalone)::

    python -m evaluation.eval_pruning

Or import helpers directly from test/analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.ground_truth_loader import get_dopant_elements, load_ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PruningMetrics:
    """Recall/precision metrics for one pipeline stage vs the ground truth."""
    stage: str
    n_candidates: int
    n_gt_positives: int
    n_recalled: int
    recall: float
    precision: float | None          # precision over ALL survivors
    precision_known: float | None    # precision over survivors with known GT class
    missed: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  {self.stage}",
            f"{'='*60}",
            f"  Candidates       : {self.n_candidates}",
            f"  GT positives     : {self.n_gt_positives}",
            f"  Recalled         : {self.n_recalled}",
            f"  Recall           : {self.recall:.1%}",
        ]
        if self.precision is not None:
            lines.append(f"  Precision (all)  : {self.precision:.1%}  ← includes unknowns")
        if self.precision_known is not None:
            lines.append(f"  Precision (known): {self.precision_known:.1%}  ← excludes unknowns")
        if self.missed:
            lines.append(f"  Missed GT        : {', '.join(self.missed)}")
        return "\n".join(lines)


@dataclass
class PerDopantRow:
    """One row in the per-dopant breakdown table."""
    element: str
    gt_class: str          # confirmed_successful | confirmed_limited | confirmed_failed | unknown
    oxidation_state: int | None
    survived_stage1: bool
    survived_stage2: bool
    survived_stage3: bool
    filtered_at_stage: str | None  # "Stage 1" | "Stage 2" | "Stage 3" | None (survived)
    reason: str            # e.g. "mismatch > 35%" for Stage 2, "" if survived


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function (unchanged API from Phase 2)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_pruning(
    stage_candidates: list[dict],
    ground_truth_path=None,
    site_filter: str = "TM_octahedral",
    gt_classes: list[str] | None = None,
    stage_label: str = "Stage",
) -> PruningMetrics:
    """Compute recall and precision for a stage output vs the ground truth.

    Args:
        stage_candidates:  List of candidate dicts (must contain ``"element"``).
        ground_truth_path: Path to the ground truth JSON; defaults to
                           ``data/known_dopants/nmc_layered_oxide.json``.
        site_filter:       Only evaluate dopants on this site.
        gt_classes:        Ground truth classes counted as positives.
                           Defaults to ``["confirmed_successful",
                           "confirmed_limited"]``.
        stage_label:       Human-readable label for the metrics output.

    Returns:
        ``PruningMetrics`` dataclass.
    """
    if gt_classes is None:
        gt_classes = ["confirmed_successful", "confirmed_limited"]

    gt = load_ground_truth(ground_truth_path)
    gt_elements = set(
        get_dopant_elements(gt, site_filter=site_filter, classes=gt_classes)
    )
    all_gt_elements = set(
        get_dopant_elements(gt, site_filter=site_filter, classes=None)
    )

    candidate_elements = {c["element"] for c in stage_candidates}

    recalled = gt_elements & candidate_elements
    missed = sorted(gt_elements - candidate_elements)

    recall = len(recalled) / len(gt_elements) if gt_elements else 0.0
    precision = (
        len(recalled) / len(candidate_elements) if candidate_elements else None
    )
    # Precision among survivors with known experimental outcome
    known_survivors = candidate_elements & all_gt_elements
    gt_positive_survivors = candidate_elements & gt_elements
    precision_known = (
        len(gt_positive_survivors) / len(known_survivors) if known_survivors else None
    )

    return PruningMetrics(
        stage=stage_label,
        n_candidates=len(candidate_elements),
        n_gt_positives=len(gt_elements),
        n_recalled=len(recalled),
        recall=recall,
        precision=precision,
        precision_known=precision_known,
        missed=missed,
    )


def print_metrics(m: PruningMetrics) -> None:
    """Print a PruningMetrics summary to stdout."""
    print(m)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dopant breakdown
# ─────────────────────────────────────────────────────────────────────────────


def per_dopant_breakdown(
    stage_results: dict,
    ground_truth_path=None,
) -> list[PerDopantRow]:
    """Build a per-dopant table showing at which stage each GT element was filtered.

    Args:
        stage_results:     Dict with keys ``stage1_candidates``,
                           ``stage2_candidates``, ``stage3_candidates``
                           (each a list of dicts with ``"element"``).
        ground_truth_path: Path to ground truth JSON.

    Returns:
        List of ``PerDopantRow``, one per GT dopant.
    """
    gt = load_ground_truth(ground_truth_path)
    all_gt = gt.get("dopants", {})

    s1 = {c["element"] for c in stage_results.get("stage1_candidates", [])}
    s2 = {c["element"] for c in stage_results.get("stage2_candidates", [])}
    s3 = {c["element"] for c in stage_results.get("stage3_candidates", [])}

    # Build per-element info for Stage 2 filtering reason
    s2_info = {c["element"]: c for c in stage_results.get("stage2_candidates", [])}
    # Stage 1 output has oxidation_state; Stage 2 adds mismatch_pct
    s1_info = {c["element"]: c for c in stage_results.get("stage1_candidates", [])}

    rows = []
    for element, info in sorted(all_gt.items()):
        in_s1 = element in s1
        in_s2 = element in s2
        in_s3 = element in s3

        filtered_at = None
        reason = ""
        if not in_s1:
            filtered_at = "Stage 1"
            reason = "failed SMACT EN/charge-neutrality"
        elif not in_s2:
            filtered_at = "Stage 2"
            c = s1_info.get(element, {})
            mismatch = c.get("mismatch_pct")
            reason = f"radius mismatch {mismatch:.1%}" if mismatch is not None else "radius mismatch"
        elif not in_s3:
            filtered_at = "Stage 3"
            reason = "substitution probability below threshold"

        rows.append(PerDopantRow(
            element=element,
            gt_class=info.get("ground_truth_class", "unknown"),
            oxidation_state=info.get("oxidation_state"),
            survived_stage1=in_s1,
            survived_stage2=in_s2,
            survived_stage3=in_s3,
            filtered_at_stage=filtered_at,
            reason=reason,
        ))

    return rows


def print_per_dopant_table(rows: list[PerDopantRow]) -> None:
    """Print the per-dopant breakdown as a formatted text table."""
    header = f"{'El':>3}  {'GT Class':<22}  {'OS':>3}  {'S1':>3}  {'S2':>3}  {'S3':>3}  {'Filtered at':<12}  Reason"
    print(f"\n{header}")
    print("-" * len(header))
    for r in rows:
        s1 = "✓" if r.survived_stage1 else "✗"
        s2 = "✓" if r.survived_stage2 else "✗"
        s3 = "✓" if r.survived_stage3 else "✗"
        filt = r.filtered_at_stage or "—  (survived)"
        os_str = str(r.oxidation_state) if r.oxidation_state is not None else "?"
        print(f"{r.element:>3}  {r.gt_class:<22}  {os_str:>3}  {s1:>3}  {s2:>3}  {s3:>3}  {filt:<12}  {r.reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Oxidation-state category breakdown
# ─────────────────────────────────────────────────────────────────────────────


def os_category_breakdown(
    stage_results: dict,
    ground_truth_path=None,
    gt_classes: list[str] | None = None,
) -> dict[str, dict]:
    """Compute recall by oxidation-state category for GT positives.

    Categories:
    - isovalent_3+   : dopants substituting as 3+ (same as Co3+)
    - aliovalent_2+  : 2+ dopants
    - aliovalent_4+  : 4+ dopants
    - aliovalent_56+ : 5+ or 6+ dopants

    Returns:
        Dict keyed by category with keys
        ``{"n_gt": int, "n_recalled": int, "recall": float, "elements": list}``.
    """
    if gt_classes is None:
        gt_classes = ["confirmed_successful", "confirmed_limited"]

    gt = load_ground_truth(ground_truth_path)
    all_gt = gt.get("dopants", {})

    s3_elements = {c["element"] for c in stage_results.get("stage3_candidates", [])}

    categories: dict[str, list[str]] = {
        "isovalent_3+": [],
        "aliovalent_2+": [],
        "aliovalent_4+": [],
        "aliovalent_56+": [],
    }

    for element, info in all_gt.items():
        if info.get("ground_truth_class") not in gt_classes:
            continue
        os = info.get("oxidation_state")
        if os == 3:
            categories["isovalent_3+"].append(element)
        elif os == 2:
            categories["aliovalent_2+"].append(element)
        elif os == 4:
            categories["aliovalent_4+"].append(element)
        elif os in (5, 6):
            categories["aliovalent_56+"].append(element)

    result = {}
    for cat, elements in categories.items():
        recalled = [e for e in elements if e in s3_elements]
        n_gt = len(elements)
        n_recalled = len(recalled)
        result[cat] = {
            "n_gt": n_gt,
            "n_recalled": n_recalled,
            "recall": n_recalled / n_gt if n_gt else 0.0,
            "elements": sorted(elements),
            "recalled": sorted(recalled),
            "missed": sorted(set(elements) - set(recalled)),
        }

    return result


def print_os_breakdown(breakdown: dict[str, dict]) -> None:
    """Print OS category breakdown as a text table."""
    print(f"\n{'Category':<16}  {'GT':>4}  {'Recalled':>8}  {'Recall':>7}  Elements")
    print("-" * 70)
    for cat, d in breakdown.items():
        els = ", ".join(d["elements"])
        missed = f"  [missed: {', '.join(d['missed'])}]" if d["missed"] else ""
        print(f"{cat:<16}  {d['n_gt']:>4}  {d['n_recalled']:>8}  {d['recall']:>6.0%}  {els}{missed}")


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive RQ1 report
# ─────────────────────────────────────────────────────────────────────────────


def run_full_rq1(
    config_path=None,
    ground_truth_path=None,
    parent_formula: str = "LiNi0.8Mn0.1Co0.1O2",
    target_site_species: str = "Co",
    target_oxidation_state: int = 3,
) -> dict:
    """Run Stages 1–3 and compute all RQ1 metrics.

    Returns a comprehensive dict with:
    - ``stage_metrics``    : list of PruningMetrics for stages 1-3
    - ``per_dopant``       : list of PerDopantRow
    - ``os_breakdown``     : dict from os_category_breakdown()
    - ``funnel_counts``    : {"stage0": all_elements, "stage1": n, "stage2": n, "stage3": n}
    - ``execution_log``    : pipeline log
    """
    from graph.entry_points import run_stages_1_3

    state = run_stages_1_3(
        parent_formula=parent_formula,
        target_site_species=target_site_species,
        target_oxidation_state=target_oxidation_state,
        config_path=config_path,
    )

    metrics = []
    for stage_key, label in [
        ("stage1_candidates", "Stage 1 — SMACT"),
        ("stage2_candidates", "Stage 2 — Radius"),
        ("stage3_candidates", "Stage 3 — Substitution"),
    ]:
        m = evaluate_pruning(
            state.get(stage_key, []),
            ground_truth_path=ground_truth_path,
            stage_label=label,
        )
        metrics.append(m)

    rows = per_dopant_breakdown(state, ground_truth_path=ground_truth_path)
    os_bd = os_category_breakdown(state, ground_truth_path=ground_truth_path)

    return {
        "state": state,
        "stage_metrics": metrics,
        "per_dopant": rows,
        "os_breakdown": os_bd,
        "funnel_counts": {
            "stage0": state.get("stage1_os_combinations", "?"),
            "stage1": len(state.get("stage1_candidates", [])),
            "stage2": len(state.get("stage2_candidates", [])),
            "stage3": len(state.get("stage3_candidates", [])),
        },
        "execution_log": state.get("execution_log", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    print("=" * 60)
    print("  RQ1: Pruning Recall & Precision — NMC Co³⁺")
    print("=" * 60)

    report = run_full_rq1()

    print("\nFunnel:")
    fc = report["funnel_counts"]
    print(f"  Element-OS combinations screened : {fc['stage0']}")
    print(f"  After Stage 1 (SMACT)            : {fc['stage1']}")
    print(f"  After Stage 2 (Radius)           : {fc['stage2']}")
    print(f"  After Stage 3 (Substitution)     : {fc['stage3']}")

    for m in report["stage_metrics"]:
        print_metrics(m)

    print("\n\nPer-dopant breakdown:")
    print_per_dopant_table(report["per_dopant"])

    print("\n\nOxidation-state category breakdown (Stage 3 survivors):")
    print_os_breakdown(report["os_breakdown"])
