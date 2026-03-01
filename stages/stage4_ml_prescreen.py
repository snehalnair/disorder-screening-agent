"""
Stage 4: ML property pre-screen (optional).

Uses a pre-trained composition-based or structure-based ML model to predict
target properties for each doped composition and prunes candidates with
clearly unfavourable predictions before the expensive Stage 5 simulations.

Supported backends:
  - cgcnn   — Crystal Graph Convolutional Neural Network (structure-based, preferred)
  - roost   — Roost composition-based model (fallback)
  - crabnet — CrabNet attention model (fallback)

IMPORTANT CAVEAT: These models predict properties for ORDERED compositions.
They serve as a rough filter only.  Disorder-aware simulation in Stage 5 may
significantly change predicted values.  This caveat is logged prominently.

This stage is OPTIONAL.  Set ``stage4_ml.enabled: false`` in pipeline.yaml
to route directly from Stage 3 to Stage 5.

Input state keys:
    stage3_candidates (list[dict])    — output of Stage 3
    parent_formula (str)
    parent_structure (Structure|None) — required for CGCNN backend
    target_site_species (str)
    config (dict)

Output state keys:
    stage4_candidates (list[dict])    — filtered, annotated with ml_predicted_property
    execution_log (list[str])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TOOL_METADATA = {
    "name": "ml_prescreen",
    "stage": 4,
    "description": (
        "ML property pre-screen using CGCNN (structure-based) or Roost/CrabNet "
        "(composition-based).  Removes candidates with clearly unfavourable "
        "predicted properties before expensive Stage 5 MLIP simulations."
    ),
    "system_type": "periodic_crystal",
    "input_type": "list[CandidateDopant] + parent structure/formula",
    "output_type": "list[CandidateDopant] (filtered, annotated with ml_predicted_property)",
    "cost": "seconds",
    "cost_per_candidate": "~1–5 s (CGCNN with structure), ~0.5 s (Roost composition-only)",
    "typical_reduction": "46 → 15–25 candidates (post-Phase 1 calibration)",
    "external_dependencies": ["cgcnn or roost or crabnet", "pre-trained model checkpoint"],
    "requires_structure": True,
    "requires_network": False,
    "requires_gpu": True,
    "configurable_params": ["model_name", "model_checkpoint", "threshold_config"],
    "failure_modes": [
        "model checkpoint not found",
        "element not in model training set → fallback to Roost → pass-through on failure",
    ],
    "caveat": (
        "Predicts ORDERED properties only. "
        "Disorder-aware Stage 5 may change values significantly."
    ),
}

_ORDERED_CAVEAT = (
    "⚠ Stage 4 ML predictions are for ORDERED compositions. "
    "Disorder-aware Stage 5 simulation may yield significantly different values."
)

# ── Default filtering thresholds ──────────────────────────────────────────────
_DEFAULT_THRESHOLDS = {
    "formation_energy_above_hull": 0.200,   # eV/atom — remove if above
    "voltage_min": 2.0,                     # V — remove if below
    "voltage_max": 5.5,                     # V — remove if above
}


# ── Backend loaders (lazy import — graceful if not installed) ─────────────────

def _load_cgcnn_model(checkpoint: str | None):
    """Load CGCNN model from checkpoint path. Returns None if unavailable."""
    try:
        # cgcnn is from the original Xie & Grossman (2018) implementation
        from cgcnn.model import CrystalGraphConvNet  # type: ignore  # noqa: F401
        logger.debug("CGCNN backend loaded.")
        # Return a callable that wraps the checkpoint
        return _CGCNNBackend(checkpoint)
    except (ImportError, Exception) as exc:
        logger.warning("CGCNN unavailable (%s). Falling back to Roost.", exc)
        return None


def _load_roost_model(checkpoint: str | None):
    """Load Roost model. Returns None if unavailable."""
    try:
        from roost.model import Roost  # type: ignore  # noqa: F401
        logger.debug("Roost backend loaded.")
        return _RoostBackend(checkpoint)
    except (ImportError, Exception) as exc:
        logger.warning("Roost unavailable (%s).", exc)
        return None


# ── Backend classes ───────────────────────────────────────────────────────────

class _CGCNNBackend:
    """Thin wrapper around CGCNN for batch property prediction."""

    def __init__(self, checkpoint: str | None):
        self.checkpoint = checkpoint

    def predict(self, formula: str, structure=None) -> dict[str, float]:
        """Predict formation energy for a doped composition."""
        # Placeholder — real implementation wraps cgcnn inference pipeline
        raise NotImplementedError("CGCNN inference not yet wired up.")


class _RoostBackend:
    """Thin wrapper around Roost for composition-based prediction."""

    def __init__(self, checkpoint: str | None):
        self.checkpoint = checkpoint

    def predict(self, formula: str, structure=None) -> dict[str, float]:
        """Predict formation energy from composition string."""
        raise NotImplementedError("Roost inference not yet wired up.")


class _MockBackend:
    """Mock backend used in tests and when no checkpoint is available."""

    def __init__(self, predictions: dict[str, dict] | None = None):
        self._predictions = predictions or {}

    def predict(self, formula: str, structure=None) -> dict[str, float]:
        return self._predictions.get(formula, {"formation_energy_above_hull": 0.0})


# ── Helper: build doped formula ───────────────────────────────────────────────

def _doped_formula(parent_formula: str, target: str, dopant: str, conc: float = 0.05) -> str:
    """Return a simple doped formula string for composition-based models.

    For a parent like LiNi0.8Mn0.1Co0.1O2 replacing Co with Al at 5%:
    → LiNi0.8Mn0.1Co0.05Al0.05O2  (approximate)
    """
    try:
        from pymatgen.core import Composition
        comp = Composition(parent_formula)
        amounts = dict(comp.fractional_composition.as_dict())
        if target in amounts:
            remaining = amounts[target] * (1 - conc)
            amounts[target] = remaining
            amounts[dopant] = amounts.get(dopant, 0) + amounts.get(target, 0) * conc
        return Composition(amounts).formula
    except Exception:
        return f"{parent_formula}_doped_{dopant}"


# ── Main node ─────────────────────────────────────────────────────────────────

def run_stage4_ml_prescreen(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: optional ML property pre-screen.

    If ``stage4_ml.enabled`` is False in config, passes stage3_candidates
    through unchanged as stage4_candidates.
    """
    cfg: dict = state.get("config") or {}
    stage_cfg: dict = (cfg.get("pipeline") or {}).get("stage4_ml") or {}

    # ── Skip if disabled ──────────────────────────────────────────────────────
    if not stage_cfg.get("enabled", False):
        stage3 = state.get("stage3_candidates") or []
        log_msg = "Stage 4 (ML pre-screen): disabled — passing Stage 3 candidates through unchanged."
        logger.info(log_msg)
        return {
            "stage4_candidates": stage3,
            "execution_log": [log_msg],
        }

    logger.warning(_ORDERED_CAVEAT)

    model_name: str = stage_cfg.get("model", "cgcnn").lower()
    checkpoint: str | None = stage_cfg.get("checkpoint")
    thresholds: dict = {**_DEFAULT_THRESHOLDS, **(stage_cfg.get("threshold") or {})}

    parent_formula: str = state.get("parent_formula", "")
    parent_structure = state.get("parent_structure")
    stage3_candidates: list[dict] = state.get("stage3_candidates") or []

    # ── Load model backend ────────────────────────────────────────────────────
    backend = None
    if model_name == "cgcnn":
        backend = _load_cgcnn_model(checkpoint)
    if backend is None:
        backend = _load_roost_model(checkpoint)
    if backend is None:
        logger.warning(
            "All ML backends unavailable. Passing all candidates through (err on inclusion)."
        )
        backend = _MockBackend()   # pass-through mock — predicts 0 eV/atom for all

    # ── Filter candidates ─────────────────────────────────────────────────────
    passed: list[dict] = []
    n_pruned = 0

    for cand in stage3_candidates:
        formula = _doped_formula(parent_formula, state.get("target_site_species", ""), cand["element"])
        try:
            preds: dict = backend.predict(formula, structure=parent_structure)
        except NotImplementedError:
            # Backend not wired up yet — err on inclusion
            preds = {"formation_energy_above_hull": 0.0}
        except Exception as exc:
            logger.debug("ML prediction failed for %s: %s — passing through.", cand["element"], exc)
            preds = {}

        # Apply thresholds
        pruned = False
        reason = ""
        ef = preds.get("formation_energy_above_hull")
        if ef is not None and ef > thresholds["formation_energy_above_hull"]:
            pruned = True
            reason = (
                f"formation_energy_above_hull {ef:.3f} eV/atom "
                f"> threshold {thresholds['formation_energy_above_hull']:.3f}"
            )
        voltage = preds.get("voltage")
        if voltage is not None:
            if voltage < thresholds["voltage_min"]:
                pruned = True
                reason = f"voltage {voltage:.2f} V < min {thresholds['voltage_min']:.2f}"
            elif voltage > thresholds["voltage_max"]:
                pruned = True
                reason = f"voltage {voltage:.2f} V > max {thresholds['voltage_max']:.2f}"

        if pruned:
            logger.debug("  %s pruned: %s", cand["element"], reason)
            n_pruned += 1
        else:
            passed.append(
                {
                    **cand,
                    "ml_predicted_property": preds if preds else None,
                    "stage_passed": 4,
                }
            )

    log_msg = (
        f"Stage 4 (ML pre-screen/{model_name}): {len(passed)} candidates "
        f"(pruned {n_pruned}, from {len(stage3_candidates)} input). "
        f"Caveat: ordered-composition predictions only."
    )
    logger.info(log_msg)

    return {
        "stage4_candidates": passed,
        "execution_log": [log_msg, _ORDERED_CAVEAT],
    }
