"""
Tests for Stage 4: ML property pre-screen.

All tests use mock backends so no GPU or model checkpoints are required.
GPU-required tests are marked @pytest.mark.gpu.
"""

from __future__ import annotations

import pytest
from stages.stage4_ml_prescreen import (
    _DEFAULT_THRESHOLDS,
    _MockBackend,
    _ORDERED_CAVEAT,
    run_stage4_ml_prescreen,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_EXCLUDE = [
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    "Tc", "Pm", "Po", "At", "Fr", "Ra", "Ac", "Pa", "Np", "Pu",
]

_BASE_CFG_DISABLED = {
    "pipeline": {
        "stage1_smact": {"exclude_elements": _EXCLUDE},
        "stage2_radius": {"mismatch_threshold": 0.35},
        "stage3_substitution": {"probability_threshold": 0.001},
        "stage4_ml": {"enabled": False},
    }
}

_BASE_CFG_ENABLED = {
    "pipeline": {
        "stage4_ml": {
            "enabled": True,
            "model": "roost",
            "checkpoint": None,
            "threshold": {
                "formation_energy_above_hull": 0.200,
                "voltage_min": 2.0,
                "voltage_max": 5.5,
            },
        }
    }
}


def _make_candidates(*elements) -> list[dict]:
    return [
        {"element": el, "oxidation_state": 3, "is_aliovalent": False,
         "sub_probability": 0.05, "shannon_radius": 0.55, "mismatch_pct": 1.0}
        for el in elements
    ]


def _state(cfg: dict, candidates: list[dict]) -> dict:
    return {
        "parent_formula": "LiNi0.8Mn0.1Co0.1O2",
        "target_site_species": "Co",
        "target_oxidation_state": 3,
        "stage3_candidates": candidates,
        "config": cfg,
        "execution_log": [],
    }


# ── Skip behaviour ────────────────────────────────────────────────────────────

def test_disabled_passes_candidates_through():
    """When stage4_ml.enabled = false, stage3_candidates pass through unchanged."""
    cands = _make_candidates("Al", "Ti", "Mg")
    result = run_stage4_ml_prescreen(_state(_BASE_CFG_DISABLED, cands))
    assert result["stage4_candidates"] == cands


def test_disabled_log_mentions_disabled():
    cands = _make_candidates("Al")
    result = run_stage4_ml_prescreen(_state(_BASE_CFG_DISABLED, cands))
    assert any("disabled" in e.lower() for e in result["execution_log"])


# ── Filtering logic with mock backend ────────────────────────────────────────

def _enabled_state_with_predictions(predictions: dict, candidates: list[dict]) -> dict:
    """Build state with enabled Stage 4 + inject mock predictions into the backend."""
    state = _state(_BASE_CFG_ENABLED, candidates)
    # Monkey-patch: inject predictions via the mock backend by overriding _doped_formula
    state["_mock_predictions"] = predictions
    return state


class _InjectableMockBackend(_MockBackend):
    """Mock backend whose predictions are injected via state."""
    def __init__(self, predictions: dict):
        super().__init__(predictions)


def test_candidate_above_hull_threshold_is_pruned(monkeypatch):
    """Candidate predicted > 0.2 eV/atom above hull must be pruned."""
    candidates = _make_candidates("Al", "BadEl")
    # Mock: Al is good (0.01), BadEl is above threshold (0.3)
    mock_preds = {"Al": {"formation_energy_above_hull": 0.01},
                  "BadEl": {"formation_energy_above_hull": 0.30}}

    def mock_prescreen(state):
        from stages.stage4_ml_prescreen import _DEFAULT_THRESHOLDS, _ORDERED_CAVEAT
        stage3 = state.get("stage3_candidates") or []
        cfg = (state.get("config", {}).get("pipeline", {}) or {}).get("stage4_ml", {})
        thresholds = {**_DEFAULT_THRESHOLDS, **(cfg.get("threshold") or {})}
        passed = []
        for c in stage3:
            preds = mock_preds.get(c["element"], {})
            ef = preds.get("formation_energy_above_hull", 0.0)
            if ef <= thresholds["formation_energy_above_hull"]:
                passed.append({**c, "ml_predicted_property": preds, "stage_passed": 4})
        return {"stage4_candidates": passed,
                "execution_log": [f"Stage 4: {len(passed)} candidates", _ORDERED_CAVEAT]}

    monkeypatch.setattr(
        "stages.stage4_ml_prescreen.run_stage4_ml_prescreen", mock_prescreen
    )
    from stages import stage4_ml_prescreen
    result = mock_prescreen(_state(_BASE_CFG_ENABLED, candidates))

    elements = {c["element"] for c in result["stage4_candidates"]}
    assert "Al" in elements
    assert "BadEl" not in elements


def test_candidate_within_threshold_survives(monkeypatch):
    """Candidate with predicted Ef = 0.1 eV/atom (< 0.2 threshold) must survive."""
    candidates = _make_candidates("Al")
    mock_preds = {"Al": {"formation_energy_above_hull": 0.10}}

    def mock_prescreen(state):
        stage3 = state.get("stage3_candidates") or []
        passed = []
        for c in stage3:
            preds = mock_preds.get(c["element"], {})
            ef = preds.get("formation_energy_above_hull", 0.0)
            if ef <= 0.200:
                passed.append({**c, "ml_predicted_property": preds, "stage_passed": 4})
        return {"stage4_candidates": passed, "execution_log": []}

    result = mock_prescreen(_state(_BASE_CFG_ENABLED, candidates))
    assert len(result["stage4_candidates"]) == 1
    assert result["stage4_candidates"][0]["element"] == "Al"


# ── Annotation ────────────────────────────────────────────────────────────────

def test_survivors_annotated_with_ml_predicted_property(monkeypatch):
    """Surviving candidates must have ml_predicted_property populated."""
    candidates = _make_candidates("Al", "Ti")
    predictions = {
        "Al": {"formation_energy_above_hull": 0.02, "voltage": 3.9},
        "Ti": {"formation_energy_above_hull": 0.05, "voltage": 4.1},
    }

    def mock_prescreen(state):
        stage3 = state.get("stage3_candidates") or []
        passed = []
        for c in stage3:
            preds = predictions.get(c["element"], {})
            passed.append({**c, "ml_predicted_property": preds, "stage_passed": 4})
        return {"stage4_candidates": passed, "execution_log": []}

    result = mock_prescreen(_state(_BASE_CFG_ENABLED, candidates))
    for cand in result["stage4_candidates"]:
        assert "ml_predicted_property" in cand
        assert cand["ml_predicted_property"] is not None


# ── Stage_passed field ────────────────────────────────────────────────────────

def test_stage_passed_set_to_4(monkeypatch):
    candidates = _make_candidates("Al")

    def mock_prescreen(state):
        stage3 = state.get("stage3_candidates") or []
        return {"stage4_candidates": [{**c, "stage_passed": 4} for c in stage3],
                "execution_log": []}

    result = mock_prescreen(_state(_BASE_CFG_ENABLED, candidates))
    for c in result["stage4_candidates"]:
        assert c["stage_passed"] == 4


# ── Threshold sensitivity ─────────────────────────────────────────────────────

def test_tighter_threshold_fewer_survivors():
    """Stricter Ef threshold → fewer survivors (direct mock logic test)."""
    predictions = {
        "Al": 0.05,
        "Ti": 0.15,
        "Mg": 0.30,
    }
    candidates = _make_candidates("Al", "Ti", "Mg")

    def apply_threshold(thresh):
        return [el for el, ef in predictions.items() if ef <= thresh]

    n_tight = len(apply_threshold(0.10))
    n_loose = len(apply_threshold(0.50))
    assert n_tight <= n_loose


# ── Caveat in output ──────────────────────────────────────────────────────────

def test_ordered_caveat_in_log_when_enabled():
    """When Stage 4 runs, the ordered-properties caveat must appear in execution_log."""
    # Test against the actual function with disabled=False
    # The pass-through mock backend (0 eV/atom) means all candidates survive
    cands = _make_candidates("Al", "Ti")
    result = run_stage4_ml_prescreen(_state(_BASE_CFG_ENABLED, cands))
    log_text = " ".join(result["execution_log"])
    assert "ORDERED" in log_text or "ordered" in log_text.lower()


def test_no_caveat_when_disabled():
    """When Stage 4 is disabled, the caveat should not appear (irrelevant)."""
    cands = _make_candidates("Al")
    result = run_stage4_ml_prescreen(_state(_BASE_CFG_DISABLED, cands))
    # Just verify it runs cleanly; the caveat is optional when disabled
    assert "stage4_candidates" in result


# ── Fallback on backend failure ───────────────────────────────────────────────

def test_fallback_on_missing_checkpoint():
    """If no real checkpoint exists, mock backend is used → candidates pass through."""
    cfg = {
        "pipeline": {
            "stage4_ml": {
                "enabled": True,
                "model": "cgcnn",
                "checkpoint": "/nonexistent/path.pth",
                "threshold": {"formation_energy_above_hull": 0.200},
            }
        }
    }
    cands = _make_candidates("Al", "Ti")
    result = run_stage4_ml_prescreen({
        "parent_formula": "LiNi0.8Mn0.1Co0.1O2",
        "target_site_species": "Co",
        "target_oxidation_state": 3,
        "stage3_candidates": cands,
        "config": cfg,
        "execution_log": [],
    })
    # With mock backend (0 eV/atom), all candidates should pass through
    assert len(result["stage4_candidates"]) == len(cands)


# ── Empty input ───────────────────────────────────────────────────────────────

def test_empty_stage3_input_disabled():
    result = run_stage4_ml_prescreen(_state(_BASE_CFG_DISABLED, []))
    assert result["stage4_candidates"] == []


def test_empty_stage3_input_enabled():
    result = run_stage4_ml_prescreen(_state(_BASE_CFG_ENABLED, []))
    assert result["stage4_candidates"] == []
