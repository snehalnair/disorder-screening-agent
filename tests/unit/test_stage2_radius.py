"""Unit tests for Stage 2: Shannon ionic radius filter."""

import pytest
from stages.stage1_smact import run_stage1_smact
from stages.stage2_radius import run_stage2_radius

_EXCLUDE = [
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    "Tc", "Pm", "Po", "At", "Fr", "Ra", "Ac", "Pa", "Np", "Pu",
]

BASE_STATE = {
    "target_site_species": "Co",
    "target_oxidation_state": 3,
    "target_coordination_number": 6,
    "parent_formula": "LiNi0.8Mn0.1Co0.1O2",
    "constraints": {},
    "config": {
        "pipeline": {
            "stage1_smact": {"exclude_elements": _EXCLUDE},
            "stage2_radius": {
                "mismatch_threshold": 0.35,   # calibrated for ≥90% recall
                "tolerance_factor_max": 4.18,
            },
        }
    },
    "execution_log": [],
}


def _run_through_s1(state: dict) -> dict:
    """Run Stage 1 and merge results into state for Stage 2 input."""
    s1_out = run_stage1_smact(state)
    return {**state, **s1_out, "execution_log": []}


# ── Basic output shape ────────────────────────────────────────────────────────

def test_returns_candidates():
    state = _run_through_s1(BASE_STATE)
    result = run_stage2_radius(state)
    assert "stage2_candidates" in result
    assert isinstance(result["stage2_candidates"], list)
    assert len(result["stage2_candidates"]) > 0


def test_candidate_schema():
    state = _run_through_s1(BASE_STATE)
    result = run_stage2_radius(state)
    for c in result["stage2_candidates"]:
        assert "element" in c
        assert "oxidation_state" in c
        assert "shannon_radius" in c
        assert "mismatch_pct" in c
        assert c["shannon_radius"] > 0
        assert c["mismatch_pct"] >= 0


# ── Known radius-compatible dopants survive ───────────────────────────────────

def test_radius_compatible_dopants_survive():
    """Al3+, Fe3+, Cr3+, V3+, Ga3+ are all close in radius to Co3+ (0.545 Å)."""
    state = _run_through_s1(BASE_STATE)
    result = run_stage2_radius(state)
    elements = {c["element"] for c in result["stage2_candidates"]}
    expected = {"Al", "Fe", "Cr", "V", "Ga"}
    missed = expected - elements
    assert not missed, f"Radius-compatible dopants missed by Stage 2: {missed}"


# ── Mismatch threshold enforcement ───────────────────────────────────────────

def test_mismatch_within_threshold():
    state = _run_through_s1(BASE_STATE)
    result = run_stage2_radius(state)
    threshold_pct = (
        state["config"]["pipeline"]["stage2_radius"]["mismatch_threshold"] * 100
    )
    for c in result["stage2_candidates"]:
        assert c["mismatch_pct"] <= threshold_pct + 1e-6, (
            f"{c['element']}^{c['oxidation_state']}+ mismatch "
            f"{c['mismatch_pct']:.2f}% exceeds {threshold_pct:.0f}%"
        )


def test_tighter_threshold_yields_fewer_candidates():
    import copy
    state_tight = copy.deepcopy(_run_through_s1(BASE_STATE))
    state_tight["config"]["pipeline"]["stage2_radius"]["mismatch_threshold"] = 0.05

    state_loose = copy.deepcopy(_run_through_s1(BASE_STATE))
    state_loose["config"]["pipeline"]["stage2_radius"]["mismatch_threshold"] = 0.50

    n_tight = len(run_stage2_radius(state_tight)["stage2_candidates"])
    n_loose = len(run_stage2_radius(state_loose)["stage2_candidates"])
    assert n_tight <= n_loose


# ── Funnel: Stage 2 ≤ Stage 1 ────────────────────────────────────────────────

def test_stage2_le_stage1():
    state = _run_through_s1(BASE_STATE)
    result = run_stage2_radius(state)
    assert len(result["stage2_candidates"]) <= len(state["stage1_candidates"])


# ── Execution log ─────────────────────────────────────────────────────────────

def test_execution_log_appended():
    state = _run_through_s1(BASE_STATE)
    result = run_stage2_radius(state)
    assert len(result["execution_log"]) == 1
    assert "Stage 2" in result["execution_log"][0]


# ── Missing host radius raises ────────────────────────────────────────────────

def test_missing_host_radius_raises():
    state = _run_through_s1(BASE_STATE)
    state["target_site_species"] = "Xx"  # non-existent element
    with pytest.raises(ValueError, match="Shannon radius not found"):
        run_stage2_radius(state)
