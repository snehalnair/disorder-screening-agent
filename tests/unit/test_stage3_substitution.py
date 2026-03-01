"""Unit tests for Stage 3: Hautier-Ceder substitution probability filter."""

import pytest
from stages.stage1_smact import run_stage1_smact
from stages.stage2_radius import run_stage2_radius
from stages.stage3_substitution import run_stage3_substitution

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
            "stage2_radius": {"mismatch_threshold": 0.35},
            "stage3_substitution": {"probability_threshold": 0.001},
        }
    },
    "execution_log": [],
}


def _run_s1_s2(state: dict) -> dict:
    s1 = run_stage1_smact(state)
    s2 = run_stage2_radius({**state, **s1, "execution_log": []})
    return {**state, **s1, **s2, "execution_log": []}


# ── Basic output shape ────────────────────────────────────────────────────────

def test_returns_candidates():
    state = _run_s1_s2(BASE_STATE)
    result = run_stage3_substitution(state)
    assert "stage3_candidates" in result
    assert isinstance(result["stage3_candidates"], list)
    assert len(result["stage3_candidates"]) > 0


def test_candidate_schema():
    state = _run_s1_s2(BASE_STATE)
    result = run_stage3_substitution(state)
    for c in result["stage3_candidates"]:
        assert "element" in c
        assert "oxidation_state" in c
        assert "sub_probability" in c
        assert c["sub_probability"] >= 0.001


# ── Known chemically similar dopants survive ──────────────────────────────────

def test_isovalent_d_block_dopants_survive():
    """Al3+, Fe3+, Cr3+, V3+, Ga3+ have well-established Co3+ similarity."""
    state = _run_s1_s2(BASE_STATE)
    result = run_stage3_substitution(state)
    elements = {c["element"] for c in result["stage3_candidates"]}
    expected = {"Al", "Fe", "Cr", "V", "Ga"}
    missed = expected - elements
    assert not missed, f"Chemically similar dopants missed by Stage 3: {missed}"


# ── Probability threshold enforcement ────────────────────────────────────────

def test_all_above_threshold():
    state = _run_s1_s2(BASE_STATE)
    result = run_stage3_substitution(state)
    for c in result["stage3_candidates"]:
        assert c["sub_probability"] >= 0.001, (
            f"{c['element']}^{c['oxidation_state']}+ probability "
            f"{c['sub_probability']} below threshold"
        )


def test_higher_threshold_yields_fewer_candidates():
    state_low = _run_s1_s2(BASE_STATE)
    state_low["config"]["pipeline"]["stage3_substitution"]["probability_threshold"] = 0.001

    state_high = _run_s1_s2(BASE_STATE)
    state_high["config"]["pipeline"]["stage3_substitution"]["probability_threshold"] = 0.01

    n_low = len(run_stage3_substitution(state_low)["stage3_candidates"])
    n_high = len(run_stage3_substitution(state_high)["stage3_candidates"])
    assert n_high <= n_low


# ── Sorted by probability ─────────────────────────────────────────────────────

def test_sorted_descending_by_probability():
    state = _run_s1_s2(BASE_STATE)
    result = run_stage3_substitution(state)
    probs = [c["sub_probability"] for c in result["stage3_candidates"]]
    assert probs == sorted(probs, reverse=True)


# ── Funnel: Stage 3 ≤ Stage 2 ────────────────────────────────────────────────

def test_stage3_le_stage2():
    state = _run_s1_s2(BASE_STATE)
    result = run_stage3_substitution(state)
    assert len(result["stage3_candidates"]) <= len(state["stage2_candidates"])


# ── Execution log ─────────────────────────────────────────────────────────────

def test_execution_log_appended():
    state = _run_s1_s2(BASE_STATE)
    result = run_stage3_substitution(state)
    assert len(result["execution_log"]) == 1
    assert "Stage 3" in result["execution_log"][0]


# ── Empty input is handled gracefully ────────────────────────────────────────

def test_empty_stage2_input():
    state = {**_run_s1_s2(BASE_STATE), "stage2_candidates": []}
    result = run_stage3_substitution(state)
    assert result["stage3_candidates"] == []
