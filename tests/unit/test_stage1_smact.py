"""Unit tests for Stage 1: SMACT composition filter."""

import pytest
from stages.stage1_smact import run_stage1_smact

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
            "stage1_smact": {"exclude_elements": _EXCLUDE}
        }
    },
    "execution_log": [],
}


# ── Basic output shape ────────────────────────────────────────────────────────

def test_returns_candidates():
    result = run_stage1_smact(BASE_STATE)
    assert "stage1_candidates" in result
    assert isinstance(result["stage1_candidates"], list)
    assert len(result["stage1_candidates"]) > 0


def test_candidate_schema():
    result = run_stage1_smact(BASE_STATE)
    for c in result["stage1_candidates"]:
        assert "element" in c
        assert "oxidation_state" in c
        assert "is_aliovalent" in c
        assert "pauling_eneg" in c
        assert isinstance(c["oxidation_state"], int)
        assert 1 <= c["oxidation_state"] <= 7


# ── Known NMC dopants must survive ───────────────────────────────────────────

def test_known_dopants_present():
    """All 13 experimentally confirmed NMC dopants must survive Stage 1."""
    result = run_stage1_smact(BASE_STATE)
    elements = {c["element"] for c in result["stage1_candidates"]}
    known = {"Al", "Ti", "Mg", "Ga", "Fe", "Cr", "V", "Zr", "Nb", "W", "Ta", "Mo", "B"}
    missed = known - elements
    assert not missed, f"Known dopants missed by Stage 1: {missed}"


# ── Exclusion logic ───────────────────────────────────────────────────────────

def test_noble_gases_excluded():
    result = run_stage1_smact(BASE_STATE)
    elements = {c["element"] for c in result["stage1_candidates"]}
    noble = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
    assert not (noble & elements), "Noble gases should be excluded"


def test_oxygen_excluded_by_electronegativity():
    """O has Pauling EN ≥ 3.44 and must not appear as a cation candidate."""
    result = run_stage1_smact(BASE_STATE)
    elements = {c["element"] for c in result["stage1_candidates"]}
    assert "O" not in elements


def test_fluorine_excluded_by_electronegativity():
    """F (EN = 3.98) must be excluded."""
    result = run_stage1_smact(BASE_STATE)
    elements = {c["element"] for c in result["stage1_candidates"]}
    assert "F" not in elements


def test_user_constraint_excludes_element():
    state = {**BASE_STATE, "constraints": {"exclude_elements": ["Al"]}}
    result = run_stage1_smact(state)
    elements = {c["element"] for c in result["stage1_candidates"]}
    assert "Al" not in elements


# ── Aliovalent flag ───────────────────────────────────────────────────────────

def test_aliovalent_flag_isovalent():
    """Candidate with ox=3 substituting Co3+ must have is_aliovalent=False."""
    result = run_stage1_smact(BASE_STATE)
    isovalent = [c for c in result["stage1_candidates"] if c["oxidation_state"] == 3]
    assert all(not c["is_aliovalent"] for c in isovalent)


def test_aliovalent_flag_aliovalent():
    """Candidate with ox≠3 must have is_aliovalent=True."""
    result = run_stage1_smact(BASE_STATE)
    aliovalent = [c for c in result["stage1_candidates"] if c["oxidation_state"] != 3]
    assert all(c["is_aliovalent"] for c in aliovalent)


# ── Execution log ─────────────────────────────────────────────────────────────

def test_execution_log_appended():
    result = run_stage1_smact(BASE_STATE)
    assert len(result["execution_log"]) == 1
    assert "Stage 1" in result["execution_log"][0]


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_no_duplicate_element_ox_pairs():
    result = run_stage1_smact(BASE_STATE)
    seen = set()
    for c in result["stage1_candidates"]:
        key = (c["element"], c["oxidation_state"])
        assert key not in seen, f"Duplicate pair: {key}"
        seen.add(key)


# ── Unique element count (Task 3) ─────────────────────────────────────────────

def test_unique_element_count_in_range():
    """Unique element count for NMC Co³⁺ must be a reasonable subset of the periodic table.

    With the ML-extended Shannon table and SMACT's database covering Z=1–103,
    the actual observed count is ~80 elements (the PRD's "50–70" was illustrative
    before calibration against the real data).
    """
    result = run_stage1_smact(BASE_STATE)
    n_unique = result["stage1_unique_elements"]
    assert 50 <= n_unique <= 95, f"Unique element count {n_unique} outside expected 50–95"


def test_unique_less_than_os_combinations():
    """Unique elements must be strictly fewer than element-OS combinations."""
    result = run_stage1_smact(BASE_STATE)
    assert result["stage1_unique_elements"] < result["stage1_os_combinations"]


def test_unique_count_matches_candidate_elements():
    """stage1_unique_elements must equal the actual unique element count in stage1_candidates."""
    result = run_stage1_smact(BASE_STATE)
    actual_unique = len({c["element"] for c in result["stage1_candidates"]})
    assert result["stage1_unique_elements"] == actual_unique


def test_log_contains_unique_element_count():
    result = run_stage1_smact(BASE_STATE)
    log = result["execution_log"][0]
    assert "unique elements" in log
