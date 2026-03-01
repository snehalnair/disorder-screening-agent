"""Tests for io/parse_input.py — PipelineInput validation (Tier 1)."""

from __future__ import annotations

import pytest

from pipeline_io.parse_input import (
    PipelineInput,
    ValidationError,
    pipeline_input_from_dict,
    validate_pipeline_input,
)


def _valid() -> PipelineInput:
    return PipelineInput(
        parent_formula="LiNi0.8Mn0.1Co0.1O2",
        target_species="Co",
        target_oxidation_state=3,
        target_coordination_number=6,
        concentrations=[0.05, 0.10],
        supercell_size=[2, 2, 2],
        n_sqs_realisations=3,
    )


# ── Happy path ────────────────────────────────────────────────────────────────

def test_valid_input_passes():
    inp = validate_pipeline_input(_valid())
    assert inp.parent_formula == "LiNi0.8Mn0.1Co0.1O2"


def test_valid_from_dict():
    inp = pipeline_input_from_dict(
        {
            "parent_formula": "LiCoO2",
            "target_species": "Co",
            "target_oxidation_state": 3,
        }
    )
    assert inp.target_species == "Co"


# ── Formula validation ────────────────────────────────────────────────────────

def test_invalid_formula_raises():
    inp = _valid()
    inp.parent_formula = "NotAFormula!!!"
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("parent_formula" in e for e in exc_info.value.errors)


def test_empty_formula_raises():
    inp = _valid()
    inp.parent_formula = ""
    with pytest.raises(ValidationError):
        validate_pipeline_input(inp)


# ── Target species ────────────────────────────────────────────────────────────

def test_species_not_in_formula_raises():
    inp = _valid()
    inp.target_species = "Zr"   # not in NMC
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("target_species" in e for e in exc_info.value.errors)


# ── Oxidation state ───────────────────────────────────────────────────────────

def test_negative_oxidation_state_raises():
    inp = _valid()
    inp.target_oxidation_state = -1
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("target_oxidation_state" in e for e in exc_info.value.errors)


def test_zero_oxidation_state_raises():
    inp = _valid()
    inp.target_oxidation_state = 0
    with pytest.raises(ValidationError):
        validate_pipeline_input(inp)


# ── Coordination number ───────────────────────────────────────────────────────

def test_invalid_cn_raises():
    inp = _valid()
    inp.target_coordination_number = 5
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("coordination_number" in e for e in exc_info.value.errors)


def test_valid_cns_pass():
    for cn in (4, 6, 8, 12):
        inp = _valid()
        inp.target_coordination_number = cn
        validate_pipeline_input(inp)   # must not raise


# ── Concentrations ────────────────────────────────────────────────────────────

def test_concentration_above_1_raises():
    inp = _valid()
    inp.concentrations = [0.05, 1.5]
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("concentration" in e for e in exc_info.value.errors)


def test_concentration_zero_raises():
    inp = _valid()
    inp.concentrations = [0.0, 0.10]
    with pytest.raises(ValidationError):
        validate_pipeline_input(inp)


def test_concentration_1_is_valid():
    inp = _valid()
    inp.concentrations = [1.0]
    validate_pipeline_input(inp)   # 100% dopant is edge case but valid


# ── Supercell ─────────────────────────────────────────────────────────────────

def test_negative_supercell_raises():
    inp = _valid()
    inp.supercell_size = [2, -1, 2]
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("supercell" in e for e in exc_info.value.errors)


def test_zero_supercell_raises():
    inp = _valid()
    inp.supercell_size = [0, 2, 2]
    with pytest.raises(ValidationError):
        validate_pipeline_input(inp)


# ── n_sqs_realisations ────────────────────────────────────────────────────────

def test_zero_sqs_raises():
    inp = _valid()
    inp.n_sqs_realisations = 0
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("n_sqs" in e for e in exc_info.value.errors)


# ── specific_dopant ───────────────────────────────────────────────────────────

def test_valid_specific_dopant_passes():
    inp = _valid()
    inp.specific_dopant = "Al"
    inp.specific_dopant_os = 3
    validate_pipeline_input(inp)


def test_invalid_specific_dopant_raises():
    inp = _valid()
    inp.specific_dopant = "Unobtanium"
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("specific_dopant" in e for e in exc_info.value.errors)


def test_specific_dopant_os_without_dopant_raises():
    inp = _valid()
    inp.specific_dopant = None
    inp.specific_dopant_os = 3
    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)
    assert any("specific_dopant_os" in e for e in exc_info.value.errors)


# ── Multiple errors collected ─────────────────────────────────────────────────

def test_multiple_errors_reported_together():
    """All validation errors must be reported in one raise, not just the first."""
    inp = _valid()
    inp.target_oxidation_state = -1       # error 1
    inp.concentrations = [0.0, 2.0]       # error 2 + 3
    inp.n_sqs_realisations = 0            # error 4

    with pytest.raises(ValidationError) as exc_info:
        validate_pipeline_input(inp)

    assert len(exc_info.value.errors) >= 3


# ── Tier 2/3 stub ─────────────────────────────────────────────────────────────

def test_tier2_raises_not_implemented():
    from pipeline_io.parse_input import tier2_parse_natural_language
    with pytest.raises(NotImplementedError):
        tier2_parse_natural_language("screen NMC for Al dopants", {})
