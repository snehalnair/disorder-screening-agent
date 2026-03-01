"""
Tests for stages/stage5/property_calculator.py.

Uses LiCoO₂ fixture (from conftest.py) + MockMLIPCalculator.
No GPU required.
"""

from __future__ import annotations

import pytest

from stages.stage5.calculators import MockMLIPCalculator
from stages.stage5.property_calculator import (
    PROPERTY_REGISTRY,
    compute_formation_energy_above_hull,
    compute_lattice_params,
    compute_li_ni_exchange_energy,
    compute_ordered_properties,
    compute_properties,
    compute_volume_change,
    compute_average_voltage,
)


_SUPERCELL_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


@pytest.fixture
def mock_calc():
    return MockMLIPCalculator()


@pytest.fixture
def lco_supercell(lco_structure):
    s = lco_structure.copy()
    s.make_supercell(_SUPERCELL_222)
    return s


# ── LiCoO₂ proxy note ────────────────────────────────────────────────────────
# LiCoO₂ has no Ni. Tests for li_ni_exchange require a Ni-containing structure.

@pytest.fixture
def li_ni_structure():
    """Minimal Li/Ni structure for exchange energy tests."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.hexagonal(2.87, 14.2)
    # Simplified: Li at z=0.5, Ni at z=0.0, O at z=0.26/0.74
    species = ["Li", "Ni", "O", "O"]
    coords = [
        [0, 0, 0.5],
        [0, 0, 0.0],
        [0, 0, 0.26],
        [0, 0, 0.74],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def li_ni_supercell(li_ni_structure):
    s = li_ni_structure.copy()
    s.make_supercell(_SUPERCELL_222)
    return s


# ── Li/Ni exchange energy ─────────────────────────────────────────────────────

def test_li_ni_exchange_returns_float(li_ni_supercell, mock_calc):
    """compute_li_ni_exchange_energy returns a float on a Li+Ni structure."""
    result = compute_li_ni_exchange_energy(li_ni_supercell, mock_calc)
    assert isinstance(result, float)


def test_li_ni_exchange_no_ni_returns_none(lco_supercell, mock_calc):
    """LiCoO₂ has no Ni — must return None."""
    result = compute_li_ni_exchange_energy(lco_supercell, mock_calc)
    assert result is None


def test_li_ni_exchange_no_li_returns_none(li_ni_structure, mock_calc):
    """Structure without Li must return None."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.hexagonal(2.87, 14.2)
    # Only Ni and O
    s = Structure(lattice, ["Ni", "O", "O"], [
        [0, 0, 0.0],
        [0, 0, 0.26],
        [0, 0, 0.74],
    ])
    s.make_supercell(_SUPERCELL_222)
    result = compute_li_ni_exchange_energy(s, mock_calc)
    assert result is None


def test_swapped_structure_preserves_composition(li_ni_supercell, mock_calc):
    """After swap, structure has same number of Li and Ni as before."""
    n_li_before = sum(1 for s in li_ni_supercell if s.species_string == "Li")
    n_ni_before = sum(1 for s in li_ni_supercell if s.species_string == "Ni")

    # The function computes exchange; we verify it doesn't alter the original
    _ = compute_li_ni_exchange_energy(li_ni_supercell, mock_calc)

    n_li_after = sum(1 for s in li_ni_supercell if s.species_string == "Li")
    n_ni_after = sum(1 for s in li_ni_supercell if s.species_string == "Ni")
    assert n_li_after == n_li_before
    assert n_ni_after == n_ni_before


# ── Voltage ───────────────────────────────────────────────────────────────────

def test_voltage_returns_float(lco_supercell, mock_calc):
    """compute_average_voltage returns a float for LiCoO₂."""
    result = compute_average_voltage(lco_supercell, mock_calc)
    assert isinstance(result, float)


def test_voltage_no_li_returns_none(mock_calc):
    """Structure without Li returns None."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.0)
    s = Structure(lattice, ["Co"], [[0, 0, 0]])
    result = compute_average_voltage(s, mock_calc)
    assert result is None


def test_delithiated_structure_has_no_li(lco_supercell, mock_calc):
    """Verify that internal delithiation removes all Li."""
    from stages.stage5.property_calculator import _remove_species
    delith = _remove_species(lco_supercell, "Li")
    li_remaining = sum(1 for s in delith if s.species_string == "Li")
    assert li_remaining == 0


# ── Volume change ─────────────────────────────────────────────────────────────

def test_volume_change_returns_non_negative(lco_supercell, mock_calc):
    """Volume change must be non-negative (it's |ΔV|/V × 100)."""
    result = compute_volume_change(lco_supercell, mock_calc)
    assert result is not None
    assert result >= 0.0


def test_volume_change_is_float(lco_supercell, mock_calc):
    result = compute_volume_change(lco_supercell, mock_calc)
    assert isinstance(result, float)


def test_volume_change_no_li_returns_none(mock_calc):
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.0)
    s = Structure(lattice, ["Co"], [[0, 0, 0]])
    result = compute_volume_change(s, mock_calc)
    assert result is None


# ── Lattice parameters ────────────────────────────────────────────────────────

def test_lattice_params_returns_dict(lco_supercell, mock_calc):
    result = compute_lattice_params(lco_supercell, mock_calc)
    assert isinstance(result, dict)


def test_lattice_params_has_required_keys(lco_supercell, mock_calc):
    result = compute_lattice_params(lco_supercell, mock_calc)
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        assert key in result, f"Missing key: {key!r}"


def test_lattice_params_positive_lengths(lco_supercell, mock_calc):
    result = compute_lattice_params(lco_supercell, mock_calc)
    assert result["a"] > 0
    assert result["b"] > 0
    assert result["c"] > 0


# ── Formation energy ──────────────────────────────────────────────────────────

def test_formation_energy_returns_float(lco_supercell, mock_calc):
    result = compute_formation_energy_above_hull(lco_supercell, mock_calc)
    assert isinstance(result, float)


def test_formation_energy_uses_precomputed_value(lco_supercell, mock_calc):
    """If final_energy_per_atom kwarg is provided, it should be returned directly."""
    result = compute_formation_energy_above_hull(
        lco_supercell, mock_calc, final_energy_per_atom=-5.42
    )
    assert result == pytest.approx(-5.42)


# ── Property registry ─────────────────────────────────────────────────────────

def test_property_registry_all_callable():
    for name, fn in PROPERTY_REGISTRY.items():
        assert callable(fn), f"PROPERTY_REGISTRY[{name!r}] is not callable"


def test_property_registry_has_all_five_properties():
    expected = {"li_ni_exchange", "voltage", "formation_energy", "volume_change", "lattice_params"}
    assert expected == set(PROPERTY_REGISTRY.keys())


# ── compute_properties dispatcher ─────────────────────────────────────────────

def test_compute_properties_dispatches_subset(lco_supercell, mock_calc):
    """Only requested properties are computed."""
    result = compute_properties(
        relaxed_structure=lco_supercell,
        parent_structure=lco_supercell,
        calculator=mock_calc,
        target_properties=["lattice_params", "formation_energy"],
    )
    assert "lattice_params" in result
    assert "formation_energy" in result
    # Other properties should NOT be in result
    assert "voltage" not in result
    assert "li_ni_exchange" not in result


def test_compute_properties_unknown_property_returns_none(lco_supercell, mock_calc):
    result = compute_properties(
        relaxed_structure=lco_supercell,
        parent_structure=lco_supercell,
        calculator=mock_calc,
        target_properties=["nonexistent_property"],
    )
    assert result.get("nonexistent_property") is None


def test_compute_properties_empty_list(lco_supercell, mock_calc):
    result = compute_properties(
        relaxed_structure=lco_supercell,
        parent_structure=lco_supercell,
        calculator=mock_calc,
        target_properties=[],
    )
    assert result == {}


# ── Ordered-cell properties ───────────────────────────────────────────────────

def test_compute_ordered_properties_returns_dict(lco_structure, mock_calc):
    """compute_ordered_properties for Al@25% returns a property dict."""
    result = compute_ordered_properties(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        calculator=mock_calc,
        target_properties=["lattice_params", "formation_energy"],
        max_steps=50,
    )
    assert isinstance(result, dict)
    assert "lattice_params" in result
    assert "formation_energy" in result


def test_compute_ordered_properties_same_keys_as_disordered(lco_structure, mock_calc):
    """Ordered and disordered should produce the same property keys."""
    from stages.stage5.sqs_generator import generate_sqs
    from stages.stage5.mlip_relaxation import relax_structure

    target_props = ["lattice_params", "formation_energy"]

    # Ordered
    ordered = compute_ordered_properties(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        calculator=mock_calc,
        target_properties=target_props,
        max_steps=30,
    )

    # Disordered (SQS)
    sqs_list = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    relax_res = relax_structure(
        structure=sqs_list[0],
        calculator=mock_calc,
        fmax=0.1,
        max_steps=30,
        filter_type="None",
    )
    disordered = compute_properties(
        relaxed_structure=relax_res.relaxed_structure,
        parent_structure=lco_structure,
        calculator=mock_calc,
        target_properties=target_props,
    )

    assert set(ordered.keys()) == set(disordered.keys())
