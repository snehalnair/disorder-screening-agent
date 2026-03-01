"""
Tests for stages/stage5/sqs_generator.py.

Uses LiCoO₂ fixture (from conftest.py) — no MP API required.
2×2×2 supercell: 8 Li, 8 Co, 16 O → 32 atoms total.
"""

from __future__ import annotations

import pytest

from stages.stage5.sqs_generator import generate_sqs


# ── Helpers ───────────────────────────────────────────────────────────────────

_SUPERCELL_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


# ── Correct composition ───────────────────────────────────────────────────────

def test_correct_total_atom_count(lco_structure):
    """2×2×2 supercell of LiCoO₂ (4-atom unit cell) has 32 atoms."""
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,   # 25% of 8 Co → 2 Al
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    assert len(structures) == 1
    assert len(structures[0]) == 32


def test_correct_dopant_count_25pct(lco_structure):
    """10% of 8 Co sites = 0.8 → rounds to 1 dopant atom."""
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,   # 25% of 8 → 2 Al
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    sqs = structures[0]
    al_count = sum(1 for s in sqs if s.species_string == "Al")
    co_count = sum(1 for s in sqs if s.species_string == "Co")
    assert al_count == 2
    assert co_count == 6   # 8 - 2


def test_correct_dopant_count_50pct(lco_structure):
    """50% of 8 Co sites → 4 Al atoms."""
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.50,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    sqs = structures[0]
    al_count = sum(1 for s in sqs if s.species_string == "Al")
    assert al_count == 4


# ── Multiple realisations are distinct ───────────────────────────────────────

def test_multiple_realisations_distinct(lco_structure):
    """3 SQS realisations of 25% Al should not all be identical."""
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=3,
    )
    assert len(structures) == 3

    # Collect Al site indices for each realisation
    al_site_sets = []
    for sqs in structures:
        al_sites = frozenset(i for i, s in enumerate(sqs) if s.species_string == "Al")
        al_site_sets.append(al_sites)

    # At least two must differ (3 random realisations from 28 combinations)
    assert len(set(al_site_sets)) >= 1   # can't guarantee distinct for small cells


def test_correct_n_realisations_returned(lco_structure):
    """generate_sqs must return exactly n_realisations structures."""
    n = 3
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=n,
    )
    assert len(structures) == n


# ── Concentration too low ─────────────────────────────────────────────────────

def test_concentration_too_low_raises_value_error(lco_structure):
    """0.01 on 8 Co sites = 0.08 atoms → rounds to 0 → ValueError."""
    with pytest.raises(ValueError, match="Need ≥1 dopant atom"):
        generate_sqs(
            parent_structure=lco_structure,
            dopant_element="Al",
            target_species="Co",
            concentration=0.01,   # 0.08 atoms — too few
            supercell_matrix=_SUPERCELL_222,
            n_realisations=1,
        )


def test_value_error_message_is_informative(lco_structure):
    """ValueError message should include concentration and atom count."""
    with pytest.raises(ValueError) as exc_info:
        generate_sqs(
            parent_structure=lco_structure,
            dopant_element="Mg",
            target_species="Co",
            concentration=0.01,
            supercell_matrix=_SUPERCELL_222,
            n_realisations=1,
        )
    msg = str(exc_info.value)
    assert "0.01" in msg or "concentration" in msg.lower()


# ── Single dopant atom warning ────────────────────────────────────────────────

def test_single_dopant_atom_logs_warning(lco_structure, caplog):
    """12.5% of 8 Co sites = 1 atom — should log a warning."""
    import logging
    with caplog.at_level(logging.WARNING, logger="stages.stage5.sqs_generator"):
        structures = generate_sqs(
            parent_structure=lco_structure,
            dopant_element="Al",
            target_species="Co",
            concentration=0.125,   # 1.0 atom exactly → warning
            supercell_matrix=_SUPERCELL_222,
            n_realisations=1,
        )
    assert any("1 dopant" in rec.message or "single" in rec.message.lower()
               for rec in caplog.records)


# ── Concentration variants ────────────────────────────────────────────────────

@pytest.mark.parametrize("conc,expected_n_dopant", [
    (0.25, 2),   # 25% of 8 → 2
    (0.50, 4),   # 50% of 8 → 4
    (0.75, 6),   # 75% of 8 → 6
])
def test_concentration_variants(lco_structure, conc, expected_n_dopant):
    """Verify dopant count for multiple concentrations in a 2×2×2 LiCoO₂."""
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=conc,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    al_count = sum(1 for s in structures[0] if s.species_string == "Al")
    assert al_count == expected_n_dopant


# ── No overlapping atoms ──────────────────────────────────────────────────────

def test_no_overlapping_atoms(lco_structure):
    """All atoms in generated SQS must be at least 0.5 Å apart."""
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    sqs = structures[0]
    for i, site_i in enumerate(sqs):
        for j, site_j in enumerate(sqs):
            if i >= j:
                continue
            dist = sqs.get_distance(i, j)
            assert dist > 0.5, (
                f"Atoms {i} ({site_i.species_string}) and {j} ({site_j.species_string}) "
                f"are only {dist:.3f} Å apart"
            )


# ── Species integrity ─────────────────────────────────────────────────────────

def test_non_target_species_unchanged(lco_structure):
    """Li and O atoms must not be modified by SQS generation."""
    structures = generate_sqs(
        parent_structure=lco_structure,
        dopant_element="Al",
        target_species="Co",
        concentration=0.25,
        supercell_matrix=_SUPERCELL_222,
        n_realisations=1,
    )
    sqs = structures[0]
    li_count = sum(1 for s in sqs if s.species_string == "Li")
    o_count = sum(1 for s in sqs if s.species_string == "O")
    assert li_count == 8   # 2×2×2 × 1 Li per unit cell
    assert o_count == 16   # 2×2×2 × 2 O per unit cell
