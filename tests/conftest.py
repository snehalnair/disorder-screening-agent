"""
Shared pytest fixtures for the disorder-screening-agent test suite.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def lco_structure():
    """
    Minimal LiCoO₂ R-3m layered structure for testing.

    Uses LiCoO₂ as proxy for NMC: same space group (R-3m), same Co site
    geometry, simpler composition. Built programmatically — no MP API required.

    Unit cell contains 4 atoms: 1 Li, 1 Co, 2 O.
    In a 2×2×2 supercell: 8 Li, 8 Co, 16 O → 32 atoms total.
    """
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.hexagonal(2.816, 14.08)
    species = ["Li", "Co", "O", "O"]
    coords = [
        [0.0, 0.0, 0.5],       # Li 3b
        [0.0, 0.0, 0.0],        # Co 3a
        [0.0, 0.0, 0.2602],    # O 6c
        [0.0, 0.0, 0.7398],    # O 6c
    ]
    return Structure(lattice, species, coords)
