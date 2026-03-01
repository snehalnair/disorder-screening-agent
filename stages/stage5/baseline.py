"""
Stage 5 baseline computation.

Relaxes the undoped parent structure in the same supercell as the doped
simulations to provide a reference for property comparisons.

The disorder sensitivity metric is:
    sensitivity = |property_doped - property_undoped| / |property_undoped|

Using the same supercell, MLIP, and convergence parameters as the doped
runs ensures apples-to-apples comparisons.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_baseline(
    parent_structure,
    supercell_matrix,
    calculator,
    target_properties: Optional[list[str]] = None,
    fmax: float = 0.05,
    max_steps: int = 500,
) -> dict:
    """
    Relax the undoped parent supercell and compute reference properties.

    Parameters
    ----------
    parent_structure:
        Pymatgen Structure of the undoped host material.
    supercell_matrix:
        3×3 matrix or [a, b, c] scaling applied to ``parent_structure``.
        Must match the supercell used for doped simulations.
    calculator:
        ``MLIPCalculator`` instance (real or mock) or a raw ASE Calculator.
    target_properties:
        List of property names to record in the output dict.  Currently
        records ``energy_per_atom`` and ``volume`` by default.  Extension
        to voltage etc. is Phase 4 scope.
    fmax:
        Force convergence criterion in eV/Å. Must match doped runs.
    max_steps:
        Maximum optimisation steps. Must match doped runs.

    Returns
    -------
    dict with keys:
        energy_per_atom        — eV/atom of the relaxed undoped supercell.
        volume                 — Å³ of the relaxed undoped supercell.
        lattice_params         — {"a": float, "b": float, "c": float} in Å.
        properties             — {prop: value} for requested target_properties.
        relaxation_converged   — bool.
        relaxation_steps       — int.
        abort_reason           — str or None.
    """
    from stages.stage5.mlip_relaxation import relax_structure

    if target_properties is None:
        target_properties = []

    # ── 1. Build supercell ────────────────────────────────────────────
    supercell = parent_structure.copy()
    supercell.make_supercell(supercell_matrix)
    n_atoms = len(supercell)

    logger.info(
        "Computing baseline for undoped %s supercell (%d atoms).",
        parent_structure.composition.reduced_formula,
        n_atoms,
    )

    # ── 2. Relax undoped supercell ────────────────────────────────────
    relax_result = relax_structure(
        structure=supercell,
        calculator=calculator,
        fmax=fmax,
        max_steps=max_steps,
    )

    relaxed = relax_result.relaxed_structure

    # ── 3. Extract lattice parameters ─────────────────────────────────
    lattice = relaxed.lattice
    lattice_params = {
        "a": lattice.a,
        "b": lattice.b,
        "c": lattice.c,
    }

    # ── 4. Placeholder property values (Phase 4 will fill these in) ───
    properties: dict = {}
    for prop in target_properties:
        properties[prop] = None   # to be computed by property_calculator

    logger.info(
        "Baseline relaxation %s in %d steps. E/atom = %.4f eV.",
        "converged" if relax_result.relaxation_converged else "NOT converged",
        relax_result.relaxation_steps,
        relax_result.final_energy_per_atom,
    )

    return {
        "energy_per_atom": relax_result.final_energy_per_atom,
        "volume": relaxed.volume,
        "lattice_params": lattice_params,
        "properties": properties,
        "relaxation_converged": relax_result.relaxation_converged,
        "relaxation_steps": relax_result.relaxation_steps,
        "abort_reason": relax_result.abort_reason,
    }
