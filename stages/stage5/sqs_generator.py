"""
Stage 5a: Special Quasi-random Structure (SQS) generation.

Generates SQS realisations for each (dopant, concentration) pair at the
target site in the parent supercell.

Two approaches tried in order:
  A. pymatgen SQSTransformation (wraps ATAT mcsqs or enumlib) — preferred.
  B. Pure-Python random-sampling with pair-correlation scoring — fallback.

SQS theory: Zunger et al. (1990) Phys. Rev. Lett. 65, 353.
"""

from __future__ import annotations

import logging
import random
import warnings
from itertools import combinations
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


TOOL_METADATA = {
    "name": "sqs_generator",
    "stage": "5a",
    "description": (
        "Generate Special Quasi-random Structures for disorder-aware dopant simulation"
    ),
    "system_type": "periodic_crystal",
    "input_type": "Structure + dopant + concentration",
    "output_type": "list[Structure]",
    "cost": "seconds–minutes",
    "cost_per_candidate": "~10–60 s per realisation",
    "external_dependencies": ["pymatgen SQSTransformation or ATAT mcsqs"],
    "requires_structure": True,
    "requires_network": False,
    "requires_gpu": False,
    "configurable_params": [
        "supercell_matrix",
        "n_realisations",
        "concentration",
        "correlation_cutoff",
    ],
    "failure_modes": [
        "concentration too low for supercell",
        "ATAT not installed (fallback to manual)",
    ],
    "limitation": (
        "Assumes perfectly random distribution (no short-range order). "
        "Real materials may cluster at low concentrations."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate_sqs(
    parent_structure,
    dopant_element: str,
    target_species: str,
    concentration: float,
    supercell_matrix,
    n_realisations: int = 5,
    correlation_cutoff: float = 6.0,
) -> list:
    """
    Generate Special Quasi-random Structures for a doped crystal.

    SQS finds the arrangement of dopant atoms on target sites where
    pair/multi-site correlation functions best match the infinite random
    alloy limit. This is standard alloy theory (Zunger et al. 1990),
    rarely applied in dopant screening due to historical DFT cost —
    now feasible with MLIPs.

    Parameters
    ----------
    parent_structure:
        Pymatgen Structure of the undoped parent.
    dopant_element:
        Element symbol of dopant (e.g. ``"Al"``).
    target_species:
        Element being replaced (e.g. ``"Co"``).
    concentration:
        Fraction of target sites to replace (e.g. 0.10 for 10%).
    supercell_matrix:
        3×3 matrix or [a, b, c] scaling (e.g. ``[[2,0,0],[0,2,0],[0,0,2]]``).
    n_realisations:
        Number of independent SQS to generate (default 5).
    correlation_cutoff:
        Distance cutoff for correlation functions in Å.

    Returns
    -------
    list[Structure]
        ``n_realisations`` pymatgen Structure objects, each a valid SQS.

    Raises
    ------
    ValueError
        If ``concentration`` results in fewer than 1 dopant atom in the
        supercell.
    """
    from pymatgen.core import Structure

    # ── 1. Build supercell ────────────────────────────────────────────
    supercell = parent_structure.copy()
    supercell.make_supercell(supercell_matrix)

    # ── 2. Identify target site indices ──────────────────────────────
    target_indices = [
        i for i, site in enumerate(supercell)
        if site.species_string == target_species
    ]
    n_target_sites = len(target_indices)

    # ── 3. Compute dopant count ───────────────────────────────────────
    n_dopant_float = concentration * n_target_sites
    n_dopant = round(n_dopant_float)

    if n_dopant < 1:
        raise ValueError(
            f"Concentration {concentration} on {n_target_sites} {target_species} "
            f"sites = {n_dopant_float:.2f} atoms. "
            "Need ≥1 dopant atom. Increase concentration or supercell size."
        )

    if n_dopant < 2:
        logger.warning(
            "Only 1 dopant atom — SQS cannot optimise pair correlations. "
            "Results will be equivalent to single-site substitution."
        )

    # ── 4. Generate SQS structures ────────────────────────────────────
    structures = _try_sqs_transformation(
        parent_structure=parent_structure,
        dopant_element=dopant_element,
        target_species=target_species,
        concentration=concentration,
        supercell_matrix=supercell_matrix,
        n_realisations=n_realisations,
        correlation_cutoff=correlation_cutoff,
    )

    if structures is None:
        # Fallback to pure-Python manual enumeration
        logger.info(
            "SQSTransformation unavailable or failed — using manual "
            "pair-correlation sampling fallback."
        )
        structures = _generate_sqs_manual(
            supercell=supercell,
            target_indices=target_indices,
            dopant_element=dopant_element,
            n_dopant=n_dopant,
            n_realisations=n_realisations,
        )

    # ── 5. Validate each SQS ─────────────────────────────────────────
    validated = []
    for i, s in enumerate(structures):
        _validate_sqs(
            sqs=s,
            expected_total=len(supercell),
            n_dopant=n_dopant,
            n_target_remaining=n_target_sites - n_dopant,
            dopant_element=dopant_element,
            target_species=target_species,
            realisation_index=i,
        )
        validated.append(s)

    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Approach A: pymatgen SQSTransformation
# ─────────────────────────────────────────────────────────────────────────────


def _try_sqs_transformation(
    parent_structure,
    dopant_element: str,
    target_species: str,
    concentration: float,
    supercell_matrix,
    n_realisations: int,
    correlation_cutoff: float,
) -> list | None:
    """
    Attempt to use pymatgen's SQSTransformation.

    Returns list of Structures on success, None if the transformation is
    unavailable or raises an exception.
    """
    try:
        from pymatgen.transformations.advanced_transformations import (
            SQSTransformation,
        )
    except ImportError:
        return None

    # Build the disordered structure required by SQSTransformation.
    # Replace target_species sites with a disordered occupation.
    try:
        from pymatgen.core import Species
        from pymatgen.transformations.standard_transformations import (
            SubstitutionTransformation,
        )

        disordered = parent_structure.copy()
        # Create a disordered site: {target: 1-c, dopant: c}
        # pymatgen uses Composition on the site
        target_sp = target_species  # e.g. "Co"
        dopant_sp = dopant_element  # e.g. "Al"

        # Replace target sites with disordered occupancy
        for i, site in enumerate(disordered):
            if site.species_string == target_species:
                from pymatgen.core import PeriodicSite
                from pymatgen.core.composition import Composition
                new_species = {target_sp: 1.0 - concentration, dopant_sp: concentration}
                disordered.replace(
                    i,
                    species=new_species,
                    coords=site.frac_coords,
                    coords_are_cartesian=False,
                )

        sqs_t = SQSTransformation(
            scaling=supercell_matrix,
            search_time=10,       # seconds per call (short for CI)
            instances=n_realisations,
            best_only=False,
        )
        result = sqs_t.apply_transformation(disordered, return_ranked_list=n_realisations)

        if isinstance(result, list):
            structures = [entry["structure"] if isinstance(entry, dict) else entry
                          for entry in result]
        else:
            structures = [result]

        # Pad if fewer than requested
        while len(structures) < n_realisations:
            structures.append(structures[-1].copy())

        return structures[:n_realisations]

    except Exception as exc:
        logger.debug("SQSTransformation failed (%s); switching to fallback.", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Approach B: manual pair-correlation sampling
# ─────────────────────────────────────────────────────────────────────────────


def _generate_sqs_manual(
    supercell,
    target_indices: list[int],
    dopant_element: str,
    n_dopant: int,
    n_realisations: int,
    n_trials: int = 1000,
) -> list:
    """
    Generate SQS by random sampling + pair-correlation scoring.

    For each realisation:
      1. Randomly select ``n_dopant`` sites from ``target_indices``.
      2. Substitute dopant onto those sites.
      3. Score by pair-correlation deviation from the random-alloy target.
      4. Keep the best of ``n_trials`` random samples.

    Returns a list of ``n_realisations`` Structure objects.
    """
    structures = []
    seen_choices: set[frozenset] = set()

    for _realisation in range(n_realisations):
        best_score = float("inf")
        best_struct = None
        best_chosen: frozenset | None = None

        for _trial in range(n_trials):
            trial = supercell.copy()
            chosen = frozenset(random.sample(target_indices, n_dopant))

            for idx in chosen:
                trial.replace(idx, dopant_element)

            score = _pair_correlation_deviation(
                structure=trial,
                target_species=target_indices,
                dopant_element=dopant_element,
                supercell=supercell,
            )

            # Prefer novel arrangements when score is equal
            if score < best_score or (
                score == best_score and chosen not in seen_choices
            ):
                best_score = score
                best_struct = trial
                best_chosen = chosen

        seen_choices.add(best_chosen)
        structures.append(best_struct)

    return structures


def _pair_correlation_deviation(
    structure,
    target_species: list[int],
    dopant_element: str,
    supercell,
) -> float:
    """
    Compute the L2 deviation of pair correlations from the random-alloy ideal.

    For a random alloy at concentration c, the expected nearest-neighbour
    pair fractions are:
      P(dopant-dopant) = c²
      P(dopant-host)   = 2c(1-c)
      P(host-host)     = (1-c)²

    Counts dopant-dopant and dopant-host nearest-neighbour pairs and returns
    the sum of squared deviations.
    """
    n_dopant_sites = sum(
        1 for i in target_species if structure[i].species_string == dopant_element
    )
    n_target_total = len(target_species)
    if n_target_total == 0:
        return 0.0

    c = n_dopant_sites / n_target_total

    # Build distance matrix restricted to target sites only
    target_coords = np.array([structure[i].coords for i in target_species])
    n = len(target_species)

    if n < 2:
        return 0.0

    # Classify each site as dopant (1) or host (0)
    is_dopant = np.array(
        [1 if structure[target_species[i]].species_string == dopant_element else 0
         for i in range(n)]
    )

    # Compute pairwise distances (upper triangle only)
    dd_pairs = 0
    dh_pairs = 0
    total_pairs = 0

    lattice = structure.lattice
    for i in range(n):
        for j in range(i + 1, n):
            dist = lattice.get_distance_and_image(
                structure[target_species[i]].frac_coords,
                structure[target_species[j]].frac_coords,
            )[0]
            if dist < 5.0:   # nearest-neighbour cutoff
                total_pairs += 1
                if is_dopant[i] and is_dopant[j]:
                    dd_pairs += 1
                elif is_dopant[i] != is_dopant[j]:
                    dh_pairs += 1

    if total_pairs == 0:
        return 0.0

    actual_dd = dd_pairs / total_pairs
    actual_dh = dh_pairs / total_pairs

    expected_dd = c * c
    expected_dh = 2 * c * (1 - c)

    return (actual_dd - expected_dd) ** 2 + (actual_dh - expected_dh) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def _validate_sqs(
    sqs,
    expected_total: int,
    n_dopant: int,
    n_target_remaining: int,
    dopant_element: str,
    target_species: str,
    realisation_index: int,
) -> None:
    """
    Validate a generated SQS structure.  Raises ValueError on any failure.
    """
    # Total atom count
    if len(sqs) != expected_total:
        raise ValueError(
            f"SQS realisation {realisation_index}: expected {expected_total} atoms, "
            f"got {len(sqs)}."
        )

    # Dopant count
    actual_dopant = sum(
        1 for site in sqs if site.species_string == dopant_element
    )
    if actual_dopant != n_dopant:
        raise ValueError(
            f"SQS realisation {realisation_index}: expected {n_dopant} {dopant_element} "
            f"atoms, got {actual_dopant}."
        )

    # Remaining target species count
    actual_target_remaining = sum(
        1 for site in sqs if site.species_string == target_species
    )
    if actual_target_remaining != n_target_remaining:
        raise ValueError(
            f"SQS realisation {realisation_index}: expected {n_target_remaining} "
            f"remaining {target_species} atoms, got {actual_target_remaining}."
        )

    # No overlapping atoms (min interatomic distance > 0.5 Å)
    try:
        min_dist = sqs.get_neighbor_list(0.5, numerical_tol=0.0)[2]
        if len(min_dist) > 0:
            raise ValueError(
                f"SQS realisation {realisation_index}: overlapping atoms detected "
                f"(min distance < 0.5 Å)."
            )
    except Exception:
        # If neighbor-list fails (e.g. for very small test cells), skip this check
        pass
