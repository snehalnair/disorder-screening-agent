"""
Stage 5c: Battery property calculation on relaxed structures.

Computes five target properties via a registry pattern. New properties are
added by writing a function and inserting it into PROPERTY_REGISTRY — nothing
else changes.

Properties
----------
li_ni_exchange    : Li↔Ni anti-site defect formation energy (eV). Higher = better ordered.
voltage           : Average intercalation voltage (V). Derived from delithiation energy.
formation_energy  : Raw energy per atom (eV/atom). Proxy for thermodynamic stability.
volume_change     : |ΔV|/V₀ × 100 on delithiation (%). Lower = better structural stability.
lattice_params    : {"a", "b", "c", "alpha", "beta", "gamma"} from relaxed lattice.

Also exposes compute_ordered_properties() for the ordered-cell baseline comparison —
the canonical comparison against which disorder sensitivity is measured.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


TOOL_METADATA = {
    "name": "property_calculator",
    "stage": "5c",
    "description": (
        "Compute target properties on relaxed structures "
        "(both disordered SQS and ordered reference)"
    ),
    "system_type": "periodic_crystal",
    "input_type": "Structure + MLIPCalculator + property list",
    "output_type": "dict[str, float]",
    "cost": "seconds–minutes",
    "cost_per_candidate": "~1–5 min (includes single-point calculations)",
    "external_dependencies": ["ase", "mlip calculator"],
    "requires_structure": True,
    "requires_network": False,
    "requires_gpu": True,
    "configurable_params": ["target_properties"],
    "failure_modes": [
        "Li/Ni not found for exchange energy",
        "delithiation fails",
        "element not in MLIP training set",
    ],
    "supported_properties": [
        "li_ni_exchange",
        "voltage",
        "formation_energy",
        "volume_change",
        "lattice_params",
    ],
}

# Reference Li chemical potential (bcc Li metal, standard DFT value) eV/atom
_E_LI_REF = -1.9

# Reference O₂ energy (MACE-MPA-0 value for isolated O₂ molecule / 2) eV/atom
# Standard DFT value for O₂: ~ -4.93 eV/atom (PBE). MACE may differ slightly
# but cancels in relative rankings (all dopants use the same reference).
_E_O_REF = -4.93


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_energy(structure, calculator) -> float:
    """Single-point MLIP energy for a pymatgen Structure."""
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    # Use per-element fallback if available
    if hasattr(calculator, "get_calculator_for_atoms"):
        calc = calculator.get_calculator_for_atoms(atoms)
    elif hasattr(calculator, "get_calculator"):
        calc = calculator.get_calculator()
    else:
        calc = calculator
    atoms.calc = calc
    return atoms.get_potential_energy()


def _remove_species(structure, species: str):
    """Return a copy of structure with all sites of given species removed."""
    indices_to_remove = [
        i for i, site in enumerate(structure)
        if site.species_string == species
    ]
    if not indices_to_remove:
        return structure.copy()
    s = structure.copy()
    s.remove_sites(indices_to_remove)
    return s


def _quick_relax(structure, calculator, max_steps: int = 200):
    """Quick cell + ionic relaxation for delithiated structures.

    Uses FrechetCellFilter (default) so that volume_change reflects
    real lattice response to delithiation.  max_steps raised from 100
    to 200 to accommodate the extra degrees of freedom.
    """
    from stages.stage5.mlip_relaxation import relax_structure

    result = relax_structure(
        structure=structure,
        calculator=calculator,
        fmax=0.1,          # looser criterion for quick relaxation
        max_steps=max_steps,
        # Uses default filter_type="FrechetCellFilter" for cell + ionic relaxation.
    )
    return result.relaxed_structure


# ─────────────────────────────────────────────────────────────────────────────
# Property functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_li_ni_exchange_energy(
    structure,
    calculator,
    **kwargs,
) -> Optional[float]:
    """
    Compute the energy cost of swapping one Li and one Ni between layers.

    In an R-3m layered cathode, Li occupies 3b sites (z ≈ 0.5) and Ni
    occupies 3a sites (z ≈ 0.0). Anti-site disorder swaps them. The energy
    cost indicates how prone the material is to Li/Ni mixing.

    Higher values = more energy to disorder = better cathode performance.
    Negative values = disorder thermodynamically favoured = very bad.

    Returns
    -------
    float
        Exchange energy in eV, or None if Li or Ni not found.
    """
    # Find Li and Ni sites
    li_sites = [i for i, s in enumerate(structure) if s.species_string == "Li"]
    ni_sites = [i for i, s in enumerate(structure) if s.species_string == "Ni"]

    if not li_sites:
        logger.warning("compute_li_ni_exchange_energy: no Li found in structure — returning None.")
        return None
    if not ni_sites:
        logger.warning("compute_li_ni_exchange_energy: no Ni found in structure — returning None.")
        return None

    # Pick one Li (prefer z ≈ 0.5) and one Ni (prefer z ≈ 0.0)
    li_idx = min(li_sites, key=lambda i: abs(structure[i].frac_coords[2] - 0.5))
    ni_idx = min(ni_sites, key=lambda i: abs(structure[i].frac_coords[2] - 0.0))

    # Compute original energy
    e_orig = _get_energy(structure, calculator)

    # Build swapped structure
    swapped = structure.copy()
    li_coords = swapped[li_idx].frac_coords.copy()
    ni_coords = swapped[ni_idx].frac_coords.copy()

    swapped.replace(li_idx, "Ni", coords=ni_coords, coords_are_cartesian=False)
    swapped.replace(ni_idx, "Li", coords=li_coords, coords_are_cartesian=False)

    # Compute swapped energy (single-point)
    e_swap = _get_energy(swapped, calculator)

    return float(e_swap - e_orig)


def compute_average_voltage(
    structure,
    calculator,
    **kwargs,
) -> Optional[float]:
    """
    Compute average intercalation voltage via delithiation.

    V = -(E_delith - E_lith + n_Li × E_Li_ref) / n_Li

    Returns
    -------
    float
        Average voltage in V, or None if no Li present.
    """
    n_li = sum(1 for s in structure if s.species_string == "Li")
    if n_li == 0:
        logger.warning("compute_average_voltage: no Li found — returning None.")
        return None

    e_lith = _get_energy(structure, calculator)

    delith = _remove_species(structure, "Li")
    if len(delith) == 0:
        logger.warning("compute_average_voltage: delithiation left empty structure — returning None.")
        return None

    e_delith = _get_energy(delith, calculator)

    # Standard reference energy for Li metal
    e_li_ref_total = _E_LI_REF * n_li

    voltage = -(e_delith - e_lith + e_li_ref_total) / n_li
    return float(voltage)


def compute_oxygen_vacancy_energy(
    structure,
    calculator,
    **kwargs,
) -> Optional[float]:
    """
    Compute oxygen vacancy formation energy (eV).

    E_vac = E(structure - 1 O) + E_O_ref - E(structure)

    Removes the O atom nearest to a dopant site (most relevant vacancy)
    and computes the energy cost. Lower = easier to form vacancies =
    better ionic conductor / more catalytically active.

    For CeO₂ doping studies, this is the primary screening property
    (analogous to voltage for battery cathodes).

    Returns
    -------
    float
        Vacancy formation energy in eV, or None if no O present.
    """
    o_sites = [i for i, s in enumerate(structure) if s.species_string == "O"]
    if not o_sites:
        logger.warning("compute_oxygen_vacancy_energy: no O found — returning None.")
        return None

    # Find dopant sites (non-host, non-O atoms that aren't Sr/Ba/Li etc.)
    # Heuristic: find the minority non-O species that isn't the majority cation
    species_counts = {}
    for s in structure:
        sp = s.species_string
        if sp != "O":
            species_counts[sp] = species_counts.get(sp, 0) + 1

    if not species_counts:
        logger.warning("compute_oxygen_vacancy_energy: no cations found — returning None.")
        return None

    # Dopant = least frequent non-O species
    dopant_sp = min(species_counts, key=species_counts.get)
    dopant_sites = [i for i, s in enumerate(structure) if s.species_string == dopant_sp]

    # Find O nearest to any dopant
    best_o = None
    best_dist = float("inf")
    for o_idx in o_sites:
        for d_idx in dopant_sites:
            d = structure.get_distance(o_idx, d_idx)
            if d < best_dist:
                best_dist = d
                best_o = o_idx

    if best_o is None:
        return None

    e_pristine = _get_energy(structure, calculator)

    # Remove one O
    defect = structure.copy()
    defect.remove_sites([best_o])

    # Quick relax the defect structure
    defect_relaxed = _quick_relax(defect, calculator, max_steps=200)
    e_defect = _get_energy(defect_relaxed, calculator)

    e_vac = e_defect + _E_O_REF - e_pristine
    return float(e_vac)


def compute_formation_energy_above_hull(
    structure,
    calculator,
    **kwargs,
) -> Optional[float]:
    """
    Return the energy per atom of the relaxed structure.

    This is a proxy for formation energy; true hull-distance calculation
    requires Materials Project elemental reference energies and phase diagram
    data (Phase 6 scope). The raw E/atom is already in RelaxationResult
    if available via kwargs.

    Returns
    -------
    float
        Total MLIP energy per atom (eV/atom).
    """
    # If pre-computed final energy is passed through kwargs, use it directly
    final_energy = kwargs.get("final_energy_per_atom")
    if final_energy is not None:
        return float(final_energy)

    e_total = _get_energy(structure, calculator)
    return float(e_total / len(structure))


def compute_volume_change(
    structure,
    calculator,
    **kwargs,
) -> Optional[float]:
    """
    Compute volume change on delithiation (%).

    |V_delith - V_lith| / V_lith × 100

    For Li-containing structures, this measures structural stability under cycling.
    For non-Li structures (e.g. SrTiO3), falls back to doping_volume_change
    (volume distortion from doping relative to undoped parent).

    Returns
    -------
    float
        Volume change in percent (non-negative), or None if calculation fails.
    """
    n_li = sum(1 for s in structure if s.species_string == "Li")

    if n_li == 0:
        # Fall back to doping volume change for non-Li systems
        return compute_doping_volume_change(structure, calculator, **kwargs)

    v_lith = structure.volume
    delith = _remove_species(structure, "Li")

    if len(delith) == 0:
        return None

    # Quick relaxation on delithiated structure
    try:
        relaxed_delith = _quick_relax(delith, calculator)
        v_delith = relaxed_delith.volume
    except Exception as exc:
        logger.warning("compute_volume_change: delithiated relaxation failed (%s). Using unrelaxed volume.", exc)
        v_delith = delith.volume

    if v_lith == 0:
        return None

    return float(abs(v_delith - v_lith) / v_lith * 100.0)


def compute_doping_volume_change(
    structure,
    calculator,
    **kwargs,
) -> Optional[float]:
    """
    Compute volume distortion from doping relative to undoped parent (%).

    |V_doped - V_parent| / V_parent × 100

    Used for non-battery systems (e.g. SrTiO3) where delithiation volume
    change is not applicable. Measures how much the dopant strains the lattice.

    Requires ``parent_structure`` in kwargs and assumes it has been expanded
    to the same supercell size as the doped structure.

    Returns
    -------
    float
        Volume change in percent (non-negative), or None if parent not available.
    """
    parent = kwargs.get("parent_structure")
    if parent is None:
        logger.warning("compute_doping_volume_change: no parent_structure in kwargs — returning None.")
        return None

    # The parent_structure from kwargs is the primitive cell.
    # Estimate supercell volume by atom count ratio.
    n_doped = len(structure)
    n_parent = len(parent)
    if n_parent == 0:
        return None

    # Scale parent volume to same number of formula units
    scale = n_doped / n_parent
    v_parent_scaled = parent.volume * scale
    v_doped = structure.volume

    if v_parent_scaled == 0:
        return None

    return float(abs(v_doped - v_parent_scaled) / v_parent_scaled * 100.0)


def compute_lattice_params(
    structure,
    calculator,
    **kwargs,
) -> Optional[dict]:
    """
    Extract lattice parameters from a relaxed structure.

    Returns
    -------
    dict
        {"a", "b", "c", "alpha", "beta", "gamma"} in Å / degrees.
    """
    lat = structure.lattice
    return {
        "a": float(lat.a),
        "b": float(lat.b),
        "c": float(lat.c),
        "alpha": float(lat.alpha),
        "beta": float(lat.beta),
        "gamma": float(lat.gamma),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registry and dispatcher
# ─────────────────────────────────────────────────────────────────────────────

PROPERTY_REGISTRY: dict[str, callable] = {
    "li_ni_exchange": compute_li_ni_exchange_energy,
    "voltage": compute_average_voltage,
    "formation_energy": compute_formation_energy_above_hull,
    "volume_change": compute_volume_change,
    "oxygen_vacancy": compute_oxygen_vacancy_energy,
    "lattice_params": compute_lattice_params,
}


def compute_properties(
    relaxed_structure,
    parent_structure,
    calculator,
    target_properties: list[str],
    baseline: Optional[dict] = None,
    final_energy_per_atom: Optional[float] = None,
) -> dict:
    """
    Compute target properties on a relaxed (doped) structure.

    Parameters
    ----------
    relaxed_structure:
        Relaxed doped structure from Stage 5b.
    parent_structure:
        Original undoped parent (for reference; passed through to property fns).
    calculator:
        MLIP calculator (real or mock).
    target_properties:
        List of property names from PROPERTY_REGISTRY.
    baseline:
        Undoped baseline properties dict (optional, for disorder sensitivity).
    final_energy_per_atom:
        Pre-computed E/atom from the relaxation result (avoids redundant MLIP call).

    Returns
    -------
    dict
        {property_name: value} for each requested property.
        Value is None for any property that fails.
    """
    results: dict = {}
    kwargs = {
        "parent_structure": parent_structure,
        "baseline": baseline,
        "final_energy_per_atom": final_energy_per_atom,
    }

    for prop in target_properties:
        fn = PROPERTY_REGISTRY.get(prop)
        if fn is None:
            logger.warning("Unknown property %r — skipping.", prop)
            results[prop] = None
            continue
        try:
            results[prop] = fn(relaxed_structure, calculator, **kwargs)
        except Exception as exc:
            logger.warning("Property %r failed: %s", prop, exc)
            results[prop] = None

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ordered-cell comparison
# ─────────────────────────────────────────────────────────────────────────────


def compute_ordered_properties(
    parent_structure,
    dopant_element: str,
    target_species: str,
    concentration: float,
    supercell_matrix,
    calculator,
    target_properties: list[str],
    fmax: float = 0.05,
    max_steps: int = 200,
) -> dict:
    """
    Compute properties using a simple ordered substitution (not SQS).

    Places dopant atoms at maximally-separated target sites (farthest-first),
    relaxes, and computes properties. This represents what a conventional
    ordered-cell screening study would do.

    The thesis rests on comparing these values to the SQS (disordered) values:
    ordered substitution places dopants as far apart as possible (minimising
    dopant-dopant interaction), while SQS reproduces random statistics.

    Parameters
    ----------
    parent_structure, dopant_element, target_species, concentration,
    supercell_matrix, calculator, target_properties, fmax, max_steps:
        Same semantics as generate_sqs() and relax_structure().

    Returns
    -------
    dict
        {property_name: value} — same format as compute_properties().
    """
    from stages.stage5.mlip_relaxation import relax_structure

    # ── 1. Build supercell ────────────────────────────────────────────
    supercell = parent_structure.copy()
    supercell.make_supercell(supercell_matrix)

    # ── 2. Identify target sites ──────────────────────────────────────
    target_indices = [
        i for i, site in enumerate(supercell)
        if site.species_string == target_species
    ]
    n_target = len(target_indices)
    n_dopant = max(1, round(concentration * n_target))

    if n_dopant > n_target:
        raise ValueError(
            f"n_dopant ({n_dopant}) exceeds n_target ({n_target}) sites."
        )

    # ── 3. Farthest-first selection of dopant sites ───────────────────
    chosen = _farthest_first_selection(supercell, target_indices, n_dopant)

    # ── 4. Substitute dopant ──────────────────────────────────────────
    ordered = supercell.copy()
    for idx in chosen:
        ordered.replace(idx, dopant_element)

    # ── 5. Relax (cell + ionic via FrechetCellFilter) ────────────────
    relax_res = relax_structure(
        structure=ordered,
        calculator=calculator,
        fmax=fmax,
        max_steps=max_steps,
        # Uses default filter_type="FrechetCellFilter" for cell + ionic relaxation.
        # Must match the filter used in eval_disorder.py SQS relaxations.
    )

    # ── 6. Compute properties ─────────────────────────────────────────
    return compute_properties(
        relaxed_structure=relax_res.relaxed_structure,
        parent_structure=parent_structure,
        calculator=calculator,
        target_properties=target_properties,
        final_energy_per_atom=relax_res.final_energy_per_atom,
    )


def _farthest_first_selection(structure, candidate_indices: list[int], n: int) -> list[int]:
    """
    Greedily select ``n`` sites from ``candidate_indices`` such that
    selected sites are as far apart as possible (farthest-first traversal).

    Starts from the first candidate and iteratively adds the candidate
    that maximises the minimum distance to the already-selected set.
    """
    if n >= len(candidate_indices):
        return list(candidate_indices)

    selected = [candidate_indices[0]]
    remaining = list(candidate_indices[1:])

    for _ in range(n - 1):
        best_idx = None
        best_min_dist = -1.0
        for cidx in remaining:
            min_dist = min(
                structure.get_distance(cidx, s)
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = cidx
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected
