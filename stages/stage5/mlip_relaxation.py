"""
Stage 5b: MLIP relaxation of SQS structures.

Relaxes each SQS realisation using a machine-learned interatomic potential
(MatterSim primary, MACE-MP-0 fallback). ``RelaxationMonitor`` aborts on
energy divergence, volume explosion, stagnation, or force spikes.

Public API
----------
relax_structure(structure, calculator, ...)  →  RelaxationResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


TOOL_METADATA = {
    "name": "mlip_relaxation",
    "stage": "5b",
    "description": (
        "Relax doped structures using machine-learned interatomic potentials "
        "(MatterSim or MACE-MP-0)"
    ),
    "system_type": "periodic_crystal",
    "input_type": "Structure + MLIPCalculator",
    "output_type": "RelaxationResult",
    "cost": "minutes–hours",
    "cost_per_candidate": "~10–30 min per 96-atom cell on GPU",
    "external_dependencies": ["mattersim or mace-torch", "ase"],
    "requires_structure": True,
    "requires_network": False,
    "requires_gpu": True,
    "configurable_params": ["mlip", "fmax", "max_steps", "optimizer", "filter_type"],
    "failure_modes": [
        "energy_divergence",
        "volume_explosion",
        "stagnation",
        "force_spike",
        "MLIP not installed",
    ],
}


@dataclass
class RelaxationResult:
    """Output of a single MLIP relaxation run."""

    relaxed_structure: object                     # pymatgen Structure
    initial_energy_per_atom: float
    final_energy_per_atom: float
    relaxation_converged: bool
    relaxation_steps: int
    max_force_final: float
    abort_reason: Optional[str]                   # None if converged cleanly
    monitor_history: list[dict] = field(default_factory=list)


def relax_structure(
    structure,
    calculator,
    fmax: float = 0.05,
    max_steps: int = 500,
    optimizer_name: str = "BFGS",
    monitor_config: Optional[dict] = None,
    filter_type: str = "FrechetCellFilter",
) -> RelaxationResult:
    """
    Relax a structure using an MLIP calculator with monitoring.

    Parameters
    ----------
    structure:
        Input pymatgen Structure.
    calculator:
        ``MLIPCalculator`` instance (real or mock) **or** a raw ASE
        Calculator object (for testing with ``InjectableCalculator``).
    fmax:
        Force convergence criterion in eV/Å.
    max_steps:
        Maximum optimisation steps.
    optimizer_name:
        ``"BFGS"`` (default) or ``"FIRE"`` for difficult systems.
    monitor_config:
        Dict of ``RelaxationMonitor`` keyword arguments. If ``None``,
        defaults are used.
    filter_type:
        ``"FrechetCellFilter"`` for simultaneous cell + ionic relaxation,
        ``"None"`` for ionic positions only.

    Returns
    -------
    RelaxationResult
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from stages.stage5.monitoring import RelaxationAborted, RelaxationMonitor

    # ── 1. pymatgen Structure → ASE Atoms ────────────────────────────
    atoms = AseAtomsAdaptor.get_atoms(structure)

    # ── 2. Attach calculator ─────────────────────────────────────────
    # Support both MLIPCalculator ABC and raw ASE calculators.
    # Use get_calculator_for_atoms if available (allows per-element fallback).
    if hasattr(calculator, "get_calculator_for_atoms"):
        ase_calc = calculator.get_calculator_for_atoms(atoms)
    elif hasattr(calculator, "get_calculator"):
        ase_calc = calculator.get_calculator()
    else:
        ase_calc = calculator   # already a raw ASE calculator
    atoms.calc = ase_calc

    # ── 3. Record initial energy ─────────────────────────────────────
    initial_energy_per_atom = atoms.get_potential_energy() / len(atoms)

    # ── 4. Apply cell filter ─────────────────────────────────────────
    if filter_type == "FrechetCellFilter":
        # FrechetCellFilter location changed between ASE versions
        try:
            from ase.filters import FrechetCellFilter
        except ImportError:
            try:
                from ase.constraints import FrechetCellFilter
            except ImportError:
                FrechetCellFilter = None
        if FrechetCellFilter is not None:
            filtered_atoms = FrechetCellFilter(atoms)
        else:
            # Last resort: no cell filter (positions only)
            filtered_atoms = atoms
    else:
        filtered_atoms = atoms

    # ── 5. Create optimizer ──────────────────────────────────────────
    from ase.optimize import BFGS, FIRE

    optimizer_cls = FIRE if optimizer_name.upper() == "FIRE" else BFGS
    optimizer = optimizer_cls(filtered_atoms, logfile=None)

    # ── 6. Attach RelaxationMonitor ──────────────────────────────────
    monitor = RelaxationMonitor(**(monitor_config or {}))
    optimizer.attach(monitor, interval=1, atoms=atoms)

    # ── 7. Run relaxation ────────────────────────────────────────────
    converged = False
    abort_reason = None

    try:
        optimizer.run(fmax=fmax, steps=max_steps)
        # nsteps < max_steps → converged; otherwise hit step limit
        converged = optimizer.nsteps < max_steps
    except RelaxationAborted as exc:
        converged = False
        abort_reason = exc.reason

    # ── 8. Extract results ───────────────────────────────────────────
    final_energy_per_atom = atoms.get_potential_energy() / len(atoms)

    forces = atoms.get_forces()
    max_force_final = float(np.max(np.linalg.norm(forces, axis=1)))

    from pymatgen.io.ase import AseAtomsAdaptor
    relaxed_structure = AseAtomsAdaptor.get_structure(atoms)

    return RelaxationResult(
        relaxed_structure=relaxed_structure,
        initial_energy_per_atom=initial_energy_per_atom,
        final_energy_per_atom=final_energy_per_atom,
        relaxation_converged=converged,
        relaxation_steps=optimizer.nsteps,
        max_force_final=max_force_final,
        abort_reason=abort_reason,
        monitor_history=list(monitor.history),
    )


def run_mlip_relaxation(state: dict) -> dict:
    """LangGraph node: MLIP relaxation. Delegates to relax_structure()."""
    raise NotImplementedError(
        "run_mlip_relaxation graph node not yet wired (Phase 3 Task 5)."
    )
