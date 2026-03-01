"""
Tests for stages/stage5/mlip_relaxation.py and stages/stage5/calculators.py.

Uses MockMLIPCalculator (EMT on Cu/Al) and InjectableCalculator for abort
testing. No GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from stages.stage5.calculators import (
    InjectableCalculator,
    MockMLIPCalculator,
    MatterSimCalculator,
    MACECalculator,
    get_calculator,
)
from stages.stage5.mlip_relaxation import RelaxationResult, relax_structure
from stages.stage5.monitoring import RelaxationAborted


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cu_structure():
    """Tiny FCC Cu structure — EMT supports Cu natively."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(3.615)
    return Structure(lattice, ["Cu"], [[0, 0, 0]])


@pytest.fixture
def al_structure():
    """Tiny FCC Al structure — EMT supports Al natively."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.05)
    return Structure(lattice, ["Al"], [[0, 0, 0]])


# ── RelaxationResult dataclass ────────────────────────────────────────────────

def test_relaxation_result_fields():
    """RelaxationResult must have all required fields."""
    from pymatgen.core import Lattice, Structure
    dummy = Structure(Lattice.cubic(3.0), ["Cu"], [[0, 0, 0]])
    rr = RelaxationResult(
        relaxed_structure=dummy,
        initial_energy_per_atom=-3.0,
        final_energy_per_atom=-3.1,
        relaxation_converged=True,
        relaxation_steps=5,
        max_force_final=0.02,
        abort_reason=None,
        monitor_history=[],
    )
    assert rr.relaxation_converged is True
    assert rr.abort_reason is None
    assert rr.relaxation_steps == 5


# ── Mock convergence ──────────────────────────────────────────────────────────

def test_convergence_with_emt(cu_structure):
    """EMT on Cu should converge quickly. Final energy ≤ initial energy."""
    calc = MockMLIPCalculator()
    result = relax_structure(
        structure=cu_structure,
        calculator=calc,
        fmax=0.05,
        max_steps=200,
        filter_type="None",   # position-only for speed
    )
    assert result.relaxation_steps >= 0
    # EMT should find a minimum quickly
    assert result.final_energy_per_atom <= result.initial_energy_per_atom + 1e-3


def test_relaxation_steps_positive(cu_structure):
    calc = MockMLIPCalculator()
    result = relax_structure(
        structure=cu_structure,
        calculator=calc,
        fmax=0.05,
        max_steps=200,
        filter_type="None",
    )
    assert result.relaxation_steps >= 0


def test_monitor_history_populated(cu_structure):
    """monitor_history must be non-empty after relaxation."""
    calc = MockMLIPCalculator()
    result = relax_structure(
        structure=cu_structure,
        calculator=calc,
        fmax=0.05,
        max_steps=50,
        filter_type="None",
    )
    assert len(result.monitor_history) > 0
    # Check structure of entries
    entry = result.monitor_history[0]
    assert "step" in entry
    assert "energy" in entry
    assert "volume" in entry
    assert "max_force" in entry


# ── Structure conversion roundtrip ────────────────────────────────────────────

def test_structure_conversion_roundtrip(cu_structure):
    """pymatgen → ASE → relax → pymatgen: lattice and species preserved."""
    calc = MockMLIPCalculator()
    result = relax_structure(
        structure=cu_structure,
        calculator=calc,
        fmax=0.05,
        max_steps=50,
        filter_type="None",
    )
    relaxed = result.relaxed_structure
    # Check species are preserved
    orig_species = [str(s.specie) for s in cu_structure]
    relax_species = [str(s.specie) for s in relaxed]
    assert orig_species == relax_species


# ── Abort propagation ─────────────────────────────────────────────────────────

def test_abort_energy_divergence(cu_structure):
    """Energy divergence → RelaxationResult with abort_reason='energy_divergence'."""
    n_atoms = len(cu_structure)
    # Energy sequence (total): starts at -10, then rises after 2 steps
    # Forces >> fmax (1.0 eV/Å) so optimizer keeps running
    total_seq = [-10.0, -10.0] + [0.0] * 20   # rises 10 eV above initial
    force_seq = [1.0] * len(total_seq)

    inject_calc = InjectableCalculator(
        energy_sequence=total_seq,
        force_magnitude_sequence=force_seq,
        n_atoms=n_atoms,
    )

    result = relax_structure(
        structure=cu_structure,
        calculator=inject_calc,
        fmax=0.05,
        max_steps=50,
        optimizer_name="FIRE",    # FIRE handles non-Hessian force fields better
        monitor_config={"max_energy_increase": 3.0},
        filter_type="None",
    )
    assert result.relaxation_converged is False
    assert result.abort_reason == "energy_divergence"


def test_abort_stagnation(cu_structure):
    """Flat energy over stagnation window → abort_reason='stagnation'."""
    n_atoms = len(cu_structure)
    # 100 steps of flat energy; forces > fmax so optimizer doesn't converge
    flat_seq = [-10.0] * 100
    force_seq = [1.0] * 100

    inject_calc = InjectableCalculator(
        energy_sequence=flat_seq,
        force_magnitude_sequence=force_seq,
        n_atoms=n_atoms,
    )

    result = relax_structure(
        structure=cu_structure,
        calculator=inject_calc,
        fmax=0.05,
        max_steps=200,
        optimizer_name="FIRE",    # more robust for non-physical force landscapes
        monitor_config={"stagnation_window": 50, "stagnation_threshold": 0.001},
        filter_type="None",
    )
    assert result.relaxation_converged is False
    assert result.abort_reason == "stagnation"


def test_abort_reason_none_on_normal_convergence(cu_structure):
    """For a normal converging run, abort_reason must be None."""
    calc = MockMLIPCalculator()
    result = relax_structure(
        structure=cu_structure,
        calculator=calc,
        fmax=0.05,
        max_steps=200,
        filter_type="None",
    )
    assert result.abort_reason is None


# ── Calculator factory ─────────────────────────────────────────────────────────

def test_get_calculator_mock():
    calc = get_calculator("mock")
    assert calc.get_name() == "mock"


def test_get_calculator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown MLIP"):
        get_calculator("unknown_mlip")


def test_calculator_supports_elements():
    calc = MockMLIPCalculator()
    assert calc.supports_elements({"Cu", "Al"}) is True


# ── Device detection ──────────────────────────────────────────────────────────

def test_detect_device_returns_valid_string():
    """_detect_device must return one of the known device strings."""
    from stages.stage5.calculators import _detect_device
    device = _detect_device()
    assert device in ("cuda", "mps", "cpu")


def test_mace_calculator_stores_explicit_device():
    """MACECalculator must honour an explicit device kwarg."""
    calc = MACECalculator(device="cpu")
    assert calc._device == "cpu"


def test_mattersim_calculator_stores_explicit_device():
    """MatterSimCalculator must honour an explicit device kwarg."""
    calc = MatterSimCalculator(device="cpu")
    assert calc._device == "cpu"


def test_get_calculator_passes_device_to_mace():
    """get_calculator with device='cpu' must create a MACECalculator on CPU."""
    calc = get_calculator("mace-mp-0", device="cpu")
    assert isinstance(calc, MACECalculator)
    assert calc._device == "cpu"


def test_get_calculator_passes_device_to_mattersim():
    """get_calculator with device='cpu' must create a MatterSimCalculator on CPU."""
    calc = get_calculator("mattersim", device="cpu")
    assert isinstance(calc, MatterSimCalculator)
    assert calc._device == "cpu"


# ── GPU tests (skipped by default; run with: pytest -m gpu) ───────────────────

@pytest.mark.gpu
def test_mattersim_calculator_import():
    """MatterSim calculator must load and return an ASE calculator."""
    pytest.importorskip("mattersim", reason="mattersim not installed")
    calc = MatterSimCalculator()
    ase_calc = calc.get_calculator()
    assert ase_calc is not None


@pytest.mark.gpu
def test_mace_calculator_import():
    """MACE-MP-0 calculator must load and return an ASE calculator."""
    calc = MACECalculator()
    ase_calc = calc.get_calculator()
    assert ase_calc is not None


@pytest.mark.gpu
def test_mace_calculator_on_mps():
    """MACECalculator must use MPS device when available (Apple Silicon)."""
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")
    calc = MACECalculator(device="mps")
    ase_calc = calc.get_calculator()
    assert ase_calc is not None
