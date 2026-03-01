"""
MLIP calculator interface and implementations.

Defines a clean ABC (MLIPCalculator) so that real MLIPs (MatterSim, MACE)
and test mocks conform to the same interface.

Implementations
---------------
MatterSimCalculator  — MatterSim MLIP (``pip install mattersim``).
MACECalculator       — MACE-MP-0 universal MLIP (``pip install mace-torch``).
                       Supports CUDA, Apple MPS (M1/M2/M3), and CPU automatically.
MockMLIPCalculator   — ASE EMT/LJ calculator; no GPU, fast, for testing.
InjectableCalculator — Returns pre-defined energy/force sequences; for testing
                       abort conditions without a real optimizer.

Device selection
----------------
Pass ``device="auto"`` (the default) to auto-select the best available backend:
    CUDA GPU → Apple MPS (M-series) → CPU

Override with ``device="cpu"`` if MPS gives numerical issues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def _detect_device() -> str:
    """Return the best available compute device string.

    Priority: ``cuda`` → ``mps`` (Apple Silicon) → ``cpu``.
    Returns ``"cpu"`` if PyTorch is not installed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


class MLIPCalculator(ABC):
    """Interface for MLIP calculators. Real and mock implementations."""

    @abstractmethod
    def get_calculator(self):
        """Return an ASE-compatible calculator object."""
        ...

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_version(self) -> str: ...

    @abstractmethod
    def supports_elements(self, elements: set[str]) -> bool:
        """Return True if all ``elements`` are in the MLIP training set."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Real MLIPs (require optional dependencies + GPU)
# ─────────────────────────────────────────────────────────────────────────────


class MatterSimCalculator(MLIPCalculator):
    """MatterSim MLIP calculator wrapper.

    Supports CUDA, Apple MPS, and CPU via the ``device`` parameter.
    """

    def __init__(self, device: str = "auto") -> None:
        self._device = _detect_device() if device == "auto" else device

    def get_calculator(self):
        try:
            from mattersim.forcefield import MatterSimCalculator as _MSCalc
            try:
                return _MSCalc(device=self._device)
            except TypeError:
                # Older MatterSim versions may not accept device kwarg
                return _MSCalc()
        except ImportError:
            raise ImportError(
                "MatterSim not installed. Install with: pip install mattersim\n"
                "Or set potential: 'mace-mp-0' in pipeline.yaml"
            )

    def get_name(self) -> str:
        return "mattersim"

    def get_version(self) -> str:
        try:
            import mattersim
            return mattersim.__version__
        except Exception:
            return "unknown"

    def supports_elements(self, elements: set[str]) -> bool:
        """MatterSim covers most of the periodic table — check training set file."""
        import json
        import pathlib
        path = pathlib.Path(__file__).parents[2] / "data" / "mlip_training_elements.json"
        if path.exists():
            with open(path) as f:
                supported = set(json.load(f).get("mattersim", []))
            return elements.issubset(supported)
        return True


class MACECalculator(MLIPCalculator):
    """MACE-MP-0 universal MLIP — works on CUDA, Apple MPS, and CPU.

    Install: ``pip install mace-torch``

    Device selection
    ----------------
    ``device="auto"`` (default): picks CUDA → MPS (M-series Mac) → CPU.
    ``device="mps"``:  force Apple Metal Performance Shaders.
    ``device="cpu"``:  fallback when MPS gives numerical issues.

    On an M1 Max, 96-atom supercell relaxations run in ~1-5 min on MPS
    and ~2-10 min on CPU. No external GPU or Colab required.
    """

    def __init__(self, device: str = "auto") -> None:
        self._device = _detect_device() if device == "auto" else device

    def get_calculator(self):
        try:
            from mace.calculators import mace_mp
        except ImportError:
            raise ImportError(
                "MACE not installed. Install with: pip install mace-torch"
            )

        # MPS (Apple Silicon) does not support float64; MACE model weights are
        # stored as float64 so loading directly onto MPS fails.  Fall back to
        # CPU (still fast on M1/M2/M3 for < 200 atoms) and warn the user.
        if self._device == "mps":
            try:
                return mace_mp(default_dtype="float32", device="mps")
            except (TypeError, RuntimeError):
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "MACE-MP-0: MPS float64 incompatibility — falling back to CPU. "
                    "On M1/M2/M3, CPU relaxations typically take 2–10 min per structure."
                )
                return mace_mp(default_dtype="float64", device="cpu")
        return mace_mp(default_dtype="float64", device=self._device)

    def get_name(self) -> str:
        return "mace-mp-0"

    def get_version(self) -> str:
        try:
            import mace
            return mace.__version__
        except Exception:
            return "unknown"

    def supports_elements(self, elements: set[str]) -> bool:
        # MACE-MP-0 covers all naturally occurring elements
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Mock calculators (no GPU, for testing)
# ─────────────────────────────────────────────────────────────────────────────


class MockMLIPCalculator(MLIPCalculator):
    """
    Mock calculator for testing.

    Tries ASE's built-in EMT potential first (fast, works for Al/Cu/Ni/…).
    Falls back to a simple Lennard-Jones pair potential for element sets
    that EMT does not cover (e.g. Li, O, Co in LiCoO₂).
    """

    # EMT supports this set of elements
    _EMT_ELEMENTS = {
        "H", "Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt",
        "C", "N", "O",  # limited
    }

    def get_calculator(self):
        from ase.calculators.emt import EMT
        return EMT()

    def get_calculator_for_atoms(self, atoms):
        """Return EMT if all elements are supported, otherwise Lennard-Jones."""
        symbols = set(atoms.get_chemical_symbols())
        # Check if EMT has parameters for all elements in this structure
        try:
            from ase.calculators.emt import parameters as emt_params
            if all(s in emt_params for s in symbols):
                from ase.calculators.emt import EMT
                return EMT()
        except Exception:
            pass
        # Fall back to Lennard-Jones (works for any element pair)
        from ase.calculators.lj import LennardJones
        return LennardJones()

    def get_name(self) -> str:
        return "mock"

    def get_version(self) -> str:
        return "test"

    def supports_elements(self, elements: set[str]) -> bool:
        return True


class InjectableCalculator:
    """
    Returns pre-defined energy/force sequences.

    Used to test abort conditions in RelaxationMonitor without a real MLIP.
    Tracks atom positions to detect when a new optimizer step has started
    (different positions → advance to the next sequence value).

    Parameters
    ----------
    energy_sequence:
        Total energies (eV) to return at successive optimizer steps.
        The last value is repeated once exhausted.
    force_magnitude_sequence:
        Force magnitudes (eV/Å) returned at each step. Values should be
        > ``fmax`` to prevent premature convergence. Defaults to 1.0 eV/Å.
    n_atoms:
        Fallback number of atoms when ``atoms`` arg is None.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        energy_sequence: list[float],
        force_magnitude_sequence: Optional[list[float]] = None,
        n_atoms: int = 1,
    ) -> None:
        self.energy_sequence = energy_sequence
        # Default force magnitude >> fmax so optimizer keeps running
        self.force_magnitude_sequence = (
            force_magnitude_sequence or [1.0] * len(energy_sequence)
        )
        self.n_atoms = n_atoms
        self._step = 0
        self.results: dict = {}
        self._last_pos_hash: Optional[int] = None

    # ------------------------------------------------------------------
    # Minimal ASE Calculator interface
    # ------------------------------------------------------------------

    def _pos_hash(self, atoms) -> Optional[int]:
        if atoms is None:
            return None
        try:
            return hash(atoms.get_positions().tobytes())
        except Exception:
            return None

    def _do_calculate(self, atoms) -> None:
        """Advance step counter and fill self.results."""
        idx = min(self._step, len(self.energy_sequence) - 1)
        energy = self.energy_sequence[idx]
        f_idx = min(self._step, len(self.force_magnitude_sequence) - 1)
        f_mag = self.force_magnitude_sequence[f_idx]
        n = len(atoms) if atoms is not None else self.n_atoms
        forces = np.full((n, 3), f_mag / np.sqrt(3))   # equal components
        self.results = {"energy": energy, "forces": forces}
        self._step += 1
        self._last_pos_hash = self._pos_hash(atoms)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        pos_hash = self._pos_hash(atoms)
        if pos_hash != self._last_pos_hash or not self.results:
            self._do_calculate(atoms)
        return self.results["energy"]

    def get_forces(self, atoms=None, apply_constraint=False):
        pos_hash = self._pos_hash(atoms)
        if pos_hash != self._last_pos_hash or "forces" not in self.results:
            self._do_calculate(atoms)
        return self.results["forces"]

    def get_stress(self, atoms=None):
        return np.zeros(6)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self._do_calculate(atoms)

    def check_state(self, atoms, tol=1e-15):
        return []

    def reset(self):
        self._step = 0
        self.results = {}
        self._last_pos_hash = None


def get_calculator(mlip_name: str, device: str = "auto") -> MLIPCalculator:
    """Factory: return the right MLIPCalculator for a given name.

    Parameters
    ----------
    mlip_name:
        One of ``"mattersim"``, ``"mace-mp-0"`` / ``"mace"``, ``"mock"``.
    device:
        Compute device. ``"auto"`` (default) picks CUDA → MPS → CPU.
        Pass ``"cpu"`` or ``"mps"`` to override.

    Raises
    ------
    ValueError
        If ``mlip_name`` is not recognised.
    """
    name = mlip_name.lower()
    if name == "mattersim":
        return MatterSimCalculator(device=device)
    elif name in ("mace-mp-0", "mace"):
        return MACECalculator(device=device)
    elif name == "mock":
        return MockMLIPCalculator()
    else:
        raise ValueError(
            f"Unknown MLIP: {mlip_name!r}. "
            "Valid options: 'mattersim', 'mace-mp-0', 'mock'."
        )
