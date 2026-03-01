"""
Stage 5 relaxation monitor.

Attached to an ASE optimizer via ``optimizer.attach(monitor, interval=1, atoms=atoms)``.
Called at every step; raises ``RelaxationAborted`` on four failure conditions:

1. energy_divergence  — total energy rises more than ``max_energy_increase`` eV
                        above the initial value.
2. volume_explosion   — cell volume changes by more than ``max_volume_change``
                        (relative fraction) vs. the initial volume.
3. force_spike        — any single-atom force exceeds ``max_force`` eV/Å.
4. stagnation         — energy range over the last ``stagnation_window`` steps
                        is smaller than ``stagnation_threshold`` eV.
"""

from __future__ import annotations

import numpy as np


TOOL_METADATA = {
    "name": "relaxation_monitor",
    "stage": "5b_monitor",
    "description": (
        "Attached to ASE optimizer. Aborts MLIP relaxation on energy divergence, "
        "volume explosion, force spike, or stagnation."
    ),
    "requires_gpu": False,
    "configurable_params": [
        "max_energy_increase",
        "max_volume_change",
        "stagnation_window",
        "stagnation_threshold",
        "max_force",
    ],
    "failure_modes": [
        "energy_divergence",
        "volume_explosion",
        "force_spike",
        "stagnation",
    ],
}


class RelaxationAborted(Exception):
    """Raised by RelaxationMonitor when an abort condition is triggered."""

    def __init__(self, reason: str, suggestion: str) -> None:
        self.reason = reason
        self.suggestion = suggestion
        super().__init__(f"Relaxation aborted: {reason}. {suggestion}")


class RelaxationMonitor:
    """
    Attached to ASE optimizer. Called at every optimisation step.

    Monitors for four failure conditions and aborts early with
    structured explanations.

    Parameters
    ----------
    max_energy_increase:
        Maximum allowed total-energy increase (eV) above the initial value.
        Default 3.0 eV.
    max_volume_change:
        Maximum allowed relative volume change (fraction). Default 0.30 (30%).
    stagnation_window:
        Number of consecutive steps used to detect stagnation. Default 50.
    stagnation_threshold:
        Energy range (eV) below which the run is considered stagnated. Default 1e-3.
    max_force:
        Maximum allowed force magnitude on any single atom (eV/Å). Default 50.0.
    """

    def __init__(
        self,
        max_energy_increase: float = 3.0,
        max_volume_change: float = 0.30,
        stagnation_window: int = 50,
        stagnation_threshold: float = 0.001,
        max_force: float = 50.0,
    ) -> None:
        self.max_energy_increase = max_energy_increase
        self.max_volume_change = max_volume_change
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold
        self.max_force = max_force

        self.history: list[dict] = []
        self.initial_energy: float | None = None
        self.initial_volume: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, atoms=None) -> None:
        """Called by ASE optimizer.attach() at each step."""
        if atoms is None:
            return

        energy: float = atoms.get_potential_energy()
        volume: float = atoms.get_volume()
        forces = atoms.get_forces()
        max_force_val = float(np.max(np.linalg.norm(forces, axis=1)))

        # Initialise reference values on the first call
        if self.initial_energy is None:
            self.initial_energy = energy
            self.initial_volume = volume

        self.history.append(
            {
                "step": len(self.history),
                "energy": energy,
                "volume": volume,
                "max_force": max_force_val,
            }
        )

        # ── Check 1: Energy divergence ────────────────────────────────
        if energy > self.initial_energy + self.max_energy_increase:
            raise RelaxationAborted(
                "energy_divergence",
                "MLIP likely extrapolating outside training domain. "
                "Validate this chemistry with DFT first.",
            )

        # ── Check 2: Volume explosion / collapse ──────────────────────
        vol_change = abs(volume - self.initial_volume) / self.initial_volume
        if vol_change > self.max_volume_change:
            raise RelaxationAborted(
                "volume_explosion",
                f"Volume changed by {vol_change:.0%}. "
                "Dopant may be too large/small for this site.",
            )

        # ── Check 3: Force spike ──────────────────────────────────────
        if max_force_val > self.max_force:
            raise RelaxationAborted(
                "force_spike",
                f"Max force {max_force_val:.1f} eV/Å. "
                "Unphysical forces detected. MLIP failure.",
            )

        # ── Check 4: Stagnation ───────────────────────────────────────
        if len(self.history) > self.stagnation_window:
            recent_energies = [
                h["energy"] for h in self.history[-self.stagnation_window :]
            ]
            energy_range = max(recent_energies) - min(recent_energies)
            if energy_range < self.stagnation_threshold:
                raise RelaxationAborted(
                    "stagnation",
                    f"Energy range {energy_range:.6f} eV over "
                    f"{self.stagnation_window} steps. "
                    "Try FIRE optimizer or larger supercell.",
                )

    def reset(self) -> None:
        """Reset monitor state for reuse across multiple relaxations."""
        self.history = []
        self.initial_energy = None
        self.initial_volume = None
