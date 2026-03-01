"""
Tests for stages/stage5/monitoring.py — RelaxationMonitor.

All tests use MockAtoms; no ASE optimizer or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from stages.stage5.monitoring import RelaxationAborted, RelaxationMonitor


# ── MockAtoms ─────────────────────────────────────────────────────────────────

class MockAtoms:
    """Minimal stand-in for ASE Atoms with controllable energy/volume/forces."""

    def __init__(
        self,
        energy: float,
        volume: float = 100.0,
        forces: list | None = None,
    ) -> None:
        self._energy = energy
        self._volume = volume
        # Default: single atom with tiny force
        self._forces = (
            np.array(forces, dtype=float)
            if forces is not None
            else np.array([[0.001, 0.0, 0.0]])
        )

    def get_potential_energy(self) -> float:
        return self._energy

    def get_volume(self) -> float:
        return self._volume

    def get_forces(self):
        return self._forces


# ── Helper ────────────────────────────────────────────────────────────────────

def _call_sequence(monitor, atoms_list):
    """Call monitor with each MockAtoms in atoms_list, return first exception or None."""
    for atoms in atoms_list:
        monitor(atoms)


# ── Tests: energy divergence ─────────────────────────────────────────────────

def test_energy_divergence_at_threshold_does_not_abort():
    """Energy exactly at initial + max_energy_increase should NOT abort."""
    monitor = RelaxationMonitor(max_energy_increase=3.0)
    # Step 0 sets initial_energy = -100
    monitor(MockAtoms(-100.0))
    # Step 1: energy rises to exactly initial + 3.0 = -97 → should not abort
    monitor(MockAtoms(-97.0))


def test_energy_divergence_above_threshold_aborts():
    """Energy rising more than max_energy_increase eV above initial → abort."""
    monitor = RelaxationMonitor(max_energy_increase=3.0)
    monitor(MockAtoms(-100.0))   # initial
    monitor(MockAtoms(-97.0))    # exactly at limit — no abort
    with pytest.raises(RelaxationAborted) as exc_info:
        monitor(MockAtoms(-96.9))  # 3.1 eV above initial → abort
    assert exc_info.value.reason == "energy_divergence"


def test_energy_divergence_suggestion_not_empty():
    monitor = RelaxationMonitor(max_energy_increase=1.0)
    monitor(MockAtoms(-100.0))
    with pytest.raises(RelaxationAborted) as exc_info:
        monitor(MockAtoms(-98.0))   # 2 eV above initial (> 1 eV limit)
    assert len(exc_info.value.suggestion) > 0


# ── Tests: volume explosion ───────────────────────────────────────────────────

def test_volume_explosion_25pct_does_not_abort():
    """25% volume increase (below 30% threshold) should NOT abort."""
    monitor = RelaxationMonitor(max_volume_change=0.30)
    monitor(MockAtoms(-100.0, volume=100.0))
    monitor(MockAtoms(-101.0, volume=125.0))   # 25% — ok


def test_volume_explosion_31pct_aborts():
    """31% volume increase (above 30% threshold) should abort."""
    monitor = RelaxationMonitor(max_volume_change=0.30)
    monitor(MockAtoms(-100.0, volume=100.0))
    with pytest.raises(RelaxationAborted) as exc_info:
        monitor(MockAtoms(-101.0, volume=131.0))  # 31% — abort
    assert exc_info.value.reason == "volume_explosion"


def test_volume_collapse_aborts():
    """Volume collapsing (shrinking > 30%) should also abort."""
    monitor = RelaxationMonitor(max_volume_change=0.30)
    monitor(MockAtoms(-100.0, volume=100.0))
    with pytest.raises(RelaxationAborted) as exc_info:
        monitor(MockAtoms(-101.0, volume=60.0))   # 40% reduction
    assert exc_info.value.reason == "volume_explosion"


# ── Tests: force spike ────────────────────────────────────────────────────────

def test_force_spike_aborts():
    """Force exceeding max_force (50 eV/Å default) must abort."""
    monitor = RelaxationMonitor(max_force=50.0)
    monitor(MockAtoms(-100.0))   # initial — tiny force
    big_forces = [[60.0, 0.0, 0.0]]   # 60 eV/Å norm
    with pytest.raises(RelaxationAborted) as exc_info:
        monitor(MockAtoms(-101.0, forces=big_forces))
    assert exc_info.value.reason == "force_spike"


def test_force_below_threshold_does_not_abort():
    monitor = RelaxationMonitor(max_force=50.0)
    monitor(MockAtoms(-100.0))
    monitor(MockAtoms(-101.0, forces=[[40.0, 0.0, 0.0]]))   # 40 eV/Å — ok


# ── Tests: stagnation ────────────────────────────────────────────────────────

def test_stagnation_flat_energy_aborts():
    """Flat energy over stagnation_window steps must abort."""
    monitor = RelaxationMonitor(stagnation_window=5, stagnation_threshold=0.001)
    energies = [-100.0, -100.0001, -100.0002, -100.0001, -100.0, -100.0001]
    atoms_list = [MockAtoms(e) for e in energies]
    with pytest.raises(RelaxationAborted) as exc_info:
        _call_sequence(monitor, atoms_list)
    assert exc_info.value.reason == "stagnation"


def test_no_stagnation_if_window_not_reached():
    """If fewer steps than stagnation_window, no stagnation abort."""
    monitor = RelaxationMonitor(stagnation_window=10, stagnation_threshold=0.001)
    # Only 5 calls — window is 10
    for e in [-100.0, -100.0, -100.0, -100.0, -100.0]:
        monitor(MockAtoms(e))


def test_no_stagnation_with_decreasing_energy():
    """Steadily decreasing energy must NOT trigger stagnation."""
    monitor = RelaxationMonitor(stagnation_window=5, stagnation_threshold=0.001)
    energies = [-100.0, -101.0, -101.5, -101.8, -102.0, -102.1]
    for e in energies:
        monitor(MockAtoms(e))


# ── Tests: normal convergence ─────────────────────────────────────────────────

def test_normal_decreasing_energy_no_abort():
    """Steadily decreasing energy sequence must never abort."""
    monitor = RelaxationMonitor()
    for e in [-100.0, -100.5, -101.0, -101.3, -101.5, -101.6]:
        monitor(MockAtoms(e))  # Should not raise


# ── Tests: history recording ──────────────────────────────────────────────────

def test_history_recorded_after_calls():
    """monitor.history must have one entry per call."""
    monitor = RelaxationMonitor()
    n_calls = 10
    for i in range(n_calls):
        monitor(MockAtoms(-100.0 - i * 0.1))
    assert len(monitor.history) == n_calls


def test_history_entry_has_required_keys():
    """Each history entry must contain step, energy, volume, max_force."""
    monitor = RelaxationMonitor()
    monitor(MockAtoms(-100.0, volume=200.0))
    entry = monitor.history[0]
    assert "step" in entry
    assert "energy" in entry
    assert "volume" in entry
    assert "max_force" in entry
    assert entry["step"] == 0
    assert entry["energy"] == pytest.approx(-100.0)
    assert entry["volume"] == pytest.approx(200.0)


# ── Tests: reset ──────────────────────────────────────────────────────────────

def test_reset_clears_history():
    monitor = RelaxationMonitor()
    for e in [-100.0, -101.0]:
        monitor(MockAtoms(e))
    assert len(monitor.history) == 2
    monitor.reset()
    assert len(monitor.history) == 0
    assert monitor.initial_energy is None
    assert monitor.initial_volume is None


def test_reset_recaptures_initial_energy():
    """After reset, next call must set new initial_energy."""
    monitor = RelaxationMonitor(max_energy_increase=3.0)
    monitor(MockAtoms(-100.0))   # initial = -100
    monitor.reset()
    monitor(MockAtoms(-50.0))    # new initial = -50 (would have aborted before reset)
    # This should NOT abort since new initial_energy = -50
    monitor(MockAtoms(-52.0))    # 2 eV above new initial — ok


def test_none_atoms_noop():
    """Calling monitor with atoms=None must be a no-op."""
    monitor = RelaxationMonitor()
    monitor(None)   # must not raise
    assert len(monitor.history) == 0
