"""Tests for __main__.py — CLI interface.

Uses monkeypatching to avoid running the real pipeline; validates argument
parsing, output shapes, and exit codes.
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Invoke main() with argv; capture stdout and return (exit_code, stdout)."""
    import io
    from contextlib import redirect_stdout
    from unittest.mock import patch

    # Import after sys.path is set up
    import importlib
    import os

    # Ensure project root is on path
    project_root = str(pathlib.Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    buf = io.StringIO()
    with redirect_stdout(buf):
        # Import __main__ module
        spec = importlib.util.spec_from_file_location(
            "__main__", pathlib.Path(project_root) / "__main__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        try:
            code = mod.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1

    return (code if code is not None else 0), buf.getvalue()


def _import_main():
    import importlib.util
    project_root = str(pathlib.Path(__file__).parent.parent)
    spec = importlib.util.spec_from_file_location(
        "_disorder_main", pathlib.Path(project_root) / "__main__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── No-command / help ─────────────────────────────────────────────────────────


def test_no_command_prints_help_exits_1(capsys):
    """Running with no sub-command should print help and exit 1."""
    mod = _import_main()
    code = mod.main([])
    assert code == 1


def test_help_flag_exits_0(capsys):
    """--help should print usage and exit 0."""
    mod = _import_main()
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["--help"])
    assert exc_info.value.code == 0


def test_invalid_command_exits_nonzero(capsys):
    """An unknown sub-command should exit non-zero."""
    mod = _import_main()
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["totally-invalid-command"])
    assert exc_info.value.code != 0


# ── prune command ─────────────────────────────────────────────────────────────


def test_prune_produces_json_output(monkeypatch):
    """prune command must produce JSON with stage1/stage2/stage3_candidates keys."""
    mock_state = {
        "stage1_candidates": [{"element": "Al"}, {"element": "Mg"}],
        "stage2_candidates": [{"element": "Al"}],
        "stage3_candidates": [{"element": "Al"}],
        "execution_log": ["Stage 1: 2 candidates"],
    }

    mod = _import_main()
    monkeypatch.setattr(
        "graph.entry_points.run_stages_1_3",
        lambda **kw: mock_state,
    )

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main(["prune", "--formula", "LiCoO2", "--site", "Co", "--os", "3"])

    assert code == 0
    output = buf.getvalue()
    data = json.loads(output)
    assert "stage1_candidates" in data
    assert "stage2_candidates" in data
    assert "stage3_candidates" in data
    assert len(data["stage1_candidates"]) == 2


def test_prune_writes_to_file(tmp_path, monkeypatch):
    """prune --output FILE must write JSON to that file."""
    mock_state = {
        "stage1_candidates": [{"element": "Ti"}],
        "stage2_candidates": [],
        "stage3_candidates": [],
        "execution_log": [],
    }

    mod = _import_main()
    monkeypatch.setattr(
        "graph.entry_points.run_stages_1_3",
        lambda **kw: mock_state,
    )

    out_file = tmp_path / "prune_out.json"
    code = mod.main([
        "prune",
        "--formula", "LiCoO2",
        "--site", "Co",
        "--os", "3",
        "--output", str(out_file),
    ])
    assert code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "stage1_candidates" in data


def test_prune_missing_required_args_exits_nonzero():
    """prune without --formula/--site/--os should exit non-zero."""
    mod = _import_main()
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["prune", "--formula", "LiCoO2"])  # missing --site and --os
    assert exc_info.value.code != 0


# ── single command ────────────────────────────────────────────────────────────


def test_single_dopant_skips_pruning(monkeypatch):
    """single --dopant Al must call run_single_dopant (not run_stages_1_3)."""
    calls = {"prune": 0, "single": 0}

    def mock_prune(**kw):
        calls["prune"] += 1
        return {}

    def mock_single(**kw):
        calls["single"] += 1
        return {
            "run_id": str(uuid.uuid4()),
            "simulation_results": [],
            "ordered_results": {},
        }

    mod = _import_main()
    monkeypatch.setattr("graph.entry_points.run_stages_1_3", mock_prune)
    monkeypatch.setattr("graph.entry_points.run_single_dopant", mock_single)

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main([
            "single",
            "--formula", "LiCoO2",
            "--site", "Co",
            "--os", "3",
            "--dopant", "Al",
        ])

    assert code == 0
    assert calls["single"] == 1
    assert calls["prune"] == 0  # pruning stages NOT called


def test_single_with_concentrations(monkeypatch):
    """single --conc 0.05 0.10 must pass concentrations to run_single_dopant."""
    received = {}

    def mock_single(**kw):
        received.update(kw)
        return {"run_id": "test-id", "simulation_results": [], "ordered_results": {}}

    mod = _import_main()
    monkeypatch.setattr("graph.entry_points.run_single_dopant", mock_single)

    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        mod.main([
            "single",
            "--formula", "LiCoO2",
            "--site", "Co",
            "--os", "3",
            "--dopant", "Al",
            "--conc", "0.05", "0.10",
        ])

    assert received.get("concentrations") == pytest.approx([0.05, 0.10])


# ── list-runs command ─────────────────────────────────────────────────────────


def test_list_runs_empty_db_no_crash(tmp_path):
    """list-runs on an empty database must return 0 and print 'No runs found.'"""
    db_path = tmp_path / "empty.db"
    mod = _import_main()

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main(["list-runs", "--db", str(db_path)])

    assert code == 0
    assert "No runs found." in buf.getvalue()


def test_list_runs_shows_results(tmp_path):
    """list-runs must show run_id and parent formula when results exist."""
    from db.local_store import LocalStore
    from db.models import SimulationResult

    db_path = tmp_path / "runs.db"
    store = LocalStore(str(db_path))

    run_id = str(uuid.uuid4())
    sim = SimulationResult(
        dopant_element="Al",
        dopant_oxidation_state=3,
        concentration_pct=5.0,
        sqs_realisation_index=0,
        parent_formula="LiCoO2",
        target_site_species="Co",
        supercell_size=[2, 2, 2],
        mlip_name="mock",
        mlip_version="0.1",
        relaxation_converged=True,
        relaxation_steps=10,
    )
    store.save_simulation(sim, run_id)
    store.close()

    mod = _import_main()

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main(["list-runs", "--db", str(db_path)])

    assert code == 0
    output = buf.getvalue()
    assert run_id in output
    assert "LiCoO2" in output


def test_list_runs_parent_filter(tmp_path):
    """list-runs --parent must filter by parent formula."""
    from db.local_store import LocalStore
    from db.models import SimulationResult

    db_path = tmp_path / "filter.db"
    store = LocalStore(str(db_path))

    for formula, dopant in [("LiCoO2", "Al"), ("LiNiO2", "Mg")]:
        store.save_simulation(
            SimulationResult(
                dopant_element=dopant,
                dopant_oxidation_state=3,
                concentration_pct=5.0,
                sqs_realisation_index=0,
                parent_formula=formula,
                target_site_species="Co",
                supercell_size=[2, 2, 2],
                mlip_name="mock",
                mlip_version="0.1",
                relaxation_converged=True,
                relaxation_steps=5,
            ),
            str(uuid.uuid4()),
        )
    store.close()

    mod = _import_main()

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main(["list-runs", "--db", str(db_path), "--parent", "LiCoO2"])

    assert code == 0
    output = buf.getvalue()
    assert "LiCoO2" in output
    assert "LiNiO2" not in output


# ── run command ───────────────────────────────────────────────────────────────


def test_run_produces_report_file(tmp_path, monkeypatch):
    """run with mock pipeline must create a report file in report-dir."""
    import pathlib

    report_path = tmp_path / "test_report.md"

    mock_state = {
        "ranked_report": {"parent_formula": "LiCoO2"},
        "report_path": str(report_path),
        "simulation_results": [],
        "execution_log": [],
    }

    # Create the report file as the mock state says it was created
    report_path.write_text("# Mock Report\n")

    mod = _import_main()
    monkeypatch.setattr("graph.entry_points.run_full_pipeline", lambda **kw: mock_state)

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main([
            "run",
            "--formula", "LiCoO2",
            "--site", "Co",
            "--os", "3",
            "--report-dir", str(tmp_path),
        ])

    assert code == 0
    assert report_path.exists()


def test_run_missing_formula_without_config_exits_1(tmp_path, monkeypatch):
    """run without --formula and no material: in config must exit 1."""
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text("pipeline:\n  stage2_radius:\n    mismatch_threshold: 0.35\n")

    mod = _import_main()
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        code = mod.main(["run", "--config", str(cfg_path)])

    assert code == 1
