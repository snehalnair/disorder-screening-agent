"""
Tests for Stage 4 (Viability): element safety and regulatory metadata filter.

All tests use the real data/element_metadata.json so they also validate the
metadata file content.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from stages.stage4_viability import (
    TOOL_METADATA,
    _load_element_metadata,
    _rejection_reason,
    run_stage4_viability,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_BASE_CFG = {
    "pipeline": {
        "stage4_viability": {
            "enabled": True,
            "constraints": {"non_radioactive": True, "non_toxic": True},
        },
        "database": {"local": {"path": ":memory:"}},
    }
}

_BASE_CFG_DISABLED_TOXIC = {
    "pipeline": {
        "stage4_viability": {
            "enabled": True,
            "constraints": {"non_radioactive": True, "non_toxic": False},
        }
    }
}

_BASE_CFG_ALL_OFF = {
    "pipeline": {
        "stage4_viability": {
            "constraints": {"non_radioactive": False, "non_toxic": False}
        }
    }
}


def _make_candidates(*elements) -> list[dict]:
    return [
        {"element": el, "oxidation_state": 3, "mismatch_pct": 0.05, "sub_probability": 0.01}
        for el in elements
    ]


# ── Metadata file tests ───────────────────────────────────────────────────────

class TestElementMetadata:
    def test_metadata_file_exists(self):
        meta_path = pathlib.Path(__file__).parent.parent / "data" / "element_metadata.json"
        assert meta_path.exists(), "data/element_metadata.json not found"

    def test_metadata_loads(self):
        metadata = _load_element_metadata()
        assert len(metadata) >= 29, "Expected at least 29 elements in metadata"

    def test_radioactive_elements_marked(self):
        metadata = _load_element_metadata()
        assert metadata["U"]["is_radioactive"] is True

    def test_toxic_elements_marked(self):
        metadata = _load_element_metadata()
        for el in ["As", "Cr", "Os", "Sb"]:
            assert metadata[el]["is_toxic"] is True, f"{el} should have is_toxic=True"

    def test_safe_elements_not_toxic(self):
        metadata = _load_element_metadata()
        for el in ["Al", "Fe", "Ti", "Mg", "Nb", "Zr", "W", "Ta"]:
            assert metadata[el]["is_toxic"] is False, f"{el} should not be toxic"
            assert metadata[el]["is_radioactive"] is False

    def test_required_fields_present(self):
        metadata = _load_element_metadata()
        required = {"is_radioactive", "is_toxic", "toxicity_class",
                    "eu_crm_2023", "usgs_criticality_2022", "cost_annotation"}
        for el, meta in metadata.items():
            missing = required - set(meta.keys())
            assert not missing, f"{el} missing fields: {missing}"

    def test_se_is_not_filtered(self):
        """Se has toxicity_class=moderate but is_toxic=False — annotation only."""
        metadata = _load_element_metadata()
        assert metadata["Se"]["is_toxic"] is False
        assert metadata["Se"]["toxicity_class"] == "moderate"

    def test_cr_synthesis_hazard(self):
        """Cr is flagged for synthesis hazard (Cr6+ risk during calcination)."""
        metadata = _load_element_metadata()
        assert metadata["Cr"]["synthesis_hazard"] is True
        assert metadata["Cr"]["is_toxic"] is True


# ── Rejection logic tests ─────────────────────────────────────────────────────

class TestRejectionReason:
    def setup_method(self):
        self.metadata = _load_element_metadata()

    def test_radioactive_rejected_when_flag_on(self):
        reason = _rejection_reason("U", self.metadata["U"],
                                   filter_radioactive=True, filter_toxic=True)
        assert reason == "radioactive"

    def test_radioactive_passes_when_both_flags_off(self):
        # U is both radioactive and toxic — both flags must be off to let it pass
        reason = _rejection_reason("U", self.metadata["U"],
                                   filter_radioactive=False, filter_toxic=False)
        assert reason is None

    def test_carcinogen_rejected(self):
        reason = _rejection_reason("As", self.metadata["As"],
                                   filter_radioactive=True, filter_toxic=True)
        assert reason is not None
        assert "carcinogen" in reason

    def test_os_rejected_as_high_toxicity(self):
        reason = _rejection_reason("Os", self.metadata["Os"],
                                   filter_radioactive=True, filter_toxic=True)
        assert reason is not None
        assert "high" in reason

    def test_safe_element_passes(self):
        for el in ["Al", "Ti", "Nb", "W", "Zr", "Mg", "Fe", "Ga"]:
            meta = self.metadata[el]
            reason = _rejection_reason(el, meta,
                                       filter_radioactive=True, filter_toxic=True)
            assert reason is None, f"{el} should pass viability filter"


# ── Stage function tests ──────────────────────────────────────────────────────

class TestRunStage4Viability:
    def test_removes_u_as_cr_os_sb(self):
        candidates = _make_candidates("Al", "U", "As", "Cr", "Os", "Sb", "Ti", "Nb")
        state = {"stage3_candidates": candidates, "config": _BASE_CFG}
        result = run_stage4_viability(state)

        passed_elements = {c["element"] for c in result["stage4_viability_candidates"]}
        rejected_elements = {c["element"] for c in result["stage4_viability_rejected"]}

        assert "U" in rejected_elements
        assert "As" in rejected_elements
        assert "Cr" in rejected_elements
        assert "Os" in rejected_elements
        assert "Sb" in rejected_elements
        assert "Al" in passed_elements
        assert "Ti" in passed_elements
        assert "Nb" in passed_elements

    def test_rejected_candidates_have_reason(self):
        candidates = _make_candidates("U", "As", "Al")
        state = {"stage3_candidates": candidates, "config": _BASE_CFG}
        result = run_stage4_viability(state)

        for rejected in result["stage4_viability_rejected"]:
            assert "viability_rejection_reason" in rejected
            assert rejected["viability_rejection_reason"]

    def test_passed_candidates_annotated(self):
        candidates = _make_candidates("Al", "Nb", "W")
        state = {"stage3_candidates": candidates, "config": _BASE_CFG}
        result = run_stage4_viability(state)

        for cand in result["stage4_viability_candidates"]:
            assert "eu_crm_2023" in cand
            assert "toxicity_class" in cand
            assert "usgs_criticality" in cand
            assert "cost_annotation" in cand

    def test_all_filters_off_passes_everything(self):
        candidates = _make_candidates("U", "As", "Cr", "Al")
        state = {"stage3_candidates": candidates, "config": _BASE_CFG_ALL_OFF}
        result = run_stage4_viability(state)

        assert len(result["stage4_viability_candidates"]) == 4
        assert len(result["stage4_viability_rejected"]) == 0

    def test_non_toxic_off_only_radioactive_filtered(self):
        candidates = _make_candidates("U", "As", "Al")
        state = {"stage3_candidates": candidates, "config": _BASE_CFG_DISABLED_TOXIC}
        result = run_stage4_viability(state)

        passed = {c["element"] for c in result["stage4_viability_candidates"]}
        rejected = {c["element"] for c in result["stage4_viability_rejected"]}

        assert "U" in rejected
        assert "As" in passed   # non_toxic=False → As passes
        assert "Al" in passed

    def test_empty_input(self):
        state = {"stage3_candidates": [], "config": _BASE_CFG}
        result = run_stage4_viability(state)

        assert result["stage4_viability_candidates"] == []
        assert result["stage4_viability_rejected"] == []

    def test_execution_log_populated(self):
        candidates = _make_candidates("Al", "U")
        state = {"stage3_candidates": candidates, "config": _BASE_CFG}
        result = run_stage4_viability(state)

        assert len(result["execution_log"]) >= 1
        assert "Viability" in result["execution_log"][0]

    def test_execution_log_lists_rejected(self):
        candidates = _make_candidates("U", "As", "Al")
        state = {"stage3_candidates": candidates, "config": _BASE_CFG}
        result = run_stage4_viability(state)

        log = " ".join(result["execution_log"])
        assert "U" in log
        assert "As" in log

    def test_missing_metadata_passes_through(self):
        """Elements not in metadata should pass with a warning."""
        candidates = _make_candidates("Xy")  # fake element
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"elements": {}}, f)
            tmp_path = f.name

        cfg_with_path = {
            "pipeline": {
                "stage4_viability": {
                    "constraints": {"non_radioactive": True, "non_toxic": True},
                    "metadata_path": tmp_path,
                }
            }
        }
        state = {"stage3_candidates": candidates, "config": cfg_with_path}
        result = run_stage4_viability(state)

        assert len(result["stage4_viability_candidates"]) == 1
        assert result["stage4_viability_candidates"][0]["toxicity_class"] == "unknown"

    def test_missing_metadata_file_passes_all(self):
        """If metadata file is absent, all candidates pass through."""
        candidates = _make_candidates("Al", "U")
        cfg = {
            "pipeline": {
                "stage4_viability": {
                    "constraints": {"non_radioactive": True, "non_toxic": True},
                    "metadata_path": "/nonexistent/path/element_metadata.json",
                }
            }
        }
        state = {"stage3_candidates": candidates, "config": cfg}
        result = run_stage4_viability(state)

        assert len(result["stage4_viability_candidates"]) == 2
        assert len(result["stage4_viability_rejected"]) == 0


# ── TOOL_METADATA contract ────────────────────────────────────────────────────

class TestToolMetadata:
    def test_required_keys_present(self):
        required = {"name", "stage", "cost", "requires_gpu"}
        assert required <= set(TOOL_METADATA.keys())

    def test_not_gpu_required(self):
        assert TOOL_METADATA["requires_gpu"] is False

    def test_stage_id(self):
        assert TOOL_METADATA["stage"] == "4v"
