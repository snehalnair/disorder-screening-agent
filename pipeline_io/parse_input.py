"""
Input parsing and validation for the disorder-screening pipeline.

Tier 1 (implemented): Validates a ``PipelineInput`` dataclass with all
errors collected and reported together (not just the first failure).

Tier 2/3 (LLM natural-language → PipelineInput): Out of scope for Phase 5.
Requires ``llm.tier >= 2`` in pipeline.yaml. See Level 2 PRD §6.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Input dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineInput:
    """Validated description of a single screening run.

    All fields correspond 1-to-1 with pipeline.yaml + CLI arguments.
    Use ``validate_pipeline_input()`` to create a validated instance.
    """

    # Required
    parent_formula: str                     # e.g. "LiNi0.8Mn0.1Co0.1O2"
    target_species: str                     # element being substituted, e.g. "Co"
    target_oxidation_state: int             # formal OS of target, e.g. 3

    # Optional with defaults
    target_coordination_number: int = 6     # CN of target site
    concentrations: list[float] = field(
        default_factory=lambda: [0.05, 0.10]
    )
    supercell_size: list = field(
        default_factory=lambda: [2, 2, 2]   # [a, b, c] or 3×3 matrix
    )
    n_sqs_realisations: int = 3
    target_properties: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    # Single-dopant mode (skips Stages 1–3)
    specific_dopant: Optional[str] = None          # e.g. "Al"
    specific_dopant_os: Optional[int] = None       # e.g. 3


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


class ValidationError(Exception):
    """Raised when PipelineInput fails validation.

    ``errors`` is a list of all issues found (not just the first).
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(f"  - {e}" for e in errors))


_VALID_CNS = {4, 6, 8, 12}


def validate_pipeline_input(inp: PipelineInput) -> PipelineInput:
    """Validate a PipelineInput and return it unchanged if valid.

    Collects ALL validation errors before raising so the caller sees the
    complete list in one ``ValidationError``.

    Parameters
    ----------
    inp:
        The input to validate.

    Returns
    -------
    PipelineInput
        The same object, unchanged.

    Raises
    ------
    ValidationError
        If one or more validation rules are violated.
    """
    errors: list[str] = []

    # ── parent_formula ─────────────────────────────────────────────────────
    comp = None
    try:
        from pymatgen.core import Composition
        comp = Composition(inp.parent_formula)
        if len(comp) == 0:
            errors.append(
                f"parent_formula {inp.parent_formula!r} has no elements."
            )
    except Exception:
        errors.append(
            f"parent_formula {inp.parent_formula!r} is not a valid chemical formula."
        )

    # ── target_species ─────────────────────────────────────────────────────
    if comp is not None:
        element_symbols = {str(e) for e in comp.elements}
        if inp.target_species not in element_symbols:
            errors.append(
                f"target_species {inp.target_species!r} is not present in "
                f"parent_formula {inp.parent_formula!r} "
                f"(elements: {sorted(element_symbols)})."
            )

    # ── target_oxidation_state ────────────────────────────────────────────
    if inp.target_oxidation_state <= 0:
        errors.append(
            f"target_oxidation_state must be a positive integer, "
            f"got {inp.target_oxidation_state}."
        )

    # ── target_coordination_number ────────────────────────────────────────
    if inp.target_coordination_number not in _VALID_CNS:
        errors.append(
            f"target_coordination_number must be one of {sorted(_VALID_CNS)}, "
            f"got {inp.target_coordination_number}."
        )

    # ── concentrations ────────────────────────────────────────────────────
    for c in inp.concentrations:
        if not (0.0 < c <= 1.0):
            errors.append(
                f"concentration {c} is out of range — must be in (0.0, 1.0]."
            )

    # ── supercell_size ─────────────────────────────────────────────────────
    ss = inp.supercell_size
    if ss and isinstance(ss[0], (list, tuple)):
        # 3×3 matrix
        flat_vals = [v for row in ss for v in row]
    else:
        flat_vals = list(ss)
    for v in flat_vals:
        if not isinstance(v, int) or v <= 0:
            errors.append(
                f"supercell_size entries must be positive integers, got {v!r}."
            )
            break  # report once per supercell block

    # ── n_sqs_realisations ────────────────────────────────────────────────
    if inp.n_sqs_realisations < 1:
        errors.append(
            f"n_sqs_realisations must be ≥ 1, got {inp.n_sqs_realisations}."
        )

    # ── specific_dopant ───────────────────────────────────────────────────
    if inp.specific_dopant is not None:
        try:
            from pymatgen.core import Element
            Element(inp.specific_dopant)
        except Exception:
            errors.append(
                f"specific_dopant {inp.specific_dopant!r} is not a valid element symbol."
            )

    # ── specific_dopant_os requires specific_dopant ───────────────────────
    if inp.specific_dopant_os is not None and inp.specific_dopant is None:
        errors.append(
            "specific_dopant_os is set but specific_dopant is not — "
            "both must be provided together."
        )

    if errors:
        raise ValidationError(errors)

    return inp


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors
# ─────────────────────────────────────────────────────────────────────────────


def pipeline_input_from_dict(data: dict) -> PipelineInput:
    """Build and validate a ``PipelineInput`` from a plain dict (e.g. JSON)."""
    inp = PipelineInput(
        parent_formula=data["parent_formula"],
        target_species=data["target_species"],
        target_oxidation_state=int(data["target_oxidation_state"]),
        target_coordination_number=int(
            data.get("target_coordination_number", 6)
        ),
        concentrations=data.get("concentrations", [0.05, 0.10]),
        supercell_size=data.get("supercell_size", [2, 2, 2]),
        n_sqs_realisations=int(data.get("n_sqs_realisations", 3)),
        target_properties=data.get("target_properties", []),
        constraints=data.get("constraints", {}),
        specific_dopant=data.get("specific_dopant"),
        specific_dopant_os=(
            int(data["specific_dopant_os"])
            if data.get("specific_dopant_os") is not None
            else None
        ),
    )
    return validate_pipeline_input(inp)


def tier2_parse_natural_language(text: str, config: dict) -> PipelineInput:
    """Parse natural-language input via LLM (Tier 2/3 — Level 2 scope).

    Not implemented in Phase 5. Requires ``llm.tier >= 2`` in pipeline.yaml.
    See Level 2 PRD §6.7 for full specification.

    Raises
    ------
    NotImplementedError
        Always, in Phase 5.
    """
    raise NotImplementedError(
        "Natural-language input parsing (Tier 2/3) is a Level 2 feature. "
        "Pass a structured PipelineInput dict instead, or use the CLI flags."
    )
