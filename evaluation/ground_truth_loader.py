"""
Ground truth loader for dopant evaluation.

Loads ``data/known_dopants/nmc_layered_oxide.json`` (or a user-supplied path)
and exposes helpers for extracting element lists filtered by site and class.
"""

from __future__ import annotations

import json
import pathlib

_DEFAULT_GT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "data"
    / "known_dopants"
    / "nmc_layered_oxide.json"
)


def load_ground_truth(path: str | pathlib.Path | None = None) -> dict:
    """Load and return the full ground truth JSON as a dict."""
    p = pathlib.Path(path) if path else _DEFAULT_GT_PATH
    with p.open() as f:
        return json.load(f)


def get_dopant_elements(
    ground_truth: dict,
    site_filter: str | None = None,
    classes: list[str] | None = None,
) -> list[str]:
    """Return element symbols from the ground truth that match the given filters.

    Args:
        ground_truth:  Parsed ground truth dict (from ``load_ground_truth``).
        site_filter:   If given, only include dopants at this site
                       (e.g. ``"TM_octahedral"``).
        classes:       Ground truth class values to include.  Defaults to all
                       three classes (successful, limited, failed).

    Returns:
        Sorted list of element symbols.
    """
    if classes is None:
        classes = ["confirmed_successful", "confirmed_limited", "confirmed_failed"]

    results: list[str] = []
    for symbol, info in ground_truth.get("dopants", {}).items():
        if site_filter and info.get("site") != site_filter:
            continue
        if info.get("ground_truth_class") not in classes:
            continue
        results.append(symbol)

    return sorted(results)
