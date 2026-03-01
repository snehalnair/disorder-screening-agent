"""
Ground Truth Assembly for Material Dopants
==========================================

Material-agnostic script driven by a per-material YAML config file.
Combines three approaches to build a known_dopants JSON file:
  1. Bootstrap from established literature  (config/targets/bootstrap/<id>.json)
  2. Materials Project API query            (experimentally-derived structures)
  3. Semantic Scholar API mining            (successes + failures from papers)

Output: data/known_dopants/<material_id>.json

Requirements:
    pip install mp-api requests pymatgen tqdm pyyaml

Environment variables:
    MP_API_KEY   - Materials Project API key (required for Path 2)
    S2_API_KEY   - Semantic Scholar API key  (optional, higher rate limits)

Usage:
    # Run all paths for a given material
    python scripts/dopants_ground_truth_extraction.py --material nmc_layered_oxide --path all

    # Point directly at a config file
    python scripts/dopants_ground_truth_extraction.py --config config/targets/nmc_layered_oxide.yaml

    # Bootstrap only (no API keys needed)
    python scripts/dopants_ground_truth_extraction.py --material nmc_layered_oxide --path bootstrap
"""

import json
import os
import re
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional

import requests
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Project root = parent of this script's directory
PROJECT_ROOT = Path(__file__).parent.parent

# Elements that cannot be dopants (noble gases, radioactive, etc.)
EXCLUDED_ELEMENTS = {
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    "Tc", "Pm",
    "Po", "At", "Fr", "Ra",
    "Ac", "Pa", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}

# Semantic Scholar rate limit (requests between calls)
S2_DELAY = 3.5   # seconds

# Elements to search for as dopants (covers broad periodic table candidates)
DOPANT_SEARCH_ELEMENTS = [
    "Al", "Ti", "Mg", "W", "Zr", "Ta", "Nb", "Mo", "Fe", "V", "Cr",
    "B", "Ga", "Sc", "Hf", "Y", "Sb", "Bi", "Cu", "Zn", "Si", "Sn",
    "La", "Ce", "Nd", "Ca", "Sr", "Ba", "In", "Ge", "Te", "Ru",
]

# Abstract keyword patterns for outcome classification
SUCCESS_PATTERNS = [
    r"improv(ed|es|ing)\s+(cycle|cycling|capacity|stability|rate|performance)",
    r"enhanc(ed|es|ing)\s+(electrochemical|structural|thermal)",
    r"suppress(ed|es|ing)\s+(cation\s+mixing|Li/Ni|phase\s+transition)",
    r"single[\s-]phase\s+layered",
    r"maintain(ed|s|ing)\s+layered\s+structure",
    r"stable\s+cycling",
    r"retain(ed|s|ing)\s+capacity",
    r"reduced?\s+(impedance|degradation|capacity\s+fade)",
]

FAILURE_PATTERNS = [
    r"phase\s+separation",
    r"secondary\s+phase",
    r"decompos(ed|ition|ing)",
    r"structural\s+collapse",
    r"solubility\s+limit\s+exceeded",
    r"failed?\s+(to\s+form|synthesis|doping)",
    r"impurity\s+phase",
    r"inhomogeneous",
    r"loss\s+of\s+layered\s+structure",
    r"capacity\s+degra?d(ation|ed)",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_material_config(material_id: Optional[str], config_path: Optional[str]) -> dict:
    """Load material config from a YAML file.

    Accepts either a material_id (resolved to config/targets/<id>.yaml)
    or a direct path to a YAML file.
    """
    if config_path:
        path = Path(config_path)
    elif material_id:
        path = PROJECT_ROOT / "config" / "targets" / f"{material_id}.yaml"
    else:
        raise ValueError("Provide --material or --config.")

    if not path.exists():
        raise FileNotFoundError(
            f"Material config not found at {path}.\n"
            f"Available configs: {list((PROJECT_ROOT / 'config' / 'targets').glob('*.yaml'))}"
        )

    with path.open() as f:
        cfg = yaml.safe_load(f)

    log.info("Loaded material config: %s (%s)", cfg["material_id"], cfg["display_name"])
    return cfg


def load_bootstrap(cfg: dict) -> dict:
    """Load bootstrap ground truth from the JSON file specified in config."""
    bootstrap_path = PROJECT_ROOT / cfg["bootstrap_file"]
    if not bootstrap_path.exists():
        raise FileNotFoundError(
            f"Bootstrap file not found: {bootstrap_path}.\n"
            f"Create it at {cfg['bootstrap_file']} or update bootstrap_file in the config."
        )
    with bootstrap_path.open() as f:
        data = json.load(f)

    n_success = len(data.get("confirmed_successful", []))
    n_limited = len(data.get("confirmed_limited", []))
    n_failed = len(data.get("confirmed_failed", []))
    log.info(
        "Path 1: Bootstrap loaded — %d successful, %d limited, %d failed",
        n_success, n_limited, n_failed,
    )
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Path 2: Materials Project API
# ═══════════════════════════════════════════════════════════════════════════════

def query_materials_project(cfg: dict, api_key: Optional[str] = None) -> list[dict]:
    """Query Materials Project for experimentally confirmed structures of this
    material family and extract non-base elements as candidate dopants."""
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        log.warning("MP_API_KEY not set — skipping Materials Project query.")
        return []

    log.info("Path 2: Querying Materials Project API...")

    try:
        from mp_api.client import MPRester
    except ImportError:
        log.error("mp-api not installed. Run: pip install mp-api")
        return []

    mp_cfg = cfg["mp_query"]
    base_elements = set(cfg["base_elements"])
    discovered = []

    with MPRester(api_key) as mpr:
        log.info(
            "  Searching spacegroup %d structures containing %s...",
            mp_cfg["spacegroup_number"], mp_cfg["elements"],
        )
        try:
            results = mpr.materials.summary.search(
                elements=mp_cfg["elements"],
                spacegroup_number=mp_cfg["spacegroup_number"],
                fields=[
                    "material_id", "formula_pretty", "composition",
                    "is_stable", "database_IDs",
                    "energy_above_hull", "formation_energy_per_atom",
                ],
                num_chunks=None,
            )
        except Exception as e:
            log.error("MP API query failed: %s", e)
            return []

        log.info("  Found %d structures", len(results))

        experimental = [
            r for r in results
            if hasattr(r, "database_IDs") and r.database_IDs
            and any("ICSD" in str(d).upper() for d in r.database_IDs)
        ]
        log.info("  %d have ICSD (experimental) entries", len(experimental))

        dopant_evidence: dict[str, list] = defaultdict(list)
        for r in experimental:
            comp = r.composition
            extra = {str(el) for el in comp.elements} - base_elements - EXCLUDED_ELEMENTS
            for el in extra:
                fraction = comp.get_atomic_fraction(el)
                dopant_evidence[el].append({
                    "material_id": str(r.material_id),
                    "formula": r.formula_pretty,
                    "atomic_fraction": round(float(fraction), 4),
                    "is_stable": bool(r.is_stable) if r.is_stable is not None else None,
                    "e_above_hull": round(float(r.energy_above_hull), 4)
                    if r.energy_above_hull is not None else None,
                })

        log.info("  Found %d non-base elements in experimental structures", len(dopant_evidence))
        for el, entries in sorted(dopant_evidence.items()):
            max_frac = max(e["atomic_fraction"] for e in entries)
            discovered.append({
                "element": el,
                "source": "materials_project",
                "n_structures": len(entries),
                "max_atomic_fraction": round(max_frac, 4),
                "example_material_ids": [e["material_id"] for e in entries[:5]],
                "example_formulas": [e["formula"] for e in entries[:5]],
            })
            log.info("    %s: %d structures, max fraction %.3f", el, len(entries), max_frac)

    return discovered


# ═══════════════════════════════════════════════════════════════════════════════
# Path 3: Semantic Scholar
# ═══════════════════════════════════════════════════════════════════════════════

S2_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_headers() -> dict:
    headers = {"Accept": "application/json"}
    key = os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    return headers


def _s2_search(query: str, limit: int = 50, year_range: str = "2015-2026") -> list[dict]:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "paperId,title,abstract,year,citationCount,url",
        "year": year_range,
    }
    try:
        resp = requests.get(
            f"{S2_BASE}/paper/search",
            params=params,
            headers=_s2_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.exceptions.RequestException as e:
        log.warning("S2 search failed for '%s': %s", query, e)
        return []


def _classify_abstract(abstract: str) -> str:
    if not abstract:
        return "unclear"
    text = abstract.lower()
    success_hits = sum(1 for p in SUCCESS_PATTERNS if re.search(p, text))
    failure_hits = sum(1 for p in FAILURE_PATTERNS if re.search(p, text))
    if failure_hits > 0 and failure_hits >= success_hits:
        return "failure"
    if success_hits >= 2:
        return "success"
    if success_hits == 1:
        return "likely_success"
    return "unclear"


def _extract_concentration(abstract: str) -> Optional[float]:
    if not abstract:
        return None
    m = re.search(r'x\s*=\s*(0\.\d+)', abstract)
    if m:
        return float(m.group(1)) * 100
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:mol|at|atomic)?\s*%', abstract)
    if m:
        return float(m.group(1))
    return None


def mine_semantic_scholar(cfg: dict) -> list[dict]:
    """Mine Semantic Scholar using queries defined in the material config."""
    log.info("Path 3: Mining Semantic Scholar...")

    s2_cfg = cfg["s2_queries"]
    all_papers: dict[str, dict] = {}

    # Broad queries
    log.info("  Running broad queries...")
    for query in tqdm(s2_cfg["broad"], desc="  Broad queries"):
        for p in _s2_search(query, limit=80):
            if p.get("paperId"):
                all_papers[p["paperId"]] = p
        time.sleep(S2_DELAY)
    log.info("  %d unique papers from broad queries", len(all_papers))

    # Per-element queries
    log.info("  Running per-element queries...")
    element_papers: dict[str, list] = defaultdict(list)
    material_name = cfg["display_name"]

    for el in tqdm(DOPANT_SEARCH_ELEMENTS, desc="  Element queries"):
        queries = [
            f"{material_name.split('(')[0].strip()} {el} doping",
            f"layered oxide cathode {el} substitution",
        ]
        for query in queries:
            for p in _s2_search(query, limit=30):
                pid = p.get("paperId")
                if pid:
                    all_papers[pid] = p
                    element_papers[el].append(p)
            time.sleep(S2_DELAY)
    log.info("  %d total unique papers after element queries", len(all_papers))

    # Failure-specific queries
    log.info("  Running failure-specific queries...")
    for query in tqdm(s2_cfg["failure"], desc="  Failure queries"):
        for p in _s2_search(query, limit=50):
            if p.get("paperId"):
                all_papers[p["paperId"]] = p
        time.sleep(S2_DELAY)
    log.info("  %d total unique papers after failure queries", len(all_papers))

    # Classify
    log.info("  Classifying abstracts...")
    element_evidence: dict[str, dict] = defaultdict(
        lambda: {"success_papers": [], "failure_papers": [], "unclear_papers": []}
    )
    for el, papers in element_papers.items():
        for p in papers:
            abstract = p.get("abstract", "") or ""
            classification = _classify_abstract(abstract)
            record = {
                "paperId": p["paperId"],
                "title": p.get("title", ""),
                "year": p.get("year"),
                "citationCount": p.get("citationCount", 0),
                "url": p.get("url", ""),
                "classification": classification,
                "extracted_concentration_pct": _extract_concentration(abstract),
            }
            if classification in ("success", "likely_success"):
                element_evidence[el]["success_papers"].append(record)
            elif classification == "failure":
                element_evidence[el]["failure_papers"].append(record)
            else:
                element_evidence[el]["unclear_papers"].append(record)

    results = []
    for el in sorted(element_evidence):
        ev = element_evidence[el]
        n_s = len(ev["success_papers"])
        n_f = len(ev["failure_papers"])
        n_u = len(ev["unclear_papers"])

        if n_f > 0 and n_f >= n_s:
            verdict = "likely_failure"
        elif n_s >= 2:
            verdict = "likely_success"
        elif n_s == 1:
            verdict = "possible_success"
        else:
            verdict = "insufficient_evidence"

        concentrations = [
            p["extracted_concentration_pct"]
            for p in ev["success_papers"] + ev["failure_papers"]
            if p["extracted_concentration_pct"] is not None
        ]

        results.append({
            "element": el,
            "source": "semantic_scholar",
            "verdict": verdict,
            "n_success_papers": n_s,
            "n_failure_papers": n_f,
            "n_unclear_papers": n_u,
            "max_extracted_concentration_pct": max(concentrations) if concentrations else None,
            "top_success_papers": sorted(
                ev["success_papers"], key=lambda x: x.get("citationCount", 0), reverse=True
            )[:5],
            "top_failure_papers": sorted(
                ev["failure_papers"], key=lambda x: x.get("citationCount", 0), reverse=True
            )[:5],
        })

        if n_s + n_f > 0:
            log.info("    %s: %d success, %d failure, %d unclear → %s", el, n_s, n_f, n_u, verdict)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Merge
# ═══════════════════════════════════════════════════════════════════════════════

def merge_results(cfg: dict, bootstrap: dict, mp_results: list, s2_results: list) -> dict:
    """Merge all three sources. Priority: bootstrap > MP experimental > S2 mining."""
    log.info("Merging results from all sources...")

    output = {
        "metadata": {
            "material_id": cfg["material_id"],
            "parent_material": cfg["display_name"],
            "structure_type": cfg["structure_type"],
            "space_group": cfg["space_group"],
            "target_site": cfg["target_site"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": ["bootstrap_literature", "materials_project_api", "semantic_scholar_api"],
            "notes": (
                "Bootstrap entries are high-confidence from established literature. "
                "MP entries are from experimentally confirmed ICSD structures. "
                "S2 entries are mined from abstracts and should be manually verified."
            ),
        },
        "dopants": {},
    }

    for gt_class, entries in [
        ("confirmed_successful", bootstrap.get("confirmed_successful", [])),
        ("confirmed_limited",    bootstrap.get("confirmed_limited", [])),
        ("confirmed_failed",     bootstrap.get("confirmed_failed", [])),
    ]:
        for entry in entries:
            el = entry["element"]
            output["dopants"][el] = {
                **entry,
                "ground_truth_class": gt_class,
                "confidence": "high",
                "sources": ["bootstrap_literature"],
                "mp_evidence": [],
                "s2_evidence": {},
            }

    for mp_entry in mp_results:
        el = mp_entry["element"]
        if el in output["dopants"]:
            output["dopants"][el]["mp_evidence"].append(mp_entry)
            if "materials_project" not in output["dopants"][el]["sources"]:
                output["dopants"][el]["sources"].append("materials_project")
        else:
            output["dopants"][el] = {
                "element": el,
                "ground_truth_class": "mp_experimental_only",
                "confidence": "medium",
                "sources": ["materials_project"],
                "mp_evidence": [mp_entry],
                "s2_evidence": {},
                "notes": (
                    f"Found in {mp_entry['n_structures']} experimental MP structures "
                    "but not in bootstrap literature. Needs manual verification."
                ),
            }

    for s2_entry in s2_results:
        el = s2_entry["element"]
        if el in output["dopants"]:
            output["dopants"][el]["s2_evidence"] = s2_entry
            if "semantic_scholar" not in output["dopants"][el]["sources"]:
                output["dopants"][el]["sources"].append("semantic_scholar")
        elif s2_entry["verdict"] in ("likely_success", "likely_failure"):
            gt_class = (
                "s2_likely_successful" if s2_entry["verdict"] == "likely_success"
                else "s2_likely_failed"
            )
            output["dopants"][el] = {
                "element": el,
                "ground_truth_class": gt_class,
                "confidence": "low",
                "sources": ["semantic_scholar"],
                "mp_evidence": [],
                "s2_evidence": s2_entry,
                "notes": (
                    f"Discovered from Semantic Scholar mining only. "
                    f"Verdict: {s2_entry['verdict']} based on "
                    f"{s2_entry['n_success_papers']} success / "
                    f"{s2_entry['n_failure_papers']} failure papers. "
                    "Needs manual verification."
                ),
            }

    classes: dict[str, int] = defaultdict(int)
    for d in output["dopants"].values():
        classes[d["ground_truth_class"]] += 1

    output["summary"] = {
        "total_dopants": len(output["dopants"]),
        "by_class": dict(classes),
    }

    log.info("  Final ground truth: %d elements", len(output["dopants"]))
    for cls, count in sorted(classes.items()):
        log.info("    %s: %d", cls, count)

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Assemble ground truth dopant data for a target material family."
    )
    parser.add_argument(
        "--material", "-m",
        type=str,
        default=None,
        help="Material ID (resolves to config/targets/<id>.yaml). "
             "E.g. nmc_layered_oxide",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Direct path to a material config YAML file.",
    )
    parser.add_argument(
        "--path",
        choices=["all", "bootstrap", "mp", "semantic"],
        default="all",
        help="Which data source(s) to run (default: all).",
    )
    parser.add_argument(
        "--mp-key",
        type=str,
        default=None,
        help="Materials Project API key (or set MP_API_KEY env var).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override output file path (default: data/known_dopants/<material_id>.json).",
    )
    args = parser.parse_args()

    if args.mp_key:
        os.environ["MP_API_KEY"] = args.mp_key

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_material_config(args.material, args.config)

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "known_dopants" / f"{cfg['material_id']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Dopant Ground Truth Assembly: %s", cfg["display_name"])
    log.info("=" * 60)

    # ── Run selected paths ───────────────────────────────────────────────────
    bootstrap_data: dict = {}
    mp_data: list = []
    s2_data: list = []

    if args.path in ("all", "bootstrap"):
        bootstrap_data = load_bootstrap(cfg)

    if args.path in ("all", "mp"):
        mp_data = query_materials_project(cfg)

    if args.path in ("all", "semantic"):
        s2_data = mine_semantic_scholar(cfg)

    # Bootstrap always needed as the foundation for merging
    if not bootstrap_data:
        bootstrap_data = load_bootstrap(cfg)

    merged = merge_results(cfg, bootstrap_data, mp_data, s2_data)

    with output_path.open("w") as f:
        json.dump(merged, f, indent=2, default=str)

    log.info("\nGround truth written to: %s", output_path)
    log.info("Total dopants: %d", merged["summary"]["total_dopants"])
    log.info("")
    log.info("Next steps:")
    log.info("  1. Review s2_only / mp_experimental_only entries manually")
    log.info("  2. Add DOIs to bootstrap entries")
    log.info("  3. To add a new material: create config/targets/<id>.yaml "
             "and config/targets/bootstrap/<id>.json, then rerun with --material <id>")


if __name__ == "__main__":
    main()
