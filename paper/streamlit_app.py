#!/usr/bin/env python3
"""
Disorder-Risk Predictor — Streamlit App
========================================

Upload a CIF file (or select a preset material), and the app computes
whether ordered-cell dopant screening is safe or whether disorder-aware
(SQS-ensemble) screening is required.

Usage:
    pip install streamlit pymatgen numpy
    streamlit run paper/streamlit_app.py

Based on:
    R = property_scope × sublattice_anisotropy + 0.3 × (n_TM_species − 1)
    R > 1.0  →  UNSAFE (disorder-aware screening required)
    R ≤ 1.0  →  SAFE   (ordered-only screening sufficient)
"""

import io
import tempfile

import numpy as np
import streamlit as st

# ── Transition metals for sublattice detection ──────────────────────────
TM_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
}

# Also include common non-TM cation targets (for perovskites, etc.)
CATION_ELEMENTS = TM_ELEMENTS | {
    "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "Al", "Ga", "In",
    "Sn", "Pb", "Bi", "Sb", "Ge", "Si",
}

RISK_THRESHOLD = 1.0
RHO_THRESHOLD = 0.50

# ── Preset materials ────────────────────────────────────────────────────
PRESETS = {
    "— Enter formula or upload CIF —": None,
    "LiCoO₂ (layered R-3m)": {"anisotropy": 1.94, "n_tm": 1, "structure": "layered R-3m"},
    "LiNiO₂ (layered R-3m)": {"anisotropy": 1.93, "n_tm": 1, "structure": "layered R-3m"},
    "NMC811 (layered R-3m)": {"anisotropy": 1.93, "n_tm": 3, "structure": "layered R-3m"},
    "LiMn₂O₄ (spinel Fd-3m)": {"anisotropy": 1.00, "n_tm": 1, "structure": "spinel Fd-3m"},
    "SrTiO₃ (perovskite Pm-3m)": {"anisotropy": 1.00, "n_tm": 1, "structure": "perovskite Pm-3m"},
    "LiFePO₄ (olivine Pnma)": {"anisotropy": 1.21, "n_tm": 1, "structure": "olivine Pnma"},
    "Na₃V₂(PO₄)₃ (NASICON R-3c)": {"anisotropy": 1.15, "n_tm": 1, "structure": "NASICON R-3c"},
}


def fetch_structure_from_mp(formula):
    """Fetch crystal structure from Materials Project by formula.

    Uses the legacy MP API (no key required for basic queries) or
    pymatgen's MPRester if available.
    """
    try:
        # Try mp-api first (requires mp_api package)
        from mp_api.client import MPRester
        with MPRester() as mpr:
            docs = mpr.summary.search(
                formula=formula,
                fields=["material_id", "structure", "energy_above_hull"],
            )
            if docs:
                # Pick the most stable (lowest e_above_hull)
                docs.sort(key=lambda d: d.energy_above_hull or 0)
                return docs[0].structure, str(docs[0].material_id)
    except Exception:
        pass

    try:
        # Fallback: pymatgen's MPRester (legacy API)
        from pymatgen.ext.matproj import MPRester as LegacyMPRester
        with LegacyMPRester() as mpr:
            results = mpr.get_structures(formula, final=True)
            if results:
                return results[0], "MP (legacy)"
    except Exception:
        pass

    # Fallback: build from pymatgen's common structures
    try:
        from pymatgen.core import Structure, Lattice

        # Try to parse as a known composition and build from spacegroup
        # This handles simple cases like "NaCl", "SrTiO3", etc.
        from pymatgen.core import Composition
        comp = Composition(formula)
        # Can't auto-generate without more info
        return None, f"Could not fetch {formula} from Materials Project. Please upload a CIF file."
    except Exception as e:
        return None, str(e)

    return None, "Structure not found"


def compute_anisotropy_from_structure(struct, target_species=None):
    """Compute sublattice anisotropy from a pymatgen Structure.

    Returns (anisotropy, nn_dist, next_dist, n_tm_species, target_species, details).
    """
    from pymatgen.core import Structure

    # Identify target sublattice (transition metals or user-specified)
    all_species = set(str(sp) for sp in struct.species)
    tm_in_struct = all_species & TM_ELEMENTS

    if target_species:
        target_set = {target_species}
    elif tm_in_struct:
        target_set = tm_in_struct
    else:
        # Fallback: use the most common non-O, non-anion species
        from collections import Counter
        counts = Counter(str(sp) for sp in struct.species
                         if str(sp) not in {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"})
        if counts:
            target_set = {counts.most_common(1)[0][0]}
        else:
            return None, None, None, 0, None, "No cation sublattice found"

    # Get target site indices
    target_indices = [i for i, sp in enumerate(struct.species) if str(sp) in target_set]
    n_tm_species = len(target_set)

    if len(target_indices) < 2:
        return None, None, None, n_tm_species, target_set, "Need at least 2 target sites"

    # Compute all pairwise distances within the sublattice
    dists = []
    for i, idx_i in enumerate(target_indices):
        for j, idx_j in enumerate(target_indices):
            if idx_i < idx_j:
                d = struct.get_distance(idx_i, idx_j)
                if d < 12.0:  # reasonable cutoff
                    dists.append(d)

    if len(dists) < 2:
        return None, None, None, n_tm_species, target_set, "Too few distances found"

    dists.sort()

    # Cluster distances into shells (0.3 A tolerance)
    shells = []
    for d in dists:
        placed = False
        for shell in shells:
            if abs(d - shell[0]) < 0.3:
                shell.append(d)
                placed = True
                break
        if not placed:
            shells.append([d])

    if len(shells) < 2:
        return 1.0, np.mean(shells[0]), np.mean(shells[0]), n_tm_species, target_set, "Single shell — isotropic"

    nn_dist = np.mean(shells[0])
    next_dist = np.mean(shells[1])
    anisotropy = next_dist / nn_dist

    details = (
        f"Target sublattice: {', '.join(sorted(target_set))}\n"
        f"Sites found: {len(target_indices)}\n"
        f"NN distance: {nn_dist:.3f} A ({len(shells[0])} pairs)\n"
        f"Next shell: {next_dist:.3f} A ({len(shells[1])} pairs)\n"
        f"Anisotropy ratio: {anisotropy:.2f}\n"
        f"Distinct TM species: {n_tm_species}"
    )

    return anisotropy, nn_dist, next_dist, n_tm_species, target_set, details


def compute_risk(anisotropy, property_scope, n_tm):
    """R = property_scope * anisotropy + 0.3 * (n_tm - 1)"""
    return property_scope * anisotropy + 0.3 * (n_tm - 1)


def risk_explanation(anisotropy, property_scope, n_tm, risk):
    """Generate human-readable explanation of the risk score."""
    parts = []

    # Anisotropy contribution
    if anisotropy > 1.5:
        parts.append(
            f"**High sublattice anisotropy** ({anisotropy:.2f}): "
            f"The target-site sublattice is strongly 2D (layered). "
            f"Dopant arrangement within a layer has long-range effects on "
            f"interlayer interactions, making global properties sensitive to disorder."
        )
    elif anisotropy > 1.1:
        parts.append(
            f"**Moderate sublattice anisotropy** ({anisotropy:.2f}): "
            f"The sublattice has some directional asymmetry. "
            f"Disorder sensitivity is borderline — depends on property type."
        )
    else:
        parts.append(
            f"**Low sublattice anisotropy** ({anisotropy:.2f}): "
            f"The sublattice is approximately isotropic (3D). "
            f"Each dopant site samples a similar local environment regardless "
            f"of arrangement, so disorder has minimal effect on rankings."
        )

    # Property scope contribution
    if property_scope == 1:
        parts.append(
            "**Global property** (voltage, volume change): "
            "These properties depend on the total energy difference between "
            "two states (lithiated vs delithiated), so they integrate over "
            "the entire structure and are sensitive to long-range order."
        )
    else:
        parts.append(
            "**Local property** (formation energy, defect energy): "
            "These properties are dominated by the local coordination "
            "environment of the dopant, so they are relatively insensitive "
            "to how other dopants are arranged farther away."
        )

    # Multi-TM contribution
    if n_tm > 1:
        parts.append(
            f"**Multi-component sublattice** ({n_tm} TM species): "
            f"Additional compositional disorder from {n_tm} transition metals "
            f"on the same sublattice increases the configuration space "
            f"and adds +{0.3 * (n_tm - 1):.1f} to the risk score."
        )

    # Overall verdict
    if risk > RISK_THRESHOLD:
        verdict = (
            f"**Prediction: UNSAFE** (R = {risk:.2f} > {RISK_THRESHOLD})\n\n"
            "Disorder-aware screening (e.g., SQS ensembles) is recommended. "
            "Ordered-only calculations may produce misleading dopant rankings."
        )
    else:
        verdict = (
            f"**Prediction: SAFE** (R = {risk:.2f} ≤ {RISK_THRESHOLD})\n\n"
            "Ordered-only screening should produce reliable dopant rankings. "
            "SQS ensembles are not necessary for this property."
        )

    return parts, verdict


# ── Streamlit UI ────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Disorder-Risk Predictor",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Disorder-Risk Predictor")
    st.markdown(
        "**Does chemical disorder change computational dopant rankings?**  \n"
        "Enter a formula, upload a CIF, or select a preset material to find out."
    )
    st.markdown("---")

    # ── Sidebar: inputs ──
    with st.sidebar:
        st.header("Input")

        input_method = st.radio(
            "How to provide the structure:",
            ["Preset material", "Enter formula", "Upload CIF"],
        )

        preset = None
        cif_file = None
        formula_input = None
        target_sp = None

        if input_method == "Preset material":
            preset = st.selectbox(
                "Select material",
                [k for k in PRESETS.keys() if k != "— Enter formula or upload CIF —"],
            )
        elif input_method == "Enter formula":
            formula_input = st.text_input(
                "Material formula",
                placeholder="e.g., LiCoO2, NaMnO2, SrTiO3",
                help="Fetches the most stable structure from Materials Project."
            )
            target_sp = st.text_input(
                "Target species (dopant sublattice)",
                placeholder="e.g., Co, Fe, Ti, Mn",
                help="Which sublattice will be doped. Auto-detects TM sites if blank."
            )
        elif input_method == "Upload CIF":
            cif_file = st.file_uploader("Upload CIF file", type=["cif"])
            target_sp = st.text_input(
                "Target species (optional)",
                placeholder="e.g., Co, Fe, Ti",
                help="Which sublattice to analyse. Leave blank for auto-detect (TM sites)."
            )

        st.markdown("---")
        st.header("Property type")
        prop_type = st.radio(
            "What property are you screening?",
            [
                "Voltage / volume change (global)",
                "Formation energy / defect energy (local)",
            ],
            help=(
                "Global properties (voltage, volume change) depend on total "
                "energy differences and are more sensitive to disorder. "
                "Local properties (formation energy) depend mainly on the "
                "dopant's immediate coordination environment."
            ),
        )
        property_scope = 1 if "global" in prop_type.lower() else 0

        st.markdown("---")
        st.markdown(
            "**Reference:**  \n"
            "*Does disorder change the ranking?*  \n"
            "Nair et al. (2026)"
        )

    # ── Main content ──
    anisotropy = None
    n_tm = 1
    details = None
    struct_name = None

    if input_method == "Preset material" and preset:
        data = PRESETS[preset]
        anisotropy = data["anisotropy"]
        n_tm = data["n_tm"]
        struct_name = preset
        details = (
            f"Structure type: {data['structure']}\n"
            f"Anisotropy: {anisotropy:.2f}\n"
            f"TM species: {n_tm}"
        )

    elif input_method == "Enter formula" and formula_input:
        with st.spinner(f"Fetching structure for {formula_input} from Materials Project..."):
            struct, mp_id = fetch_structure_from_mp(formula_input)
        if struct is None:
            st.error(f"Could not fetch structure: {mp_id}")
        else:
            st.success(f"Loaded {formula_input} ({mp_id}, {len(struct)} atoms)")
            struct_name = f"{formula_input} ({mp_id})"
            target = target_sp.strip() if target_sp else None
            anisotropy, nn_dist, next_dist, n_tm, target_set, details = \
                compute_anisotropy_from_structure(struct, target)
            if anisotropy is None:
                st.error(f"Could not compute anisotropy: {details}")

    elif input_method == "Upload CIF" and cif_file is not None:
        try:
            from pymatgen.core import Structure

            # Write to temp file for pymatgen
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w") as tmp:
                content = cif_file.getvalue().decode("utf-8")
                tmp.write(content)
                tmp_path = tmp.name

            struct = Structure.from_file(tmp_path)
            struct_name = cif_file.name

            target = target_sp.strip() if target_sp else None
            anisotropy, nn_dist, next_dist, n_tm, target_set, details = \
                compute_anisotropy_from_structure(struct, target)

            if anisotropy is None:
                st.error(f"Could not compute anisotropy: {details}")
                return

        except Exception as e:
            st.error(f"Error parsing CIF: {e}")
            return

    if anisotropy is not None:
        # Compute risk
        risk = compute_risk(anisotropy, property_scope, n_tm)
        is_safe = risk <= RISK_THRESHOLD

        # ── Result display ──
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.metric("Risk Score (R)", f"{risk:.2f}")
        with col2:
            st.metric("Threshold", f"{RISK_THRESHOLD:.1f}")
        with col3:
            if is_safe:
                st.success("✅ SAFE — Ordered screening OK")
            else:
                st.error("⚠️ UNSAFE — Use disorder-aware screening")

        st.markdown("---")

        # ── Formula breakdown ──
        st.subheader("Risk Score Decomposition")

        formula_col1, formula_col2 = st.columns(2)

        with formula_col1:
            st.latex(
                r"R = \underbrace{%d}_{\text{property scope}} "
                r"\times \underbrace{%.2f}_{\text{anisotropy}} "
                r"+ 0.3 \times \underbrace{(%d - 1)}_{\text{n\_TM - 1}} "
                r"= %.2f" % (property_scope, anisotropy, n_tm, risk)
            )

        with formula_col2:
            if details:
                st.code(details, language=None)

        st.markdown("---")

        # ── Explanation ──
        st.subheader("Why this prediction?")
        parts, verdict = risk_explanation(anisotropy, property_scope, n_tm, risk)

        for part in parts:
            st.markdown(part)

        st.markdown("---")
        st.markdown(verdict)

        # ── Confidence note ──
        st.markdown("---")
        st.subheader("Confidence & Caveats")

        # Distance from threshold
        margin = abs(risk - RISK_THRESHOLD)
        if margin < 0.2:
            confidence = "Low"
            conf_color = "orange"
            conf_note = (
                f"The risk score ({risk:.2f}) is close to the threshold ({RISK_THRESHOLD}). "
                f"This is a borderline case — consider running a small SQS test "
                f"(3-5 realisations) to verify."
            )
        elif margin < 0.5:
            confidence = "Medium"
            conf_color = "blue"
            conf_note = (
                f"The risk score ({risk:.2f}) is moderately separated from the threshold. "
                f"The prediction is likely correct but has not been validated for "
                f"every possible structure type."
            )
        else:
            confidence = "High"
            conf_color = "green"
            conf_note = (
                f"The risk score ({risk:.2f}) is well separated from the threshold. "
                f"High confidence in this prediction based on validation across "
                f"7 materials and 4 structure types."
            )

        st.markdown(f"**Confidence: :{conf_color}[{confidence}]** (margin from threshold: {margin:.2f})")
        st.info(conf_note)

        st.markdown(
            "**General caveats:**\n"
            "- Validated on 7 materials (layered, spinel, perovskite, olivine, NASICON)\n"
            "- Predictor is conservative: may flag borderline cases as UNSAFE "
            "(false positives) but has zero false negatives in validation set\n"
            "- Assumes dopant concentration ~5-15% on the target sublattice\n"
            "- Does not account for strong Jahn-Teller or charge-ordering effects"
        )

        # ── Comparison table ──
        st.markdown("---")
        st.subheader("Validation Dataset")

        import pandas as pd
        val_data = [
            {"Material": "LiCoO₂", "Structure": "layered R-3m", "Property": "Voltage",
             "R": 1.94, "Actual ρ": -0.25, "Prediction": "UNSAFE ✓"},
            {"Material": "LiCoO₂", "Structure": "layered R-3m", "Property": "Ef",
             "R": 0.0, "Actual ρ": 0.76, "Prediction": "SAFE ✓"},
            {"Material": "LiNiO₂", "Structure": "layered R-3m", "Property": "Voltage",
             "R": 1.93, "Actual ρ": -0.06, "Prediction": "UNSAFE ✓"},
            {"Material": "LiNiO₂", "Structure": "layered R-3m", "Property": "Ef",
             "R": 0.0, "Actual ρ": 0.82, "Prediction": "SAFE ✓"},
            {"Material": "NMC811", "Structure": "layered R-3m", "Property": "Voltage",
             "R": 2.53, "Actual ρ": 0.09, "Prediction": "UNSAFE ✓"},
            {"Material": "LiMn₂O₄", "Structure": "spinel Fd-3m", "Property": "Voltage",
             "R": 1.0, "Actual ρ": 0.95, "Prediction": "SAFE ✓"},
            {"Material": "SrTiO₃", "Structure": "perovskite Pm-3m", "Property": "ΔV",
             "R": 1.0, "Actual ρ": 0.94, "Prediction": "SAFE ✓"},
            {"Material": "LiFePO₄", "Structure": "olivine Pnma", "Property": "Voltage",
             "R": 1.21, "Actual ρ": 0.99, "Prediction": "UNSAFE (FP)"},
            {"Material": "LiFePO₄", "Structure": "olivine Pnma", "Property": "Ef",
             "R": 0.0, "Actual ρ": 1.00, "Prediction": "SAFE ✓"},
        ]
        df = pd.DataFrame(val_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("FP = false positive (predictor said UNSAFE but actual ρ ≥ 0.50). Zero false negatives.")

    else:
        # Landing state
        st.info(
            "👈 Select a preset material or upload a CIF file in the sidebar to get started."
        )

        st.markdown(
            """
            ### How it works

            The disorder-risk score **R** predicts whether chemical disorder
            (random dopant placement) will change computational dopant rankings
            compared to ordered-cell calculations.

            **Formula:**
            """
        )
        st.latex(r"R = \text{property\_scope} \times \text{sublattice\_anisotropy} + 0.3 \times (n_{\text{TM}} - 1)")

        st.markdown(
            """
            | Descriptor | Meaning | Source |
            |-----------|---------|--------|
            | **sublattice_anisotropy** | Ratio of 2nd to 1st neighbor shell distance on the dopant sublattice | Crystal structure (CIF) |
            | **property_scope** | 0 for local properties (Ef), 1 for global (voltage, ΔV) | User selection |
            | **n_TM** | Number of distinct transition metal species on the target sublattice | Crystal structure |

            **Decision rule:** R > 1.0 → disorder-aware screening required
            """
        )


if __name__ == "__main__":
    main()
