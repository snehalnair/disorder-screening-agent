"""
Generate element_costs.json for all 118 elements.

Covers Z=1–103 with full SMACT HHI data, and Z=104–118 (superheavy synthetic
elements) with null HHI entries. Prices come from element_prices.csv (base
values) overlaid with manual_prices (authoritative 2024 spot prices for key
battery elements). EU CRM status from the 2023/2024 critical raw materials list.

Output: data/element_costs.json
"""

import csv
import json
from pathlib import Path

import smact
from smact import element_dictionary, ordered_elements

# ── EU Critical Raw Materials (2023/2024 list, including Strategic Raw Materials) ──
EU_CRMS = {
    "Al", "Sb", "As", "Ba", "Be", "Bi", "B", "Co", "C", "F",
    "Ga", "Ge", "Hf", "He", "Li", "Mg", "Mn", "Ni", "Nb", "P",
    "Sc", "Si", "Sr", "Ta", "Ti", "W", "V", "Cu",
    # REEs
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y",
    # PGMs
    "Pt", "Pd", "Ir", "Rh", "Ru", "Os",
}

# ── Authoritative 2024/2025 spot/contract prices (override CSV where needed) ──
MANUAL_PRICES: dict[str, float] = {
    "Li":  75.00,   # ~$14 k/t Li₂CO₃ → elemental equivalent
    "Co":  28.00,   # LME avg 2024
    "Ni":  16.80,   # LME cash avg 2024
    "Mn":   2.30,   # electrolytic manganese metal
    "C":    0.80,   # natural graphite (flake) export avg
    "Cu":   9.00,   # LME avg
    "Al":   2.40,   # LME avg
    "Si":   3.50,   # silicon metal
    "V":   30.00,   # ferrovanadium
    "Ti":  11.00,   # titanium sponge
}

# ── IUPAC symbols for Z=104–118 (superheavy / synthetic) ──
SUPERHEAVY: list[tuple[int, str]] = [
    (104, "Rf"), (105, "Db"), (106, "Sg"), (107, "Bh"), (108, "Hs"),
    (109, "Mt"), (110, "Ds"), (111, "Rg"), (112, "Cn"), (113, "Nh"),
    (114, "Fl"), (115, "Mc"), (116, "Lv"), (117, "Ts"), (118, "Og"),
]


def load_prices_from_csv(path: Path) -> dict[str, float]:
    prices: dict[str, float] = {}
    if not path.exists():
        raise FileNotFoundError(
            f"Price table CSV not found at {path}. "
            "Run this script from the data/ directory or provide the correct path."
        )
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            try:
                prices[symbol] = float(row["price_usd_per_kg"])
            except (KeyError, ValueError):
                raise ValueError(
                    f"Invalid or missing price_usd_per_kg for '{symbol}' in {path}"
                )
    return prices


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    prices_path = data_dir / "element_prices.csv"

    csv_prices = load_prices_from_csv(prices_path)
    combined_prices: dict[str, float] = {**csv_prices, **MANUAL_PRICES}

    elem_dict = element_dictionary()
    # Z=1–103: use SMACT element objects for HHI data
    smact_syms = ordered_elements(1, 103)

    element_costs: list[dict] = []

    # ── Z = 1–103 (SMACT-covered) ──────────────────────────────────────────────
    for atomic_number, symbol in enumerate(smact_syms, start=1):
        elem = elem_dict[symbol]
        price = combined_prices.get(symbol)
        if price is None:
            raise RuntimeError(
                f"No price found for {symbol} (Z={atomic_number}). "
                "Add it to element_prices.csv."
            )
        element_costs.append(
            {
                "symbol": symbol,
                "atomic_number": atomic_number,
                "HHI_production": elem.HHI_p,
                "HHI_reserves": elem.HHI_r,
                "price_usd_per_kg": price,
                "is_EU_CRM": symbol in EU_CRMS,
            }
        )

    # ── Z = 104–118 (superheavy / synthetic — no HHI or commercial price) ─────
    for atomic_number, symbol in SUPERHEAVY:
        price = combined_prices.get(symbol, 1e12)   # nominal; effectively infinite
        element_costs.append(
            {
                "symbol": symbol,
                "atomic_number": atomic_number,
                "HHI_production": None,
                "HHI_reserves": None,
                "price_usd_per_kg": price,
                "is_EU_CRM": False,
            }
        )

    output_path = data_dir / "element_costs.json"  # data/element_costs.json
    with output_path.open("w") as f:
        json.dump(element_costs, f, indent=2)

    print(
        f"Generated {output_path.name} with {len(element_costs)} entries "
        f"(Z=1–103 with HHI, Z=104–118 superheavy placeholders)."
    )


if __name__ == "__main__":
    main()
