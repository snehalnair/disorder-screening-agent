"""
Rebuild shannon_radii.json from SMACT's ML-extended Shannon table.

Source: smact/data/shannon_radii_ML_extended.csv  (990 rows)
  Columns: ion, charge, coordination, crystal_radius, ionic_radius, reference

Output format (matching existing shannon_radii.json schema):
  {
    "Co": {
      "3": {
        "VI": {"r_crystal": 0.685, "remark": "Low Spin", "spin": "", "r_ionic": 0.545},
        ...
      },
      ...
    },
    ...
  }

Coordination numbers are stored as Roman numerals (I–XIV) so they are
compatible with the existing consumers in nodes/radius_screen.py.

Rows that cannot be converted (unrecognised CN, blank ion/charge) are
skipped with a warning so the file is always valid.
"""

import csv
import json
import pathlib
import smact

# ── Integer CN → Roman numeral ────────────────────────────────────────────────
CN_TO_ROMAN: dict[int, str] = {
    1: "I",   2: "II",   3: "III",  4: "IV",   5: "V",
    6: "VI",  7: "VII",  8: "VIII", 9: "IX",  10: "X",
    11: "XI", 12: "XII", 14: "XIV",
}


def _parse_cn(raw: str) -> str | None:
    """
    Convert a raw coordination string to a Roman numeral key.
    The ML-extended CSV stores CNs as plain integers ('4', '6', '12')
    or with a suffix ('4_n').  Returns None for anything unrecognised.
    """
    raw = raw.strip().split("_")[0]   # strip '_n' / '_sq' suffixes
    try:
        return CN_TO_ROMAN[int(raw)]
    except (ValueError, KeyError):
        return None


def main() -> None:
    smact_data_dir = pathlib.Path(smact.__file__).parent / "data"
    src_csv = smact_data_dir / "shannon_radii_ML_extended.csv"

    if not src_csv.exists():
        raise FileNotFoundError(
            f"Expected SMACT ML-extended radii at {src_csv}. "
            "Ensure smact>=2.5 is installed."
        )

    output_path = pathlib.Path(__file__).parent.parent / "data" / "shannon_radii.json"

    radii: dict[str, dict] = {}
    skipped = 0
    loaded = 0

    with src_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ion = (row.get("ion") or "").strip()
            charge_raw = (row.get("charge") or "").strip()
            cn_raw = (row.get("coordination") or "").strip()
            r_crystal_raw = (row.get("crystal_radius") or "").strip()
            r_ionic_raw = (row.get("ionic_radius") or "").strip()
            reference = (row.get("reference") or "").strip()

            # ── Validate required fields ──────────────────────────────────────
            if not ion or not charge_raw:
                skipped += 1
                continue

            cn_roman = _parse_cn(cn_raw)
            if cn_roman is None:
                print(f"  SKIP {ion}^{charge_raw} CN={cn_raw!r} — unrecognised coordination")
                skipped += 1
                continue

            try:
                r_crystal = round(float(r_crystal_raw), 4)
                r_ionic = round(float(r_ionic_raw), 4)
            except ValueError:
                print(f"  SKIP {ion}^{charge_raw} CN={cn_roman} — non-numeric radius")
                skipped += 1
                continue

            # ── Build nested dict ─────────────────────────────────────────────
            radii.setdefault(ion, {})
            radii[ion].setdefault(charge_raw, {})

            # If CN key already exists keep whichever was loaded first
            # (Shannon originals come first in the CSV; ML entries follow).
            if cn_roman not in radii[ion][charge_raw]:
                radii[ion][charge_raw][cn_roman] = {
                    "r_crystal": r_crystal,
                    "remark": reference if reference != "Shannon" else "",
                    "spin": "",
                    "r_ionic": r_ionic,
                }
                loaded += 1

    # ── Report ────────────────────────────────────────────────────────────────
    total_ions = sum(
        len(cn_dict)
        for os_dict in radii.values()
        for cn_dict in os_dict.values()
    )
    print(
        f"Loaded {loaded} entries from CSV  |  "
        f"Unique (element, OS, CN) triples: {total_ions}  |  "
        f"Skipped: {skipped}"
    )

    with output_path.open("w") as f:
        json.dump(radii, f, indent=2, sort_keys=True)

    print(f"Written → {output_path}  ({total_ions} ion entries across {len(radii)} elements)")


if __name__ == "__main__":
    main()
