#!/usr/bin/env python3
"""
Consolidate the granular sub-indicators from `Crisis Indicator Data` sheets
across all INFORM Severity monthly snapshots.

Where `consolidate.py` produces the composite severity index, this script
extracts the decomposed primitives (PIN at each humanitarian-conditions
level, displacement, access constraints, fatalities, duration) so that
downstream analysis can rely on auditable indicators rather than on a
composite whose scale changed in Feb 2026.

Reads:   Data/Third-Party/DRMKC-INFORM/snapshots/*.xlsx
Writes:  Data/Third-Party/DRMKC-INFORM/inform_indicators_long.csv

Schema of the output (long-format; one row per crisis × month):
    CRISIS ID, COUNTRY, ISO3,
    affected, displaced, injured, fatalities,
    pin_level_1, pin_level_2, pin_level_3, pin_level_4, pin_level_5,
    access_limited, access_restricted, impediments_bureaucratic,
    year, month, snapshot

Tolerates schema drift across 2020 → 2026 by fuzzy-matching column names
(case-insensitive, normalised whitespace).

Requires pandas + openpyxl (installed in dashboard/.venv).
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

HERE = Path(__file__).resolve().parent
SNAPS = HERE / "snapshots"
OUT = HERE / "inform_indicators_long.csv"

SHEET_CANDIDATES = ("Crisis Indicator Data", "Crisis indicator data")

# Canonical short name → patterns to match in the raw column name
# (lowercase, whitespace-collapsed substring matching)
INDICATOR_PATTERNS = {
    "affected":            ["total # of people affected"],
    "displaced":           ["displaced"],
    "injured":             ["injuries"],
    "fatalities":          ["fatalities"],
    "pin_level_1":         ["minimal humanitarian needs", "(level 1)"],
    "pin_level_2":         ["stressed humanitarian", "(level 2)"],
    "pin_level_3":         ["moderate humanitarian", "(level 3)"],
    "pin_level_4":         ["severe humanitarian", "(level 4)"],
    "pin_level_5":         ["extreme humanitarian", "(level 5)"],
    "access_limited":      ["limited access"],
    "access_restricted":   ["restricted access"],
    "impediments_bureaucratic": ["impediments to entry"],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def match_column(raw_cols: list[str], needles: list[str]) -> str | None:
    """Return the first column name whose normalised form contains all needles."""
    for col in raw_cols:
        n = norm(col)
        if all(nd.lower() in n for nd in needles):
            return col
    return None


def load_one(path: Path) -> pd.DataFrame | None:
    for sheet in SHEET_CANDIDATES:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "ISO3" in df.columns:
            return df
    return None


def main() -> int:
    files = sorted(SNAPS.glob("*.xlsx"))
    if not files:
        print("No snapshots. Run download.py first.", file=sys.stderr)
        return 1

    rows = []
    missing_stats = {k: 0 for k in INDICATOR_PATTERNS}
    for f in files:
        m = re.match(r"(\d{6})", f.name)
        if not m:
            continue
        year, month = int(m.group(1)[:4]), int(m.group(1)[4:])

        df = load_one(f)
        if df is None:
            print(f"  skip {f.name} (no Crisis Indicator Data sheet)")
            continue

        df = df[df["ISO3"].notna() & (df["ISO3"].astype(str).str.len() == 3)]

        # Resolve each canonical indicator against the actual columns
        rec = {
            "CRISIS ID": df.get("CRISIS ID", df.get("Crisis ID")),
            "COUNTRY":   df.get("COUNTRY", df.get("Country")),
            "ISO3":      df["ISO3"],
        }
        for short, needles in INDICATOR_PATTERNS.items():
            col = match_column(list(df.columns), needles)
            if col:
                rec[short] = pd.to_numeric(df[col], errors="coerce")
            else:
                rec[short] = pd.Series([pd.NA] * len(df), index=df.index)
                missing_stats[short] += 1

        out = pd.DataFrame(rec)
        out["year"] = year
        out["month"] = month
        out["snapshot"] = f"{year}-{month:02d}"
        rows.append(out)

    if not rows:
        print("Nothing consolidated.", file=sys.stderr)
        return 1

    long = pd.concat(rows, ignore_index=True)
    # Drop all-NaN indicator rows (usually header detritus that snuck past ISO3 filter)
    long.to_csv(OUT, index=False)

    print(f"Wrote {OUT}  ({len(long):,} rows · {OUT.stat().st_size/1024:.0f} KB)")
    print(f"Countries: {long['ISO3'].nunique()}  |  Snapshots: {long['snapshot'].nunique()}")
    print("\nIndicator availability across snapshots (higher = worse):")
    for k, v in missing_stats.items():
        pct = 100 * v / len(files)
        note = "" if v == 0 else f"  ({pct:.0f}% of snapshots missing this column)"
        print(f"  {k:30s} missing in {v}/{len(files)} snapshots{note}")

    # Spot-check: latest snapshot for a priority country
    latest = long[(long["snapshot"] == long["snapshot"].max()) & (long["ISO3"] == "SDN")]
    if not latest.empty:
        print(f"\nSDN spot-check ({long['snapshot'].max()}):")
        for col in ["affected", "displaced", "pin_level_4", "pin_level_5", "access_restricted"]:
            v = latest[col].iloc[0] if col in latest.columns else None
            print(f"  {col:25s} {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
