#!/usr/bin/env python3
"""
Consolidate the INFORM Severity monthly snapshots into one tidy long-format CSV.

Reads:   Data/Third-Party/DRMKC-INFORM/snapshots/*.xlsx
Writes:  Data/Third-Party/DRMKC-INFORM/inform_severity_long.csv

Schema of the output:
    CRISIS ID, COUNTRY, ISO3, severity (1–10), category (1–5),
    year, month, snapshot

Tolerates schema drift across 2020 → 2026 by trying both `INFORM Severity -
country` and `INFORM Severity - all crises` sheets, and the first few
plausible header rows.

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
OUT = HERE / "inform_severity_long.csv"

SHEETS = ("INFORM Severity - country", "INFORM Severity - all crises")
HEADER_CANDIDATES = (1, 2, 3)


def load_one(path: Path) -> pd.DataFrame | None:
    for sheet in SHEETS:
        for header in HEADER_CANDIDATES:
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header, dtype=str)
            except Exception:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            if "ISO3" in df.columns and "INFORM Severity Index" in df.columns:
                return df
    return None


def main() -> int:
    files = sorted(SNAPS.glob("*.xlsx"))
    if not files:
        print("No snapshots found. Run download.py first.", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        m = re.match(r"(\d{6})", f.name)
        if not m:
            continue
        year, month = int(m.group(1)[:4]), int(m.group(1)[4:])
        df = load_one(f)
        if df is None:
            print(f"  skip {f.name}  (no matching sheet)")
            continue
        df = df[df["ISO3"].notna() & (df["ISO3"].str.len() == 3)]
        cols = [c for c in ("CRISIS ID", "COUNTRY", "ISO3",
                            "INFORM Severity Index", "INFORM Severity category")
                if c in df.columns]
        sub = df[cols].copy()
        sub["year"] = year
        sub["month"] = month
        sub["snapshot"] = f"{year}-{month:02d}"
        rows.append(sub)

    if not rows:
        print("Nothing consolidated.", file=sys.stderr)
        return 1

    long = pd.concat(rows, ignore_index=True)
    long["severity"] = pd.to_numeric(long.get("INFORM Severity Index"), errors="coerce")
    long["category"] = pd.to_numeric(long.get("INFORM Severity category"), errors="coerce")
    long = long.drop(columns=["INFORM Severity Index", "INFORM Severity category"], errors="ignore")
    long = long.dropna(subset=["severity"]).reset_index(drop=True)

    long.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  ({len(long):,} rows · {OUT.stat().st_size/1024:.0f} KB)")
    print(f"Countries: {long['ISO3'].nunique()}  |  Snapshots: {long['snapshot'].nunique()}  |  Crises: {long['CRISIS ID'].nunique()}")
    print(f"Severity range: [{long['severity'].min():.2f}, {long['severity'].max():.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
