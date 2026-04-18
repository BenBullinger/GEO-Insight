#!/usr/bin/env python3
"""Materialise analysis/enriched.parquet.

Rebuilds the full L1–L5 enriched frame from source CSVs and writes it to
`analysis/enriched.parquet` (gitignored). The analysis app prefers this
snapshot when fresh, so running this after a data refresh keeps cold
starts fast.

Usage:
    python scripts/refresh_enriched.py [--year YEAR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))

import features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    path = features.save_enriched_frame(year=args.year)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(REPO)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
