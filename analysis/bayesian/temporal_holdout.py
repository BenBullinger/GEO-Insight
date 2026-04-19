"""Temporal held-out validation — fit on 2024 data, predict 2025 picks.

The held-out test that actually matters operationally: would last year's
fit have predicted this year's underfunded-emergency selections? We
build the enriched frame separately for the 2024 and 2025 cycles, fit
the cross-sectional Bayesian model on each, and score both rankings
against four external benchmarks. The 2024 fit never sees 2025 data,
so its precision against CERF UFE 2025 w1 (selected March 2025) is a
genuine out-of-sample predictive metric.

A full AR(1) temporal model is sketched in proposal/methodology.md §5
but is not yet identified by our data: the six attributes are annual
(HRP, FTS), so we have at most two time points per country across
2024-2025, well below what AR(1) inference needs to discriminate
persistence from noise. INFORM severity is monthly and could anchor a
partial-temporal model; that is on the explicit roadmap.

Usage:
    dashboard/.venv/bin/python -m analysis.bayesian.temporal_holdout
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import features
    from analysis import validation as val

    print("[temporal] building 2024 enriched frame (training cycle)…")
    df24 = features.build_enriched_frame(year=2024)
    print(f"           n = {len(df24)} countries; "
          f"HRP-eligible (per_pin_gap observed) = {df24['per_pin_gap'].notna().sum()}")

    print("[temporal] loading 2025 enriched frame (concurrent baseline)…")
    df25 = features.load_cached_enriched_frame()
    if df25 is None:
        df25 = features.build_enriched_frame(year=2025)
    print(f"           n = {len(df25)} countries; "
          f"HRP-eligible = {df25['per_pin_gap'].notna().sum()}")

    rank24 = (-df24.dropna(subset=["theta_median"])["theta_median"]).rank(method="average")
    rank25 = (-df25.dropna(subset=["theta_median"])["theta_median"]).rank(method="average")

    benchmarks = [
        ("CERF UFE 2024 w2 (Dec 2024)", set(val.load_cerf_ufe(2024).query("window == 2")["iso3"]), 10),
        ("CERF UFE 2025 w1 (Mar 2025)", set(val.load_cerf_ufe(2025).query("window == 1")["iso3"]), 10),
        ("CERF UFE 2025 w2 (Dec 2025)", set(val.load_cerf_ufe(2025).query("window == 2")["iso3"]),  7),
        ("CARE BTS 2024",               set(val.load_care_bts(2024)["iso3"]),                       10),
    ]

    print()
    print(f"{'benchmark':<32s}  {'k':>3s}  {'2024-fit (held-out)':>22s}  {'2025-fit (concurrent)':>22s}")
    print("-" * 84)
    for name, bset, k in benchmarks:
        t24 = set(rank24.nsmallest(k).index.astype(str))
        t25 = set(rank25.nsmallest(k).index.astype(str))
        print(
            f"{name:<32s}  {k:>3d}  "
            f"{len(t24 & bset):>3d}/{k:<3d} ({len(t24 & bset)/k:.2f})         "
            f"{len(t25 & bset):>3d}/{k:<3d} ({len(t25 & bset)/k:.2f})"
        )

    top10_24 = sorted(rank24.nsmallest(10).index.astype(str))
    top10_25 = sorted(rank25.nsmallest(10).index.astype(str))
    overlap = sorted(set(top10_24) & set(top10_25))

    print()
    print(f"[temporal] 2024-fit top-10: {top10_24}")
    print(f"[temporal] 2025-fit top-10: {top10_25}")
    print(f"[temporal] year-over-year top-10 overlap: {len(overlap)}/10")
    print(f"           shared:    {overlap}")
    print(f"           2024 only: {sorted(set(top10_24) - set(top10_25))}")
    print(f"           2025 only: {sorted(set(top10_25) - set(top10_24))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
