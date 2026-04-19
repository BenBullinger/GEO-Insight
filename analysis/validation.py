"""External-benchmark validation for the Bayesian posterior over overlookedness.

Loads the two curated benchmark files in Data/Third-Party/Benchmarks/ and
provides set-overlap and rank-correlation metrics against a rank Series
(typically derived from `theta_median`).

Used by analysis/views/validation.py (Mode F in the analysis app).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parent.parent / "Data" / "Third-Party" / "Benchmarks"


# ─── loaders ────────────────────────────────────────────────────────────────
def load_cerf_ufe(year: int | None = None) -> pd.DataFrame:
    """Return CERF UFE selections as a DataFrame with cols [iso3, year, window,
    country_name]. When `year` is None, returns all years."""
    path = BENCH / "cerf_ufe.csv"
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "year", "window", "country_name"])
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if year is not None:
        df = df[df["year"] == year]
    return df


def load_care_bts(year: int | None = None) -> pd.DataFrame:
    """CARE Breaking the Silence ranked top-10 per year."""
    path = BENCH / "care_bts.csv"
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "year", "rank", "country_name"])
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    if year is not None:
        df = df[df["year"] == year]
    return df


# ─── metrics ────────────────────────────────────────────────────────────────
def overlap_at_k(
    our_rank: pd.Series, benchmark_set: Iterable[str], k: int
) -> dict:
    """our_rank: Series indexed by ISO3, lower value = more overlooked (smaller is better)."""
    bench_set = set(benchmark_set)
    ranks = our_rank.dropna()
    if len(ranks) == 0:
        return {
            "k": k,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "intersection": [],
            "framework_only": [],
            "benchmark_only": sorted(bench_set),
            "our_top_k": [],
        }
    top_k_iso = set(ranks.nsmallest(k).index.astype(str))
    inter = sorted(top_k_iso & bench_set)
    fw_only = sorted(top_k_iso - bench_set)
    bm_only = sorted(bench_set - top_k_iso)
    return {
        "k": k,
        "precision_at_k": len(inter) / k if k else 0.0,
        "recall_at_k": len(inter) / len(bench_set) if bench_set else 0.0,
        "intersection": inter,
        "framework_only": fw_only,
        "benchmark_only": bm_only,
        "our_top_k": sorted(top_k_iso),
    }


def spearman_on_intersection(
    our_rank: pd.Series, bench_rank: pd.Series
) -> tuple[float, int]:
    """Spearman ρ computed on the ISO3 set that appears in both rankings.
    Returns (rho, n_common); rho is NaN if n_common < 3.
    """
    our = our_rank.dropna()
    bench = bench_rank.dropna()
    common = sorted(set(our.index.astype(str)) & set(bench.index.astype(str)))
    if len(common) < 3:
        return float("nan"), len(common)
    ours = our.loc[common].astype(float)
    theirs = bench.loc[common].astype(float)
    rho = ours.corr(theirs, method="spearman")
    return float(rho) if pd.notna(rho) else float("nan"), len(common)


def agreement_table(
    our_rank: pd.Series, benchmark: pd.DataFrame, k: int = 10
) -> pd.DataFrame:
    """Build a per-country agreement table: our rank, whether in benchmark, result class."""
    if "iso3" not in benchmark.columns:
        return pd.DataFrame(
            columns=["iso3", "country_name", "our_rank", "in_benchmark", "result"]
        )
    bench_map = benchmark.set_index("iso3")
    ranks = our_rank.dropna().sort_values()
    top_k = set(ranks.head(k).index.astype(str))
    bench_set = set(benchmark["iso3"].astype(str))

    rows = []
    # 1) intersection + framework-only (rows drawn from our top-k)
    for iso in ranks.index.astype(str):
        in_bench = iso in bench_set
        in_top = iso in top_k
        if in_top or in_bench:
            rows.append(
                {
                    "iso3": iso,
                    "country_name": (
                        bench_map.loc[iso, "country_name"] if in_bench else ""
                    ),
                    "our_rank": float(ranks.loc[iso]),
                    "our_top_k": in_top,
                    "in_benchmark": in_bench,
                    "result": (
                        "both" if (in_top and in_bench)
                        else "framework-only" if in_top
                        else "benchmark-only"
                    ),
                }
            )
    # 2) benchmark-only countries not in our ranking at all
    for iso in bench_set - set(ranks.index.astype(str)):
        rows.append(
            {
                "iso3": iso,
                "country_name": str(bench_map.loc[iso, "country_name"]),
                "our_rank": float("nan"),
                "our_top_k": False,
                "in_benchmark": True,
                "result": "benchmark-only (unscored)",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["result", "our_rank"], na_position="last")
        .reset_index(drop=True)
    )
