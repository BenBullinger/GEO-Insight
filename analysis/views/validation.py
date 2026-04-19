"""Mode F — Validation against two independent external benchmarks.

CERF UFE (OCHA underfunded-emergencies allocations) and CARE Breaking the
Silence (under-reported humanitarian crises) provide human-curated ground
truth on two orthogonal dimensions of overlookedness. We compute
set-overlap metrics (precision / recall at K) against CERF UFE and a
Spearman rank correlation against CARE's 1–10 ranking.

The view ranks by the posterior median of the latent overlookedness θ
(Level 5, `theta_median`). Higher θ = more overlooked.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from validation import (
    agreement_table,
    load_care_bts,
    load_cerf_ufe,
    overlap_at_k,
    spearman_on_intersection,
)


def _completeness_filter(enriched: pd.DataFrame) -> pd.DataFrame:
    if "completeness" not in enriched.columns:
        return enriched
    mode = st.radio(
        "Completeness filter (applied to our ranking before validation)",
        [
            "All (≥ 1/2 = 3 of 6 observed — widest pool, partials included)",
            "High (≥ 2/3 = 4 of 6)",
            "Strict (completeness = 1.0 — all six attributes observed)",
        ],
        index=0,
        horizontal=True,
    )
    thr = {"All": 0.5, "High": 4 / 6, "Strict": 1.0}[mode.split(" ", 1)[0]]
    return enriched[enriched["completeness"] >= thr - 1e-9]


def render(enriched: pd.DataFrame, lens, registry) -> None:
    st.title("Validation — two independent benchmarks")
    st.caption(
        "Bayesian posterior median of overlookedness θ vs CERF UFE and CARE "
        "Breaking the Silence. Independent human curation on two orthogonal "
        "axes (underfunding, under-reporting). Covered in `proposal/proposal.pdf` §8."
    )

    if "theta_median" not in enriched.columns or enriched["theta_median"].isna().all():
        st.info(
            "No `theta_median` in the enriched frame. The Bayesian posterior is "
            "computed only for HRP-eligible countries (those with an observed "
            "per_pin_gap)."
        )
        return

    sub = _completeness_filter(enriched)
    # Convert posterior median into a rank (lower rank value = more overlooked)
    # so the existing overlap_at_k / agreement_table helpers can consume it.
    ranks = (-sub["theta_median"].dropna()).rank(method="average")
    st.markdown(f"**Our pool after filter: {len(ranks)} countries.**")

    # ════════════════════════════════════════════════════════════════════
    # CERF UFE
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### CERF Underfunded Emergencies")
    year_choice = st.selectbox(
        "CERF UFE cycle",
        options=["2025 (both windows combined)", "2025 (window 1, March)", "2025 (window 2, December)", "2024 (window 2)"],
        index=0,
    )
    cerf_all = load_cerf_ufe()
    if cerf_all.empty:
        st.warning(
            "No CERF UFE data on disk. Expected at "
            "`Data/Third-Party/Benchmarks/cerf_ufe.csv`."
        )
    else:
        if year_choice.startswith("2025 (both"):
            cerf_sel = cerf_all[cerf_all["year"] == 2025]
        elif year_choice == "2025 (window 1, March)":
            cerf_sel = cerf_all[(cerf_all["year"] == 2025) & (cerf_all["window"] == 1)]
        elif year_choice == "2025 (window 2, December)":
            cerf_sel = cerf_all[(cerf_all["year"] == 2025) & (cerf_all["window"] == 2)]
        else:
            cerf_sel = cerf_all[(cerf_all["year"] == 2024) & (cerf_all["window"] == 2)]
        cerf_set = set(cerf_sel["iso3"].astype(str).unique())
        st.caption(f"CERF UFE selection size: {len(cerf_set)} countries.")

        cols = st.columns(4)
        for i, k in enumerate([5, 10, 15, 20]):
            m = overlap_at_k(ranks, cerf_set, k)
            cols[i].metric(
                f"Overlap @ top-{k}",
                f"{m['precision_at_k']:.0%}",
                help=(
                    f"{len(m['intersection'])} of our top-{k} appear in the "
                    f"{len(cerf_set)}-country CERF UFE list. "
                    f"Recall of CERF UFE: {m['recall_at_k']:.0%}."
                ),
            )

        # Agreement table
        st.markdown("#### Agreement breakdown (top-10)")
        agree = agreement_table(
            ranks,
            cerf_sel.rename(columns={"country_name": "country_name"})[
                ["iso3", "country_name"]
            ].drop_duplicates(subset=["iso3"]),
            k=10,
        )
        st.caption(
            "**both** = our top-10 ∩ CERF UFE (we agree). "
            "**framework-only** = we rank high, CERF does not pick. "
            "**benchmark-only** = CERF picks, we rank outside top-10 (or don't score)."
        )
        st.dataframe(agree, use_container_width=True, height=360, hide_index=True)

    # ════════════════════════════════════════════════════════════════════
    # CARE Breaking the Silence
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### CARE Breaking the Silence")
    care = load_care_bts()
    if care.empty:
        st.warning(
            "No CARE BTS data on disk. Expected at "
            "`Data/Third-Party/Benchmarks/care_bts.csv`."
        )
    else:
        year_opts = sorted(care["year"].dropna().unique().astype(int).tolist(), reverse=True)
        care_year = st.selectbox("CARE BTS year", year_opts, index=0)
        care_sel = care[care["year"] == care_year]
        care_set = set(care_sel["iso3"].astype(str).unique())
        st.caption(f"CARE BTS list size: {len(care_set)} countries (ranked 1–10).")

        cols = st.columns(3)
        for i, k in enumerate([10, 15, 20]):
            m = overlap_at_k(ranks, care_set, k)
            cols[i].metric(
                f"Overlap @ top-{k}",
                f"{m['precision_at_k']:.0%}",
                help=(
                    f"{len(m['intersection'])} of our top-{k} appear in CARE's top-10. "
                    f"Recall of CARE: {m['recall_at_k']:.0%}."
                ),
            )

        # Spearman on intersection: CARE has explicit ranks 1–10
        bench_rank = care_sel.set_index("iso3")["rank"]
        rho, n = spearman_on_intersection(ranks, bench_rank)
        c1, c2 = st.columns(2)
        c1.metric(
            "Spearman ρ (overlap ordering)",
            f"{rho:+.2f}" if pd.notna(rho) else "n/a",
            help=(
                f"Rank correlation on the {n} ISO3 in both lists. "
                "Positive = we order the overlapping countries consistently with CARE."
            ),
        )
        c2.metric("Common countries", n)

        st.markdown("#### Agreement breakdown (top-10)")
        agree2 = agreement_table(
            ranks,
            care_sel[["iso3", "country_name"]].drop_duplicates(subset=["iso3"]),
            k=10,
        )
        st.dataframe(agree2, use_container_width=True, height=360, hide_index=True)
