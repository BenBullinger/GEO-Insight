"""Mode D — Per-crisis profile card scoped to the lens.

All lens properties with values, ranks, provenance, plus a rank-based radar.
Rank used rather than raw values so axes are comparable regardless of unit.

Also drills down into the multi-row L1/L2 properties — sector-level coverage
and donor breakdown — which cannot live as scalar columns in the country-
indexed enriched frame.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import features


def render(enriched: pd.DataFrame, lens, registry) -> None:
    st.title(f"Profile — {lens.name}")
    st.caption(f"Lens: {lens.question}")

    cols = [c for c in lens.properties if c in enriched.columns]
    if not cols:
        st.warning("No lens properties available.")
        return

    iso_choices = sorted(enriched.index.astype(str).tolist())
    default_iso = st.query_params.get("country", iso_choices[0])
    if default_iso not in iso_choices:
        default_iso = iso_choices[0]
    iso = st.selectbox("Country (ISO3)", iso_choices,
                       index=iso_choices.index(default_iso),
                       key=f"profile_iso_{lens.id}")
    row = enriched.loc[iso, cols]

    # Ranks: higher rank = more extreme (descending for most metrics)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(enriched[c])]
    ranks = enriched[numeric_cols].rank(ascending=False, na_option="bottom")
    n = len(enriched)

    # ── Bayesian posterior card (when applicable) ─────────────────────
    _render_posterior_card(enriched, iso)

    # Value table — includes a one-line description next to each property
    st.markdown(f"### {iso} · lens values")
    info_rows = []
    for c in cols:
        val = row[c]
        rank = int(ranks.loc[iso, c]) if c in numeric_cols and pd.notna(ranks.loc[iso, c]) else None
        prop = registry.properties.get(c)
        info_rows.append(
            {
                "property": c,
                "description": (prop.description if prop and prop.description else "—"),
                "value": (f"{val:.3g}" if pd.api.types.is_numeric_dtype(enriched[c]) and pd.notna(val) else str(val)),
                "rank": f"{rank}/{n}" if rank else "—",
                "level": prop.level if prop else "?",
                "unit": prop.unit if prop else "",
            }
        )
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, height=300, hide_index=True)

    # Radar on rank fractions (1 = most extreme, 0 = least extreme)
    if numeric_cols:
        rank_frac = 1 - (ranks.loc[iso, numeric_cols] / n)
        fig = go.Figure(
            go.Scatterpolar(
                r=rank_frac.tolist() + [rank_frac.iloc[0]],
                theta=numeric_cols + [numeric_cols[0]],
                fill="toself",
                name=iso,
            )
        )
        fig.update_layout(
            polar={"radialaxis": {"range": [0, 1], "tickformat": ".0%"}},
            height=520,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
            showlegend=False,
        )
        st.markdown("#### Lens rank radar (outer = most extreme within lens)")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Property provenance"):
        for c in cols:
            st.markdown(registry.tooltip(c))
            st.markdown("---")

    # ── Multi-row drill-downs (spec.yaml country_sector_year / country_donor_year) ──
    sector_df, donor_df = _load_breakdowns()

    sector_rows = sector_df[sector_df["iso3"] == iso]
    if not sector_rows.empty:
        st.markdown("### Sector breakdown")
        st.caption(
            "L1 `pin_by_sector`, `requirements_by_sector`, `funding_by_sector` · "
            "L2 `coverage_by_sector`. Derived from the HNO × FTS cluster join."
        )
        show = sector_rows[[
            "cluster",
            "pin_by_sector",
            "requirements_by_sector",
            "funding_by_sector",
            "coverage_by_sector",
        ]].copy()
        show = show.sort_values("coverage_by_sector", na_position="last").reset_index(drop=True)
        show["pin_by_sector"] = show["pin_by_sector"].map(_fmt_int)
        show["requirements_by_sector"] = show["requirements_by_sector"].map(_fmt_usd)
        show["funding_by_sector"] = show["funding_by_sector"].map(_fmt_usd)
        show["coverage_by_sector"] = show["coverage_by_sector"].map(_fmt_pct)
        show.columns = ["Sector", "PIN", "Requested", "Funded", "Coverage"]
        st.dataframe(show, use_container_width=True, hide_index=True, height=320)

    donor_rows = donor_df[donor_df["iso3"] == iso]
    if not donor_rows.empty:
        st.markdown("### Donor breakdown")
        st.caption(
            "L1 `funding_by_donor`. Single-destination FTS incoming flows only "
            "(rows whose destLocations is a single ISO3)."
        )
        top = donor_rows.nlargest(15, "funding_by_donor").copy()
        total = donor_rows["funding_by_donor"].sum()
        top["share"] = top["funding_by_donor"] / total if total else 0
        show = top[["donor", "funding_by_donor", "share"]].copy()
        show["funding_by_donor"] = show["funding_by_donor"].map(_fmt_usd)
        show["share"] = show["share"].map(_fmt_pct)
        show.columns = ["Donor", "USD contributed", "Share"]
        st.dataframe(show, use_container_width=True, hide_index=True, height=320)
        if len(donor_rows) > 15:
            st.caption(f"Showing top 15 of {len(donor_rows)} donors.")


# ─── Helpers ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_breakdowns() -> tuple[pd.DataFrame, pd.DataFrame]:
    return features.load_sector_breakdown(), features.load_donor_breakdown()


@st.cache_data(show_spinner=False)
def _load_benchmarks() -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (CERF UFE 2024 w2, CERF UFE 2025 w1, CERF UFE 2025 w2, CARE BTS 2024) ISO3 sets."""
    from validation import load_cerf_ufe, load_care_bts
    cerf_all = load_cerf_ufe()
    care_all = load_care_bts()
    def _cerf(yr: int, win: int) -> set[str]:
        if cerf_all.empty: return set()
        s = cerf_all[(cerf_all["year"] == yr) & (cerf_all["window"] == win)]
        return set(s["iso3"].astype(str).unique())
    def _care(yr: int) -> set[str]:
        if care_all.empty: return set()
        return set(care_all[care_all["year"] == yr]["iso3"].astype(str).unique())
    return _cerf(2024, 2), _cerf(2025, 1), _cerf(2025, 2), _care(2024)


def _render_posterior_card(enriched: pd.DataFrame, iso: str) -> None:
    """Lead a country profile with the Bayesian posterior summary.

    Shows theta_median ± 90 % CI, CI width, completeness, typology cell,
    and benchmark membership across the four CERF/CARE lists. For non-HRP-
    eligible countries (no theta_median), shows a short explanatory note.
    """
    if "theta_median" not in enriched.columns:
        return
    if iso not in enriched.index:
        return
    row = enriched.loc[iso]
    theta = row.get("theta_median")
    if pd.isna(theta):
        st.info(
            f"**{iso}** is not in the HRP-eligible pool — the Bayesian model "
            "does not produce a posterior for this country (per_pin_gap is "
            "not observed). See the *Candidate pool* page of the data dashboard."
        )
        return

    lo = row.get("theta_ci_lo")
    hi = row.get("theta_ci_hi")
    width = row.get("theta_ci_width")
    completeness = row.get("completeness")
    typology = row.get("typology_cell", "—")

    # Rank within the HRP-eligible pool
    fit = enriched[enriched["theta_median"].notna()]
    rank = int((-fit["theta_median"]).rank(method="average").loc[iso]) if iso in fit.index else None
    pool_n = len(fit)

    # Benchmark memberships
    b24w2, b25w1, b25w2, care24 = _load_benchmarks()
    bench = []
    for label, bset in [("CERF UFE 2024 w2", b24w2), ("CERF UFE 2025 w1", b25w1),
                        ("CERF UFE 2025 w2", b25w2), ("CARE BTS 2024", care24)]:
        if iso in bset:
            bench.append(label)

    st.markdown(f"### {iso} · Bayesian posterior")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "θ median",
        f"{theta:+.3f}",
        help="Posterior median of the latent overlookedness θ. Higher = more overlooked.",
    )
    ci_str = f"[{lo:+.3f}, {hi:+.3f}]" if (pd.notna(lo) and pd.notna(hi)) else "—"
    c2.metric("90 % CI", ci_str, help=f"Width = {width:.3f}" if pd.notna(width) else "")
    if rank is not None:
        c3.metric("Rank", f"{rank} / {pool_n}", help="Rank within the HRP-eligible pool by posterior median.")
    else:
        c3.metric("Rank", "—")
    c4.metric(
        "Completeness",
        f"{completeness:.2f}" if pd.notna(completeness) else "—",
        help="Fraction of the six attributes observed for this country.",
    )

    if typology and typology != "—":
        st.markdown(f"**Typology cell:** `{typology}`")
    if bench:
        st.markdown(
            "**External benchmarks (2024–25):** "
            + " · ".join(f"`{b}`" for b in bench)
        )
    else:
        st.caption("Not on any CERF UFE or CARE BTS list in 2024–25.")
    st.markdown("---")


def _fmt_int(x: object) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"


def _fmt_usd(x: object) -> str:
    if pd.isna(x):
        return "—"
    v = float(x)
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def _fmt_pct(x: object) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x) * 100:.1f}%"
