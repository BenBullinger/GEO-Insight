"""Geo-Insight — Data Landscape Dashboard.

Interactive, local. Run:
    cd dashboard
    .venv/bin/streamlit run app.py

Purpose: orient the team across the 5 official datasets before committing to a
methodology. Surfaces, per dataset: country coverage, year range, top-line
figures, and a preview of the intra-crisis cluster-equity signal that the
proposal depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, page_header, COLORS  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "Data"

st.set_page_config(
    page_title="Geo-Insight — Data Sources",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading HNO…")
def load_hno(year: int) -> pd.DataFrame:
    return pd.read_csv(
        DATA / "hno" / f"hpc_hno_{year}.csv", skiprows=[1], low_memory=False
    )


@st.cache_data(show_spinner="Loading FTS global…")
def load_fts_global() -> pd.DataFrame:
    df = pd.read_csv(DATA / "fts" / "fts_requirements_funding_global.csv", skiprows=[1])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading FTS cluster-level…")
def load_fts_cluster() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "fts" / "fts_requirements_funding_cluster_global.csv", skiprows=[1]
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading FTS incoming (donors)…")
def load_fts_incoming() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "fts" / "fts_incoming_funding_global.csv",
        skiprows=[1],
        low_memory=False,
    )
    df["budgetYear"] = pd.to_numeric(df["budgetYear"], errors="coerce")
    df["amountUSD"] = pd.to_numeric(df["amountUSD"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading population (CoD-PS admin0)…")
def load_population() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "cod-ps" / "cod_population_admin0.csv", skiprows=[1], low_memory=False
    )
    # Keep all-age + both-gender rows and aggregate to country-year
    df["Gender"] = df["Gender"].astype(str).str.lower()
    df["Age_range"] = df["Age_range"].astype(str).str.lower()
    tot = df[df["Gender"].isin({"all", "total", "t"})]
    tot = tot[tot["Age_range"].isin({"all", "total", "t"})]
    if tot.empty:
        # fall back: sum everything (may double-count; flagged in UI)
        tot = df
    return (
        tot.groupby(["ISO3", "Country", "Reference_year"], as_index=False)[
            "Population"
        ]
        .sum()
        .rename(columns={"Reference_year": "year"})
    )


@st.cache_data(show_spinner="Loading HRP plans…")
def load_hrp() -> pd.DataFrame:
    return pd.read_csv(
        DATA / "hrp" / "humanitarian-response-plans.csv",
        skiprows=[1],
        low_memory=False,
    )


@st.cache_data(show_spinner="Loading CBPF projects…")
def load_cbpf_projects() -> pd.DataFrame:
    return pd.read_csv(DATA / "cbpf" / "cbpf_project_summary.csv", low_memory=False)


@st.cache_data(show_spinner="Loading CBPF contributions…")
def load_cbpf_contributions() -> pd.DataFrame:
    return pd.read_csv(DATA / "cbpf" / "cbpf_contributions.csv", low_memory=False)


@st.cache_data(show_spinner="Loading INFORM Severity (monthly panel)…")
def load_inform() -> pd.DataFrame:
    path = DATA / "Third-Party" / "DRMKC-INFORM" / "inform_severity_long.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    df["category"] = pd.to_numeric(df["category"], errors="coerce")
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    return df.dropna(subset=["severity", "ISO3", "date"])


@st.cache_data(show_spinner="Loading INFORM sub-indicators…")
def load_inform_indicators() -> pd.DataFrame:
    path = DATA / "Third-Party" / "DRMKC-INFORM" / "inform_indicators_long.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("affected", "displaced", "injured", "fatalities",
                "pin_level_1", "pin_level_2", "pin_level_3", "pin_level_4", "pin_level_5",
                "access_limited", "access_restricted", "impediments_bureaucratic"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    return df.dropna(subset=["ISO3", "date"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def hno_country_pin(year: int) -> pd.DataFrame:
    """Sum PIN across clusters per country, filtering to non-null Country ISO3."""
    df = load_hno(year)
    df = df[df["Country ISO3"].notna()]
    # HNO rows mix per-cluster and per-admin granularities. Use Cluster column
    # cluster != ALL filter is unreliable; safest is to sum over unique
    # (country, cluster) after deduplication at cluster granularity.
    # Strategy: take rows where admin levels are null (country-level aggregate)
    country_level = df[
        df.get("Admin 1 PCode").isna() & df.get("Admin 2 PCode").isna()
    ]
    if country_level.empty:
        country_level = df
    agg = (
        country_level.groupby(["Country ISO3", "Cluster"], as_index=False)["In Need"]
        .sum()
        .groupby("Country ISO3", as_index=False)["In Need"]
        .sum()
    )
    return agg.rename(columns={"Country ISO3": "iso3", "In Need": "pin"})


def countries_per_dataset(year: int) -> dict[str, set[str]]:
    hno = hno_country_pin(year)
    fts = load_fts_global()
    fts_cl = load_fts_cluster()
    hrp = load_hrp()
    cbpf_p = load_cbpf_projects()

    hrp_years = hrp["years"].fillna("").astype(str)
    hrp_current = hrp[hrp_years.str.contains(str(year))]
    hrp_iso = set()
    for s in hrp_current["locations"].dropna():
        for code in str(s).split("|"):
            code = code.strip()
            if len(code) == 3 and code.isalpha():
                hrp_iso.add(code.upper())

    return {
        "HNO": set(hno["iso3"].dropna().astype(str).str.upper()),
        "FTS (country)": set(
            fts[fts["year"] == year]["countryCode"]
            .dropna()
            .astype(str)
            .str.upper()
        ),
        "FTS (cluster)": set(
            fts_cl[fts_cl["year"] == year]["countryCode"]
            .dropna()
            .astype(str)
            .str.upper()
        ),
        "HRP": hrp_iso,
        "CBPF": set(
            cbpf_p[cbpf_p["AllocationYear"] == year]
            .get("PooledFundName", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str[:3]
            .str.upper()
        ),  # crude — real ISO3 mapping via fund name prefix is imperfect
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

def page_overview() -> None:
    st.title("Data sources")
    st.caption(
        "Six upstream datasets — HNO, HRP, FTS, CoD-PS, CBPF, INFORM — "
        "audited for shape, country coverage, and top-line figures. This is "
        "the provenance layer underneath the semantic-analysis surface on :8502."
    )

    hno25 = load_hno(2025)
    fts = load_fts_global()
    fts_cl = load_fts_cluster()
    hrp = load_hrp()
    cbpf_p = load_cbpf_projects()
    incoming = load_fts_incoming()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HNO 2025", f"{len(hno25):,} rows", f"{hno25['Country ISO3'].nunique()} countries")
    c2.metric("FTS country × year", f"{len(fts):,}", f"years {int(fts['year'].min())}–{int(fts['year'].max())}")
    c3.metric("FTS cluster × year", f"{len(fts_cl):,}", f"{fts_cl['countryCode'].nunique()} countries")
    c4.metric("HRP plans", f"{len(hrp):,}", "1999–2026")
    c5.metric("CBPF projects", f"{len(cbpf_p):,}", f"{cbpf_p['AllocationYear'].nunique()} years")

    st.markdown("### Top-line 2025 figures")
    country_pin = hno_country_pin(2025)
    total_pin = country_pin["pin"].sum()
    fts25 = fts[fts["year"] == 2025]
    req = fts25["requirements"].sum()
    fund = fts25["funding"].sum()
    cov = 100 * fund / req if req else 0
    inc25 = incoming[incoming["budgetYear"] == 2025]["amountUSD"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total PIN (HNO 2025)", f"{total_pin/1e6:,.1f} M people")
    m2.metric("Global requirements 2025", f"${req/1e9:,.1f} B")
    m3.metric("Global funding received 2025", f"${fund/1e9:,.1f} B")
    m4.metric("Global coverage 2025", f"{cov:.1f} %", help="Σ funding ÷ Σ requirements")

    st.markdown("### What each tab answers")
    st.markdown(
        """
        - **Cross-dataset coverage** — which countries appear in HNO, HRP, FTS, CBPF simultaneously?
        - **Needs (HNO)** — where is PIN concentrated, at what admin level, across which sectors?
        - **Funding (FTS)** — country-year requirements and receipts; coverage distribution.
        - **Severity (INFORM)** — monthly severity panels; sub-indicator availability.
        - **Pooled funds (CBPF)** — project-level allocations and donor contributions.
        - **Plans (HRP)** — plan metadata, year range, scope.

        Analytic views (sector-coverage inequality, donor concentration, gap scores,
        four-cell typology, external-benchmark validation) live on the
        **semantic-analysis** surface at http://localhost:8502.
        """
    )


def page_coverage() -> None:
    st.title("Cross-dataset coverage")
    st.caption(
        "Which countries appear in which datasets. Determines the set of crises "
        "we can reason about with the full methodology."
    )

    year = st.slider("Plan year", 2018, 2026, value=2025, step=1)
    sets = countries_per_dataset(year)

    cols = st.columns(len(sets))
    for col, (name, s) in zip(cols, sets.items()):
        col.metric(name, f"{len(s)} countries")

    # Countries present in ALL core datasets (excluding CBPF which is partial)
    core = ["HNO", "FTS (country)", "FTS (cluster)", "HRP"]
    core_inter = set.intersection(*(sets[k] for k in core)) if all(sets[k] for k in core) else set()
    st.markdown(
        f"**Full-stack countries** (present in HNO + FTS country + FTS cluster + HRP for {year}): "
        f"**{len(core_inter)}**"
    )

    # Membership matrix
    all_iso = set().union(*sets.values())
    rows = []
    for iso in sorted(all_iso):
        rows.append({name: iso in s for name, s in sets.items()} | {"ISO3": iso})
    mem = pd.DataFrame(rows).set_index("ISO3")
    mem["count"] = mem.sum(axis=1)
    st.markdown("#### Membership matrix")
    st.dataframe(
        mem.sort_values("count", ascending=False).reset_index(),
        use_container_width=True,
        height=420,
    )

    st.markdown("#### Coverage choropleth (full-stack countries highlighted)")
    choro_df = pd.DataFrame({"iso3": sorted(all_iso)})
    choro_df["in_all_core"] = choro_df["iso3"].isin(core_inter)
    fig = px.choropleth(
        choro_df,
        locations="iso3",
        locationmode="ISO-3",
        color="in_all_core",
        color_discrete_map={True: COLORS["accent"], False: COLORS["faint"]},
        title=f"Countries present in HNO + FTS (country & cluster) + HRP for {year}",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0}, height=500)
    st.plotly_chart(fig, use_container_width=True)


def page_hno() -> None:
    st.title("Needs — HNO")
    st.caption("People in Need by country × sector, from OCHA HPC HNO.")

    year = st.selectbox("Year", [2025, 2024], index=0)
    df = load_hno(year)

    country_pin = hno_country_pin(year)
    st.metric(f"Total PIN {year}", f"{country_pin['pin'].sum()/1e6:,.1f} M")

    fig = px.choropleth(
        country_pin,
        locations="iso3",
        locationmode="ISO-3",
        color="pin",
        color_continuous_scale="Reds",
        title=f"PIN by country, {year}",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0}, height=460)
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("#### Top-20 countries by PIN")
        top = country_pin.sort_values("pin", ascending=False).head(20)
        st.plotly_chart(
            px.bar(top, x="pin", y="iso3", orientation="h").update_layout(
                yaxis={"categoryorder": "total ascending"}, height=500, margin={"l":0,"r":0,"t":10,"b":0}
            ),
            use_container_width=True,
        )
    with colB:
        st.markdown("#### PIN by cluster for a selected country")
        iso_choices = sorted(df["Country ISO3"].dropna().unique())
        iso_pick = st.selectbox("Country", iso_choices, key="hno_country")
        sub = df[df["Country ISO3"] == iso_pick]
        sub = sub[sub.get("Admin 1 PCode").isna() & sub.get("Admin 2 PCode").isna()]
        by_cluster = (
            sub.groupby("Cluster", as_index=False)["In Need"].sum().sort_values("In Need", ascending=False)
        )
        st.plotly_chart(
            px.bar(by_cluster, x="In Need", y="Cluster", orientation="h").update_layout(
                yaxis={"categoryorder": "total ascending"}, height=500, margin={"l":0,"r":0,"t":10,"b":0}
            ),
            use_container_width=True,
        )


def page_fts() -> None:
    st.title("Funding — FTS")
    st.caption("Requirements and funding by country × year.")

    fts = load_fts_global()
    year = st.slider("Year", 2018, 2026, value=2025, step=1)
    sub = fts[fts["year"] == year].copy()
    sub["coverage"] = (sub["funding"] / sub["requirements"]).clip(upper=1.5)

    c1, c2, c3 = st.columns(3)
    c1.metric("Countries reporting", sub["countryCode"].nunique())
    c2.metric("Total requirements", f"${sub['requirements'].sum()/1e9:,.1f}B")
    c3.metric("Total funding", f"${sub['funding'].sum()/1e9:,.1f}B")

    st.markdown("#### Coverage ratio by country")
    fig = px.choropleth(
        sub,
        locations="countryCode",
        locationmode="ISO-3",
        color="coverage",
        color_continuous_scale="RdYlGn",
        range_color=(0, 1),
        title=f"FTS coverage ratio, {year}",
        hover_data=["requirements", "funding"],
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0}, height=460)
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Lowest coverage (needs ≥ $100M)")
        big = sub[sub["requirements"] >= 100_000_000].sort_values("coverage").head(20)
        st.plotly_chart(
            px.bar(big, x="coverage", y="countryCode", orientation="h",
                   color="requirements", color_continuous_scale="Reds").update_layout(
                yaxis={"categoryorder": "total descending"}, height=500, margin={"l":0,"r":0,"t":10,"b":0}
            ),
            use_container_width=True,
        )
    with colB:
        st.markdown("#### Coverage trend — select countries")
        iso_choices = sorted(fts["countryCode"].dropna().unique())
        picks = st.multiselect("Countries", iso_choices, default=["SDN", "YEM", "SOM", "COD", "UKR"])
        if picks:
            trend = fts[fts["countryCode"].isin(picks)].copy()
            trend["coverage"] = (trend["funding"] / trend["requirements"]).clip(upper=1.5)
            st.plotly_chart(
                px.line(trend, x="year", y="coverage", color="countryCode", markers=True)
                .update_layout(yaxis_tickformat=".0%", height=500, margin={"l":0,"r":0,"t":10,"b":0}),
                use_container_width=True,
            )


# Analytic drill-downs (sector equity, donor concentration) moved to the
# semantic-analysis surface on :8502 — avoiding duplication. This app is now
# strictly a "data sources" audit of the five upstream datasets.


def page_cbpf() -> None:
    st.title("Pooled funds — CBPF")
    st.caption("Country-Based Pooled Funds projects and contributions.")

    projects = load_cbpf_projects()
    contribs = load_cbpf_contributions()

    c1, c2, c3 = st.columns(3)
    c1.metric("Projects", f"{len(projects):,}")
    c2.metric("Pooled funds", projects["PooledFundName"].nunique())
    c3.metric("Total project budget", f"${projects['Budget'].sum()/1e9:,.1f}B")

    year = st.slider("Allocation year", int(projects["AllocationYear"].min()),
                     int(projects["AllocationYear"].max()), value=2024, step=1, key="cbpf_year")
    sub = projects[projects["AllocationYear"] == year]

    st.markdown(f"#### Top pooled funds (allocation budget, {year})")
    by_fund = (
        sub.groupby("PooledFundName", as_index=False)["Budget"].sum().sort_values("Budget", ascending=False)
    )
    st.plotly_chart(
        px.bar(by_fund, x="Budget", y="PooledFundName", orientation="h").update_layout(
            yaxis={"categoryorder": "total ascending"}, height=420, margin={"l":0,"r":0,"t":10,"b":0}
        ),
        use_container_width=True,
    )

    st.markdown(f"#### Top donors to CBPFs ({contribs['FiscalYear'].max():g})")
    top_cfy = contribs[contribs["FiscalYear"] == contribs["FiscalYear"].max()]
    by_donor = (
        top_cfy.groupby("DonorName", as_index=False)["PaidAmt"].sum().sort_values("PaidAmt", ascending=False).head(20)
    )
    st.plotly_chart(
        px.bar(by_donor, x="PaidAmt", y="DonorName", orientation="h").update_layout(
            yaxis={"categoryorder": "total ascending"}, height=500, margin={"l":0,"r":0,"t":10,"b":0}
        ),
        use_container_width=True,
    )


def page_inform() -> None:
    st.title("Severity — INFORM (monthly panel)")
    st.caption(
        "EU JRC DRMKC Index for Risk Management. Per-crisis severity (1–5 ordinal "
        "category + 1–10 continuous index) assessed every month. Source for "
        "attribute $a_4$ and the panel-data backbone for the temporal task."
    )

    df = load_inform()
    if df.empty:
        st.warning(
            "`Data/Third-Party/DRMKC-INFORM/inform_severity_long.csv` is missing. "
            "Run `python3 Data/Third-Party/DRMKC-INFORM/download.py` then "
            "`dashboard/.venv/bin/python Data/Third-Party/DRMKC-INFORM/consolidate.py` "
            "— or just re-run `./run.sh`."
        )
        return

    st.warning(
        "**Scale discontinuity — February 2026.** INFORM rescaled the continuous "
        "index from 1–5 to 0–10 in its Feb 2026 release. Pre-/post-Feb-2026 *index* "
        "values are **not directly comparable**. The 1–5 **category** is stable "
        "across the break and is this page's default time-series. See the "
        "[DRMKC page](https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Severity/) "
        "and `proposal/metric_cards.md` for the formal definition."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Countries", df["ISO3"].nunique())
    c3.metric("Crises", df["CRISIS ID"].nunique() if "CRISIS ID" in df.columns else "—")
    c4.metric("Months", f"{df['snapshot'].min()} → {df['snapshot'].max()}")

    # Snapshot choropleth — always on category (stable across rescaling)
    snapshots = sorted(df["snapshot"].unique())
    snap = st.select_slider("Snapshot", options=snapshots, value=snapshots[-1])
    snap_df = df[df["snapshot"] == snap]

    st.markdown(f"#### Severity category — {snap}")
    fig = px.choropleth(
        snap_df,
        locations="ISO3",
        locationmode="ISO-3",
        color="category",
        color_continuous_scale="OrRd",
        range_color=(1, 5),
        hover_name="COUNTRY",
        hover_data={"severity": ":.1f", "category": True, "ISO3": False},
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=440)
    st.plotly_chart(fig, use_container_width=True)

    # Time-series — category by default, index with explicit warning
    st.markdown("#### Trajectories")
    metric_choice = st.radio(
        "Time-series metric",
        ["Category (1–5, stable)", "Index (1–10, NOT comparable across 2026-02)"],
        horizontal=True,
    )
    iso_choices = sorted(df["ISO3"].unique())
    defaults = [c for c in ("SDN", "YEM", "SOM", "COD", "UKR", "AFG") if c in iso_choices]
    picks = st.multiselect("Countries", iso_choices, default=defaults, key="sev_picks")
    if picks:
        sub = df[df["ISO3"].isin(picks)].sort_values("date")
        y = "category" if metric_choice.startswith("Category") else "severity"
        y_range = [1, 5] if y == "category" else [1, 10]
        y_label = ("INFORM Severity category (1–5)" if y == "category"
                   else "INFORM Severity index (1–10, rescaled Feb 2026)")
        fig2 = px.line(sub, x="date", y=y, color="ISO3", markers=True,
                       hover_data=["COUNTRY", "snapshot"])
        fig2.update_layout(
            yaxis={"range": y_range, "title": y_label},
            margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=420,
        )
        # Mark the Feb 2026 methodology break on both plots for context
        fig2.add_vline(x="2026-02-01", line_dash="dash", line_color=COLORS["subtle"])
        fig2.add_annotation(x="2026-02-01", y=y_range[1], yref="y",
                            text="methodology rescaling", showarrow=False,
                            font=dict(size=10, color=COLORS["muted"]), yshift=-6)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Sub-indicators — granular, auditable alternatives to the composite")
    st.caption(
        "Extracted from the `Crisis Indicator Data` sheet of each monthly xlsx. "
        "Use these primitives to avoid the composite-index failure modes."
    )
    ind = load_inform_indicators()
    if ind.empty:
        st.info("Sub-indicator CSV not present. Run `consolidate_indicators.py` to generate.")
        return

    indicator_choice = st.selectbox(
        "Indicator",
        [
            "pin_level_5 (PIN in level 5 — catastrophic conditions)",
            "pin_level_4 (PIN in level 4 — severe)",
            "displaced",
            "access_restricted",
            "access_limited",
            "impediments_bureaucratic",
            "fatalities",
            "affected",
        ],
    )
    col = indicator_choice.split(" ", 1)[0]
    picks2 = st.multiselect(
        "Countries", iso_choices, default=defaults, key="ind_picks"
    )
    if picks2:
        sub = ind[ind["ISO3"].isin(picks2)].sort_values("date")
        fig4 = px.line(sub, x="date", y=col, color="ISO3", markers=True,
                       hover_data=["COUNTRY", "snapshot"])
        fig4.update_layout(
            yaxis_title=col, margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=420,
        )
        # Same methodology-rescaling annotation — indicators themselves don't jump,
        # but consistent markup helps when comparing across views.
        fig4.add_vline(x="2026-02-01", line_dash="dot", line_color=COLORS["subtle"])
        st.plotly_chart(fig4, use_container_width=True)

    # PIN-composition view for a single country
    st.markdown("#### PIN composition by humanitarian-conditions level — single country")
    ccol1, ccol2 = st.columns([1, 3])
    iso_pick = ccol1.selectbox("Country", iso_choices, index=iso_choices.index("SDN") if "SDN" in iso_choices else 0)
    levels = ["pin_level_1", "pin_level_2", "pin_level_3", "pin_level_4", "pin_level_5"]
    sub_country = ind[ind["ISO3"] == iso_pick].sort_values("date")[["date"] + levels]
    sub_country = sub_country.melt(id_vars="date", var_name="level", value_name="pin")
    sub_country["level"] = sub_country["level"].str.replace("pin_level_", "Phase ", regex=False)
    fig5 = px.area(
        sub_country, x="date", y="pin", color="level",
        color_discrete_map={
            "Phase 1": "#FEF7E8", "Phase 2": "#FCDD8C",
            "Phase 3": "#F4A462", "Phase 4": "#E76F51", "Phase 5": "#96281B",
        },
    )
    fig5.update_layout(
        yaxis_title="People (stacked)", margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=360,
    )
    ccol2.plotly_chart(fig5, use_container_width=True)

    with st.expander("Raw snapshot data (category + index + sub-indicator joins)"):
        show = snap_df.merge(
            ind[ind["snapshot"] == snap][["CRISIS ID"] + levels + ["displaced", "access_restricted"]],
            on="CRISIS ID", how="left",
        )
        st.dataframe(
            show.sort_values("category", ascending=False)[
                ["ISO3", "COUNTRY", "CRISIS ID", "category", "severity",
                 "pin_level_4", "pin_level_5", "displaced", "access_restricted"]
            ],
            use_container_width=True, height=420,
        )


def page_hrp() -> None:
    st.title("Response plans — HRP")
    st.caption("OCHA HPC response plans (flash appeals, HRPs, refugee plans).")

    hrp = load_hrp()
    st.metric("Total plans on record", f"{len(hrp):,}")

    year = st.slider("Plan year (contains)", 2018, 2026, value=2025, step=1, key="hrp_year")
    sub = hrp[hrp["years"].fillna("").astype(str).str.contains(str(year))]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Plans active in {year}", len(sub))
    m2.metric("Original requirements", f"${sub['origRequirements'].sum()/1e9:,.1f}B")
    m3.metric("Revised requirements", f"${sub['revisedRequirements'].sum()/1e9:,.1f}B")

    st.dataframe(
        sub[["code", "planVersion", "categories", "locations", "years",
             "origRequirements", "revisedRequirements", "startDate", "endDate"]]
          .sort_values("revisedRequirements", ascending=False),
        use_container_width=True,
        height=500,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Geo-Insight")
st.sidebar.caption("Data sources · what's in our upstream datasets")
section = st.sidebar.radio(
    "Dataset",
    [
        "Overview",
        "Cross-dataset coverage",
        "Needs (HNO)",
        "Funding (FTS)",
        "Severity (INFORM)",
        "Pooled funds (CBPF)",
        "Plans (HRP)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Analytic views — sector equity, donor concentration, gap scores — live on the "
    "**semantic-analysis** surface at http://localhost:8502. This app is the "
    "provenance layer underneath it.\n\n"
    "Raw data: `Data/` (reproduce with `python3 Data/download.py`).\n\n"
    "Methodology: `proposal/proposal.pdf`."
)

PAGES = {
    "Overview": page_overview,
    "Cross-dataset coverage": page_coverage,
    "Needs (HNO)": page_hno,
    "Funding (FTS)": page_fts,
    "Severity (INFORM)": page_inform,
    "Pooled funds (CBPF)": page_cbpf,
    "Plans (HRP)": page_hrp,
}
PAGES[section]()
