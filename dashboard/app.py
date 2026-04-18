"""GEO-Insight — Data Landscape Dashboard.

Interactive, local. Run:
    cd dashboard
    .venv/bin/streamlit run app.py

Purpose: orient the team across the 5 official datasets before committing to a
methodology. Surfaces, per dataset: country coverage, year range, top-line
figures, and a preview of the intra-crisis cluster-equity signal that the
proposal depends on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "Data"

st.set_page_config(
    page_title="GEO-Insight — Data Landscape",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    st.title("Data landscape")
    st.caption(
        "Five official sources from `task/challenge.md`, downloaded into `Data/`. "
        "This view orients us before committing to a methodology."
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

    st.markdown("### Orienting questions this dashboard helps answer")
    st.markdown(
        """
        - **Cross-dataset coverage** — which countries appear in HNO, HRP, FTS, CBPF simultaneously?
        - **Needs** — where is PIN concentrated, and across which sectors?
        - **Funding** — where is the coverage ratio lowest?
        - **Sector equity** — for a given country, is the intra-crisis coverage spread tight or lopsided? (preview of the Gini term in §5.4 of the proposal)
        - **Donors** — how concentrated is the donor set for each crisis? (HHI preview)
        - **Pooled funds / plans** — what are the CBPF allocations and HRP plan targets look like on the ground?
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
        color_discrete_map={True: "#0072BC", False: "#D8DEE4"},
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
                   color="requirements", color_continuous_scale="Blues").update_layout(
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


def page_equity_preview() -> None:
    st.title("Sector-equity preview")
    st.caption(
        "For a given country/year, cluster-level coverage ratios. If these spread widely, "
        "the intra-crisis Gini (proposal §5.4) flags the crisis as sector-starved."
    )

    fts_cl = load_fts_cluster()
    year = st.slider("Year", 2018, 2026, value=2025, step=1, key="eq_year")

    sub = fts_cl[fts_cl["year"] == year].copy()
    iso_choices = sorted(sub["countryCode"].dropna().unique())
    iso = st.selectbox("Country", iso_choices, key="eq_iso")

    crisis = sub[sub["countryCode"] == iso].copy()
    crisis["coverage"] = (crisis["funding"] / crisis["requirements"]).clip(upper=1.5)
    crisis = crisis.sort_values("coverage")

    # Simple unweighted Gini on cluster coverage ratios (PIN-weighted version in pipeline)
    cov = crisis["coverage"].dropna().values
    if len(cov) >= 2 and cov.sum() > 0:
        cov_sorted = sorted(cov)
        n = len(cov_sorted)
        cum = sum((i + 1) * x for i, x in enumerate(cov_sorted))
        gini = (2 * cum) / (n * sum(cov_sorted)) - (n + 1) / n
    else:
        gini = float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters", len(crisis))
    m2.metric("Total requirements", f"${crisis['requirements'].sum()/1e6:,.0f}M")
    m3.metric("Aggregate coverage", f"{(crisis['funding'].sum()/max(crisis['requirements'].sum(),1))*100:.1f}%")
    m4.metric("Unweighted cluster Gini", f"{gini:.3f}" if pd.notna(gini) else "n/a",
              help="Illustrative — full methodology uses PIN-weighted Gini (proposal Eq. 2).")

    st.plotly_chart(
        px.bar(
            crisis,
            x="coverage",
            y="cluster",
            orientation="h",
            color="coverage",
            color_continuous_scale="RdYlGn",
            range_color=(0, 1),
            hover_data=["requirements", "funding"],
            title=f"Cluster coverage ratios — {iso} {year}",
        ).update_layout(yaxis={"categoryorder": "total ascending"}, height=520),
        use_container_width=True,
    )
    st.dataframe(crisis[["cluster", "requirements", "funding", "coverage"]], use_container_width=True)


def page_donors() -> None:
    st.title("Donors — FTS incoming")
    st.caption("Donor flows from `fts_incoming_funding_global.csv`. Preview of HHI computation.")

    inc = load_fts_incoming()
    year = st.slider("Budget year", 2018, 2026, value=2025, step=1, key="donor_year")
    sub = inc[inc["budgetYear"] == year].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", len(sub))
    c2.metric("Donors (srcOrganization)", sub["srcOrganization"].nunique())
    c3.metric("Total received", f"${sub['amountUSD'].sum()/1e9:,.1f}B")

    st.markdown("#### Top-20 donors globally")
    top_donors = (
        sub.groupby("srcOrganization", as_index=False)["amountUSD"]
        .sum()
        .sort_values("amountUSD", ascending=False)
        .head(20)
    )
    st.plotly_chart(
        px.bar(top_donors, x="amountUSD", y="srcOrganization", orientation="h").update_layout(
            yaxis={"categoryorder": "total ascending"}, height=500, margin={"l":0,"r":0,"t":10,"b":0}
        ),
        use_container_width=True,
    )

    st.markdown("#### Donor concentration for a selected country")
    # crude: pick rows where destLocations exactly matches an ISO3
    sub["destLocations"] = sub["destLocations"].fillna("").astype(str)
    single_country = sub[sub["destLocations"].str.len() == 3]
    iso_choices = sorted(single_country["destLocations"].dropna().unique())
    if iso_choices:
        iso = st.selectbox("Country (single-destination rows only)", iso_choices, key="hhi_iso")
        crisis = single_country[single_country["destLocations"] == iso]
        by_donor = (
            crisis.groupby("srcOrganization", as_index=False)["amountUSD"]
            .sum()
            .sort_values("amountUSD", ascending=False)
        )
        total = by_donor["amountUSD"].sum()
        if total > 0:
            shares = by_donor["amountUSD"] / total
            hhi = float((shares ** 2).sum())
        else:
            hhi = float("nan")
        m1, m2, m3 = st.columns(3)
        m1.metric("Donors", len(by_donor))
        m2.metric("Total to country", f"${total/1e6:,.0f}M")
        m3.metric("HHI", f"{hhi:.3f}" if pd.notna(hhi) else "n/a",
                  help="0 = perfectly diversified, 1 = single donor. Multi-country transactions excluded here.")
        st.plotly_chart(
            px.pie(by_donor.head(15), values="amountUSD", names="srcOrganization",
                   title=f"Top-15 donor shares, {iso} {year}"),
            use_container_width=True,
        )


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

st.sidebar.title("GEO-Insight")
st.sidebar.caption("Data landscape · Datathon 2026")
section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Cross-dataset coverage",
        "Needs (HNO)",
        "Funding (FTS)",
        "Sector equity preview",
        "Donors (FTS)",
        "Pooled funds (CBPF)",
        "Plans (HRP)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Raw data in `Data/` (reproduce with `python3 Data/download.py`).\n\n"
    "Methodology: `proposal/proposal.pdf`."
)

PAGES = {
    "Overview": page_overview,
    "Cross-dataset coverage": page_coverage,
    "Needs (HNO)": page_hno,
    "Funding (FTS)": page_fts,
    "Sector equity preview": page_equity_preview,
    "Donors (FTS)": page_donors,
    "Pooled funds (CBPF)": page_cbpf,
    "Plans (HRP)": page_hrp,
}
PAGES[section]()
