# GEO-Insight — Data Landscape Dashboard

Interactive Streamlit dashboard that surveys the five official datasets we downloaded. Purpose: orient the team before committing to a methodology — see what's there, see the gaps, preview the key signals (cluster-coverage Gini, donor HHI, cross-dataset intersection).

**This is exploratory, not the final tool.** It's a survey surface, meant for team discussion.

## Setup (one time)

```bash
cd dashboard
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
cd dashboard
.venv/bin/streamlit run app.py
# opens http://localhost:8501 in your default browser
```

First load takes ~10–15 seconds to cache the ~270 MB of CSVs; subsequent navigation is fast.

Stop with `Ctrl-C` in the terminal. Streamlit auto-reloads on file save, so edit `app.py` and see changes immediately.

## Sections

| Section | What's there |
|---|---|
| **Overview** | Top-line metrics, orienting questions |
| **Cross-dataset coverage** | Which countries appear in HNO ∩ HRP ∩ FTS (global + cluster) for a chosen year — the set of crises we can reason about with the full methodology |
| **Needs (HNO)** | Choropleth of PIN, top-20 countries, per-country cluster breakdown |
| **Funding (FTS)** | Coverage-ratio choropleth, lowest-coverage countries ≥ $100M need, multi-country time-series |
| **Sector equity preview** | Cluster-level coverage ratios for a selected country + **computed unweighted cluster Gini** (full methodology uses PIN-weighted; this is a signal preview) |
| **Donors (FTS)** | Top-20 global donors, per-country donor pie + **computed HHI** for single-destination transactions |
| **Pooled funds (CBPF)** | Top pooled funds, top CBPF donors |
| **Plans (HRP)** | Active plans by year, requirements, plan metadata table |

## Known shortcuts / caveats

- **HNO**: PIN is summed across clusters for country totals. The HNO file mixes country-level and admin1/2 granularities; we filter to country-level aggregates where possible and fall back to summing all rows otherwise.
- **Sector equity Gini shown here is unweighted** across clusters — the proposal uses a **PIN-weighted** Gini. The unweighted version is a fast proxy for the signal.
- **Donor HHI** is computed only on FTS incoming rows where `destLocations` is a single ISO3. Multi-country rows need a proper allocation rule (see proposal §4.3).
- **CBPF country mapping** in the overview page is crude (prefix of fund name); the real ISO3 ↔ fund mapping lives in the project-level data.

These are known and documented in the proposal. The dashboard's purpose is to surface the data, not compute the final scores.

## Files

```
dashboard/
├── app.py             single-file Streamlit app
├── requirements.txt   streamlit, pandas, plotly
├── .gitignore         excludes .venv/
└── README.md          this file
```

## Add a page / chart

Each section is a function in `app.py`. To add a new view:

1. Write a `def page_foo():` function using `st` + `px`.
2. Add it to the `PAGES` dict at the bottom and to the sidebar `radio` list.

Data loaders are already cached with `@st.cache_data` so adding a view that reuses the same CSV is free.
