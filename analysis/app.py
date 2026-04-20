"""Geo-Insight — Semantic Analysis.

Thin orchestrator. Reads sidebar (lens × mode) or a "Featured views" shortcut
bar at the top of the page, loads the enriched frame from disk cache if
fresh (else re-derives), and dispatches to the chosen view.

The eight lenses × six modes = 48-cell grid remains fully reachable via the
sidebar. The featured strip above it lets a first-time visitor reach the
load-bearing views in one click without exploring the grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `import ontology`, `import features`, `from views import ...` from the
# analysis/ folder regardless of CWD; expose the shared _theme helper that
# lives next door in dashboard/; and put the repo root on sys.path so absolute
# imports like `from analysis.bayesian import hierarchical` (used by
# composites.py and the Bayesian CLI scripts) resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ontology import Registry
import features
from views import atlas, pca as pca_view, cluster, profile, cross_lens, validation as validation_view
from _theme import apply_theme, back_to_landing, COLORS  # noqa: E402

st.set_page_config(
    page_title="Geo-Insight — Semantic Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ─── Registry + enriched frame (cached) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading ontology…")
def get_registry() -> Registry:
    return Registry.load()


@st.cache_data(show_spinner="Assembling enriched frame (L1–L5)…")
def get_enriched():
    # Prefer the on-disk parquet snapshot when fresh; fall back to a full
    # re-derivation otherwise. See `scripts/refresh_enriched.py` to materialise.
    cached = features.load_cached_enriched_frame()
    if cached is not None:
        return cached
    return features.build_enriched_frame()


REGISTRY = get_registry()


# ─── Default lens + mode (session_state so the featured-view buttons can
# change them, and the sidebar widgets pick up the change on next render) ──
st.session_state.setdefault("lens_id", "geo_insight_score")
st.session_state.setdefault("mode_id", "atlas")


# ─── Sidebar ───────────────────────────────────────────────────────────────
back_to_landing()
st.sidebar.title("Geo-Insight")
st.sidebar.caption("Semantic analysis · eight lenses × six modes")

lens_order = sorted(REGISTRY.lenses.keys(), key=lambda lid: (lid != "geo_insight_score", lid))


def _label(lid: str) -> str:
    return REGISTRY.lenses[lid].name


st.sidebar.selectbox(
    "Lens",
    lens_order,
    index=lens_order.index(st.session_state["lens_id"]),
    format_func=_label,
    key="lens_id",
)
st.sidebar.radio(
    "Mode",
    ["atlas", "pca", "cluster", "profile", "cross_lens", "validation"],
    index=["atlas", "pca", "cluster", "profile", "cross_lens", "validation"].index(st.session_state["mode_id"]),
    format_func=lambda m: REGISTRY.modes[m].name,
    key="mode_id",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data-sources audit: [localhost:8501](http://localhost:8501).\n\n"
    "Methodology: [localhost:7777/methodology/](http://localhost:7777/methodology/)."
)


# ─── Featured-views shortcut strip ─────────────────────────────────────────
FEATURED = [
    ("Top ten overlooked",          "Bayesian posterior ranking on the HRP-eligible pool",  "geo_insight_score", "atlas"),
    ("Validation",                  "vs CERF UFE and CARE Breaking the Silence",            "geo_insight_score", "validation"),
    ("Four-cell typology",          "Sectoral inequality × posterior CI width",             "geo_insight_score", "cross_lens"),
    ("Country profile",             "Drill into one crisis across the layers",              "geo_insight_score", "profile"),
    ("Funding pressure",            "Ranked by coverage-gap intensity",                     "funding_pressure",  "atlas"),
]


def _select_featured(lens_id: str, mode_id: str) -> None:
    st.session_state["lens_id"] = lens_id
    st.session_state["mode_id"] = mode_id


# ─── Calibration & validation snapshot (above Featured Views) ──────────────
_CAL = [
    ("ρ = 0.89",  "SVI medians vs NUTS gold standard",     "Hierarchical Bayesian fit validated by MCMC on every run."),
    ("≥ 0.91",    "posterior predictive coverage, all 6 attributes", "Marginals are calibrated (target 0.90)."),
    ("5 / 10",    "CERF UFE March 2025 · precision @ 10",   "Half the UN's underfunded-emergency picks in our independent top-10."),
    ("4 / 10",    "held-out 2024-fit → March 2025",         "2024-only fit predicts CERF's March-2025 picks without seeing 2025 data."),
]
st.markdown(
    f"<div style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; "
    f"color: {COLORS['accent']}; font-weight: 600; margin-bottom: 10px;'>"
    f"Calibration &amp; validation snapshot</div>",
    unsafe_allow_html=True,
)
_cal_cols = st.columns(len(_CAL))
for _col, (_val, _label, _hlp) in zip(_cal_cols, _CAL):
    _col.metric(_label, _val, help=_hlp)
st.markdown("")  # small breathing room

st.markdown(
    f"<div style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; "
    f"color: {COLORS['accent']}; font-weight: 600; margin-bottom: 10px;'>"
    f"Featured views</div>",
    unsafe_allow_html=True,
)

cols = st.columns(len(FEATURED))
for col, (title, sub, lens_id, mode_id) in zip(cols, FEATURED):
    selected = (st.session_state["lens_id"] == lens_id and st.session_state["mode_id"] == mode_id)
    prefix = "• " if selected else ""
    with col:
        st.button(
            f"{prefix}{title}",
            help=sub,
            on_click=_select_featured,
            args=(lens_id, mode_id),
            use_container_width=True,
            type=("primary" if selected else "secondary"),
        )
        st.markdown(
            f"<div style='font-size: 0.72rem; color: {COLORS['muted']}; "
            f"line-height: 1.4; margin-top: 4px; margin-bottom: 14px;'>{sub}</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")


# ─── Route ─────────────────────────────────────────────────────────────────
lens = REGISTRY.lenses[st.session_state["lens_id"]]
mode_id = st.session_state["mode_id"]
enriched = get_enriched()

VIEWS = {
    "atlas":       atlas.render,
    "pca":         pca_view.render,
    "cluster":     cluster.render,
    "profile":     profile.render,
    "cross_lens":  cross_lens.render,
    "validation":  validation_view.render,
}

VIEWS[mode_id](enriched, lens, REGISTRY)
