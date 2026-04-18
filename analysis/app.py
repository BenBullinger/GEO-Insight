"""GEO-Insight — Unsupervised Analysis (lens-first).

Thin orchestrator. Reads sidebar → loads the enriched frame (Levels 1-3 for
Phase 1) → dispatches to the chosen view mode with (enriched, lens, registry).

Phase 1 wires the Funding Pressure lens end-to-end. Other lenses show a
"ships in Phase 2/3" message so the shape of the final UI is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `import ontology`, `import features`, `from views import ...` from the
# analysis/ folder regardless of CWD; also expose the shared _theme helper
# that lives next door in dashboard/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))

from ontology import Registry
import features
from views import atlas, pca as pca_view, cluster, profile, cross_lens, validation as validation_view
from _theme import apply_theme, COLORS  # noqa: E402

st.set_page_config(
    page_title="GEO-Insight — Semantic Analysis",
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

# Phase 3: all eight lenses wired end-to-end.
ACTIVE_LENSES = {
    "magnitude",
    "intensity",
    "severity_composition",
    "funding_pressure",
    "donor_fragility",
    "temporal_dynamics",
    "access_friction",
    "geo_insight_score",
}


# ─── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("GEO-Insight")
st.sidebar.caption("Unsupervised analysis · Cache Me if You Can · Phase 1")

lens_order = ["funding_pressure"] + [
    lid for lid in REGISTRY.lenses if lid != "funding_pressure"
]


def _label(lid: str) -> str:
    return REGISTRY.lenses[lid].name


lens_id = st.sidebar.selectbox("Lens", lens_order, format_func=_label)
lens = REGISTRY.lenses[lens_id]

mode_id = st.sidebar.radio(
    "Mode",
    ["atlas", "pca", "cluster", "profile", "cross_lens", "validation"],
    format_func=lambda m: REGISTRY.modes[m].name,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Companion: data-exploration dashboard on **:8501**.\n\n"
    "Spec: `analysis/spec.yaml`.\n\n"
    "Metric provenance: `proposal/metric_cards.md`.\n\n"
    "Methodology: `proposal/proposal.pdf`."
)


# ─── Route ─────────────────────────────────────────────────────────────────
enriched = get_enriched()

VIEWS = {
    "atlas": atlas.render,
    "pca": pca_view.render,
    "cluster": cluster.render,
    "profile": profile.render,
    "cross_lens": cross_lens.render,
    "validation": validation_view.render,
}

VIEWS[mode_id](enriched, lens, REGISTRY)
