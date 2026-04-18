"""Mode B — PCA scoped to lens properties.

Components are named by their dominant loadings, so PC1 within a lens reads
as something like "magnitude of underfunding" rather than an unnamed axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def render(enriched: pd.DataFrame, lens, registry) -> None:
    st.title(f"PCA — {lens.name}")
    st.caption(f"Lens: {lens.question}")

    cols = [
        c
        for c in lens.properties
        if c in enriched.columns and pd.api.types.is_numeric_dtype(enriched[c])
    ]
    if len(cols) < 4:
        st.info(
            f"Lens has only {len(cols)} numeric properties in the enriched frame. "
            "PCA is suppressed below 4 (see `analysis/spec.yaml` → ui.modes.pca)."
        )
        return

    X = enriched[cols].dropna()
    if len(X) < 5:
        st.warning("Fewer than 5 complete rows for PCA — widen eligibility.")
        return

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=min(Xs.shape[1], 6))
    Z = pca.fit_transform(Xs)

    # ── Scree + loadings ──
    var = pca.explained_variance_ratio_
    scree = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(len(var))],
            "Variance explained": var,
            "Cumulative": np.cumsum(var),
        }
    )
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### Scree")
        fig = px.bar(scree, x="Component", y="Variance explained")
        fig.add_scatter(
            x=scree["Component"],
            y=scree["Cumulative"],
            mode="lines+markers",
            name="cumulative",
            yaxis="y2",
        )
        fig.update_layout(
            yaxis={"tickformat": ".0%"},
            yaxis2={"overlaying": "y", "side": "right", "tickformat": ".0%"},
            height=340,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Loadings")
        loadings = pd.DataFrame(
            pca.components_.T,
            index=cols,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        )
        st.dataframe(loadings.round(3), use_container_width=True)
        st.caption(
            "Each column is a principal component; entries show how each lens property contributes. "
            "The dominant loading is the natural name for that component."
        )

    # ── Biplot ──
    st.markdown("#### PC1 × PC2 biplot")
    scores = (
        pd.DataFrame(Z[:, :2], columns=["PC1", "PC2"], index=X.index)
        .reset_index()
        .rename(columns={"index": "ISO3", X.index.name or "index": "ISO3"})
    )
    colour_by = st.selectbox(
        "Colour points by", ["(none)"] + cols, key=f"pca_color_{lens.id}"
    )
    fig = px.scatter(
        scores,
        x="PC1",
        y="PC2",
        text="ISO3",
        color=None
        if colour_by == "(none)"
        else enriched.loc[scores["ISO3"], colour_by].values,
        color_continuous_scale="Viridis",
        labels={"color": colour_by if colour_by != "(none)" else ""},
    )
    fig.update_traces(textposition="top center", textfont={"size": 9})

    # Loading arrows
    load2 = pca.components_[:2].T
    arrow_scale = max(abs(scores["PC1"]).max(), abs(scores["PC2"]).max()) * 0.9
    for i, name in enumerate(cols):
        x, y = load2[i, 0] * arrow_scale, load2[i, 1] * arrow_scale
        fig.add_annotation(
            x=x, y=y, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor="#E99C2D",
        )
        fig.add_annotation(
            x=x * 1.08, y=y * 1.08, text=name, showarrow=False,
            font=dict(size=10, color="#E99C2D"),
        )
    fig.update_layout(height=560, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
