"""Mode C — Clustering scoped to the lens. k-means + hierarchical.

Centroids reported in ORIGINAL units (un-standardised) so that a reader can
say what the cluster means without squinting at z-scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def render(enriched: pd.DataFrame, lens, registry) -> None:
    st.title(f"Clustering — {lens.name}")
    st.caption(f"Lens: {lens.question}")

    cols = [
        c
        for c in lens.properties
        if c in enriched.columns and pd.api.types.is_numeric_dtype(enriched[c])
    ]
    if len(cols) < 2:
        st.info("Need at least 2 numeric lens properties to cluster.")
        return

    X = enriched[cols].dropna()
    if len(X) < 5:
        st.warning("Fewer than 5 complete rows available.")
        return

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # ── Silhouette curve ──
    st.markdown("#### k-means silhouette curve")
    ks = list(range(2, min(11, len(X))))
    sil = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
        sil.append(silhouette_score(Xs, km.labels_))
    sil_df = pd.DataFrame({"k": ks, "silhouette": sil})
    best_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    fig = px.line(sil_df, x="k", y="silhouette", markers=True)
    fig.update_layout(height=250, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Silhouette-optimal k = **{best_k}**.")

    k = st.slider("k", 2, max(10, best_k), best_k, key=f"km_k_{lens.id}")
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    labels = km.labels_

    # Centroids in ORIGINAL units
    means = X.mean().values
    stds = X.std().values
    centroids_orig = km.cluster_centers_ * stds + means
    centroid_df = pd.DataFrame(
        centroids_orig,
        columns=cols,
        index=[f"Cluster {i}" for i in range(k)],
    )
    centroid_df["n"] = pd.Series(labels).value_counts().sort_index().values

    st.markdown("#### Centroids (original units)")
    st.dataframe(centroid_df.round(3), use_container_width=True)

    # PC-space scatter coloured by cluster
    if Xs.shape[1] >= 2:
        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)
        scatter = pd.DataFrame(
            {
                "PC1": Z[:, 0],
                "PC2": Z[:, 1],
                "cluster": labels.astype(str),
                "ISO3": X.index.tolist(),
            }
        )
        fig = px.scatter(
            scatter, x="PC1", y="PC2", color="cluster", text="ISO3", hover_name="ISO3"
        )
        fig.update_traces(textposition="top center", textfont={"size": 9})
        fig.update_layout(
            height=500, legend_title_text="cluster",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Membership"):
        st.dataframe(
            pd.DataFrame({"ISO3": X.index, "cluster": labels}).sort_values(["cluster", "ISO3"]),
            use_container_width=True, height=400,
        )
