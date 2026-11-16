"""Unsupervised mode: find natural groups when there is no column to predict.

Tries KMeans, Ward hierarchical clustering and a Gaussian mixture for each k, keeps
the combination with the best silhouette score, and describes every cluster in the
original units so a person can name it."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import key_series, parse_numeric_series
from .profiler import DatasetProfile

MAX_SILHOUETTE_ROWS = 4000


@dataclass
class ClusterResult:
    inputs: list[str]
    algorithm: str
    k: int
    silhouette: float
    search: pd.DataFrame                 # algorithm, k, silhouette
    labels: pd.Series                    # cluster id per row, aligned to the input index
    sizes: pd.Series
    profile: pd.DataFrame                # one row per cluster, original units
    pca: pd.DataFrame                    # pc1, pc2, cluster
    notes: list[str] = field(default_factory=list)


def default_cluster_inputs(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    """Everything except identifiers, constants, free text, dates and near-empty columns."""
    keep = []
    for c in df.columns:
        cp = profile.columns.get(c)
        if cp is None or cp.is_constant or cp.is_likely_identifier or cp.is_free_text \
                or cp.is_datetime or cp.missing_ratio > 0.5:
            continue
        if cp.is_high_cardinality:
            continue
        keep.append(c)
    return keep


def _design(df: pd.DataFrame, cols: list[str], profile: DatasetProfile):
    num = [c for c in cols if profile.columns[c].is_numeric or profile.columns[c].is_boolean]
    cat = [c for c in cols if c not in num]
    X = df[cols].copy()
    for c in num:
        X[c] = parse_numeric_series(X[c]) if not pd.api.types.is_numeric_dtype(X[c]) else X[c].astype(float)
    for c in cat:
        X[c] = key_series(X[c])
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    return X, pre.fit_transform(X), num, cat


def _fit(algo: str, Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    if algo == "KMeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    if algo == "Hierarchical (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=seed).fit(Z).predict(Z)


def run_clustering(df: pd.DataFrame, profile: DatasetProfile, inputs: list[str] | None = None,
                   k_range: tuple[int, int] = (2, 8), random_state: int = 42,
                   fixed_k: int | None = None) -> ClusterResult:
    cols = list(inputs) if inputs else default_cluster_inputs(df, profile)
    if not cols:
        raise ValueError("No usable input columns for clustering")
    if len(df) < 10:
        raise ValueError("Need at least 10 rows to find clusters")
    X, Z, num, cat = _design(df, cols, profile)
    notes = []
    n = len(Z)
    rng = np.random.RandomState(random_state)
    sil_idx = rng.choice(n, MAX_SILHOUETTE_ROWS, replace=False) if n > MAX_SILHOUETTE_ROWS else np.arange(n)
    if n > MAX_SILHOUETTE_ROWS:
        notes.append(f"Silhouette scores computed on a {MAX_SILHOUETTE_ROWS:,}-row sample.")
    algos = ["KMeans", "Hierarchical (Ward)", "Gaussian mixture"]
    ks = [fixed_k] if fixed_k else list(range(k_range[0], min(k_range[1], n - 1) + 1))
    if n > 8000:
        algos = ["KMeans", "Gaussian mixture"]  # Ward is quadratic in rows
        notes.append("Hierarchical clustering skipped: too many rows for it.")
    rows, labels_by = [], {}
    for algo in algos:
        for k in ks:
            try:
                lab = _fit(algo, Z, k, random_state)
                if len(set(lab)) < 2:
                    continue
                sil = float(silhouette_score(Z[sil_idx], lab[sil_idx]))
            except Exception:  # noqa: BLE001 — one failing combination must not stop the search
                continue
            rows.append({"algorithm": algo, "k": k, "silhouette": sil})
            labels_by[(algo, k)] = lab
    if not rows:
        raise ValueError("Clustering failed for every setting tried")
    search = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    best = search.iloc[0]
    lab = labels_by[(best["algorithm"], int(best["k"]))]
    # order clusters by size so "Cluster 1" is always the biggest
    order = pd.Series(lab).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[v] for v in lab])
    labels = pd.Series(lab, index=df.index, name="cluster")
    sizes = labels.value_counts().sort_index()

    prof_rows = {}
    for cid in sizes.index:
        m = labels == cid
        r = {"size": int(m.sum()), "share": float(m.mean())}
        for c in num:
            r[c] = float(X.loc[m, c].mean())
        for c in cat:
            vc = X.loc[m, c].value_counts(normalize=True)
            r[c] = f"{vc.index[0]} ({vc.iloc[0]:.0%})" if len(vc) else ""
        prof_rows[cid] = r
    cluster_profile = pd.DataFrame(prof_rows).T
    cluster_profile.index.name = "cluster"

    pcs = PCA(n_components=2, random_state=random_state).fit_transform(Z) if Z.shape[1] >= 2 else \
        np.c_[Z[:, 0], np.zeros(n)]
    pca = pd.DataFrame({"pc1": pcs[:, 0], "pc2": pcs[:, 1], "cluster": lab}, index=df.index)
    if not fixed_k and int(best["k"]) in (ks[0], ks[-1]) and len(ks) > 2:
        edge = "smallest" if int(best["k"]) == ks[0] else "largest"
        notes.append(f"The best score sits at the {edge} number of groups tried (k={int(best['k'])}); "
                     "compare the other options in the search table or fix k yourself.")
    if float(best["silhouette"]) < 0.25:
        notes.append("Silhouette is below 0.25: the groups overlap a lot, so treat them as soft segments, "
                     "not sharp categories.")
    return ClusterResult(cols, str(best["algorithm"]), int(best["k"]), float(best["silhouette"]),
                         search, labels, sizes, cluster_profile, pca, notes)
