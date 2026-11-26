"""
paga_nodes.py
=============

Step 10 of TACTC.ipynb: expression clusters and PAGA of the spot nodes from RNA alone (no spatial information), and
the k sensitivity of the clustering.

10-1 Preprocessing (scanpy tutorials: 3k PBMC clustering, spatial basic analysis)
     raw counts of the node spots (all genes) → normalize_total(1e4) → log1p → highly_variable_genes(n_top_genes,
     flavor 'seurat') → scale(max_value=10)  → z (nodes × HVGs)
10-2 Clusters and PAGA (Wolf et al., Genome Biology 2019): z → PCA → sc.pp.neighbors → sc.tl.leiden (flavor
     'igraph', 2 iterations; resolution searched to give k clusters) → sc.tl.paga → sc.pl.paga (layout) →
     sc.tl.umap(init_pos='paga'). Optional cell-number edge weights (off by default).
10-3 k sensitivity (descriptive): for several k,
       split_half_ari : fit on one side of a random straight cut, transfer labels to the other side (kNN in the
                        fitted PCA space), ARI with the full-data labels (mean over both sides of each cut)
       <feature>_z    : separation (η²) of independent features (Visium proteins; node morphology) by cluster,
                        against the shift null of the label map
       coherence      : share of lattice edges inside a cluster
Cluster summaries: cell-weighted group composition, cells and morphology per cluster.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .node_graphs import _sep_z, _transfer, random_cuts, shift_maps

__all__ = ["preprocess", "cluster_features", "leiden_to_k", "branch", "k_sensitivity", "cluster_summary"]


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# ---------------------------------------------------------------------------
# 10-1 preprocessing
# ---------------------------------------------------------------------------
def preprocess(spots_adata, nodes, n_top_genes: int = 2000, verbose: bool = True):
    """Standard scanpy preprocessing of the node spots. Returns an AnnData: X = scaled log HVG expression,
    raw = log-normalised all genes, layers['counts'] (HVG counts), obsm spatial, protein, obsp lattice,
    obs node_weight, n_hex, dominant_group."""
    import scanpy as sc

    a = spots_adata[nodes.obs_names].copy()
    prot = a.obsm["protein"] if "protein" in a.obsm else None
    for k in list(a.obsm.keys()):
        del a.obsm[k]
    if prot is not None:
        a.obsm["protein"] = prot
    for k in list(a.uns.keys()):
        if k not in ("spatial", "mpp"):
            del a.uns[k]
    a.obsm["spatial"] = np.asarray(nodes.obsm["spatial"], float)
    a.obsp["lattice"] = nodes.obsp["spatial_connectivities"]
    for c in ("node_weight", "n_hex", "dominant_group"):
        a.obs[c] = nodes.obs[c].to_numpy()
    counts = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(int(n_top_genes), a.n_vars), flavor="seurat")
    a.raw = a
    hv = a.var["highly_variable"].to_numpy()
    a = a[:, hv].copy()
    a.layers["counts"] = counts[:, hv]
    sc.pp.scale(a, max_value=10)
    a.uns["preprocess"] = dict(n_top_genes=int(n_top_genes), target_sum=1e4, hvg_flavor="seurat", scale_max=10)
    if verbose:
        print(f"[10-1] {a.n_obs:,} nodes × {a.n_vars:,} HVGs (normalize_total → log1p → HVG → scale)")
    return a


# ---------------------------------------------------------------------------
# graph + Leiden (one implementation for the clusters, the half fits and the k sensitivity)
# ---------------------------------------------------------------------------
def leiden_to_k(adata, k: Optional[int], key: str, resolution: float = 1.0, seed: int = 0, lo: float = 0.01,
                hi: float = 20.0, iters: int = 30):
    """sc.tl.leiden (flavor 'igraph', n_iterations 2) at `resolution`, or with the resolution bisected (log scale)
    until it gives k clusters (closest found if exact k is not reachable). Returns the resolution used."""
    import scanpy as sc

    kw = dict(flavor="igraph", n_iterations=2, directed=False, random_state=seed, key_added=key)
    if not k:
        sc.tl.leiden(adata, resolution=resolution, **kw)
        return float(resolution)
    a, b = np.log(lo), np.log(hi)
    best = None
    for _ in range(iters):
        r = float(np.exp((a + b) / 2))
        sc.tl.leiden(adata, resolution=r, **kw)
        nk = adata.obs[key].nunique()
        if best is None or abs(nk - k) < abs(best[1] - k):
            best = (r, nk)
        if nk == k:
            return r
        a, b = (np.log(r), b) if nk < k else (a, np.log(r))
    sc.tl.leiden(adata, resolution=best[0], **kw)
    return float(best[0])


def cluster_features(F: np.ndarray, k: Optional[int], n_pcs: int = 30, n_neighbors: int = 15, resolution: float = 1.0,
                     cell_weights: Optional[np.ndarray] = None, rows: Optional[np.ndarray] = None, seed: int = 0):
    """PCA → sc.pp.neighbors → (optional cell-number edge weights) → Leiden on the rows of F.
    Returns (labels, resolution, working AnnData with obsm X_pca and the neighbours graph, fitted PCA)."""
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from sklearn.decomposition import PCA

    idx = np.arange(len(F)) if rows is None else np.flatnonzero(rows) if rows.dtype == bool else np.asarray(rows)
    X = F[idx]
    nc = int(max(2, min(50, X.shape[1] - 1, len(idx) - 1)))
    pca = PCA(n_components=nc, random_state=seed).fit(X)
    P = pca.transform(X).astype(np.float32)
    t = ad.AnnData(np.zeros((len(idx), 1), np.float32))
    t.obsm["X_pca"] = P[:, :int(min(n_pcs, nc))]
    sc.pp.neighbors(t, n_neighbors=int(min(n_neighbors, len(idx) - 1)), use_rep="X_pca", random_state=seed)
    if cell_weights is not None:
        C = t.obsp["connectivities"].tocoo()
        cw = np.asarray(cell_weights, float)[idx]
        w = np.sqrt(np.clip(cw[C.row], 0, None) * np.clip(cw[C.col], 0, None))
        w = w / max(np.median(w[w > 0]) if (w > 0).any() else 1.0, 1e-12)
        t.obsp["connectivities"] = sparse.csr_matrix((C.data * w, (C.row, C.col)), shape=C.shape)
    res = leiden_to_k(t, k, "leiden", resolution, seed)
    return t.obs["leiden"].astype(str).to_numpy(), res, t, pca


def _coherence(lattice, lab: np.ndarray) -> float:
    C = lattice.tocoo()
    same = lab[C.row] == lab[C.col]
    return float((C.data * same).sum() / max(C.data.sum(), 1e-12))


# ---------------------------------------------------------------------------
# 10-2 clusters and PAGA
# ---------------------------------------------------------------------------
def _features(pre) -> np.ndarray:
    """Scaled HVG expression z. A zero block of the same width is appended so that the PCA input is identical to the
    one used for the published example run (it does not change the principal components)."""
    Z = _dense(pre.X).astype(float)
    return np.hstack([Z, np.zeros_like(Z)])


def branch(pre, tag: str = "std", k: Optional[int] = None, resolution: float = 1.0, n_pcs: int = 30,
           n_neighbors: int = 15, weight_by_cells: bool = False, paga_threshold: float = 0.03, seed: int = 0,
           verbose: bool = True):
    """Clustering + PAGA + UMAP(init_pos='paga'), stored in `pre` (in place): obs['leiden_<tag>'],
    obsm['X_pca_<tag>'], ['X_umap_<tag>'], obsp['<tag>_connectivities'], uns['paga_<tag>'], uns['<tag>_params']."""
    import scanpy as sc

    F = _features(pre)
    cw = pre.obs["node_weight"].to_numpy(float) if weight_by_cells else None
    lab, res, t, _ = cluster_features(F, k, n_pcs, n_neighbors, resolution=resolution, cell_weights=cw, seed=seed)
    t.obs["leiden"] = pd.Categorical(lab)
    sc.tl.paga(t, groups="leiden")
    sc.pl.paga(t, threshold=paga_threshold, plot=False)
    sc.tl.umap(t, init_pos="paga", random_state=seed)
    key = f"leiden_{tag}"
    pre.obs[key] = pd.Categorical(lab, categories=sorted(set(lab), key=lambda x: int(x) if x.isdigit() else x))
    pre.obsm[f"X_pca_{tag}"] = t.obsm["X_pca"]
    pre.obsm[f"X_umap_{tag}"] = t.obsm["X_umap"]
    pre.obsp[f"{tag}_connectivities"] = t.obsp["connectivities"]
    p = {kk: t.uns["paga"][kk] for kk in t.uns["paga"]}
    p["groups"] = key
    pre.uns[f"paga_{tag}"] = p
    pre.uns[f"{tag}_params"] = dict(k=k, resolution=float(res), n_pcs=int(n_pcs), n_neighbors=int(n_neighbors),
                                    weight_by_cells=bool(weight_by_cells), paga_threshold=paga_threshold, seed=seed)
    if verbose:
        print(f"[10 {tag}] {len(set(lab))} clusters (resolution {res:.3g}) | cell weights "
              f"{'on' if weight_by_cells else 'off'} | coherence {_coherence(pre.obsp['lattice'], lab):.3f}")
    return pre


# ---------------------------------------------------------------------------
# 10-3 k sensitivity
# ---------------------------------------------------------------------------
def k_sensitivity(pre, independent: Dict[str, pd.DataFrame], ks: Sequence[int] = (4, 5, 6, 7, 8), n_pcs: int = 30,
                  n_neighbors: int = 15, weight_by_cells: bool = False, n_split: int = 10, n_shift: int = 200,
                  min_shift_um: float = 1000.0, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Per k: clusters found, resolution, split-half ARI (and the per-fit values), coherence and the shift-null z of
    each independent feature set (module docstring, 10-3)."""
    from sklearn.metrics import adjusted_rand_score

    F = _features(pre)
    xy = np.asarray(pre.obsm["spatial"], float)
    cw = pre.obs["node_weight"].to_numpy(float) if weight_by_cells else None
    rows = []
    for k in ks:
        k = int(k)
        cuts = random_cuts(xy, n_split, seed)
        shifts = shift_maps(xy, n_shift, min_shift_um, seed)
        lab, res, _, _ = cluster_features(F, k, n_pcs, n_neighbors, cell_weights=cw, seed=seed)
        aris = []
        for c in cuts:
            side = xy @ c["u"] < c["t"]
            for fit in (side, ~side):
                if fit.sum() < 3 * k or (~fit).sum() < 3:
                    continue
                lf, _, tf, pf = cluster_features(F, k, n_pcs, n_neighbors, cell_weights=cw, rows=fit, seed=seed)
                Po = pf.transform(F[~fit])[:, :tf.obsm["X_pca"].shape[1]]
                aris.append(adjusted_rand_score(lab[~fit], _transfer(tf.obsm["X_pca"], lf, Po)))
        row = dict(k_target=k, clusters=int(len(np.unique(lab))), resolution=float(res),
                   split_half_ari=float(np.mean(aris)) if aris else np.nan,
                   coherence=_coherence(pre.obsp["lattice"], lab),
                   ari_values=json.dumps([round(float(x), 6) for x in aris]))   # per fit (2 per cut), for CIs
        for name, df in independent.items():
            z = _sep_z(df.reindex(pre.obs_names), lab, shifts)
            row[f"{name}_z"] = float(np.nanmean(z)) if np.isfinite(z).any() else np.nan
        rows.append(row)
        if verbose:
            print(f"[k {k}] found {row['clusters']} | split-half ARI {row['split_half_ari']:.3f} | "
                  + " | ".join(f"{n} z {row[n + '_z']:.2f}" for n in independent) + f" | coherence {row['coherence']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cluster summaries
# ---------------------------------------------------------------------------
def cluster_summary(nodes, labels) -> pd.DataFrame:
    """Per cluster: nodes, cells, cell-weighted group composition, mean morphology, top group."""
    groups = list(nodes.uns["nodes"]["groups"])
    Ch = pd.DataFrame(np.asarray(nodes.obsm["cell_counts_hex"], float), index=nodes.obs_names, columns=groups)
    lab = pd.Series(np.asarray(labels).astype(str), index=nodes.obs_names, name="cluster")
    cells = Ch.sum(axis=1)
    comp = Ch.groupby(lab).sum()
    comp = comp.div(comp.sum(axis=1), axis=0).add_prefix("frac_")
    out = pd.DataFrame({"nodes": lab.value_counts(), "cells": cells.groupby(lab).sum()}).join(comp)
    if "morphology" in nodes.obsm:
        M = nodes.obsm["morphology"].astype(float)
        wm = M.mul(cells, axis=0).groupby(lab).sum().div(cells.groupby(lab).sum(), axis=0)
        out = out.join(wm.add_prefix("mean_"))
    out["top_group"] = comp.idxmax(axis=1).str.replace("frac_", "", regex=False)
    return out.sort_values("cells", ascending=False)
