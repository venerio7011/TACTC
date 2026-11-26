"""
spot_nodes.py
=============

Step 5 of TACTC.ipynb: one node per hexagon (Visium spot centre), at the resolution RNA was actually measured.
It replaces cell-level type assignment and per-cell RNA: within a spot the arrangement of cell types and the differences
between its cells cannot be identified from spot RNA, so later steps (graphs, correlations, Leiden / PAGA) use these nodes.

Node attributes (one AnnData; slots follow the pipeline plan)
  X                       : log1p(counts per 10k) of the selected genes. Normalisation removes the factor shared by all
                            genes of a node (cell number, capture efficiency, hexagon ratio), so distances / correlations
                            reflect what is expressed, not how much
  layers['counts']        : measured spot counts (selected genes)
  layers['hex_scaled']    : counts × hexagon ratio (absolute hexagon totals; only for amounts, not for correlations)
  layers['decomp_<i>']    : the counts' share of group i (spot decomposition with group signatures; sums over i to
                            'counts'), group order in uns['nodes']['groups']
  obs                     : n_hex (node weight = DAPI cells in the hexagon), n_circle, n_ring, hex_ratio,
                            ratio_unstable (fewer than min_circle circle cells), node_weight, dominant_group
  obsm                    : spatial (µm), cell_counts_hex / _circle / _ring, group_frac (hexagon composition),
                            program_usage (step-1 RNA shares per program), morphology (per-node nuclear summaries)
  obsp                    : spatial_connectivities / spatial_distances = hexagon lattice (6 neighbours), edge weight
                            √(n_i n_j) / median → nodes with few cells influence the graph less
  varm['group_signature'] : gene × group per-cell UMI (non-negative least squares over spots; reference-free)

Group signatures and decomposition follow visiumif2niche 10c step 1 with spot-derived signatures:
  μ_G(g) from  y[s,g] ≈ Σ_G n_circle[s,G] μ_G(g)   and   Y[s,g,G] = y[s,g] · c[s,G] μ_G(g) / Σ_G' c[s,G'] μ_G'(g)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["select_genes", "group_signatures", "decompose", "lattice_graph", "node_morphology", "build_spot_nodes",
           "node_frame"]


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def select_genes(vis, n_top: int = 2000, extra: Sequence[str] = (), verbose: bool = True) -> List[str]:
    """Highly variable genes of the spots (scanpy 'seurat_v3' on counts; 'seurat' on log if scikit-misc is missing)
    + extra genes present in the data."""
    import scanpy as sc

    a = vis.copy()
    n = min(int(n_top), a.n_vars)
    try:
        sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat_v3")
    except Exception:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat")
    genes = list(a.var_names[a.var["highly_variable"].to_numpy()])
    add = [g for g in extra if g in vis.var_names and g not in genes]
    if verbose:
        print(f"[nodes] {len(genes):,} variable genes + {len(add)} extra genes")
    return genes + add


def group_signatures(vis, circle_counts: pd.DataFrame, genes: Sequence[str], min_cells: int = 1,
                     verbose: bool = True) -> pd.DataFrame:
    """gene × group per-cell UMI by non-negative least squares over spots with ≥ min_cells circle cells."""
    from scipy.optimize import nnls

    N = circle_counts.reindex(vis.obs_names).fillna(0).astype(float)
    use = N.sum(axis=1) >= min_cells
    A = N[use].to_numpy()
    Y = _dense(vis[use.to_numpy(), list(genes)].X).astype(float)
    M = np.zeros((len(genes), A.shape[1]))
    for j in range(len(genes)):
        M[j] = nnls(A, Y[:, j])[0]
    sig = pd.DataFrame(M, index=list(genes), columns=list(N.columns))
    if verbose:
        print(f"[nodes] group signatures from {int(use.sum()):,} spots (per-cell UMI over the selected genes: "
              + ", ".join(f"{g} {v:.0f}" for g, v in sig.sum().items()) + ")")
    return sig


def decompose(counts: np.ndarray, comp: np.ndarray, signatures: np.ndarray) -> np.ndarray:
    """Y[s,g,G]: spot counts (S × genes) split over groups by composition (S × G) × signature (genes × G).
    Genes absent from every signature are split by composition. Returns G × S × genes (sums over G to counts)."""
    mu = signatures.T                                        # G × genes
    tot = comp @ mu                                          # S × genes
    out = np.empty((comp.shape[1],) + counts.shape, dtype=np.float32)
    for g in range(comp.shape[1]):
        share = np.divide(comp[:, g:g + 1] * mu[g][None, :], tot, out=np.zeros_like(tot), where=tot > 0)
        zero = tot <= 0
        if zero.any():
            share[zero] = np.broadcast_to(comp[:, g:g + 1], tot.shape)[zero]
        out[g] = (counts * share).astype(np.float32)
    return out


def lattice_graph(xy_um: np.ndarray, weights: Optional[np.ndarray] = None, factor: float = 1.2):
    """Hexagon lattice: nodes within factor × spot pitch are neighbours (6 in the interior).
    weights (node cell numbers) → edge weight √(n_i n_j) / median over edges; None → 1.
    Returns (connectivities, distances µm, pitch µm)."""
    from scipy import sparse
    from scipy.spatial import cKDTree

    xy = np.asarray(xy_um, float)
    tree = cKDTree(xy)
    pitch = float(np.median(tree.query(xy, k=2)[0][:, 1]))
    pr = tree.query_pairs(factor * pitch, output_type="ndarray")
    i, j = pr[:, 0], pr[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    if weights is not None:
        w = np.sqrt(np.clip(np.asarray(weights, float)[i], 0, None) * np.clip(np.asarray(weights, float)[j], 0, None))
        med = np.median(w[w > 0]) if (w > 0).any() else 1.0
        w = w / med
    else:
        w = np.ones(len(i))
    n = len(xy)
    rows, cols = np.r_[i, j], np.r_[j, i]
    C = sparse.csr_matrix((np.r_[w, w], (rows, cols)), shape=(n, n))
    D = sparse.csr_matrix((np.r_[d, d], (rows, cols)), shape=(n, n))
    return C, D, pitch


def node_morphology(cells_ad, spots: Sequence[str], pitch_um: float) -> pd.DataFrame:
    """Per hexagon from its DAPI cells (circle + ring): density (cells / mm²), nuclear area mean / median / CV and
    mean log intensity of each Mesmer channel."""
    obs = cells_ad.obs
    m = obs["region"].astype(str).isin(["circle", "ring"]).to_numpy()
    o = obs[m]
    key = o["spot"].astype(str)
    area = o["area_um2"].astype(float)
    g = area.groupby(key)
    df = pd.DataFrame({"area_mean_um2": g.mean(), "area_median_um2": g.median(),
                       "area_cv": g.std() / g.mean()})
    hex_area_mm2 = np.sqrt(3) / 2 * pitch_um ** 2 / 1e6
    df["density_per_mm2"] = key.value_counts() / hex_area_mm2
    if "if_intensity" in cells_ad.obsm:
        I = cells_ad.obsm["if_intensity"]
        I = I.iloc[np.flatnonzero(m)]
        for c in I.columns:
            df[f"log_{c}_mean"] = np.log1p(I[c].astype(float).clip(lower=0)).groupby(key.to_numpy()).mean()
    df = df.reindex(pd.Index(spots, dtype=str))
    df.columns = [re.sub(r"[^A-Za-z0-9_.-]+", "_", str(c)) for c in df.columns]
    return df


def build_spot_nodes(spots4, hexa, hc, cells_ad, genes: Sequence[str], signatures: Optional[pd.DataFrame] = None,
                     min_cells: int = 1, min_circle: int = 2, ratio_cap_quantile: float = 0.99,
                     verbose: bool = True):
    """Spot-node AnnData (see the module docstring).

    spots4     : step-4 spot AnnData (raw counts in X; obsm topic_theta, uns step1 / step4)
    hexa, hc   : hexagons.HexAssignment / HexCounts
    cells_ad   : step-4 cell AnnData (for morphology)
    genes      : genes to carry (select_genes)
    signatures : gene × group (group_signatures); computed here if None
    min_cells  : nodes need ≥ min_cells DAPI cells in the hexagon and step-1 shares
    min_circle : hexagons with fewer circle cells get ratio_unstable = True; their ratio is capped at the
                 ratio_cap_quantile of all hexagons for hex_scaled
    """
    import anndata as ad
    from scipy import sparse

    groups = [str(g) for g in spots4.uns["step1"]["groups"]]
    sp = hexa.spots.reindex(spots4.obs_names)
    keep = (sp["n_hex"].fillna(0).to_numpy() >= min_cells) & sp["has_shares"].fillna(False).to_numpy().astype(bool)
    spots = spots4.obs_names[keep]
    s4 = spots4[spots]
    genes = [g for g in genes if g in s4.var_names]
    Y = _dense(s4[:, genes].X).astype(np.float64)
    if signatures is None:
        signatures = group_signatures(spots4, hc.circle, genes, verbose=verbose)
    signatures = signatures.reindex(index=genes, columns=groups).fillna(0.0)

    Ch = hc.hex.reindex(index=spots, columns=groups).fillna(0).to_numpy(float)
    nh = Ch.sum(axis=1)
    comp = np.divide(Ch, nh[:, None], out=np.zeros_like(Ch), where=nh[:, None] > 0)
    n_circ = sp.loc[spots, "n_circle"].to_numpy(float)
    ratio = sp.loc[spots, "hex_ratio_members"].to_numpy(float)
    unstable = n_circ < min_circle
    finite = np.isfinite(ratio) & ~unstable
    cap = float(np.quantile(ratio[finite], ratio_cap_quantile)) if finite.any() else 1.0
    ratio_used = np.where(np.isfinite(ratio), np.minimum(ratio, cap), cap)
    ratio_used = np.where(unstable, np.minimum(ratio_used, cap), ratio_used)

    tot = Y.sum(axis=1, keepdims=True)
    Xn = np.log1p(np.divide(Y, tot, out=np.zeros_like(Y), where=tot > 0) * 1e4).astype(np.float32)
    a = ad.AnnData(X=Xn, obs=pd.DataFrame(index=pd.Index(spots, name="spot")),
                   var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    a.layers["counts"] = sparse.csr_matrix(Y.astype(np.float32))
    a.layers["hex_scaled"] = sparse.csr_matrix((Y * ratio_used[:, None]).astype(np.float32))
    D = decompose(Y, comp, signatures.to_numpy(float))
    for i in range(len(groups)):
        a.layers[f"decomp_{i}"] = sparse.csr_matrix(D[i])

    a.obs["n_hex"] = nh.astype(int)
    a.obs["n_circle"] = n_circ.astype(int)
    a.obs["n_ring"] = sp.loc[spots, "n_ring"].to_numpy(int)
    a.obs["hex_ratio"] = ratio
    a.obs["hex_ratio_used"] = ratio_used
    a.obs["ratio_unstable"] = unstable
    a.obs["node_weight"] = nh.astype(float)
    a.obs["total_counts"] = tot.ravel()
    a.obs["dominant_group"] = pd.Categorical(np.array(groups, dtype=object)[np.argmax(Ch, axis=1)])

    mpp = float(spots4.uns.get("mpp", 1.0))
    xy = s4.obs[["x", "y"]].to_numpy(float) * mpp
    a.obsm["spatial"] = xy
    for key, df in (("cell_counts_hex", hc.hex), ("cell_counts_circle", hc.circle), ("cell_counts_ring", hc.ring)):
        a.obsm[key] = df.reindex(index=spots, columns=groups).fillna(0).to_numpy(float)
    a.obsm["group_frac"] = comp
    if "topic_theta" in s4.obsm:
        a.obsm["program_usage"] = np.asarray(s4.obsm["topic_theta"], float)
    C, Dm, pitch = lattice_graph(xy, weights=nh)
    a.obsp["spatial_connectivities"] = C
    a.obsp["spatial_distances"] = Dm
    morph = node_morphology(cells_ad, spots, pitch)
    morph.index = a.obs_names
    a.obsm["morphology"] = morph
    a.varm["group_signature"] = signatures.to_numpy(float)
    st = spots4.uns.get("step1", {})
    a.uns["nodes"] = {"groups": groups, "programs": list(st.get("topics", [])),
                      "pitch_um": pitch, "ratio_cap": cap, "min_cells": int(min_cells), "min_circle": int(min_circle),
                      "decomp_layers": [f"decomp_{i}" for i in range(len(groups))],
                      "n_nodes": int(a.n_obs), "n_genes": int(a.n_vars)}
    if verbose:
        deg = np.asarray((C > 0).sum(axis=1)).ravel()
        print(f"[nodes] {a.n_obs:,} spot nodes × {a.n_vars:,} genes, {len(groups)} groups | cells per node median "
              f"{np.median(nh):.0f} | lattice pitch {pitch:.1f} µm, degree median {np.median(deg):.0f} | "
              f"{int(unstable.sum())} nodes with < {min_circle} circle cells (ratio capped at {cap:.2f})")
    return a


def node_frame(nodes, key: str) -> pd.DataFrame:
    """obsm[key] (cell_counts_* / group_frac / program_usage) of the node AnnData as a DataFrame."""
    cols = nodes.uns["nodes"]["programs"] if key == "program_usage" else nodes.uns["nodes"]["groups"]
    return pd.DataFrame(np.asarray(nodes.obsm[key]), index=nodes.obs_names, columns=list(cols))
