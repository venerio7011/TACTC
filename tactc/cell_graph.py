"""
cell_graph.py
=============

Step 2 of TACTC.ipynb: every DAPI (Mesmer) cell of the whole image becomes a node of one spatial graph G = (V, E),
not only the cells inside Visium spots.

Contents
----
* build_graph  : edges between neighbouring cells from their centres (µm)
                   'delaunay' : Delaunay triangulation, edges longer than max_edge_um removed (default; adapts to local
                                density and does not jump across empty areas)
                   'knn'      : k nearest neighbours (symmetrised), also cut at max_edge_um
                 → (connectivities, distances): symmetric sparse cell × cell matrices (1 / µm)
* graph_stats  : degree distribution, isolated cells, connected components
* cells_anndata: one cell-level AnnData (slots follow the pipeline plan)
                   obs  : cell_id, x_um, y_um, area_um2 (+ region / spot columns added in steps 3–4)
                   obsm : spatial (x, y µm), spatial_px (x, y full-resolution px), if_intensity (Mesmer channel means)
                   obsp : spatial_connectivities, spatial_distances (squidpy names, so squidpy / scanpy can use them)
                   X    : empty (cells × 0); per-cell RNA is added in step 6
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["build_graph", "graph_stats", "cells_anndata", "META_COLS"]

META_COLS = {"area", "y", "x", "spot", "domain", "domain_raw", "x_um", "y_um", "n_cells", "row", "col", "rna_domain"}


def _sym(i: np.ndarray, j: np.ndarray, d: np.ndarray, n: int):
    """Symmetric connectivity (1) and distance (µm) matrices from unique undirected pairs i < j."""
    from scipy import sparse

    rows, cols = np.r_[i, j], np.r_[j, i]
    D = sparse.csr_matrix((np.r_[d, d].astype(float), (rows, cols)), shape=(n, n))
    C = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return C, D


def build_graph(xy_um: np.ndarray, method: str = "delaunay", max_edge_um: float = 30.0, k: int = 6,
                verbose: bool = True) -> Tuple["object", "object"]:
    """Spatial cell graph from cell centres in µm. Returns (connectivities, distances) as CSR matrices."""
    from scipy.spatial import Delaunay, cKDTree

    xy = np.asarray(xy_um, float)
    n = len(xy)
    if n < 3:
        raise ValueError("Need at least 3 cells for a graph.")
    if method == "delaunay":
        tri = Delaunay(xy)
        s = tri.simplices
        e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
        e.sort(axis=1)
        e = np.unique(e, axis=0)
    elif method == "knn":
        _, nb = cKDTree(xy).query(xy, k=k + 1)
        e = np.column_stack([np.repeat(np.arange(n), k), nb[:, 1:].ravel()])
        e.sort(axis=1)
        e = np.unique(e, axis=0)
    else:
        raise ValueError("method must be 'delaunay' or 'knn'")
    d = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
    keep = (d <= max_edge_um) & (d > 0)
    C, D = _sym(e[keep, 0], e[keep, 1], d[keep], n)
    if verbose:
        st = graph_stats(C)
        print(f"[graph] {n:,} cells, {int(keep.sum()):,} edges ({method}, ≤ {max_edge_um:g} µm; "
              f"{int((~keep).sum()):,} longer edges removed) | degree median {st['degree_median']:.0f} "
              f"(5–95% {st['degree_5']:.0f}–{st['degree_95']:.0f}) | isolated {st['isolated']:,} | "
              f"components {st['components']:,} (largest {st['largest_component_frac']:.1%})")
    return C, D


def graph_stats(C) -> Dict[str, float]:
    """Degree distribution, isolated cells, connected components of a connectivity matrix."""
    from scipy.sparse.csgraph import connected_components

    deg = np.asarray((C > 0).sum(axis=1)).ravel()
    ncomp, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab)
    return dict(n_cells=int(C.shape[0]), n_edges=int(C.nnz // 2), degree_median=float(np.median(deg)),
                degree_5=float(np.percentile(deg, 5)), degree_95=float(np.percentile(deg, 95)),
                isolated=int((deg == 0).sum()), components=int(ncomp),
                largest_component_frac=float(sizes.max() / len(lab)))


def cells_anndata(cells: pd.DataFrame, mpp: float, conn=None, dist=None, channels: Optional[list] = None):
    """Cell-level AnnData from the Mesmer cell table (index = cell_id; columns area (px), y, x (px), channels)."""
    import anndata as ad
    from scipy import sparse

    xy_px = cells[["x", "y"]].to_numpy(float)
    obs = pd.DataFrame({"cell_id": cells.index.astype(np.int64),
                        "x_um": xy_px[:, 0] * mpp, "y_um": xy_px[:, 1] * mpp,
                        "area_um2": cells["area"].to_numpy(float) * mpp * mpp},
                       index=pd.Index(cells.index.astype(str), name="cell"))
    a = ad.AnnData(X=sparse.csr_matrix((len(cells), 0), dtype=np.float32), obs=obs)
    a.obsm["spatial"] = xy_px * mpp
    a.obsm["spatial_px"] = xy_px
    chans = channels if channels is not None else [c for c in cells.columns if c not in META_COLS]
    chans = [c for c in chans if pd.api.types.is_numeric_dtype(cells[c])]
    if chans:
        I = cells[chans].astype(np.float32).copy()
        I.columns = [str(c).replace("/", "_") for c in I.columns]
        I.index = a.obs_names
        a.obsm["if_intensity"] = I
    if conn is not None:
        a.obsp["spatial_connectivities"] = conn.tocsr()
    if dist is not None:
        a.obsp["spatial_distances"] = dist.tocsr()
    a.uns["mpp"] = float(mpp)
    return a
