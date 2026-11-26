"""
quality.py
==========

TACTC.ipynb 14-4: are the differences between core and scattered nodes explained by spot quality?
Added after the primary results (docs/VALIDATION.md, change log 2026-11-25); it does not change the verdict.

Quality of a node = RNA count over the clustering genes (HVGs) and the number of nuclei in its hexagon.
Outcomes are per-node results of section 13: label agreement across reclustered halves (13-1, P1) and whether the
cluster was predicted correctly from same-spot proteins and morphology with spatial-block folds (13-2, P2).

quality_table      : per node: cluster, set, boundary, RNA count, detected genes, nuclei, x / y (µm)
filter_crosstab    : a standard filter (≥ min_cells nuclei and ≥ min_counts counts) × core / scattered: nodes and
                     mean outcome per cell, and core − scattered among nodes passing the filter
matched_difference : each scattered node matched to the k core nodes of the same cluster (and boundary status) with
                     the most similar log RNA and log nucleus counts (standardised over all nodes; caliper in SD);
                     core − scattered over matched scattered nodes
adjusted_difference: OLS of the outcome on scattered (0/1) + cluster dummies + boundary + log RNA + log nuclei;
                     core − scattered = − coefficient of scattered; optionally restricted to nodes passing the filter
Intervals: spatial block bootstrap (blocks of block_um; every block size in block_sizes is reported). Nodes in one
block are resampled together, so neighbouring nodes do not count as independent. Point estimates do not depend on
the block size.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = ["quality_table", "filter_crosstab", "matched_difference", "adjusted_difference", "quality_check"]


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def quality_table(pre, label_key: str = "leiden_std", set_key: str = "support_set", boundary_key: str = "boundary",
                  counts_layer: str = "counts", cells_key: str = "n_hex") -> pd.DataFrame:
    """Per node quality from the step-11 AnnData (support_step11.h5ad)."""
    Y = pre.layers[counts_layer]
    umi = np.asarray(Y.sum(axis=1)).ravel().astype(float)
    genes = np.asarray((Y > 0).sum(axis=1)).ravel().astype(float)
    xy = np.asarray(pre.obsm["spatial"], float)
    return pd.DataFrame(dict(cluster=pre.obs[label_key].astype(str).to_numpy(),
                             set=pre.obs[set_key].astype(str).to_numpy(),
                             boundary=pre.obs[boundary_key].to_numpy(bool),
                             umi_hvg=umi, genes_hvg=genes, cells_hex=pre.obs[cells_key].to_numpy(float),
                             x_um=xy[:, 0], y_um=xy[:, 1]), index=pre.obs_names)


def _block_index(xy: np.ndarray, block_um: float) -> np.ndarray:
    b = np.floor(xy / block_um).astype(np.int64)
    return np.unique(b[:, 0] * 100003 + b[:, 1], return_inverse=True)[1]


def _block_boot(stat, xy: np.ndarray, block_um: float, n_boot: int, rng) -> np.ndarray:
    """Statistic over resamples of whole spatial blocks; stat(idx) gets row indices (with repeats)."""
    blk = _block_index(xy, block_um)
    members = [np.flatnonzero(blk == b) for b in range(blk.max() + 1)]
    nb = len(members)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, nb, nb)
        out[i] = stat(np.concatenate([members[j] for j in pick]))
    return out[np.isfinite(out)]


def _ci_rows(est: float, xy: np.ndarray, stat, block_sizes: Sequence[float], n_boot: int, seed: int) -> Dict:
    row = {}
    for bs in block_sizes:
        b = _block_boot(stat, xy, bs, n_boot, np.random.default_rng(seed))
        tag = f"{int(bs)}um"
        row[f"ci_low_{tag}"] = float(np.quantile(b, 0.025)) if len(b) else np.nan
        row[f"ci_high_{tag}"] = float(np.quantile(b, 0.975)) if len(b) else np.nan
    return row


def filter_crosstab(Q: pd.DataFrame, outcome: pd.Series, name: str, min_cells: float = 10, min_counts: float = 500,
                    block_sizes: Sequence[float] = (500.0, 1000.0), n_boot: int = 2000, seed: int = 0):
    """(table per filter × set, row with core − scattered among nodes passing the filter)."""
    y = outcome.reindex(Q.index).to_numpy(float)
    passing = (Q["cells_hex"] >= min_cells) & (Q["umi_hvg"] >= min_counts)
    T = (pd.DataFrame(dict(qc_pass=passing.to_numpy(), set=Q["set"].to_numpy(), y=y))
         .groupby(["qc_pass", "set"])["y"].agg(nodes="size", mean="mean").reset_index())
    T.insert(0, "outcome", name)
    m = passing.to_numpy() & np.isfinite(y)
    xy = Q[["x_um", "y_um"]].to_numpy(float)[m]
    yy, core = y[m], (Q["set"].to_numpy()[m] == "core")

    def stat(idx):
        c, s = core[idx], ~core[idx]
        return yy[idx][c].mean() - yy[idx][s].mean() if c.any() and s.any() else np.nan

    est = stat(np.arange(len(yy)))
    row = dict(outcome=name, analysis="quality filter passed", n_scattered=int((~core).sum()),
               core=float(yy[core].mean()), scattered=float(yy[~core].mean()), difference=float(est))
    row.update(_ci_rows(est, xy, stat, block_sizes, n_boot, seed))
    return T, row


def matched_difference(Q: pd.DataFrame, outcome: pd.Series, name: str, exact: Sequence[str] = ("cluster", "boundary"),
                       k: int = 3, caliper: float = 0.25, block_sizes: Sequence[float] = (500.0, 1000.0),
                       n_boot: int = 2000, seed: int = 0) -> Dict:
    """core − scattered over scattered nodes matched to k quality-similar core nodes (module docstring)."""
    y = outcome.reindex(Q.index).to_numpy(float)
    Z = np.c_[np.log1p(Q["umi_hvg"].to_numpy(float)), np.log1p(Q["cells_hex"].to_numpy(float))]
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
    sets = Q["set"].to_numpy()
    keys = Q[list(exact)].astype(str).agg("|".join, axis=1).to_numpy()
    rows_s, d, ys, yc, bal = [], [], [], [], []
    n_sc = int((sets == "scattered").sum())
    for key in np.unique(keys):
        s_idx = np.flatnonzero((keys == key) & (sets == "scattered") & np.isfinite(y))
        c_idx = np.flatnonzero((keys == key) & (sets == "core") & np.isfinite(y))
        if len(s_idx) == 0 or len(c_idx) < k:
            continue
        dist, j = cKDTree(Z[c_idx]).query(Z[s_idx], k=k)
        dist, j = np.atleast_2d(dist), np.atleast_2d(j)
        ok = dist.max(axis=1) <= caliper * np.sqrt(2)
        for n in np.flatnonzero(ok):
            cm = c_idx[j[n]]
            rows_s.append(s_idx[n]); ys.append(y[s_idx[n]]); yc.append(y[cm].mean())
            bal.append((Q["umi_hvg"].iloc[s_idx[n]], Q["umi_hvg"].iloc[cm].mean(),
                        Q["cells_hex"].iloc[s_idx[n]], Q["cells_hex"].iloc[cm].mean()))
    ys, yc = np.asarray(ys), np.asarray(yc)
    d = yc - ys
    xy = Q[["x_um", "y_um"]].to_numpy(float)[np.asarray(rows_s, int)]
    stat = (lambda idx: d[idx].mean()) if len(d) else (lambda idx: np.nan)
    est = float(d.mean()) if len(d) else np.nan
    B = np.asarray(bal) if bal else np.full((1, 4), np.nan)
    row = dict(outcome=name, analysis="matched (" + " + ".join(exact) + ")", n_scattered=n_sc, n_matched=len(d),
               core=float(yc.mean()) if len(d) else np.nan, scattered=float(ys.mean()) if len(d) else np.nan,
               difference=est, umi_scattered=float(np.median(B[:, 0])), umi_core=float(np.median(B[:, 1])),
               cells_scattered=float(np.median(B[:, 2])), cells_core=float(np.median(B[:, 3])))
    if len(d):
        row.update(_ci_rows(est, xy, stat, block_sizes, n_boot, seed))
    return row


def _ols_scattered(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1]


def adjusted_difference(Q: pd.DataFrame, outcome: pd.Series, name: str, restrict: Optional[np.ndarray] = None,
                        block_sizes: Sequence[float] = (500.0, 1000.0), n_boot: int = 2000, seed: int = 0) -> Dict:
    """core − scattered from OLS with cluster dummies, boundary, log RNA and log nuclei (module docstring)."""
    y = outcome.reindex(Q.index).to_numpy(float)
    m = np.isfinite(y) if restrict is None else (np.isfinite(y) & np.asarray(restrict, bool))
    q = Q[m]
    cl = pd.get_dummies(q["cluster"], drop_first=True, dtype=float).to_numpy()
    X = np.c_[np.ones(len(q)), (q["set"].to_numpy() == "scattered").astype(float), cl,
              q["boundary"].to_numpy(float), np.log1p(q["umi_hvg"].to_numpy(float)),
              np.log1p(q["cells_hex"].to_numpy(float))]
    yy = y[m]
    xy = q[["x_um", "y_um"]].to_numpy(float)

    def stat(idx):
        if X[idx, 1].min() == X[idx, 1].max():
            return np.nan
        return -_ols_scattered(yy[idx], X[idx])

    est = float(stat(np.arange(len(yy))))
    row = dict(outcome=name, analysis="adjusted" + ("" if restrict is None else ", quality filter passed"),
               n_scattered=int(X[:, 1].sum()), difference=est)
    row.update(_ci_rows(est, xy, stat, block_sizes, n_boot, seed))
    return row


def quality_check(Q: pd.DataFrame, outcomes: Dict[str, pd.Series], min_cells: float = 10, min_counts: float = 500,
                  k: int = 3, caliper: float = 0.25, block_sizes: Sequence[float] = (500.0, 1000.0),
                  n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """All analyses for every outcome → dict(summary, crosstab, medians)."""
    rows, tabs = [], []
    passing = ((Q["cells_hex"] >= min_cells) & (Q["umi_hvg"] >= min_counts)).to_numpy()
    for name, y in outcomes.items():
        T, r = filter_crosstab(Q, y, name, min_cells, min_counts, block_sizes, n_boot, seed)
        tabs.append(T)
        rows += [matched_difference(Q, y, name, ("cluster", "boundary"), k, caliper, block_sizes, n_boot, seed),
                 matched_difference(Q, y, name, ("cluster",), k, caliper, block_sizes, n_boot, seed),
                 adjusted_difference(Q, y, name, None, block_sizes, n_boot, seed),
                 adjusted_difference(Q, y, name, passing, block_sizes, n_boot, seed), r]
    S = pd.DataFrame(rows)
    med = Q.groupby(["set", "boundary"])[["umi_hvg", "genes_hvg", "cells_hex"]].median()
    med_cl = Q.groupby(["cluster", "set"])[["umi_hvg", "cells_hex"]].median().unstack()
    if verbose:
        b = int(block_sizes[0])
        for r in S.itertuples():
            print(f"[14-4] {r.outcome:<10} {r.analysis:<36} core − scattered {r.difference:.3f} "
                  f"(95% CI {getattr(r, f'ci_low_{b}um'):.3f} to {getattr(r, f'ci_high_{b}um'):.3f}, {b} µm blocks)")
    return dict(summary=S, crosstab=pd.concat(tabs, ignore_index=True), medians=med, medians_by_cluster=med_cl)
