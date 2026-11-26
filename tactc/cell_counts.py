"""
cell_counts.py
==============

Step 1 of TACTC.ipynb: per-spot RNA shares of each group (spOT-NMF programs, named and merged) + DAPI cell counts
→ cell counts per group per spot.

  a. RNA share   q(s,g): program usages merged into groups (rows sum to 1). A usage is a share of the spot's
                 transcripts, so groups with more RNA per cell are overrepresented.
  b. correction  f_g = relative cells per unit of RNA (= 1 / relative RNA per cell). Without a reference it is estimated
                 from the spots themselves (rna_correction='fit', method of visiumif2niche spot_counts):
                 N_s ≈ Σ_g q(s,g)·UMI_s·c0·f_g over spots with ≥ min_cells cells, bounded least squares with a ridge
                 pulling f_g toward 1 (ridge larger = weaker correction), geometric mean of f fixed to 1.
                 Or set directly: {group: relative RNA per cell} (f = 1 / value), or False (f = 1).
  c. cell share  c(s,g) ∝ q(s,g)·f_g
  d. counts      E = N_s × c (fractional); integers = round(N_s) split by largest remainder
  e. interval    5–95% over n_draws, each draw = f refitted on a bootstrap resample of spots + a multinomial draw of
                 round(N_s) cells (uncertainty of the correction and of the integer split; not of the deconvolution)

Also: protein_check (cell share vs matching Visium protein), add_to_anndata (one spot AnnData with all step-1
outputs), frame (obsm → DataFrame), plot_maps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

__all__ = ["fit_rna_factor", "largest_remainder", "to_cell_counts", "CellCounts", "protein_check", "PROTEIN_RULES",
           "add_to_anndata", "frame", "plot_maps"]

# group → matching Visium proteins (tested only when a group of that exact name exists)
PROTEIN_RULES: Dict[str, List[str]] = {
    "T/NK": ["CD3E", "CD8A", "CD4"],
    "B/Plasma": ["MS4A1", "CD19", "PAX5", "SDC1"],
    "Macrophage/Mono": ["CD163", "CD68", "CD14", "ITGAM"],
    "Microglia": ["CD68", "ITGAM", "PTPRC"],
    "Endothelial": ["PECAM1"],
    "Mural": ["ACTA2"],
}


def fit_rna_factor(Q: np.ndarray, umi: np.ndarray, N: np.ndarray, ridge: float = 0.1,
                   bounds=(0.2, 5.0)) -> np.ndarray:
    """Solve N_s ≈ Σ_g Q[s,g]·umi_s·c0·f_g for f_g (ridge toward 1); geometric mean of f weighted by RNA share = 1.
    Same as visiumif2niche spot_counts._fit_rna."""
    from scipy.optimize import lsq_linear

    c0 = N.sum() / umi.sum()
    A = Q * (umi * c0)[:, None]
    G = A.shape[1]
    lam = np.sqrt(ridge * (A ** 2).sum(axis=0) / max(1, G))
    A2 = np.vstack([A, np.diag(lam)])
    b2 = np.r_[N, lam]
    f = lsq_linear(A2, b2, bounds=bounds).x
    return f / np.exp(np.average(np.log(f), weights=Q.sum(axis=0) + 1e-9))


def largest_remainder(E: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Turn each row E[s] into non-negative integers summing to total[s] (keeping proportions, largest remainder)."""
    E = np.clip(np.nan_to_num(np.asarray(E, float)), 0, None)
    total = np.asarray(total).astype(int)
    s = E.sum(axis=1, keepdims=True)
    W = np.divide(E, s, out=np.full_like(E, 1.0 / E.shape[1]), where=s > 0) * total[:, None]
    base = np.floor(W).astype(int)
    rem = total - base.sum(axis=1)
    frac = W - base
    rank = np.argsort(-frac, axis=1, kind="stable")
    add = np.zeros_like(base)
    cols = np.arange(E.shape[1])[None, :]
    add[np.arange(len(E))[:, None], rank] = (cols < rem[:, None]).astype(int)
    return base + add


@dataclass
class CellCounts:
    rna_share: pd.DataFrame        # spot × group, share of transcripts (named programs)
    cell_frac: pd.DataFrame        # spot × group, share of cells (after the RNA-per-cell correction)
    expected: pd.DataFrame         # spot × group, N_s × cell_frac (fractional)
    counts: pd.DataFrame           # spot × group, integers summing to round(N_s)
    lo: pd.DataFrame               # 5% over draws
    hi: pd.DataFrame               # 95% over draws
    n_cells: pd.Series             # spot → DAPI cell count used (fractional)
    rna_factor: pd.Series          # group → f_g (relative cells per unit RNA; 1 / relative RNA per cell)
    params: Dict = field(default_factory=dict)

    _TABLES = ("rna_share", "cell_frac", "expected", "counts", "lo", "hi")

    @property
    def groups(self) -> List[str]:
        return list(self.counts.columns)

    def summary(self) -> pd.DataFrame:
        """Per group: total cells, share of all cells, spots with ≥ 1 cell, mean RNA vs cell share, relative RNA per cell."""
        c = self.counts
        return pd.DataFrame({
            "cells": c.sum().astype(int),
            "cell_share": (c.sum() / max(float(c.to_numpy().sum()), 1.0)).round(4),
            "spots_with_cells": (c > 0).sum(),
            "mean_rna_share": self.rna_share.mean().round(4),
            "mean_cell_share": self.cell_frac.mean().round(4),
            "rel_rna_per_cell": (1.0 / self.rna_factor).round(3),
        }).sort_values("cells", ascending=False)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        for t in self._TABLES:
            getattr(self, t).to_csv(os.path.join(out_dir, f"{t}.csv"))
        self.n_cells.rename("n_cells").to_csv(os.path.join(out_dir, "n_cells.csv"))
        self.rna_factor.rename("rna_factor").to_csv(os.path.join(out_dir, "rna_factor.csv"))
        json.dump(self.params, open(os.path.join(out_dir, "cell_counts_params.json"), "w"), indent=1, default=str)
        self.summary().to_csv(os.path.join(out_dir, "summary.csv"))
        print(f"[save] cell counts per group → {out_dir} (counts.csv, expected.csv, lo/hi.csv, cell_frac.csv, …)")

    @classmethod
    def load(cls, out_dir: str) -> "CellCounts":
        t = {k: pd.read_csv(os.path.join(out_dir, f"{k}.csv"), index_col=0) for k in cls._TABLES}
        n = pd.read_csv(os.path.join(out_dir, "n_cells.csv"), index_col=0).iloc[:, 0]
        f = pd.read_csv(os.path.join(out_dir, "rna_factor.csv"), index_col=0).iloc[:, 0]
        p = json.load(open(os.path.join(out_dir, "cell_counts_params.json")))
        return cls(n_cells=n, rna_factor=f, params=p, **t)


def to_cell_counts(shares: pd.DataFrame, n_cells: pd.Series, umi: pd.Series,
                   rna_correction: Union[str, bool, Dict[str, float]] = "fit", ridge: float = 0.1,
                   min_cells: float = 2.0, n_draws: int = 200, seed: int = 0, verbose: bool = True) -> CellCounts:
    """Group RNA shares + DAPI cell counts → cell counts per group (see the module docstring).

    shares         : spot × group RNA shares (programs.name_topics output). Spots missing here get 0 cells.
    n_cells        : spot → DAPI cell count of the capture circle (dapi_counts.SpotCells.counts)
    umi            : spot → total UMI (vis.obs['total_counts'])
    rna_correction : 'fit' | False | {group: relative RNA per cell}
    """
    spots = pd.Index(n_cells.index)
    Q = shares.reindex(spots).astype(float)
    groups = list(Q.columns)
    ok = Q.notna().all(axis=1) & (Q.sum(axis=1) > 0)
    Qv = Q.where(ok, 0.0).to_numpy()
    Qv = np.divide(Qv, Qv.sum(axis=1, keepdims=True), out=np.zeros_like(Qv), where=Qv.sum(axis=1, keepdims=True) > 0)
    N = n_cells.astype(float).clip(lower=0).to_numpy()
    U = umi.reindex(spots).astype(float).fillna(0).to_numpy()
    fit_idx = np.flatnonzero(ok.to_numpy() & (N >= min_cells) & (U > 0))

    if isinstance(rna_correction, dict):
        mode = "fixed"
        r = np.array([float(rna_correction.get(g, 1.0)) for g in groups])
        fit = lambda idx: 1.0 / r
    elif rna_correction == "fit":
        mode = "fit"
        if len(fit_idx) < 3 * len(groups):
            raise ValueError(f"Too few spots for the RNA correction ({len(fit_idx)}). Use rna_correction=False.")
        fit = lambda idx: fit_rna_factor(Qv[idx], U[idx], N[idx], ridge)
    elif rna_correction is False or rna_correction is None:
        mode = "off"
        fit = lambda idx: np.ones(len(groups))
    else:
        raise ValueError("rna_correction must be 'fit', False, or {group: relative RNA per cell}")

    def compose(fv):
        W = Qv * fv
        s = W.sum(axis=1, keepdims=True)
        return np.divide(W, s, out=np.zeros_like(W), where=s > 0)

    f = fit(fit_idx)
    C = compose(f)
    E = C * N[:, None]
    tot = np.where(ok.to_numpy(), np.rint(N), 0).astype(int)
    counts = largest_remainder(E, tot)

    rng = np.random.default_rng(seed)
    D = np.zeros((n_draws, len(spots), len(groups)), dtype=np.int32)
    F = np.empty((n_draws, len(groups)))
    for d in range(n_draws):
        fd = fit(rng.choice(fit_idx, len(fit_idx), replace=True)) if mode == "fit" else f
        F[d] = fd
        Cd = compose(fd)
        idx = np.flatnonzero(tot > 0)
        if len(idx):
            Pd = Cd[idx] / Cd[idx].sum(axis=1, keepdims=True)
            D[d, idx] = rng.multinomial(tot[idx], Pd)
    lo, hi = np.percentile(D, [5, 95], axis=0, method="nearest")

    mk = lambda A: pd.DataFrame(A, index=spots, columns=groups)
    res = CellCounts(rna_share=mk(np.where(ok.to_numpy()[:, None], Qv, np.nan)),
                     cell_frac=mk(np.where(ok.to_numpy()[:, None], C, np.nan)), expected=mk(E), counts=mk(counts),
                     lo=mk(lo.astype(int)), hi=mk(hi.astype(int)), n_cells=pd.Series(N, index=spots),
                     rna_factor=pd.Series(f, index=groups, name="rna_factor"),
                     params=dict(rna_correction=mode if mode != "fixed" else {k: float(v) for k, v in
                                                                              rna_correction.items()},
                                 ridge=float(ridge), min_cells=float(min_cells), n_draws=int(n_draws), seed=int(seed),
                                 n_spots=int(len(spots)), n_fit_spots=int(len(fit_idx)),
                                 n_spots_without_shares=int((~ok).sum()),
                                 rna_factor_5_95={g: [float(np.percentile(F[:, j], 5)), float(np.percentile(F[:, j], 95))]
                                                  for j, g in enumerate(groups)}))
    if verbose:
        print(f"[counts] {int(counts.sum()):,} cells in {int((tot > 0).sum()):,} spots, {len(groups)} groups | "
              f"RNA correction {mode} ({len(fit_idx):,} spots)"
              + (f" | {int((~ok).sum())} spots without shares (0 cells)" if (~ok).any() else ""))
    return res


def _protein_norm(vis) -> Optional[pd.DataFrame]:
    if "protein" not in vis.obsm:
        return None
    p = vis.obsm["protein"]
    p = p[[c for c in p.columns if not str(c).lower().startswith(("mouse_", "rat_", "isotype"))]]
    return np.log1p(p.div(p.sum(axis=1).replace(0, 1), axis=0) * 1e4)


def protein_check(frac: pd.DataFrame, vis, rules: Optional[Dict[str, List[str]]] = None, min_rho: float = 0.1,
                  n_shift: int = 200, min_shift_um: float = 1000.0, seed: int = 0) -> pd.DataFrame:
    """Spearman ρ between each group's cell share and its matching Visium proteins (spot level, measured apart from
    RNA), against the shift null of visiumif2niche (protein map read at the spot nearest to the position shifted by
    ≥ min_shift_um; keeps both maps' spatial autocorrelation, breaks their positional correspondence).
    Uses the CytAssist antibody panel (obsm['protein']), not IF image channels. Only groups named as keys of rules are
    tested. For the best protein of each group: z = (ρ − null mean) / null sd, p_shift = one-sided (positive)
    empirical p. passes = ρ ≥ min_rho and p_shift ≤ 0.05. n_shift = 0 skips the null."""
    from scipy.stats import spearmanr

    from .node_graphs import shift_maps

    prot = _protein_norm(vis)
    if prot is None:
        raise ValueError("vis has no proteins (obsm['protein']).")
    rules = rules or PROTEIN_RULES
    idx = frac.dropna().index.intersection(prot.index)
    mpp = float(vis.uns.get("mpp", 1.0))
    xy = vis.obs.loc[idx, ["x", "y"]].to_numpy(float) * mpp
    shifts = shift_maps(xy, n_shift, min_shift_um, seed) if n_shift else []
    rows = []
    for g, ps in rules.items():
        ps = [p for p in ps if p in prot]
        if g not in frac or not ps:
            rows.append(dict(group=g, proteins=", ".join(ps), best_protein=None, rho=np.nan, all_rho="",
                             null_mean=np.nan, null_sd=np.nan, z_shift=np.nan, p_shift=np.nan, passes=None,
                             note="group not in result" if g not in frac else "no matching protein in the panel"))
            continue
        f = frac.loc[idx, g].to_numpy(float)
        rr = {p: float(spearmanr(f, prot.loc[idx, p].to_numpy(float)).statistic) for p in ps}
        best = max(rr, key=lambda k: -np.inf if np.isnan(rr[k]) else rr[k])
        pv = prot.loc[idx, best].to_numpy(float)
        nul = np.array([spearmanr(f[ii], pv[jj]).statistic for ii, jj in shifts], float)
        nul = nul[np.isfinite(nul)]
        if len(nul) > 2:
            m, sd = float(nul.mean()), float(nul.std())
            z = (rr[best] - m) / sd if sd > 0 else np.nan
            p = float((1 + (nul >= rr[best]).sum()) / (1 + len(nul)))
        else:
            m = sd = z = p = np.nan
        passes = bool(rr[best] >= min_rho and (p <= 0.05 if np.isfinite(p) else True))
        rows.append(dict(group=g, proteins=", ".join(ps), best_protein=best, rho=round(rr[best], 3),
                         all_rho=", ".join(f"{q} {v:.2f}" for q, v in rr.items()), null_mean=round(m, 3) if m == m else m,
                         null_sd=round(sd, 3) if sd == sd else sd, z_shift=round(z, 2) if z == z else z, p_shift=p,
                         passes=passes, note=f"{len(nul)} shifts" if len(nul) else "no shift null"))
    return pd.DataFrame(rows)


def add_to_anndata(vis, cc: CellCounts, lda=None, spot_cells=None, topic_names: Optional[Dict[str, str]] = None):
    """Copy of vis holding the step-1 outputs (one spot AnnData; slots follow the pipeline plan).

    obs  : n_dapi, n_dapi_centroid, has_shares, n_cells_assigned, dominant_group
    obsm : topic_theta (spot × program usage), rna_share, cell_frac, cell_expected, cell_counts, cell_counts_lo, cell_counts_hi
           (float arrays; columns in uns['step1']['topics'] / ['groups']; frame(adata, key) returns a DataFrame.
           Arrays rather than DataFrames because h5ad cannot store column names containing '/', e.g. 'T/NK')
    var  : in_corpus (gene used by spOT-NMF);  varm['topic_beta'] : gene × program (0 for genes not used)
    uns  : step1 (topics, groups, topic → group, K choice, parameters, RNA factors)
    """
    a = vis.copy()
    for df in (a.obs, a.var):                        # h5ad cannot store an index name that is also a different column
        if df.index.name is not None and df.index.name in df.columns:
            df.index.name = None
    for k in list(a.obsm.keys()):
        if isinstance(a.obsm[k], pd.DataFrame) and any("/" in str(c) for c in a.obsm[k].columns):
            a.obsm[k] = a.obsm[k].rename(columns=lambda c: str(c).replace("/", "_"))
    s = a.obs_names
    groups = [str(c) for c in cc.counts.columns]
    a.obs["n_dapi"] = cc.n_cells.reindex(s).to_numpy(float)
    if spot_cells is not None:
        a.obs["n_dapi_centroid"] = spot_cells.n_centroid.reindex(s).fillna(0).astype(int).to_numpy()
    a.obs["has_shares"] = cc.rna_share.reindex(s).notna().all(axis=1).to_numpy()
    C = cc.counts.reindex(s).fillna(0)
    a.obs["n_cells_assigned"] = C.sum(axis=1).astype(int).to_numpy()
    a.obs["dominant_group"] = pd.Categorical(np.where(C.sum(axis=1) > 0, C.idxmax(axis=1), "none"))
    tabs = dict(rna_share=cc.rna_share, cell_frac=cc.cell_frac, cell_expected=cc.expected, cell_counts=cc.counts,
                cell_counts_lo=cc.lo, cell_counts_hi=cc.hi)
    for k, df in tabs.items():
        d = df.reindex(index=s, columns=groups).astype(float)
        a.obsm[k] = (d.fillna(0.0) if k.startswith("cell_counts") else d).to_numpy()
    st = {"groups": groups, "group_obsm": list(tabs),
          "cell_counts_params": json.dumps(cc.params, default=str),
          "rel_rna_per_cell": [float(1.0 / cc.rna_factor.get(g, np.nan)) for g in groups]}
    if lda is not None:
        topics = [str(t) for t in lda.topics]
        a.obsm["topic_theta"] = lda.theta.reindex(index=s, columns=topics).astype(float).to_numpy()
        B = lda.beta.T.reindex(a.var_names).astype(float)
        a.var["in_corpus"] = B.notna().all(axis=1).to_numpy()
        a.varm["topic_beta"] = B.fillna(0.0)[topics].to_numpy()
        st.update(topics=topics, topic_to_group=[str((topic_names or {}).get(t, t)) for t in topics],
                  k=int(lda.k), k_choice=json.dumps(lda.k_choice), lda_params=json.dumps(lda.params, default=str))
    a.uns["step1"] = st
    return a


def frame(adata, key: str) -> pd.DataFrame:
    """obsm[key] of the step-1 AnnData as a spot × group (or × topic) DataFrame."""
    cols = adata.uns["step1"]["topics"] if key == "topic_theta" else adata.uns["step1"]["groups"]
    return pd.DataFrame(np.asarray(adata.obsm[key]), index=adata.obs_names, columns=list(cols))


def plot_maps(df: pd.DataFrame, vis, ncols: int = 4, size: float = 4.0, cmap: str = "magma", title: str = ""):
    """Per-column spot maps (x, y in µm)."""
    import matplotlib.pyplot as plt

    cols = list(df.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    mpp = float(vis.uns.get("mpp", 1.0))
    d = df.reindex(vis.obs_names)
    x, y = vis.obs["x"].to_numpy(float) * mpp, vis.obs["y"].to_numpy(float) * mpp
    for ax, c in zip(axes.ravel(), cols):
        v = d[c].to_numpy(float)
        vmax = np.nanquantile(v, 0.99) if np.isfinite(v).any() else 1.0
        im = ax.scatter(x, y, c=v, s=size, cmap=cmap, vmin=0, vmax=vmax if vmax > 0 else 1.0, linewidths=0)
        ax.set_title(str(c), fontsize=9)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.6)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
