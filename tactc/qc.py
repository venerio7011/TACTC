"""
qc.py
=====

Visium spot / gene quality control (TACTC.ipynb 1-1; copied unchanged from the visiumif2niche qc.py). Usable for any tissue type.

Spot metrics: UMI count (excluding MT), detected genes, MT fraction, (if present) protein isotype-control fraction, whether at the tissue edge.

Two criteria are kept apart.
  * global: deviates from the tissue-wide median by ≥ nmads × MAD.
    Broad areas of low UMI may be tissue with little RNA (normal brain, around necrosis, sparse cells),
    so this alone does not remove spots; they are only flagged as 'regionally low (regional_low)'.
  * local: lower (MT: higher) than the median of surrounding spots (about 2 rings, ~18 spots) by ≥ nmads × MAD.
    Spots abruptly different from their surroundings are likely technical problems such as bubbles·folds·tears·edge damage → removal candidates.
  * absolute floor: UMI < min_counts or genes < min_genes carry too little information and are removed.
  MT is removed when high both globally and locally, or extremely high in the global distribution (2 × nmads × MAD; high spots gathered in
  a band, as with edge damage, escape the local criterion). In probe-based FFPE, MT fraction does not follow scRNA's 'dying cell'
  criterion, so a distribution-based criterion is used instead of a fixed threshold (e.g. 20%).
Genes: genes detected in fewer than min_spots spots are removed (spot total_counts keep their original values).

Contents: spot_qc → QCResult(apply, save, summary), plot_qc
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["spot_qc", "QCResult", "plot_qc", "DEFAULT_REMOVE"]

DEFAULT_REMOVE = ("too_few", "low_counts_local", "low_genes_local", "high_mt")


def _mad(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    m = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float(m) if m > 0 else float(np.std(v) or 1.0)


def _neighbors(xy: np.ndarray, rings: float):
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    spacing = float(np.median(d[:, 1]))
    nb = tree.query_ball_point(xy, rings * spacing * 1.05)
    nb = [np.array([j for j in n if j != i], dtype=np.int64) for i, n in enumerate(nb)]
    ring1 = np.array([int((np.linalg.norm(xy[n] - xy[i], axis=1) < 1.5 * spacing).sum()) for i, n in enumerate(nb)])
    return nb, ring1, spacing


def _local_median(v: np.ndarray, nb) -> np.ndarray:
    return np.array([np.median(v[n]) if len(n) >= 3 else np.nan for n in nb])


@dataclass
class QCResult:
    metrics: pd.DataFrame          # spot × metrics + verdict columns
    thresholds: Dict[str, float]
    keep_genes: pd.Index
    removed_genes: pd.Index
    params: Dict = field(default_factory=dict)

    @property
    def keep(self) -> pd.Series:
        return ~self.metrics["remove"]

    def summary(self) -> pd.DataFrame:
        m = self.metrics
        flags = [c for c in m.columns if m[c].dtype == bool]
        return pd.DataFrame({"spots": [int(m[c].sum()) for c in flags],
                             "fraction": [round(float(m[c].mean()), 4) for c in flags]}, index=flags)

    def apply(self, vis, genes: bool = True):
        """Copy with spots (and genes if genes=True) removed according to the verdicts. QC metrics are attached to obs."""
        out = vis[self.keep.reindex(vis.obs_names).fillna(False).to_numpy()].copy()
        if genes:
            out = out[:, out.var_names.isin(self.keep_genes)].copy()
        cols = ["log10_counts", "n_genes", "mt_frac", "regional_low", "edge"]
        for c in cols:
            if c in self.metrics:
                out.obs[f"qc_{c}"] = self.metrics.loc[out.obs_names, c].to_numpy()
        out.uns["qc"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                         for k, v in dict(self.thresholds, **self.params).items()}
        return out

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.metrics.to_csv(os.path.join(out_dir, "qc_spots.csv"))
        pd.Series(self.removed_genes, name="gene").to_csv(os.path.join(out_dir, "qc_removed_genes.csv"), index=False)
        json.dump(dict(thresholds=self.thresholds, params=self.params,
                       summary=self.summary()["spots"].to_dict()),
                  open(os.path.join(out_dir, "qc_summary.json"), "w"), indent=1, default=str)
        print(f"[save] QC → {out_dir}/qc_spots.csv, qc_removed_genes.csv, qc_summary.json")


def spot_qc(vis, nmads: float = 3.0, min_counts: float = 500, min_genes: float = 250, min_spots: int = 10,
            rings: float = 2.0, remove: Sequence[str] = DEFAULT_REMOVE, verbose: bool = True) -> QCResult:
    """Spot / gene QC metrics and verdicts.

    vis        : output of cell2loc.load_visium (obs: x, y, total_counts; obsm: MT, protein)
    nmads      : how many MADs from the median count as abnormal (log scale)
    min_counts, min_genes : absolute floors (below these there is almost no information)
    min_spots  : minimum number of spots a gene must be detected in to be kept
    rings      : neighbourhood range for local comparison (multiples of spot spacing; 2 = about 18 spots)
    remove     : verdicts used for removal. Default = absolute floor, locally low UMI·genes, high MT (both global and local)
                 adding e.g. 'low_counts_global' also removes regionally low spots (not recommended)
    """
    X = vis.X
    counts = vis.obs["total_counts"].to_numpy(float) if "total_counts" in vis.obs else np.asarray(X.sum(1)).ravel()
    n_genes = np.asarray((X > 0).sum(1)).ravel().astype(float)
    m = pd.DataFrame({"total_counts": counts, "n_genes": n_genes}, index=vis.obs_names)
    m["log10_counts"] = np.log10(np.maximum(counts, 1))
    m["log10_genes"] = np.log10(np.maximum(n_genes, 1))
    has_mt = "MT" in vis.obsm and np.asarray(vis.obsm["MT"]).shape[1] > 0
    if has_mt:
        mt = np.asarray(vis.obsm["MT"], dtype=float).sum(axis=1)
        m["mt_frac"] = mt / np.maximum(mt + counts, 1)
        lm = np.log(np.clip(m["mt_frac"], 1e-4, 1 - 1e-4) / (1 - np.clip(m["mt_frac"], 1e-4, 1 - 1e-4)))
    else:
        m["mt_frac"] = np.nan
    prot = vis.obsm.get("protein") if hasattr(vis.obsm, "get") else None
    if prot is not None:
        iso = [c for c in prot.columns if "igg" in c.lower()]
        if iso:
            ptot = prot.sum(axis=1).to_numpy(float)
            m["isotype_frac"] = prot[iso].sum(axis=1).to_numpy(float) / np.maximum(ptot, 1)

    xy = vis.obs[["x", "y"]].to_numpy(float)
    nb, ring1, spacing = _neighbors(xy, rings)
    m["n_neighbors"] = ring1
    m["edge"] = ring1 < 4

    th = {}

    def flag(col, v, lower: bool):
        med, mad = float(np.median(v[np.isfinite(v)])), _mad(v)
        loc = _local_median(v, nb)
        if lower:
            th[f"{col}_global_min"] = med - nmads * mad
            g = v < med - nmads * mad
            l_ = (v - loc) < -nmads * mad
        else:
            th[f"{col}_global_max"] = med + nmads * mad
            g = v > med + nmads * mad
            l_ = (v - loc) > nmads * mad
        return g, l_                                  # fewer than 3 neighbours → loc = NaN → comparison is False

    g, l_ = flag("log10_counts", m["log10_counts"].to_numpy(), True)
    m["low_counts_global"], m["low_counts_local"] = g, l_
    g, l_ = flag("log10_genes", m["log10_genes"].to_numpy(), True)
    m["low_genes_global"], m["low_genes_local"] = g, l_
    if has_mt:
        g, l_ = flag("logit_mt", lm.to_numpy(), False)
        m["high_mt_global"], m["high_mt_local"] = g, l_
        v = lm.to_numpy()
        ext = float(np.median(v) + 2 * nmads * _mad(v))
        m["high_mt_extreme"] = v > ext
        m["high_mt"] = (g & l_) | m["high_mt_extreme"].to_numpy()
        th["mt_frac_global_max"] = float(1 / (1 + np.exp(-th["logit_mt_global_max"])))
        th["mt_frac_extreme"] = float(1 / (1 + np.exp(-ext)))
    else:
        m["high_mt"] = False
    m["too_few"] = (counts < min_counts) | (n_genes < min_genes)
    th["min_counts"], th["min_genes"] = float(min_counts), float(min_genes)

    rm = np.zeros(len(m), bool)
    for c in remove:
        if c not in m:
            raise KeyError(f"Unknown verdict: {c}. Possible values: {[c_ for c_ in m.columns if m[c_].dtype == bool]}")
        rm |= m[c].to_numpy(bool)
    m["remove"] = rm
    m["regional_low"] = (m["low_counts_global"] | m["low_genes_global"]) & ~rm

    det = np.asarray((X > 0).sum(0)).ravel()
    keep_genes = vis.var_names[det >= min_spots]
    removed_genes = vis.var_names[det < min_spots]
    res = QCResult(metrics=m, thresholds={k: round(float(v), 4) for k, v in th.items()}, keep_genes=keep_genes,
                   removed_genes=removed_genes,
                   params=dict(nmads=nmads, min_spots=min_spots, rings=rings, remove=list(remove),
                               spacing_px=round(spacing, 1), has_mt=has_mt))
    if verbose:
        q = lambda c: np.round(np.quantile(m[c].dropna(), [0.05, 0.5, 0.95]), 3).tolist()
        print(f"[qc] {len(m):,} spots | UMI 5/50/95% {q('total_counts')} | genes {q('n_genes')}"
              + (f" | MT fraction {q('mt_frac')}" if has_mt else " | no MT genes (MT verdict skipped)"))
        why = ", ".join(f"{c} {int(m[c].sum())}" for c in remove)
        print(f"[qc] removed {int(rm.sum())} spots ({why}; with overlaps) | regionally low but kept (regional_low) "
              f"{int(m['regional_low'].sum())} | edge {int(m['edge'].sum())} | "
              f"{len(removed_genes)} genes removed (detected in < {min_spots} spots)")
    return res


def plot_qc(qc: QCResult, vis, figsize=(16, 9)):
    """Top: UMI·gene·MT maps. Bottom: verdict map (red = removed, orange = regionally low but kept, grey = pass) and distributions."""
    import matplotlib.pyplot as plt

    m = qc.metrics.reindex(vis.obs_names)
    xy = vis.obs[["x", "y"]].to_numpy(float)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for ax, c, t in zip(axes[0], ["log10_counts", "n_genes", "mt_frac"], ["log10 UMI", "genes detected", "MT fraction"]):
        v = m[c].to_numpy(float)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=3, cmap="viridis", linewidths=0,
                        vmin=np.nanquantile(v, 0.01) if np.isfinite(v).any() else None,
                        vmax=np.nanquantile(v, 0.99) if np.isfinite(v).any() else None)
        ax.set_title(t); fig.colorbar(sc, ax=ax, shrink=0.7)
    ax = axes[1][0]
    col = np.where(m["remove"], "crimson", np.where(m["regional_low"], "orange", "0.8"))
    ax.scatter(xy[:, 0], xy[:, 1], c=col, s=3, linewidths=0)
    ax.set_title(f"removed {int(m['remove'].sum())} (red) · regional-low kept {int(m['regional_low'].sum())} (orange)")
    for ax in list(axes[0]) + [axes[1][0]]:
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1][1]
    ax.hist(m["log10_counts"], bins=60, color="0.6")
    ax.axvline(qc.thresholds["log10_counts_global_min"], color="orange", ls="--", label="global −nMAD")
    ax.axvline(np.log10(qc.thresholds["min_counts"]), color="crimson", label="min_counts")
    ax.set_xlabel("log10 UMI"); ax.legend(fontsize=7)
    ax = axes[1][2]
    if m["mt_frac"].notna().any():
        ax.scatter(m["log10_counts"], m["mt_frac"], s=2, c=np.where(m["remove"], "crimson", "0.5"))
        ax.axhline(qc.thresholds.get("mt_frac_global_max", np.nan), color="orange", ls="--", label="global (flag)")
        ax.axhline(qc.thresholds.get("mt_frac_extreme", np.nan), color="crimson", ls="--", label="extreme (remove)")
        ax.legend(fontsize=7)
        ax.set_xlabel("log10 UMI"); ax.set_ylabel("MT fraction")
    else:
        ax.axis("off")
    fig.tight_layout()
    return fig
