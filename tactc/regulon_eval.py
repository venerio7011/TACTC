"""
regulon_eval.py
===============

Used by TACTC.ipynb 13-3 (dapi_support.regulon_support): where are the step-12 regulons region-specific, and which of
those calls survive three exclusion criteria?

exclusion(grn, nodes, labels, xy, …) per (regulon, region) pair
  C1 spatial     : d = (mean activity inside − outside) / SD, against the shift null of the label map
                   (|z| and BH q over pairs)
  C2 composition : the same on activity residualised on the node's group composition (residual_activity)
  C3 replication : the sign of d is kept in both halves of random straight cuts
  naive          : the same contrast with an ordinary test and no null (for comparison)
calibration: the |z| rate with shifted label maps as pseudo-regions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .node_graphs import _sep_z, random_cuts, shift_maps

__all__ = ["region_scores", "calibration", "residual_activity", "exclusion"]

Z_THR = 3.0


# ---------------------------------------------------------------------------
# region specificity of the regulons for one labelling
# ---------------------------------------------------------------------------
def _d(X: np.ndarray, M: np.ndarray, sd: np.ndarray, min_in: int) -> np.ndarray:
    """regions × regulons: (mean inside − mean outside) / sd. X units × regulons, M units × regions (0/1)."""
    n_in = M.sum(axis=0)
    s_in = M.T @ X
    m_in = s_in / np.maximum(n_in, 1)[:, None]
    m_out = (X.sum(axis=0)[None, :] - s_in) / np.maximum(len(X) - n_in, 1)[:, None]
    d = (m_in - m_out) / sd[None, :]
    d[(n_in < min_in) | (len(X) - n_in < min_in)] = np.nan
    return d


def _onehot(lab: np.ndarray, regions: Sequence[str]) -> np.ndarray:
    return np.column_stack([lab == r for r in regions]).astype(float)


def _activity(grn, names: Sequence[str]):
    """group → (unit positions in `names`, units × regulons array, regulon names); units outside `names` dropped."""
    pos = pd.Index(names)
    out = {}
    for g, A in grn.activity.items():
        if A.shape[1] == 0:
            continue
        idx = pos.get_indexer(A.index)
        ok = idx >= 0
        out[g] = (idx[ok], A.to_numpy(float)[ok], list(A.columns))
    return out


def region_scores(grn, labels: pd.Series, xy: np.ndarray, n_shift: int = 200, min_shift_um: float = 1000.0,
                  min_units_in: int = 5, seed: int = 0, shifts=None, min_null_frac: float = 0.5) -> pd.DataFrame:
    """Long table group, regulon, region, d, z (shift null of the label map). labels: node-indexed (order of xy).
    z is left NaN when fewer than max(20, min_null_frac × shifts) shifts give a valid value for that region."""
    names = list(labels.index)
    lab = labels.astype(str).to_numpy(dtype=object)
    regions = sorted(pd.unique(lab[lab != ""]), key=lambda x: (len(x), x))
    shifts = shifts if shifts is not None else shift_maps(xy, n_shift, min_shift_um, seed)
    shifted = []
    for ii, jj in shifts:
        ls = np.full(len(lab), "", dtype=object)
        ls[ii] = lab[jj]
        shifted.append(ls)
    rows = []
    for g, (idx, X, regs) in _activity(grn, names).items():
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, np.nan)
        D = _d(X, _onehot(lab[idx], regions), sd, min_units_in)
        N = np.array([_d(X, _onehot(ls[idx], regions), sd, min_units_in) for ls in shifted])
        with np.errstate(invalid="ignore", divide="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                Z = (D - np.nanmean(N, axis=0)) / np.nanstd(N, axis=0) if len(N) > 2 else np.full_like(D, np.nan)
        if len(N):          # a region that is empty / too small in most shifts has an unreliable null sd → no z
            Z[np.isfinite(N).sum(axis=0) < max(20, int(min_null_frac * len(N)))] = np.nan
        for ri, r in enumerate(regions):
            for ki, k in enumerate(regs):
                rows.append((g, k, r, D[ri, ki], Z[ri, ki]))
    return pd.DataFrame(rows, columns=["group", "regulon", "region", "d", "z"])


# ---------------------------------------------------------------------------
# S1 / S2 / S3 / S5
# ---------------------------------------------------------------------------
def calibration(grn, labels: pd.Series, xy: np.ndarray, n_maps: int = 10, n_shift: int = 200,
                min_shift_um: float = 1000.0, z_thr: float = Z_THR, min_units_in: int = 5, seed: int = 0):
    """S3: |z| ≥ z_thr rate when shifted copies of the label map are used as pseudo-regions.
    Returns (rate, threshold): threshold = z_thr, or the 99.73 % quantile of the pseudo-region |z| if that is larger
    (the rate a normal z would give at 3), so that S1 / S2 / S4 keep a ≈ 0.27 % false rate."""
    lab = labels.astype(str).to_numpy(dtype=object)
    pseudo = shift_maps(xy, n_maps, min_shift_um, seed + 1000)
    shifts = shift_maps(xy, n_shift, min_shift_um, seed)
    zs = []
    for ii, jj in pseudo:
        ls = np.full(len(lab), "", dtype=object)
        ls[ii] = lab[jj]
        sc = region_scores(grn, pd.Series(ls, index=labels.index), xy, min_units_in=min_units_in, shifts=shifts)
        sc = sc[sc["region"] != ""]
        z = sc["z"].to_numpy(float)
        zs.append(np.abs(z[np.isfinite(z)]))
    z = np.concatenate(zs) if zs else np.array([])
    if not len(z):
        return np.nan, z_thr
    return float((z >= z_thr).mean()), float(max(z_thr, np.quantile(z, 0.9973)))


# ---------------------------------------------------------------------------
# Step 16 (paper claim ②): excluding niche regulons without spatial meaning, standard PAGA vs TACTC
# (earlier version; not run by TACTC.ipynb)
#   C1 spatial   : shift-null z → normal p → BH over all (regulon, region) of the branch; q ≤ α and |z| ≥ the
#                  pseudo-region calibrated threshold (max(3, 99.73 % quantile of |z| over shifted label maps))
#   C2 not composition : the same on the residual of each regulon's activity after OLS on node composition (all groups),
#                  log total UMI and log cells of the group; same sign as C1
#   C3 replicated: over random straight cuts, both halves' d have the sign of the whole-tissue d in ≥ 80 % of the cuts
#                  where both are computable (≥ 5 such cuts)
#   kept = C1 ∧ C2 ∧ C3; reason = first failed condition. Without-null rule: Welch t inside vs outside, BH.
# ---------------------------------------------------------------------------
def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]
        o = np.argsort(v)
        r = v[o] * len(v) / (np.arange(len(v)) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty_like(v)
        qq[o] = np.clip(r, 0, 1)
        q[ok] = qq
    return q


class _Act:
    """Minimal stand-in for GRNResult with other activity tables (same regulons)."""

    def __init__(self, activity, regulons):
        self.activity, self.regulons = activity, regulons


def residual_activity(grn, nodes):
    """Per group: regulon activity minus its OLS fit on node composition (all groups' cell fractions), log total UMI
    and log cells of the group (C2)."""
    names = pd.Index(nodes.obs_names)
    groups = list(nodes.uns["nodes"]["groups"])
    Ch = np.asarray(nodes.obsm["cell_counts_hex"], float)
    F = Ch / np.maximum(Ch.sum(axis=1, keepdims=True), 1e-12)
    cnt = nodes.layers["counts"]
    umi = np.log1p(np.asarray(cnt.sum(axis=1)).ravel())
    out = {}
    for g, A in grn.activity.items():
        if A.shape[1] == 0:
            out[g] = A
            continue
        idx = names.get_indexer(A.index)
        gi = groups.index(g) if g in groups else None
        cols = [np.ones(len(idx)), umi[idx]] + [F[idx, j] for j in range(F.shape[1] - 1)]
        if gi is not None:
            cols.append(np.log1p(Ch[idx, gi]))
        X = np.column_stack(cols)
        Y = A.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        out[g] = pd.DataFrame(Y - X @ beta, index=A.index, columns=A.columns)
    return _Act(out, grn.regulons)


def _half_signs(grn, labels: pd.Series, xy: np.ndarray, cuts: List[dict], min_units_in: int):
    """(group, regulon, region) → list over cuts of (d half 1, d half 2)."""
    names = list(labels.index)
    lab = labels.astype(str).to_numpy(dtype=object)
    regions = sorted(pd.unique(lab), key=lambda x: (len(x), x))
    out = {}
    for g, (idx, X, regs) in _activity(grn, names).items():
        D = []
        for c in cuts:
            side = (xy @ c["u"] < c["t"])[idx]
            pair = []
            for h in (side, ~side):
                Xh = X[h]
                sd = Xh.std(axis=0)
                sd = np.where(sd > 0, sd, np.nan)
                pair.append(_d(Xh, _onehot(lab[idx][h], regions), sd, min_units_in))
            D.append(pair)
        for ri, r in enumerate(regions):
            for ki, k in enumerate(regs):
                out[(g, k, r)] = [(p[0][ri, ki], p[1][ri, ki]) for p in D]
    return out


def _naive(grn, labels: pd.Series, min_units_in: int) -> pd.DataFrame:
    from scipy.stats import ttest_ind

    names = list(labels.index)
    lab = labels.astype(str).to_numpy(dtype=object)
    regions = sorted(pd.unique(lab), key=lambda x: (len(x), x))
    rows = []
    for g, (idx, X, regs) in _activity(grn, names).items():
        for r in regions:
            m = lab[idx] == r
            if m.sum() < min_units_in or (~m).sum() < min_units_in:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                t, p = ttest_ind(X[m], X[~m], axis=0, equal_var=False)
            for ki, k in enumerate(regs):
                rows.append((g, k, r, float(p[ki])))
    df = pd.DataFrame(rows, columns=["group", "regulon", "region", "naive_p"])
    df["naive_q"] = _bh(df["naive_p"].to_numpy())
    return df


def exclusion(grn, nodes, labels: pd.Series, xy: np.ndarray, n_shift: int = 200, min_shift_um: float = 1000.0,
              n_maps: int = 10, n_cut: int = 10, alpha: float = 0.05, rep_min: float = 0.8, min_cuts: int = 5,
              min_units_in: int = 5, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """C1–C3 for every (regulon, region) of one region definition (labels: node-indexed, order of xy)."""
    from scipy.stats import norm

    shifts = shift_maps(xy, n_shift, min_shift_um, seed)
    sc = region_scores(grn, labels, xy, min_units_in=min_units_in, shifts=shifts)
    _, thr = calibration(grn, labels, xy, n_maps, n_shift, min_shift_um, Z_THR, min_units_in, seed)
    sc["q"] = _bh(2 * norm.sf(np.abs(sc["z"].to_numpy(float))))
    sc["thr"] = thr
    sc["C1"] = (sc["q"] <= alpha) & (np.abs(sc["z"]) >= thr)
    res = residual_activity(grn, nodes)
    sa = region_scores(res, labels, xy, min_units_in=min_units_in, shifts=shifts)
    _, thr_a = calibration(res, labels, xy, n_maps, n_shift, min_shift_um, Z_THR, min_units_in, seed)
    sa["q_adj"] = _bh(2 * norm.sf(np.abs(sa["z"].to_numpy(float))))
    sc = sc.merge(sa.rename(columns={"d": "d_adj", "z": "z_adj"}), on=["group", "regulon", "region"], how="left")
    sc["thr_adj"] = thr_a
    sc["C2"] = (sc["q_adj"] <= alpha) & (np.abs(sc["z_adj"]) >= thr_a) & (np.sign(sc["z_adj"]) == np.sign(sc["z"]))
    halves = _half_signs(grn, labels, xy, random_cuts(xy, n_cut, seed), min_units_in)
    rep, ncut = [], []
    for g, k, r, d in sc[["group", "regulon", "region", "d"]].itertuples(index=False):
        pr = [(a, b) for a, b in halves.get((g, k, r), []) if np.isfinite(a) and np.isfinite(b)]
        ncut.append(len(pr))
        rep.append(np.mean([(np.sign(a) == np.sign(d)) and (np.sign(b) == np.sign(d)) for a, b in pr]) if pr else np.nan)
    sc["rep_frac"], sc["n_cuts"] = rep, ncut
    sc["C3"] = (sc["n_cuts"] >= min_cuts) & (sc["rep_frac"] >= rep_min)
    sc["kept"] = sc["C1"] & sc["C2"] & sc["C3"]
    sc["reason"] = np.where(sc["kept"], "kept", np.where(~sc["C1"], "C1 not spatial",
                            np.where(~sc["C2"], "C2 composition", "C3 not replicated")))
    sc = sc.merge(_naive(grn, labels, min_units_in), on=["group", "regulon", "region"], how="left")
    sc["naive_call"] = sc["naive_q"] <= alpha
    if verbose:
        print(f"[16] {labels.name}: {len(sc):,} (regulon, region) pairs | without null {int(sc['naive_call'].sum())} | "
              f"C1 {int(sc['C1'].sum())} → C1∧C2 {int((sc['C1'] & sc['C2']).sum())} → kept {int(sc['kept'].sum())} "
              f"(thresholds |z| {thr:.2f}, residual {thr_a:.2f})")
    return sc


