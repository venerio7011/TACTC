"""
transition.py
=============

TACTC.ipynb 14-3 (exploratory, not pre-registered): spot nodes that sit between two expression clusters — mixing or
a transition state?

A node whose expression lies between clusters A and B can be
  (a) a mixed spot: cells of A and cells of B in one 55 µm capture area (expected where A and B touch in the tissue);
  (b) a transition state: cells whose own expression is intermediate (possible far from any B tissue).
Spot RNA alone cannot separate the two. This module uses the tissue position and a count-level mixture null.

Steps (all on objects saved by steps 10, 11 and 13-4)
  1. Expression boundary: share of the node's expression-kNN weight (pre.obsp['<tag>_connectivities']) on other
     clusters ≥ min_mix; the partner B = the other cluster with the most weight.
  2. Path: A–B is a PAGA edge (connectivity ≥ paga_threshold) and the node's DPT pseudotime lies between the median
     pseudotimes of A's and B's core nodes. Nodes passing 1 and 2 are 'examined'.
  3. Mixture null (counts, HVGs): reference profiles = mean count proportions of each cluster's core nodes without
     a DAPI link to another cluster. Fit p_node ≈ Σ_c β_c profile_c (weighted NNLS over all clusters, gene weight
     1 / √mean proportion) → relative residual r. Null: the node rebuilt as a real mixture — counts of a random A
     reference node and a random B reference node (never the node itself), thinned to the node's depth with the
     node's fitted A : B share — refitted the same way (n_null times).
     p_mix = (1 + #null r ≥ observed r) / (1 + n_null). Small p_mix = not explained as a mixture of real A and B spots.
  4. Specificity check: the same test on expression-interior nodes (expr_mix < min_mix / 2, sample of n_interior)
     with a one-cluster null (rebuilt from a random reference node of its own cluster). These are not between
     clusters, so their rejection rate shows how often within-cluster heterogeneity alone rejects. Examined nodes
     must reject clearly more often (block bootstrap CI of the difference above 0) before any candidate is read.
  5. Location in the tissue (context, not a criterion): near = DAPI link to a partner node or one within near_um;
     far = no partner node within far_um; middle = otherwise. In a continuum the partner-labelled half of the
     gradient lies next to the node, so 'near' does not mean mixing (synthetic check, 2026-11-25).

Candidates = examined and BH q_mix ≤ q_max (over examined nodes). Exploratory: the criteria were fixed before this
analysis was run on the real data (2026-11-25) but after the step 11–13 results were seen.
Limits: a transition state whose profile is an exact linear blend of A and B cannot be told from mixing by RNA.
Spot data cannot place the state in single cells.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.spatial import cKDTree

__all__ = ["transition_nodes"]


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    v = p[ok]
    o = np.argsort(v)
    r = v[o] * len(v) / np.arange(1, len(v) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(v)
    out[o] = np.minimum(r, 1.0)
    q[ok] = out
    return q


def _fit(p: np.ndarray, WM: np.ndarray, w: np.ndarray):
    """Weighted NNLS of one proportion vector on the reference profiles → (beta, relative residual)."""
    b, res = nnls(WM, w * p)
    return b, float(res / max(np.linalg.norm(w * p), 1e-12))


def _thin(rng, y: np.ndarray, prob: float) -> np.ndarray:
    if prob <= 1.0:
        return rng.binomial(y.astype(np.int64), prob)
    return rng.poisson(y * prob)


def _block_ids(xy: np.ndarray, block_um: float) -> np.ndarray:
    b = np.floor(xy / block_um).astype(np.int64)
    return np.unique(b[:, 0] * 100003 + b[:, 1], return_inverse=True)[1]


def transition_nodes(pre, pseudotime: np.ndarray, tag: str = "std", label_key: Optional[str] = None,
                     set_key: str = "support_set", dapi_key: str = "dapi_links", paga: Optional[pd.DataFrame] = None,
                     paga_threshold: float = 0.03, min_mix: float = 0.3, near_um: float = 150.0,
                     far_um: float = 300.0, n_null: int = 200, n_interior: int = 300, alpha: float = 0.05,
                     q_max: float = 0.1,
                     block_um: float = 500.0, n_boot: int = 500, seed: int = 0,
                     verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Returns dict(nodes, pairs, test). `pre` = support_step11.h5ad (steps 10–11); `pseudotime` = DPT on all nodes
    (13-4 pt_all), aligned with pre.obs_names."""
    rng = np.random.default_rng(seed)
    label_key = label_key or f"leiden_{tag}"
    lab = pre.obs[label_key].astype(str).to_numpy()
    cats = sorted(set(lab), key=lambda c: (len(c), c))
    code = pd.Index(cats).get_indexer(lab)
    n, K = len(lab), len(cats)
    sets = pre.obs[set_key].astype(str).to_numpy()
    xy = np.asarray(pre.obsm["spatial"], float)
    pt = np.asarray(pseudotime, float)
    if len(pt) != n:
        raise ValueError("pseudotime must have one value per node of pre.")
    if paga is None:
        P = _dense(pre.uns[f"paga_{tag}"]["connectivities"])
        pcats = list(pre.obs[label_key].cat.categories) if hasattr(pre.obs[label_key], "cat") else cats
        paga = pd.DataFrame(P, index=[str(c) for c in pcats], columns=[str(c) for c in pcats])
    paga = paga.copy()
    paga.index = paga.index.astype(str)
    paga.columns = paga.columns.astype(str)
    PG = paga.loc[cats, cats].to_numpy(float)

    # 1. expression boundary
    C = sparse.csr_matrix(pre.obsp[f"{tag}_connectivities"])
    Cc = C.tocoo()
    Wk = np.zeros((n, K))
    np.add.at(Wk, (Cc.row, code[Cc.col]), Cc.data)
    tot = Wk.sum(1)
    own_w = Wk[np.arange(n), code]
    mix = np.where(tot > 0, 1 - own_w / np.maximum(tot, 1e-12), 0.0)
    Wo = Wk.copy()
    Wo[np.arange(n), code] = -1
    partner = Wo.argmax(1)
    expr_boundary = mix >= min_mix

    # 2. path: PAGA edge and intermediate pseudotime
    core = sets == "core"
    med = np.array([np.nanmedian(pt[(code == k) & core]) if ((code == k) & core).any() else np.nan
                    for k in range(K)])
    lo = np.minimum(med[code], med[partner])
    hi = np.maximum(med[code], med[partner])
    paga_edge = PG[code, partner] >= paga_threshold
    between = np.isfinite(pt) & (pt > lo) & (pt < hi)
    examined = expr_boundary & paga_edge & between

    # 3. tissue location
    A = sparse.csr_matrix(pre.obsp[dapi_key])
    A.data = (A.data > 0).astype(np.float64)
    tissue_boundary = np.zeros(n, bool)
    Ac = A.tocoo()
    np.logical_or.at(tissue_boundary, Ac.row, code[Ac.row] != code[Ac.col])
    dapi_partner = np.zeros(n, bool)
    np.logical_or.at(dapi_partner, Ac.row, code[Ac.col] == partner[Ac.row])
    dist = np.full(n, np.inf)
    for k in range(K):
        m = code == k
        if not m.any():
            continue
        tree = cKDTree(xy[m])
        idx = np.flatnonzero(partner == k)
        if len(idx):
            dist[idx] = tree.query(xy[idx])[0]
    location = np.where(dapi_partner | (dist <= near_um), "near", np.where(dist >= far_um, "far", "middle"))

    # 3. mixture null on HVG counts
    Y = _dense(pre.layers["counts"]).astype(float)
    N = Y.sum(1)
    Pm = Y / np.maximum(N, 1)[:, None]
    ref = core & ~tissue_boundary & (N > 0)
    prof = np.vstack([Pm[ref & (code == k)].mean(0) if (ref & (code == k)).sum() else np.zeros(Y.shape[1])
                      for k in range(K)])
    w = 1.0 / np.sqrt(Pm[N > 0].mean(0) + 1e-6)
    WM = (w[:, None] * prof.T)
    pools = [np.flatnonzero(ref & (code == k)) for k in range(K)]

    def mix_p(i, a_, b_, share):
        """Observed residual of node i and its p under the mixture null (share of a_ = share)."""
        r = _fit(Pm[i], WM, w)[1]
        pa, pb = pools[a_][pools[a_] != i], pools[b_][pools[b_] != i]
        if len(pa) < 5 or (share < 1 and len(pb) < 5):
            return r, np.nan
        null = np.empty(n_null)
        for t in range(n_null):
            j = rng.choice(pa)
            y = _thin(rng, Y[j], share * N[i] / max(N[j], 1))
            if share < 1:
                k = rng.choice(pb)
                y = y + _thin(rng, Y[k], (1 - share) * N[i] / max(N[k], 1))
            ny = y.sum()
            null[t] = _fit(y / max(ny, 1), WM, w)[1] if ny > 0 else np.nan
        null = null[np.isfinite(null)]
        return r, (1 + (null >= r).sum()) / (1 + len(null))

    resid = np.full(n, np.nan)
    share_own = np.full(n, np.nan)
    p_mix = np.full(n, np.nan)
    for i in np.flatnonzero(examined & (N > 0)):
        b = _fit(Pm[i], WM, w)[0]
        s_ = b[code[i]] + b[partner[i]]
        share_own[i] = float(b[code[i]] / s_) if s_ > 0 else 0.5
        resid[i], p_mix[i] = mix_p(i, code[i], partner[i], share_own[i])
    q_mix = _bh(p_mix)
    candidate = examined & (q_mix <= q_max)

    # 4. specificity check on expression-interior nodes (one-cluster null)
    interior_pool = np.flatnonzero((mix < min_mix / 2) & core & (N > 0) & ~examined)
    interior = rng.choice(interior_pool, min(n_interior, len(interior_pool)), replace=False) \
        if len(interior_pool) else np.array([], int)
    p_int = np.full(n, np.nan)
    for i in interior:
        resid[i], p_int[i] = mix_p(i, code[i], code[i], 1.0)

    nodes = pd.DataFrame(dict(cluster=lab, set=sets, partner=np.array(cats)[partner], expr_mix=mix,
                              expr_boundary=expr_boundary, paga_edge=paga_edge, pseudotime=pt,
                              pt_between=between, examined=examined, tissue_boundary=tissue_boundary,
                              dapi_link_partner=dapi_partner, dist_partner_um=dist, location=location,
                              mix_residual=resid, share_own=share_own, p_mix=p_mix, q_mix=q_mix,
                              candidate=candidate, interior_check=np.isin(np.arange(n), interior),
                              p_interior=p_int), index=pre.obs_names)

    E = nodes[nodes["examined"]]
    pairs = (E.assign(pair=E["cluster"] + "→" + E["partner"], reject=E["p_mix"] <= alpha)
              .groupby(["pair", "location"])
              .agg(nodes=("reject", "size"), reject_share=("reject", "mean"),
                   median_residual=("mix_residual", "median"), candidates=("candidate", "sum"))
              .reset_index())

    # 5. examined vs interior rejection rate, spatial block bootstrap CI
    blk = _block_ids(xy, block_um)
    ex_m = examined & np.isfinite(p_mix)
    in_m = np.isfinite(p_int)
    rej = np.where(ex_m, p_mix <= alpha, np.where(in_m, p_int <= alpha, False))
    ok = ex_m | in_m

    def diff(idx):
        e, c = ex_m[idx], in_m[idx]
        if e.sum() == 0 or c.sum() == 0:
            return np.nan
        return rej[idx][e].mean() - rej[idx][c].mean()

    obs = diff(np.arange(n))
    ub = np.unique(blk[ok])
    by = {b: np.flatnonzero((blk == b) & ok) for b in ub}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        boots.append(diff(np.concatenate([by[b] for b in pick])))
    boots = np.array([b for b in boots if np.isfinite(b)])
    ci_low = float(np.quantile(boots, 0.025)) if len(boots) else np.nan
    specific = bool(np.isfinite(ci_low) and ci_low > 0)
    loc_rej = {f"reject_{L}": float(rej[ex_m & (location == L)].mean()) if (ex_m & (location == L)).any() else np.nan
               for L in ("near", "middle", "far")}
    test = pd.DataFrame([dict(
        expression_boundary=int(expr_boundary.sum()), examined=int(ex_m.sum()),
        near=int((ex_m & (location == "near")).sum()), middle=int((ex_m & (location == "middle")).sum()),
        far=int((ex_m & (location == "far")).sum()), interior_checked=int(in_m.sum()),
        reject_examined=float(rej[ex_m].mean()) if ex_m.any() else np.nan,
        reject_interior=float(rej[in_m].mean()) if in_m.any() else np.nan, difference=obs, ci_low=ci_low,
        ci_high=float(np.quantile(boots, 0.975)) if len(boots) else np.nan, specific=specific, **loc_rej,
        candidates=int(candidate.sum()) if specific else 0, candidates_before_check=int(candidate.sum()),
        alpha=alpha, q_max=q_max, min_mix=min_mix, near_um=near_um, far_um=far_um, n_null=n_null,
        block_um=block_um)])
    if not specific:
        nodes["candidate"] = False
    if verbose:
        t = test.iloc[0]
        print(f"[14-3] expression-boundary nodes {int(t.expression_boundary):,} | examined (PAGA edge + between in "
              f"pseudotime) {int(t.examined):,}: near {int(t.near)}, middle {int(t.middle)}, far {int(t.far)}")
        print(f"[14-3] not explained as a mixture (p ≤ {alpha}): examined {t.reject_examined:.2f} vs interior "
              f"{t.reject_interior:.2f} (n {int(t.interior_checked)}), difference {t.difference:.2f} "
              f"(95% CI {t.ci_low:.2f}–{t.ci_high:.2f}) | by location near {t.reject_near:.2f}, "
              f"middle {t.reject_middle:.2f}, far {t.reject_far:.2f}")
        if specific:
            print(f"[14-3] candidates (q_mix ≤ {q_max}): {int(t.candidates)}")
        else:
            print("[14-3] the mixture test is not specific on this section (examined nodes do not reject more often "
                  "than interior nodes): no candidates are called.")
    return dict(nodes=nodes, pairs=pairs, test=test)
