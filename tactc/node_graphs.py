"""
node_graphs.py
==============

Step 6 of TACTC.ipynb on the spot nodes (step 5), plus helpers shared with steps 9–10.

Step 6. Group-specific node profiles and per-group graphs
  With one signature per group for the whole section, the spot decomposition gives every group in a node the same
  fold change (y / expected mixture), so group-specific spatial change cannot be seen. Local signatures fix this
  (visiumif2niche 10c idea of local profiles): for node s, per-cell expression μ_G(s) of every group is estimated by
  ridge-regularised least squares over its neighbourhood (nodes within radius_um, Gaussian weights σ):
      minimise Σ_k w_sk ‖ỹ_k − Σ_G n_kG μ_G(s)‖² + α_s Σ_G ‖μ_G(s) − μ_G‖²        (α_s = ridge × mean diag)
  ỹ_k = counts / capture factor (node total vs expected total), n_kG = circle cells (their RNA was captured).
  Neighbours differ in composition, so a change is attributed to the group whose abundance follows it.
  Group graph G: lattice edges between nodes that both contain G, weight
      max(0, cos(r_i, r_j)) · √(n_iG n_jG) / median,   r = per-gene standardised log fold change of μ_G(s) vs μ_G
  → obsp['graph_<i>'] (i = group index).

Shared helpers: shift_maps (shift null), random_cuts (split-half), label transfer, η² separation vs the shift null,
independent_features (Visium proteins, node morphology). Clustering, PAGA and the cluster summaries are in
paga_nodes.py (step 10).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["shift_maps", "random_cuts", "local_signatures", "group_graphs",
           "independent_features"]


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def shift_maps(xy: np.ndarray, n_shift: int = 200, min_shift_um: float = 1000.0, seed: int = 0,
               min_frac: float = 0.2):
    """Shift null maps (visiumif2niche benchmark._shift_maps): per shift, (positions, node nearest to position + v)
    for nodes whose shifted position stays in the tissue. Shifts shorter than min_shift_um are not used."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    spacing = float(np.median(tree.query(xy, k=2)[0][:, 1]))
    span = xy.max(0) - xy.min(0)
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n_shift and tries < n_shift * 100:
        tries += 1
        v = rng.uniform(-span, span)
        if np.linalg.norm(v) < min_shift_um:
            continue
        d, j = tree.query(xy + v)
        ok = d <= spacing * 0.75
        if ok.sum() >= max(30, min_frac * len(xy)):
            out.append((np.flatnonzero(ok), j[ok]))
    if len(out) < n_shift:
        print(f"  note: only {len(out)} of {n_shift} shifts ≥ {min_shift_um:g} µm keep ≥ {min_frac:.0%} of the nodes "
              "in the tissue (lower min_shift_um for small sections)")
    return out


def random_cuts(xy: np.ndarray, n_split: int = 10, seed: int = 0, min_frac: float = 0.3) -> List[dict]:
    """Random straight cuts leaving ≥ min_frac of the nodes on either side (visiumif2niche scanpy_compare)."""
    rng = np.random.default_rng(seed)
    cuts = []
    for _ in range(n_split):
        th = rng.uniform(0, np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        pr = xy @ u
        cuts.append(dict(u=u, t=float(rng.uniform(*np.quantile(pr, [min_frac, 1 - min_frac])))))
    return cuts


def _rownorm(W):
    from scipy import sparse

    s = np.asarray(W.sum(axis=1)).ravel()
    return sparse.diags(np.where(s > 0, 1.0 / s, 0.0)) @ W


# ---------------------------------------------------------------------------
# step 6
# ---------------------------------------------------------------------------
def local_signatures(counts: np.ndarray, n_circle: np.ndarray, xy: np.ndarray, mu_global: np.ndarray,
                     radius_um: float = 250.0, sigma_um: float = 100.0, ridge: float = 0.1) -> np.ndarray:
    """Per-node per-cell expression of every group: G × S × genes (float32, ≥ 0). See the module docstring.
    counts: S × genes, n_circle: S × G, mu_global: genes × G."""
    from scipy.spatial import cKDTree

    Y = np.asarray(counts, float)
    N = np.asarray(n_circle, float)
    Mg = np.asarray(mu_global, float)                         # genes × G
    expect = (N @ Mg.T).sum(axis=1)
    f = np.where(expect > 0, Y.sum(axis=1) / np.maximum(expect, 1e-12), 1.0)
    f = np.where(np.isfinite(f) & (f > 0), f, 1.0)
    Yt = Y / f[:, None]
    S, G = N.shape
    out = np.empty((G, S, Y.shape[1]), dtype=np.float32)
    tree = cKDTree(xy)
    nbs = tree.query_ball_point(xy, radius_um)
    eye = np.eye(G)
    for s in range(S):
        k = np.asarray(nbs[s], int)
        w = np.exp(-((xy[k] - xy[s]) ** 2).sum(axis=1) / (2 * sigma_um ** 2))
        Nw = N[k] * w[:, None]
        A0 = Nw.T @ N[k]
        alpha = ridge * max(float(np.mean(np.diag(A0))), 1e-6)
        A = A0 + alpha * eye
        B = Nw.T @ Yt[k] + alpha * Mg.T
        mu = np.linalg.solve(A, B)
        out[:, s, :] = np.clip(mu, 0, None)
    return out


def _resid_profiles(mu_s: np.ndarray, mu_g: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """log fold change of the local profile vs the global one (per-node normalised), standardised per gene over mask."""
    P = mu_s / np.maximum(mu_s.sum(axis=1, keepdims=True), 1e-12) * 1e4
    q = mu_g / max(mu_g.sum(), 1e-12) * 1e4
    R = np.log1p(P) - np.log1p(q)[None, :]
    Z = np.zeros_like(R)
    if mask.sum() > 2:
        x = R[mask]
        sd = x.std(axis=0)
        Z[mask] = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return Z


def group_graphs(nodes, local: np.ndarray, min_cells: int = 1) -> Dict[str, object]:
    """Per-group lattice graphs with expression-similarity edge weights (module docstring, step 6).
    Returns {f'graph_{i}': csr} for obsp; also stores obs['has_group_<i>']."""
    from scipy import sparse

    L = nodes.obsp["spatial_connectivities"].tocoo()
    i0, j0 = L.row[L.row < L.col], L.col[L.row < L.col]
    Ch = np.asarray(nodes.obsm["cell_counts_hex"], float)
    Mg = np.asarray(nodes.varm["group_signature"], float)
    n = nodes.n_obs
    out = {}
    for g in range(Ch.shape[1]):
        mask = Ch[:, g] >= min_cells
        Z = _resid_profiles(local[g], Mg[:, g], mask)
        both = mask[i0] & mask[j0]
        a, b = i0[both], j0[both]
        if not len(a):
            out[f"graph_{g}"] = sparse.csr_matrix((n, n))
            continue
        na, nb = np.linalg.norm(Z[a], axis=1), np.linalg.norm(Z[b], axis=1)
        cos = (Z[a] * Z[b]).sum(axis=1) / np.maximum(na * nb, 1e-12)
        size = np.sqrt(Ch[a, g] * Ch[b, g])
        size = size / max(np.median(size), 1e-12)
        w = np.clip(cos, 0, 1) * size
        out[f"graph_{g}"] = sparse.csr_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    return out


# ---------------------------------------------------------------------------
# evaluation helpers shared with step 10 (paga_nodes)
# ---------------------------------------------------------------------------
def _transfer(P_fit: np.ndarray, lab_fit: np.ndarray, P_new: np.ndarray, k: int = 15) -> np.ndarray:
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=int(min(k, len(P_fit))))
    clf.fit(P_fit, lab_fit)
    return clf.predict(P_new)


def _eta2(v: np.ndarray, lab: np.ndarray) -> float:
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    if len(v) < 3 or v.std() == 0:
        return np.nan
    m = v.mean()
    tot = ((v - m) ** 2).sum()
    btw = sum(((v[lab == c].mean() - m) ** 2) * (lab == c).sum() for c in np.unique(lab))
    return float(btw / tot)


def _sep_z(feats: pd.DataFrame, lab: np.ndarray, shifts) -> pd.Series:
    out = {}
    for c in feats.columns:
        v = feats[c].to_numpy(float)
        obs = _eta2(v, lab)
        nul = np.array([_eta2(v[ii], lab[jj]) for ii, jj in shifts], float)
        nul = nul[np.isfinite(nul)]
        out[c] = (obs - nul.mean()) / nul.std() if len(nul) > 2 and nul.std() > 0 else np.nan
    return pd.Series(out)


def independent_features(nodes, spots_adata=None) -> Dict[str, pd.DataFrame]:
    """Features not used for clustering: Visium proteins (log CP10k, isotypes removed) and node morphology."""
    out = {}
    if spots_adata is not None and "protein" in spots_adata.obsm:
        P = spots_adata.obsm["protein"]
        P = P.reindex(nodes.obs_names)
        P = P[[c for c in P.columns if not str(c).lower().startswith(("mouse_", "rat_", "isotype"))]]
        out["protein"] = np.log1p(P.div(P.sum(axis=1).replace(0, 1), axis=0) * 1e4)
    if "morphology" in nodes.obsm:
        M = nodes.obsm["morphology"]
        keep = [c for c in M.columns if c in ("density_per_mm2", "area_mean_um2", "area_cv") or c.startswith("log_")]
        out["morphology"] = M[keep].astype(float)
    return out
