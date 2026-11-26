"""
cross_groups.py
===============

Step 9 of TACTC.ipynb: per gene, how the expression of group A relates to group B on the spot-node graph.
Group expression per node = log1p of the local per-cell signature μ_A(s) (node_graphs.local_signatures, step 6).

Two statistics per gene
  overlap   (A, B unordered): Pearson r across nodes containing both A and B of log μ_A(s,g) and log μ_B(s,g)
            → co-variation of the two groups where they coexist
  neighbour (A outcome, B exposure): OLS t of x_B in   log μ_A(s,g) ~ 1 + own composition + x_B(s),
            x_B(s) = share of B among the cells of the lattice neighbours of s (s itself excluded)
            → A's expression next to B, beyond A's own node composition (contact-dependent signal)

Two nulls, both required
  shift null         : the B side is read at the node nearest to s + v (|v| ≥ min_shift_um; visiumif2niche shift
                       null). Keeps each map's spatial autocorrelation, breaks the positional correspondence.
                       (Nodes whose shifted position leaves the tissue take a random valid shifted value.)
  decomposition null : counts simulated with NO spatial change and NO interaction,
                       y_sim[s,g] ~ NB(mean = f_s Σ_G n_circle[s,G] μ_G(g), per-gene dispersion from the data),
                       same composition and capture factors; local signatures and both statistics recomputed.
                       Captures what the estimation alone produces (A and B estimated from the same counts, and
                       neighbour composition entering the local fits). The shift null cannot see this: shifting
                       breaks the within-node coupling, so a pure estimation artefact would look significant.
  z = (observed − null mean) / null sd per null. With few null replicates the sd is noisy and z comes out too wide (in a
  synthetic test without any interaction: sd ≈ 1.3, i.e. false calls). With empirical_null=True (default) each pair's z
  is recentred and rescaled by the median and MAD across genes (Efron's empirical null; assumes most genes have no
  effect), then two-sided normal p and BH per statistic and null — over all pairs × genes (fdr_scope='all', default,
  conservative) or within each pair (fdr_scope='pair'). Raw z are kept as *_zraw_*.
  call = q_shift ≤ α and q_decomp ≤ α and both z have the same sign.

Output: a separate AnnData (obs = ordered group pairs A|B, var = genes).

Checks
  replication_split   : the two halves of the tissue analysed separately; calls of one half checked in the other
                        against a chance level (visiumif2niche's lesson: hit lists must be shown to replicate)
  pseudo_pair_control : one group split at random into two identical pseudo-groups; calls between them are false by
                        construction (a negative control on the real data that does not rely on the simulation model)
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .node_graphs import local_signatures, shift_maps

__all__ = ["group_log_profiles", "cross_group_tests", "replication_split", "pseudo_pair_control"]


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def group_log_profiles(local: np.ndarray) -> np.ndarray:
    """log1p of the local per-cell signatures (G × S × genes)."""
    return np.log1p(np.clip(local, 0, None)).astype(np.float32)


def _colcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=0) / den, np.nan)


def _neighbor_share(lattice, Ch: np.ndarray) -> np.ndarray:
    """S × G share of each group among the cells of the lattice neighbours (own node excluded)."""
    Adj = (lattice > 0).astype(float)
    tot = np.asarray(Adj @ Ch)
    s = tot.sum(axis=1, keepdims=True)
    return np.divide(tot, s, out=np.zeros_like(tot), where=s > 0)


def _ols_t_batch(Y: np.ndarray, C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """t of each column of X (n × m) in Y ~ C + x, for all genes (columns of Y) at once → m × genes."""
    Q, R = np.linalg.qr(C)
    keep = np.abs(np.diag(R)) > 1e-10 * max(np.abs(np.diag(R)).max(), 1e-300)
    Q = Q[:, keep]
    Yr = Y - Q @ (Q.T @ Y)
    Xr = X - Q @ (Q.T @ X)
    xx = (Xr * Xr).sum(axis=0)
    yy = (Yr * Yr).sum(axis=0)
    B = Xr.T @ Yr
    beta = B / np.where(xx > 0, xx, np.nan)[:, None]
    df = Y.shape[0] - Q.shape[1] - 1
    rss = yy[None, :] - beta ** 2 * xx[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.clip(rss, 0, None) / max(df, 1) / xx[:, None])
        return beta / se


def _stats(L: np.ndarray, Ch: np.ndarray, frac: np.ndarray, xshare: np.ndarray, pairs_u, pairs_o, masks,
           shifts=None, rng=None, outcome_rows=None):
    """Observed (shifts=None) or shift-null statistics. Returns dict of arrays."""
    out_ov, out_nb = {}, {}
    G = L.shape[0]
    C_full = np.column_stack([np.ones(len(frac)), frac[:, :-1]]) if frac.shape[1] > 1 else np.ones((len(frac), 1))
    for a, b in pairs_u:
        S = masks[a] & masks[b]
        if shifts is None:
            out_ov[(a, b)] = _colcorr(L[a][S], L[b][S])
        else:
            vals = []
            for ii, jj in shifts:
                m = masks[a][ii] & masks[b][jj]
                if m.sum() < 10:
                    continue
                vals.append(_colcorr(L[a][ii[m]], L[b][jj[m]]))
            out_ov[(a, b)] = np.array(vals)
    for a, b in pairs_o:
        rows = outcome_rows[a]
        Y = L[a][rows]
        C = C_full[rows]
        x = xshare[rows, b]
        if shifts is None:
            out_nb[(a, b)] = _ols_t_batch(Y, C, x[:, None])[0]
        else:
            pos = -np.ones(len(frac), int)
            pos[rows] = np.arange(len(rows))
            Xs = []
            for ii, jj in shifts:
                xs = np.full(len(rows), np.nan)
                inrow = pos[ii] >= 0
                xs[pos[ii[inrow]]] = xshare[jj[inrow], b]
                miss = ~np.isfinite(xs)
                if miss.all():
                    continue
                if miss.any():
                    xs[miss] = rng.choice(xs[~miss], miss.sum(), replace=True)
                Xs.append(xs)
            out_nb[(a, b)] = _ols_t_batch(Y, C, np.column_stack(Xs)) if Xs else np.empty((0, Y.shape[1]))
    return out_ov, out_nb


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


def _nb_dispersion(Y: np.ndarray, Mu: np.ndarray) -> np.ndarray:
    num = ((Y - Mu) ** 2 - Mu).sum(axis=0)
    den = (Mu ** 2).sum(axis=0)
    return np.clip(np.divide(num, den, out=np.zeros_like(num), where=den > 0), 0, 10)


def cross_group_tests(nodes, local: Optional[np.ndarray] = None, min_cells: int = 1, min_nodes: int = 30,
                      n_shift: int = 200, min_shift_um: float = 1000.0, n_decomp: int = 200, alpha: float = 0.05,
                      radius_um: float = 250.0, sigma_um: float = 100.0, ridge: float = 0.1, seed: int = 0,
                      empirical_null: bool = True, fdr_scope: str = "all", verbose: bool = True):
    """Overlap and neighbour statistics with shift and decomposition nulls → AnnData (obs = A|B pairs, var = genes).
    radius_um, sigma_um, ridge must match the step-6 local signatures (they are re-estimated on the simulations)."""
    import anndata as ad

    groups = list(nodes.uns["nodes"]["groups"])
    genes = list(nodes.var_names)
    xy = np.asarray(nodes.obsm["spatial"], float)
    Ch = np.asarray(nodes.obsm["cell_counts_hex"], float)
    Nc = np.asarray(nodes.obsm["cell_counts_circle"], float)
    frac = np.asarray(nodes.obsm["group_frac"], float)
    Y = _dense(nodes.layers["counts"]).astype(float)
    Mg = np.asarray(nodes.varm["group_signature"], float)
    lattice = nodes.obsp["spatial_connectivities"]
    if local is None:
        local = local_signatures(Y, Nc, xy, Mg, radius_um, sigma_um, ridge)
    L = group_log_profiles(local)
    G = len(groups)
    masks = [Ch[:, g] >= min_cells for g in range(G)]
    deg = np.asarray((lattice > 0).sum(axis=1)).ravel()
    xshare = _neighbor_share(lattice, Ch)
    okg = [g for g in range(G) if masks[g].sum() >= min_nodes]
    pairs_u = [(a, b) for i, a in enumerate(okg) for b in okg[i + 1:] if (masks[a] & masks[b]).sum() >= min_nodes]
    pairs_o = [(a, b) for a in okg for b in okg if a != b]
    outcome_rows = {a: np.flatnonzero(masks[a] & (deg > 0)) for a in okg}
    if not pairs_o:
        raise ValueError(f"Fewer than two groups have ≥ {min_nodes} nodes; lower min_nodes.")
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"[cross] {len(okg)} groups with ≥ {min_nodes} nodes | {len(pairs_u)} overlap pairs, {len(pairs_o)} "
              f"neighbour pairs × {len(genes):,} genes")

    obs_ov, obs_nb = _stats(L, Ch, frac, xshare, pairs_u, pairs_o, masks, outcome_rows=outcome_rows)
    shifts = shift_maps(xy, n_shift, min_shift_um, seed)
    sh_ov, sh_nb = _stats(L, Ch, frac, xshare, pairs_u, pairs_o, masks, shifts=shifts, rng=rng,
                          outcome_rows=outcome_rows)
    if verbose:
        print(f"[cross] shift null: {len(shifts)} shifts ≥ {min_shift_um:g} µm")

    # decomposition null
    expect = Nc @ Mg.T
    f = np.where(expect.sum(axis=1) > 0, Y.sum(axis=1) / np.maximum(expect.sum(axis=1), 1e-12), 1.0)
    Mu = expect * f[:, None]
    phi = _nb_dispersion(Y, Mu)
    dec_ov = {k: [] for k in obs_ov}
    dec_nb = {k: [] for k in obs_nb}
    pos = phi > 0
    kshape = 1.0 / np.maximum(phi[pos], 1e-8)
    for r in range(int(n_decomp)):
        lam = Mu.copy()
        if pos.any():                                   # gamma-Poisson: mean Mu, variance Mu + φ Mu²
            lam[:, pos] = rng.gamma(kshape[None, :], Mu[:, pos] / kshape[None, :])
        Ys = rng.poisson(np.clip(lam, 0, None)).astype(float)
        Ls = group_log_profiles(local_signatures(Ys, Nc, xy, Mg, radius_um, sigma_um, ridge))
        o1, o2 = _stats(Ls, Ch, frac, xshare, pairs_u, pairs_o, masks, outcome_rows=outcome_rows)
        for k in o1:
            dec_ov[k].append(o1[k])
        for k in o2:
            dec_nb[k].append(o2[k])
        if verbose and (r + 1) % max(1, n_decomp // 5) == 0:
            print(f"[cross] decomposition null {r + 1}/{n_decomp}")

    def z_of(obs, null):
        null = np.asarray(null, float)
        if null.ndim != 2 or len(null) < 3:
            return np.full_like(obs, np.nan)
        m, s = np.nanmean(null, axis=0), np.nanstd(null, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(s > 0, (obs - m) / s, np.nan)

    from scipy.stats import norm

    if fdr_scope not in ("all", "pair"):
        raise ValueError("fdr_scope must be 'all' (BH over all pairs × genes) or 'pair' (BH within each pair)")

    def q_of(z):
        p = 2 * norm.sf(np.abs(z))
        if fdr_scope == "pair":
            return np.vstack([_bh(row) for row in p]) if len(p) else p
        return _bh(p.ravel()).reshape(z.shape)

    def calibrate(z):
        if not empirical_null or not len(z):
            return z
        med = np.nanmedian(z, axis=1, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(z - med), axis=1, keepdims=True)
        return (z - med) / np.where(mad > 0, mad, 1.0)

    nG = len(genes)
    U = {u: i for i, u in enumerate(pairs_u)}
    ov = {"r": np.vstack([obs_ov[u] for u in pairs_u]) if pairs_u else np.zeros((0, nG))}
    ov["zraw_shift"] = np.vstack([z_of(obs_ov[u], sh_ov[u]) for u in pairs_u]) if pairs_u else np.zeros((0, nG))
    ov["zraw_decomp"] = np.vstack([z_of(obs_ov[u], dec_ov[u]) for u in pairs_u]) if pairs_u else np.zeros((0, nG))
    ov["z_shift"], ov["z_decomp"] = calibrate(ov["zraw_shift"]), calibrate(ov["zraw_decomp"])
    ov["q_shift"], ov["q_decomp"] = q_of(ov["z_shift"]), q_of(ov["z_decomp"])
    nb = {"t": np.vstack([obs_nb[k] for k in pairs_o])}
    nb["zraw_shift"] = np.vstack([z_of(obs_nb[k], sh_nb[k]) for k in pairs_o])
    nb["zraw_decomp"] = np.vstack([z_of(obs_nb[k], dec_nb[k]) for k in pairs_o])
    nb["z_shift"], nb["z_decomp"] = calibrate(nb["zraw_shift"]), calibrate(nb["zraw_decomp"])
    nb["q_shift"], nb["q_decomp"] = q_of(nb["z_shift"]), q_of(nb["z_decomp"])
    for d in (ov, nb):
        d["call"] = ((d["q_shift"] <= alpha) & (d["q_decomp"] <= alpha)
                     & (np.sign(d["z_shift"]) == np.sign(d["z_decomp"]))).astype(float)

    rows, M = [], {f"overlap_{k}": [] for k in ov}
    M.update({f"neighbor_{k}": [] for k in nb})
    for i, (a, b) in enumerate(pairs_o):
        u = U.get((min(a, b), max(a, b)))
        for k in ov:
            M[f"overlap_{k}"].append(ov[k][u] if u is not None else np.full(nG, np.nan))
        for k in nb:
            M[f"neighbor_{k}"].append(nb[k][i])
        rows.append(dict(pair=f"{groups[a]}|{groups[b]}", typeA=groups[a], typeB=groups[b],
                         n_overlap_nodes=int((masks[a] & masks[b]).sum()), n_outcome_nodes=int(len(outcome_rows[a]))))
    M = {k: np.vstack(v).astype(np.float32) for k, v in M.items()}
    M["neighbor_call"] = np.nan_to_num(M["neighbor_call"])
    M["overlap_call"] = np.nan_to_num(M["overlap_call"])
    n_ov_calls = int(np.nansum(ov["call"])) if len(ov["call"]) else 0

    obs = pd.DataFrame(rows).set_index("pair")
    obs.index = obs.index.astype(str)
    a = ad.AnnData(X=np.nan_to_num(M["neighbor_z_shift"]), obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    for k, v in M.items():
        a.layers[k] = v
    a.obs["overlap_calls_up"] = ((M["overlap_call"] > 0) & (M["overlap_z_decomp"] > 0)).sum(axis=1)
    a.obs["overlap_calls_down"] = ((M["overlap_call"] > 0) & (M["overlap_z_decomp"] < 0)).sum(axis=1)
    a.obs["neighbor_calls_up"] = ((M["neighbor_call"] > 0) & (M["neighbor_z_decomp"] > 0)).sum(axis=1)
    a.obs["neighbor_calls_down"] = ((M["neighbor_call"] > 0) & (M["neighbor_z_decomp"] < 0)).sum(axis=1)
    a.uns["cross_groups"] = {"groups": groups, "alpha": alpha, "min_cells": min_cells, "min_nodes": min_nodes,
                             "n_shift": len(shifts), "min_shift_um": min_shift_um, "n_decomp": int(n_decomp),
                             "radius_um": radius_um, "sigma_um": sigma_um, "ridge": ridge, "seed": seed,
                             "empirical_null": bool(empirical_null), "fdr_scope": fdr_scope,
                             "nb_dispersion_median": float(np.median(phi))}
    if verbose:
        print(f"[cross] calls (both nulls, q ≤ {alpha}): overlap {n_ov_calls} (unordered pair × gene), "
              f"neighbour {int(M['neighbor_call'].sum())}")
    return a


# ---------------------------------------------------------------------------
# checks: discovery → replication split, pseudo-pair negative control
# ---------------------------------------------------------------------------
def _halves(xy: np.ndarray):
    """Split along the long axis of the tissue at the median → two halves of equal node number."""
    c = xy - xy.mean(axis=0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    pr = c @ u
    return pr < np.median(pr)


def replication_split(nodes, verbose: bool = True, **kw):
    """Discovery → replication between the two halves of the tissue (split along its long axis).
    Each half is analysed on its own (local signatures, both nulls); calls of one half are checked in the other:
    replicated = both calibrated z (shift and decomposition) have the discovery sign and two-sided p ≤ 0.05.
    background = the same criterion for a random sign over all tests of the replication half (chance level).
    kw: cross_group_tests arguments (min_shift_um may need to be lower for half sections).
    Returns (summary DataFrame, hits DataFrame, {'half_1': AnnData, 'half_2': AnnData})."""
    from scipy.stats import norm

    xy = np.asarray(nodes.obsm["spatial"], float)
    side = _halves(xy)
    res = {}
    for name, m in (("half_1", side), ("half_2", ~side)):
        sub = nodes[m].copy()
        for k in [k for k in sub.layers.keys() if k is not None and str(k).startswith("local_")]:
            del sub.layers[k]
        if verbose:
            print(f"[replication] {name}: {sub.n_obs:,} nodes")
        res[name] = cross_group_tests(sub, local=None, verbose=False, **kw)
    rows, hits = [], []
    for disc, rep in (("half_1", "half_2"), ("half_2", "half_1")):
        D, R = res[disc], res[rep]
        pairs = [p for p in D.obs_names if p in set(R.obs_names)]
        genes = [g for g in D.var_names if g in set(R.var_names)]
        di, ri = [list(D.obs_names).index(p) for p in pairs], [list(R.obs_names).index(p) for p in pairs]
        dg, rg = [list(D.var_names).index(g) for g in genes], [list(R.var_names).index(g) for g in genes]
        for stat in ("neighbor", "overlap"):
            call = D.layers[f"{stat}_call"][np.ix_(di, dg)] > 0
            sgn = np.sign(D.layers[f"{stat}_z_decomp"][np.ix_(di, dg)])
            zs, zd = R.layers[f"{stat}_z_shift"][np.ix_(ri, rg)], R.layers[f"{stat}_z_decomp"][np.ix_(ri, rg)]
            sig = (2 * norm.sf(np.abs(zs)) <= 0.05) & (2 * norm.sf(np.abs(zd)) <= 0.05)
            ok = sig & (np.sign(zs) == sgn) & (np.sign(zd) == sgn)
            conc = sig & (np.sign(zs) == np.sign(zd))
            bg = 0.5 * float(np.nanmean(conc)) if conc.size else np.nan
            n_disc = int(call.sum())
            n_rep = int((call & ok).sum())
            rate = n_rep / n_disc if n_disc else np.nan
            rows.append(dict(discovery=disc, replication=rep, statistic=stat, discovered=n_disc, replicated=n_rep,
                             rate=rate, background=bg, enrichment=rate / bg if n_disc and bg > 0 else np.nan))
            for a, b in zip(*np.nonzero(call)):
                hits.append(dict(discovery=disc, statistic=stat, pair=pairs[a], gene=genes[b],
                                 z_decomp_discovery=float(D.layers[f"{stat}_z_decomp"][di[a], dg[b]]),
                                 z_decomp_replication=float(zd[a, b]), z_shift_replication=float(zs[a, b]),
                                 replicated=bool(ok[a, b])))
    S = pd.DataFrame(rows)
    if verbose:
        for r in rows:
            print(f"[replication] {r['discovery']} → {r['replication']} {r['statistic']}: {r['replicated']}/{r['discovered']}"
                  f" replicated (rate {r['rate']:.2f} vs chance {r['background']:.3f})" if r["discovered"] else
                  f"[replication] {r['discovery']} → {r['replication']} {r['statistic']}: no calls in the discovery half")
    return S, pd.DataFrame(hits), res


def pseudo_pair_control(nodes, group: Optional[str] = None, seed: int = 0, verbose: bool = True, **kw):
    """Negative control on the real data: the cells of one group are split at random (binomial, per node and per
    circle / ring) into two pseudo-groups with the same true expression, and step 9 is re-run. Every call between the
    two pseudo-groups is false by construction; their z show how well the nulls are calibrated without relying on
    the simulation model. group: default = the group with the most cells. kw: cross_group_tests arguments.
    Returns (summary DataFrame, result AnnData)."""
    import anndata as ad

    groups = list(nodes.uns["nodes"]["groups"])
    Ch = np.asarray(nodes.obsm["cell_counts_hex"], float)
    g = groups.index(group) if group is not None else int(np.argmax(Ch.sum(axis=0)))
    rng = np.random.default_rng(seed)
    new = nodes.copy()
    for k in [k for k in new.layers.keys() if k is not None and (str(k).startswith("local_") or str(k).startswith("decomp_"))]:
        del new.layers[k]
    Cc = np.asarray(nodes.obsm["cell_counts_circle"], float)
    Cr = np.asarray(nodes.obsm["cell_counts_ring"], float)
    ca = rng.binomial(np.rint(Cc[:, g]).astype(int), 0.5)
    ra = rng.binomial(np.rint(Cr[:, g]).astype(int), 0.5)
    cb, rb = np.rint(Cc[:, g]) - ca, np.rint(Cr[:, g]) - ra
    Cc2 = np.column_stack([np.delete(Cc, g, axis=1), ca, cb])
    Cr2 = np.column_stack([np.delete(Cr, g, axis=1), ra, rb])
    Ch2 = Cc2 + Cr2
    new.obsm["cell_counts_circle"], new.obsm["cell_counts_ring"], new.obsm["cell_counts_hex"] = Cc2, Cr2, Ch2
    s = Ch2.sum(axis=1, keepdims=True)
    new.obsm["group_frac"] = np.divide(Ch2, s, out=np.zeros_like(Ch2), where=s > 0)
    M = np.asarray(nodes.varm["group_signature"], float)
    new.varm["group_signature"] = np.column_stack([np.delete(M, g, axis=1), M[:, g], M[:, g]])
    na, nb = f"{groups[g]} (pseudo A)", f"{groups[g]} (pseudo B)"
    st = dict(new.uns["nodes"])
    st["groups"] = [x for i, x in enumerate(groups) if i != g] + [na, nb]
    new.uns["nodes"] = st
    if verbose:
        print(f"[negative control] {groups[g]} split at random into two pseudo-groups "
              f"({int(Ch2[:, -2].sum()):,} / {int(Ch2[:, -1].sum()):,} cells)")
    res = cross_group_tests(new, local=None, verbose=False, **kw)
    rows = []
    for pair in (f"{na}|{nb}", f"{nb}|{na}"):
        if pair not in res.obs_names:
            continue
        i = list(res.obs_names).index(pair)
        for stat in ("overlap", "neighbor"):
            raw = res.layers["overlap_r" if stat == "overlap" else "neighbor_t"][i]
            rows.append(dict(pair=pair, statistic=stat, tests=int(np.isfinite(raw).sum()),
                             false_calls=int(np.nansum(res.layers[f"{stat}_call"][i])),
                             raw_mean=float(np.nanmean(raw)),
                             z_shift_mean=float(np.nanmean(res.layers[f"{stat}_z_shift"][i])),
                             z_shift_sd=float(np.nanstd(res.layers[f"{stat}_z_shift"][i])),
                             z_decomp_mean=float(np.nanmean(res.layers[f"{stat}_z_decomp"][i])),
                             z_decomp_sd=float(np.nanstd(res.layers[f"{stat}_z_decomp"][i]))))
    S = pd.DataFrame(rows)
    if verbose and len(S):
        for r in rows:
            print(f"[negative control] {r['pair']} {r['statistic']}: {r['false_calls']} false calls / {r['tests']} tests"
                  f" | raw mean {r['raw_mean']:.3f} | z_decomp {r['z_decomp_mean']:.2f} ± {r['z_decomp_sd']:.2f}")
    return S, res
