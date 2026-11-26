"""
paper_checks.py
===============

Sections 14 and 15 of TACTC.ipynb: supplementary checks, run at the end on the saved results.

13-1 Composition vs a standard baseline
     Pipeline group shares (step 1) and a scanpy marker score (sc.tl.score_genes on log-normalised spots, the usual
     baseline without a reference) are both correlated with the matching Visium proteins. Per group: Spearman ρ of
     each (mean over its matching proteins), shift-null z of each, and the paired difference with a spatial block
     bootstrap 95% CI (blocks of block_um; visiumif2niche 13-4 A).
13-3 Step 9 against naive testing
     The same statistics with textbook p-values (t / Fisher-z, spots as independent samples, no shift or decomposition
     null) → calls on the real data and on the pseudo-pair negative control (where every call is false).
13-4 Planted-effect simulation (recovery)
     Counts simulated from the real node composition and capture factors (negative binomial around
     Σ cells × section-wide signature) with a planted effect: group A's expression of n_genes genes rises with the
     share of group B among the neighbouring nodes' cells. Step 9 is re-run → recall of the planted genes for A | B,
     false calls elsewhere. Optimistic in one respect: the simulation uses the same generative model as the
     decomposition null.
13-5 RNA-independent morphology check
     Spearman ρ of each group's share with node morphology (density, nuclear area / CV, channel intensity), against the
     shift null — e.g. tumour share vs nuclear area, immune share vs density.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .node_graphs import shift_maps

__all__ = ["BASELINE_MAP", "block_bootstrap", "marker_scores", "composition_vs_baseline",
           "naive_calls", "planted_simulation", "recovery", "morphology_check"]

# pipeline group → marker set of programs.MARKERS used for the marker-score baseline
BASELINE_MAP = {"T/NK": "T/NK cell", "B/Plasma": "B/Plasma cell", "Macrophage/Mono": "Macrophage",
                "Microglia": "Microglia", "Endothelial": "Endothelial", "Mural": "Mural (pericyte/SMC)",
                "Oligodendrocyte": "Oligodendrocyte", "Neuron": "Neuron"}


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def _blocks(xy: np.ndarray, block_um: float) -> List[np.ndarray]:
    b = np.floor(xy / block_um).astype(np.int64)
    code = pd.factorize(pd.Series(b[:, 0]).astype(str) + "_" + pd.Series(b[:, 1]).astype(str))[0]
    return [np.flatnonzero(code == c) for c in range(code.max() + 1)]


def block_bootstrap(stat, xy: np.ndarray, n_boot: int = 500, block_um: float = 500.0, seed: int = 0):
    """Spatial block bootstrap of stat(indices) → (estimate, 2.5%, 97.5%, two-sided bootstrap p of 0)."""
    rng = np.random.default_rng(seed)
    blocks = _blocks(xy, block_um)
    est = stat(np.arange(len(xy)))
    bs = np.array([stat(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]))
                   for _ in range(n_boot)], float)
    bs = bs[np.isfinite(bs)]
    if not len(bs):
        return est, np.nan, np.nan, np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / (len(bs) + 1)))
    return est, float(lo), float(hi), p


def marker_scores(spots, markers: Dict[str, Sequence[str]], seed: int = 0) -> pd.DataFrame:
    """Spot × marker set: sc.tl.score_genes on normalize_total(1e4) + log1p (standard reference-free baseline)."""
    import scanpy as sc

    a = spots.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    out = {}
    for name, genes in markers.items():
        g = [x for x in genes if x in a.var_names]
        if len(g) < 2:
            continue
        sc.tl.score_genes(a, g, score_name="_s", random_state=seed)
        out[name] = a.obs["_s"].to_numpy(float)
    return pd.DataFrame(out, index=a.obs_names)


def _protein(vis) -> Optional[pd.DataFrame]:
    if "protein" not in vis.obsm:
        return None
    p = vis.obsm["protein"]
    p = p[[c for c in p.columns if not str(c).lower().startswith(("mouse_", "rat_", "isotype"))]]
    return np.log1p(p.div(p.sum(axis=1).replace(0, 1), axis=0) * 1e4)


def composition_vs_baseline(frac: pd.DataFrame, baseline: pd.DataFrame, vis, rules: Dict[str, List[str]],
                            baseline_map: Optional[Dict[str, str]] = None, n_shift: int = 200,
                            min_shift_um: float = 1000.0, n_boot: int = 500, block_um: float = 500.0,
                            seed: int = 0) -> pd.DataFrame:
    """Per group: ρ(pipeline share, protein) and ρ(marker score, protein) (mean over matching proteins), their shift-null
    z, and the paired difference with a spatial block-bootstrap CI."""
    bmap = baseline_map or BASELINE_MAP
    prot = _protein(vis)
    if prot is None:
        raise ValueError("No Visium proteins (obsm['protein']).")
    idx = frac.dropna().index.intersection(prot.index).intersection(baseline.index)
    mpp = float(vis.uns.get("mpp", 1.0))
    xy = vis.obs.loc[idx, ["x", "y"]].to_numpy(float) * mpp
    shifts = shift_maps(xy, n_shift, min_shift_um, seed)
    rows = []
    for g, ps in rules.items():
        ps = [p for p in ps if p in prot]
        b = bmap.get(g)
        if g not in frac or not ps or b not in baseline:
            continue
        f = frac.loc[idx, g].to_numpy(float)
        m = baseline.loc[idx, b].to_numpy(float)
        P = prot.loc[idx, ps].to_numpy(float)

        def rho(v, ii=slice(None), jj=None):
            jj = ii if jj is None else jj
            return float(np.nanmean([_spearman(v[ii], P[jj, k]) for k in range(P.shape[1])]))

        rp, rb = rho(f), rho(m)

        def z(v, obs):
            nul = np.array([rho(v, ii, jj) for ii, jj in shifts], float)
            nul = nul[np.isfinite(nul)]
            return (obs - nul.mean()) / nul.std() if len(nul) > 2 and nul.std() > 0 else np.nan

        est, lo, hi, p = block_bootstrap(lambda ii: rho(f, ii) - rho(m, ii), xy, n_boot, block_um, seed)
        rows.append(dict(group=g, baseline=b, proteins=", ".join(ps), rho_pipeline=rp, rho_baseline=rb,
                         z_pipeline=z(f, rp), z_baseline=z(m, rb), difference=est, ci_low=lo, ci_high=hi,
                         p_boot=p, verdict="pipeline better" if lo > 0 else ("baseline better" if hi < 0 else "no difference")))
    return pd.DataFrame(rows)


def naive_calls(cross, n_groups: int, alpha: float = 0.05) -> pd.DataFrame:
    """Per pair: calls with textbook p-values (spots as independent samples, no nulls) vs the pipeline's calls."""
    from scipy.stats import t as tdist

    from .cross_groups import _bh

    rows, Pn, Po = [], [], []
    for i, pair in enumerate(cross.obs_names):
        n_out = float(cross.obs["n_outcome_nodes"].iloc[i])
        n_ov = float(cross.obs["n_overlap_nodes"].iloc[i])
        t = cross.layers["neighbor_t"][i].astype(float)
        Pn.append(2 * tdist.sf(np.abs(t), max(n_out - n_groups - 1, 1)))
        r = np.clip(cross.layers["overlap_r"][i].astype(float), -0.999999, 0.999999)
        to = r * np.sqrt(max(n_ov - 2, 1) / (1 - r ** 2))
        Po.append(2 * tdist.sf(np.abs(to), max(n_ov - 2, 1)))
    Qn = _bh(np.vstack(Pn).ravel()).reshape(len(Pn), -1)
    Qo = _bh(np.vstack(Po).ravel()).reshape(len(Po), -1)
    for i, pair in enumerate(cross.obs_names):
        rows.append(dict(pair=pair, neighbor_naive=int(np.nansum(Qn[i] <= alpha)),
                         neighbor_pipeline=int(np.nansum(cross.layers["neighbor_call"][i])),
                         overlap_naive=int(np.nansum(Qo[i] <= alpha)),
                         overlap_pipeline=int(np.nansum(cross.layers["overlap_call"][i]))))
    return pd.DataFrame(rows).set_index("pair")


def planted_simulation(nodes, group_a: str, group_b: str, n_genes: int = 20, effect: float = 1.0, seed: int = 0):
    """Node AnnData copy with simulated counts and a planted neighbour effect (module docstring, 13-4).
    effect: log-scale change of group A's expression of the planted genes per SD of B's neighbour share.
    Returns (AnnData, planted genes)."""
    from .cross_groups import _nb_dispersion, _neighbor_share

    groups = list(nodes.uns["nodes"]["groups"])
    a, b = groups.index(group_a), groups.index(group_b)
    rng = np.random.default_rng(seed)
    Y = nodes.layers["counts"]
    Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y, float)
    Nc = np.asarray(nodes.obsm["cell_counts_circle"], float)
    Ch = np.asarray(nodes.obsm["cell_counts_hex"], float)
    M = np.asarray(nodes.varm["group_signature"], float)          # genes × G
    expect = Nc @ M.T
    f = np.where(expect.sum(axis=1) > 0, Y.sum(axis=1) / np.maximum(expect.sum(axis=1), 1e-12), 1.0)
    comp_a = (Nc[:, [a]] * M[:, a][None, :]) * f[:, None]
    Mu = expect * f[:, None]
    phi = _nb_dispersion(Y, Mu)
    x = _neighbor_share(nodes.obsp["spatial_connectivities"], Ch)[:, b]
    zx = (x - x.mean()) / max(x.std(), 1e-12)
    # planted genes: expressed by group A (top signature share among the genes)
    share_a = M[:, a] / np.maximum(M.sum(axis=1), 1e-12)
    cand = np.argsort(-share_a * (M[:, a] > np.median(M[:, a])))[: max(4 * n_genes, n_genes)]
    planted = np.sort(rng.choice(cand, n_genes, replace=False))
    Mu2 = Mu.copy()
    Mu2[:, planted] += comp_a[:, planted] * (np.exp(effect * zx)[:, None] - 1)
    Mu2 = np.clip(Mu2, 0, None)
    pos = phi > 0
    lam = Mu2.copy()
    if pos.any():
        k = 1.0 / np.maximum(phi[pos], 1e-8)
        lam[:, pos] = rng.gamma(k[None, :], Mu2[:, pos] / k[None, :])
    from scipy import sparse
    sim = nodes.copy()
    for key in [k for k in sim.layers.keys() if k is not None and k != "counts"]:
        del sim.layers[key]
    sim.layers["counts"] = sparse.csr_matrix(rng.poisson(lam).astype(np.float32))
    return sim, [nodes.var_names[i] for i in planted]


def recovery(cross_sim, group_a: str, group_b: str, planted: Sequence[str]) -> Dict[str, float]:
    """Recall of the planted genes (neighbour statistic, A | B) and false calls elsewhere."""
    pair = f"{group_a}|{group_b}"
    i = list(cross_sim.obs_names).index(pair)
    called = cross_sim.layers["neighbor_call"][i] > 0
    genes = np.asarray(cross_sim.var_names)
    isp = np.isin(genes, list(planted))
    other = cross_sim.layers["neighbor_call"].copy()
    other[i, isp] = 0
    z = cross_sim.layers["neighbor_z_decomp"][i]
    return dict(pair=pair, planted=int(isp.sum()), recovered=int((called & isp).sum()),
                recall=float((called & isp).sum() / max(isp.sum(), 1)),
                false_calls_same_pair=int((called & ~isp).sum()), false_calls_other_pairs=int(np.nansum(other) - (called & ~isp).sum()),
                median_z_planted=float(np.nanmedian(z[isp])), median_z_other=float(np.nanmedian(z[~isp])))


def morphology_check(nodes, n_shift: int = 200, min_shift_um: float = 1000.0, seed: int = 0) -> pd.DataFrame:
    """Spearman ρ of every group's cell share with every node-morphology feature, against the shift null."""
    groups = list(nodes.uns["nodes"]["groups"])
    F = np.asarray(nodes.obsm["group_frac"], float)
    Mo = nodes.obsm["morphology"].astype(float)
    xy = np.asarray(nodes.obsm["spatial"], float)
    shifts = shift_maps(xy, n_shift, min_shift_um, seed)
    rows = []
    for gi, g in enumerate(groups):
        for c in Mo.columns:
            v = Mo[c].to_numpy(float)
            r = _spearman(F[:, gi], v)
            nul = np.array([_spearman(F[ii, gi], v[jj]) for ii, jj in shifts], float)
            nul = nul[np.isfinite(nul)]
            z = (r - nul.mean()) / nul.std() if len(nul) > 2 and nul.std() > 0 else np.nan
            rows.append(dict(group=g, feature=c, rho=r, z_shift=z))
    out = pd.DataFrame(rows)
    from scipy.stats import norm
    from .cross_groups import _bh
    out["q"] = _bh(2 * norm.sf(np.abs(out["z_shift"].to_numpy(float))))
    return out
