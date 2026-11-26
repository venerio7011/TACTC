"""
programs.py
===========

Step 1 of TACTC.ipynb: reference-free deconvolution of Visium spots into K gene-expression programs with spOT-NMF
(optimal-transport NMF; Abdelkareem et al., bioRxiv 2025; PyPI 'spot-nmf', GPL-3.0). Pure Python (PyTorch), so no R.

Flow
  1. genes      : 'hvg' (top n_genes variable genes, scanpy) or 'overdispersed' (spOT-NMF's own GAM-based selection).
                  The OT ground cost is gene × gene, so memory grows with n_genes² (2,000 genes ≈ 32 MB per matrix).
  2. run        : spotnmf.models.run_spotnmf (or run_spotnmf_consensus over n_iter replicates, which also reports
                  cross-replicate program stability). Usages W (spot × program) sum to 1 per spot; spectra H
                  (gene × program) are distributions over genes.
  3. usage refit: spOT-NMF's own usages are softmin read-outs whose scale depends on the entropy regularisation
                  (they correlate with the true shares but are pulled toward equal shares). With usage_refit='nnls'
                  (default) each spot's normalised expression is refitted as a non-negative mixture of the learned
                  program spectra (as in spOT-NMF's consensus refit), giving calibrated RNA shares. 'ot' keeps the
                  raw usages (saved as theta_ot in both cases).
  4. Programs   : theta (spot × program), beta = Hᵀ (program × gene, rows sum to 1), cached in out_dir
  4. naming     : topic_summary (top genes, best marker sets, best proteins) → TOPIC_NAMES → name_topics (same name
                  = merged)
The usages are shares of the spot's (row-normalised) expression, i.e. of RNA, not of cells; cell_counts.py converts them.

Settings that mattered in a synthetic test (5 groups, 293 spots): with spOT-NMF's defaults (w = h = 0.01) and its
overdispersed genes the programs matched the true groups poorly (mean r 0.17); with all 400 genes and w = h = 0.1 they
matched well (mean r 0.95; STdeconvolve on the same data 0.62). The raw usages were mis-scaled (RNA-share error 0.135,
cell-share error after the RNA correction 0.128); the NNLS refit brought them to 0.017 and 0.030. Defaults here are
therefore w = h = 0.1, 2,000 variable genes and the NNLS refit. Check on real data with topic_summary and the protein
check.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["SPOT_DEFAULTS", "MARKERS", "run_spotnmf", "Programs", "topic_summary", "name_topics"]

SPOT_DEFAULTS = dict(k=8, gene_mode="hvg", n_genes=2000, extra_genes=(), w=0.1, h=0.1, eps=0.02, max_iter=100,
                     consensus=False, n_iter=10, seed=0, dtype="float32", usage_refit="nnls")

# marker sets for naming programs (glioblastoma; copied from visiumif2niche topics.GBM_MARKERS). Replace for other tissues.
MARKERS: Dict[str, List[str]] = {
    "Neuron": ["RBFOX3", "SNAP25", "SYT1", "GRIN1", "NRGN", "STMN2", "CHGA", "HPCA", "CPLX2", "SNCB", "GAD1", "GAD2", "SLC17A7", "VGF", "GABRA1"],
    "Oligodendrocyte": ["MBP", "PLP1", "MOG", "MAG", "MOBP", "ST18", "OPALIN", "CLDN11", "TMEM144", "ERMN"],
    "Microglia": ["P2RY12", "CX3CR1", "TMEM119", "SALL1", "CSF1R", "C1QA", "C1QB", "C1QC", "AIF1", "TYROBP", "CD74"],
    "Macrophage": ["CD163", "MRC1", "CD14", "FCGR3A", "LYZ", "S100A8", "S100A9", "VSIG4", "IFI30", "GPNMB", "TGFBI", "FCER1G"],
    "T/NK cell": ["CD3E", "CD3D", "CD2", "CD8A", "IL7R", "GZMK", "NKG7", "GZMB", "CCL5", "TRAC"],
    "B/Plasma cell": ["MS4A1", "CD79A", "CD19", "IGKC", "IGHG1", "JCHAIN", "MZB1"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "FLT1", "ESAM", "EMCN", "ADGRL4", "NOSTRIN", "ABCB1"],
    "Mural (pericyte/SMC)": ["RGS5", "PDGFRB", "ACTA2", "MYH11", "NOTCH3", "MYL9", "TAGLN", "CD248"],
    "Fibroblast/ECM": ["COL1A1", "COL1A2", "COL3A1", "COL6A3", "DCN", "LUM", "OGN", "FN1"],
    "Erythrocyte": ["HBB", "HBA1", "HBA2"],
    "MES-like": ["CHI3L1", "CD44", "VIM", "ANXA1", "SERPINE1", "TIMP1", "S100A10", "LGALS3", "CLIC1", "ANXA2"],
    "Hypoxia": ["VEGFA", "ADM", "NDRG1", "BNIP3", "CA9", "ERO1A", "PGK1", "HILPDA", "SLC2A1", "SPP1"],
    "AC-like": ["AQP4", "GFAP", "SLC1A3", "CST3", "S100B", "MLC1", "HEY1", "SPARCL1", "ALDOC", "GJA1", "NDRG2", "EGFR"],
    "OPC-like": ["OLIG1", "OLIG2", "PDGFRA", "BCAN", "NCAN", "DLL3", "SOX8", "LHFPL3", "GPR17", "PTPRZ1"],
    "NPC-like": ["SOX4", "SOX11", "DCX", "CD24", "DLX5", "DLX6-AS1", "STMN4", "ELAVL4", "TUBB2B", "MYCN"],
    "Cycling": ["MKI67", "TOP2A", "CENPF", "NUSAP1", "TYMS", "PCNA", "MCM2", "RRM2", "CDK1", "HIST1H1B"],
    "Interferon": ["ISG15", "IFI6", "IFIT1", "IFIT3", "MX1", "OAS1", "IFI44L", "RSAD2"],
    "Stress": ["FOS", "JUN", "HSPA1A", "HSPA1B", "HSPA6", "DNAJB1", "EGR1", "ATF3"],
}


@dataclass
class Programs:
    theta: pd.DataFrame            # spot × program RNA share (rows sum to 1; NNLS refit unless usage_refit='ot')
    beta: pd.DataFrame             # program × gene (rows sum to 1)
    k: int
    k_choice: Dict = field(default_factory=dict)
    params: Dict = field(default_factory=dict)
    theta_ot: Optional[pd.DataFrame] = None    # spOT-NMF's raw usages

    @property
    def topics(self) -> List[str]:
        return list(self.theta.columns)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.theta.to_csv(os.path.join(out_dir, "theta.csv"))
        self.beta.to_csv(os.path.join(out_dir, "beta.csv"))
        if self.theta_ot is not None:
            self.theta_ot.to_csv(os.path.join(out_dir, "theta_ot.csv"))
        json.dump(dict(k=self.k, k_choice=self.k_choice, params=self.params),
                  open(os.path.join(out_dir, "programs.json"), "w"), indent=1, default=str)

    @classmethod
    def load(cls, out_dir: str) -> "Programs":
        meta = json.load(open(os.path.join(out_dir, "programs.json")))
        ot = os.path.join(out_dir, "theta_ot.csv")
        return cls(theta=pd.read_csv(os.path.join(out_dir, "theta.csv"), index_col=0),
                   beta=pd.read_csv(os.path.join(out_dir, "beta.csv"), index_col=0),
                   k=int(meta["k"]), k_choice=meta.get("k_choice", {}), params=meta.get("params", {}),
                   theta_ot=pd.read_csv(ot, index_col=0) if os.path.exists(ot) else None)

    @staticmethod
    def exists(out_dir: str) -> bool:
        return all(os.path.exists(os.path.join(out_dir, f)) for f in ("theta.csv", "beta.csv", "programs.json"))


def _spots_hash(vis) -> str:
    h = hashlib.sha1("\n".join(vis.obs_names).encode())
    h.update(str(float(np.asarray(vis.X.sum()))).encode())
    return h.hexdigest()[:16]


def _genes(vis, mode: str, n_genes: int, extra: Sequence[str], verbose: bool) -> List[str]:
    if mode == "hvg":
        from .spot_nodes import select_genes
        return select_genes(vis, n_genes, extra, verbose=verbose)
    if mode == "overdispersed":
        import anndata as ad
        from spotnmf import hvg

        a = ad.AnnData(X=vis.X.copy(), obs=pd.DataFrame(index=vis.obs_names), var=pd.DataFrame(index=vis.var_names))
        with contextlib.redirect_stdout(io.StringIO()):
            _, od = hvg.compute_overdispersed_genes(a, save_dir=None, is_show=False, n_top_genes=None, verbose=False)
        genes = [g for g in vis.var_names if g in set(od)]
        genes += [g for g in extra if g in vis.var_names and g not in genes]
        if verbose:
            print(f"[spot] {len(genes):,} overdispersed genes (spOT-NMF selection)")
        return genes
    raise ValueError("gene_mode must be 'hvg' or 'overdispersed'")


def run_spotnmf(vis, out_dir: str, rerun: bool = False, device: Optional[str] = None, verbose: bool = True,
                **kw) -> Programs:
    """Run spOT-NMF on the post-QC spots (raw counts in vis.X) and return the programs. Cached in out_dir (reused when
    the settings and spots are the same; rerun=True forces a new fit).

    kw: k, gene_mode ('hvg' | 'overdispersed'), n_genes, extra_genes, w, h (entropy regularisation of usages / spectra;
        larger = smoother), eps (entropic OT smoothing), max_iter, consensus, n_iter, seed, dtype ('float32' | 'float64'),
        usage_refit ('nnls' calibrated RNA shares (default) | 'ot' raw spOT-NMF usages)
    device: None = GPU if available, else CPU
    """
    import anndata as ad

    p = dict(SPOT_DEFAULTS, **kw)
    p["extra_genes"] = list(p["extra_genes"])
    p.update(n_spots=int(vis.n_obs), spots_hash=_spots_hash(vis))
    keys = [k for k in p if k not in ("device",)]
    if not rerun and Programs.exists(out_dir):
        old = Programs.load(out_dir)
        if all(old.params.get(k) == p.get(k) for k in keys):
            if verbose:
                print(f"[spot] read the saved result ({out_dir}); rerun=True to refit")
            return old
    import torch
    from spotnmf import models

    genes = _genes(vis, p["gene_mode"], int(p["n_genes"]), p["extra_genes"], verbose)
    if len(genes) <= int(p["k"]):
        raise ValueError(f"Only {len(genes)} genes for k = {p['k']} programs.")
    X = vis[:, genes].X
    X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float64)
    keep = X.sum(axis=1) > 0
    b = ad.AnnData(X=X[keep], obs=pd.DataFrame(index=vis.obs_names[keep]),
                   var=pd.DataFrame({"highly_variable": True}, index=pd.Index(genes)))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mp = dict(w=float(p["w"]), h=float(p["h"]), eps=float(p["eps"]), max_iter=int(p["max_iter"]),
              dtype=p["dtype"], device=dev)
    if verbose:
        print(f"[spot] spOT-NMF k = {p['k']} on {int(keep.sum()):,} spots × {len(genes):,} genes ({dev}, {p['dtype']}"
              + (f", consensus over {p['n_iter']} replicates" if p["consensus"] else "") + ") …")
    stability = None
    with contextlib.redirect_stdout(io.StringIO()):
        if p["consensus"]:
            out = models.run_spotnmf_consensus(b, components=int(p["k"]), n_iter=int(p["n_iter"]), seed=int(p["seed"]),
                                               return_stability=True, **mp)
            res, losses, stability = out
        else:
            res, losses = models.run_spotnmf(b, components=int(p["k"]), seed=int(p["seed"]), **mp)
    W = res["topics_per_spot"].astype(float)
    H = res["genes_per_topic"].astype(float)
    names = [f"Program {i + 1}" for i in range(W.shape[1])]
    W.columns = names
    H.columns = names
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0)
    beta = H.T
    beta = beta.div(beta.sum(axis=1).replace(0, np.nan), axis=0)
    theta = W
    if p["usage_refit"] == "nnls":
        from scipy.optimize import nnls

        Bm = beta.fillna(0).to_numpy().T                                  # genes × programs
        Yn = X[keep] / X[keep].sum(axis=1, keepdims=True)
        A = np.array([nnls(Bm, yv)[0] for yv in Yn])
        A = A / np.where(A.sum(axis=1, keepdims=True) > 0, A.sum(axis=1, keepdims=True), 1.0)
        theta = pd.DataFrame(A, index=W.index, columns=names)
    elif p["usage_refit"] != "ot":
        raise ValueError("usage_refit must be 'nnls' or 'ot'")
    p["genes"] = genes
    p["device"] = dev
    p["n_outer_iterations"] = int(len(losses)) if losses is not None else None
    if stability is not None:
        p["stability"] = stability if isinstance(stability, (int, float)) else json.loads(json.dumps(stability, default=str))
    import importlib.metadata as md
    p["versions"] = {k: md.version(k) for k in ("spot-nmf", "torch")}
    out = Programs(theta=theta, beta=beta, k=int(p["k"]), k_choice={"chosen": int(p["k"])}, params=p, theta_ot=W)
    out.save(out_dir)
    if verbose:
        lost = vis.n_obs - len(W)
        print(f"[spot] {len(W):,} spots × {W.shape[1]} programs → {out_dir}"
              + (f" | {lost} spots without counts in the selected genes" if lost else "")
              + (f" | stability {stability}" if stability is not None and isinstance(stability, (int, float)) else ""))
    return out


def _lognorm(vis):
    """spot × gene log1p(counts per 10k) as a sparse CSC matrix."""
    from scipy import sparse
    X = sparse.csr_matrix(vis.X).astype(np.float64)
    tot = np.asarray(X.sum(axis=1)).ravel()
    X = sparse.diags(1e4 / np.maximum(tot, 1)) @ X
    X.data = np.log1p(X.data)
    return X.tocsc()


def topic_summary(res: Programs, vis, markers: Optional[Dict[str, Sequence[str]]] = None, n_genes: int = 12,
                  min_markers: int = 2) -> pd.DataFrame:
    """Per topic: mean proportion, spots where it is the largest topic, top genes (beta), best marker sets and best
    proteins (Spearman of theta with the marker-set score / protein across spots). For naming topics (TOPIC_NAMES)."""
    from scipy.stats import spearmanr

    markers = MARKERS if markers is None else markers
    th = res.theta
    spots = th.index.intersection(vis.obs_names)
    th = th.loc[spots]
    sub = vis[spots]
    X = _lognorm(sub)
    gi = {g: i for i, g in enumerate(sub.var_names)}
    scores = {}
    for name, gs in markers.items():
        idx = [gi[g] for g in gs if g in gi]
        if len(idx) >= min_markers:
            scores[name] = np.asarray(X[:, idx].mean(axis=1)).ravel()
    prot = None
    if "protein" in sub.obsm:
        P = sub.obsm["protein"]
        P = P[[c for c in P.columns if not str(c).lower().startswith(("mouse_", "rat_", "isotype"))]]
        prot = np.log1p(P.div(P.sum(axis=1).replace(0, 1), axis=0) * 1e4)
    dom = th.idxmax(axis=1)
    rows = []
    for t in th.columns:
        v = th[t].to_numpy()
        r = {"topic": t, "mean_prop": round(float(v.mean()), 4), "dominant_spots": int((dom == t).sum()),
             "top_genes": ", ".join(res.beta.loc[t].sort_values(ascending=False).index[:n_genes])}
        rr = {k: spearmanr(v, s).statistic for k, s in scores.items()}
        best = sorted(rr.items(), key=lambda kv: -np.nan_to_num(kv[1], nan=-9))[:3]
        r["best_markers"] = ", ".join(f"{k} {x:.2f}" for k, x in best)
        if prot is not None:
            pr = {c: spearmanr(v, prot[c].to_numpy()).statistic for c in prot.columns}
            bp = sorted(pr.items(), key=lambda kv: -np.nan_to_num(kv[1], nan=-9))[:3]
            r["best_proteins"] = ", ".join(f"{k} {x:.2f}" for k, x in bp)
        rows.append(r)
    return pd.DataFrame(rows).set_index("topic")


def name_topics(theta: pd.DataFrame, names: Optional[Dict[str, str]] = None, drop: Sequence[str] = ()) -> pd.DataFrame:
    """spot × group shares: topics renamed by names (unnamed topics keep their name), topics with the same name summed,
    topics (or names) in drop removed and the rest renormalised to 1."""
    names = names or {}
    g = pd.Series({t: names.get(t, t) for t in theta.columns})
    keep = [t for t in theta.columns if t not in drop and g[t] not in drop]
    Q = theta[keep].T.groupby(g[keep].to_numpy(), sort=False).sum().T
    s = Q.sum(axis=1)
    return Q.div(s.where(s > 0), axis=0)


