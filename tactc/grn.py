"""
grn.py
======

Step 12 of TACTC.ipynb: per-group gene regulatory networks with pySCENIC (GRNBoost2 → cisTarget → AUCell) on the
spot nodes, then a prioritisation that adds pre-specified bonus points from steps 9 and 11.

Units (what one "sample" is)
  RNA is measured per spot, so a per-cell network cannot be identified: cells of one group in one node share the same
  estimate. The unit is therefore (node × group): for group g, every node with ≥ min_cells cells of g, with g's local
  per-cell expression (step-6 local signatures, re-estimated here for the GRN gene set), normalised per unit
  (counts per 10k → log1p). visiumif2niche used the same unit ((spot × type) profiles, 13-1); normalising per unit
  removes factors shared by all genes of a unit, which otherwise correlate unrelated gene pairs.

Network (visiumif2niche 13-1, same settings)
  genes   : candidates = the n_candidates most variable genes of the spots + all detected TFs (allTFs_hg38);
            per group: targets = top n_top by variance among genes detected in ≥ min_frac of its units,
            regulators = top n_tf by variance among detected TFs
  GRNBoost2 : arboreto's per-target gradient boosting (same settings / early stopping), run in parallel with joblib
              (arboreto's dask scheduler is incompatible with recent dask)
  cisTarget : pySCENIC 0.12.1 prune2df / df2regulons with the hg38 v10 clust ranking databases (±10 kb, −500/+100 bp),
              NES ≥ 3; regulon = TF + motif-supported targets (leading edge)
  AUCell    : per unit, area under the recovery curve of the regulon targets within the top 5% of the unit's ranking
Checkpoints per group (out_dir) so a disconnected runtime resumes.

Region specificity (base score)
  For regulon k of group g and step-11 region r: d = (mean AUC of g's units in r − mean outside) / SD, against the
  shift null of the region map (regions rebuilt from shifted cluster labels, as in step 11) → z_region.

Prioritisation (bonus points, weights fixed before looking at the ranking; components are reported)
  bonus_9  (step 9, cross-group AnnData): for group g, genes with a neighbour or overlap call (both nulls) where g is
           the outcome / a partner. = tf_bonus (1 if the regulon's TF is called) + min(−log10 p, cap) of the
           hypergeometric enrichment of called genes among the regulon's targets.
  bonus_11 (step 11, region AnnData): for region r and group g, the largest positive z among enrichment,
           edge_density and cross_weight that has q ≤ 0.05 (0 otherwise), capped.
  total = z_region + [z_region > 0] · (w9 · bonus_9 + w11 · bonus_11): the bonuses only lift a regulon in a region where
  it is more active than outside; rows without a base score are not ranked. Rank by total (and by z_region alone).
Output: a separate AnnData — obs = regulons (group | regulon), var = regions, X = total, layers z_region, d,
bonus_9, bonus_11; obs: group, tf, n_targets, tf_called, n_targets_called, partners; uns: tables and parameters.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["CISTARGET", "download_cistarget", "candidate_genes", "group_profiles", "select_genes", "grnboost2",
           "prune", "aucell", "run_grn", "GRNResult"]

_BASE = "https://resources.aertslab.org/cistarget"
CISTARGET = {
    "db_10kb": f"{_BASE}/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/"
               "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
    "db_500bp": f"{_BASE}/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/"
                "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
    "motifs": f"{_BASE}/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
    "tfs": f"{_BASE}/tf_lists/allTFs_hg38.txt",
}


def _compat():
    """Make pySCENIC 0.12.1 work with current NumPy / pandas:
    * NumPy 2 removed np.object / np.bool / … → aliases restored before import (visiumif2niche regulons._compat)
    * pandas ≥ 3 no longer passes the grouping column to groupby().apply(), which breaks
      pyscenic.utils.modules4top_factors (KeyError 'target') → replaced by an equivalent version without apply
      (per target the n largest weights: sort by weight, then groupby().head(n))."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, typ in (("object", object), ("bool", bool), ("int", int), ("float", float)):
            if not hasattr(np, name):
                setattr(np, name, typ)
    import pyscenic.utils as pu

    if getattr(pu, "_tactc_patched", False):
        return

    def modules4top_factors(adjacencies, n, context=frozenset()):
        df = (adjacencies.sort_values(pu.COLUMN_NAME_WEIGHT, ascending=False, kind="stable")
              .groupby(pu.COLUMN_NAME_TARGET, sort=False).head(n))
        for tf_name, df_grp in df.groupby(by=pu.COLUMN_NAME_TF):
            if len(df_grp) > 0:
                yield pu.Regulon(name=tf_name, context=frozenset([f"top{n}perTarget"]).union(context),
                                 transcription_factor=tf_name,
                                 gene2weight=list(zip(df_grp[pu.COLUMN_NAME_TARGET].values,
                                                      df_grp[pu.COLUMN_NAME_WEIGHT].values)),
                                 gene2occurrence=[])

    pu.modules4top_factors = modules4top_factors
    pu._tactc_patched = True


def download_cistarget(cache_dir: str, resources: Optional[Dict[str, str]] = None, verbose: bool = True) -> Dict[str, str]:
    """Download the cisTarget resources into cache_dir once (Drive). resources = {key: url or local path} to override."""
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for k, url in dict(CISTARGET, **(resources or {})).items():
        if os.path.exists(url):
            out[k] = url
            continue
        path = os.path.join(cache_dir, os.path.basename(url))
        if not os.path.exists(path):
            if verbose:
                print(f"[cistarget] downloading {os.path.basename(url)}")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        out[k] = path
    if verbose:
        print("[cistarget] " + ", ".join(f"{k}: {os.path.getsize(v) / 1e6:.0f} MB" for k, v in out.items()))
    return out


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def candidate_genes(spots, tf_names: Sequence[str], n_candidates: int = 4000, min_frac: float = 0.05,
                    verbose: bool = True) -> List[str]:
    """The n_candidates most variable genes of the spots (log CP10k variance among genes detected in ≥ min_frac of
    spots) + all detected TFs."""
    X = spots.X
    det = np.asarray((X > 0).mean(axis=0)).ravel()
    tot = np.asarray(X.sum(axis=1)).ravel()
    from scipy import sparse
    Xn = sparse.diags(1e4 / np.maximum(tot, 1)) @ sparse.csr_matrix(X).astype(float)
    Xn.data = np.log1p(Xn.data)
    mean = np.asarray(Xn.mean(axis=0)).ravel()
    var = np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel() - mean ** 2
    ok = det >= min_frac
    names = np.asarray(spots.var_names)
    order = np.argsort(-np.where(ok, var, -np.inf))
    top = [names[i] for i in order[:int(n_candidates)] if ok[i]]
    tfset = set(tf_names)
    tfs = [g for g, o in zip(names, ok) if o and g in tfset and g not in set(top)]
    if verbose:
        print(f"[grn] candidate genes: {len(top):,} variable + {len(tfs):,} additional detected TFs "
              f"({sum(g in tfset for g in top) + len(tfs):,} TFs in total)")
    return top + tfs


def group_profiles(spots, nodes, genes: Sequence[str], min_cells: int = 1, min_units: int = 100,
                   radius_um: float = 250.0, sigma_um: float = 100.0, ridge: float = 0.1, verbose: bool = True):
    """Per group: DataFrame unit (node) × gene of log1p(CP10k) of the group's local per-cell expression.
    Returns ({group: DataFrame}, signatures gene × group)."""
    from .node_graphs import local_signatures
    from .spot_nodes import group_signatures

    groups = list(nodes.uns["nodes"]["groups"])
    s = spots[nodes.obs_names, list(genes)]
    Y = _dense(s.X).astype(float)
    Nc = pd.DataFrame(np.asarray(nodes.obsm["cell_counts_circle"], float), index=nodes.obs_names, columns=groups)
    sig = group_signatures(s, Nc, list(genes), verbose=False)
    local = local_signatures(Y, Nc.to_numpy(), np.asarray(nodes.obsm["spatial"], float), sig.to_numpy(),
                             radius_um, sigma_um, ridge)
    Ch = np.asarray(nodes.obsm["cell_counts_hex"], float)
    out = {}
    for g, name in enumerate(groups):
        m = Ch[:, g] >= min_cells
        if m.sum() < min_units:
            if verbose:
                print(f"[grn] {name}: {int(m.sum())} units < {min_units} → skipped")
            continue
        P = local[g][m].astype(float)
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12) * 1e4
        out[name] = pd.DataFrame(np.log1p(P).astype(np.float32), index=nodes.obs_names[m], columns=list(genes))
    del local
    if verbose:
        print(f"[grn] units per group: " + ", ".join(f"{k} {len(v):,}" for k, v in out.items()))
    return out, sig


def select_genes(df: pd.DataFrame, tf_names: Sequence[str], n_top: int = 3000, n_tf: int = 500,
                 min_frac: float = 0.05):
    """(targets, regulators): top n_top by variance among genes detected in ≥ min_frac of units; top n_tf TFs."""
    det = (df > 0).mean()
    ok = det.index[det >= min_frac]
    var = df[ok].var().sort_values(ascending=False)
    tfset = set(tf_names)
    regs = [g for g in var.index if g in tfset][:n_tf]
    return list(var.index[:n_top]), regs


# ---------------------------------------------------------------------------
# GRNBoost2 / cisTarget / AUCell (visiumif2niche regulons.py)
# ---------------------------------------------------------------------------
def _one_target(tf_matrix, tf_names, target, y, seed):
    import arboreto.core as ac

    tm, tn = ac.clean(tf_matrix, list(tf_names), target)
    return ac.infer_partial_network(regressor_type="GBM", regressor_kwargs=ac.SGBM_KWARGS, tf_matrix=tm,
                                    tf_matrix_gene_names=tn, target_gene_name=target, target_gene_expression=y,
                                    include_meta=False, early_stop_window_length=ac.EARLY_STOP_WINDOW_LENGTH, seed=seed)


def grnboost2(df: pd.DataFrame, tf_names: Sequence[str], targets: Optional[Sequence[str]] = None, n_jobs: int = -1,
              seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """GRNBoost2 (arboreto settings), parallel over targets with joblib. Returns TF, target, importance."""
    from joblib import Parallel, delayed

    tfs = [t for t in tf_names if t in df.columns]
    targets = list(targets or df.columns)
    tf_matrix = df[tfs].to_numpy(np.float32)
    t0 = time.time()
    parts = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_one_target)(tf_matrix, tfs, g, df[g].to_numpy(np.float32), seed) for g in targets)
    adj = pd.concat(parts, ignore_index=True).sort_values("importance", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"[grnboost2] {len(df):,} units | {len(targets):,} targets × {len(tfs)} TFs → {len(adj):,} links "
              f"({time.time() - t0:.0f}s)")
    return adj


def prune(adj: pd.DataFrame, df: pd.DataFrame, res: Dict[str, str], n_workers: int = 2, nes_threshold: float = 3.0,
          db_keys: Sequence[str] = ("db_10kb", "db_500bp"), rank_threshold: int = 1500, verbose: bool = True):
    """Co-expression modules → motif-enrichment pruning (pySCENIC cisTarget) → regulons.
    Returns (regulon table: regulon, tf, target, weight; motif table)."""
    _compat()
    from ctxcore.rnkdb import FeatherRankingDatabase
    from pyscenic.prune import df2regulons, prune2df
    from pyscenic.utils import modules_from_adjacencies

    t0 = time.time()
    modules = modules_from_adjacencies(adj, df)
    dbs = [FeatherRankingDatabase(fname=res[k], name=os.path.basename(res[k])) for k in db_keys if k in res]
    rt = int(min(rank_threshold, min(db.total_genes for db in dbs) - 1))      # pySCENIC prune2df default 1500 (capped for small DBs)
    motifs = prune2df(dbs, modules, res["motifs"], nes_threshold=nes_threshold, num_workers=max(n_workers, len(dbs)),
                      client_or_address="custom_multiprocessing", rank_threshold=rt)
    regs = df2regulons(motifs) if len(motifs) else []
    rows = [dict(regulon=r.name, tf=r.transcription_factor, target=g, weight=float(w))
            for r in regs for g, w in r.gene2weight.items()]
    tab = pd.DataFrame(rows, columns=["regulon", "tf", "target", "weight"])
    if verbose:
        print(f"[cistarget] {len(modules):,} modules → {len(motifs):,} motif enrichments → {len(regs)} regulons "
              f"({time.time() - t0:.0f}s)")
    return tab, motifs


def aucell(df: pd.DataFrame, gene_sets: Dict[str, Sequence[str]], auc_frac: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """AUCell: per unit, area under the recovery curve of the set's genes within the top auc_frac of the unit's ranking,
    divided by the maximum possible area (0–1). Ties broken at random."""
    X = df.to_numpy(np.float32)
    n_units, G = X.shape
    max_rank = max(1, int(np.ceil(auc_frac * G)))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(X.shape), -X), axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n_units)[:, None], order] = np.arange(G)[None, :]
    cols = pd.Index(df.columns)
    out = {}
    for name, genes in gene_sets.items():
        gi = cols.get_indexer([g for g in genes if g in cols])
        gi = gi[gi >= 0]
        if not len(gi):
            continue
        r = rank[:, gi]
        area = np.where(r < max_rank, max_rank - r, 0).sum(axis=1)
        out[name] = area / (min(len(gi), max_rank) * max_rank)
    return pd.DataFrame(out, index=df.index)


@dataclass
class GRNResult:
    regulons: pd.DataFrame                  # group, regulon, tf, target, weight
    summary: pd.DataFrame                   # per group: units, targets, tfs, links, regulons
    activity: Dict[str, pd.DataFrame]       # group → unit × regulon AUCell
    params: Dict = field(default_factory=dict)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.regulons.to_csv(os.path.join(out_dir, "regulons.csv"), index=False)
        self.summary.to_csv(os.path.join(out_dir, "grn_summary.csv"))
        for g, a in self.activity.items():
            a.to_csv(os.path.join(out_dir, f"aucell_{_safe(g)}.csv.gz"))
        json.dump(dict(params=self.params, groups=list(self.activity)), open(os.path.join(out_dir, "grn_params.json"), "w"),
                  indent=1, default=str)
        print(f"[save] GRN → {out_dir} (regulons.csv, grn_summary.csv, aucell_<group>.csv.gz)")

    @classmethod
    def load(cls, out_dir: str) -> "GRNResult":
        m = json.load(open(os.path.join(out_dir, "grn_params.json")))
        act = {g: pd.read_csv(os.path.join(out_dir, f"aucell_{_safe(g)}.csv.gz"), index_col=0) for g in m["groups"]}
        return cls(pd.read_csv(os.path.join(out_dir, "regulons.csv")),
                   pd.read_csv(os.path.join(out_dir, "grn_summary.csv"), index_col=0), act, m.get("params", {}))


def _safe(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


class _Checkpoint:
    def __init__(self, out_dir: Optional[str], group: str, key: Dict):
        self.dir = os.path.join(out_dir, "checkpoints", _safe(group)) if out_dir else None
        self.key = json.dumps(key, sort_keys=True, default=str)

    def _p(self, name):
        return os.path.join(self.dir, f"{name}.csv.gz")

    def read(self, name):
        if not self.dir or not os.path.exists(self._p(name)) or not os.path.exists(os.path.join(self.dir, "key.json")):
            return None
        if open(os.path.join(self.dir, "key.json")).read() != self.key:
            return None
        return pd.read_csv(self._p(name))

    def write(self, name, df):
        if not self.dir:
            return
        os.makedirs(self.dir, exist_ok=True)
        df.to_csv(self._p(name), index=False)
        with open(os.path.join(self.dir, "key.json"), "w") as f:
            f.write(self.key)


def run_grn(profiles: Dict[str, pd.DataFrame], res: Dict[str, str], groups: Optional[Sequence[str]] = None,
            n_top: int = 3000, n_tf: int = 500, min_frac: float = 0.05, nes_threshold: float = 3.0,
            min_targets: int = 10, auc_frac: float = 0.05, n_jobs: int = -1, n_workers: int = 2, seed: int = 0,
            out_dir: Optional[str] = None, verbose: bool = True) -> GRNResult:
    """Per group: GRNBoost2 → cisTarget → AUCell (module docstring). out_dir: checkpoints per group."""
    tf_all = [l.strip() for l in open(res["tfs"]) if l.strip()]
    regs, summ, act = [], [], {}
    for g in (groups or list(profiles)):
        df = profiles[g]
        targets, tfs = select_genes(df, tf_all, n_top, n_tf, min_frac)
        genes = list(dict.fromkeys(targets + tfs))
        sub = df[genes]
        if verbose:
            print(f"--- {g}: {len(sub):,} units, {len(targets):,} targets, {len(tfs)} regulator TFs")
        if len(tfs) < 2:
            print(f"  skipped: fewer than 2 regulator TFs")
            continue
        ck = _Checkpoint(out_dir, g, dict(units=len(sub), targets=sorted(targets), tfs=sorted(tfs), seed=seed,
                                          nes=nes_threshold))
        adj = ck.read("adj")
        if adj is None:
            adj = grnboost2(sub, tfs, targets=targets, n_jobs=n_jobs, seed=seed, verbose=verbose)
            ck.write("adj", adj)
        elif verbose:
            print(f"[checkpoint] {g}: {len(adj):,} GRNBoost2 links read")
        tab = ck.read("regulons")
        if tab is None:
            tab, _ = prune(adj, sub, res, n_workers=n_workers, nes_threshold=nes_threshold, verbose=verbose)
            ck.write("regulons", tab)
        elif verbose:
            print(f"[checkpoint] {g}: {tab['regulon'].nunique()} regulons read")
        sets = {k: list(v) for k, v in tab.groupby("regulon")["target"] if len(v) >= min_targets}
        act[g] = aucell(df, sets, auc_frac, seed) if sets else pd.DataFrame(index=df.index)
        regs.append(tab.assign(group=g))
        summ.append(dict(group=g, units=len(sub), targets=len(targets), tfs=len(tfs), links=len(adj),
                         regulons=int(tab["regulon"].nunique()), regulons_scored=len(sets)))
    R = pd.concat(regs, ignore_index=True) if regs else pd.DataFrame(columns=["regulon", "tf", "target", "weight", "group"])
    return GRNResult(R[["group", "regulon", "tf", "target", "weight"]],
                     pd.DataFrame(summ).set_index("group") if summ else pd.DataFrame(), act,
                     dict(n_top=n_top, n_tf=n_tf, min_frac=min_frac, nes_threshold=nes_threshold,
                          min_targets=min_targets, auc_frac=auc_frac, seed=seed,
                          databases=[os.path.basename(res[k]) for k in ("db_10kb", "db_500bp") if k in res]))


# ---------------------------------------------------------------------------
# region specificity and prioritisation
# ---------------------------------------------------------------------------
