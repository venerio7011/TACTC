"""
trajectory.py
=============

Used by TACTC.ipynb 13-4: tumour-state scores, the OPC/NPC-like root and diffusion pseudotime (DPT) on a branch's
kNN graph.

Root (default: the OPC/NPC-like region)
  state scores (sc.tl.score_genes on the log-normalised node expression, programs.MARKERS), z-scored over nodes;
  candidates = nodes whose tumour-group cell share ≥ 0.5 (groups with 'tumor' in the name; all nodes if none or < 20);
  root = argmax over candidates of (OPC + NPC)/2 − max(AC, MES).
Pseudotime: diffusion map + DPT on the branch's own kNN graph (pre.obsp['<tag>_connectivities']).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .node_graphs import shift_maps

__all__ = ["STATE_SETS", "state_scores", "choose_root", "pseudotime"]

STATE_SETS = ("OPC-like", "NPC-like", "AC-like", "MES-like", "Hypoxia", "Cycling")


def _raw_adata(pre):
    import anndata as ad
    r = pre.raw.to_adata() if pre.raw is not None else pre
    return ad.AnnData(X=r.X, obs=pd.DataFrame(index=pre.obs_names), var=pd.DataFrame(index=r.var_names))


def state_scores(pre, markers: Optional[Dict[str, Sequence[str]]] = None, sets: Sequence[str] = STATE_SETS,
                 seed: int = 0) -> pd.DataFrame:
    """node × state score (sc.tl.score_genes on log-normalised expression), z-scored over nodes."""
    import scanpy as sc
    from .programs import MARKERS

    markers = markers or MARKERS
    a = _raw_adata(pre)
    out = {}
    for s in sets:
        genes = [g for g in markers.get(s, []) if g in a.var_names]
        if len(genes) < 2:
            continue
        sc.tl.score_genes(a, genes, score_name="_s", random_state=seed)
        v = a.obs["_s"].to_numpy(float)
        out[s] = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
    return pd.DataFrame(out, index=pre.obs_names)


def _tumour_share(nodes, names: Sequence[str]) -> Optional[np.ndarray]:
    groups = list(nodes.uns["nodes"]["groups"])
    tum = [i for i, g in enumerate(groups) if "tumor" in str(g).lower() or "tumour" in str(g).lower()]
    if not tum:
        return None
    pos = pd.Index(nodes.obs_names).get_indexer(names)
    Ch = np.asarray(nodes.obsm["cell_counts_hex"], float)[pos]
    return Ch[:, tum].sum(axis=1) / np.maximum(Ch.sum(axis=1), 1e-12)


def choose_root(pre, nodes, scores: pd.DataFrame, min_tumour_share: float = 0.5, n_alt: int = 10,
                verbose: bool = True) -> Dict[str, object]:
    """Root = argmax of (OPC + NPC)/2 − max(AC, MES) over tumour-rich nodes (module docstring)."""
    need = [s for s in ("OPC-like", "NPC-like") if s in scores]
    if not need:
        raise ValueError("No OPC-like / NPC-like marker genes found in the data; pass markers= or set the root by hand.")
    other = [s for s in ("AC-like", "MES-like") if s in scores]
    score = scores[need].mean(axis=1) - (scores[other].max(axis=1) if other else 0.0)
    share = _tumour_share(nodes, list(pre.obs_names))
    cand = np.ones(len(score), bool) if share is None else share >= min_tumour_share
    if cand.sum() < 20:
        cand = np.ones(len(score), bool)
    s = score.where(cand)
    order = s.dropna().sort_values(ascending=False)
    root = order.index[0]
    out = dict(root=root, root_index=int(pre.obs_names.get_loc(root)), score=score, candidates=cand,
               alternatives=list(order.index[1:n_alt + 1]), n_candidates=int(cand.sum()),
               tumour_groups_found=share is not None)
    if verbose:
        print(f"[15] root {root}: OPC/NPC − max(AC, MES) = {order.iloc[0]:.2f} | {int(cand.sum())} candidate nodes "
              f"({'tumour share ≥ ' + str(min_tumour_share) if share is not None else 'no tumour group: all nodes'})")
    return out


def pseudotime(pre, tag: str, root_index: int, n_dcs: int = 10) -> np.ndarray:
    """DPT on the branch's own kNN graph (diffusion map → dpt, scanpy defaults)."""
    import anndata as ad
    import scanpy as sc

    C = pre.obsp[f"{tag}_connectivities"].tocsr()
    t = ad.AnnData(X=np.zeros((pre.n_obs, 1), np.float32), obs=pd.DataFrame(index=pre.obs_names))
    t.obsp["connectivities"] = C
    t.obsp["distances"] = C.copy()
    t.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"method": "umap", "n_neighbors": int(pre.uns.get(f"{tag}_params", {}).get("n_neighbors", 15))}}
    sc.tl.diffmap(t, n_comps=max(n_dcs + 1, 15))
    t.uns["iroot"] = int(root_index)
    sc.tl.dpt(t, n_dcs=n_dcs)
    return t.obs["dpt_pseudotime"].to_numpy(float)


