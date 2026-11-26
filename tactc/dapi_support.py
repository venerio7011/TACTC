"""
dapi_support.py
===============

Steps 11 and 13 of TACTC.ipynb: which parts of the expression clusters (PAGA, step 10, branch 'std') are supported by
the DAPI cell graph, and whether the unsupported parts behave like false positives.

Idea: expression (PAGA) and cell positions (DAPI graph) are built independently. Where they agree —
nodes of one expression cluster that form a connected patch of tissue in the DAPI graph — the cluster is supported
("core", PAGA ∩ DAPI). Nodes of a cluster that sit in small, isolated pieces are expression-only ("scattered",
PAGA only) and are the candidate false positives. Step 13 tests that prediction with measurements that were not used
to define the sets.

11-1 DAPI node graph
     Two spot nodes (hexagons) are linked when at least `min_edges` edges of the DAPI cell graph (step 2: Delaunay
     between nuclei, ≤ 30 µm) join a cell of one hexagon to a cell of the other. Gaps without cells (tissue tears,
     necrosis, the section edge) therefore break the link, unlike the plain hexagon lattice.
11-2 Core / scattered nodes (permutation null)
     For cluster c, keep only c's nodes and their DAPI links; each node gets the size of its connected component.
     Null: the same number of nodes placed at random (n_perm permutations) → node-level component sizes.
     p = (1 + #null ≥ observed) / (1 + #null); core if p ≤ alpha, else scattered. A shift null is not used here:
     shifting the label map keeps its patches, so it cannot ask whether a cluster forms patches at all.
11-3 Tissue-annotated PAGA (descriptive)
     For clusters i, j: DAPI links between core nodes of i and core nodes of j, also as a share of the smaller core
     boundary, and separately the links and nodes that involve scattered (fragment) nodes. A PAGA edge whose cores
     touch (≥ min_links and ≥ min_share) is 'supported'; one whose contact is made mainly by fragments is 'through
     fragments'; otherwise 'weak contact' or 'no contact'. boundary_nodes / boundary_neighbors list which clusters
     the boundary nodes touch. No null is used, because a shift null makes two disjoint clusters overlap and a label
     permutation is dominated by the scattered pieces.
11-4 Planted controls (known answer)
     A random scattered set of nodes and a contiguous patch of the same size are given new labels; the classifier
     must call the patch core and the random set scattered. If it does not, the step-13 comparison is not valid.
13   Validation of core vs scattered (features not used to define the sets)
     13-1 expression reproducibility: re-cluster on one side of a random cut, transfer the labels to the other side,
          match them to the full labels; per-node agreement, core vs scattered (also among cluster-boundary nodes only).
     13-2 independent measurements: k-NN prediction of the cluster from Visium proteins and node morphology,
          cross-validated with whole spatial blocks held out (plus a buffer) so that tissue neighbours do not leak
          into training; accuracy core vs scattered, overall and among boundary nodes, with a label-permutation
          chance level and a spatial block bootstrap CI. Random folds are reported alongside for comparison.
     13-3 regulons: regulon_eval.exclusion (C1–C3, naive test) on regions "<cluster>|core" and "<cluster>|scattered";
          share of tested pairs with |z| ≥ 2, core vs scattered (pairs without a z are not counted).
     13-4 trajectory: PAGA edges with tissue support (11-3); pseudotime on all nodes vs on core nodes only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .node_graphs import _dense, _transfer, random_cuts

__all__ = ["node_dapi_graph", "core_scattered", "edge_support", "planted_controls", "boundary_nodes", "boundary_neighbors",
           "support_labels", "label_reproducibility", "independent_agreement", "regulon_support", "regulon_summary",
           "pseudotime_core"]


# ---------------------------------------------------------------------------
# 11-1 DAPI node graph
# ---------------------------------------------------------------------------
def node_dapi_graph(cells_ad, node_names: Sequence[str], min_edges: int = 1, verbose: bool = True):
    """(W, A): W[s, t] = number of DAPI cell-graph edges between cells of hexagon s and hexagon t (s ≠ t);
    A = W ≥ min_edges (0/1, symmetric). cells_ad: step-2–4 cell AnnData with obs['spot'] and
    obsp['spatial_connectivities']."""
    names = pd.Index([str(x) for x in node_names])
    pos = names.get_indexer(cells_ad.obs["spot"].astype(str).to_numpy())
    C = sparse.triu(cells_ad.obsp["spatial_connectivities"], k=1).tocoo()
    a, b = pos[C.row], pos[C.col]
    ok = (a >= 0) & (b >= 0) & (a != b)
    n = len(names)
    W = sparse.coo_matrix((np.ones(int(ok.sum())), (a[ok], b[ok])), shape=(n, n)).tocsr()
    W = (W + W.T).tocsr()
    A = (W >= min_edges).astype(np.int8).tocsr()
    A.eliminate_zeros()
    if verbose:
        deg = np.asarray(A.sum(axis=1)).ravel()
        ncomp, _ = connected_components(A, directed=False)
        print(f"[11-1] DAPI node graph: {n:,} nodes, {A.nnz // 2:,} links (≥ {min_edges} cell edges) | "
              f"median degree {np.median(deg):.0f} | isolated {int((deg == 0).sum())} | components {ncomp}")
    return W, A


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _codes(labels) -> tuple:
    lab = np.asarray(labels).astype(str)
    cats = sorted(pd.unique(lab), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    return pd.Categorical(lab, categories=cats).codes.astype(np.int64), list(cats)


def _comp_sizes(A, code: np.ndarray) -> np.ndarray:
    """Size of the connected component of every labelled node (code ≥ 0) in the graph of A restricted to edges
    between nodes with the same code. NaN for unlabelled nodes."""
    C = A.tocoo()
    keep = (code[C.row] >= 0) & (code[C.row] == code[C.col])
    G = sparse.coo_matrix((np.ones(int(keep.sum())), (C.row[keep], C.col[keep])), shape=A.shape).tocsr()
    _, cid = connected_components(G, directed=False)
    size = np.bincount(cid)[cid].astype(float)
    size[code < 0] = np.nan
    return size


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(pv)
        out[o] = np.clip(r, 0, 1)
        q[ok] = out
    return q


def boundary_nodes(A, labels) -> np.ndarray:
    """True for nodes with at least one DAPI link to a node of another cluster."""
    code, _ = _codes(labels)
    C = A.tocoo()
    diff = code[C.row] != code[C.col]
    out = np.zeros(A.shape[0], bool)
    out[C.row[diff]] = True
    return out


# ---------------------------------------------------------------------------
# 11-2 core / scattered
# ---------------------------------------------------------------------------
def core_scattered(A, labels, n_perm: int = 200, alpha: float = 0.05, seed: int = 0,
                   verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Per node: cluster, component size, permutation p, set ('core' | 'scattered'); per cluster: summary."""
    code, cats = _codes(labels)
    n = len(code)
    obs = _comp_sizes(A, code)
    rng = np.random.default_rng(seed)
    p = np.full(n, np.nan)
    rows = []
    for k, c in enumerate(cats):
        members = np.flatnonzero(code == k)
        m = len(members)
        null = []
        for _ in range(n_perm):
            pc = np.full(n, -1, np.int64)
            pc[rng.choice(n, m, replace=False)] = 0
            s = _comp_sizes(A, pc)
            null.append(s[pc == 0])
        null = np.sort(np.concatenate(null))
        ge = len(null) - np.searchsorted(null, obs[members], side="left")
        p[members] = (1 + ge) / (1 + len(null))
        thr = float(np.quantile(null, 1 - alpha))
        core = p[members] <= alpha
        rows.append(dict(cluster=c, nodes=m, core=int(core.sum()), scattered=int((~core).sum()),
                         core_share=float(core.mean()) if m else np.nan, null_size_q95=thr,
                         largest_component=float(np.nanmax(obs[members])) if m else np.nan,
                         components=int(len(np.unique(np.round(obs[members], 6)))) if m else 0))
    node = pd.DataFrame(dict(cluster=np.asarray(cats, object)[code], component_size=obs, p_component=p,
                             set=np.where(p <= alpha, "core", "scattered")))
    node["boundary"] = boundary_nodes(A, labels)
    summ = pd.DataFrame(rows)
    if verbose:
        print(f"[11-2] core (PAGA ∩ DAPI) {int((node['set'] == 'core').sum()):,} nodes | scattered (PAGA only) "
              f"{int((node['set'] == 'scattered').sum()):,} | permutations {n_perm}, alpha {alpha}")
    return dict(nodes=node, clusters=summ)


def support_labels(labels, sets) -> np.ndarray:
    """'<cluster>|core' / '<cluster>|scattered' per node."""
    return np.char.add(np.char.add(np.asarray(labels).astype(str), "|"), np.asarray(sets).astype(str))


# ---------------------------------------------------------------------------
# 11-3 PAGA edges in the tissue (contact between core regions)
# ---------------------------------------------------------------------------
def edge_support(A, labels, sets, paga: Optional[pd.DataFrame] = None, paga_threshold: float = 0.03,
                 min_links: int = 10, min_share: float = 0.02, verbose: bool = True) -> pd.DataFrame:
    """Tissue-annotated PAGA: contact in the tissue between every pair of clusters, split by whether the contact is
    made by core nodes or by scattered (fragment) nodes.
    links           = DAPI links between a core node of i and a core node of j
    share           = links / the smaller of the two clusters' core boundary links (links from core nodes to any
                      other cluster's core nodes)
    touching        = links ≥ min_links and share ≥ min_share  (the cores touch)
    fragment_links  = DAPI links between i and j with at least one scattered end (core–scattered or
                      scattered–scattered)
    fragment_nodes  = scattered nodes of i or j with at least one DAPI link to the other cluster
    tissue_contact  = 'cores touch' if touching; else 'through fragments' if fragment_links > links (the contact
                      that exists is made mainly by fragments); else 'weak' if links + fragment_links > 0; else 'none'
    link_mark       = for PAGA edges: 'supported' | 'through fragments' | 'weak contact' | 'no contact';
                      for other pairs: 'contact without PAGA edge' when the cores touch, else ''
    edge_type       = the original four classes ('PAGA ∩ DAPI', 'PAGA only', 'DAPI only', 'neither'), unchanged
    Descriptive (no null): a shift null would overlap two disjoint clusters, and a label permutation is dominated
    by the scattered pieces."""
    code, cats = _codes(labels)
    K = len(cats)
    core = np.asarray(sets).astype(str) == "core"
    C = sparse.triu(A, k=1).tocoo()                   # each link once
    ci, cj = code[C.row], code[C.col]
    diff = ci != cj
    both_core = core[C.row] & core[C.col]
    M = np.zeros((K, K))
    F = np.zeros((K, K))
    np.add.at(M, (ci[diff & both_core], cj[diff & both_core]), 1.0)
    np.add.at(F, (ci[diff & ~both_core], cj[diff & ~both_core]), 1.0)
    M, F = M + M.T, F + F.T
    bnd = M.sum(axis=1)
    # scattered nodes of cluster a with a link to cluster b
    Cf = A.tocoo()
    sel = (code[Cf.row] != code[Cf.col]) & ~core[Cf.row]
    pairs = np.unique(np.c_[Cf.row[sel], code[Cf.col[sel]]], axis=0) if sel.any() else np.zeros((0, 2), np.int64)
    N = np.zeros((K, K))
    if len(pairs):
        np.add.at(N, (code[pairs[:, 0]], pairs[:, 1]), 1.0)
    rows = []
    for a in range(K):
        for b in range(a + 1, K):
            links = float(M[a, b])
            share = links / max(min(bnd[a], bnd[b]), 1.0)
            conn = float(paga.loc[cats[a], cats[b]]) if paga is not None and cats[a] in paga.index and cats[b] in paga.columns else np.nan
            rows.append(dict(c1=cats[a], c2=cats[b], paga_connectivity=conn,
                             paga_edge=bool(np.isfinite(conn) and conn >= paga_threshold), links=links,
                             share=float(share), touching=bool(links >= min_links and share >= min_share),
                             fragment_links=float(F[a, b]), fragment_nodes=float(N[a, b] + N[b, a])))
    T = pd.DataFrame(rows)
    T["edge_type"] = np.where(T["paga_edge"] & T["touching"], "PAGA ∩ DAPI",
                              np.where(T["paga_edge"], "PAGA only", np.where(T["touching"], "DAPI only", "neither")))
    T["tissue_contact"] = np.where(T["touching"], "cores touch",
                                   np.where(T["fragment_links"] > T["links"], "through fragments",
                                            np.where(T["links"] + T["fragment_links"] > 0, "weak", "none")))
    mark = {"cores touch": "supported", "through fragments": "through fragments", "weak": "weak contact",
            "none": "no contact"}
    T["link_mark"] = np.where(T["paga_edge"], T["tissue_contact"].map(mark),
                              np.where(T["touching"], "contact without PAGA edge", ""))
    if verbose:
        P = T[T["paga_edge"]]
        print(f"[11-3] PAGA edges {len(P)}: supported {int((P['link_mark'] == 'supported').sum())}, through fragments "
              f"{int((P['link_mark'] == 'through fragments').sum())}, weak {int((P['link_mark'] == 'weak contact').sum())}, "
              f"no contact {int((P['link_mark'] == 'no contact').sum())} | cores touching without a PAGA edge "
              f"{int((T['link_mark'] == 'contact without PAGA edge').sum())} (cores touch: ≥ {min_links} links and "
              f"≥ {min_share:.0%} of the smaller core boundary)")
    return T


def boundary_neighbors(A, labels, sets) -> Dict[str, pd.DataFrame]:
    """Which clusters the boundary nodes touch (node counts, not link counts; a node touching two clusters is
    counted in both columns).
    by_cluster : rows (cluster, set of the node), columns = touched cluster
    by_set     : rows (cluster, set), columns = (touched cluster, set of the touched node)"""
    code, cats = _codes(labels)
    sets = np.asarray(sets).astype(str)
    lab = np.asarray(cats, object)[code]
    C = A.tocoo()
    m = code[C.row] != code[C.col]
    E = pd.DataFrame(dict(node=C.row[m], own=lab[C.row[m]], set=sets[C.row[m]], other=lab[C.col[m]],
                          other_set=sets[C.col[m]]))
    N1 = E.drop_duplicates(["node", "other"])
    N2 = E.drop_duplicates(["node", "other", "other_set"])
    return dict(by_cluster=pd.crosstab([N1["own"], N1["set"]], N1["other"]),
                by_set=pd.crosstab([N2["own"], N2["set"]], [N2["other"], N2["other_set"]]))


# ---------------------------------------------------------------------------
# 11-4 planted controls
# ---------------------------------------------------------------------------
def planted_controls(A, labels, xy: np.ndarray, frac: float = 0.03, n_perm: int = 200, alpha: float = 0.05,
                     seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """A random set and a contiguous patch (same size) get new labels; share of each called core."""
    rng = np.random.default_rng(seed + 7)
    n = len(xy)
    m = max(10, int(frac * n))
    centre = xy[rng.integers(n)]
    patch = np.argsort(((xy - centre) ** 2).sum(axis=1))[:m]
    rest = np.setdiff1d(np.arange(n), patch)
    rand = rng.choice(rest, m, replace=False)
    lab = np.asarray(labels).astype(object).copy()
    lab[patch] = "planted_patch"
    lab[rand] = "planted_random"
    res = core_scattered(A, lab, n_perm=n_perm, alpha=alpha, seed=seed, verbose=False)["nodes"]
    T = pd.DataFrame([dict(control="planted_patch", nodes=m, expected="core",
                           share_core=float((res["set"].to_numpy()[patch] == "core").mean())),
                      dict(control="planted_random", nodes=m, expected="scattered",
                           share_core=float((res["set"].to_numpy()[rand] == "core").mean()))])
    T["share_as_expected"] = np.where(T["expected"] == "core", T["share_core"], 1 - T["share_core"])
    T["passes"] = T["share_as_expected"] >= 0.9
    if verbose:
        print("[11-4] planted controls: " + " | ".join(f"{r.control} {r.share_as_expected:.0%} as expected"
                                                     for r in T.itertuples()))
    return T


# ---------------------------------------------------------------------------
# 13-1 expression reproducibility
# ---------------------------------------------------------------------------
def _match(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Relabel pred to the reference labels by maximum overlap (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    pu, ru = np.unique(pred), np.unique(ref)
    M = np.array([[np.sum((pred == a) & (ref == b)) for b in ru] for a in pu])
    r, c = linear_sum_assignment(-M)
    mp = {pu[i]: ru[j] for i, j in zip(r, c)}
    return np.array([mp.get(x, "__unmatched__") for x in pred], dtype=object)


def label_reproducibility(pre, labels, sets, boundary: np.ndarray, k: Optional[int] = None, n_cut: int = 10,
                          n_pcs: int = 30, n_neighbors: int = 15, n_boot: int = 2000, seed: int = 0,
                          verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Per node: how often a clustering fitted on the other side of a random cut gives it its full-data cluster."""
    from .paga_nodes import cluster_features

    Z = _dense(pre.X).astype(float)
    xy = np.asarray(pre.obsm["spatial"], float)
    ref = np.asarray(labels).astype(str)
    k = k or len(np.unique(ref))
    sets = np.asarray(sets).astype(str)
    agree, seen, per_fit = np.zeros(len(ref)), np.zeros(len(ref)), []
    for c in random_cuts(xy, n_cut, seed):
        side = xy @ c["u"] < c["t"]
        for fit in (side, ~side):
            if fit.sum() < 3 * k or (~fit).sum() < 3:
                continue
            lf, _, tf, pf = cluster_features(Z, k, n_pcs, n_neighbors, rows=fit, seed=seed)
            Po = pf.transform(Z[~fit])[:, :tf.obsm["X_pca"].shape[1]]
            pred = _match(_transfer(tf.obsm["X_pca"], lf, Po), ref[~fit])
            ok = (pred == ref[~fit]).astype(float)
            idx = np.flatnonzero(~fit)
            agree[idx] += ok
            seen[idx] += 1
            row = {}
            for s in ("core", "scattered"):
                for nm, msk in (("all", np.ones(len(idx), bool)), ("boundary", boundary[idx])):
                    mm = (sets[idx] == s) & msk
                    row[f"{s}_{nm}"] = float(ok[mm].mean()) if mm.any() else np.nan
            per_fit.append(row)
    F = pd.DataFrame(per_fit)
    rng = np.random.default_rng(seed)
    rows = []
    for nm in ("all", "boundary"):
        d = (F[f"core_{nm}"] - F[f"scattered_{nm}"]).dropna().to_numpy()
        b = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)]) if len(d) > 1 else np.array([np.nan])
        rows.append(dict(nodes=nm, core=float(F[f"core_{nm}"].mean()), scattered=float(F[f"scattered_{nm}"].mean()),
                         difference=float(d.mean()) if len(d) else np.nan, ci_low=float(np.nanpercentile(b, 2.5)),
                         ci_high=float(np.nanpercentile(b, 97.5)), fits=int(len(d))))
    S = pd.DataFrame(rows)
    node = pd.DataFrame(dict(agreement=np.where(seen > 0, agree / np.maximum(seen, 1), np.nan), fits=seen),
                        index=pre.obs_names)
    if verbose:
        r = S.iloc[0]
        print(f"[13-1] label agreement core {r.core:.3f} vs scattered {r.scattered:.3f} "
              f"(difference {r.difference:.3f}, 95% CI {r.ci_low:.3f}–{r.ci_high:.3f}; {r.fits} fits)")
    return dict(summary=S, per_fit=F, nodes=node)


# ---------------------------------------------------------------------------
# 13-2 independent measurements
# ---------------------------------------------------------------------------
def _blocks(xy: np.ndarray, block_um: float) -> List[np.ndarray]:
    b = np.floor(xy / block_um).astype(np.int64)
    key = b[:, 0] * 100003 + b[:, 1]
    return [np.flatnonzero(key == u) for u in np.unique(key)]


def _block_ids(xy: np.ndarray, block_um: float) -> np.ndarray:
    b = np.floor(xy / block_um).astype(np.int64)
    return np.unique(b[:, 0] * 100003 + b[:, 1], return_inverse=True)[1]


def independent_agreement(feats: Dict[str, pd.DataFrame], node_names: Sequence[str], labels, sets, xy: np.ndarray,
                          boundary: Optional[np.ndarray] = None, n_neighbors: int = 15, n_folds: int = 5,
                          n_perm: int = 20, block_um: float = 500.0, buffer_um: float = 200.0, n_boot: int = 500,
                          n_shift: int = 50, min_shift_um: float = 1000.0, seed: int = 0,
                          verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Cross-validated k-NN prediction of the cluster from features not used for clustering; accuracy core vs
    scattered, chance level from permuted labels, spatial block bootstrap CI of the difference.

    Two fold schemes (rows `cv`):
      'random'        stratified random folds (the scheme run on 2026-11-25; kept for comparison). A core node's
                      tissue neighbours sit in the training folds and the features are spatially smooth, so this
                      favours core nodes (spatial leakage).
      'spatial_block' whole `block_um` blocks are held out together, and training nodes closer than `buffer_um` to
                      any test node are dropped. This is the scheme used for P2 (change log 2026-11-25).
    Each scheme is summarised over all nodes and, when `boundary` is given, over cluster-boundary nodes only.

    Shift null (spatial_block only): leakage is not removed completely by blocks when the features vary smoothly over
    more than the buffer (synthetic check: features with no label information still gave core 0.35 vs scattered 0.12).
    The label map and the core / scattered sets are therefore shifted together (≥ `min_shift_um`, wrapped inside the
    bounding box, read at the nearest node), which keeps the patches but breaks their link to the features, and the
    same CV is run. `p_shift` = (1 + #null difference ≥ observed) / (1 + n_shift); P2 needs p_shift ≤ 0.05."""
    from scipy.spatial import cKDTree
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    X = pd.concat([df.reindex(node_names) for df in feats.values()], axis=1)
    X = X.loc[:, X.notna().mean() > 0.9].fillna(X.median())
    X = StandardScaler().fit_transform(X.to_numpy(float))
    y = np.asarray(labels).astype(str)
    sets = np.asarray(sets).astype(str)
    xy = np.asarray(xy, float)
    bnd = None if boundary is None else np.asarray(boundary, bool)
    bid = _block_ids(xy, block_um)
    tree = cKDTree(xy)

    def folds_random(yy, s):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        return [te for _, te in skf.split(X, yy)]

    def folds_block(s):
        r = np.random.default_rng(s)
        f_of_block = r.permutation(np.arange(bid.max() + 1) % n_folds)
        return [np.flatnonzero(f_of_block[bid] == f) for f in range(n_folds)]

    def cv_correct(yy, test_sets, buffer):
        out = np.full(len(yy), np.nan)
        for te in test_sets:
            if not len(te):
                continue
            tr = np.ones(len(yy), bool)
            tr[te] = False
            if buffer > 0:
                near = tree.query_ball_point(xy[te], r=buffer)
                tr[np.unique(np.concatenate([np.asarray(v, int) for v in near]))] = False
            tr = np.flatnonzero(tr)
            if len(tr) < n_neighbors:
                continue
            clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X[tr], yy[tr])
            out[te] = clf.predict(X[te]) == yy[te]
        return out

    blocks = _blocks(xy, block_um)
    rng = np.random.default_rng(seed)
    lo, span = xy.min(0), np.ptp(xy, 0) + 1e-9

    def shifted_maps(n):
        out, tries = [], 0
        while len(out) < n and tries < 100 * n:
            tries += 1
            d = rng.uniform(0, span)
            dd = np.minimum(d, span - d)
            if np.hypot(*dd) < min_shift_um:
                continue
            j = tree.query(lo + np.mod(xy - lo + d, span))[1]
            out.append(j)
        return out

    def diff(ok_, yy_sets, keep):
        a = ok_[keep & (yy_sets == "core") & np.isfinite(ok_)]
        b = ok_[keep & (yy_sets == "scattered") & np.isfinite(ok_)]
        return a.mean() - b.mean() if len(a) and len(b) else np.nan

    null_diffs = {}
    if n_shift:
        maps = shifted_maps(n_shift)
        for i, j in enumerate(maps):
            ys, ss = y[j], sets[j]
            okn = cv_correct(ys, folds_block(seed + 1000 + i), buffer_um)
            null_diffs.setdefault("all", []).append(diff(okn, ss, np.ones(len(y), bool)))
            if bnd is not None:
                # boundary status moves with the shifted label map (read at the same source node)
                null_diffs.setdefault("boundary", []).append(diff(okn, ss, bnd[j]))
    rows, per_node = [], {}
    for cv in ("random", "spatial_block"):
        if cv == "random":
            ok = cv_correct(y, folds_random(y, seed), 0.0)
        else:
            ok = cv_correct(y, folds_block(seed), buffer_um)
        per_node[cv] = ok
        chance = []
        for i in range(n_perm):
            yp = rng.permutation(y)
            te = folds_random(yp, seed + 1 + i) if cv == "random" else folds_block(seed + 1 + i)
            chance.append(cv_correct(yp, te, 0.0 if cv == "random" else buffer_um))
        subsets = [("all", np.ones(len(y), bool))]
        if bnd is not None:
            subsets.append(("boundary", bnd))
        for name, keep in subsets:
            c_m = keep & (sets == "core") & np.isfinite(ok)
            s_m = keep & (sets == "scattered") & np.isfinite(ok)
            diffs = []
            for _ in range(n_boot):
                ii = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
                a, b = ok[ii][c_m[ii]], ok[ii][s_m[ii]]
                if len(a) and len(b):
                    diffs.append(a.mean() - b.mean())
            ch_c = float(np.nanmean([np.nanmean(o[keep & (sets == "core")]) for o in chance]))
            ch_s = float(np.nanmean([np.nanmean(o[keep & (sets == "scattered")]) for o in chance]))
            obs = float(ok[c_m].mean() - ok[s_m].mean())
            nd = np.asarray(null_diffs.get(name, []), float) if cv == "spatial_block" else np.array([])
            nd = nd[np.isfinite(nd)]
            rows.append(dict(cv=cv, nodes=name, core=float(ok[c_m].mean()), scattered=float(ok[s_m].mean()),
                             chance_core=ch_c, chance_scattered=ch_s,
                             difference=float(ok[c_m].mean() - ok[s_m].mean()),
                             ci_low=float(np.percentile(diffs, 2.5)), ci_high=float(np.percentile(diffs, 97.5)),
                             n_core=int(c_m.sum()), n_scattered=int(s_m.sum()), features=int(X.shape[1]),
                             block_um=block_um, buffer_um=(0.0 if cv == "random" else buffer_um),
                             null_difference_mean=(float(nd.mean()) if len(nd) else np.nan),
                             null_difference_q95=(float(np.percentile(nd, 95)) if len(nd) else np.nan),
                             p_shift=(float((1 + (nd >= obs).sum()) / (1 + len(nd))) if len(nd) else np.nan),
                             n_shift=int(len(nd))))
    S = pd.DataFrame(rows)
    if verbose:
        for r in S.itertuples():
            print(f"[13-2] {r.cv:13s} {r.nodes:8s}: core {r.core:.3f} (chance {r.chance_core:.3f}) vs scattered "
                  f"{r.scattered:.3f} (chance {r.chance_scattered:.3f}); difference {r.difference:.3f} "
                  f"(95% CI {r.ci_low:.3f}–{r.ci_high:.3f})"
                  + (f"; shift null mean {r.null_difference_mean:.3f}, p_shift {r.p_shift:.3f}" if r.n_shift else ""))
    N = pd.DataFrame({f"correct_{k}": v for k, v in per_node.items()}, index=list(node_names))
    N["set"] = sets
    if bnd is not None:
        N["boundary"] = bnd
    return dict(summary=S, nodes=N)


# ---------------------------------------------------------------------------
# 13-3 regulons
# ---------------------------------------------------------------------------
def regulon_support(grn, nodes, node_names: Sequence[str], support_lab, xy: np.ndarray, n_shift: int = 200,
                    min_shift_um: float = 1000.0, n_maps: int = 10, n_cut: int = 10, alpha: float = 0.05,
                    seed: int = 0, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """regulon_eval.exclusion on regions '<cluster>|core' / '<cluster>|scattered'; summary by set."""
    from scipy.stats import fisher_exact

    from .regulon_eval import exclusion

    labels = pd.Series(np.asarray(support_lab).astype(str), index=list(node_names), name="support")
    T = exclusion(grn, nodes, labels, np.asarray(xy, float), n_shift=n_shift, min_shift_um=min_shift_um,
                  n_maps=n_maps, n_cut=n_cut, alpha=alpha, seed=seed, verbose=verbose)
    T["set"] = T["region"].astype(str).str.split("|").str[-1]
    T["cluster"] = T["region"].astype(str).str.split("|").str[0]
    T["abs_z"] = T["z"].abs()
    S, R = regulon_summary(T)
    if verbose:
        print("[13-3] " + " | ".join(f"{r.set}: {r.tested}/{r.pairs} pairs tested, |z| ≥ 2 {r.z_ge_2}, C1 {r.C1}, "
                                    f"kept {r.kept}, naive {r.naive_calls}" for r in S.itertuples()))
        if "fisher_tested_p" in S:
            r = S.iloc[0]
            print(f"[13-3] Fisher (tested pairs) p = {r.fisher_tested_p:.3g}, OR {r.odds_ratio_tested:.2f}; "
                  f"max p leaving out one core region {r.fisher_tested_loo_max_p:.3g} | "
                  f"as first run (all pairs, untested counted as |z| < 2) p = {r.fisher_all_pairs_p:.3g}")
    return dict(pairs=T, summary=S, regions=R)


def regulon_summary(T: pd.DataFrame):
    """P3 summary from the pair table of `regulon_support` (also usable on a saved 13-3_regulon_pairs.csv).

    Only pairs with a finite shift-null z are counted ('tested'); a pair whose region is too small for the test has no
    z and is neither a hit nor a miss. The first run (2026-11-25) counted untested pairs as |z| < 2, which favours the
    larger core regions; that p is kept as `fisher_all_pairs_p`. `fisher_tested_loo_max_p` is the largest p after
    leaving out any one core region (is the result carried by a single region?)."""
    from scipy.stats import fisher_exact

    T = T.copy()
    T["abs_z"] = pd.to_numeric(T["z"], errors="coerce").abs()
    T["tested"] = np.isfinite(T["abs_z"])
    T["hit"] = T["tested"] & (T["abs_z"] >= 2)
    for col in ("C1", "kept", "naive_call"):
        T[col] = T[col].astype(str).str.lower().eq("true") if T[col].dtype == object else T[col].astype(bool)
    rows = []
    for s, g in T.groupby("set"):
        rows.append(dict(set=s, pairs=len(g), tested=int(g["tested"].sum()), regions=g["region"].nunique(),
                         regions_tested=int(g.loc[g["tested"], "region"].nunique()),
                         naive_calls=int(g["naive_call"].sum()), z_ge_2=int(g["hit"].sum()),
                         share_z_ge_2_tested=float(g["hit"].sum() / max(g["tested"].sum(), 1)),
                         C1=int(g["C1"].sum()), kept=int(g["kept"].sum()),
                         median_abs_z=float(g.loc[g["tested"], "abs_z"].median())))
    S = pd.DataFrame(rows)
    R = (T.groupby("region").agg(set=("set", "first"), pairs=("z", "size"), tested=("tested", "sum"),
                                 z_ge_2=("hit", "sum"), C1=("C1", "sum"), kept=("kept", "sum"))
          .reset_index())
    if {"core", "scattered"} <= set(S["set"]):
        def fisher(tt):
            c, s = tt[tt["set"] == "core"], tt[tt["set"] == "scattered"]
            tab = [[int(c["hit"].sum()), int(c["tested"].sum() - c["hit"].sum())],
                   [int(s["hit"].sum()), int(s["tested"].sum() - s["hit"].sum())]]
            return fisher_exact(tab, alternative="greater")
        odds, p = fisher(T)
        loo = [fisher(T[T["region"] != r])[1] for r in T.loc[(T["set"] == "core") & T["tested"], "region"].unique()]
        c, s = S.set_index("set").loc["core"], S.set_index("set").loc["scattered"]
        odds_all, p_all = fisher_exact([[c.z_ge_2, c.pairs - c.z_ge_2], [s.z_ge_2, s.pairs - s.z_ge_2]],
                                       alternative="greater")
        S["fisher_tested_p"] = float(p)
        S["odds_ratio_tested"] = float(odds)
        S["fisher_tested_loo_max_p"] = float(max(loo)) if loo else np.nan
        S["fisher_all_pairs_p"] = float(p_all)
        S["odds_ratio_all_pairs"] = float(odds_all)
    return S, R


# ---------------------------------------------------------------------------
# 13-4 trajectory
# ---------------------------------------------------------------------------
def pseudotime_core(pre, tag: str, core_mask: np.ndarray, root_index: int, n_dcs: int = 10) -> Dict[str, object]:
    """DPT on the whole kNN graph and on the graph restricted to core nodes (component containing the root)."""
    import anndata as ad
    import scanpy as sc
    from scipy.stats import spearmanr

    from .trajectory import pseudotime

    full = pseudotime(pre, tag, root_index, n_dcs)
    C = pre.obsp[f"{tag}_connectivities"].tocsr()
    core = np.flatnonzero(core_mask)
    if root_index not in set(core.tolist()):
        raise ValueError("root is not a core node; choose a core root (see TACTC.ipynb 13-4).")
    Cs = C[core][:, core]
    _, cid = connected_components(Cs, directed=False)
    r_local = int(np.flatnonzero(core == root_index)[0])
    keep = np.flatnonzero(cid == cid[r_local])
    nodes = core[keep]
    Ck = Cs[keep][:, keep]
    t = ad.AnnData(X=np.zeros((len(nodes), 1), np.float32), obs=pd.DataFrame(index=pre.obs_names[nodes]))
    t.obsp["connectivities"] = Ck
    t.obsp["distances"] = Ck.copy()
    t.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"method": "umap", "n_neighbors": 15}}
    sc.tl.diffmap(t, n_comps=max(n_dcs + 1, 15))
    t.uns["iroot"] = int(np.flatnonzero(nodes == root_index)[0])
    sc.tl.dpt(t, n_dcs=n_dcs)
    sub = np.full(pre.n_obs, np.nan)
    sub[nodes] = t.obs["dpt_pseudotime"].to_numpy(float)
    ok = np.isfinite(sub) & np.isfinite(full)
    rho = float(spearmanr(full[ok], sub[ok])[0]) if ok.sum() > 3 else np.nan
    return dict(full=full, core_only=sub, spearman=rho, nodes_used=int(ok.sum()), core_nodes=int(len(core)))
