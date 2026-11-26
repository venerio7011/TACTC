"""
hexagons.py
===========

Steps 3–4 of TACTC.ipynb.

Step 3: hexagon expansion
  Visium spots (55 µm circles) sit 100 µm apart, so most cells lie between spots where no RNA was captured.
  Each spot is expanded to its hexagon (Voronoi cell of the spot lattice, bounded by the midlines to the neighbouring
  spots); a whole cell goes to the one hexagon containing its centre. The lattice is built from ALL in-tissue spots
  (before QC), so cells of QC-removed spots stay with their own spot instead of being pulled into a neighbour.
  Per cell, the region is recorded:
    circle     : ≥ min_capture_frac (0.5) of the cell area inside its spot's capture circle → RNA measured
    ring       : in the hexagon but outside the circle → RNA not measured (extrapolated from the spot)
    qc_removed : in the hexagon of a spot removed by QC
    no_shares  : in the hexagon of a QC-passed spot without step-1 shares (e.g. dropped by STdeconvolve)
    outside    : not in any in-tissue spot hexagon (tissue edge beyond the lattice, off-tissue debris)
  Ratio per spot: hex_ratio = DAPI cells in the hexagon / DAPI cells in the circle (step-1 area-overlap count).

Step 4: scaling the step-1 counts to the hexagon
  Assumption (as in visiumif2niche 9-1b): ring cells have the same type composition as the circle cells of their spot.
  Scaling the circle counts by hex_ratio is the same as N_hex × cell share, so the counts are built from the actual
  numbers of cell nodes, which keeps them exact integers that match the graph:
    circle counts = cell share × n_circle cells,  ring counts = cell share × n_ring cells (each split by largest
    remainder), hexagon counts = circle + ring.  Every spot's hexagon counts sum to its number of cell nodes, which
    the assignment in step 5 needs.
  lo / hi: 5–95% of multinomial draws of n_hex cells with the cell shares (integer-split uncertainty only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .cell_counts import largest_remainder

__all__ = ["spot_lattice", "assign_hexagons", "HexAssignment", "scale_counts", "HexCounts", "add_to_spots",
           "annotate_cells", "REGIONS"]

REGIONS = ["circle", "ring", "qc_removed", "no_shares", "outside"]


def spot_lattice(xy: np.ndarray):
    """Visium spot lattice from spot centres: (centre spacing in the units of xy, unit vectors of the 3 neighbour
    directions). Same as visiumif2niche spot_counts.spot_lattice."""
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, float)
    d, j = cKDTree(xy).query(xy, k=7)
    spacing = float(np.median(d[:, 1]))
    v = (xy[j[:, 1:]] - xy[:, None, :]).reshape(-1, 2)
    v = v[np.abs(np.linalg.norm(v, axis=1) - spacing) < 0.15 * spacing]
    ang = np.arctan2(v[:, 1], v[:, 0])
    th = np.angle(np.mean(np.exp(6j * ang))) / 6           # mean direction with 60° period
    U = np.array([[np.cos(th + k * np.pi / 3), np.sin(th + k * np.pi / 3)] for k in range(3)])
    return spacing, U


@dataclass
class HexAssignment:
    cells: pd.DataFrame        # cell_id → spot ('' if none), region, capture_frac
    spots: pd.DataFrame        # QC-passed spot → n_hex, n_circle, n_ring, n_dapi_area, hex_ratio, hex_ratio_members
    spacing_px: float
    params: Dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        c = self.cells["region"].value_counts().reindex(REGIONS, fill_value=0)
        return pd.DataFrame({"cells": c, "share": (c / max(len(self.cells), 1)).round(4)})

    def save(self, out_dir: str):
        import os

        os.makedirs(out_dir, exist_ok=True)
        self.cells.to_csv(os.path.join(out_dir, "cell_hexagons.csv.gz"))
        self.spots.to_csv(os.path.join(out_dir, "spot_hexagons.csv"))
        json.dump(dict(spacing_px=self.spacing_px, params=self.params), open(os.path.join(out_dir, "hex_params.json"),
                                                                             "w"), indent=1, default=str)
        print(f"[save] hexagon assignment → {out_dir}/cell_hexagons.csv.gz, spot_hexagons.csv")

    @classmethod
    def load(cls, out_dir: str) -> "HexAssignment":
        import os

        c = pd.read_csv(os.path.join(out_dir, "cell_hexagons.csv.gz"), index_col=0, keep_default_na=False)
        c["capture_frac"] = c["capture_frac"].astype(float)
        s = pd.read_csv(os.path.join(out_dir, "spot_hexagons.csv"), index_col=0)
        m = json.load(open(os.path.join(out_dir, "hex_params.json")))
        return cls(c, s, float(m["spacing_px"]), m.get("params", {}))


def assign_hexagons(cells: pd.DataFrame, lattice_spots: pd.DataFrame, qc_spots, spot_cells, shares_spots=None,
                    min_capture_frac: float = 0.5, verbose: bool = True) -> HexAssignment:
    """Assign every cell to the hexagon of its spot and label its region.

    cells         : Mesmer table (index = cell_id; x, y px)
    lattice_spots : ALL in-tissue spots before QC (index = barcode; x, y px), e.g. vis_raw.obs
    qc_spots      : barcodes that passed QC (vis.obs_names)
    spot_cells    : step-1 dapi_counts.SpotCells (circle overlap per cell; area-overlap counts per spot)
    shares_spots  : barcodes with step-1 shares (None = all QC-passed spots)
    """
    from scipy.spatial import cKDTree

    sxy = lattice_spots[["x", "y"]].to_numpy(float)
    names = lattice_spots.index.astype(str).to_numpy()
    spacing, U = spot_lattice(sxy)
    cxy = cells[["x", "y"]].to_numpy(float)
    _, j = cKDTree(sxy).query(cxy)
    rel = cxy - sxy[j]
    inside = (np.abs(rel @ U.T) <= spacing / 2 * (1 + 1e-9)).all(axis=1)
    spot = np.where(inside, names[j], "")

    qc = pd.Index(pd.Index(qc_spots).astype(str))
    sh = qc if shares_spots is None else pd.Index(pd.Index(shares_spots).astype(str))
    cap = spot_cells.overlap.groupby("cell_id")["frac"].sum().clip(upper=1.0)
    capture = cap.reindex(cells.index.astype(np.int64)).fillna(0.0).to_numpy()
    in_qc = pd.Index(spot).isin(qc)
    in_sh = pd.Index(spot).isin(sh)
    region = np.full(len(cells), "outside", dtype=object)
    region[inside & ~in_qc] = "qc_removed"
    region[inside & in_qc & ~in_sh] = "no_shares"
    ok = inside & in_qc & in_sh
    region[ok & (capture >= min_capture_frac)] = "circle"
    region[ok & (capture < min_capture_frac)] = "ring"
    cdf = pd.DataFrame({"spot": spot, "region": region, "capture_frac": capture},
                       index=pd.Index(cells.index.astype(np.int64), name="cell_id"))

    member = cdf[cdf["spot"].isin(qc)]
    circ_mask = member["capture_frac"] >= min_capture_frac
    n_hex = member.groupby("spot").size().reindex(qc, fill_value=0)
    n_circ = member[circ_mask].groupby("spot").size().reindex(qc, fill_value=0)
    n_area = spot_cells.counts.reindex(qc).fillna(0.0).astype(float)
    sdf = pd.DataFrame({"n_hex": n_hex.astype(int), "n_circle": n_circ.astype(int),
                        "n_ring": (n_hex - n_circ).astype(int), "n_dapi_area": n_area,
                        "hex_ratio": (n_hex / n_area.where(n_area > 0)).astype(float),
                        "hex_ratio_members": (n_hex / n_circ.where(n_circ > 0)).astype(float),
                        "has_shares": qc.isin(sh)}, index=qc)
    res = HexAssignment(cdf, sdf, float(spacing), dict(min_capture_frac=float(min_capture_frac),
                                                        n_lattice_spots=int(len(names)), n_qc_spots=int(len(qc))))
    if verbose:
        r = float(np.pi * spot_cells.radius_px ** 2) if np.isfinite(spot_cells.radius_px) else np.nan
        area_ratio = (np.sqrt(3) / 2 * spacing ** 2) / r if r == r else np.nan
        s = res.summary()["cells"]
        q = sdf["hex_ratio"].quantile([0.1, 0.5, 0.9]).round(2).tolist()
        print(f"[hex] spacing {spacing:.1f} px, hexagon / circle area {area_ratio:.2f}× | cells: "
              + ", ".join(f"{k} {int(s[k]):,}" for k in REGIONS)
              + f" | hex / circle DAPI ratio 10% {q[0]}, median {q[1]}, 90% {q[2]}")
    return res


@dataclass
class HexCounts:
    hex: pd.DataFrame          # spot × group integer counts in the hexagon (= circle + ring)
    circle: pd.DataFrame       # spot × group integer counts of circle (RNA-measured) cells
    ring: pd.DataFrame         # spot × group integer counts of ring (extrapolated) cells
    expected: pd.DataFrame     # spot × group cell share × n_hex (fractional)
    lo: pd.DataFrame           # 5% of multinomial draws of the hexagon counts
    hi: pd.DataFrame           # 95%
    params: Dict = field(default_factory=dict)

    _TABLES = ("hex", "circle", "ring", "expected", "lo", "hi")

    def summary(self) -> pd.DataFrame:
        h, c, r = self.hex.sum(), self.circle.sum(), self.ring.sum()
        return pd.DataFrame({"hex_cells": h.astype(int), "circle_cells": c.astype(int), "ring_cells": r.astype(int),
                             "hex_share": (h / max(float(h.sum()), 1.0)).round(4)}).sort_values("hex_cells",
                                                                                               ascending=False)

    def save(self, out_dir: str):
        import os

        os.makedirs(out_dir, exist_ok=True)
        for t in self._TABLES:
            getattr(self, t).to_csv(os.path.join(out_dir, f"hex_{t}.csv"))
        json.dump(self.params, open(os.path.join(out_dir, "hex_counts_params.json"), "w"), indent=1, default=str)
        self.summary().to_csv(os.path.join(out_dir, "hex_summary.csv"))
        print(f"[save] hexagon cell counts → {out_dir}/hex_hex.csv, hex_circle.csv, hex_ring.csv, …")

    @classmethod
    def load(cls, out_dir: str) -> "HexCounts":
        import os

        t = {k: pd.read_csv(os.path.join(out_dir, f"hex_{k}.csv"), index_col=0) for k in cls._TABLES}
        p = json.load(open(os.path.join(out_dir, "hex_counts_params.json")))
        return cls(params=p, **t)


def scale_counts(hexa: HexAssignment, cell_frac: pd.DataFrame, n_draws: int = 200, seed: int = 0,
                 verbose: bool = True) -> HexCounts:
    """Step-1 cell shares × hexagon cell numbers → integer counts per group (circle, ring, hexagon)."""
    spots = hexa.spots.index
    C = cell_frac.reindex(spots).astype(float)
    groups = list(C.columns)
    ok = C.notna().all(axis=1) & (C.sum(axis=1) > 0) & hexa.spots["has_shares"]
    Cv = np.where(ok.to_numpy()[:, None], C.fillna(0).to_numpy(), 0.0)
    s = Cv.sum(axis=1, keepdims=True)
    Cv = np.divide(Cv, s, out=np.zeros_like(Cv), where=s > 0)
    okv = ok.to_numpy()
    n_c = np.where(okv, hexa.spots["n_circle"].to_numpy(), 0)
    n_r = np.where(okv, hexa.spots["n_ring"].to_numpy(), 0)
    circ = largest_remainder(Cv * n_c[:, None], n_c)
    ring = largest_remainder(Cv * n_r[:, None], n_r)
    hexc = circ + ring
    n_h = n_c + n_r
    rng = np.random.default_rng(seed)
    D = np.zeros((n_draws, len(spots), len(groups)), dtype=np.int32)
    idx = np.flatnonzero(n_h > 0)
    for d in range(n_draws):
        if len(idx):
            D[d, idx] = rng.multinomial(n_h[idx], Cv[idx])
    lo, hi = np.percentile(D, [5, 95], axis=0, method="nearest")
    mk = lambda A: pd.DataFrame(A, index=spots, columns=groups)
    res = HexCounts(hex=mk(hexc), circle=mk(circ), ring=mk(ring), expected=mk(Cv * n_h[:, None]),
                    lo=mk(lo.astype(int)), hi=mk(hi.astype(int)),
                    params=dict(n_draws=int(n_draws), seed=int(seed), n_spots=int(len(spots)),
                                n_spots_counted=int(okv.sum()), n_cells_hex=int(hexc.sum()),
                                n_cells_circle=int(circ.sum()), n_cells_ring=int(ring.sum())))
    if verbose:
        print(f"[hex counts] {int(hexc.sum()):,} cells in {int((n_h > 0).sum()):,} hexagons "
              f"(circle {int(circ.sum()):,}, ring {int(ring.sum()):,}; ring / circle {ring.sum() / max(circ.sum(), 1):.2f})"
              f", {len(groups)} groups")
    return res


def add_to_spots(spots_adata, hexa: HexAssignment, hc: HexCounts):
    """Copy of the step-1 spot AnnData with the step 3–4 outputs.

    obs  : n_hex, n_circle_cells, n_ring_cells, hex_ratio, hex_ratio_members
    obsm : cell_counts_hex, cell_counts_circle, cell_counts_ring, cell_expected_hex, cell_counts_hex_lo / _hi
           (columns = uns['step1']['groups'])
    uns  : step4 (parameters)
    """
    a = spots_adata.copy()
    s = a.obs_names
    groups = list(a.uns["step1"]["groups"])
    sp = hexa.spots.reindex(s)
    a.obs["n_hex"] = sp["n_hex"].fillna(0).astype(int).to_numpy()
    a.obs["n_circle_cells"] = sp["n_circle"].fillna(0).astype(int).to_numpy()
    a.obs["n_ring_cells"] = sp["n_ring"].fillna(0).astype(int).to_numpy()
    a.obs["hex_ratio"] = sp["hex_ratio"].astype(float).to_numpy()
    a.obs["hex_ratio_members"] = sp["hex_ratio_members"].astype(float).to_numpy()
    for key, df in (("cell_counts_hex", hc.hex), ("cell_counts_circle", hc.circle), ("cell_counts_ring", hc.ring),
                    ("cell_expected_hex", hc.expected), ("cell_counts_hex_lo", hc.lo), ("cell_counts_hex_hi", hc.hi)):
        a.obsm[key] = df.reindex(index=s, columns=groups).fillna(0).astype(float).to_numpy()
    a.uns["step4"] = {"hex_counts_params": json.dumps(hc.params, default=str),
                      "hex_params": json.dumps(dict(spacing_px=hexa.spacing_px, **hexa.params), default=str)}
    return a


def annotate_cells(cell_adata, hexa: HexAssignment):
    """Adds the step-3 labels to the cell AnnData in place: obs['spot'] ('' = none), obs['region'] (REGIONS),
    obs['capture_frac'], obs['rna_measured'] (region == 'circle')."""
    c = hexa.cells.reindex(cell_adata.obs["cell_id"].to_numpy())
    cell_adata.obs["spot"] = c["spot"].fillna("").astype(str).to_numpy()
    cell_adata.obs["region"] = pd.Categorical(c["region"].fillna("outside").to_numpy(), categories=REGIONS)
    cell_adata.obs["capture_frac"] = c["capture_frac"].fillna(0.0).astype(float).to_numpy()
    cell_adata.obs["rna_measured"] = (cell_adata.obs["region"] == "circle").to_numpy()
    return cell_adata
