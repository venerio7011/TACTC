"""
dapi_counts.py
==============

Number of DAPI (Mesmer nuclear / whole-cell) segmented cells inside each Visium spot circle (TACTC.ipynb 1-2).
This is the cell count of the area where RNA was actually captured; step 1 splits it into cell types with RCTD.

Contents
----
* load_cells    : reads the Mesmer cell table (index = cell_id = mask label; columns area, y, x in full-resolution px)
* open_mask     : opens mask.tif without loading it all into RAM
* cells_in_spots: per spot, cells counted from the pixel overlap of the mask with the spot circle (diameter 55 µm).
                  A cell on the circle edge counts by the fraction of its area inside (0–1). Centre-based counts are kept too.
* SpotCells     : result (counts = area-overlap count, n_centroid, overlap table) with save / load / exists
* spot_radius_px: spot radius in full-resolution px (from scalefactors_json)

Copied from visiumif2niche spot_counts.cells_per_spot / spagcn_domains.open_mask (same algorithm) so that TACTC does
not depend on the old modules/ folder. Hexagon expansion (steps 2–4) is added later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["load_cells", "open_mask", "cells_in_spots", "SpotCells", "spot_radius_px"]


def load_cells(path: str) -> pd.DataFrame:
    """Nucleus segmentation table (e.g. Mesmer; see input/README.md). index = cell_id (mask label)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cell table not found: {path}\n  Provide a nucleus segmentation table "
                                "(see input/README.md) and point CELLS_CSV to it.")
    cells = pd.read_csv(path, index_col=0)
    missing = [c for c in ("area", "y", "x") if c not in cells.columns]
    if missing:
        raise KeyError(f"The cell table needs columns area, y, x (missing {missing}). Columns: {list(cells.columns)[:12]} …")
    cells.index = cells.index.astype(np.int64)
    cells.index.name = "cell_id"
    return cells


def open_mask(path: str):
    """Open mask.tif without loading it all into RAM (uncompressed → memmap, compressed → zarr, full load on failure)."""
    import tifffile

    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    try:
        return tifffile.memmap(path, mode="r")
    except Exception:
        pass
    try:
        import zarr

        z = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
        if not hasattr(z, "shape"):
            z = z[sorted(z.array_keys())[0]]
        return z
    except Exception:
        return tifffile.imread(path)


def spot_radius_px(vis) -> float:
    """Spot radius in full-resolution px (spot_diameter_fullres / 2)."""
    return float(vis.uns["spatial"][next(iter(vis.uns["spatial"]))]["scalefactors"]["spot_diameter_fullres"]) / 2


@dataclass
class SpotCells:
    counts: pd.Series        # spot → cell count (sum of overlap fractions, fractional)
    n_centroid: pd.Series    # spot → cells whose centre is inside the circle (integer)
    overlap: pd.DataFrame    # spot, cell_id, pixels, frac (fraction of the cell area inside the spot)
    radius_px: float

    def members(self, min_frac: float = 0.5) -> pd.Series:
        """cell_id → spot (cells with ≥ min_frac of their area inside the spot)."""
        ov = self.overlap[self.overlap["frac"] >= min_frac]
        return pd.Series(ov["spot"].to_numpy(), index=pd.Index(ov["cell_id"].to_numpy(), name="cell_id"), name="spot")

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"n_cells": self.counts, "n_centroid": self.n_centroid}).to_csv(
            os.path.join(out_dir, "spot_cell_counts.csv"))
        self.overlap.to_csv(os.path.join(out_dir, "spot_cell_overlap.csv.gz"), index=False)
        with open(os.path.join(out_dir, "radius_px.txt"), "w") as f:
            f.write(f"{self.radius_px:.6f}\n")
        print(f"[save] DAPI cells per spot → {out_dir}/spot_cell_counts.csv, spot_cell_overlap.csv.gz")

    @classmethod
    def load(cls, out_dir: str) -> "SpotCells":
        c = pd.read_csv(os.path.join(out_dir, "spot_cell_counts.csv"), index_col=0)
        ov = pd.read_csv(os.path.join(out_dir, "spot_cell_overlap.csv.gz"))
        rp = os.path.join(out_dir, "radius_px.txt")
        r = float(open(rp).read().strip()) if os.path.exists(rp) else float("nan")
        return cls(c["n_cells"].astype(float), c["n_centroid"].astype(int), ov, r)

    @staticmethod
    def exists(out_dir: str) -> bool:
        return all(os.path.exists(os.path.join(out_dir, f)) for f in ("spot_cell_counts.csv", "spot_cell_overlap.csv.gz"))

    def aligned(self, spots) -> "SpotCells":
        """Copy restricted / reindexed to the given spots (missing spots count 0 cells)."""
        spots = pd.Index(spots)
        ov = self.overlap[self.overlap["spot"].isin(spots)].reset_index(drop=True)
        return SpotCells(self.counts.reindex(spots).fillna(0.0), self.n_centroid.reindex(spots).fillna(0).astype(int),
                         ov, self.radius_px)


def cells_in_spots(mask, cells: pd.DataFrame, vis, band_rows: int = 2048, verbose: bool = True) -> SpotCells:
    """Count cells per spot from the overlap of mask pixels with spot circles.

    mask     : label mask (numpy / tifffile memmap / zarr from open_mask). label = cell_id
    cells    : index = cell_id, columns area (px), y, x
    band_rows: mask rows read at once (memory ≈ band_rows × width × 4 bytes)

    Contribution of one cell = (pixels inside the spot circle) / (cell area), so cells on the edge count between 0 and 1.
    The gap between Visium spots (45 µm) is larger than a cell, so no cell straddles two spots.
    """
    r = spot_radius_px(vis)
    H, W = mask.shape[:2]
    sx, sy = vis.obs["x"].to_numpy(float), vis.obs["y"].to_numpy(float)
    names = vis.obs_names.to_numpy()
    area = cells["area"]
    order = np.argsort(sy)
    rows = []
    for b0 in range(0, H, band_rows):
        sel = order[(sy[order] >= b0) & (sy[order] < b0 + band_rows)]
        if b0 == 0:                                    # spots centred above the top of the image
            sel = np.r_[order[sy[order] < 0], sel]
        if b0 + band_rows >= H:                        # last band: spots centred below it
            sel = np.r_[sel, order[sy[order] >= b0 + band_rows]]
        if not len(sel):
            continue
        y0 = max(0, int(np.floor(sy[sel].min() - r)) - 1)
        y1 = min(H, int(np.ceil(sy[sel].max() + r)) + 2)
        x0 = max(0, int(np.floor(sx[sel].min() - r)) - 1)
        x1 = min(W, int(np.ceil(sx[sel].max() + r)) + 2)
        if y1 <= y0 or x1 <= x0:
            continue
        band = np.asarray(mask[y0:y1, x0:x1])
        for i in sel:
            a0, a1 = max(y0, int(np.floor(sy[i] - r))), min(y1, int(np.ceil(sy[i] + r)) + 1)
            c0, c1 = max(x0, int(np.floor(sx[i] - r))), min(x1, int(np.ceil(sx[i] + r)) + 1)
            if a1 <= a0 or c1 <= c0:
                continue
            sub = band[a0 - y0:a1 - y0, c0 - x0:c1 - x0]
            yy, xx = np.ogrid[a0:a1, c0:c1]
            inside = (yy - sy[i]) ** 2 + (xx - sx[i]) ** 2 <= r * r
            lab = sub[inside]
            lab = lab[lab > 0]
            if not lab.size:
                continue
            u, n = np.unique(lab, return_counts=True)
            rows.append(pd.DataFrame({"spot": names[i], "cell_id": u.astype(np.int64), "pixels": n}))
    ov = (pd.concat(rows, ignore_index=True) if rows
          else pd.DataFrame({"spot": pd.Series(dtype=str), "cell_id": pd.Series(dtype=np.int64),
                             "pixels": pd.Series(dtype=np.int64)}))
    ov = ov[ov["cell_id"].isin(area.index)].reset_index(drop=True)
    ov["frac"] = (ov["pixels"] / area.reindex(ov["cell_id"]).to_numpy()).clip(upper=1.0)
    counts = ov.groupby("spot")["frac"].sum().reindex(vis.obs_names, fill_value=0.0).astype(float).rename("n_cells")
    cxy = cells.loc[ov["cell_id"], ["x", "y"]].to_numpy(float)
    sxy = vis.obs.loc[ov["spot"], ["x", "y"]].to_numpy(float)
    cen = ((cxy - sxy) ** 2).sum(axis=1) <= r * r
    n_cen = (pd.Series(cen, index=ov.index).groupby(ov["spot"]).sum()
             .reindex(vis.obs_names, fill_value=0).astype(int).rename("n_centroid"))
    res = SpotCells(counts, n_cen, ov, float(r))
    if verbose:
        q = counts.quantile([0.1, 0.5, 0.9]).round(1).tolist()
        part = float((ov["frac"] < 0.999).mean()) if len(ov) else 0.0
        print(f"[dapi] cells per spot (area overlap): 10% {q[0]}, median {q[1]}, 90% {q[2]} | "
              f"{int((counts < 0.5).sum())} empty spots | cells straddling the edge {part:.0%} | "
              f"median |area − centre count| {float((counts - n_cen).abs().median()):.1f}")
    return res
