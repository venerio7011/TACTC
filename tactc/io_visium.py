"""
io_visium.py
============

Reads 10x Visium (CytAssist) data for TACTC.ipynb (TACTC). Copied from the visiumif2niche cell2loc.py loader so that
TACTC does not depend on the old modules/ folder.

Contents
----
* load_visium : filtered_feature_bc_matrix (.tar / folder / .h5) + spatial (.tar / folder) → AnnData.
                Genes (Gene Expression) and proteins (Antibody Capture) are separated; spot coordinates are full-resolution
                image px (obsm['spatial'] = x, y); uns['mpp'] = µm/px from the 55 µm spot diameter.
* is_counts   : True if a matrix holds non-negative integers (raw UMI counts)

Coordinate convention: tissue_positions.csv holds px of the full-resolution image given to Space Ranger (pxl_row = y,
pxl_col = x). Mesmer cells (cells.y, cells.x) from the same image use the same coordinates.
"""

from __future__ import annotations

import glob
import json
import os
import tarfile
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["load_visium", "is_counts", "SPOT_DIAMETER_UM"]

SPOT_DIAMETER_UM = 55.0  # Visium spot diameter

_ARCHIVE_EXT = (".tar", ".tar.gz", ".tgz", ".tar.tar")


def _untar(path: str, dest: str):
    with tarfile.open(path) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12
            root = os.path.realpath(dest)
            for m in tf.getmembers():
                if not os.path.realpath(os.path.join(dest, m.name)).startswith(root + os.sep):
                    raise RuntimeError(f"The tar contains a path pointing outside the folder: {m.name}")
            tf.extractall(dest)


def _extract(path: str, dest: str) -> str:
    """Safely extract a tar(.gz) into dest and return dest.

    * Skipped if already extracted. But if only the marker file exists and the content is empty (e.g. an earlier run was interrupted), it is extracted again.
    * If it contains another tar(.gz) (double compression, depending on how it was downloaded), that is extracted in place as well.
    """
    os.makedirs(dest, exist_ok=True)
    stamp = os.path.join(dest, ".extracted_" + os.path.basename(path))
    has_files = any(not f.startswith(".") for f in os.listdir(dest))
    if not (os.path.exists(stamp) and has_files):
        _untar(path, dest)
        for inner in sorted(glob.glob(os.path.join(dest, "**", "*"), recursive=True)):
            if os.path.isfile(inner) and inner.endswith(_ARCHIVE_EXT) and tarfile.is_tarfile(inner):
                _untar(inner, os.path.dirname(inner))
        open(stamp, "w").close()
    return dest


def _find(root: str, names: Sequence[str]) -> Optional[str]:
    for n in names:
        hits = sorted(glob.glob(os.path.join(root, "**", n), recursive=True))
        if hits:
            return hits[0]
    return None


def _resolve(path: str, work_dir: str, sub: str) -> str:
    """If a file (tar), extract it to work_dir/sub and return that folder; a folder or .h5 is returned as is.
    Raises an error explaining the cause if the path is missing or the tar is broken (instead of silently continuing with an empty result)."""
    if not os.path.exists(path):
        d = os.path.dirname(path) or "."
        near = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*"))
                      if sub.split("_")[0].lower() in os.path.basename(f).lower()) if os.path.isdir(d) else []
        raise FileNotFoundError(f"File/folder not found: {path}\n"
                                + (f"  similar names in the same folder: {near}" if near else
                                   f"  no file containing '{sub}' in folder {d}. Check the upload location and name."))
    if os.path.isfile(path):
        if tarfile.is_tarfile(path):
            return _extract(path, os.path.join(work_dir, sub))
        if path.lower().endswith(_ARCHIVE_EXT):
            raise ValueError(f"Cannot open as tar: {path} (size {os.path.getsize(path):,} bytes). "
                             "Check that the Drive upload/sync has finished and the download is complete.")
    return path


def load_visium(matrix: str, spatial: str, work_dir: str = "/content/visium", in_tissue_only: bool = True,
                remove_mt: bool = True, library_id: str = "visium", verbose: bool = True):
    """Read Visium data into AnnData.

    matrix  : .tar(.gz) / folder / .h5 of filtered_feature_bc_matrix
    spatial : .tar(.gz) / folder of spatial
    work_dir: local folder to extract tars into (faster than Drive)

    Returns AnnData
      X                  : gene UMI counts (spot × gene, var_names = gene symbols, var['gene_ids'] = Ensembl)
      obsm['spatial']    : (x, y) full-resolution px.  obs['x'], obs['y'] hold the same values
      obsm['protein']    : protein (Antibody Capture) count DataFrame (if present)
      obsm['MT']         : mitochondrial gene counts (with remove_mt=True they are removed from X and put here)
      uns['spatial']     : image + scalefactors for scanpy plotting
      uns['mpp']         : µm/px = 55 / spot_diameter_fullres
    """
    import anndata as ad
    import scanpy as sc

    matrix = _resolve(matrix, work_dir, "matrix")
    spatial = _resolve(spatial, work_dir, "spatial")

    if os.path.isfile(matrix) and matrix.endswith(".h5"):
        a = sc.read_10x_h5(matrix, gex_only=False)
        a.var["gene_symbols"] = a.var_names.astype(str)
        a.var_names = pd.Index(a.var["gene_ids"].astype(str).to_numpy())
    else:
        mtx = _find(matrix, ["matrix.mtx.gz", "matrix.mtx"])
        if mtx is None:
            raise FileNotFoundError(f"Could not find matrix.mtx(.gz) in {matrix}.")
        a = sc.read_10x_mtx(os.path.dirname(mtx), var_names="gene_ids", gex_only=False)
    a.var_names_make_unique()
    ftype = a.var.get("feature_types", pd.Series("Gene Expression", index=a.var_names)).astype(str)

    prot = None
    is_ab = (ftype == "Antibody Capture").to_numpy()
    if is_ab.any():
        p = a[:, is_ab]
        prot = pd.DataFrame(p.X.toarray() if hasattr(p.X, "toarray") else np.asarray(p.X),
                            index=a.obs_names, columns=p.var_names.astype(str))
    g = a[:, (ftype == "Gene Expression").to_numpy()].copy()
    g.var["gene_ids"] = g.var_names.astype(str)
    # an unnamed index: a named one (e.g. "gene_symbols") clashes with the column of that name once
    # var_names_make_unique renames duplicated symbols, and h5ad writing then fails
    g.var_names = pd.Index(g.var["gene_symbols"].astype(str).to_numpy())
    g.var_names_make_unique()

    sp_dir = os.path.dirname(_find(spatial, ["scalefactors_json.json"]) or "")
    if not sp_dir:
        raise FileNotFoundError(f"Could not find scalefactors_json.json in {spatial}.")
    pos_path = _find(sp_dir, ["tissue_positions.csv", "tissue_positions_list.csv"])
    if pos_path.endswith("_list.csv"):
        pos = pd.read_csv(pos_path, header=None, index_col=0,
                          names=["barcode", "in_tissue", "array_row", "array_col",
                                 "pxl_row_in_fullres", "pxl_col_in_fullres"])
    else:
        pos = pd.read_csv(pos_path, index_col=0)
    pos = pos.reindex(g.obs_names)
    if pos["in_tissue"].isna().any():
        raise ValueError("Some barcodes of the matrix are not in tissue_positions. Check that the files are from the same sample.")
    for c in ["in_tissue", "array_row", "array_col"]:
        g.obs[c] = pos[c].astype(int).to_numpy()
    g.obs["y"] = pos["pxl_row_in_fullres"].astype(float).to_numpy()
    g.obs["x"] = pos["pxl_col_in_fullres"].astype(float).to_numpy()
    g.obsm["spatial"] = g.obs[["x", "y"]].to_numpy()
    if prot is not None:
        g.obsm["protein"] = prot.loc[g.obs_names]

    sf = json.load(open(os.path.join(sp_dir, "scalefactors_json.json")))
    images = {}
    try:
        from matplotlib.image import imread

        for k in ["hires", "lowres"]:
            f = os.path.join(sp_dir, f"tissue_{k}_image.png")
            if os.path.exists(f):
                images[k] = imread(f)
    except Exception:
        pass
    g.uns["spatial"] = {library_id: {"images": images, "scalefactors": sf}}
    g.uns["mpp"] = SPOT_DIAMETER_UM / float(sf["spot_diameter_fullres"])

    if in_tissue_only:
        g = g[g.obs["in_tissue"] == 1].copy()
    if remove_mt:
        mt = g.var_names.str.upper().str.startswith("MT-")
        if mt.any():
            g.obsm["MT"] = pd.DataFrame(g[:, mt].X.toarray() if hasattr(g.X, "toarray") else g[:, mt].X,
                                        index=g.obs_names, columns=g.var_names[mt])
            g = g[:, ~mt].copy()
    g.obs["total_counts"] = np.asarray(g.X.sum(axis=1)).ravel()
    if verbose:
        print(f"[visium] {g.n_obs:,} spots × {g.n_vars:,} genes"
              + (f", protein {prot.shape[1]}" if prot is not None else "")
              + f" | spot diameter {sf['spot_diameter_fullres']:.1f} px → {g.uns['mpp']:.4f} µm/px")
    return g


def is_counts(X, n: int = 20000) -> bool:
    """True if the first n stored values are non-negative integers (raw UMI counts)."""
    v = X.data if hasattr(X, "data") else np.asarray(X).ravel()
    v = np.asarray(v[:n], dtype=np.float64)
    return bool(v.size == 0 or (np.all(v >= 0) and np.allclose(v, np.round(v))))
