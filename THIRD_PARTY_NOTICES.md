# Third-party software and data

The MIT license in `LICENSE` covers the code in this repository only. The notebook installs and calls the packages below at run time and reads the data below; none of them are redistributed here. Each keeps its own license. Confirm the licenses on each project's page before redistributing anything.

## Software

| Package | Used for | License |
|---|---|---|
| spOT-NMF (`spot-nmf`) | expression programs, section 1-3 | GPL-3.0 |
| pySCENIC | regulons, section 12 (optional) | GPL-3.0 |
| arboreto, ctxcore | GRNBoost2, cisTarget, section 12 | BSD 3-Clause |
| scanpy, anndata | data handling, clustering, PAGA, diffusion pseudotime | BSD 3-Clause |
| leidenalg, igraph | Leiden clustering | GPL-3.0 / GPL-2.0-or-later |
| PyTorch | spOT-NMF backend | BSD-style |
| numpy, pandas, scipy, scikit-learn, matplotlib, h5py, tifffile, zarr, joblib, psutil | general | BSD-style / permissive |
| DeepCell (Mesmer) | nucleus segmentation of the example data (run separately; not called by TACTC) | Modified Apache 2.0 — non-commercial use only |

## Data and resources

| Resource | Used for | Terms |
|---|---|---|
| 10x Genomics *CytAssist FFPE Protein Expression Human Glioblastoma* | example data | 10x Genomics dataset terms |
| aertslab cisTarget databases and motif annotations (hg38) | regulon pruning, section 12 (optional) | see resources.aertslab.org |

## Code adapted from the author's earlier work

`tactc/qc.py`, `io_visium.py`, `dapi_counts.py`, the marker sets in `programs.py`, the RNA-per-cell correction in `cell_counts.py`, the hexagon lattice in `hexagons.py`, the spot decomposition in `spot_nodes.py` and the pySCENIC wrappers in `grn.py` are adapted from visiumif2niche (Jibum Seo, MIT License).

## Citing

If you use TACTC, please also cite the methods it builds on: scanpy (Wolf et al., Genome Biol 2018), PAGA (Wolf et al., Genome Biol 2019), Leiden (Traag et al., Sci Rep 2019), diffusion pseudotime (Haghverdi et al., Nat Methods 2016), spOT-NMF (Abdelkareem et al., bioRxiv 2025), SCENIC / pySCENIC (Aibar et al., Nat Methods 2017; Van de Sande et al., Nat Protoc 2020) and, if you used it for segmentation, Mesmer (Greenwald et al., Nat Biotechnol 2022).
