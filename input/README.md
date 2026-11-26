# Input files

Put the input files here. Everything in `input/` except this file is ignored by git.

```
input/
├── visium/
│   ├── filtered_feature_bc_matrix.h5      # or a filtered_feature_bc_matrix/ folder, or .tar.gz
│   └── spatial/                           # or spatial.tar.gz
└── segmentation/
    ├── cells.csv
    └── mask.tif
```

## `visium/` — Space Ranger outputs

| File | Accepted forms | Notes |
|---|---|---|
| `filtered_feature_bc_matrix` | `.h5`, folder (`matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`), `.tar.gz` / `.tar` | Gene Expression and, if present, Antibody Capture features (used as same-spot proteins in sections 1-6, 13 and 14) |
| `spatial` | folder or `.tar.gz` / `.tar` | must contain `tissue_positions(.csv / _list.csv)` and `scalefactors_json.json` |

Files downloaded from 10x Genomics carry the sample name as a prefix (e.g. `<sample>_filtered_feature_bc_matrix.h5`, `<sample>_spatial.tar.gz`); rename them to the names above. Archives are extracted into a local scratch folder on first use.

## `segmentation/` — nuclei of the same section

Any nucleus (or whole-cell) segmentation of the section's image can be used, as long as it is on the **full-resolution image grid of the Visium spatial coordinates**.

**`cells.csv`** — one row per segmented cell

| Column | Content |
|---|---|
| index (first column) | cell id = the label of the cell in `mask.tif` (integer) |
| `area` | area in pixels |
| `y`, `x` | centroid in full-resolution pixels |
| other numeric columns (optional) | per-cell channel intensities (e.g. mean DAPI); used as nuclear features |

**`mask.tif`** — integer label image (0 = background, otherwise the cell id), same pixel grid as `y`, `x`. It is read once to count cells inside each spot's capture circle.

The example data were segmented with Mesmer (DeepCell) on the DAPI channel of the immunofluorescence image. DeepCell needs a personal access token; keep it out of the repository (it is in `.gitignore` patterns such as `.env`).
