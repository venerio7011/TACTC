# TACTC — Tissue Adjacency Cross-Check of Transcriptomic Clusters

TACTC checks which parts of a spot-level expression clustering, and of its PAGA graph, are supported by the tissue.

Spot clusters are often read as tissue compartments, and small, isolated pieces of a cluster are often reported as findings — tumor islands, invasion foci, transition zones. TACTC marks these pieces with evidence that the clustering did not use:

1. **Expression clusters and PAGA from RNA alone** (standard scanpy: HVGs → PCA → kNN → Leiden → PAGA; no spatial information).
2. **A nuclear-adjacency graph from the image alone**: segmented nuclei are joined by Delaunay edges of at most 30 µm, so gaps without cells (tears, necrosis, section edges) break the graph; two spots are linked when a nucleus of one is adjacent to a nucleus of the other.
3. **Crossing the two**, only at the end:
   - **spot marks** — each spot is *core* (part of a patch of its cluster larger than expected for randomly placed spots; permutation test) or *scattered* (an isolated fragment); spots touching another cluster are marked as *boundary* spots;
   - **tissue-annotated PAGA** — each PAGA edge is marked *supported* (the clusters' cores touch), *through fragments* (contact made mainly by scattered spots), *weak contact* or *no contact*.
4. **Tests of the marks with same-spot measurements** that neither graph used (antibody panel, nuclear morphology), plus reclustering reproducibility, regulon specificity and spot-quality checks.

## Crossing instead of combining

Combining tissue position with expression clusters and PAGA-type trajectories is not new. SpaGCN (Hu et al., *Nat Methods* 2021) and BANKSY (Singhal et al., *Nat Genet* 2024) build position into clustering, and the pseudo-time-space method of stLearn (PSTS; Pham et al., *Nat Commun* 2023) combines spatial distance with pseudotime distance between clusters to build spatial trajectories. These methods aim at better clusters and trajectories; because position is used to make them, position can no longer serve as independent evidence for them.

TACTC keeps the two apart. It does not change the clusters or the trajectory; it adds marks to an existing analysis, so that disagreement between expression and tissue can be seen and judged by a third measurement. The same check can be applied to clusters from a spatial method, as long as it uses evidence that the method did not use. TACTC has not been benchmarked against these methods.

## Repository

| Path | Content |
|---|---|
| `TACTC.ipynb` | the pipeline (Google Colab or local Jupyter), sections 0–15 |
| `tactc/` | the modules (Python package) |
| `input/README.md` | which input files are needed and where to put them |
| `docs/VALIDATION.md` | the validation criteria, fixed before the example run, and their change log |

Main sections of the notebook

| Section | Step |
|---|---|
| 1 | spot QC, nuclei per spot, expression programs (spOT-NMF), cell counts per group |
| 2–5 | nucleus graph, hexagon expansion of spots, spot nodes |
| 10 | expression clusters and PAGA (RNA only) |
| **11** | **TACTC: nuclear-adjacency graph, core / scattered spots, tissue-annotated PAGA, planted controls** (11-5 recomputes the PAGA marks on saved results) |
| 12 | per-group regulons with pySCENIC (optional) |
| 13 | validation of the marks: reproducibility (P1), same-spot proteins and morphology (P2), regulons (P3), trajectory |
| 14 | supplementary checks, including spots between clusters: mixing or transition? (14-3, exploratory) and spot quality vs core / scattered with spatial block bootstrap intervals (14-4) |
| 6, 9, 15 | optional group-level analyses |

Key outputs (`output/dapi_support/`): `nodes.csv` (per spot: cluster, component size, permutation p, core / scattered, boundary), `clusters.csv`, `edges.csv` (tissue-annotated PAGA), `boundary_neighbors.csv`, `planted_controls.csv`, and `output/support_step11.h5ad`.

## Quick start

**Google Colab**
1. Copy this repository to Google Drive, e.g. `MyDrive/TACTC`.
2. Put the input files in `input/` as described in [`input/README.md`](input/README.md).
3. Open `TACTC.ipynb` in Colab (GPU runtime recommended for spOT-NMF), check `WORK_DIR` in section 0 and run the sections in order. After a runtime restart, run section 0 and then the `-R` cell of the section you continue from.

**Local Jupyter**
```bash
git clone <this repository> TACTC
cd TACTC
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps spot-nmf rbo
jupyter lab TACTC.ipynb
```
Run the notebook from the repository root; `WORK_DIR` is then the current folder.

TACTC itself (section 11) needs only cluster labels, a PAGA matrix and nucleus positions with their spot assignment, and runs in minutes on a CPU. The full notebook is dominated by spOT-NMF (GPU) and, if run, pySCENIC (hours).

## Inputs

- Space Ranger outputs of a Visium (CytAssist) section: `filtered_feature_bc_matrix` and `spatial`. Antibody Capture features, if present, are used as same-spot proteins.
- A nucleus segmentation of the same section's image: a cell table (`cells.csv`) and a label mask (`mask.tif`). Any segmentation tool can be used.

Details: [`input/README.md`](input/README.md).

## Example data

The example run used the public 10x Genomics dataset *CytAssist FFPE Protein Expression Human Glioblastoma* (one section, 5,591 spots, 178,753 segmented nuclei; nuclei segmented with Mesmer).

## Citation

A manuscript is in preparation; until then, please cite this repository (`CITATION.cff`) and the methods listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

MIT License, © 2026 Jibum Seo. Third-party packages are installed at run time and keep their own licenses (spOT-NMF and pySCENIC are GPL-3.0); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
