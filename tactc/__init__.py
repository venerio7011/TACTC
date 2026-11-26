"""
tactc
=====

TACTC (Tissue Adjacency Cross-Check of Transcriptomic Clusters): the modules used by TACTC.ipynb.
Expression clusters and PAGA are built from RNA alone, a nuclear-adjacency graph from the image alone, and the
two are crossed only at the end (dapi_support: core / scattered spots, tissue-annotated PAGA).

Step 1 (TACTC.ipynb 1-0 … 1-6)
* io_visium    : Visium loading (genes, proteins, coordinates)                     ← from visiumif2niche cell2loc.py
* qc           : spot / gene QC (same rules as visiumif2niche 9-1a)                 ← copy of visiumif2niche qc.py
* dapi_counts  : DAPI (Mesmer) cell count inside each spot circle                  ← from visiumif2niche spot_counts.py
* programs     : reference-free deconvolution into K programs with spOT-NMF (optimal-transport NMF), naming
* cell_counts  : program RNA shares + DAPI counts → cell counts per group (RNA-per-cell correction from visiumif2niche)

Steps 2–4 (TACTC.ipynb 2 … 4-R)
* cell_graph   : all DAPI cells as nodes of one spatial graph (Delaunay / kNN); the cell-level AnnData
* hexagons     : hexagon expansion, cell regions (circle / ring / …), hex / circle ratio, step-1 counts scaled to hexagons

Step 5 (TACTC.ipynb 5 … 5-R)
* spot_nodes   : one node per hexagon (spot centre): cell counts, normalised / scaled / per-group expression,
                 morphology, cell-weighted hexagon lattice graph

Steps 6 and 9 (TACTC.ipynb 6, 9 … 9-R)
* node_graphs  : local group signatures, per-group graphs (6); shared shift-null / split-half helpers
* cross_groups : overlap and neighbour correlations between groups with shift and decomposition nulls (9),
                 replication split and pseudo-pair negative control

Step 10 (TACTC.ipynb 10 … 10-R)
* paga_nodes   : preprocessing, expression clusters and PAGA (no spatial information), k sensitivity

Step 11 (TACTC.ipynb 11 … 11-R) and step 13 (validation, pre-registered, see docs/VALIDATION.md)
* dapi_support : DAPI node graph from the cell graph; core (PAGA ∩ DAPI) vs scattered (PAGA only) nodes with a
                 permutation null; tissue-annotated PAGA and boundary neighbours; planted controls (11);
                 validation of core vs scattered — expression reproducibility, independent measurements,
                 regulons (via regulon_eval.exclusion), pseudotime on core nodes (13)

Step 12 (TACTC.ipynb 12 … 12-R)
* grn          : per-group pySCENIC (GRNBoost2 → cisTarget → AUCell) on (node × group) units

Used by step 13 and the supplement (14)
* regulon_eval : exclusion (C1 shift-null FDR, C2 beyond composition, C3 split-half replication, naive test)
* trajectory   : state scores, OPC/NPC-like root, DPT
* paper_checks : composition vs marker-score baseline, morphology check (14); step 9 checks (15, optional)
* transition   : transition-node candidates between clusters, mixture null + interior specificity check (14-3, exploratory)
* quality      : spot quality vs core / scattered: quality filter, matched and adjusted differences, spatial block
                 bootstrap intervals (14-4, added after the primary results)

"""

__version__ = "0.1.0"

__all__ = ["io_visium", "qc", "dapi_counts", "programs", "cell_counts", "cell_graph", "hexagons", "spot_nodes", "node_graphs", "cross_groups", "paga_nodes", "grn", "paper_checks", "regulon_eval", "trajectory", "dapi_support", "transition", "quality"]
