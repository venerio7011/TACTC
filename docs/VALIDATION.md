# Validation criteria and change log

The criteria below were fixed on 2026-11-24, before TACTC (sections 10, 11 and 13 of `TACTC.ipynb`) was run on the example data. Changes made after seeing the data are listed in the change log, and the original results are reported alongside the changed ones.

## Hypothesis

Expression clusters and PAGA (section 10, RNA only) and the nuclear-adjacency graph (section 11, image only) are built separately and then crossed. Spots that both support — part of a contiguous patch of their cluster in the tissue (**core**) — are reliable; spots grouped by expression only — isolated fragments (**scattered**) — are more likely to be false positives.

## Definitions (section 11)

- **Nuclear-adjacency graph.** Two spot nodes (hexagons) are linked when at least one Delaunay edge between nuclei (≤ 30 µm) joins a nucleus of one to a nucleus of the other.
- **Core / scattered.** For cluster c, keep c's nodes and the links among them; each node gets the size of its connected component. Null: the same number of nodes placed at random (200 permutations). p = (1 + #null ≥ observed) / (1 + #null); core if p ≤ 0.05, else scattered. A shift null is not used: shifting the label map keeps its patches, so it cannot ask whether a cluster forms patches.
- **Validity condition (11-4).** A planted contiguous patch (3% of nodes) must be ≥ 90% core and a planted random set of the same size ≥ 90% scattered; otherwise section 13 is not interpreted.
- **Tissue-annotated PAGA (11-3, descriptive).** Cores of two clusters touch when ≥ 10 links join their core nodes and these are ≥ 2% of the smaller core boundary. A PAGA edge (connectivity ≥ 0.03) is *supported* when the cores touch, *through fragments* when links involving scattered nodes outnumber core–core links, otherwise *weak contact* or *no contact*. No null (a shift null overlaps two disjoint clusters; a permutation is dominated by the fragments).

## Tests (section 13): core vs scattered, with information not used to define them

| | Test | Pass rule |
|---|---|---|
| P1 | Reproducibility: 10 random straight cuts × both sides; recluster one side, transfer labels to the other (kNN in PCA space), match to the full labels (Hungarian). Share of nodes keeping their cluster, core − scattered | fit-level bootstrap 95% CI lower bound > 0 |
| P2 | Same-spot measurements: kNN prediction of the cluster from the antibody panel and nuclear morphology, with 500 µm spatial-block folds and a 200 µm buffer. Accuracy, core − scattered, for all nodes and boundary nodes only | spatial block bootstrap CI lower bound > 0, core accuracy above chance, and shift-null p ≤ 0.05 (label map and core / scattered shifted together, 50 shifts ≥ 1 mm) |
| P3 | Regulons: `regulon_eval.exclusion` on regions `<cluster>|core` and `<cluster>|scattered`; share of testable (regulon, region) pairs with \|shift-null z\| ≥ 2, core vs scattered | one-sided Fisher p < 0.05 (core higher); leave-one-core-region-out maximum p reported |
| supporting | P1 on boundary nodes only (nodes linked to another cluster) | if P1 passes but fails on boundary nodes, report that the difference may reflect boundary position |
| supporting | Trajectory: PAGA edges with touching cores; Spearman of pseudotime on all nodes vs on core nodes only (root = OPC/NPC-like rule restricted to core nodes) | descriptive |

**Verdict:** at least 2 of P1–P3 pass → supported; none → not supported; otherwise partially supported.

## Change log

| Date | Change | Reason | After seeing the data? |
|---|---|---|---|
| 2026-11-24 | Criteria fixed | — | No |
| 2026-11-25 | P2: random 5-fold → spatial-block folds + buffer + shift null. P3: untestable pairs (no z) excluded from the denominator; leave-one-region-out added | Both were flaws in how the comparisons were computed. P2: on synthetic data with no label information, random folds passed (core 0.98 vs scattered 0.005; spatial leakage); blocks alone still passed (0.35 vs 0.12), so the shift null was added (leakage only: p 0.48, fail; false fragments: p 0.03, pass; true fragments: p 1.0, fail). P3: untestable pairs were counted as |z| < 2, which penalized small scattered regions | Yes. The original values (random folds; all pairs) are reported alongside |
| 2026-11-25 | Tissue-annotated PAGA: fragment contacts and link marks added to `edge_support` | Report how PAGA edges without core contact touch in the tissue | Yes; the core–core rule is unchanged |
| 2026-11-25 | Post hoc sensitivity analysis: P1 and P2 adjusted for spot quality (RNA count, nuclei per hexagon) by matching and regression | Scattered spots had much lower RNA and nucleus counts than core spots | Yes; the verdict rule is unchanged |
| 2026-11-26 | The quality analysis moved into `tactc/quality.py` and notebook cell 14-4, with spatial block bootstrap intervals (500 µm primary, 1,000 µm sensitivity) instead of node resampling | Neighbouring nodes are not independent; node-level intervals were too narrow | Yes; point estimates unchanged |
| 2026-11-25 | 14-3 spots between clusters (exploratory) | Mixing or transition state? | Yes; criteria fixed before 14-3 was run, not part of the verdict |
