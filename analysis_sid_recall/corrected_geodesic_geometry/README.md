# True-geodesic ORC and corrected boundaries (Issue #134)

Corrects the two defects left by #133 and recomputes the **raw (unadjusted)**
Spearman study at L1/L2/L3. Offline only: no Stage 2/3 training, no loss,
codebook or tokenizer changes, no model-curvature selection.

## D1 — the ORC ground metric

#133 used a 0/1/2-clipped cost, which is not the quotient metric: for support
vertices of an adjacent pair the true shortest-path distance can be 3, and paths
may pass through nodes outside the support. `geodesic.support_distances` builds
the exact cost matrix (adjacency = 1, full-graph two-hop product = 2, otherwise 3)
and `geodesic.w1_exact` solves the transportation problem exactly as a sparse LP
(the earlier 3-threshold max-flow identity is **not** valid for three cost
classes and has been removed). `orc_edge_clipped` is kept only as the explicitly
labelled diagnostic.

Fixtures: on the 4-path, the central edge's true ORC is **0.0** while the clipped
value is **+0.25**; a graph whose shorter support-pair path leaves the support is
tested; the LP matches the reference transport to 1e-9.

## D2 — boundary edges

`cut = sum_B deg(i) - 2*E_B(S)` already counts each undirected crossing edge once,
so `boundary_edges = cut` (the `/2` is gone). Invariants tested: cut equals the
enumerated crossing-edge count, and the sum of the group cuts equals 2x the
cross-prefix item edges. Conductance keeps using the cut and stays in [0, 1].

## Files

| file | contents |
|---|---|
| `geodesic.py` | exact support distances, LP transport, true/clipped edge ORC, cut enumerators |
| `run_measure.py` | per-SID measurement with checkpoint/resume; paired old-vs-new ORC on the **same seeded edges** |
| `run_stats.py` | raw Spearman family (Holm), the paired audit, the #133 comparison, figures, REPORT |
| `features_by_level.csv|json` | per SID and level: corrected intra boundary/geometry, quotient/Forman, true and clipped ORC with coverage |
| `support_L1_L2_L3.csv` | census accounting and boundary-mismatch counts |
| `spearman_by_level.csv` | one row per feature with n, rho, p, Holm p, CI |
| `statistics.json`, `completeness.json`, `timing.json`, `figure_audit.json` | results, exactness ledger, performance, figure coverage |
| `tests/test_geodesic.py` | 13 tests: P4 fixture, outside-support path, LP oracle, symmetry, metric inequalities, boundaries, coverage statuses, provenance, figure honesty |

## Reproduce

```bash
python3 analysis_sid_recall/corrected_geodesic_geometry/run_measure.py --resume
python3 analysis_sid_recall/corrected_geodesic_geometry/run_stats.py
python3 -m unittest discover -s analysis_sid_recall/corrected_geodesic_geometry/tests
```

## Exactness ledger

`completeness.json`: intra/inter metrics exact for every group; Forman on every
quotient edge; ORC is `SAMPLED_ESTIMATE` with 16 edges per level per SID and the
per-row coverage recorded; all-edge ORC is `BLOCKED_NOT_FULL_CENSUS` (about
25k/300k/314k quotient edges per SID at L1/L2/L3); quotient Gromov stays
`BLOCKED_NOT_FULL_CENSUS` and is reported with the halved convention
`(max pair-sum − second)/2`.
