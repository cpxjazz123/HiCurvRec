# What is established as of 2026-11-02 (issues #128-#132)

Every number below is reproducible from the artefacts named with it. Claims are
split into **withdrawn**, **established observation**, **established invariant**
and **still open**; nothing here asserts a causal geometry -> Recall benefit.

## 1. Withdrawn: the curvature-recall association (#128)

* The pre-fix profile correlation (L1 minus L3 curvature vs Recall@10, rho
  -0.184) came from a one-direction (upper-triangular) behaviour graph and a
  bounded ground metric. With the undirected graph and the exact metric it is
  **-0.047**, and every feature is null: L1 -0.075, L2 +0.026, L3 -0.077.
  `analysis_sid_recall/curvature_before_issue128/before_after_summary.json`.
* The estimator converges at **64 sampled edges and 4 null replicates**
  (|delta mean| / SE <= 0.02; estimator SE 0.004-0.008 against an across-arm sd
  of 0.036-0.126). `analysis_sid_recall/curvature/convergence/`.
* The coefficient is **not stable across budgets**: 16 vs 32 edges moves L3 from
  -0.049 to +0.145, and the null-replicate count alone moves L2 from -0.136 to
  -0.263 (p 0.25 -> 0.026). `analysis_sid_recall/CONVERGENCE_FINDINGS.md`.

## 2. Established: the raw graph statistics are resolution proxies (#129-#131)

| quantity | value | control |
|---|---|---|
| raw quotient curvature vs Recall (L2 / L3) | -0.401 / -0.371 (Holm 0.002 / 0.001) | correlation with log quotient edges -0.929 / -0.821; partial given resolution **-0.005 / +0.022** |
| standardised curvature z_l | null at all levels (Holm 0.61 / 0.25 / 0.25) | — |
| joint model with z1,z2,z3 | leave-one-series-out RMSE 0.00349 -> **0.00926 (worse)** | #129, `layerwise_curvature/results.json` |
| interaction-strength block (#130) | step A->B **-7.5%** (worse); D recovers +11.2% over C but 0.00366 vs controls 0.00359 | `interaction_strength/statistics.json` |
| weighted ORC vs Recall | partial +0.046 / -0.060 / -0.293 after resolution controls | same |

## 3. Established observation: where the organising signal actually is

* **Interaction strength is concentrated inside the first-level prefix**: median
  per-arm retention excess over the size-preserving null **+0.084**; same-prefix
  share by strength tertile 0.041 -> 0.063 -> 0.088 at L1. Across arms the excess
  tracks Recall only at L1 (+0.524, p 2e-4, Holm 0.005).
  `interaction_strength/REPORT.md`.
* **L1 multi-item groups are cyclic, non-tree, low-conductance subgraphs**, and
  they cover all items; refinement replaces them with singletons: item coverage
  by multi-item groups 1.000 -> 0.655 -> 0.152. **L3 is 92.9% singleton groups
  and is labelled STRUCTURALLY_UNINFORMATIVE** for clustering geometry.
  `local_prefix_geometry/REPORT.md`.
* Against fixed-size and activity-stratified shuffles the L1 geometry differs
  (transitivity delta +0.124 / +0.112); the L2 parent-conditioned delta is
  +0.051 and the L3 +0.000.

## 4. Geometry vs Recall: the corrected raw-Spearman table (issue #133)

Issue #132's intra-group numbers and its +9.8% RMSE are **superseded**: those
metrics mixed singletons into the multi-item averages, mixed weighted degrees with
an unweighted cut, and counted triangles with a dense `int8` cube that overflows
on large groups (on a real 1,000-item group the old path returns 481 triangles;
the exact count is 524).

Corrected measurement (108 rows, every metric with raw Spearman, permutation p,
bootstrap CI and Holm over the predeclared 81-test family):

| block | feature (level) | rho | Holm p | n |
|---|---|---|---|---|
| intra | transitivity (L1 / L2 / L3) | **+0.828 / +0.814 / +0.583** | 0.015 / 0.015 / 0.015 | 108 / 82 / 81 |
| intra | conductance (L1 / L2 / L3) | **-0.604 / -0.164 / -0.406** | 0.015 / 1.000 / 0.016 | 107 / 108 / 108 |
| intra | largest component (L1) | +0.573 | 0.015 | 108 |
| intra | cycle rank / node (L1) | +0.449 | 0.015 | 108 |
| inter | density (L2 / L3) | **-0.451 / -0.479** | 0.015 / 0.015 | 108 |
| inter | transitivity (L2 / L3) | **-0.465 / -0.438** | 0.015 / 0.015 | 108 |
| inter | degree median (L3) | -0.562 | 0.015 | 108 |
| inter | Forman mean (L3) | **+0.420** | 0.015 | 108 |
| inter | singleton share (L2 / L3) | +0.419 / +0.485 | 0.015 / 0.015 | 108 |
| curvature | ORC mean, true geodesic (#134; L1 / L2 / L3) | +0.130 / **-0.415** / **-0.401** | 1.000 / 0.014 / 0.014 | 107 / 108 / 108 |
| curvature | ORC mean, #133 clipped metric (superseded) | +0.112 / -0.451 / -0.420 | 1.000 / 0.015 / 0.015 | 107 / 108 / 108 |

* The intra-group geometry (cyclic, non-tree, low-conductance subgraphs) tracks
  Recall most strongly at L1 and still at L3 at the **group** level; the inter-group
  quotient geometry tracks it at L2/L3 mainly through density/degree/transitivity
  (denser and more transitive quotients go with *lower* Recall).
* Ollivier-Ricci is a **sampled estimate** (16 quotient edges per level per SID,
  per-row coverage 6e-4 at L1 and 5e-5 at L2/L3; all-edge ORC is
  `BLOCKED_NOT_FULL_CENSUS`) and is therefore reported separately; Forman
  curvature covers every quotient edge exactly.
* **Issue #134 corrected the ground metric.** The old 0/1/2 cost clip was not the
  quotient metric: on the same seeded edges every L2/L3 SID has support pairs
  whose true shortest path has 3 hops, and the clip biased the sampled curvature
  upward. Paired audit (same edges): mean Δκ = -0.0154 (L2) / -0.0280 (L3), max
  |Δκ| = 0.31, sign flips 14 (L2) / 1 (L3); L1 has no distance-3 pairs, so its
  ORC is unchanged. The clipped numbers above are **superseded**; the corrected
  table and the boundary fix (`boundary_edges = cut`, the earlier `/2` removed)
  live in `analysis_sid_recall/corrected_geodesic_geometry/`. The boundary and
  conductance correlations are scale-invariant and identical to #133.
* Raw correlations are unadjusted by design (the issue's primary analysis);
  partial correlations given resolution, alignment, collision and code usage are
  secondary and live in `intra_inter_geometry/statistics.json`. A raw coefficient
  here is not evidence that geometry causes Recall.
## 5. Established invariants (now enforced by tests)

* SID-recall admission is **directory-name equality**; the join key is
  `sid_sha256`, never the variant name (182 rows, 168 names). `CLAUDE.md` 64b.
* A SID digest mismatch is **`HASH_MISMATCH` and the pair is excluded**; 108
  pairs are content-matched, 2 mismatched. A prospective Stage 3 must use
  `layerwise_curvature/attest.py` (fail-closed on hash or item order).
* The behaviour graph is undirected (symmetrised) before any degree or curvature
  use; statistics never read validation/test labels; undefined quantities are
  written `UNDEFINED_*` or `BLOCKED_*`, never zero; a figure with supported data
  may not render "no defined values" (a test caught a real instance of this).
* Regression suites: #129 16 tests, #130 10, #131 12, #132 13, #133 (intra/inter
  geometry) as before, #134 13 tests (P4 fixture: true ORC 0.0 vs clipped 0.25,
  outside-support shortest paths, LP vs reference transport, cut identities).

## 6. Status

* Issue #132 is complete: the full census ran on 108 SIDs (2 excluded after a
  measured digest mismatch), every occupied group of every level is counted, and
  the completeness ledger records the metric coverage and the two blockers
  (`full_census_geometry/completeness.json`).
* Open for a future issue, not for this analysis: any claim of cross-family
  replication, and any causal statement from graph geometry to model curvature.
  Both require a controlled Stage 2 -> Stage 3 experiment, which no issue here
  launched.
