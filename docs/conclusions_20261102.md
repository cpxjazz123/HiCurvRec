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
## 4b. Inter-prefix geometry variants (issue #135)

* The frozen registry (70 candidates, sha256 `5625618…`, written before the scan)
  measures A local-to-global aggregation, B parent-conditioned sibling quotients,
  C null-relative versions against three label-shuffle designs (20 deterministic
  replicates each) and D a secondary multi-item-only subgraph. Baselines are
  reproduced exactly: quotient density L2 **-0.451**, quotient transitivity L2
  **-0.465** (identical to #133).
* **A local aggregation does not improve the association.** The strongest local
  entries are local-transitivity p10 at L2/L3 (-0.513 / -0.541) and local-transitivity
  mean/median at L2 (-0.458 / -0.468), all p=2e-4 but Holm **0.061**, and inside the
  76 `curvature_RQ-VAE` arms they collapse to -0.08…-0.40.
* **B raw sibling geometry is weak.** Sibling density is -0.145 at L2 (p 0.13) and
  +0.332 at L3 (n 87, Holm 0.40, within-76 -0.02); sibling transitivity is likewise
  not associated.
* **C is where the signal is, and it is a normalisation effect.** Sibling density
  relative to a size-preserving global shuffle reaches **+0.688** (z) and **+0.723**
  (ratio) at L2, and +0.618/+0.722 with the degree-stratified shuffle; the paired
  |rho| gain over global density is +0.269/+0.272 with bootstrap CIs excluding 0,
  and within the 76 arms it stays +0.35…+0.39. **But Holm is 0.061** (304-test
  family), and the *parent-conditioned* null of the same feature is **-0.346**:
  the gain comes from the global size-preserving null, not from within-parent
  structure, so it is not a hierarchy-preserving effect.
* **D** (multi-item-only induced subgraph) gives density L2 -0.487, transitivity
  L2 -0.583 (n 84, singleton-heavy rows excluded) and Forman mean L3 +0.521, all
  Holm 0.061 and within-76 -0.15…-0.19.
* **Verdict.** Under the predeclared rule (coverage >= 0.95, paired |rho| CI above
  zero, direction stable within 76) three C candidates are a **GAIN**, but the
  evidence is **borderline** (Holm 0.061). Under a stricter rule requiring
  Holm < 0.05 the verdict is **NO STABLE GAIN**. Every L2/L3 association here is
  to a large degree a cross-series effect; an independent SID family is required.
* Cost: 12 shards, median 725 s/SID, peak RSS 0.95 GB, 78k null applications
  (nulls are 96% of the time; the L2/L3 quotient `A^2` has ~46-50M non-zeros).

## 4c. AF3 curvature and the original-graph within/cross gap (issue #136)

* Frozen convention on every graph: `F_H(e)=4-deg_H(u)-deg_H(v)`,
  `T_H(e)=|N_H(u)∩N_H(v)|`, `AF3_H(e)=F_H(e)+3T_H(e)`, with degree and triangles
  taken in the graph the edge belongs to (induced / quotient / original). 108
  rows, 2 digest mismatches excluded, all `RECORD_HASH_MATCH`; one global
  original-graph curvature cache over **315,391** edges (1.2 s).
* **The original-edge gap is the strongest geometry association measured so far
  and it survives every control tested.** `gap_af3` vs R@10: Pearson **+0.809 /
  +0.773 / +0.600** and Spearman **+0.760 / +0.685 / +0.524** at L1/L2/L3
  (n 107/108/108, Holm 0.013 over the 260 defined metric×outcome entries); NDCG@5
  at L1 reaches r +0.832 / rho +0.814. Within the 76 `curvature_RQ-VAE` arms it
  stays r +0.526/+0.608/+0.451 and rho +0.522/+0.457/+0.253.
* **Incremental evidence.** Controlling hard-prefix edge retention, the partial
  Pearson of the gap vs R@10 is **+0.79 / +0.75 / +0.59** (p = 2e-4); controlling
  L1 intra transitivity it is +0.43 (rho +0.22), and controlling L2/L3 quotient
  density +0.53/+0.47 (rho +0.62/+0.49). The gap is therefore not just frontier
  size or a re-encoding of transitivity/density.
* **Triangles matter, but not for the gap mean.** The ordinary-Forman gap
  correlates almost identically with the outcomes (r +0.826 at L1 NDCG@5), and the
  AF3 increment *given* the Forman gap is only partial r +0.32/+0.26/+0.48
  (rho +0.59/+0.51/+0.44). Where AF3 does change the picture: the intra AF3
  negative-edge fraction at L1 (r +0.573 / rho +0.670, Holm 0.013) is a defined
  statistic while the ordinary-Forman negative fraction is degenerate; and the L1
  quotient AF3 mean flips sign against the ordinary-Forman mean (r +0.394 vs
  -0.423). The intra AF3 *mean* itself has a Pearson/Spearman sign conflict
  (r +0.40, rho -0.32 at L1): outlier-driven, mixed-arm, not stable.
* **Blocked, not filled in.** D3/prefix alignment is `NOT_AVAILABLE` (no
  implementation in the repository; edge retention is the named substitute);
  AF3-vs-ORC on the identical sampled edges is `NOT_AVAILABLE_BY_DESIGN` (#134
  stores aggregates only); R@20/NDCG@20 are `NOT_EVALUATED`.
* Cost: median 4.9 s/SID, max 5.5 s, phases intra 116 s / quotient 254 s /
  gap 6.9 s / nulls 122 s, peak RSS 1.28 GB; 324 support rows.
* Not claimed: AF3 is not the hyperbolic sectional curvature `-c_l`, a large gap
  does not prove hyperbolic necessity, and the gap is not deduced from
  `mean intra - mean quotient`.

## 4d. Audit of #136: correct inference and negative-curvature semantics (issue #137)

* **B1 (permutation null).** #136 resampled outcomes **with replacement** and did not
  re-rank its Spearman sample: that is a bootstrap null, not the finite-sample
  permutation null. The audit uses a uniform **bijection** of all `n` outcome
  indices per draw and `p=(1+#{|stat|>=|obs|})/(B+1)`; bootstrap stays a
  with-replacement **paired** CI. `stats_core.py` is validated against an
  exhaustive n=5/6 enumeration and `scipy.stats.permutation_test` (pairings, exact).
* **B2 (Holm).** #136 adjusted `min(p_r, p_rho)` once and wrote one `holm_p`. The
  audit reports `p_pearson`, `p_spearman`, `holm_pearson`, `holm_spearman`, each
  adjusted within its own column over 66 metric-levels x 4 outcomes x 2
  coefficients = **528 potential tests** (528 defined, 0 undefined; the issue's
  520/8 expectation described #136's field set, where four entries were degenerate
  and the mislabelled field has been replaced by a defined one).
* **B3 (negative-curvature semantics).** The old `intra_af3_negative_fraction` was
  the fraction of **groups whose mean AF3 is negative**. New fields:
  `group_mean_negative_fraction`, `edge_negative_fraction` (negative edges over
  all eligible induced edges) and `mean_group_negative_edge_fraction`, with
  ordinary-Forman analogues (`feature_semantics_audit.csv`). They are not
  interchangeable: at L1 the group-mean fraction is r +0.573 / rho +0.670
  (Holm 0.0048) while the true edge fraction is r +0.113 / rho +0.197 (Holm 1.000);
  at L2 the edge fraction is Pearson-informative (r -0.423, Holm 0.017) but rank-null
  (rho -0.180, Holm 1.000). **No single negative-curvature claim survives both
  coefficient types.**
* **H1 still holds.** `Gap_AF3` vs R@10: r +0.809 / +0.773 / +0.600 and rho
  +0.760 / +0.685 / +0.524 at L1/L2/L3, Holm **0.0132** in both columns, within the
  76 arms r +0.526 / +0.608 / +0.451, and the association survives removing
  Cook-flagged arms (L1 r +0.732, L3 r +0.514). The corrected inference and the
  per-column Holm leave the strong effect intact.
* **H2/H3/H4 are `MIXED/NULL_SENSITIVE`.** Beyond `{Gap_Forman, edge retention,
  L1 transitivity | quotient density}`, the three-control partial is significant
  for both coefficient types only at **L2** (+0.341/+0.269); at L1 the Pearson
  partial is -0.088 (p 0.35) while the rank partial is +0.232 (p 0.016), and at L3
  the Pearson partial is +0.379 (p 2e-4) while the rank partial is -0.034 (p 0.73).
  Leave-series-out RMSE improves at L1/L2 (0.00639->0.00411, 0.00751->0.00725) but
  **worsens at L3** (0.00681->0.00748). Under the degree-stratified label shuffle
  the gap z changes sign at L2/L3 (because that null's mean is strongly negative)
  although the **null-centred gap stays positive at every level**; no null scheme
  was chosen by its Recall correlation.
* **Retracted from #136**: the "negative-edge fraction at L1 rho +0.670" claim
  (it was the group-mean fraction), every old p/Holm value, the claim that the gap
  adds beyond the controls at every level, and the implication that the gap result
  is null-insensitive. What stands is the raw gap association.
* Blocked: D3 `NOT_AVAILABLE`, per-SID collision/code usage `NOT_INCLUDED`
  (resolution controls are occupied-prefix count and singleton share),
  R@20/NDCG@20 `NOT_EVALUATED`. Cost: 108 rows, one cached curvature pass,
  median 4.8 s/SID; 32 audit tests.

## 4e. Curvature-guided prefix ranking, the first full run (2026-11-03)

* **Arm.** `stage2_RQ-VAE/TIGER_curvature-prefix-ranking/`, copied from the last
  complete mechanism `TIGER_poincare-frechet-successor-codebook`. AF3 cohesion on
  the TRAIN behaviour graph (`315,391` edges, median `-128.0`, scale `177.17`)
  selects a cohesive and a bridge neighbour per anchor inside the batch; a ranking
  loss asks the cohesive neighbour's soft prefix overlap to exceed the bridge
  one's at all three levels. The overlap is the inner product of the softmax of
  the negative Poincare distance from each level's residual tangent to that
  level's codebook; behaviour-off (random triples, unit weights) and geometry-off
  (squared Euclidean cost) exist in code and are exercised only by tests.
* **Admission and run.** Machine admission PASS (constants, geometry contract,
  one-epoch entry, 24,587 unique four-token SIDs, four-rank probe); one-epoch
  gradient smoke PASS (24 steps, no dead/non-finite terms, every step norm
  positive); training on four cards, 3000 epochs, 0.238 s/epoch, 12 minutes.
  Final loss `5.66e-4`, recon `8.93e-5`.
* **Pre-training signal.** The existing SID partitions already rank cohesive
  neighbours above bridges: +8.8 pp at L1 (positive in 98.1% of the 108 arms),
  +1.46 pp at L2, +0.61 pp at L3 (`analysis_sid_recall/curvature_prefix_signal/`).
* **Stage2 diagnostics** (technical validity only, no threshold):
  Hard-L1 ER **5.02** (reference 16.39), L2 distinct prefixes **1,985**
  (reference 16,145 -- a much coarser L1, hence fewer L2 prefixes), L3 collision
  **0.831**, L3 refinement gain **2.56 bits**, conditional G3 gain **-0.027**.
  Recorded before the Stage3 score was read only in the sense that the SID exists
  first; the record itself was written after the run and is dated as such.
* **Stage3 result (first SID)**: `test_recall@10` **0.054057**, R@5 0.036508,
  NDCG@5 0.024011, NDCG@10 0.029662, n_eval 57,439. The frozen target is `> 0.065`
  and the Resample baseline is 0.0637372, so this arm **does not beat the baseline**
  and is recorded as a negative result, not hidden.
* **The owner's LEARNED collision gate (final rule, 2026-11-03).** Stage2
  admission now requires the three-level collision to be below the owner's
  threshold **as learned by training**; export-time dedup is forbidden and is
  detected mechanically (the gate reads the trainer's final collision from the
  run's metrics and refuses an exported SID that disagrees with it, or a learned
  value above the threshold, or a missing metrics file). The threshold was set at
  20% and relaxed by the owner to **35%** after the measured floor.
* **Measured iteration record** (learned collision, gate definition): curved
  Frechet assignment 0.831; Euclidean assignment with per-level Sinkhorn 0.334;
  plus the hierarchical diversity term with a per-item nearest-code temperature
  0.332; an L3 usage penalty 0.442 and an L2 pair-usage penalty 0.449 (both worse,
  reverted); product assignment over the shared latent 0.999 (collapsed,
  reverted). The frozen baseline itself is 0.2745, i.e. the original 20% gate was
  stricter than every existing arm.
* **Final gate-passing run**: the restored configuration (Euclidean assignment,
  per-level Sinkhorn, AF3 ranking term, hierarchical diversity, per-item
  temperature) learned **0.3421**, the exported SID equals the log (no
  post-processing), and the gate admits it. Layered metrics `399aa8d9…json`:
  collision 0.3421, L2 prefixes 8,305, Hard-L1 ER 8.85, L3 refinement 1.85 bits,
  conditional G3 -0.037.
* **Stage3 result of the gate-passing SID**: `test_recall@10` **0.053605**, R@5
  0.035359, NDCG@5 0.023199, NDCG@10 0.029067, n_eval 57,439. Together with the
  earlier runs (0.054057 and 0.052334) the arm is a consistent **negative result**
  against the 0.065 target and the 0.0637372 baseline; the collision gate is
  satisfied by the learned assignment, not by an export step, and the broader
  claim of the mechanism is not supported.

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
