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

## 4. The single surviving predictive signal, and its limits

* L1 local geometry improves the leave-one-series-out RMSE over the controls:
  **0.00472 -> 0.00437 (+7.5%)**, and its L1 summary keeps a partial correlation
  of **+0.526** given log occupied prefixes, A1, collision and L1 code usage.
  `local_prefix_geometry/statistics.json`.
* It is **not independent replication**: the strict A2 subset is one mechanism
  series (leave-one-series-out `UNDEFINED_SINGLE_SERIES`), and #131 sampled only
  groups of 2-64 items (L1 item coverage 5.7% under that rule). Issue #132 is
  re-running the census without any cap to decide whether the number survives.

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
* Regression suites: #129 16 tests, #130 10, #131 12, #132 13.

## 6. What is still open

* Issue #132: full-census L1/L2/L3 geometry (108 SIDs, every occupied group, no
  size or count cap). Structural metrics are exact; the four-point deviation is
  exact only on components up to 24 nodes and the detour is
  `BLOCKED_NOT_FULL_CENSUS`; both are recorded with counts in
  `full_census_geometry/completeness.json`. The L1 predictive number above will
  be re-evaluated on that census before any claim is repeated.
