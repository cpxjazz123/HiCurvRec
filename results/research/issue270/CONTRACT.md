# Issue #270 frozen analysis contract

Written before any correlation is computed. Parent: #269 (calibrated per-triple curvature, Stage3
R@10 `0.045230592455`) and #265 (R@10 `0.052594926791900976`). This is an **analysis-only**
contract: no training, no mechanism change, no public-infra change.

## 1. Question and falsifiable predictions

**Question.** Which triplet-level geometry improvements, under which SID conditions, stably
explain or predict downstream recommendation changes? Explicitly attempt to falsify
"curvature-distortion reduction necessarily improves recommendation".

| id | prediction (pre-registered) | would falsify |
|---|---|---|
| P1 | Arm-level mean A.1 curvature-error reduction is positively associated with R@10 change across archived arms | Spearman <= 0 with a CI excluding 0 on the positive side |
| P2 | The association is conditional on SID structural preservation: only arms whose prefix/L2-split/collision fidelity is preserved show the positive association | curvature improves but SID collapses and recall still rises, or no interaction |
| P3 | #269's loss is attributable to SID capacity/prefix compression, not to curvature error: its A.1 error improves while L2 prefix count and recall fall | #269's A.1 error is not better than #265's under the common ruler |
| P4 | Arm-level associations are dominated by mechanism family/protocol (single vs four cards) rather than the curvature metric | within-family association remains strong and protocol does not explain variance |

## 2. Cohort and identity

- Every Stage3 run with `test_final.json` is enumerated (204 rows).
- **Usable geometry cohort**: exact SID + checkpoint + complete predictions (57,439 rows) = 42 arms
  (`COHORT.json`). 12 are `record_verified` against the immutable run's recorded sha256; 30 are
  `arm_dir_legacy` because the pre-`_runs` Stage3 record carries no SID/checkpoint path. Legacy
  bindings are used read-only and marked; no hash is invented.
- Missing/duplicate SIDs are reported, and arms without a recoverable SID/checkpoint or without
  complete predictions are **excluded** from geometry (they may still appear in the inventory and
  recall tables). No imputed value is used.
- **TIGER is the only baseline**; it is read-only and was not retrained. Historical TIGER reruns
  (`0.040199`–`0.055537`) are reported as a range, not collapsed.

## 3. Frozen triple, reference and target definitions

`assets/train/visible_undirected.npz` (sha256 `097a719a…`), `assets/v8_runtime_layout/targets.npy`
(sha256 `8c0ff9f8…`) and `target_offsets.npy` (`33f30b29…`). 28,391,948 legal rows; the A.1 target
is unchanged:

```
kappa_graph(m;b,c) = ( S2(m) + N*g(b,c)^2/4 - (S2(b)+S2(c))/2 ) / N
```

Directed labels stay directed; the graph distance `g` is the undirected shortest path. Structure
strata come from the directed training graph and the canonical visible graph; no new graph is built.

Three quantities are never mixed:

1. **Δ_A.1** — error of the equal-weight three-expert A.1 proxy on the hard q under the **common
   fixed ruler** (below).
2. **J** — graph-distance relative distortion of the fixed Euclidean metric on the hard q, on the
   frozen 20,000-row / 256-reference sample (full-triplet J is not claimed).
3. **kappa_t** — the #269 fitted per-triple curvature, used only as an extra common ruler on the
   same 20,000-row sample; it is not a target and is not refitted per arm.

## 4. Common fixed ruler

All arms are evaluated with **one scale**: the frozen `tangent_scale` of the curvature family's
common parent #265 (`0.08433422446250916`, `results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/…/rqvae_best.pth`;
recorded in `analysis/COMMON_RULER.json`). Change recorded 2026-11-27: the TIGER baseline has no
`tangent_scale` key, so the #265 parent scale that defines the common tangent space is used; #269's
own `0.0825626328587532` is reported as a sensitivity. The distance is the baseline's equal-weight three-expert
`fixed_three_squared_numpy(q_i, q_j, scale=SEALED_SCALE)` and the A.1 moment identity over the full
component references, exactly the `tools/issue251.full_evaluation._moments` path used by the frozen
evaluation. **No arm-specific scale is used for the common-ruler numbers**; the arm's own
`tangent_scale` is reported as a sensitivity only.

Hard q is reconstructed exactly from the checkpoint codebooks and the exported SID; no encoder
forward is needed for q. Continuous `z` is reported only where the immutable source snapshot is
recoverable (#265, #267, #269) and never substitutes for q.

## 5. Metrics (all pre-registered)

Per triplet `t`:
- signed error `e_t = predicted_t - target_t`; absolute error `|e_t|`;
- sign correctness `sign(predicted)=sign(target)`; corrected sign indicator;
- within-centre ranking: for each centre with >=2 valid rows, Spearman between predicted and target
  (centre-equal aggregation);
- Pearson/Spearman over all rows (micro) and centre-equal macro means.

Per item: mean signed/absolute error over the triplets whose centre is the item, weighted variants
(row-equal, centre-equal, training-transition-frequency-weighted). Weights come only from the
training graph.

Per SID triplet: L1/L1+L2/L1+L2+L3 prefix sharing among (m,b,c); LCP lengths; full collision;
whether the shared prefix equals the ground-truth chain/fork/join rule (treated as a hypothesis,
not a label). Per item: prefix purity (share of its prefix group with the same structural role),
prefix group size, L2 split (distinct L2 under its L1), L3 refinement gain, effective rank,
quantisation error, collision.

Recommendation: HIT@5/10 and NDCG@10 from the saved predictions, paired to TIGER by `global_row`
(the frozen test order). The target item's per-item curvature/SID deltas are attached to each
sample; a sample may map to multiple triplets, so triplet-level attribution uses
`1/deg(item)`-style normalisation and is reported separately from item-level attribution.

## 6. Strata, aggregation and statistics

Strata: directed 3-node graphlet class (chain, fork, join, reciprocal, triangle/closure, other),
orbit role of m/b/c, edge direction, sign and magnitude of the target, centre degree decile,
training transition frequency decile, co-occurrence decile, triangle-closure flag, centre triplet
participation decile.

Aggregation: **macro** (centre-equal) and **micro** (row-equal), plus training-frequency-weighted;
no high-frequency centre may dominate the headline.

Statistics:
- arm-level: Spearman/Kendall/Pearson between mean curvature-error delta and R@10 delta, with a
  **family-cluster bootstrap** (2,000 draws; resampling families, not rows) and exact permutation
  p-values where feasible;
- item-level: paired deltas vs TIGER, cluster bootstrap by item/centre, Spearman and sign tests;
- four quadrants: Δcurvature error (improve/worsen) x SID fidelity (improve/worsen), with hit-change
  rates and Wilson intervals;
- multiple strata: Benjamini–Hochberg FDR at q=0.10, reported with raw p;
- **never** treat 28.4M correlated rows as independent; effective n is arms/centres, reported;
- leave-one-family-out: the sign and CI of the arm-level association are recomputed excluding each
  family; a relation that flips sign is reported as unstable;
- all candidate composite indicators are fixed here (no weight search): the composite is the
  z-scored mean of (A.1 error delta, sign-accuracy delta, within-centre Spearman delta) and is used
  only if its single components are reported too.

**Development vs test.** Geometry and SID metrics come from training-side assets; the recommendation
linkage uses the frozen historical test predictions **only as explanatory error analysis**, never for
weight/threshold/model selection. No validation predictions exist (Stage3 `NO_EVAL=True`); this is
recorded as an evidence gap (`E1`).

## 7. Exclusions and failure policy

- Arm excluded from geometry if SID/checkpoint missing or predictions incomplete; excluded rows are
  listed with the reason, never imputed.
- Any arm whose exported SID does not reproduce the checkpoint's q is flagged and excluded from the
  geometry relationship (reported, not silently fixed).
- If P1–P4 are not supported, the verdict states "not identifiable with the available independent
  runs" and reports the evidence gap; no monotone relation is manufactured.

## 8. Post-registration outcome (2026-11-27)

**Status: RESOLVED — analysis-only, null relation.**

- Arm-level association: Spearman −0.068, family-cluster bootstrap 95 % CI [−0.51, +0.35],
  permutation p = 0.66 over n = 42 arms; Kendall −0.060; the Pearson +0.41 is carried by
  scale-outlier arms and contradicts every rank statistic. Leave-one-family-out stays in
  [−0.26, +0.01] (one family at a time); the single-card stratum is n = 3 and degenerate.
- P2: compressed stratum ρ = −0.90 (n = 5), preserved ρ = +0.065 (n = 37);
  difference CI [−0.85, +1.44], includes 0 → **not supported**.
- P3: on the Σ absolute ruler #269 improves A.1 by 0.249 over #265 while L2 groups fall
  170 → 113 (127 splits, 93 merges) and ΔR = −0.00736; the relative J and the arm's own κ_t
  ruler reverse the A.1 sign (#265 J 0.168 vs #269 0.288) → **supported only on one ruler;
  not a causal proof**. `DIAGNOSIS.md` holds the per-item and L2-group reconstruction.
- P4: between-family SD 0.00356 vs within-family 0.00094 (22 families) → **supported**.
- C2 (rerun #265/#269) **not triggered**: the pre-registered rule runs the intervention only
  when a positive association exists; with a null association the rerun would not test the
  registered claim. The causal export-geometry design is written in `VERDICT.md` §"Proposed
  causal test" and must be registered as a separate issue if executed.
- All 42 arms were re-evaluated under one 57,439-row target snapshot and one graph asset;
  the 12 immutable-run records are hash-verified, the 30 arm-dir records are marked.
- Every number cited here is reproducible from `analysis/RELATION.json` and the per-arm
  `assets/geometry/*.npz|*.json`; receipts are under `receipts/`.
