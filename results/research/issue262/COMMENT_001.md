# Issue #262 comment — completed analysis

Commit: `7d31e04aace88062f5309fcb6c3fb22acba65e17` (main; remote SHA matched).
No Stage2/Stage3/TIGER training, no optimizer update, no SID rewrite and no
change to public infrastructure or frozen assets. Evidence is under
`results/research/issue262/`, with the frozen contract in `CONTRACT.md` and the
acceptance answers in `VERDICT.md`.

## Corrections to #261

- Stage3 validation is not absent for R01–R04. The identity-bound records at
  `results/research/issue256/rounds/r01..r04/recommendation_validation/` have
  `training=false`, `test_access=false`, the final checkpoint/SID/test-final
  identity, and independently reduced `rank*.jsonl` shards. Their reported
  validation R@5/R@10/NDCG values reproduce to ≤1e-10. 100/101/106/107 remain
  NA; test was never substituted.
- The #261 `+0.958` is a **cross-arm Pearson of a within-arm distance/label
  Spearman**, using the visible-graph `jaccard_in` field. On the strict pairs
  that truly share an in-neighbour in the bank's source graph, the true directed
  Jaccard mean is 0.2035, while that visible field is 0.0041. Recomputing the
  same cross-arm association with the directed label gives **−0.8682**. The
  original headline is therefore not a shared-in fidelity measurement; this is
  a structure-test association, not a recommendation-test effect.

## H1 / H2 / H3

- **H1 — PARTIALLY_SUPPORTED.** Target common-in rate 0.4033 vs
  degree/popularity-matched negative 0.2695; paired difference +0.1338
  [0.1300, 0.1378] (2,000 bootstrap draws). Relation-matched negative:
  +0.1070; both sides without direct/reverse/two-hop: +0.0611 (n=30,771). The
  raw and relation-matched target advantage is real, but the target also has
  more direct/reverse/two-hop and prefix structure, and the paired logistic
  common-in coefficient is not positive once those controls are present. This
  is not an independent causal topology effect.
- **H2 — SUPPORTED.** The historical +0.958 is a visible-graph artefact at the
  cross-arm layer; the true directed label reverses the sign. The association
  is not evidence that shared-in fidelity improves recommendation.
- **H3 — SUPPORTED_AS_DESCRIPTIVE.** Every arm was re-evaluated on all
  28,391,948 valid frozen triples. Lower curvature MAE is associated with higher
  test R@10 at n=8 (Pearson −0.724), but the five/expert/role groups differ on
  architecture, capacity and multiple metrics; no causal direction is inferred.

## Completed measurements

- Full predicted-vs-target curvature Pearson/Spearman/RMSE/MAE and per-sign
  precision/recall/F1 for all eight arms, with role and directed-total-degree
  strata (`CURVATURE_FULL_*.json`; exact coverage check).
- Actual-codebook per-level top1/top2 margin, selected rank, normalized gap,
  residual, commitment/codebook values and effective ranks
  (`QUANT_GRAD_SWITCH_*.json`). MAE reproduces #261's micro MAE to rounding.
- One common fixed-batch endpoint forward/autograd per arm: raw reconstruction,
  curvature and combined losses, per-module gradient norms/directions. Every
  state-dict checksum is unchanged; optimizer steps = 0; parameters unchanged.
- Read-only switches for all eight arms. No-intervention replay is exact for
  100/101/103–107; 102 changes 0.7402% on both CPU and an L40S replay, so its
  switch fractions are reported relative to the frozen SID, not as a
  zero-baseline causal effect (`GPU_SWITCH_102.json`).
- Historical per-module gradient trajectories and an R2 source tar copy remain
  NA with concrete evidence; no values were imputed.

## Verification

- `python3 -m pytest -q results/research/issue262/tests` → 8 passed.
- `INFRA_SNAPSHOT_START/END` and `INFRA_DIFF.json` → PASS, no changed/missing
  public path or frozen asset.
- `TRAINING.json` and all endpoint receipts → zero training runs, zero
  optimizer steps, zero SID/checkpoint writes.

The issue is ready to close as COMPLETED.
