# Issue #262 — shared-in, curvature and quantisation diagnostics

Analysis only. No Stage2/Stage3/TIGER training, no optimizer update, no SID rewrite.
Eight completed R2 bundles 100–107; test is the only recommendation outcome.

## Identity and corrections

- Audit: 8 arms; #261 inventory identities rechecked.
- Stage3 validation found and independently reduced for 102, 103, 104, 105.
- #261's blanket validation-NA claim is corrected; 100/101/106/107 remain NA.
- The #261 `+0.958` row is a **cross-arm Pearson of a within-arm distance/label Spearman**, not a shared-in fidelity value.

## Primary shared-in result (validation, directed graph)

- Target common-in rate: 0.403280
- Matched-negative common-in rate: 0.269486
- Paired difference: 0.133794 [0.129981, 0.137816] (2,000 bootstrap draws)
- Relation-matched (direct/reverse/two-hop) difference: 0.107035
- Both sides without direct/reverse/two-hop: 0.061064 (n=30771)

The raw target advantage is positive in every degree stratum, but the target also contains direct/reverse/two-hop structure and a much higher SID prefix match. The paired logistic/prefix controls are retained in `MATRIX.json` and `SHARED_IN.json`; the raw difference is not a causal effect.

## Corrected held-out structure

| bank | pairs | visible J mean | true directed J mean | cross-arm test-R@10 Pearson (visible) | cross-arm test-R@10 Pearson (true) |
|---|---:|---:|---:|---:|---:|
| `struct_test_shared_in` | 2877 | 0.004107 | 0.203495 | 0.958448 | -0.868240 |
| `struct_test_shared_out` | 2877 | 0.004098 | 0.215756 | 0.840888 | -0.754803 |
| `struct_val_shared_in` | 2881 | 0.004388 | 0.204425 | 0.827706 | -0.901428 |
| `struct_val_shared_out` | 2876 | 0.004454 | 0.195731 | 0.807895 | -0.710102 |

The struct test/validation banks are structure populations, not the recommendation test population. The corrected directed label has a much larger mean Jaccard; its cross-arm association is not the #261 visible-feature association and is not evidence that shared-in fidelity improves recommendation.

## New full-table curvature and quantisation

| idx | test R@10 | validation R@10 | curve Pearson | curve Spearman | curve RMSE | curve MAE | sign acc | quant error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.049287 | NA | 0.1539 | 0.1318 | 2.7357 | 2.3444 | 0.3403 | 0.008098 |
| 101 | 0.046432 | NA | 0.1886 | 0.1678 | 2.7601 | 2.3601 | 0.3627 | 0.008693 |
| 102 | 0.052734 | 0.057644 | -0.0108 | -0.1022 | 2.2277 | 1.8815 | 0.3442 | 0.007227 |
| 103 | 0.052665 | 0.057313 | 0.0142 | -0.0627 | 2.2601 | 1.9111 | 0.3407 | 0.007673 |
| 104 | 0.049009 | 0.054301 | -0.0299 | -0.1170 | 2.2998 | 1.9387 | 0.3404 | 0.006471 |
| 105 | 0.051516 | 0.055903 | 0.0212 | -0.0473 | 2.2537 | 1.9059 | 0.3377 | 0.007616 |
| 106 | 0.050349 | NA | -0.0053 | -0.0870 | 2.2746 | 1.9246 | 0.3351 | 0.007134 |
| 107 | 0.049600 | NA | 0.0325 | -0.0506 | 2.2500 | 1.9002 | 0.3468 | 0.008575 |

The complete 28,391,948-triple pass per arm replaces the missing #261 predicted/target correlation. MAE for 102 matches the #261 micro MAE (1.882 vs 1.881) within rounding; the exact correlation is new.

## Hypothesis verdicts

- **H1 — PARTIALLY_SUPPORTED.** The raw target common-in advantage is
  +0.1338 [0.1300, 0.1378] over the degree/popularity-matched negative. It
  remains +0.1070 under a relation-matched negative and +0.0611 when both
  target and negative have no direct, reverse or two-hop relation (n=30,771).
  The target also has much higher direct/reverse/two-hop and SID prefix
  sharing; the paired logistic coefficient for common-in is not positive once
  those controls are present. The raw result is not an independent causal
  topology effect.
- **H2 — SUPPORTED.** The historical +0.958 is the cross-arm correlation of
  the **visible-graph** feature. On the same strict pairs, the directed label
  that actually generated the shared-in bank has mean Jaccard 0.2035 rather
  than 0.0041; its cross-arm Pearson is −0.8682. The original headline was
  therefore not a shared-in fidelity measurement. This is a structure-test
  association, not a recommendation-test effect.
- **H3 — SUPPORTED_AS_DESCRIPTIVE.** The new full-table curvature metrics are
  reported but remain architecture/capacity-confounded at n=8. Lower curvature
  MAE is associated with higher test R@10 (Pearson −0.724), but 100/101 carry
  much larger errors and the five/expert/role groups differ on several axes.
  No causal direction is inferred.

## Switch caveat

The no-intervention replay reproduces the frozen exported SID exactly on
100, 101, 103–107. On 102 it changes 0.7402% of items on CPU and on a read-only
L40S replay (identical value), so 102's switch fractions are relative to the
frozen SID rather than a zero-baseline replay. `GPU_SWITCH_102.json` records the
cross-device check. No switch is a retrained effect.

## Endpoint gradients, quantisation margins and switches

Every arm has a fixed-batch endpoint forward/autograd result, per-level top1/top2 margin, selected-code rank, residual, commitment/codebook value and read-only switch probes. See `QUANT_GRAD_SWITCH_*.json`; no optimizer step occurred.

## Correlations with test R@10 (n=8, exploratory)

| predictor | Pearson | Spearman | n |
|---|---:|---:|---:|
| `curve_pearson` | -0.6734 | -0.4286 | 8 |
| `curve_spearman` | -0.6696 | -0.3810 | 8 |
| `curve_rmse` | -0.7328 | -0.8095 | 8 |
| `curve_mae` | -0.7244 | -0.8095 | 8 |
| `validation_recall@10` | 0.9806 | 1.0000 | 4 |
| `validation_recall@5` | 0.9228 | 1.0000 | 4 |
| `quant_total_error_new` | -0.4165 | -0.3095 | 8 |
| `heldout_true_shared_in_spearman` | -0.8682 | -0.7143 | 8 |
| `heldout_visible_shared_in_spearman` | 0.9584 | 0.9048 | 8 |

Full predictor list, constants/NA, Fisher intervals, leave-one-out and exclude-100/107 sensitivity are in `CORRELATIONS_262.json` and `SENSITIVITY_262.json`. n=8 is exploratory; no causal or multiplicity-adjusted claim.

## Missing evidence

Historical per-module gradient trajectories remain NA because the archived training stream records aggregate/level gradients only. R2 source tar is not present in the retained local R2 download; the local source tree and candidate hash are checked instead. See `MISSING.json`.

## Artifacts

`AUDIT.json`, `MATRIX.json`, `CORRELATIONS_262.json`, `SENSITIVITY_262.json`,
`SHARED_IN.json`, `CURVATURE_FULL_<idx>.json`,
`QUANT_GRAD_SWITCH_<idx>.json`, `INFRA.md`, `INFRA_SNAPSHOT_START.json`, tests.
