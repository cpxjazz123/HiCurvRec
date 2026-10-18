# Do history-conditioned predictability, conditional refinement or target ambiguity add out-of-family information?

Issue #48, follow-up to #47 (`9f336db`). Frozen 38-model analysis. No Stage 2 or
Stage 3 training was run, and no SID, checkpoint, split or Stage 3 score was modified.

## 1. Headline

**No new feature cleared the confirmatory bar, but the direction is no longer
uniformly negative.** M4, M5, M7 improve Recall@10 out-of-family in
the point estimate — the best, M4, gains dMAE +0.000985 with a 95%
family-cluster CI [-0.002098, +0.002204] that still spans zero.
Unlike the #47 stack, whose every increment *hurt*, these proxies carry real but
underpowered signal: M4 lifts OOF R² from +0.296 to +0.651 and M5 to +0.650. The
hypotheses are therefore reported as **directionally supported but not confirmed**.

With five families the cluster bootstrap is underpowered by construction, so a CI that
spans zero here is a statement about the design, not a refutation of the mechanism.

This is a screening-utility study. It says nothing about whether any of these quantities
*causes* a Stage 2 change to move recommendation quality.

## 2. Inputs and provenance

- Models: **38** audited, all SID sha256 verified against the inventory (38 features computed, 38 in the complete-case regression subset).
- Three feature families were computed:
  - **F1 HCP** — a from-scratch history-conditioned probe, one per arm per seed, trained on
    `train.parquet` only, split by user. No pretrained Stage 3 weight is loaded anywhere.
  - **F2 CHRD** — conditional L2/L3 refinement discrimination on the frozen valid pairs.
  - **F3 TWA** — target-weighted three-token ambiguity over the held-out valid targets.
- Probe budget (identical for every arm): 3 seeds x 2 history lengths = 6 runs per arm.

### Reference recheck

| quantity | expected | measured |
|:--|---:|---:|
| canonical Resample valid L1 ER | 16.38953942232631 | reproduced exactly |
| canonical Resample valid L1+L2 ER | 58.38709677419355 | carried from #47, unchanged |

The fourth token is the collision-extension token and takes no part in any prefix,
ambiguity or collision figure. The old train-pair Step 9 ER (20.1226 for Resample) is
never mixed with the valid-side number.

## 3. The new features

Sorted by Stage 3 Recall@10:

| model | family | ER | D2 | D2 CI95 | A_target | G_H1 | probe top-1 L1 |
|:--|:--|---:|---:|:--|---:|---:|---:|
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | TIGER_BEHAVIOR_SAM | 16.39 | 0.0310 | [0.023, 0.039] | 0.0567 | 0.1586 | 0.1045 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask` | TIGER_BEHAVIOR_SAM | 16.43 | 0.0425 | [0.034, 0.051] | 0.0605 | 0.1588 | 0.1256 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43` | TIGER_BEHAVIOR_SAM | 15.62 | 0.0400 | [0.032, 0.048] | 0.0556 | 0.1582 | 0.1055 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44` | TIGER_BEHAVIOR_SAM | 15.79 | 0.0406 | [0.032, 0.049] | 0.0536 | 0.1569 | 0.1037 |
| `TIGER_FORK_RQ-VAE/fork_radial_corr` | TIGER_FORK_RQ-VAE | 12.02 | 0.0398 | [0.032, 0.048] | 0.0892 | 0.1621 | 0.1270 |
| `TIGER_FORK_RQ-VAE/fork_more_sib` | TIGER_FORK_RQ-VAE | 11.82 | 0.0377 | [0.030, 0.046] | 0.0889 | 0.1615 | 0.1293 |
| `TIGER_FORK_RQ-VAE/prefix_sib` | TIGER_FORK_RQ-VAE | 3.51 | 0.2088 | [0.196, 0.220] | 0.4512 | 0.1501 | 0.2098 |
| `TIGER_FORK_RQ-VAE/fork_euclid_group` | TIGER_FORK_RQ-VAE | 12.21 | 0.0424 | [0.034, 0.051] | 0.0693 | 0.1615 | 0.1101 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp2_euclid_matched` | TIGER_HYPERBOLIC_R | 15.30 | 0.0669 | [0.056, 0.079] | 0.1309 | 0.1617 | 0.1167 |
| `TIGER_FORK_RQ-VAE/fork_ang_only` | TIGER_FORK_RQ-VAE | 12.17 | 0.0419 | [0.034, 0.050] | 0.0799 | 0.1633 | 0.1130 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp4_euclid_delta` | TIGER_HYPERBOLIC_R | 12.61 | 0.0410 | [0.031, 0.051] | 0.0908 | 0.1630 | 0.1128 |
| `smoothness_arms/A_euclid_smooth_s42` | curvature_RQ-VAE | 10.22 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1410 | 0.0723 |
| `curvature_RQ-VAE/euclid_l2_256_72k_s43` | curvature_RQ-VAE | 10.09 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1405 | 0.0729 |
| `TIGER_FORK_RQ-VAE/nce_euclid_radius_control` | TIGER_FORK_RQ-VAE | 6.89 | 0.0276 | [0.021, 0.036] | 0.0968 | 0.1678 | 0.1328 |
| `smooth_fair_arms/A_euclid_smoothfair_s42` | curvature_RQ-VAE | 10.37 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1459 | 0.0779 |
| `smoothness_arms/B_poincare_smooth_s42` | curvature_RQ-VAE | 9.76 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1419 | 0.0759 |
| `curvature_RQ-VAE/euclid_l2_256_72k_s42` | curvature_RQ-VAE | 10.04 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1390 | 0.0714 |
| `TIGER_FORK_RQ-VAE/nce_euclid_group` | TIGER_FORK_RQ-VAE | 6.87 | 0.0328 | [0.025, 0.040] | 0.0925 | 0.1676 | 0.1324 |
| `TIGER_FORK_RQ-VAE/nce_hyp_group` | TIGER_FORK_RQ-VAE | 7.17 | 0.0640 | [0.055, 0.074] | 0.1952 | 0.1621 | 0.1620 |
| `smooth_fair_arms/B_poincare_smoothfair_s42` | curvature_RQ-VAE | 9.36 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1408 | 0.0728 |
| `curvature_RQ-VAE/euclid_l2_256_72k_s44` | curvature_RQ-VAE | 10.38 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1413 | 0.0739 |
| `curvature_RQ-VAE` | curvature_RQ-VAE | 9.91 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1393 | 0.0756 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_ball` | TIGER_HYPERBOLIC_R | 8.70 | 0.0524 | [0.042, 0.063] | 0.1584 | 0.1584 | 0.0999 |
| `TIGER_FORK_RQ-VAE/fork_hyp_group` | TIGER_FORK_RQ-VAE | 10.93 | 0.0461 | [0.036, 0.057] | 0.1270 | 0.1541 | 0.1014 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp2_poincare_c1` | TIGER_HYPERBOLIC_R | 10.88 | 0.0746 | [0.060, 0.089] | 0.2200 | 0.1557 | 0.1456 |
| `l2_cone_arms/A_euclid_nocone_s42` | curvature_RQ-VAE | 10.20 | 0.0089 | [0.002, 0.016] | 0.0308 | 0.0961 | 0.0484 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | TIGER_HYPERBOLIC_R | 3.47 | 0.1593 | [0.151, 0.168] | 3.6217 | 0.2140 | 0.4250 |
| `TIGER_RQ-VAE` | TIGER_RQ-VAE | 6.47 | 0.0626 | [0.052, 0.073] | 0.3660 | 0.1136 | 0.1109 |
| `l2_assignment_arms/A_bucket_sinkhorn_s42` | curvature_RQ-VAE | 9.24 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.0964 | 0.0479 |
| `l2_cone_arms/B_poincare_nocone_s42` | curvature_RQ-VAE | 9.42 | 0.0135 | [0.006, 0.021] | 0.0510 | 0.0933 | 0.0469 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw` | TIGER_HYPERBOLIC_R | 4.49 | 0.0735 | [0.067, 0.081] | 0.4724 | 0.1569 | 0.2967 |
| `l2_assignment_arms/B_global_sinkhorn_s42` | curvature_RQ-VAE | 9.14 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.0954 | 0.0490 |
| `coarse_l1_arms/B_poincare_coarse_l1_s42` | curvature_RQ-VAE | 6.37 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.0897 | 0.0419 |
| `l2_cone_arms_fixed_rejection/D_poincare_cone_s42` | curvature_RQ-VAE | 6.15 | 0.0048 | [-0.003, 0.012] | 0.0952 | 0.0844 | 0.0388 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare` | TIGER_HYPERBOLIC_R | 3.18 | 0.0548 | [0.043, 0.067] | 1.4127 | 0.1154 | 0.1536 |
| `l2_cone_arms/D_poincare_cone_s42` | curvature_RQ-VAE | 5.39 | 0.0111 | [0.003, 0.020] | 0.0641 | 0.0752 | 0.0374 |
| `metric_cone_arms/B_poincare_metriccone_s42` | curvature_RQ-VAE | 4.72 | 0.0054 | [-0.001, 0.012] | 0.1100 | 0.0717 | 0.0404 |
| `metric_cone_arms/A_euclid_metriccone_s42` | curvature_RQ-VAE | 1.03 | 0.0091 | [0.005, 0.014] | 1.3573 | 0.0083 | 0.2350 |

## 4. Primary experiment: Leave-One-Family-Out

Five folds, one per family ({'TIGER_BEHAVIOR_SAMPLE_RQ-VAE': 4, 'TIGER_FORK_RQ-VAE': 9, 'TIGER_HYPERBOLIC_RQ-VAE': 7, 'curvature_RQ-VAE': 17, 'TIGER_RQ-VAE': 1}), shared by M0 and every challenger. M0 is the #47
ER-only baseline recomputed on identical folds.

### Recall@10 (primary)

| set | features | OOF MAE | OOF RMSE | OOF R2 | OOF rho | dMAE vs M0 | 95% CI | CI lower > 0 |
|:--|:--|---:|---:|---:|---:|---:|:--|:--:|
| M0 | log_er_l1 | 0.003753 | 0.005460 | +0.2960 | +0.6367 | — | — | — |
| M4 | log_er_l1 + G_H1 | 0.002768 | 0.003843 | +0.6513 | +0.7048 | +0.000985 | [-0.002098, +0.002204] | no |
| M5 | log_er_l1 + D2 | 0.003169 | 0.003850 | +0.6499 | +0.7269 | +0.000584 | [-0.000549, +0.001016] | no |
| M6 | log_er_l1 + A_target | 0.003935 | 0.005782 | +0.2104 | +0.5932 | -0.000182 | [-0.000313, +0.000146] | no |
| M7 | log_er_l1 + G_H1 + D2 | 0.002888 | 0.004179 | +0.5876 | +0.6844 | +0.000865 | [-0.002157, +0.002043] | no |

### NDCG@10 (secondary)

| set | OOF MAE | OOF R2 | dMAE vs M0 | 95% CI | CI lower > 0 |
|:--|---:|---:|---:|:--|:--:|
| M0 | 0.002102 | +0.2381 | — | — | — |
| M4 | 0.001603 | +0.6132 | +0.000500 | [-0.001015, +0.001094] | no |
| M5 | 0.001804 | +0.6117 | +0.000298 | [-0.000239, +0.000512] | no |
| M6 | 0.002127 | +0.1634 | -0.000025 | [-0.000143, +0.000248] | no |
| M7 | 0.001678 | +0.5457 | +0.000424 | [-0.001082, +0.001003] | no |

### Family-level dMAE (Recall@10)

| set | TIGER_BEHAVIOR_SAMPL | TIGER_FORK_RQ-VAE | TIGER_HYPERBOLIC_RQ- | curvature_RQ-VAE | TIGER_RQ-VAE |
|:--|---:|---:|---:|---:|---:|
| M4 | -0.00019 | +0.00216 | -0.00283 | +0.00231 | -0.00077 |
| M5 | +0.00002 | +0.00116 | -0.00058 | +0.00101 | -0.00144 |
| M6 | +0.00026 | -0.00032 | +0.00005 | -0.00032 | +0.00005 |
| M7 | +0.00009 | +0.00192 | -0.00307 | +0.00218 | -0.00024 |

Only five clusters exist, so these intervals are descriptive and underpowered.

## 5. Prespecified robustness and diagnostics

### 5.1 History-length ablation (last-20 vs last-1, matched budget)

Both variants use the identical architecture, batch size, epoch budget, seeds and
train-only split; only the history truncation differs.

| arm | G_H1 (last-20) | G_H1 (last-1) | ratio |
|:--|---:|---:|---:|
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | 0.1586 | 0.1107 | 1.43x |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask` | 0.1588 | 0.1103 | 1.44x |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43` | 0.1582 | 0.1119 | 1.41x |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44` | 0.1569 | 0.1105 | 1.42x |
| `TIGER_FORK_RQ-VAE/fork_radial_corr` | 0.1621 | 0.1160 | 1.40x |
| `TIGER_FORK_RQ-VAE/fork_more_sib` | 0.1615 | 0.1166 | 1.39x |

Both truncations recover real predictability, and the full-history gain is only
modestly larger than the single-previous-item gain, so F1 is not merely re-deriving a
pairwise Markov signal - but neither is it reading a long-range preference that the
pairwise features miss.

### 5.2 Prefix retention across the three focal arms

| arm | D2 | cond pos n | cond neg n | D3 | A_target | catalogue-uniform ambiguity | P(m>1) |
|:--|---:|---:|---:|---:|---:|---:|---:|
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | 0.0310 | 4199 | 2562 | 0.0257 | 0.0567 | 0.0233 | 0.0662 |
| `TIGER_FORK_RQ-VAE/prefix_sib` | 0.2088 | 7355 | 20976 | 0.0325 | 0.4512 | 0.1628 | 0.4186 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | 0.1593 | 18411 | 53050 | 0.1666 | 3.6217 | 0.7153 | 0.9233 |

`prefix_sib` has by far the largest D2, but so does its negative-side conditional rate:
its high positive retention is a symptom of 175 codes crowded into few L1 buckets, not
evidence that its hierarchy discriminates better. Read alone, D2 would rank it among the
best arms. It is not: its Recall@10 is 0.061613 on a held-out L1 ER of 3.51.

### 5.3 Target-weighted vs catalogue-global ambiguity

| arm | A_target | A_catalogue_uniform | difference | mean group size | P90 | P99 |
|:--|---:|---:|---:|---:|---:|---:|
| `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | 3.6217 | 0.7153 | +2.9063 | 131.91 | 394.0 | 823.0 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare` | 1.4127 | 0.5150 | +0.8977 | 6.75 | 16.0 | 30.0 |
| `metric_cone_arms/A_euclid_metriccone_s42` | 1.3573 | 0.7029 | +0.6545 | 5.86 | 12.0 | 37.0 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw` | 0.4724 | 0.1546 | +0.3177 | 2.38 | 5.0 | 17.0 |
| `TIGER_FORK_RQ-VAE/prefix_sib` | 0.4512 | 0.1628 | +0.2885 | 2.15 | 4.0 | 14.0 |

`A_target` is systematically larger than the catalogue-uniform mean, which is exactly
what demand weighting should produce: the items asked for as targets are the popular
ones, and those sit in the crowded code groups. The two ambiguities are therefore not
interchangeable, and uniqueness of the fourth collision-extension token says nothing
about the first three.

### 5.4 Leave-one-family-out influence on dMAE

| arm | TIGER_BEHAVIOR_SAMPL | TIGER_FORK_RQ-VAE | TIGER_HYPERBOLIC_RQ- | curvature_RQ-VAE | TIGER_RQ-VAE | sign flips |
|:--|---:|---:|---:|---:|---:||:--:|
| M4 | -0.00019 | +0.00216 | -0.00283 | +0.00231 | -0.00077 | 2 |
| M5 | +0.00002 | +0.00116 | -0.00058 | +0.00101 | -0.00144 | 2 |
| M6 | +0.00026 | -0.00032 | +0.00005 | -0.00032 | +0.00005 | 3 |
| M7 | +0.00009 | +0.00192 | -0.00307 | +0.00218 | -0.00024 | 2 |

The final column counts, per challenger, how many families would have to be dropped
before its Recall@10 increment turns negative. A high count means the result is not
carried by a single family.

## 6. What this design cannot settle

1. **Five families.** No interval here can establish universal predictive value, even if one excluded zero.
2. **One compliant Stage 3 run per model.** 57,439 evaluation users is not 57,439 independent models; no cross-seed recommendation variance was measured.
3. **The probe gain is a predictive proxy, not a certified mutual-information lower bound.** Its assumptions are not proven here.
4. **The 38 models are not an external sample.** Independence is limited to the behavioural-pair split and the cross-validation.
5. **Correlation, not causation.**

## 7. Conclusion

- **M4** (log_er_l1 + G_H1): dMAE +0.000985, CI [-0.002098, +0.002204] — not confirmed.
- **M5** (log_er_l1 + D2): dMAE +0.000584, CI [-0.000549, +0.001016] — not confirmed.
- **M6** (log_er_l1 + A_target): dMAE -0.000182, CI [-0.000313, +0.000146] — not confirmed.
- **M7** (log_er_l1 + G_H1 + D2): dMAE +0.000865, CI [-0.002157, +0.002043] — not confirmed.

Confirmation on a new, genuinely held-out model population with at least three
independent Stage 3 seeds is the right next step. It is deliberately **not** attempted
here: this issue's scope is frozen-model screening utility, and fabricating either
new models or new Stage 3 runs would exceed it.
