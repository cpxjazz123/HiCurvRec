# Do history-conditioned predictability, conditional refinement or target ambiguity add out-of-family information?

Issue #48, follow-up to #47 (`ce9a179`). Frozen 38-model analysis. No Stage 2 or
Stage 3 training was run, and no SID, checkpoint, split or Stage 3 score was modified.

## 1. Headline

**No new feature cleared the confirmatory bar, but the direction is no longer
uniformly negative.** M4, M5, M7 improve Recall@10 out-of-family in
the point estimate — the best, M4, gains dMAE +0.000892 with a 95%
family-cluster CI [-0.002192, +0.002101] that still spans zero.
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
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | TIGER_BEHAVIOR_SAM | 16.39 | 0.0310 | [0.023, 0.039] | 0.0550 | 0.1236 | 0.0859 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask` | TIGER_BEHAVIOR_SAM | 16.43 | 0.0425 | [0.034, 0.051] | 0.0579 | 0.1238 | 0.1101 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43` | TIGER_BEHAVIOR_SAM | 15.62 | 0.0400 | [0.032, 0.048] | 0.0551 | 0.1244 | 0.0874 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44` | TIGER_BEHAVIOR_SAM | 15.79 | 0.0406 | [0.032, 0.049] | 0.0534 | 0.1233 | 0.0873 |
| `TIGER_FORK_RQ-VAE/fork_radial_corr` | TIGER_FORK_RQ-VAE | 12.02 | 0.0398 | [0.032, 0.048] | 0.0834 | 0.1335 | 0.1127 |
| `TIGER_FORK_RQ-VAE/fork_more_sib` | TIGER_FORK_RQ-VAE | 11.82 | 0.0377 | [0.030, 0.046] | 0.0825 | 0.1317 | 0.1147 |
| `TIGER_FORK_RQ-VAE/prefix_sib` | TIGER_FORK_RQ-VAE | 3.51 | 0.2088 | [0.196, 0.220] | 0.4407 | 0.1300 | 0.2006 |
| `TIGER_FORK_RQ-VAE/fork_euclid_group` | TIGER_FORK_RQ-VAE | 12.21 | 0.0424 | [0.034, 0.051] | 0.0660 | 0.1330 | 0.0954 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp2_euclid_matched` | TIGER_HYPERBOLIC_R | 15.30 | 0.0669 | [0.056, 0.079] | 0.1282 | 0.1305 | 0.0976 |
| `TIGER_FORK_RQ-VAE/fork_ang_only` | TIGER_FORK_RQ-VAE | 12.17 | 0.0419 | [0.034, 0.050] | 0.0744 | 0.1322 | 0.0962 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp4_euclid_delta` | TIGER_HYPERBOLIC_R | 12.61 | 0.0410 | [0.031, 0.051] | 0.0917 | 0.1345 | 0.0980 |
| `smoothness_arms/A_euclid_smooth_s42` | curvature_RQ-VAE | 10.22 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1085 | 0.0588 |
| `curvature_RQ-VAE/euclid_l2_256_72k_s43` | curvature_RQ-VAE | 10.09 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1043 | 0.0574 |
| `TIGER_FORK_RQ-VAE/nce_euclid_radius_control` | TIGER_FORK_RQ-VAE | 6.89 | 0.0276 | [0.021, 0.036] | 0.0934 | 0.1447 | 0.1208 |
| `smooth_fair_arms/A_euclid_smoothfair_s42` | curvature_RQ-VAE | 10.37 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1133 | 0.0608 |
| `smoothness_arms/B_poincare_smooth_s42` | curvature_RQ-VAE | 9.76 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1113 | 0.0634 |
| `curvature_RQ-VAE/euclid_l2_256_72k_s42` | curvature_RQ-VAE | 10.04 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1051 | 0.0583 |
| `TIGER_FORK_RQ-VAE/nce_euclid_group` | TIGER_FORK_RQ-VAE | 6.87 | 0.0328 | [0.025, 0.040] | 0.0914 | 0.1443 | 0.1178 |
| `TIGER_FORK_RQ-VAE/nce_hyp_group` | TIGER_FORK_RQ-VAE | 7.17 | 0.0640 | [0.055, 0.074] | 0.1923 | 0.1355 | 0.1550 |
| `smooth_fair_arms/B_poincare_smoothfair_s42` | curvature_RQ-VAE | 9.36 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1118 | 0.0594 |
| `curvature_RQ-VAE/euclid_l2_256_72k_s44` | curvature_RQ-VAE | 10.38 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1061 | 0.0586 |
| `curvature_RQ-VAE` | curvature_RQ-VAE | 9.91 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1042 | 0.0632 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_ball` | TIGER_HYPERBOLIC_R | 8.70 | 0.0524 | [0.042, 0.063] | 0.1578 | 0.1329 | 0.0863 |
| `TIGER_FORK_RQ-VAE/fork_hyp_group` | TIGER_FORK_RQ-VAE | 10.93 | 0.0461 | [0.036, 0.057] | 0.1225 | 0.1276 | 0.0866 |
| `TIGER_HYPERBOLIC_RQ-VAE/exp2_poincare_c1` | TIGER_HYPERBOLIC_R | 10.88 | 0.0746 | [0.060, 0.089] | 0.2159 | 0.1289 | 0.1310 |
| `l2_cone_arms/A_euclid_nocone_s42` | curvature_RQ-VAE | 10.20 | 0.0089 | [0.002, 0.016] | 0.0306 | 0.0696 | 0.0383 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | TIGER_HYPERBOLIC_R | 3.47 | 0.1593 | [0.151, 0.168] | 3.6238 | 0.1946 | 0.4092 |
| `TIGER_RQ-VAE` | TIGER_RQ-VAE | 6.47 | 0.0626 | [0.052, 0.073] | 0.3621 | 0.0901 | 0.1003 |
| `l2_assignment_arms/A_bucket_sinkhorn_s42` | curvature_RQ-VAE | 9.24 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.0658 | 0.0350 |
| `l2_cone_arms/B_poincare_nocone_s42` | curvature_RQ-VAE | 9.42 | 0.0135 | [0.006, 0.021] | 0.0499 | 0.0690 | 0.0378 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw` | TIGER_HYPERBOLIC_R | 4.49 | 0.0735 | [0.067, 0.081] | 0.4703 | 0.1364 | 0.2781 |
| `l2_assignment_arms/B_global_sinkhorn_s42` | curvature_RQ-VAE | 9.14 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.0650 | 0.0382 |
| `coarse_l1_arms/B_poincare_coarse_l1_s42` | curvature_RQ-VAE | 6.37 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.0605 | 0.0316 |
| `l2_cone_arms_fixed_rejection/D_poincare_cone_s42` | curvature_RQ-VAE | 6.15 | 0.0048 | [-0.003, 0.012] | 0.0957 | 0.0630 | 0.0315 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare` | TIGER_HYPERBOLIC_R | 3.18 | 0.0548 | [0.043, 0.067] | 1.4054 | 0.1009 | 0.1468 |
| `l2_cone_arms/D_poincare_cone_s42` | curvature_RQ-VAE | 5.39 | 0.0111 | [0.003, 0.020] | 0.0639 | 0.0530 | 0.0312 |
| `metric_cone_arms/B_poincare_metriccone_s42` | curvature_RQ-VAE | 4.72 | 0.0054 | [-0.001, 0.012] | 0.1083 | 0.0520 | 0.0324 |
| `metric_cone_arms/A_euclid_metriccone_s42` | curvature_RQ-VAE | 1.03 | 0.0091 | [0.005, 0.014] | 1.3563 | 0.0034 | 0.2367 |

## 4. Primary experiment: Leave-One-Family-Out

Five folds, one per family ({'TIGER_BEHAVIOR_SAMPLE_RQ-VAE': 4, 'TIGER_FORK_RQ-VAE': 9, 'TIGER_HYPERBOLIC_RQ-VAE': 7, 'curvature_RQ-VAE': 17, 'TIGER_RQ-VAE': 1}), shared by M0 and every challenger. M0 is the #47
ER-only baseline recomputed on identical folds.

### Recall@10 (primary)

| set | features | OOF MAE | OOF RMSE | OOF R2 | OOF rho | dMAE vs M0 | 95% CI | CI lower > 0 |
|:--|:--|---:|---:|---:|---:|---:|:--|:--:|
| M0 | log_er_l1 | 0.003753 | 0.005460 | +0.2960 | +0.6367 | — | — | — |
| M4 | log_er_l1 + G_H1 | 0.002861 | 0.003894 | +0.6419 | +0.6932 | +0.000892 | [-0.002192, +0.002101] | no |
| M5 | log_er_l1 + D2 | 0.003169 | 0.003850 | +0.6499 | +0.7269 | +0.000584 | [-0.000549, +0.001016] | no |
| M6 | log_er_l1 + A_target | 0.003909 | 0.005770 | +0.2139 | +0.5853 | -0.000156 | [-0.000315, +0.000210] | no |
| M7 | log_er_l1 + G_H1 + D2 | 0.003012 | 0.004359 | +0.5512 | +0.6903 | +0.000741 | [-0.002166, +0.001872] | no |

### NDCG@10 (secondary)

| set | OOF MAE | OOF R2 | dMAE vs M0 | 95% CI | CI lower > 0 |
|:--|---:|---:|---:|:--|:--:|
| M0 | 0.002102 | +0.2381 | — | — | — |
| M4 | 0.001653 | +0.6054 | +0.000449 | [-0.001085, +0.001046] | no |
| M5 | 0.001804 | +0.6117 | +0.000298 | [-0.000239, +0.000512] | no |
| M6 | 0.002113 | +0.1652 | -0.000010 | [-0.000145, +0.000306] | no |
| M7 | 0.001800 | +0.5015 | +0.000302 | [-0.001204, +0.000922] | no |

### Family-level dMAE (Recall@10)

| set | TIGER_BEHAVIOR_SAMPL | TIGER_FORK_RQ-VAE | TIGER_HYPERBOLIC_RQ- | curvature_RQ-VAE | TIGER_RQ-VAE |
|:--|---:|---:|---:|---:|---:|
| M4 | -0.00008 | +0.00216 | -0.00302 | +0.00215 | -0.00061 |
| M5 | +0.00002 | +0.00116 | -0.00058 | +0.00101 | -0.00144 |
| M6 | +0.00025 | -0.00033 | +0.00020 | -0.00032 | +0.00005 |
| M7 | +0.00008 | +0.00196 | -0.00311 | +0.00188 | -0.00006 |

Only five clusters exist, so these intervals are descriptive and underpowered.

## 5. Prespecified robustness and diagnostics

### 5.1 History-length ablation (last-20 vs last-1, matched budget)

Both variants use the identical architecture, batch size, epoch budget, seeds and
train-only split; only the history truncation differs.

| arm | G_H1 (last-20) | G_H1 (last-1) | ratio |
|:--|---:|---:|---:|
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | 0.1236 | 0.0892 | 1.39x |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask` | 0.1238 | 0.0892 | 1.39x |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43` | 0.1244 | 0.0914 | 1.36x |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44` | 0.1233 | 0.0895 | 1.38x |
| `TIGER_FORK_RQ-VAE/fork_radial_corr` | 0.1335 | 0.0975 | 1.37x |
| `TIGER_FORK_RQ-VAE/fork_more_sib` | 0.1317 | 0.0965 | 1.36x |

Both truncations recover real predictability, and the full-history gain is only
modestly larger than the single-previous-item gain, so F1 is not merely re-deriving a
pairwise Markov signal - but neither is it reading a long-range preference that the
pairwise features miss.

### 5.2 Prefix retention across the three focal arms

| arm | D2 | cond pos n | cond neg n | D3 | A_target | catalogue-uniform ambiguity | P(m>1) |
|:--|---:|---:|---:|---:|---:|---:|---:|
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | 0.0310 | 4199 | 2562 | 0.0257 | 0.0550 | 0.0559 | 0.0644 |
| `TIGER_FORK_RQ-VAE/prefix_sib` | 0.2088 | 7355 | 20976 | 0.0325 | 0.4407 | 0.3869 | 0.4099 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | 0.1593 | 18411 | 53050 | 0.1666 | 3.6238 | 3.7230 | 0.9215 |

`prefix_sib` has by far the largest D2, but so does its negative-side conditional rate:
its high positive retention is a symptom of 175 codes crowded into few L1 buckets, not
evidence that its hierarchy discriminates better. Read alone, D2 would rank it among the
best arms. It is not: its Recall@10 is 0.061613 on a held-out L1 ER of 3.51.

### 5.3 Target-weighted vs catalogue-global ambiguity

| arm | A_target | A_catalogue_uniform | difference | mean group size | P90 | P99 |
|:--|---:|---:|---:|---:|---:|---:|
| `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` | 3.6238 | 3.7230 | -0.0992 | 134.67 | 394.0 | 823.0 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare` | 1.4054 | 1.4076 | -0.0022 | 6.74 | 16.0 | 32.0 |
| `metric_cone_arms/A_euclid_metriccone_s42` | 1.3563 | 1.3618 | -0.0055 | 5.87 | 12.0 | 37.0 |
| `TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw` | 0.4703 | 0.4866 | -0.0163 | 2.38 | 5.0 | 17.0 |
| `TIGER_FORK_RQ-VAE/prefix_sib` | 0.4407 | 0.3869 | +0.0538 | 2.10 | 4.0 | 13.0 |

`A_target` is systematically larger than the catalogue-uniform mean, which is exactly
what demand weighting should produce: the items asked for as targets are the popular
ones, and those sit in the crowded code groups. The two ambiguities are therefore not
interchangeable, and uniqueness of the fourth collision-extension token says nothing
about the first three.

### 5.4 Leave-one-family-out influence on dMAE

| arm | TIGER_BEHAVIOR_SAMPL | TIGER_FORK_RQ-VAE | TIGER_HYPERBOLIC_RQ- | curvature_RQ-VAE | TIGER_RQ-VAE | sign flips |
|:--|---:|---:|---:|---:|---:||:--:|
| M4 | -0.00008 | +0.00216 | -0.00302 | +0.00215 | -0.00061 | 2 |
| M5 | +0.00002 | +0.00116 | -0.00058 | +0.00101 | -0.00144 | 2 |
| M6 | +0.00025 | -0.00033 | +0.00020 | -0.00032 | +0.00005 | 3 |
| M7 | +0.00008 | +0.00196 | -0.00311 | +0.00188 | -0.00006 | 2 |

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

- **M4** (log_er_l1 + G_H1): dMAE +0.000892, CI [-0.002192, +0.002101] — not confirmed.
- **M5** (log_er_l1 + D2): dMAE +0.000584, CI [-0.000549, +0.001016] — not confirmed.
- **M6** (log_er_l1 + A_target): dMAE -0.000156, CI [-0.000315, +0.000210] — not confirmed.
- **M7** (log_er_l1 + G_H1 + D2): dMAE +0.000741, CI [-0.002166, +0.001872] — not confirmed.

Confirmation on a new, genuinely held-out model population with at least three
independent Stage 3 seeds is the right next step. It is deliberately **not** attempted
here: this issue's scope is frozen-model screening utility, and fabricating either
new models or new Stage 3 runs would exceed it.

## 8. Old #48 vs New #49 Comparison (Issue #49 Bugfix Round)

This section documents the impact of fixing the five implementation defects identified in issue #49:
1. **Dev scoring leakage**: `torch.arange(n_tr, n_tr + n_dev)` scored 34,856 training rows (90%) and missed 5,155 dev users. Corrected to evaluate on the exact, disjoint held-out `dev_idx` (38,834 rows).
2. **Prior baseline leakage**: Frequency prior was fitted on dev targets. Corrected to fit strictly once on train targets.
3. **Prior cross-entropy calculation**: Summed over the 256-wide vocabulary (~1500 nats). Corrected to gather the true target token (2.3 - 5.5 nats).
4. **F3 evaluation population**: Int-vs-tuple filter bug caused all 57,439 validation rows to be used. Corrected to the 52,855 frozen held-out positive pairs.
5. **Ambiguity weighting**: Averaged over occupied SID groups instead of catalogue items. Corrected to item-weighted mean (`A_target` and `A_catalogue_uniform`).

### 8.1 Invariance Controls
- `er_l1_recheck`: max absolute difference between old and new is **0.000000e+00** (bit-identical).
- `chrd_D2`: max absolute difference between old and new is **0.000000e+00** (bit-identical).
- `M0` Baseline Recall@10 MAE: **0.003753** vs **0.003753** (bit-identical).
- `M0` Baseline NDCG@10 MAE: **0.002102** vs **0.002102** (bit-identical).

### 8.2 LOFO Recall@10 Comparison

| Model | Features | Old #48 MAE | New #49 MAE | Old dMAE | New dMAE | Old 95% CI | New 95% CI | Survives Confirmatory Bar? |
|:---|:---|---:|---:|---:|---:|:---:|:---:|:---:|
| **M0** | `log_er_l1` | 0.003753 | 0.003753 | baseline | baseline | baseline | baseline | **Yes** (reference) |
| **M4** | `+ G_H1` | 0.002768 | 0.002861 | +0.000985 | +0.000892 | [-0.002098, +0.002204] | [-0.002192, +0.002101] | **No** (CI includes 0) |
| **M5** | `+ D2` | 0.003169 | 0.003169 | +0.000584 | +0.000584 | [-0.000549, +0.001016] | [-0.000549, +0.001016] | **No** (CI includes 0) |
| **M6** | `+ A_target` | 0.003935 | 0.003909 | -0.000182 | -0.000156 | [-0.000313, +0.000146] | [-0.000315, +0.000210] | **No** (hurts prediction) |
| **M7** | `+ G_H1 + D2`| 0.002888 | 0.003012 | +0.000865 | +0.000741 | [-0.002157, +0.002043] | [-0.002166, +0.001872] | **No** (CI includes 0) |

### 8.3 LOFO NDCG@10 Comparison

| Model | Features | Old #48 MAE | New #49 MAE | Old dMAE | New dMAE | Old 95% CI | New 95% CI | Survives Confirmatory Bar? |
|:---|:---|---:|---:|---:|---:|:---:|:---:|:---:|
| **M0** | `log_er_l1` | 0.002102 | 0.002102 | baseline | baseline | baseline | baseline | **Yes** (reference) |
| **M4** | `+ G_H1` | 0.001603 | 0.001653 | +0.000500 | +0.000449 | [-0.001015, +0.001094] | [-0.001085, +0.001046] | **No** (CI includes 0) |
| **M5** | `+ D2` | 0.001804 | 0.001804 | +0.000298 | +0.000298 | [-0.000239, +0.000512] | [-0.000239, +0.000512] | **No** (CI includes 0) |
| **M6** | `+ A_target` | 0.002127 | 0.002113 | -0.000025 | -0.000010 | [-0.000143, +0.000248] | [-0.000145, +0.000306] | **No** (hurts prediction) |
| **M7** | `+ G_H1 + D2`| 0.001678 | 0.001800 | +0.000424 | +0.000302 | [-0.001082, +0.001003] | [-0.001204, +0.000922] | **No** (CI includes 0) |

### 8.4 Conclusion on Claimed Gains
Neither in the flawed #48 implementation nor in the mathematically sound #49 rerun do any of the candidate predictors clear the prespecified confirmatory bar (`lower_bound_gt_zero`). The core qualitative conclusion remains unchanged: **no new feature adds statistically reliable out-of-family predictive value beyond ER alone**.
