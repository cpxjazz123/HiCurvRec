# Issue #270 relation report

Generated from `analysis/RELATION.json`; every cell is a re-read of an archived
artefact or a recomputation over the frozen assets. No test prediction is used for
selection; the recommendation linkage is explanatory error analysis.

## 1. Cohort and arm-level table

n = **42** arms; single-card 3, four-card 39.

| arm | family | protocol | ΔR@10 | ΔA.1 (TIGER − arm) | L2 ratio | J | J κ_t |
|---|---|---|---:|---:|---:|---:|---:|
| `TIGER_behavior_edge_contrastive_rq` | behavior_contrast | multi | +0.001045 | -3.572274 | 0.431 | 2.6625 | NA |
| `TIGER_chain_align_rq` | alignment | multi | +0.000905 | +2.129819 | 1.067 | 0.8181 | NA |
| `TIGER_Uniform_KNN_control` | control | multi | +0.000557 | +4.793954 | 1.113 | 0.1868 | NA |
| `TIGER_g4_closure_A` | graphlet_closure | multi | +0.000313 | +1.101081 | 1.201 | 0.8162 | NA |
| `TIGER_Uniform_KNN_RQ` | control | multi | +0.000087 | -124.964973 | 0.688 | 67.7028 | NA |
| `TIGER_RQ-VAE` | baseline | multi | +0.000000 | +0.000000 | 1.000 | 0.6916 | 1.1732 |
| `TIGER_kappa_sk_only_rq` | curvature_budget | multi | -0.000157 | +4.783854 | 0.967 | 0.1984 | NA |
| `TIGER_af3_curvature_drive_rq` | af3_curvature | multi | -0.000400 | +4.727666 | 0.997 | 0.2090 | NA |
| `TIGER_af3_graphgeo_rq` | af3_geometry | multi | -0.000627 | +4.778931 | 0.994 | 0.2007 | NA |
| `TIGER_cos_contrastive_rq` | curve_contrast | multi | -0.000644 | +0.139158 | 1.130 | 1.2918 | NA |
| `TIGER_graphlet_role_E` | graphlet_role | multi | -0.000870 | +0.873105 | 1.241 | 0.8945 | NA |
| `TIGER_g4_closure_B` | graphlet_closure | multi | -0.001045 | +1.489312 | 1.246 | 0.8006 | NA |
| `TIGER_graphlet_role_QR` | graphlet_role | multi | -0.001097 | -0.574103 | 1.262 | 0.9523 | NA |
| `TIGER_baseline_recal_rq` | control | multi | -0.001654 | +4.739391 | 0.986 | 0.1982 | NA |
| `TIGER_curve_contrastive_rq` | curve_contrast | multi | -0.001706 | -7.165674 | 1.075 | 4.2321 | NA |
| `TIGER_i234_A1` | i234 | multi | -0.002194 | +4.707507 | 1.320 | 0.1274 | NA |
| `TIGER_i234_A2` | i234 | multi | -0.002194 | +4.574997 | 1.332 | 0.1307 | NA |
| `TIGER_g237_sem_rq` | semantic | multi | -0.002420 | +5.415669 | 1.106 | 0.3042 | NA |
| `TIGER_module_budget_prefix_rq` | budget | multi | -0.002472 | +5.427752 | 1.161 | 0.3510 | NA |
| `TIGER_periodic_budget_prefix_rq` | budget | multi | -0.002542 | +5.398182 | 1.199 | 0.3096 | NA |
| `TIGER_euclid_local_curve_supervision_rq` | curvature_point | multi | -0.002611 | +5.063267 | 1.481 | 0.1832 | 0.1683 |
| `TIGER_degree_temp_rq` | control | multi | -0.002716 | +4.800267 | 0.996 | 0.2061 | NA |
| `TIGER_i234_A3` | i234 | multi | -0.002803 | +4.577162 | 1.327 | 0.1282 | NA |
| `TIGER_i234_A4` | i234 | multi | -0.003290 | +4.605577 | 1.330 | 0.1180 | NA |
| `TIGER_frozen_topology_gnn_rq` | gnn | multi | -0.003308 | +5.419944 | 1.206 | 0.3347 | NA |
| `TIGER_role_conditioned_l3_rq` | role_condition | multi | -0.003691 | +5.403333 | 1.191 | 0.3116 | NA |
| `TIGER_relative_curvature_ranking_rq` | curvature_rank | multi | -0.004422 | +5.105263 | 1.439 | 0.1755 | 0.1692 |
| `TIGER_role_conditioned_l2_rq` | role_condition | multi | -0.004857 | +5.384664 | 1.303 | 0.3167 | NA |
| `TIGER_topology_residual_l2_rq` | topology_residual | multi | -0.005606 | +5.409057 | 1.046 | 0.3052 | NA |
| `TIGER_gnn_attention_sectional_rq` | gnn | multi | -0.005919 | +4.964811 | 1.238 | 0.1177 | NA |
| `TIGER_role_conditioned_l1_rq` | role_condition | multi | -0.006198 | +5.370506 | 1.255 | 0.3169 | NA |
| `TIGER_graphlet_role_R` | graphlet_role | multi | -0.007034 | -98.021437 | 2.094 | 11.4779 | NA |
| `TIGER_graphlet_codebook_rq` | graphlet_codebook | single | -0.008113 | +5.088900 | 0.506 | 0.3351 | NA |
| `TIGER_three_expert_attention_rq` | attention_geometry | multi | -0.008775 | +4.949138 | 1.408 | 0.1230 | NA |
| `TIGER_graphlet_codebook_global_rq` | graphlet_codebook | single | -0.008827 | +5.187474 | 0.581 | 0.2996 | NA |
| `TIGER_calibrated_kappa_rq` | curvature_calibrated | single | -0.009976 | +5.311956 | 0.847 | 0.3259 | 0.2883 |
| `TIGER_graphlet_role_Q` | graphlet_role | multi | -0.010132 | -272.023201 | 1.975 | 38.3028 | NA |
| `TIGER_drq_D1` | other | multi | -0.010829 | -814.788519 | 1.476 | 85.5557 | NA |
| `TIGER_graphlet_role_T` | graphlet_role | multi | -0.011839 | -1044.112172 | 2.018 | 125.3545 | NA |
| `TIGER_i234_A5` | i234 | multi | -0.012117 | +4.319978 | 1.075 | 0.1323 | NA |
| `TIGER_g4_closure_C` | graphlet_closure | multi | -0.014346 | -104.799761 | 2.022 | 62.2640 | NA |
| `TIGER_g4_closure_D` | graphlet_closure | multi | -0.016975 | -7.094465 | 2.260 | 3.9314 | NA |

## 2. Headline arm-level association (P1)

| statistic | value |
|---|---:|
| Spearman(Δcurvature improvement, ΔR@10) | -0.068069 |
| Pearson | 0.410546 |
| Kendall | -0.060430 |
| permutation p (5,000) | 0.664467 |
| family-cluster bootstrap 95 % CI | [-0.511462, 0.351775] |
| single-card Spearman (n=3) | -1.000000 |
| four-card Spearman (n=39) | -0.016499 |

### Leave-one-family-out

| held out | n | Spearman |
|---|---:|---:|
| `af3_curvature` | 41 | -0.061414 |
| `af3_geometry` | 41 | -0.062285 |
| `alignment` | 41 | -0.047824 |
| `attention_geometry` | 41 | -0.056187 |
| `baseline` | 41 | -0.045037 |
| `behavior_contrast` | 41 | -0.040159 |
| `budget` | 40 | -0.068765 |
| `control` | 38 | 0.008535 |
| `curvature_budget` | 41 | -0.062633 |
| `curvature_calibrated` | 41 | -0.051832 |
| `curvature_point` | 41 | -0.056013 |
| `curvature_rank` | 41 | -0.055839 |
| `curve_contrast` | 40 | -0.033304 |
| `gnn` | 40 | -0.046625 |
| `graphlet_closure` | 38 | -0.171582 |
| `graphlet_codebook` | 40 | -0.035930 |
| `graphlet_role` | 37 | -0.262107 |
| `i234` | 37 | -0.038644 |
| `other` | 41 | -0.143473 |
| `role_condition` | 39 | -0.035326 |
| `semantic` | 41 | -0.069428 |
| `topology_residual` | 41 | -0.057232 |

### Conditional on SID preservation (P2)

| stratum | n | Spearman |
|---|---:|---:|
| l2_ratio<0.9 | 5 | -0.900000 |
| l2_ratio>=0.9 | 37 | 0.065082 |
| difference (compressed − preserved) | | [-0.848121, 1.437446] |

## 3. Item-level pairing (explanatory, test predictions)

Pooled items×arms: n=117629, Spearman(ΔA.1, Δhit) = **-0.026167**, item-cluster bootstrap 95 % CI [-0.042709, -0.010530], sign concordance 0.492246 (ties where either Δ is zero: 0.842114).

| arm | Spearman ΔA.1 vs Δhit | sign concordance |
|---|---:|---:|
| `TIGER_graphlet_role_Q` | -0.192352 | 0.826577 |
| `TIGER_graphlet_role_R` | -0.144417 | 0.743182 |
| `TIGER_drq_D1` | -0.114566 | 0.733333 |
| `TIGER_graphlet_role_T` | -0.085285 | 0.842697 |
| `TIGER_g4_closure_A` | -0.024341 | 0.525386 |
| `TIGER_gnn_attention_sectional_rq` | -0.022768 | 0.358108 |
| `TIGER_degree_temp_rq` | -0.019750 | 0.417808 |
| `TIGER_i234_A5` | -0.017631 | 0.334802 |
| `TIGER_g4_closure_D` | -0.016968 | 0.735849 |
| `TIGER_relative_curvature_ranking_rq` | -0.014759 | 0.361111 |
| `TIGER_calibrated_kappa_rq` | -0.012135 | 0.517301 |
| `TIGER_i234_A3` | -0.007871 | 0.409302 |
| `TIGER_g4_closure_B` | -0.006604 | 0.521253 |
| `TIGER_three_expert_attention_rq` | -0.003850 | 0.288416 |
| `TIGER_g237_sem_rq` | -0.000613 | 0.415755 |
| `TIGER_Uniform_KNN_RQ` | -0.000473 | 0.504255 |
| `TIGER_g4_closure_C` | 0.000389 | 0.819249 |
| `TIGER_frozen_topology_gnn_rq` | 0.001895 | 0.394799 |
| `TIGER_baseline_recal_rq` | 0.003085 | 0.475556 |
| `TIGER_i234_A1` | 0.004476 | 0.415525 |
| `TIGER_af3_curvature_drive_rq` | 0.004969 | 0.494407 |
| `TIGER_kappa_sk_only_rq` | 0.005200 | 0.470455 |
| `TIGER_i234_A2` | 0.005666 | 0.446121 |
| `TIGER_euclid_local_curve_supervision_rq` | 0.008329 | 0.402326 |
| `TIGER_graphlet_role_QR` | 0.008369 | 0.508584 |
| `TIGER_module_budget_prefix_rq` | 0.009123 | 0.412844 |
| `TIGER_role_conditioned_l1_rq` | 0.009283 | 0.331111 |
| `TIGER_role_conditioned_l2_rq` | 0.009748 | 0.347126 |
| `TIGER_graphlet_codebook_global_rq` | 0.010454 | 0.558730 |
| `TIGER_curve_contrastive_rq` | 0.015536 | 0.532110 |
| `TIGER_chain_align_rq` | 0.015943 | 0.471831 |
| `TIGER_graphlet_role_E` | 0.017043 | 0.500000 |
| `TIGER_cos_contrastive_rq` | 0.017702 | 0.487745 |
| `TIGER_role_conditioned_l3_rq` | 0.019245 | 0.375000 |
| `TIGER_Uniform_KNN_control` | 0.020299 | 0.453488 |
| `TIGER_af3_graphgeo_rq` | 0.020355 | 0.474057 |
| `TIGER_periodic_budget_prefix_rq` | 0.020511 | 0.427602 |
| `TIGER_i234_A4` | 0.020555 | 0.399083 |
| `TIGER_behavior_edge_contrastive_rq` | 0.021234 | 0.467811 |
| `TIGER_topology_residual_l2_rq` | 0.025266 | 0.354167 |
| `TIGER_graphlet_codebook_rq` | 0.034978 | 0.553120 |

### Four quadrants (pooled counts are per arm in RELATION.json)

| arm | curv improve + SID improve | curv improve + SID worsen | curv worsen + SID improve | curv worsen + SID worsen |
|---|---:|---:|---:|---:|
| `TIGER_Uniform_KNN_RQ` | — | +0.000000 (n=1) | -0.001400 (n=1689) | -0.000707 (n=1179) |
| `TIGER_Uniform_KNN_control` | -0.000050 (n=558) | -0.002660 (n=2113) | +0.020866 (n=26) | +0.001134 (n=172) |
| `TIGER_af3_curvature_drive_rq` | +0.008016 (n=537) | -0.002838 (n=2124) | +0.017013 (n=17) | +0.003848 (n=191) |
| `TIGER_af3_graphgeo_rq` | +0.002379 (n=622) | -0.002903 (n=2043) | -0.000086 (n=23) | +0.002631 (n=181) |
| `TIGER_baseline_recal_rq` | -0.000069 (n=605) | -0.002328 (n=2057) | +0.013514 (n=23) | +0.003893 (n=184) |
| `TIGER_behavior_edge_contrastive_rq` | -0.003040 (n=582) | -0.003686 (n=146) | +0.003544 (n=1735) | -0.009165 (n=406) |
| `TIGER_calibrated_kappa_rq` | +0.000207 (n=835) | -0.004617 (n=1908) | +0.017575 (n=11) | -0.005952 (n=115) |
| `TIGER_chain_align_rq` | -0.000709 (n=309) | -0.001791 (n=1497) | +0.002916 (n=159) | -0.001422 (n=904) |
| `TIGER_cos_contrastive_rq` | -0.000318 (n=218) | -0.004447 (n=1181) | +0.004092 (n=199) | -0.003331 (n=1271) |
| `TIGER_curve_contrastive_rq` | +0.001224 (n=99) | -0.000400 (n=389) | +0.000359 (n=397) | -0.003188 (n=1984) |
| `TIGER_degree_temp_rq` | +0.000942 (n=667) | -0.005212 (n=1989) | +0.003342 (n=25) | -0.003733 (n=188) |
| `TIGER_drq_D1` | — | +0.000000 (n=34) | +0.019832 (n=71) | -0.017060 (n=2764) |
| `TIGER_euclid_local_curve_supervision_rq` | +0.003275 (n=291) | -0.006052 (n=2451) | -0.015300 (n=9) | -0.009611 (n=118) |
| `TIGER_frozen_topology_gnn_rq` | +0.002428 (n=322) | -0.006460 (n=2440) | +0.000000 (n=6) | +0.004251 (n=101) |
| `TIGER_g237_sem_rq` | +0.001557 (n=824) | -0.005649 (n=1945) | -0.022222 (n=3) | +0.000946 (n=97) |
| `TIGER_g4_closure_A` | +0.003804 (n=254) | +0.000810 (n=1554) | +0.010082 (n=115) | -0.002919 (n=946) |
| `TIGER_g4_closure_B` | +0.003361 (n=262) | +0.000588 (n=1491) | +0.012481 (n=141) | -0.006224 (n=975) |
| `TIGER_g4_closure_C` | — | — | +0.015861 (n=65) | -0.018581 (n=2804) |
| `TIGER_g4_closure_D` | +0.000000 (n=11) | -0.015397 (n=634) | +0.000402 (n=40) | -0.022539 (n=2184) |
| `TIGER_gnn_attention_sectional_rq` | -0.005773 (n=328) | -0.008829 (n=2436) | +0.010406 (n=11) | -0.012405 (n=94) |
| `TIGER_graphlet_codebook_global_rq` | -0.000023 (n=1960) | -0.000943 (n=758) | +0.001071 (n=55) | -0.001527 (n=96) |
| `TIGER_graphlet_codebook_rq` | +0.000447 (n=2050) | -0.001470 (n=646) | -0.004102 (n=69) | +0.010765 (n=104) |
| `TIGER_graphlet_role_E` | -0.001088 (n=214) | +0.001491 (n=1448) | +0.018957 (n=132) | -0.001728 (n=1075) |
| `TIGER_graphlet_role_Q` | +0.000000 (n=1) | -0.002579 (n=40) | -0.016723 (n=100) | -0.017791 (n=2728) |
| `TIGER_graphlet_role_QR` | -0.005739 (n=218) | -0.002849 (n=1464) | +0.004042 (n=128) | -0.007854 (n=1059) |
| `TIGER_graphlet_role_R` | -0.023810 (n=7) | -0.004175 (n=240) | +0.008269 (n=127) | -0.016978 (n=2495) |
| `TIGER_graphlet_role_T` | +0.000000 (n=1) | +0.000000 (n=19) | -0.011291 (n=110) | -0.020099 (n=2739) |
| `TIGER_i234_A1` | +0.007283 (n=474) | -0.005657 (n=2192) | -0.023958 (n=23) | -0.004657 (n=180) |
| `TIGER_i234_A2` | -0.000233 (n=446) | -0.003873 (n=2242) | +0.035302 (n=14) | -0.002633 (n=167) |
| `TIGER_i234_A3` | +0.006027 (n=464) | -0.005955 (n=2207) | +0.049272 (n=21) | -0.005288 (n=177) |
| `TIGER_i234_A4` | +0.000362 (n=595) | -0.007661 (n=2086) | +0.068484 (n=17) | -0.014792 (n=171) |
| `TIGER_i234_A5` | -0.005895 (n=975) | -0.014180 (n=1683) | +0.001706 (n=34) | -0.021177 (n=177) |
| `TIGER_kappa_sk_only_rq` | +0.002837 (n=613) | -0.002452 (n=2066) | -0.018453 (n=16) | +0.004183 (n=174) |
| `TIGER_module_budget_prefix_rq` | -0.003310 (n=326) | -0.005431 (n=2440) | +0.000000 (n=3) | -0.002438 (n=100) |
| `TIGER_periodic_budget_prefix_rq` | +0.002762 (n=331) | -0.005107 (n=2430) | +0.000000 (n=2) | +0.001169 (n=106) |
| `TIGER_relative_curvature_ranking_rq` | -0.002397 (n=298) | -0.007827 (n=2439) | -0.031642 (n=8) | +0.003477 (n=124) |
| `TIGER_role_conditioned_l1_rq` | -0.004492 (n=313) | -0.009071 (n=2452) | +0.000000 (n=2) | -0.003511 (n=102) |
| `TIGER_role_conditioned_l2_rq` | -0.006998 (n=370) | -0.008301 (n=2396) | -0.100000 (n=2) | -0.000409 (n=101) |
| `TIGER_role_conditioned_l3_rq` | -0.001765 (n=334) | -0.007908 (n=2428) | +0.076923 (n=5) | -0.004542 (n=102) |
| `TIGER_three_expert_attention_rq` | -0.002211 (n=209) | -0.012387 (n=2548) | +0.025926 (n=9) | -0.011318 (n=103) |
| `TIGER_topology_residual_l2_rq` | +0.000144 (n=330) | -0.008785 (n=2429) | +0.000000 (n=3) | -0.005854 (n=107) |

## 4. #265 vs #269 (P3)

| quantity | value |
|---|---:|
| ΔA.1 (269 − 265) | -0.248689 |
| ΔA.1 (265 − TIGER) | -5.063267 |
| ΔA.1 (269 − TIGER) | -5.311956 |
| ΔR@10 (269 − 265) | -0.007364 |
| item Spearman(ΔA.1, Δhit) 269−265 | -0.009114 |
| item sign concordance | 0.515596 |
| #265 L2 groups | 170 |
| #265 L2 groups split across #269 groups | 127 |
| #269 L2 groups merging multiple #265 groups | 93 |
| #269 hit10 gain / loss vs TIGER | 1118 / 1691 |
| #265 J on the κ_t ruler | 0.168275 |
| #269 J on the κ_t ruler | 0.288252 |

### Rulers disagree on whether #269 improved at all

| arm | A.1 MAE (Σ ruler) | J (Σ ruler) | J (κ_t ruler) |
|---|---:|---:|---:|
| `TIGER_RQ-VAE` | 7.309 | 0.692 | 1.173 |
| `TIGER_euclid_local_curve_supervision_rq` (#265) | 2.246 | 0.183 | 0.168 |
| `TIGER_relative_curvature_ranking_rq` (#267) | 2.204 | 0.175 | 0.169 |
| `TIGER_calibrated_kappa_rq` (#269) | 1.997 | 0.326 | 0.288 |

A.1 (absolute MAE) says #269 beats #265 by 0.249; the relative J and the κ_t ruler say the opposite. The 'curvature improvement' premise is itself ruler-dependent, so no single-ruler claim is used as evidence for a recommendation effect.

Mediation (observational; not causal): curvature improvement -> SID compression rho = -0.281420, SID compression -> ΔR@10 rho = -0.471618, partial curvature -> ΔR given SID rho = -0.252735.

The Pearson +0.41 is carried by scale-outlier arms (|ΔA.1| in the hundreds) and disagrees with every rank statistic; it is not evidence of a monotone relation.

## 5. Family/protocol variance (P4)

Between-family SD of ΔR@10: **0.003559**; mean within-family SD: **0.000938** over 22 families.

## 6. Counterexamples and class strata

### `TIGER_calibrated_kappa_rq`
- curvature improved but hit lost (target items): []
- curvature worsened but hit gained: []

| class mask | items | mean ΔA.1 | mean Δhit | samples |
|---|---:|---:|---:|---:|
| 5 | 381 | -5.160528 | -0.005107 | 5519 |
| 9 | 626 | -4.718442 | -0.003417 | 5239 |
| 6 | 1172 | -5.217543 | -0.003133 | 12536 |
| 10 | 690 | -5.189821 | -0.001987 | 9127 |

### `TIGER_euclid_local_curve_supervision_rq`
- curvature improved but hit lost (target items): [5799]
- curvature worsened but hit gained: [12068]

| class mask | items | mean ΔA.1 | mean Δhit | samples |
|---|---:|---:|---:|---:|
| 6 | 1172 | -5.266738 | -0.006704 | 12536 |
| 10 | 690 | -5.176072 | -0.006532 | 9127 |
| 9 | 626 | -4.797529 | -0.005516 | 5239 |
| 5 | 381 | -5.153402 | +0.001746 | 5519 |

## 7. Verdict against the pre-registered decision table

| claim | pre-registered falsifier | observed | outcome |
|---|---|---|---|
| P1 monotone curvature->R | Spearman <= 0 (CI excluding 0 on the positive side) | Spearman -0.068069, CI [-0.511462, 0.351775], perm p 0.664467 | **not supported / not identifiable** (null; cannot claim a positive relation) |
| P2 conditional on SID | no interaction / curvature improves while SID collapses and R rises | stratum difference CI [-0.848121, 1.437446] includes 0 (compressed n=5, rho=-0.90; preserved n=37, rho=+0.065) | **not supported** |
| P3 #269 loss via SID | #269 A.1 not better than #265 under the common ruler | ΔA.1 -0.248689 (better), L2 170 -> 113 groups, (#269) fewer groups, split 127, merge 93, ΔR -0.007364; the κ_t ruler reverses the A.1 sign | **supported on the Σ common ruler; ruler-dependent; not a causal proof** |
| P4 family/protocol dominance | within-family association strong and protocol does not explain variance | between/within SD = 0.003559/0.000938; single-card n=3, four-card n=39 | **supported** |

## 8. FDR

| test | p | BH q |
|---|---:|---:|
| arm_spearman_P1 | 0.664467 | 0.664467 |
