# R2 pairs, one pass over 54 pairs

Sample: 200000 TRAIN-only transitions, 200000 random pairs (seed 20261109).

| NN | arm | R@10 | L1 codes | L1 ER | L1 max | L2 | L3 | L3 coll | gain bits |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `TIGER_c117_hyperbolic_item_semantic_id/9e1a7c3fc3e7` | 0.0556 | 240 | 91.3 | 0.035 | 6233 | 19515 | 0.206 | 0.33 |
| 02 | `TIGER_transition_busemann_gate/9f7ad2511666` | 0.0548 | 243 | 94.8 | 0.036 | 6674 | 19756 | 0.196 | 0.32 |
| 03 | `TIGER_curriculum-behaviour-rq/53279ad875c0` | 0.0548 | 249 | 126.0 | 0.026 | 7439 | 20516 | 0.166 | 0.26 |
| 04 | `TIGER_successor_residual_barycenter/f30066606b4c` | 0.0544 | 187 | 96.2 | 0.029 | 7083 | 18005 | 0.268 | 0.45 |
| 05 | `TIGER_tangent_successor_residual/7d73b0b626ce` | 0.0542 | 244 | 121.5 | 0.024 | 7223 | 20418 | 0.170 | 0.27 |
| 06 | `TIGER_successor-geodesic-coarse/64539cfe3956` | 0.0540 | 28 | 14.7 | 0.175 | 1044 | 11922 | 0.515 | 1.04 |
| 07 | `TIGER_successor_entropy_curvature/3afb4d7ff660` | 0.0506 | 256 | 43.9 | 0.219 | 3952 | 13702 | 0.443 | 0.84 |
| 08 | `TIGER_curvature-embedding-rq/d3a1b37e9ea4` | 0.0490 | 255 | 167.8 | 0.016 | 14021 | 24333 | 0.010 | 0.01 |
| 09 | `TIGER_factorized_joint_codebook` | 0.0514 | 220 | 68.9 | 0.051 | 5376 | 18939 | 0.230 | 0.38 |
| 10 | `TIGER_af3sign_capacity` | 0.0545 | 177 | 110.6 | 0.026 | 6701 | 20420 | 0.169 | 0.27 |
| 11 | `TIGER_M1_learned_kappa` | 0.0563 | 42 | 23.0 | 0.141 | 2933 | 16834 | 0.315 | 0.55 |
| 12 | `TIGER_M3_role_profiles` | 0.0564 | 13 | 5.1 | 0.371 | 982 | 11783 | 0.521 | 1.06 |
| 13 | `TIGER_M4_post_sid_af3` | 0.0562 | 156 | 89.2 | 0.034 | 5222 | 19471 | 0.208 | 0.34 |
| 14 | `TIGER_M2_radial_manifold` | 0.0571 | 15 | 7.4 | 0.205 | 593 | 9007 | 0.634 | 1.45 |
| 15 | `TIGER_mixed_rq_L` | 0.0560 | 49 | 25.7 | 0.095 | 3085 | 17272 | 0.298 | 0.51 |
| 16 | `TIGER_mixed_rq_E` | 0.0542 | 87 | 56.6 | 0.040 | 4655 | 19274 | 0.216 | 0.35 |
| 17 | `TIGER_mixed_rq_A` | 0.0559 | 50 | 25.0 | 0.115 | 3019 | 16873 | 0.314 | 0.54 |
| 18 | `TIGER_mixed_rq_G` | 0.0557 | 64 | 32.4 | 0.092 | 3714 | 17685 | 0.281 | 0.48 |
| 19 | `TIGER_mixed_rq_F0` | 0.0573 | 79 | 44.1 | 0.059 | 4151 | 18264 | 0.257 | 0.43 |
| 20 | `TIGER_community-balance-l1-rq` | 0.0543 | 256 | 242.9 | 0.008 | 7661 | 21088 | 0.142 | 0.22 |
| 21 | `TIGER_behaviour-pairs-levels-rq` | 0.0558 | 70 | 31.5 | 0.066 | 3688 | 16881 | 0.313 | 0.54 |
| 22 | `TIGER_behaviour-pairs-curved-rq` | 0.0565 | 79 | 35.7 | 0.091 | 4110 | 17608 | 0.284 | 0.48 |
| 23 | `TIGER_curvature-stratum-sinkhorn-v1` | 0.0431 | 256 | 251.2 | 0.005 | 955 | 9510 | 0.613 | 1.37 |
| 24 | `TIGER_RQ-VAE` | 0.0402 | 247 | 120.1 | 0.035 | 7414 | 20223 | 0.177 | 0.28 |
| 25 | `TIGER_behaviour_code_cohesion` | 0.0546 | 12 | 3.8 | 0.421 | 574 | 11517 | 0.532 | 1.09 |
| 26 | `TIGER_cooccur_coherent_sid` | 0.0559 | 242 | 122.6 | 0.035 | 7375 | 20412 | 0.170 | 0.27 |
| 27 | `TIGER_incoming_context_barycenter` | 0.0585 | 104 | 2.4 | 0.693 | 853 | 8545 | 0.652 | 1.52 |
| 28 | `TIGER_cogeo_region_argmax` | 0.0398 | 2 | 1.0 | 1.000 | 16 | 1557 | 0.937 | 3.98 |
| 29 | `TIGER_COOC_RANK_FRECHET` | 0.0500 | 254 | 141.1 | 0.030 | 6810 | 20130 | 0.181 | 0.29 |
| 30 | `TIGER_BRANCH_CONDITIONED_FRECHET_RESIDUAL` | 0.0546 | 253 | 100.7 | 0.031 | 9047 | 22007 | 0.105 | 0.16 |
| 31 | `TIGER_interest_signature_hypball` | 0.0512 | 16 | 15.4 | 0.103 | 1684 | 16790 | 0.317 | 0.55 |
| 32 | `TIGER_branch-geodesic-rank` | 0.0497 | 27 | 25.5 | 0.060 | 3156 | 22858 | 0.070 | 0.11 |
| 33 | `TIGER_poincare-frechet-successor-codebook` | 0.0532 | 203 | 49.0 | 0.049 | 1867 | 8007 | 0.674 | 1.62 |
| 34 | `TIGER_BRad-RQ` | 0.0306 | 12 | 11.5 | 0.112 | 60 | 170 | 0.993 | 7.18 |
| 35 | `TIGER_GDELTA-SUCC-PULL-RQ` | 0.0409 | 13 | 12.5 | 0.102 | 162 | 16174 | 0.342 | 0.60 |
| 36 | `TIGER_HGS-TREE` | 0.0513 | 4 | 2.0 | 0.516 | 123 | 1203 | 0.951 | 4.35 |
| 37 | `TIGER_CURVALLOC-LORENTZ-RQ` | 0.0419 | 2 | 1.1 | 0.972 | 13 | 56 | 0.998 | 8.78 |
| 38 | `TIGER_EQVOTE-HYP-RQ` | 0.0552 | 242 | 123.0 | 0.023 | 7312 | 20341 | 0.173 | 0.27 |
| 39 | `TIGER_curvature-prefix-ranking` | 0.0528 | 114 | 29.0 | 0.102 | 4828 | 19346 | 0.213 | 0.35 |
| 40 | `TIGER_curvature-prefix-ranking-euclid` | 0.0540 | 151 | 31.0 | 0.158 | 5120 | 19528 | 0.206 | 0.33 |
| 41 | `TIGER_curvature-prefix-ranking-nobehaviour` | 0.0525 | 134 | 30.6 | 0.090 | 5079 | 19545 | 0.205 | 0.33 |
| 42 | `TIGER_curvature-prefix-ranking-forman` | 0.0536 | 152 | 82.1 | 0.038 | 8587 | 19518 | 0.206 | 0.33 |
| 43 | `TIGER_community-l1-rq` | 0.0526 | 256 | 86.3 | 0.035 | 6280 | 19244 | 0.217 | 0.35 |
| 44 | `TIGER_community-pairs-guarded-rq` | 0.0433 | 187 | 101.6 | 0.048 | 6575 | 20425 | 0.169 | 0.27 |
| 45 | `TIGER_community-repulse-l1-rq` | 0.0542 | 255 | 84.5 | 0.036 | 6318 | 19089 | 0.224 | 0.37 |
| 46 | `TIGER_community-balance-repulse-rq` | 0.0532 | 256 | 254.2 | 0.005 | 8575 | 21952 | 0.107 | 0.16 |
| 47 | `TIGER_community_graph_recon` | 0.0558 | 246 | 124.4 | 0.026 | 7432 | 20476 | 0.167 | 0.26 |
| 48 | `TIGER_BEHAVIOR_R48` | 0.0619 | 240 | 222.1 | 0.011 | 15495 | 23582 | 0.041 | 0.06 |
| 49 | `TIGER_BEHAVIOR_SAMPLE_RQ-VAE__resample` | 0.0637 | 238 | 223.3 | 0.010 | 15860 | 23675 | 0.037 | 0.05 |
| 50 | `TIGER_behavior_residual_geometry_transport__r8_control` | 0.0634 | 238 | 217.9 | 0.010 | 15655 | 23609 | 0.040 | 0.06 |
| 51 | `TIGER_bootstrap_code_stability__resample` | 0.0634 | 238 | 221.5 | 0.011 | 15711 | 23571 | 0.041 | 0.06 |
| 52 | `TIGER_c106_directed_assignment_usage_balance__r8_control` | 0.0624 | 236 | 215.8 | 0.014 | 15580 | 23515 | 0.044 | 0.06 |
| 53 | `TIGER_c106_directed_codeword_margin_geometry__r8_control` | 0.0637 | 242 | 225.9 | 0.011 | 15724 | 23533 | 0.043 | 0.06 |
| 54 | `TIGER_c1_behaviour_conditioned_residual_target__r8_control` | 0.0610 | 241 | 201.0 | 0.012 | 14725 | 23420 | 0.047 | 0.07 |

## Spearman against test_recall@10 (n = 54 pairs with Stage 3)

| metric | rho | 95% bootstrap |
|---|---|---|
| l1_codes | 0.045 | [-0.278, 0.338] |
| l1_er | 0.274 | [-0.044, 0.555] |
| l1_max_share | -0.284 | [-0.565, 0.018] |
| l2_prefixes | 0.397 | [0.080, 0.645] |
| l3_distinct | 0.335 | [0.026, 0.597] |
| l3_collision | -0.335 | [-0.597, -0.026] |
| l3_gain_bits | -0.335 | [-0.597, -0.026] |
| prefix_overlap@1_gap | 0.331 | [0.067, 0.583] |
| prefix_overlap@2_gap | -0.189 | [-0.478, 0.125] |
| prefix_overlap@3_gap | -0.173 | [-0.453, 0.147] |

## Same, one row per distinct item_sids.json (n = 54)

| metric | rho |
|---|---|
| l1_codes | 0.045 |
| l1_er | 0.274 |
| l1_max_share | -0.284 |
| l2_prefixes | 0.397 |
| l3_distinct | 0.335 |
| l3_collision | -0.335 |
| l3_gain_bits | -0.335 |
| prefix_overlap@1_gap | 0.331 |
| prefix_overlap@2_gap | -0.189 |
| prefix_overlap@3_gap | -0.173 |
