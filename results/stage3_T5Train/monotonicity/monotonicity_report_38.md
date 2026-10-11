# Multi-model SID behaviour vs Recall@10: monotonicity analysis

Setting `test_transition_neg10` over 38 models. Offline only; nothing was retrained.


## Rank correlations against Recall@10

| metric | n | Spearman rho | p | Kendall tau | p | verdict |
|---|---|---|---|---|---|---|
| L1_p_share_pos | 38 | +0.311 | 0.0577 | +0.275 | 0.0153 | no_association |
| L1_p_share_neg | 38 | +0.106 | 0.525 | +0.081 | 0.474 | no_association |
| L1_gap | 38 | +0.559 | 0.000262 | +0.442 | 9.23e-05 | positive |
| L1_er | 38 | +0.809 | 7.82e-10 | +0.679 | 2.01e-09 | positive |
| L1_L2_gap | 38 | +0.163 | 0.328 | +0.080 | 0.492 | no_association |
| L1_L2_er | 27 | +0.844 | 3.1e-08 | +0.704 | 1.27e-08 | positive |
| L1_L2_L3_gap | 38 | +0.252 | 0.127 | +0.151 | 0.194 | no_association |
| L2_given_L1_retention | 38 | +0.129 | 0.441 | +0.036 | 0.76 | no_association |
| L3_given_L2_retention | 24 | +0.338 | 0.106 | +0.254 | 0.087 | no_association |
| collision_rate | 38 | -0.212 | 0.202 | -0.142 | 0.221 | no_association |
| L1_prefixes | 38 | -0.216 | 0.193 | -0.174 | 0.146 | no_association |
| L1_max_group | 38 | +0.036 | 0.828 | +0.007 | 0.95 | no_association |
| L1_gini | 38 | +0.062 | 0.713 | +0.021 | 0.85 | no_association |


## One model vs ten negatives

| metric | n | rho (10 neg) | rho (1 neg) | verdicts |
|---|---|---|---|---|
| L1_p_share_pos | 38 | +0.311 | +0.311 | no_association / no_association |
| L1_p_share_neg | 38 | +0.106 | +0.057 | no_association / no_association |
| L1_gap | 38 | +0.559 | +0.559 | positive / positive |
| L1_er | 38 | +0.809 | +0.805 | positive / positive |
| L1_L2_gap | 38 | +0.163 | +0.164 | no_association / no_association |
| L1_L2_er | 27 | +0.844 | +0.780 | positive / positive |
| L1_L2_L3_gap | 38 | +0.252 | +0.252 | no_association / no_association |
| L2_given_L1_retention | 38 | +0.129 | +0.129 | no_association / no_association |
| L3_given_L2_retention | 24 | +0.338 | +0.338 | no_association / no_association |
| collision_rate | 38 | -0.212 | -0.212 | no_association / no_association |
| L1_prefixes | 38 | -0.216 | -0.216 | no_association / no_association |
| L1_max_group | 38 | +0.036 | +0.036 | no_association / no_association |
| L1_gini | 38 | +0.062 | +0.062 | no_association / no_association |


## Leave-one-out stability

| metric | rho full | rho min | rho max | sign flips | n dropped |
|---|---|---|---|---|---|
| L1_p_share_pos | +0.311 | +0.261 | +0.408 | 0 | 38 |
| L1_p_share_neg | +0.106 | +0.045 | +0.199 | 0 | 38 |
| L1_gap | +0.559 | +0.523 | +0.636 | 0 | 38 |
| L1_er | +0.809 | +0.793 | +0.893 | 0 | 38 |
| L1_L2_gap | +0.163 | +0.118 | +0.232 | 0 | 38 |
| L1_L2_er | +0.844 | +0.826 | +0.915 | 0 | 27 |
| L1_L2_L3_gap | +0.252 | +0.212 | +0.311 | 0 | 38 |
| L2_given_L1_retention | +0.129 | +0.083 | +0.197 | 0 | 38 |
| L3_given_L2_retention | +0.338 | +0.253 | +0.443 | 0 | 24 |
| collision_rate | -0.212 | -0.263 | -0.148 | 0 | 38 |
| L1_prefixes | -0.216 | -0.321 | -0.180 | 0 | 38 |
| L1_max_group | +0.036 | -0.019 | +0.123 | 7 | 38 |
| L1_gini | +0.062 | +0.011 | +0.128 | 0 | 38 |


## Subgroups

| metric | subgroup | n | rho | verdict |
|---|---|---|---|---|
| L1_p_share_pos | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.800 | no_association |
| L1_p_share_pos | family:TIGER_FORK_RQ-VAE | 9 | +0.067 | no_association |
| L1_p_share_pos | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.714 | no_association |
| L1_p_share_pos | family:curvature_RQ-VAE | 17 | +0.596 | positive |
| L1_p_share_pos | seeds:resample_s42_s43_s44 | 3 | +1.000 | positive |
| L1_p_share_pos | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | +0.500 | no_association |
| L1_p_share_neg | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | -0.800 | no_association |
| L1_p_share_neg | family:TIGER_FORK_RQ-VAE | 9 | +0.117 | no_association |
| L1_p_share_neg | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.857 | negative |
| L1_p_share_neg | family:curvature_RQ-VAE | 17 | +0.189 | no_association |
| L1_p_share_neg | seeds:resample_s42_s43_s44 | 3 | -0.500 | no_association |
| L1_p_share_neg | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | +1.000 | positive |
| L1_gap | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.800 | no_association |
| L1_gap | family:TIGER_FORK_RQ-VAE | 9 | +0.400 | no_association |
| L1_gap | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.607 | no_association |
| L1_gap | family:curvature_RQ-VAE | 17 | +0.946 | positive |
| L1_gap | seeds:resample_s42_s43_s44 | 3 | +1.000 | positive |
| L1_gap | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | +0.500 | no_association |
| L1_er | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.600 | no_association |
| L1_er | family:TIGER_FORK_RQ-VAE | 9 | +0.483 | no_association |
| L1_er | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | +0.929 | positive |
| L1_er | family:curvature_RQ-VAE | 17 | +0.956 | positive |
| L1_er | seeds:resample_s42_s43_s44 | 3 | +0.500 | no_association |
| L1_er | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | -0.500 | no_association |
| L1_L2_gap | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | -0.400 | no_association |
| L1_L2_gap | family:TIGER_FORK_RQ-VAE | 9 | +0.267 | no_association |
| L1_L2_gap | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.607 | no_association |
| L1_L2_gap | family:curvature_RQ-VAE | 17 | -0.668 | negative |
| L1_L2_gap | seeds:resample_s42_s43_s44 | 3 | -1.000 | negative |
| L1_L2_gap | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | n/a | undetermined |
| L1_L2_er | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.800 | no_association |
| L1_L2_er | family:TIGER_FORK_RQ-VAE | 9 | +0.283 | no_association |
| L1_L2_er | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | +0.964 | positive |
| L1_L2_er | family:curvature_RQ-VAE | 6 | +0.943 | positive |
| L1_L2_er | seeds:resample_s42_s43_s44 | 3 | +1.000 | positive |
| L1_L2_er | seeds:euclid_l2_256_72k_s42_s43_s44 | 0 | n/a | undetermined |
| L1_L2_L3_gap | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.400 | no_association |
| L1_L2_L3_gap | family:TIGER_FORK_RQ-VAE | 9 | +0.783 | positive |
| L1_L2_L3_gap | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.321 | no_association |
| L1_L2_L3_gap | family:curvature_RQ-VAE | 17 | -0.697 | negative |
| L1_L2_L3_gap | seeds:resample_s42_s43_s44 | 3 | +0.500 | no_association |
| L1_L2_L3_gap | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | n/a | undetermined |
| L2_given_L1_retention | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | -0.400 | no_association |
| L2_given_L1_retention | family:TIGER_FORK_RQ-VAE | 9 | -0.017 | no_association |
| L2_given_L1_retention | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.607 | no_association |
| L2_given_L1_retention | family:curvature_RQ-VAE | 17 | -0.723 | negative |
| L2_given_L1_retention | seeds:resample_s42_s43_s44 | 3 | -1.000 | negative |
| L2_given_L1_retention | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | n/a | undetermined |
| L3_given_L2_retention | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.800 | no_association |
| L3_given_L2_retention | family:TIGER_FORK_RQ-VAE | 9 | +0.583 | no_association |
| L3_given_L2_retention | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | +0.536 | no_association |
| L3_given_L2_retention | family:curvature_RQ-VAE | 3 | +1.000 | positive |
| L3_given_L2_retention | seeds:resample_s42_s43_s44 | 3 | +1.000 | positive |
| L3_given_L2_retention | seeds:euclid_l2_256_72k_s42_s43_s44 | 0 | n/a | undetermined |
| collision_rate | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | -0.800 | no_association |
| collision_rate | family:TIGER_FORK_RQ-VAE | 9 | -0.600 | no_association |
| collision_rate | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.857 | negative |
| collision_rate | family:curvature_RQ-VAE | 17 | -0.760 | negative |
| collision_rate | seeds:resample_s42_s43_s44 | 3 | -0.500 | no_association |
| collision_rate | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | n/a | undetermined |
| L1_prefixes | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.800 | no_association |
| L1_prefixes | family:TIGER_FORK_RQ-VAE | 9 | -0.118 | no_association |
| L1_prefixes | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | +0.667 | no_association |
| L1_prefixes | family:curvature_RQ-VAE | 17 | +0.408 | no_association |
| L1_prefixes | seeds:resample_s42_s43_s44 | 3 | +1.000 | positive |
| L1_prefixes | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | n/a | undetermined |
| L1_max_group | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.800 | no_association |
| L1_max_group | family:TIGER_FORK_RQ-VAE | 9 | -0.317 | no_association |
| L1_max_group | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.750 | no_association |
| L1_max_group | family:curvature_RQ-VAE | 17 | -0.147 | no_association |
| L1_max_group | seeds:resample_s42_s43_s44 | 3 | +1.000 | positive |
| L1_max_group | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | -0.500 | no_association |
| L1_gini | family:TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 4 | +0.400 | no_association |
| L1_gini | family:TIGER_FORK_RQ-VAE | 9 | -0.417 | no_association |
| L1_gini | family:TIGER_HYPERBOLIC_RQ-VAE | 7 | -0.214 | no_association |
| L1_gini | family:curvature_RQ-VAE | 17 | -0.181 | no_association |
| L1_gini | seeds:resample_s42_s43_s44 | 3 | +0.500 | no_association |
| L1_gini | seeds:euclid_l2_256_72k_s42_s43_s44 | 3 | -1.000 | negative |


## Group-size-preserving null

| model | metric | observed | null mean | null 95% CI | exceeds null |
|---|---|---|---|---|---|
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample | L1_gap | 0.0776 | 0.0000 | [-0.0006, 0.0007] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask | L1_gap | 0.0827 | -0.0001 | [-0.0008, 0.0006] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43 | L1_gap | 0.0767 | -0.0000 | [-0.0006, 0.0006] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44 | L1_gap | 0.0757 | 0.0000 | [-0.0010, 0.0013] | True |
| TIGER_FORK_RQ-VAE/fork_radial_corr | L1_gap | 0.0864 | 0.0000 | [-0.0009, 0.0008] | True |
| TIGER_FORK_RQ-VAE/fork_more_sib | L1_gap | 0.0847 | 0.0001 | [-0.0008, 0.0011] | True |
| TIGER_FORK_RQ-VAE/prefix_sib | L1_gap | 0.1004 | 0.0000 | [-0.0024, 0.0022] | True |
| TIGER_FORK_RQ-VAE/fork_euclid_group | L1_gap | 0.0751 | 0.0000 | [-0.0008, 0.0008] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp2_euclid_matched | L1_gap | 0.0758 | 0.0000 | [-0.0008, 0.0007] | True |
| TIGER_FORK_RQ-VAE/fork_ang_only | L1_gap | 0.0770 | 0.0000 | [-0.0009, 0.0011] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp4_euclid_delta | L1_gap | 0.0740 | -0.0000 | [-0.0008, 0.0008] | True |
| smoothness_arms/A_euclid_smooth_s42 | L1_gap | 0.0466 | -0.0000 | [-0.0005, 0.0005] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s43 | L1_gap | 0.0476 | 0.0000 | [-0.0006, 0.0008] | True |
| TIGER_FORK_RQ-VAE/nce_euclid_radius_control | L1_gap | 0.0836 | -0.0001 | [-0.0013, 0.0011] | True |
| smoothness_arms/B_poincare_smooth_s42 | L1_gap | 0.0443 | -0.0001 | [-0.0010, 0.0006] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s42 | L1_gap | 0.0447 | -0.0000 | [-0.0007, 0.0009] | True |
| TIGER_FORK_RQ-VAE/nce_euclid_group | L1_gap | 0.0846 | -0.0001 | [-0.0010, 0.0009] | True |
| TIGER_FORK_RQ-VAE/nce_hyp_group | L1_gap | 0.0903 | -0.0000 | [-0.0007, 0.0007] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s44 | L1_gap | 0.0461 | 0.0000 | [-0.0007, 0.0006] | True |
| curvature_RQ-VAE | L1_gap | 0.0448 | -0.0000 | [-0.0007, 0.0007] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_ball | L1_gap | 0.0609 | -0.0000 | [-0.0009, 0.0007] | True |
| TIGER_FORK_RQ-VAE/fork_hyp_group | L1_gap | 0.0656 | -0.0001 | [-0.0010, 0.0007] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp2_poincare_c1 | L1_gap | 0.0840 | 0.0000 | [-0.0010, 0.0010] | True |
| l2_cone_arms/A_euclid_nocone_s42 | L1_gap | 0.0385 | -0.0000 | [-0.0006, 0.0007] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw | L1_gap | 0.2389 | 0.0002 | [-0.0029, 0.0033] | True |
| TIGER_RQ-VAE | L1_gap | 0.0750 | -0.0000 | [-0.0013, 0.0012] | True |
| l2_assignment_arms/A_bucket_sinkhorn_s42 | L1_gap | 0.0369 | -0.0001 | [-0.0008, 0.0006] | True |
| l2_cone_arms/B_poincare_nocone_s42 | L1_gap | 0.0367 | 0.0000 | [-0.0007, 0.0008] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw | L1_gap | 0.1813 | 0.0002 | [-0.0019, 0.0020] | True |
| l2_assignment_arms/B_global_sinkhorn_s42 | L1_gap | 0.0362 | -0.0001 | [-0.0006, 0.0005] | True |
| coarse_l1_arms/B_poincare_coarse_l1_s42 | L1_gap | 0.0240 | -0.0000 | [-0.0007, 0.0006] | True |
| l2_cone_arms/D_poincare_cone_s42 | L1_gap | 0.0197 | -0.0000 | [-0.0007, 0.0006] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare | L1_gap | 0.0762 | 0.0001 | [-0.0022, 0.0015] | True |
| metric_cone_arms/B_poincare_metriccone_s42 | L1_gap | 0.0199 | -0.0001 | [-0.0006, 0.0006] | True |
| metric_cone_arms/A_euclid_metriccone_s42 | L1_gap | 0.0045 | 0.0000 | [-0.0034, 0.0035] | True |
| smooth_fair_arms/A_euclid_smoothfair_s42 | L1_gap | 0.0448 | -0.0000 | [-0.0006, 0.0009] | True |
| smooth_fair_arms/B_poincare_smoothfair_s42 | L1_gap | 0.0413 | -0.0000 | [-0.0007, 0.0006] | True |
| l2_cone_arms_fixed_rejection/D_poincare_cone_s42 | L1_gap | 0.0226 | 0.0000 | [-0.0006, 0.0006] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample | L1_er | 17.5169 | 1.0061 | [0.8754, 1.1508] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask | L1_er | 18.2096 | 0.9911 | [0.8427, 1.1393] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43 | L1_er | 16.5154 | 0.9988 | [0.8709, 1.1255] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44 | L1_er | 16.7205 | 1.0160 | [0.8210, 1.3028] | True |
| TIGER_FORK_RQ-VAE/fork_radial_corr | L1_er | 12.6906 | 1.0090 | [0.8724, 1.1461] | True |
| TIGER_FORK_RQ-VAE/fork_more_sib | L1_er | 12.1346 | 1.0143 | [0.8790, 1.2013] | True |
| TIGER_FORK_RQ-VAE/prefix_sib | L1_er | 3.5316 | 1.0019 | [0.9324, 1.0680] | True |
| TIGER_FORK_RQ-VAE/fork_euclid_group | L1_er | 11.5681 | 1.0088 | [0.8745, 1.1423] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp2_euclid_matched | L1_er | 14.7579 | 1.0096 | [0.8631, 1.1439] | True |
| TIGER_FORK_RQ-VAE/fork_ang_only | L1_er | 12.1163 | 1.0090 | [0.8699, 1.2134] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp4_euclid_delta | L1_er | 13.2341 | 1.0012 | [0.8830, 1.1367] | True |
| smoothness_arms/A_euclid_smooth_s42 | L1_er | 11.5231 | 0.9929 | [0.8732, 1.1448] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s43 | L1_er | 11.2872 | 1.0075 | [0.8501, 1.2403] | True |
| TIGER_FORK_RQ-VAE/nce_euclid_radius_control | L1_er | 7.0655 | 0.9949 | [0.9002, 1.0932] | True |
| smoothness_arms/B_poincare_smooth_s42 | L1_er | 11.2561 | 0.9901 | [0.7870, 1.1588] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s42 | L1_er | 10.7302 | 0.9978 | [0.8289, 1.2578] | True |
| TIGER_FORK_RQ-VAE/nce_euclid_group | L1_er | 6.9675 | 0.9960 | [0.9233, 1.0743] | True |
| TIGER_FORK_RQ-VAE/nce_hyp_group | L1_er | 7.6368 | 1.0002 | [0.9052, 1.1148] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s44 | L1_er | 11.4785 | 1.0149 | [0.8311, 1.1713] | True |
| curvature_RQ-VAE | L1_er | 10.5141 | 1.0011 | [0.8443, 1.1917] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_ball | L1_er | 8.4745 | 0.9960 | [0.8879, 1.0962] | True |
| TIGER_FORK_RQ-VAE/fork_hyp_group | L1_er | 10.9246 | 0.9820 | [0.8591, 1.1209] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp2_poincare_c1 | L1_er | 10.7395 | 1.0029 | [0.8904, 1.1249] | True |
| l2_cone_arms/A_euclid_nocone_s42 | L1_er | 10.0665 | 1.0039 | [0.8408, 1.2092] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw | L1_er | 3.4372 | 1.0025 | [0.9679, 1.0377] | True |
| TIGER_RQ-VAE | L1_er | 6.4097 | 0.9975 | [0.8911, 1.1059] | True |
| l2_assignment_arms/A_bucket_sinkhorn_s42 | L1_er | 9.7693 | 0.9882 | [0.8049, 1.1584] | True |
| l2_cone_arms/B_poincare_nocone_s42 | L1_er | 9.7270 | 1.0072 | [0.8308, 1.2297] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw | L1_er | 4.4933 | 1.0039 | [0.9674, 1.0356] | True |
| l2_assignment_arms/B_global_sinkhorn_s42 | L1_er | 9.6013 | 0.9796 | [0.8518, 1.1473] | True |
| coarse_l1_arms/B_poincare_coarse_l1_s42 | L1_er | 6.5684 | 0.9972 | [0.8415, 1.1797] | True |
| l2_cone_arms/D_poincare_cone_s42 | L1_er | 5.8482 | 0.9944 | [0.8248, 1.1619] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare | L1_er | 3.2577 | 1.0033 | [0.9292, 1.0494] | True |
| metric_cone_arms/B_poincare_metriccone_s42 | L1_er | 5.0303 | 0.9912 | [0.8692, 1.1474] | True |
| metric_cone_arms/A_euclid_metriccone_s42 | L1_er | 1.0353 | 1.0003 | [0.9735, 1.0276] | True |
| smooth_fair_arms/A_euclid_smoothfair_s42 | L1_er | 11.4480 | 1.0044 | [0.8651, 1.2449] | True |
| smooth_fair_arms/B_poincare_smoothfair_s42 | L1_er | 10.1646 | 0.9990 | [0.8368, 1.1628] | True |
| l2_cone_arms_fixed_rejection/D_poincare_cone_s42 | L1_er | 6.5300 | 1.0110 | [0.8608, 1.1842] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample | L1_p_share_pos | 0.0823 | 0.0046 | [0.0042, 0.0052] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask | L1_p_share_pos | 0.0875 | 0.0046 | [0.0041, 0.0052] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43 | L1_p_share_pos | 0.0816 | 0.0048 | [0.0042, 0.0053] | True |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44 | L1_p_share_pos | 0.0805 | 0.0048 | [0.0041, 0.0054] | True |
| TIGER_FORK_RQ-VAE/fork_radial_corr | L1_p_share_pos | 0.0938 | 0.0062 | [0.0056, 0.0069] | True |
| TIGER_FORK_RQ-VAE/fork_more_sib | L1_p_share_pos | 0.0923 | 0.0063 | [0.0058, 0.0069] | True |
| TIGER_FORK_RQ-VAE/prefix_sib | L1_p_share_pos | 0.1400 | 0.0346 | [0.0326, 0.0368] | True |
| TIGER_FORK_RQ-VAE/fork_euclid_group | L1_p_share_pos | 0.0822 | 0.0061 | [0.0055, 0.0066] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp2_euclid_matched | L1_p_share_pos | 0.0813 | 0.0055 | [0.0051, 0.0061] | True |
| TIGER_FORK_RQ-VAE/fork_ang_only | L1_p_share_pos | 0.0839 | 0.0060 | [0.0054, 0.0067] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp4_euclid_delta | L1_p_share_pos | 0.0801 | 0.0061 | [0.0055, 0.0067] | True |
| smoothness_arms/A_euclid_smooth_s42 | L1_p_share_pos | 0.0510 | 0.0038 | [0.0033, 0.0042] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s43 | L1_p_share_pos | 0.0522 | 0.0039 | [0.0035, 0.0045] | True |
| TIGER_FORK_RQ-VAE/nce_euclid_radius_control | L1_p_share_pos | 0.0974 | 0.0121 | [0.0112, 0.0131] | True |
| smoothness_arms/B_poincare_smooth_s42 | L1_p_share_pos | 0.0486 | 0.0040 | [0.0035, 0.0044] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s42 | L1_p_share_pos | 0.0493 | 0.0039 | [0.0034, 0.0045] | True |
| TIGER_FORK_RQ-VAE/nce_euclid_group | L1_p_share_pos | 0.0988 | 0.0128 | [0.0119, 0.0138] | True |
| TIGER_FORK_RQ-VAE/nce_hyp_group | L1_p_share_pos | 0.1040 | 0.0070 | [0.0063, 0.0077] | True |
| curvature_RQ-VAE/euclid_l2_256_72k_s44 | L1_p_share_pos | 0.0504 | 0.0039 | [0.0034, 0.0044] | True |
| curvature_RQ-VAE | L1_p_share_pos | 0.0495 | 0.0039 | [0.0035, 0.0044] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_ball | L1_p_share_pos | 0.0691 | 0.0074 | [0.0068, 0.0079] | True |
| TIGER_FORK_RQ-VAE/fork_hyp_group | L1_p_share_pos | 0.0722 | 0.0064 | [0.0059, 0.0070] | True |
| TIGER_HYPERBOLIC_RQ-VAE/exp2_poincare_c1 | L1_p_share_pos | 0.0926 | 0.0089 | [0.0082, 0.0098] | True |
| l2_cone_arms/A_euclid_nocone_s42 | L1_p_share_pos | 0.0427 | 0.0039 | [0.0034, 0.0043] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw | L1_p_share_pos | 0.3369 | 0.0903 | [0.0870, 0.0939] | True |
| TIGER_RQ-VAE | L1_p_share_pos | 0.0888 | 0.0117 | [0.0105, 0.0126] | True |
| l2_assignment_arms/A_bucket_sinkhorn_s42 | L1_p_share_pos | 0.0411 | 0.0038 | [0.0032, 0.0043] | True |
| l2_cone_arms/B_poincare_nocone_s42 | L1_p_share_pos | 0.0409 | 0.0039 | [0.0034, 0.0044] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_euclid_raw | L1_p_share_pos | 0.2332 | 0.0581 | [0.0555, 0.0606] | True |
| l2_assignment_arms/B_global_sinkhorn_s42 | L1_p_share_pos | 0.0404 | 0.0039 | [0.0035, 0.0043] | True |
| coarse_l1_arms/B_poincare_coarse_l1_s42 | L1_p_share_pos | 0.0283 | 0.0039 | [0.0034, 0.0044] | True |
| l2_cone_arms/D_poincare_cone_s42 | L1_p_share_pos | 0.0238 | 0.0040 | [0.0035, 0.0045] | True |
| TIGER_HYPERBOLIC_RQ-VAE/fix_det_poincare | L1_p_share_pos | 0.1100 | 0.0306 | [0.0290, 0.0321] | True |
| metric_cone_arms/B_poincare_metriccone_s42 | L1_p_share_pos | 0.0249 | 0.0045 | [0.0041, 0.0051] | True |
| metric_cone_arms/A_euclid_metriccone_s42 | L1_p_share_pos | 0.1327 | 0.1296 | [0.1258, 0.1338] | False |
| smooth_fair_arms/A_euclid_smoothfair_s42 | L1_p_share_pos | 0.0491 | 0.0040 | [0.0036, 0.0045] | True |
| smooth_fair_arms/B_poincare_smoothfair_s42 | L1_p_share_pos | 0.0458 | 0.0040 | [0.0036, 0.0044] | True |
| l2_cone_arms_fixed_rejection/D_poincare_cone_s42 | L1_p_share_pos | 0.0267 | 0.0039 | [0.0035, 0.0045] | True |
