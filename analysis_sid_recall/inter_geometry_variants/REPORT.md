# Issue #135 - inter-prefix geometry aggregation variants vs Recall

108 content-hash-matched (SID, run) rows; excluded for a measured digest mismatch: 2. Frozen registry sha256 `56256181a4238980...` (70 candidates, written before the scan). Cohorts: all 108, curvature_RQ-VAE 76, other 32, strict73 73.

These 108 arms are an **exploratory** scan: no held-out new SID family was used, and no formula was revised after seeing rho.

## Baselines (#133 metrics, recomputed here)

## Baseline features

| candidate | n | coverage | rho | p | Holm p | 95% CI | status |
|---|---|---|---|---|---|---|---|
| `base_density_l1` | 107 | 0.99 | +0.133 | 0.1634 | nan | [-0.095, +0.357] | defined |
| `base_transitivity_l1` | 106 | 0.98 | +0.182 | 0.05859 | nan | [-0.055, +0.396] | defined |
| `base_triangles_l1` | 107 | 0.99 | +0.161 | 0.1032 | nan | [-0.066, +0.371] | defined |
| `base_cycle_rank_per_node_l1` | 108 | 1.00 | +0.194 | 0.04219 | nan | [-0.038, +0.409] | defined |
| `base_degree_median_l1` | 108 | 1.00 | +0.193 | 0.04239 | nan | [-0.037, +0.409] | defined |
| `base_degree_p90_l1` | 108 | 1.00 | +0.110 | 0.2529 | nan | [-0.127, +0.331] | defined |
| `base_forman_mean_l1` | 107 | 0.99 | -0.161 | 0.102 | nan | [-0.371, +0.067] | defined |
| `base_forman_negative_fraction_l1` | 107 | 0.99 | | | | | UNDEFINED_DEGENERATE |
| `base_density_l2` | 108 | 1.00 | -0.451 | 0.0002 | nan | [-0.608, -0.254] | defined |
| `base_transitivity_l2` | 108 | 1.00 | -0.465 | 0.0002 | nan | [-0.619, -0.272] | defined |
| `base_triangles_l2` | 108 | 1.00 | -0.145 | 0.132 | nan | [-0.359, +0.076] | defined |
| `base_cycle_rank_per_node_l2` | 108 | 1.00 | -0.235 | 0.0136 | nan | [-0.444, -0.023] | defined |
| `base_degree_median_l2` | 108 | 1.00 | -0.267 | 0.004999 | nan | [-0.459, -0.050] | defined |
| `base_degree_p90_l2` | 108 | 1.00 | -0.200 | 0.03939 | nan | [-0.408, +0.012] | defined |
| `base_forman_mean_l2` | 108 | 1.00 | +0.170 | 0.08078 | nan | [-0.048, +0.380] | defined |
| `base_forman_negative_fraction_l2` | 108 | 1.00 | -0.211 | 0.02639 | nan | [-0.392, -0.012] | defined |
| `base_density_l3` | 108 | 1.00 | -0.479 | 0.0002 | nan | [-0.636, -0.282] | defined |
| `base_transitivity_l3` | 108 | 1.00 | -0.438 | 0.0002 | nan | [-0.603, -0.239] | defined |
| `base_triangles_l3` | 108 | 1.00 | -0.420 | 0.0002 | nan | [-0.579, -0.220] | defined |
| `base_cycle_rank_per_node_l3` | 108 | 1.00 | -0.478 | 0.0002 | nan | [-0.636, -0.282] | defined |
| `base_degree_median_l3` | 108 | 1.00 | -0.562 | 0.0002 | nan | [-0.711, -0.383] | defined |
| `base_degree_p90_l3` | 108 | 1.00 | -0.476 | 0.0002 | nan | [-0.635, -0.277] | defined |
| `base_forman_mean_l3` | 108 | 1.00 | +0.420 | 0.0002 | nan | [+0.216, +0.581] | defined |
| `base_forman_negative_fraction_l3` | 108 | 1.00 | -0.218 | 0.0236 | nan | [-0.413, +0.006] | defined |

## Baselines within the 76 curvature_RQ-VAE arms

| candidate | n | coverage | rho | p | Holm p | 95% CI | status |
|---|---|---|---|---|---|---|---|
| `base_density_l1` | 75 | 0.99 | +0.073 | 0.5277 | nan | [-0.205, +0.330] | defined |
| `base_transitivity_l1` | 75 | 0.99 | +0.054 | 0.6355 | nan | [-0.221, +0.315] | defined |
| `base_triangles_l1` | 75 | 0.99 | -0.293 | 0.0114 | nan | [-0.527, -0.036] | defined |
| `base_cycle_rank_per_node_l1` | 76 | 1.00 | -0.244 | 0.03219 | nan | [-0.496, +0.028] | defined |
| `base_degree_median_l1` | 76 | 1.00 | -0.248 | 0.02899 | nan | [-0.497, +0.019] | defined |
| `base_degree_p90_l1` | 76 | 1.00 | -0.264 | 0.0196 | nan | [-0.509, +0.001] | defined |
| `base_forman_mean_l1` | 75 | 0.99 | +0.293 | 0.0116 | nan | [+0.036, +0.527] | defined |
| `base_forman_negative_fraction_l1` | 75 | 0.99 | | | | | UNDEFINED_DEGENERATE |
| `base_density_l2` | 76 | 1.00 | -0.093 | 0.4281 | nan | [-0.354, +0.180] | defined |
| `base_transitivity_l2` | 76 | 1.00 | -0.116 | 0.3327 | nan | [-0.368, +0.152] | defined |
| `base_triangles_l2` | 76 | 1.00 | +0.079 | 0.4919 | nan | [-0.194, +0.337] | defined |
| `base_cycle_rank_per_node_l2` | 76 | 1.00 | +0.110 | 0.3381 | nan | [-0.171, +0.362] | defined |
| `base_degree_median_l2` | 76 | 1.00 | +0.068 | 0.5615 | nan | [-0.225, +0.327] | defined |
| `base_degree_p90_l2` | 76 | 1.00 | +0.110 | 0.3439 | nan | [-0.176, +0.369] | defined |
| `base_forman_mean_l2` | 76 | 1.00 | -0.082 | 0.4777 | nan | [-0.339, +0.189] | defined |
| `base_forman_negative_fraction_l2` | 76 | 1.00 | +0.005 | 0.9686 | nan | [-0.227, +0.242] | defined |
| `base_density_l3` | 76 | 1.00 | -0.158 | 0.1794 | nan | [-0.397, +0.113] | defined |
| `base_transitivity_l3` | 76 | 1.00 | -0.171 | 0.1482 | nan | [-0.416, +0.101] | defined |
| `base_triangles_l3` | 76 | 1.00 | -0.052 | 0.6513 | nan | [-0.310, +0.212] | defined |
| `base_cycle_rank_per_node_l3` | 76 | 1.00 | -0.152 | 0.1992 | nan | [-0.395, +0.119] | defined |
| `base_degree_median_l3` | 76 | 1.00 | -0.215 | 0.06259 | nan | [-0.476, +0.065] | defined |
| `base_degree_p90_l3` | 76 | 1.00 | -0.157 | 0.1868 | nan | [-0.412, +0.127] | defined |
| `base_forman_mean_l3` | 76 | 1.00 | +0.035 | 0.7664 | nan | [-0.230, +0.294] | defined |
| `base_forman_negative_fraction_l3` | 76 | 1.00 | -0.080 | 0.4931 | nan | [-0.328, +0.189] | defined |

## The frozen exploratory family (primary cohort, 108)

## Family A: local-to-global aggregation on the complete quotient graph

| candidate | n | coverage | rho | p | Holm p | 95% CI | status |
|---|---|---|---|---|---|---|---|
| `A_local_transitivity_mean_l1` | 106 | 0.98 | +0.134 | 0.1686 | 1.000 | [-0.097, +0.341] | defined |
| `A_local_transitivity_mean_l2` | 108 | 1.00 | -0.458 | 0.0002 | 0.061 | [-0.608, -0.265] | defined |
| `A_local_transitivity_mean_l3` | 108 | 1.00 | -0.450 | 0.0002 | 0.061 | [-0.615, -0.245] | defined |
| `A_local_transitivity_median_l1` | 106 | 0.98 | +0.048 | 0.6215 | 1.000 | [-0.170, +0.258] | defined |
| `A_local_transitivity_median_l2` | 108 | 1.00 | -0.468 | 0.0002 | 0.061 | [-0.617, -0.279] | defined |
| `A_local_transitivity_median_l3` | 108 | 1.00 | -0.472 | 0.0002 | 0.061 | [-0.625, -0.281] | defined |
| `A_local_transitivity_p10_l1` | 106 | 0.98 | +0.172 | 0.07698 | 1.000 | [-0.060, +0.385] | defined |
| `A_local_transitivity_p10_l2` | 108 | 1.00 | -0.513 | 0.0002 | 0.061 | [-0.669, -0.321] | defined |
| `A_local_transitivity_p10_l3` | 108 | 1.00 | -0.541 | 0.0002 | 0.061 | [-0.644, -0.405] | defined |
| `A_local_transitivity_p90_l1` | 106 | 0.98 | -0.314 | 0.0014 | 0.287 | [-0.497, -0.115] | defined |
| `A_local_transitivity_p90_l2` | 108 | 1.00 | -0.443 | 0.0002 | 0.061 | [-0.599, -0.252] | defined |
| `A_local_transitivity_p90_l3` | 108 | 1.00 | -0.385 | 0.0002 | 0.061 | [-0.561, -0.174] | defined |
| `A_local_transitivity_iqr_l1` | 106 | 0.98 | -0.179 | 0.06639 | 1.000 | [-0.394, +0.046] | defined |
| `A_local_transitivity_iqr_l2` | 108 | 1.00 | -0.191 | 0.04779 | 1.000 | [-0.385, +0.025] | defined |
| `A_local_transitivity_iqr_l3` | 108 | 1.00 | -0.329 | 0.0005999 | 0.137 | [-0.518, -0.115] | defined |
| `A_triangles_per_node_mean_l1` | 108 | 1.00 | +0.210 | 0.02799 | 1.000 | [-0.020, +0.422] | defined |
| `A_triangles_per_node_mean_l2` | 108 | 1.00 | -0.316 | 0.0005999 | 0.137 | [-0.495, -0.116] | defined |
| `A_triangles_per_node_mean_l3` | 108 | 1.00 | -0.494 | 0.0002 | 0.061 | [-0.647, -0.304] | defined |
| `A_ego_density_mean_l1` | 107 | 0.99 | +0.075 | 0.4313 | 1.000 | [-0.145, +0.292] | defined |
| `A_ego_density_mean_l2` | 108 | 1.00 | -0.484 | 0.0002 | 0.061 | [-0.652, -0.289] | defined |
| `A_ego_density_mean_l3` | 108 | 1.00 | -0.264 | 0.007598 | 1.000 | [-0.470, -0.043] | defined |
| `A_ego_density_median_l1` | 107 | 0.99 | +0.016 | 0.8634 | 1.000 | [-0.203, +0.240] | defined |
| `A_ego_density_median_l2` | 108 | 1.00 | -0.483 | 0.0002 | 0.061 | [-0.644, -0.293] | defined |
| `A_ego_density_median_l3` | 108 | 1.00 | -0.135 | 0.1598 | 1.000 | [-0.346, +0.098] | defined |
| `A_ego_cycle_surplus_mean_l1` | 108 | 1.00 | +0.210 | 0.02799 | 1.000 | [-0.020, +0.422] | defined |
| `A_ego_cycle_surplus_mean_l2` | 108 | 1.00 | -0.316 | 0.0005999 | 0.137 | [-0.495, -0.116] | defined |
| `A_ego_cycle_surplus_mean_l3` | 108 | 1.00 | -0.494 | 0.0002 | 0.061 | [-0.647, -0.304] | defined |
| `A_cycle_rank_per_edge_l1` | 107 | 0.99 | +0.174 | 0.07678 | 1.000 | [-0.054, +0.378] | defined |
| `A_cycle_rank_per_edge_l2` | 108 | 1.00 | -0.235 | 0.0136 | 1.000 | [-0.444, -0.023] | defined |
| `A_cycle_rank_per_edge_l3` | 108 | 1.00 | -0.478 | 0.0002 | 0.061 | [-0.636, -0.282] | defined |
| `A_cycle_surplus_per_component_mean_l1` | 107 | 0.99 | +0.122 | 0.2154 | 1.000 | [-0.111, +0.330] | defined |
| `A_cycle_surplus_per_component_mean_l2` | 108 | 1.00 | +0.261 | 0.006999 | 1.000 | [+0.034, +0.460] | defined |
| `A_cycle_surplus_per_component_mean_l3` | 108 | 1.00 | +0.095 | 0.3221 | 1.000 | [-0.151, +0.322] | defined |
| `A_degree_mean_l1` | 108 | 1.00 | +0.194 | 0.04219 | 1.000 | [-0.038, +0.409] | defined |
| `A_degree_mean_l2` | 108 | 1.00 | -0.235 | 0.0136 | 1.000 | [-0.444, -0.023] | defined |
| `A_degree_mean_l3` | 108 | 1.00 | -0.478 | 0.0002 | 0.061 | [-0.636, -0.282] | defined |
| `A_degree_p10_l1` | 108 | 1.00 | +0.322 | 0.0007998 | 0.175 | [+0.094, +0.519] | defined |
| `A_degree_p10_l2` | 108 | 1.00 | -0.040 | 0.6613 | 1.000 | [-0.262, +0.185] | defined |
| `A_degree_p10_l3` | 108 | 1.00 | -0.391 | 0.0002 | 0.061 | [-0.570, -0.176] | defined |
| `A_degree_iqr_l1` | 108 | 1.00 | -0.134 | 0.1648 | 1.000 | [-0.349, +0.107] | defined |
| `A_degree_iqr_l2` | 108 | 1.00 | -0.140 | 0.1514 | 1.000 | [-0.360, +0.087] | defined |
| `A_degree_iqr_l3` | 108 | 1.00 | -0.420 | 0.0002 | 0.061 | [-0.594, -0.215] | defined |
| `A_degree_cv_l1` | 107 | 0.99 | -0.111 | 0.2448 | 1.000 | [-0.345, +0.114] | defined |
| `A_degree_cv_l2` | 108 | 1.00 | +0.259 | 0.007998 | 1.000 | [+0.057, +0.446] | defined |
| `A_degree_cv_l3` | 108 | 1.00 | +0.265 | 0.006999 | 1.000 | [+0.043, +0.457] | defined |
| `A_degree_p90_over_mean_l1` | 107 | 0.99 | -0.164 | 0.08478 | 1.000 | [-0.394, +0.063] | defined |
| `A_degree_p90_over_mean_l2` | 108 | 1.00 | -0.176 | 0.07179 | 1.000 | [-0.377, +0.050] | defined |
| `A_degree_p90_over_mean_l3` | 108 | 1.00 | -0.346 | 0.0003999 | 0.096 | [-0.539, -0.125] | defined |
| `A_degree_median_over_mean_l1` | 107 | 0.99 | +0.202 | 0.03659 | 1.000 | [-0.020, +0.393] | defined |
| `A_degree_median_over_mean_l2` | 108 | 1.00 | +0.196 | 0.04119 | 1.000 | [-0.020, +0.398] | defined |
| `A_degree_median_over_mean_l3` | 108 | 1.00 | +0.049 | 0.6237 | 1.000 | [-0.179, +0.267] | defined |
| `A_forman_p10_l1` | 107 | 0.99 | -0.110 | 0.2639 | 1.000 | [-0.325, +0.126] | defined |
| `A_forman_p10_l2` | 108 | 1.00 | +0.157 | 0.1026 | 1.000 | [-0.062, +0.371] | defined |
| `A_forman_p10_l3` | 108 | 1.00 | +0.363 | 0.0003999 | 0.096 | [+0.151, +0.555] | defined |
| `A_forman_p90_l1` | 107 | 0.99 | -0.213 | 0.02939 | 1.000 | [-0.420, +0.013] | defined |
| `A_forman_p90_l2` | 108 | 1.00 | +0.270 | 0.003999 | 0.760 | [+0.062, +0.465] | defined |
| `A_forman_p90_l3` | 108 | 1.00 | +0.472 | 0.0002 | 0.061 | [+0.271, +0.630] | defined |
| `A_forman_node_incident_mean_l1` | 107 | 0.99 | -0.164 | 0.09358 | 1.000 | [-0.374, +0.060] | defined |
| `A_forman_node_incident_mean_l2` | 108 | 1.00 | +0.179 | 0.06519 | 1.000 | [-0.048, +0.395] | defined |
| `A_forman_node_incident_mean_l3` | 108 | 1.00 | +0.442 | 0.0002 | 0.061 | [+0.242, +0.603] | defined |
| `A_forman_node_incident_median_l1` | 107 | 0.99 | -0.160 | 0.103 | 1.000 | [-0.368, +0.067] | defined |
| `A_forman_node_incident_median_l2` | 108 | 1.00 | +0.199 | 0.03979 | 1.000 | [-0.017, +0.410] | defined |
| `A_forman_node_incident_median_l3` | 108 | 1.00 | +0.480 | 0.0002 | 0.061 | [+0.289, +0.637] | defined |

## Family B: parent-prefix-conditioned sibling quotient geometry (L2/L3 only)

| candidate | n | coverage | rho | p | Holm p | 95% CI | status |
|---|---|---|---|---|---|---|---|
| `B_sibling_density_mean_l2` | 108 | 1.00 | -0.145 | 0.1304 | 1.000 | [-0.358, +0.084] | defined |
| `B_sibling_density_mean_l3` | 87 | 0.81 | +0.332 | 0.002 | 0.404 | [+0.091, +0.560] | defined |
| `B_sibling_density_median_parent_l2` | 108 | 1.00 | -0.134 | 0.1632 | 1.000 | [-0.350, +0.093] | defined |
| `B_sibling_density_median_parent_l3` | 87 | 0.81 | +0.233 | 0.02979 | 1.000 | [+0.001, +0.463] | defined |
| `B_sibling_density_child_weighted_l2` | 108 | 1.00 | -0.156 | 0.1044 | 1.000 | [-0.365, +0.073] | defined |
| `B_sibling_density_child_weighted_l3` | 87 | 0.81 | +0.304 | 0.003799 | 0.726 | [+0.072, +0.525] | defined |
| `B_sibling_density_item_weighted_l2` | 108 | 1.00 | -0.190 | 0.05059 | 1.000 | [-0.397, +0.043] | defined |
| `B_sibling_density_item_weighted_l3` | 87 | 0.81 | +0.163 | 0.1296 | 1.000 | [-0.078, +0.391] | defined |
| `B_sibling_transitivity_mean_l2` | 108 | 1.00 | -0.180 | 0.06259 | 1.000 | [-0.377, +0.044] | defined |
| `B_sibling_transitivity_mean_l3` | 81 | 0.75 | +0.155 | 0.1654 | 1.000 | [-0.103, +0.405] | defined |
| `B_sibling_transitivity_child_weighted_l2` | 108 | 1.00 | -0.235 | 0.0146 | 1.000 | [-0.430, -0.014] | defined |
| `B_sibling_transitivity_child_weighted_l3` | 81 | 0.75 | +0.128 | 0.2529 | 1.000 | [-0.135, +0.375] | defined |
| `B_sibling_cycle_rank_per_node_mean_l2` | 108 | 1.00 | +0.271 | 0.003599 | 0.691 | [+0.051, +0.490] | defined |
| `B_sibling_cycle_rank_per_node_mean_l3` | 108 | 1.00 | -0.081 | 0.3905 | 1.000 | [-0.292, +0.146] | defined |
| `B_sibling_degree_median_mean_l2` | 108 | 1.00 | +0.273 | 0.003399 | 0.659 | [+0.046, +0.482] | defined |
| `B_sibling_degree_median_mean_l3` | 108 | 1.00 | +0.083 | 0.3823 | 1.000 | [-0.137, +0.314] | defined |
| `B_sibling_degree_mean_mean_l2` | 108 | 1.00 | +0.364 | 0.0002 | 0.061 | [+0.132, +0.574] | defined |
| `B_sibling_degree_mean_mean_l3` | 108 | 1.00 | -0.091 | 0.3347 | 1.000 | [-0.304, +0.134] | defined |
| `B_sibling_degree_p90_mean_l2` | 108 | 1.00 | +0.433 | 0.0002 | 0.061 | [+0.215, +0.622] | defined |
| `B_sibling_degree_p90_mean_l3` | 108 | 1.00 | -0.159 | 0.1006 | 1.000 | [-0.359, +0.068] | defined |
| `B_sibling_forman_mean_l2` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_forman_mean_l3` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_forman_negative_fraction_l2` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_forman_negative_fraction_l3` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_edge_share_l2` | 108 | 1.00 | +0.421 | 0.0002 | 0.061 | [+0.185, +0.620] | defined |
| `B_sibling_edge_share_l3` | 108 | 1.00 | -0.022 | 0.8274 | 1.000 | [-0.229, +0.199] | defined |

## Family C: null-relative versions of the same structural measures

| candidate | n | coverage | rho | p | Holm p | 95% CI | status |
|---|---|---|---|---|---|---|---|
| `C_density_global_delta_l1` | 107 | 0.99 | -0.203 | 0.03619 | 1.000 | [-0.400, +0.010] | defined |
| `C_density_global_delta_l2` | 108 | 1.00 | +0.249 | 0.009998 | 1.000 | [+0.030, +0.443] | defined |
| `C_density_global_delta_l3` | 108 | 1.00 | +0.396 | 0.0002 | 0.061 | [+0.184, +0.568] | defined |
| `C_density_global_z_l1` | 100 | 0.93 | -0.116 | 0.2561 | 1.000 | [-0.311, +0.094] | defined |
| `C_density_global_z_l2` | 107 | 0.99 | -0.323 | 0.0005999 | 0.137 | [-0.496, -0.144] | defined |
| `C_density_global_z_l3` | 108 | 1.00 | -0.181 | 0.05679 | 1.000 | [-0.374, +0.026] | defined |
| `C_density_global_ratio_l1` | 107 | 0.99 | -0.033 | 0.7279 | 1.000 | [-0.241, +0.181] | defined |
| `C_density_global_ratio_l2` | 108 | 1.00 | +0.112 | 0.2482 | 1.000 | [-0.105, +0.309] | defined |
| `C_density_global_ratio_l3` | 108 | 1.00 | +0.357 | 0.0003999 | 0.096 | [+0.139, +0.536] | defined |
| `C_density_degree_stratified_delta_l1` | 107 | 0.99 | -0.382 | 0.0003999 | 0.096 | [-0.559, -0.167] | defined |
| `C_density_degree_stratified_delta_l2` | 108 | 1.00 | +0.283 | 0.0024 | 0.473 | [+0.074, +0.472] | defined |
| `C_density_degree_stratified_delta_l3` | 108 | 1.00 | +0.354 | 0.0005999 | 0.137 | [+0.141, +0.526] | defined |
| `C_density_degree_stratified_z_l1` | 93 | 0.86 | -0.489 | 0.0002 | 0.061 | [-0.630, -0.307] | defined |
| `C_density_degree_stratified_z_l2` | 107 | 0.99 | -0.498 | 0.0002 | 0.061 | [-0.646, -0.316] | defined |
| `C_density_degree_stratified_z_l3` | 108 | 1.00 | -0.277 | 0.004799 | 0.902 | [-0.456, -0.079] | defined |
| `C_density_degree_stratified_ratio_l1` | 107 | 0.99 | -0.276 | 0.003199 | 0.624 | [-0.460, -0.057] | defined |
| `C_density_degree_stratified_ratio_l2` | 108 | 1.00 | +0.019 | 0.8536 | 1.000 | [-0.189, +0.221] | defined |
| `C_density_degree_stratified_ratio_l3` | 108 | 1.00 | +0.314 | 0.0005999 | 0.137 | [+0.104, +0.485] | defined |
| `C_density_parent_conditioned_delta_l2` | 108 | 1.00 | +0.193 | 0.04459 | 1.000 | [-0.032, +0.398] | defined |
| `C_density_parent_conditioned_delta_l3` | 108 | 1.00 | +0.187 | 0.05379 | 1.000 | [-0.054, +0.404] | defined |
| `C_density_parent_conditioned_z_l2` | 107 | 0.99 | -0.204 | 0.03179 | 1.000 | [-0.400, +0.002] | defined |
| `C_density_parent_conditioned_z_l3` | 103 | 0.95 | -0.046 | 0.6497 | 1.000 | [-0.270, +0.170] | defined |
| `C_density_parent_conditioned_ratio_l2` | 108 | 1.00 | +0.051 | 0.6013 | 1.000 | [-0.164, +0.256] | defined |
| `C_density_parent_conditioned_ratio_l3` | 108 | 1.00 | +0.163 | 0.09018 | 1.000 | [-0.074, +0.381] | defined |
| `C_transitivity_global_delta_l1` | 106 | 0.98 | -0.234 | 0.0176 | 1.000 | [-0.434, -0.014] | defined |
| `C_transitivity_global_delta_l2` | 108 | 1.00 | +0.090 | 0.3471 | 1.000 | [-0.121, +0.297] | defined |
| `C_transitivity_global_delta_l3` | 108 | 1.00 | -0.125 | 0.2052 | 1.000 | [-0.345, +0.099] | defined |
| `C_transitivity_global_z_l1` | 100 | 0.93 | -0.129 | 0.198 | 1.000 | [-0.324, +0.090] | defined |
| `C_transitivity_global_z_l2` | 107 | 0.99 | +0.192 | 0.04599 | 1.000 | [-0.025, +0.405] | defined |
| `C_transitivity_global_z_l3` | 108 | 1.00 | +0.151 | 0.1164 | 1.000 | [-0.077, +0.355] | defined |
| `C_transitivity_global_ratio_l1` | 106 | 0.98 | -0.159 | 0.103 | 1.000 | [-0.365, +0.058] | defined |
| `C_transitivity_global_ratio_l2` | 108 | 1.00 | +0.158 | 0.105 | 1.000 | [-0.034, +0.349] | defined |
| `C_transitivity_global_ratio_l3` | 108 | 1.00 | -0.100 | 0.3087 | 1.000 | [-0.324, +0.118] | defined |
| `C_transitivity_degree_stratified_delta_l1` | 106 | 0.98 | -0.354 | 0.0003999 | 0.096 | [-0.535, -0.136] | defined |
| `C_transitivity_degree_stratified_delta_l2` | 108 | 1.00 | +0.172 | 0.07359 | 1.000 | [-0.044, +0.370] | defined |
| `C_transitivity_degree_stratified_delta_l3` | 108 | 1.00 | -0.038 | 0.6907 | 1.000 | [-0.250, +0.187] | defined |
| `C_transitivity_degree_stratified_z_l1` | 93 | 0.86 | -0.442 | 0.0002 | 0.061 | [-0.595, -0.248] | defined |
| `C_transitivity_degree_stratified_z_l2` | 107 | 0.99 | +0.270 | 0.002 | 0.404 | [+0.070, +0.467] | defined |
| `C_transitivity_degree_stratified_z_l3` | 108 | 1.00 | +0.196 | 0.04839 | 1.000 | [-0.006, +0.378] | defined |
| `C_transitivity_degree_stratified_ratio_l1` | 106 | 0.98 | -0.343 | 0.0005999 | 0.137 | [-0.531, -0.126] | defined |
| `C_transitivity_degree_stratified_ratio_l2` | 108 | 1.00 | +0.273 | 0.004399 | 0.831 | [+0.080, +0.446] | defined |
| `C_transitivity_degree_stratified_ratio_l3` | 108 | 1.00 | -0.025 | 0.799 | 1.000 | [-0.238, +0.193] | defined |
| `C_transitivity_parent_conditioned_delta_l2` | 108 | 1.00 | +0.211 | 0.03239 | 1.000 | [+0.010, +0.402] | defined |
| `C_transitivity_parent_conditioned_delta_l3` | 108 | 1.00 | +0.157 | 0.1038 | 1.000 | [-0.062, +0.378] | defined |
| `C_transitivity_parent_conditioned_z_l2` | 107 | 0.99 | +0.202 | 0.03399 | 1.000 | [-0.002, +0.422] | defined |
| `C_transitivity_parent_conditioned_z_l3` | 101 | 0.94 | +0.305 | 0.0022 | 0.436 | [+0.095, +0.500] | defined |
| `C_transitivity_parent_conditioned_ratio_l2` | 108 | 1.00 | +0.204 | 0.03739 | 1.000 | [+0.003, +0.400] | defined |
| `C_transitivity_parent_conditioned_ratio_l3` | 108 | 1.00 | +0.186 | 0.05599 | 1.000 | [-0.030, +0.400] | defined |
| `C_cycle_rank_per_node_global_delta_l1` | 108 | 1.00 | -0.249 | 0.0102 | 1.000 | [-0.435, -0.045] | defined |
| `C_cycle_rank_per_node_global_delta_l2` | 108 | 1.00 | +0.142 | 0.1488 | 1.000 | [-0.076, +0.342] | defined |
| `C_cycle_rank_per_node_global_delta_l3` | 108 | 1.00 | +0.380 | 0.0002 | 0.061 | [+0.162, +0.554] | defined |
| `C_cycle_rank_per_node_global_z_l1` | 103 | 0.95 | -0.138 | 0.1606 | 1.000 | [-0.352, +0.090] | defined |
| `C_cycle_rank_per_node_global_z_l2` | 108 | 1.00 | -0.342 | 0.0012 | 0.251 | [-0.512, -0.150] | defined |
| `C_cycle_rank_per_node_global_z_l3` | 108 | 1.00 | -0.181 | 0.05679 | 1.000 | [-0.374, +0.026] | defined |
| `C_cycle_rank_per_node_global_ratio_l1` | 106 | 0.98 | -0.008 | 0.9326 | 1.000 | [-0.217, +0.206] | defined |
| `C_cycle_rank_per_node_global_ratio_l2` | 108 | 1.00 | +0.110 | 0.2545 | 1.000 | [-0.107, +0.308] | defined |
| `C_cycle_rank_per_node_global_ratio_l3` | 108 | 1.00 | +0.351 | 0.0003999 | 0.096 | [+0.133, +0.532] | defined |
| `C_cycle_rank_per_node_degree_stratified_delta_l1` | 108 | 1.00 | -0.419 | 0.0002 | 0.061 | [-0.586, -0.223] | defined |
| `C_cycle_rank_per_node_degree_stratified_delta_l2` | 108 | 1.00 | +0.092 | 0.3409 | 1.000 | [-0.120, +0.289] | defined |
| `C_cycle_rank_per_node_degree_stratified_delta_l3` | 108 | 1.00 | +0.343 | 0.0005999 | 0.137 | [+0.130, +0.513] | defined |
| `C_cycle_rank_per_node_degree_stratified_z_l1` | 100 | 0.93 | -0.501 | 0.0002 | 0.061 | [-0.636, -0.319] | defined |
| `C_cycle_rank_per_node_degree_stratified_z_l2` | 108 | 1.00 | -0.512 | 0.0002 | 0.061 | [-0.660, -0.331] | defined |
| `C_cycle_rank_per_node_degree_stratified_z_l3` | 108 | 1.00 | -0.277 | 0.004799 | 0.902 | [-0.456, -0.079] | defined |
| `C_cycle_rank_per_node_degree_stratified_ratio_l1` | 106 | 0.98 | -0.252 | 0.008398 | 1.000 | [-0.436, -0.046] | defined |
| `C_cycle_rank_per_node_degree_stratified_ratio_l2` | 108 | 1.00 | +0.024 | 0.8082 | 1.000 | [-0.185, +0.223] | defined |
| `C_cycle_rank_per_node_degree_stratified_ratio_l3` | 108 | 1.00 | +0.311 | 0.0007998 | 0.175 | [+0.102, +0.481] | defined |
| `C_cycle_rank_per_node_parent_conditioned_delta_l2` | 108 | 1.00 | +0.043 | 0.6579 | 1.000 | [-0.178, +0.251] | defined |
| `C_cycle_rank_per_node_parent_conditioned_delta_l3` | 108 | 1.00 | +0.187 | 0.05519 | 1.000 | [-0.053, +0.407] | defined |
| `C_cycle_rank_per_node_parent_conditioned_z_l2` | 108 | 1.00 | -0.226 | 0.0204 | 1.000 | [-0.418, -0.018] | defined |
| `C_cycle_rank_per_node_parent_conditioned_z_l3` | 100 | 0.93 | -0.040 | 0.6993 | 1.000 | [-0.270, +0.176] | defined |
| `C_cycle_rank_per_node_parent_conditioned_ratio_l2` | 108 | 1.00 | +0.045 | 0.6383 | 1.000 | [-0.170, +0.255] | defined |
| `C_cycle_rank_per_node_parent_conditioned_ratio_l3` | 108 | 1.00 | +0.170 | 0.07499 | 1.000 | [-0.068, +0.388] | defined |
| `C_degree_median_global_delta_l1` | 108 | 1.00 | -0.130 | 0.1738 | 1.000 | [-0.327, +0.070] | defined |
| `C_degree_median_global_delta_l2` | 108 | 1.00 | +0.364 | 0.0003999 | 0.096 | [+0.164, +0.527] | defined |
| `C_degree_median_global_delta_l3` | 108 | 1.00 | +0.380 | 0.0002 | 0.061 | [+0.169, +0.557] | defined |
| `C_degree_median_global_z_l1` | 50 | 0.46 | -0.480 | 0.0005999 | 0.137 | [-0.717, -0.201] | defined |
| `C_degree_median_global_z_l2` | 56 | 0.52 | +0.421 | 0.0007998 | 0.175 | [+0.168, +0.624] | defined |
| `C_degree_median_global_z_l3` | 25 | 0.23 | +0.391 | 0.04959 | 1.000 | [-0.022, +0.747] | defined |
| `C_degree_median_global_ratio_l1` | 107 | 0.99 | +0.027 | 0.7776 | 1.000 | [-0.182, +0.249] | defined |
| `C_degree_median_global_ratio_l2` | 108 | 1.00 | +0.339 | 0.0009998 | 0.211 | [+0.143, +0.510] | defined |
| `C_degree_median_global_ratio_l3` | 108 | 1.00 | +0.373 | 0.0002 | 0.061 | [+0.165, +0.548] | defined |
| `C_degree_median_degree_stratified_delta_l1` | 108 | 1.00 | -0.318 | 0.0012 | 0.251 | [-0.497, -0.120] | defined |
| `C_degree_median_degree_stratified_delta_l2` | 108 | 1.00 | +0.050 | 0.5971 | 1.000 | [-0.167, +0.255] | defined |
| `C_degree_median_degree_stratified_delta_l3` | 108 | 1.00 | +0.328 | 0.0002 | 0.061 | [+0.073, +0.521] | defined |
| `C_degree_median_degree_stratified_z_l1` | 53 | 0.49 | -0.645 | 0.0002 | 0.061 | [-0.766, -0.465] | defined |
| `C_degree_median_degree_stratified_z_l2` | 52 | 0.48 | -0.105 | 0.4531 | 1.000 | [-0.419, +0.205] | defined |
| `C_degree_median_degree_stratified_z_l3` | 12 | 0.11 | +0.081 | 0.7998 | 1.000 | [-0.613, +0.660] | defined |
| `C_degree_median_degree_stratified_ratio_l1` | 107 | 0.99 | -0.084 | 0.3845 | 1.000 | [-0.282, +0.145] | defined |
| `C_degree_median_degree_stratified_ratio_l2` | 108 | 1.00 | +0.042 | 0.6563 | 1.000 | [-0.174, +0.245] | defined |
| `C_degree_median_degree_stratified_ratio_l3` | 108 | 1.00 | +0.325 | 0.0002 | 0.061 | [+0.073, +0.514] | defined |
| `C_degree_median_parent_conditioned_delta_l2` | 108 | 1.00 | +0.053 | 0.5915 | 1.000 | [-0.157, +0.258] | defined |
| `C_degree_median_parent_conditioned_delta_l3` | 108 | 1.00 | +0.259 | 0.005599 | 1.000 | [+0.027, +0.464] | defined |
| `C_degree_median_parent_conditioned_z_l2` | 58 | 0.54 | +0.020 | 0.895 | 1.000 | [-0.257, +0.309] | defined |
| `C_degree_median_parent_conditioned_z_l3` | 14 | 0.13 | -0.037 | 0.9038 | 1.000 | [-0.747, +0.596] | defined |
| `C_degree_median_parent_conditioned_ratio_l2` | 108 | 1.00 | -0.001 | 0.994 | 1.000 | [-0.199, +0.196] | defined |
| `C_degree_median_parent_conditioned_ratio_l3` | 108 | 1.00 | +0.255 | 0.006799 | 1.000 | [+0.027, +0.458] | defined |
| `C_forman_mean_global_delta_l1` | 107 | 0.99 | +0.247 | 0.008998 | 1.000 | [+0.028, +0.439] | defined |
| `C_forman_mean_global_delta_l2` | 108 | 1.00 | +0.019 | 0.8412 | 1.000 | [-0.186, +0.225] | defined |
| `C_forman_mean_global_delta_l3` | 108 | 1.00 | -0.037 | 0.7009 | 1.000 | [-0.250, +0.191] | defined |
| `C_forman_mean_global_z_l1` | 100 | 0.93 | +0.130 | 0.1948 | 1.000 | [-0.085, +0.325] | defined |
| `C_forman_mean_global_z_l2` | 107 | 0.99 | -0.028 | 0.7754 | 1.000 | [-0.237, +0.190] | defined |
| `C_forman_mean_global_z_l3` | 108 | 1.00 | +0.003 | 0.9738 | 1.000 | [-0.207, +0.232] | defined |
| `C_forman_mean_global_ratio_l1` | 107 | 0.99 | -0.097 | 0.3177 | 1.000 | [-0.300, +0.117] | defined |
| `C_forman_mean_global_ratio_l2` | 108 | 1.00 | -0.041 | 0.6701 | 1.000 | [-0.245, +0.166] | defined |
| `C_forman_mean_global_ratio_l3` | 108 | 1.00 | +0.026 | 0.795 | 1.000 | [-0.200, +0.236] | defined |
| `C_forman_mean_degree_stratified_delta_l1` | 107 | 0.99 | +0.403 | 0.0002 | 0.061 | [+0.186, +0.578] | defined |
| `C_forman_mean_degree_stratified_delta_l2` | 108 | 1.00 | -0.089 | 0.3557 | 1.000 | [-0.286, +0.115] | defined |
| `C_forman_mean_degree_stratified_delta_l3` | 108 | 1.00 | -0.379 | 0.0003999 | 0.096 | [-0.555, -0.174] | defined |
| `C_forman_mean_degree_stratified_z_l1` | 94 | 0.87 | +0.446 | 0.0002 | 0.061 | [+0.253, +0.607] | defined |
| `C_forman_mean_degree_stratified_z_l2` | 107 | 0.99 | -0.067 | 0.5001 | 1.000 | [-0.270, +0.141] | defined |
| `C_forman_mean_degree_stratified_z_l3` | 108 | 1.00 | -0.183 | 0.05359 | 1.000 | [-0.366, +0.034] | defined |
| `C_forman_mean_degree_stratified_ratio_l1` | 107 | 0.99 | -0.331 | 0.0007998 | 0.175 | [-0.517, -0.110] | defined |
| `C_forman_mean_degree_stratified_ratio_l2` | 108 | 1.00 | +0.054 | 0.5837 | 1.000 | [-0.148, +0.249] | defined |
| `C_forman_mean_degree_stratified_ratio_l3` | 108 | 1.00 | +0.373 | 0.0003999 | 0.096 | [+0.169, +0.550] | defined |
| `C_forman_mean_parent_conditioned_delta_l2` | 108 | 1.00 | -0.214 | 0.02539 | 1.000 | [-0.419, +0.009] | defined |
| `C_forman_mean_parent_conditioned_delta_l3` | 108 | 1.00 | -0.145 | 0.1332 | 1.000 | [-0.369, +0.081] | defined |
| `C_forman_mean_parent_conditioned_z_l2` | 107 | 0.99 | -0.214 | 0.02679 | 1.000 | [-0.426, -0.001] | defined |
| `C_forman_mean_parent_conditioned_z_l3` | 98 | 0.91 | -0.243 | 0.0182 | 1.000 | [-0.442, -0.033] | defined |
| `C_forman_mean_parent_conditioned_ratio_l2` | 108 | 1.00 | +0.220 | 0.0208 | 1.000 | [-0.003, +0.429] | defined |
| `C_forman_mean_parent_conditioned_ratio_l3` | 108 | 1.00 | +0.153 | 0.1152 | 1.000 | [-0.071, +0.373] | defined |
| `C_triangles_per_node_mean_global_delta_l1` | 108 | 1.00 | -0.327 | 0.0007998 | 0.175 | [-0.511, -0.123] | defined |
| `C_triangles_per_node_mean_global_delta_l2` | 108 | 1.00 | +0.162 | 0.08978 | 1.000 | [-0.051, +0.358] | defined |
| `C_triangles_per_node_mean_global_delta_l3` | 108 | 1.00 | +0.015 | 0.8744 | 1.000 | [-0.230, +0.245] | defined |
| `C_triangles_per_node_mean_global_z_l1` | 100 | 0.93 | -0.125 | 0.2174 | 1.000 | [-0.322, +0.090] | defined |
| `C_triangles_per_node_mean_global_z_l2` | 107 | 0.99 | +0.094 | 0.3413 | 1.000 | [-0.116, +0.307] | defined |
| `C_triangles_per_node_mean_global_z_l3` | 108 | 1.00 | +0.118 | 0.2282 | 1.000 | [-0.114, +0.328] | defined |
| `C_triangles_per_node_mean_global_ratio_l1` | 106 | 0.98 | -0.068 | 0.4841 | 1.000 | [-0.272, +0.141] | defined |
| `C_triangles_per_node_mean_global_ratio_l2` | 108 | 1.00 | +0.106 | 0.2679 | 1.000 | [-0.105, +0.305] | defined |
| `C_triangles_per_node_mean_global_ratio_l3` | 108 | 1.00 | +0.008 | 0.9334 | 1.000 | [-0.231, +0.232] | defined |
| `C_triangles_per_node_mean_degree_stratified_delta_l1` | 108 | 1.00 | -0.420 | 0.0002 | 0.061 | [-0.583, -0.223] | defined |
| `C_triangles_per_node_mean_degree_stratified_delta_l2` | 108 | 1.00 | +0.174 | 0.07239 | 1.000 | [-0.034, +0.363] | defined |
| `C_triangles_per_node_mean_degree_stratified_delta_l3` | 108 | 1.00 | +0.120 | 0.2112 | 1.000 | [-0.106, +0.333] | defined |
| `C_triangles_per_node_mean_degree_stratified_z_l1` | 93 | 0.86 | -0.466 | 0.0002 | 0.061 | [-0.612, -0.270] | defined |
| `C_triangles_per_node_mean_degree_stratified_z_l2` | 107 | 0.99 | +0.153 | 0.1166 | 1.000 | [-0.068, +0.356] | defined |
| `C_triangles_per_node_mean_degree_stratified_z_l3` | 108 | 1.00 | +0.157 | 0.1036 | 1.000 | [-0.078, +0.364] | defined |
| `C_triangles_per_node_mean_degree_stratified_ratio_l1` | 106 | 0.98 | -0.314 | 0.0012 | 0.251 | [-0.500, -0.099] | defined |
| `C_triangles_per_node_mean_degree_stratified_ratio_l2` | 108 | 1.00 | +0.140 | 0.1432 | 1.000 | [-0.073, +0.329] | defined |
| `C_triangles_per_node_mean_degree_stratified_ratio_l3` | 108 | 1.00 | +0.134 | 0.1688 | 1.000 | [-0.089, +0.340] | defined |
| `C_triangles_per_node_mean_parent_conditioned_delta_l2` | 108 | 1.00 | +0.220 | 0.0202 | 1.000 | [+0.016, +0.418] | defined |
| `C_triangles_per_node_mean_parent_conditioned_delta_l3` | 108 | 1.00 | +0.308 | 0.0012 | 0.251 | [+0.084, +0.521] | defined |
| `C_triangles_per_node_mean_parent_conditioned_z_l2` | 107 | 0.99 | +0.221 | 0.0214 | 1.000 | [+0.012, +0.446] | defined |
| `C_triangles_per_node_mean_parent_conditioned_z_l3` | 98 | 0.91 | +0.331 | 0.0003999 | 0.096 | [+0.112, +0.522] | defined |
| `C_triangles_per_node_mean_parent_conditioned_ratio_l2` | 108 | 1.00 | +0.218 | 0.023 | 1.000 | [+0.005, +0.415] | defined |
| `C_triangles_per_node_mean_parent_conditioned_ratio_l3` | 108 | 1.00 | +0.319 | 0.0007998 | 0.175 | [+0.099, +0.527] | defined |
| `C_local_transitivity_mean_global_delta_l1` | 106 | 0.98 | -0.306 | 0.002 | 0.404 | [-0.494, -0.093] | defined |
| `C_local_transitivity_mean_global_delta_l2` | 108 | 1.00 | -0.005 | 0.961 | 1.000 | [-0.206, +0.194] | defined |
| `C_local_transitivity_mean_global_delta_l3` | 108 | 1.00 | -0.055 | 0.5729 | 1.000 | [-0.279, +0.178] | defined |
| `C_local_transitivity_mean_global_z_l1` | 100 | 0.93 | -0.133 | 0.1858 | 1.000 | [-0.329, +0.084] | defined |
| `C_local_transitivity_mean_global_z_l2` | 107 | 0.99 | +0.285 | 0.003399 | 0.659 | [+0.081, +0.479] | defined |
| `C_local_transitivity_mean_global_z_l3` | 108 | 1.00 | +0.106 | 0.2779 | 1.000 | [-0.103, +0.315] | defined |
| `C_local_transitivity_mean_global_ratio_l1` | 106 | 0.98 | -0.294 | 0.002799 | 0.549 | [-0.484, -0.078] | defined |
| `C_local_transitivity_mean_global_ratio_l2` | 108 | 1.00 | +0.242 | 0.0142 | 1.000 | [+0.037, +0.430] | defined |
| `C_local_transitivity_mean_global_ratio_l3` | 108 | 1.00 | -0.053 | 0.5829 | 1.000 | [-0.270, +0.177] | defined |
| `C_local_transitivity_mean_degree_stratified_delta_l1` | 106 | 0.98 | -0.393 | 0.0002 | 0.061 | [-0.571, -0.187] | defined |
| `C_local_transitivity_mean_degree_stratified_delta_l2` | 108 | 1.00 | +0.325 | 0.0009998 | 0.211 | [+0.155, +0.484] | defined |
| `C_local_transitivity_mean_degree_stratified_delta_l3` | 108 | 1.00 | -0.025 | 0.7894 | 1.000 | [-0.252, +0.196] | defined |
| `C_local_transitivity_mean_degree_stratified_z_l1` | 94 | 0.87 | -0.421 | 0.0002 | 0.061 | [-0.581, -0.222] | defined |
| `C_local_transitivity_mean_degree_stratified_z_l2` | 107 | 0.99 | +0.446 | 0.0002 | 0.061 | [+0.281, +0.597] | defined |
| `C_local_transitivity_mean_degree_stratified_z_l3` | 108 | 1.00 | +0.228 | 0.0196 | 1.000 | [+0.025, +0.422] | defined |
| `C_local_transitivity_mean_degree_stratified_ratio_l1` | 106 | 0.98 | -0.382 | 0.0002 | 0.061 | [-0.560, -0.174] | defined |
| `C_local_transitivity_mean_degree_stratified_ratio_l2` | 108 | 1.00 | +0.450 | 0.0002 | 0.061 | [+0.287, +0.599] | defined |
| `C_local_transitivity_mean_degree_stratified_ratio_l3` | 108 | 1.00 | +0.015 | 0.867 | 1.000 | [-0.210, +0.234] | defined |
| `C_local_transitivity_mean_parent_conditioned_delta_l2` | 108 | 1.00 | -0.006 | 0.95 | 1.000 | [-0.223, +0.205] | defined |
| `C_local_transitivity_mean_parent_conditioned_delta_l3` | 108 | 1.00 | -0.136 | 0.1602 | 1.000 | [-0.352, +0.108] | defined |
| `C_local_transitivity_mean_parent_conditioned_z_l2` | 107 | 0.99 | -0.004 | 0.9748 | 1.000 | [-0.204, +0.206] | defined |
| `C_local_transitivity_mean_parent_conditioned_z_l3` | 102 | 0.94 | -0.027 | 0.7776 | 1.000 | [-0.258, +0.198] | defined |
| `C_local_transitivity_mean_parent_conditioned_ratio_l2` | 108 | 1.00 | +0.021 | 0.8246 | 1.000 | [-0.186, +0.234] | defined |
| `C_local_transitivity_mean_parent_conditioned_ratio_l3` | 108 | 1.00 | -0.122 | 0.2138 | 1.000 | [-0.339, +0.116] | defined |
| `C_sibling_density_mean_global_delta_l2` | 108 | 1.00 | +0.228 | 0.0174 | 1.000 | [+0.019, +0.444] | defined |
| `C_sibling_density_mean_global_delta_l3` | 81 | 0.75 | +0.667 | 0.0002 | 0.061 | [+0.480, +0.804] | defined |
| `C_sibling_density_mean_global_z_l2` | 106 | 0.98 | +0.688 | 0.0002 | 0.061 | [+0.556, +0.778] | defined |
| `C_sibling_density_mean_global_z_l3` | 80 | 0.74 | -0.067 | 0.5533 | 1.000 | [-0.314, +0.189] | defined |
| `C_sibling_density_mean_global_ratio_l2` | 108 | 1.00 | +0.723 | 0.0002 | 0.061 | [+0.576, +0.825] | defined |
| `C_sibling_density_mean_global_ratio_l3` | 81 | 0.75 | +0.178 | 0.1122 | 1.000 | [-0.064, +0.395] | defined |
| `C_sibling_density_mean_degree_stratified_delta_l2` | 108 | 1.00 | +0.259 | 0.007199 | 1.000 | [+0.049, +0.472] | defined |
| `C_sibling_density_mean_degree_stratified_delta_l3` | 82 | 0.76 | +0.477 | 0.0002 | 0.061 | [+0.270, +0.649] | defined |
| `C_sibling_density_mean_degree_stratified_z_l2` | 106 | 0.98 | +0.618 | 0.0002 | 0.061 | [+0.473, +0.728] | defined |
| `C_sibling_density_mean_degree_stratified_z_l3` | 81 | 0.75 | -0.165 | 0.1338 | 1.000 | [-0.405, +0.091] | defined |
| `C_sibling_density_mean_degree_stratified_ratio_l2` | 108 | 1.00 | +0.722 | 0.0002 | 0.061 | [+0.574, +0.821] | defined |
| `C_sibling_density_mean_degree_stratified_ratio_l3` | 82 | 0.76 | +0.046 | 0.6791 | 1.000 | [-0.189, +0.282] | defined |
| `C_sibling_density_mean_parent_conditioned_delta_l2` | 108 | 1.00 | +0.023 | 0.8114 | 1.000 | [-0.203, +0.240] | defined |
| `C_sibling_density_mean_parent_conditioned_delta_l3` | 87 | 0.81 | +0.080 | 0.4523 | 1.000 | [-0.149, +0.315] | defined |
| `C_sibling_density_mean_parent_conditioned_z_l2` | 106 | 0.98 | -0.346 | 0.0003999 | 0.096 | [-0.501, -0.173] | defined |
| `C_sibling_density_mean_parent_conditioned_z_l3` | 82 | 0.76 | -0.288 | 0.008398 | 1.000 | [-0.487, -0.062] | defined |
| `C_sibling_density_mean_parent_conditioned_ratio_l2` | 108 | 1.00 | -0.116 | 0.2326 | 1.000 | [-0.311, +0.101] | defined |
| `C_sibling_density_mean_parent_conditioned_ratio_l3` | 87 | 0.81 | +0.255 | 0.017 | 1.000 | [+0.019, +0.482] | defined |
| `C_sibling_transitivity_mean_global_delta_l2` | 108 | 1.00 | +0.467 | 0.0002 | 0.061 | [+0.289, +0.631] | defined |
| `C_sibling_transitivity_mean_global_delta_l3` | 34 | 0.31 | +0.592 | 0.0003999 | 0.096 | [+0.325, +0.757] | defined |
| `C_sibling_transitivity_mean_global_z_l2` | 106 | 0.98 | +0.655 | 0.0002 | 0.061 | [+0.496, +0.776] | defined |
| `C_sibling_transitivity_mean_global_z_l3` | 34 | 0.31 | +0.373 | 0.03139 | 1.000 | [+0.059, +0.609] | defined |
| `C_sibling_transitivity_mean_global_ratio_l2` | 108 | 1.00 | +0.666 | 0.0002 | 0.061 | [+0.512, +0.777] | defined |
| `C_sibling_transitivity_mean_global_ratio_l3` | 34 | 0.31 | +0.651 | 0.0002 | 0.061 | [+0.389, +0.806] | defined |
| `C_sibling_transitivity_mean_degree_stratified_delta_l2` | 108 | 1.00 | +0.419 | 0.0002 | 0.061 | [+0.234, +0.591] | defined |
| `C_sibling_transitivity_mean_degree_stratified_delta_l3` | 32 | 0.30 | +0.550 | 0.002 | 0.404 | [+0.231, +0.745] | defined |
| `C_sibling_transitivity_mean_degree_stratified_z_l2` | 106 | 0.98 | +0.505 | 0.0002 | 0.061 | [+0.324, +0.656] | defined |
| `C_sibling_transitivity_mean_degree_stratified_z_l3` | 32 | 0.30 | +0.341 | 0.05399 | 1.000 | [-0.049, +0.652] | defined |
| `C_sibling_transitivity_mean_degree_stratified_ratio_l2` | 108 | 1.00 | +0.593 | 0.0002 | 0.061 | [+0.424, +0.721] | defined |
| `C_sibling_transitivity_mean_degree_stratified_ratio_l3` | 32 | 0.30 | +0.735 | 0.0002 | 0.061 | [+0.532, +0.847] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_delta_l2` | 108 | 1.00 | -0.250 | 0.009998 | 1.000 | [-0.447, -0.041] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_delta_l3` | 81 | 0.75 | -0.174 | 0.1224 | 1.000 | [-0.391, +0.050] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_z_l2` | 106 | 0.98 | -0.349 | 0.0007998 | 0.175 | [-0.524, -0.133] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_z_l3` | 80 | 0.74 | -0.210 | 0.06059 | 1.000 | [-0.424, +0.018] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_ratio_l2` | 108 | 1.00 | -0.320 | 0.0014 | 0.287 | [-0.495, -0.130] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_ratio_l3` | 80 | 0.74 | -0.148 | 0.181 | 1.000 | [-0.356, +0.062] | defined |

## Family D: SECONDARY sensitivity: multi-item-prefix-only induced subgraph

| candidate | n | coverage | rho | p | Holm p | 95% CI | status |
|---|---|---|---|---|---|---|---|
| `D_multi_only_density_l1` | 107 | 0.99 | +0.140 | 0.1408 | 1.000 | [-0.088, +0.361] | defined |
| `D_multi_only_density_l2` | 108 | 1.00 | -0.487 | 0.0002 | 0.061 | [-0.651, -0.294] | defined |
| `D_multi_only_density_l3` | 108 | 1.00 | -0.427 | 0.0002 | 0.061 | [-0.600, -0.226] | defined |
| `D_multi_only_transitivity_l1` | 106 | 0.98 | +0.179 | 0.06479 | 1.000 | [-0.056, +0.390] | defined |
| `D_multi_only_transitivity_l2` | 84 | 0.78 | -0.583 | 0.0002 | 0.061 | [-0.750, -0.370] | defined |
| `D_multi_only_transitivity_l3` | 81 | 0.75 | -0.149 | 0.1786 | 1.000 | [-0.389, +0.107] | defined |
| `D_multi_only_cycle_rank_per_node_l1` | 108 | 1.00 | +0.198 | 0.03799 | 1.000 | [-0.034, +0.412] | defined |
| `D_multi_only_cycle_rank_per_node_l2` | 108 | 1.00 | -0.308 | 0.0005999 | 0.137 | [-0.490, -0.108] | defined |
| `D_multi_only_cycle_rank_per_node_l3` | 108 | 1.00 | -0.497 | 0.0002 | 0.061 | [-0.648, -0.304] | defined |
| `D_multi_only_degree_median_l1` | 108 | 1.00 | +0.191 | 0.04539 | 1.000 | [-0.040, +0.407] | defined |
| `D_multi_only_degree_median_l2` | 108 | 1.00 | -0.309 | 0.0007998 | 0.175 | [-0.489, -0.115] | defined |
| `D_multi_only_degree_median_l3` | 108 | 1.00 | -0.476 | 0.0002 | 0.061 | [-0.633, -0.277] | defined |
| `D_multi_only_forman_mean_l1` | 107 | 0.99 | -0.161 | 0.1018 | 1.000 | [-0.371, +0.066] | defined |
| `D_multi_only_forman_mean_l2` | 108 | 1.00 | +0.289 | 0.0016 | 0.325 | [+0.092, +0.467] | defined |
| `D_multi_only_forman_mean_l3` | 108 | 1.00 | +0.521 | 0.0002 | 0.061 | [+0.326, +0.670] | defined |
| `D_multi_only_forman_negative_fraction_l1` | 107 | 0.99 | | | | | UNDEFINED_DEGENERATE |
| `D_multi_only_forman_negative_fraction_l2` | 108 | 1.00 | -0.067 | 0.4959 | 1.000 | [-0.218, +0.091] | defined |
| `D_multi_only_forman_negative_fraction_l3` | 108 | 1.00 | -0.467 | 0.0002 | 0.061 | [-0.628, -0.272] | defined |

## Within the 76-arm curvature_RQ-VAE cohort (stability diagnostic)

## Same candidates, 76 rows

| candidate | n | coverage | rho | p | Holm p | 95% CI | status |
|---|---|---|---|---|---|---|---|
| `A_local_transitivity_mean_l1` | 75 | 0.99 | +0.044 | 0.6983 | nan | [-0.233, +0.306] | defined |
| `A_local_transitivity_mean_l2` | 76 | 1.00 | -0.082 | 0.4781 | nan | [-0.332, +0.180] | defined |
| `A_local_transitivity_mean_l3` | 76 | 1.00 | -0.124 | 0.3039 | nan | [-0.383, +0.149] | defined |
| `A_local_transitivity_median_l1` | 75 | 0.99 | +0.054 | 0.6367 | nan | [-0.219, +0.315] | defined |
| `A_local_transitivity_median_l2` | 76 | 1.00 | -0.107 | 0.3635 | nan | [-0.362, +0.155] | defined |
| `A_local_transitivity_median_l3` | 76 | 1.00 | -0.138 | 0.2507 | nan | [-0.384, +0.128] | defined |
| `A_local_transitivity_p10_l1` | 75 | 0.99 | +0.057 | 0.6123 | nan | [-0.220, +0.316] | defined |
| `A_local_transitivity_p10_l2` | 76 | 1.00 | -0.078 | 0.5011 | nan | [-0.349, +0.197] | defined |
| `A_local_transitivity_p10_l3` | 76 | 1.00 | -0.402 | 0.0002 | nan | [-0.542, -0.197] | defined |
| `A_local_transitivity_p90_l1` | 75 | 0.99 | -0.031 | 0.7846 | nan | [-0.308, +0.226] | defined |
| `A_local_transitivity_p90_l2` | 76 | 1.00 | -0.094 | 0.4217 | nan | [-0.347, +0.175] | defined |
| `A_local_transitivity_p90_l3` | 76 | 1.00 | -0.073 | 0.5391 | nan | [-0.342, +0.210] | defined |
| `A_local_transitivity_iqr_l1` | 75 | 0.99 | -0.088 | 0.4523 | nan | [-0.346, +0.186] | defined |
| `A_local_transitivity_iqr_l2` | 76 | 1.00 | -0.022 | 0.8578 | nan | [-0.284, +0.251] | defined |
| `A_local_transitivity_iqr_l3` | 76 | 1.00 | -0.073 | 0.5395 | nan | [-0.342, +0.210] | defined |
| `A_triangles_per_node_mean_l1` | 76 | 1.00 | -0.243 | 0.03179 | nan | [-0.492, +0.031] | defined |
| `A_triangles_per_node_mean_l2` | 76 | 1.00 | +0.043 | 0.7141 | nan | [-0.227, +0.291] | defined |
| `A_triangles_per_node_mean_l3` | 76 | 1.00 | -0.175 | 0.1358 | nan | [-0.422, +0.097] | defined |
| `A_ego_density_mean_l1` | 75 | 0.99 | +0.044 | 0.6993 | nan | [-0.234, +0.306] | defined |
| `A_ego_density_mean_l2` | 76 | 1.00 | -0.080 | 0.5015 | nan | [-0.337, +0.214] | defined |
| `A_ego_density_mean_l3` | 76 | 1.00 | -0.032 | 0.7968 | nan | [-0.279, +0.235] | defined |
| `A_ego_density_median_l1` | 75 | 0.99 | +0.054 | 0.6371 | nan | [-0.220, +0.315] | defined |
| `A_ego_density_median_l2` | 76 | 1.00 | -0.054 | 0.6527 | nan | [-0.301, +0.230] | defined |
| `A_ego_density_median_l3` | 76 | 1.00 | -0.085 | 0.4663 | nan | [-0.327, +0.179] | defined |
| `A_ego_cycle_surplus_mean_l1` | 76 | 1.00 | -0.243 | 0.03179 | nan | [-0.492, +0.031] | defined |
| `A_ego_cycle_surplus_mean_l2` | 76 | 1.00 | +0.043 | 0.7141 | nan | [-0.227, +0.291] | defined |
| `A_ego_cycle_surplus_mean_l3` | 76 | 1.00 | -0.175 | 0.1358 | nan | [-0.422, +0.097] | defined |
| `A_cycle_rank_per_edge_l1` | 75 | 0.99 | -0.293 | 0.0118 | nan | [-0.527, -0.031] | defined |
| `A_cycle_rank_per_edge_l2` | 76 | 1.00 | +0.110 | 0.3393 | nan | [-0.171, +0.362] | defined |
| `A_cycle_rank_per_edge_l3` | 76 | 1.00 | -0.152 | 0.1992 | nan | [-0.395, +0.119] | defined |
| `A_cycle_surplus_per_component_mean_l1` | 75 | 0.99 | -0.296 | 0.0104 | nan | [-0.529, -0.034] | defined |
| `A_cycle_surplus_per_component_mean_l2` | 76 | 1.00 | -0.036 | 0.763 | nan | [-0.307, +0.239] | defined |
| `A_cycle_surplus_per_component_mean_l3` | 76 | 1.00 | -0.208 | 0.07558 | nan | [-0.470, +0.057] | defined |
| `A_degree_mean_l1` | 76 | 1.00 | -0.244 | 0.03219 | nan | [-0.496, +0.028] | defined |
| `A_degree_mean_l2` | 76 | 1.00 | +0.110 | 0.3381 | nan | [-0.171, +0.362] | defined |
| `A_degree_mean_l3` | 76 | 1.00 | -0.152 | 0.1992 | nan | [-0.395, +0.119] | defined |
| `A_degree_p10_l1` | 76 | 1.00 | -0.196 | 0.09378 | nan | [-0.456, +0.086] | defined |
| `A_degree_p10_l2` | 76 | 1.00 | +0.181 | 0.1334 | nan | [-0.097, +0.432] | defined |
| `A_degree_p10_l3` | 76 | 1.00 | -0.264 | 0.0216 | nan | [-0.489, +0.082] | defined |
| `A_degree_iqr_l1` | 76 | 1.00 | -0.096 | 0.4183 | nan | [-0.371, +0.180] | defined |
| `A_degree_iqr_l2` | 76 | 1.00 | +0.110 | 0.3415 | nan | [-0.176, +0.369] | defined |
| `A_degree_iqr_l3` | 76 | 1.00 | -0.157 | 0.1868 | nan | [-0.412, +0.127] | defined |
| `A_degree_cv_l1` | 75 | 0.99 | -0.078 | 0.5049 | nan | [-0.331, +0.200] | defined |
| `A_degree_cv_l2` | 76 | 1.00 | +0.010 | 0.9302 | nan | [-0.259, +0.283] | defined |
| `A_degree_cv_l3` | 76 | 1.00 | +0.131 | 0.2701 | nan | [-0.134, +0.373] | defined |
| `A_degree_p90_over_mean_l1` | 75 | 0.99 | -0.071 | 0.5371 | nan | [-0.326, +0.212] | defined |
| `A_degree_p90_over_mean_l2` | 76 | 1.00 | +0.018 | 0.8766 | nan | [-0.247, +0.301] | defined |
| `A_degree_p90_over_mean_l3` | 76 | 1.00 | -0.135 | 0.2551 | nan | [-0.389, +0.144] | defined |
| `A_degree_median_over_mean_l1` | 75 | 0.99 | -0.046 | 0.6977 | nan | [-0.294, +0.239] | defined |
| `A_degree_median_over_mean_l2` | 76 | 1.00 | +0.219 | 0.05679 | nan | [-0.042, +0.459] | defined |
| `A_degree_median_over_mean_l3` | 76 | 1.00 | -0.131 | 0.2685 | nan | [-0.374, +0.120] | defined |
| `A_forman_p10_l1` | 75 | 0.99 | +0.304 | 0.008598 | nan | [+0.046, +0.531] | defined |
| `A_forman_p10_l2` | 76 | 1.00 | -0.041 | 0.7283 | nan | [-0.306, +0.233] | defined |
| `A_forman_p10_l3` | 76 | 1.00 | +0.059 | 0.6059 | nan | [-0.200, +0.325] | defined |
| `A_forman_p90_l1` | 75 | 0.99 | +0.289 | 0.0124 | nan | [+0.026, +0.527] | defined |
| `A_forman_p90_l2` | 76 | 1.00 | -0.077 | 0.5031 | nan | [-0.330, +0.210] | defined |
| `A_forman_p90_l3` | 76 | 1.00 | +0.144 | 0.2216 | nan | [-0.139, +0.400] | defined |
| `A_forman_node_incident_mean_l1` | 75 | 0.99 | +0.296 | 0.009798 | nan | [+0.034, +0.528] | defined |
| `A_forman_node_incident_mean_l2` | 76 | 1.00 | -0.098 | 0.3939 | nan | [-0.361, +0.181] | defined |
| `A_forman_node_incident_mean_l3` | 76 | 1.00 | +0.073 | 0.5465 | nan | [-0.191, +0.334] | defined |
| `A_forman_node_incident_median_l1` | 75 | 0.99 | +0.297 | 0.009398 | nan | [+0.037, +0.528] | defined |
| `A_forman_node_incident_median_l2` | 76 | 1.00 | -0.081 | 0.4751 | nan | [-0.343, +0.198] | defined |
| `A_forman_node_incident_median_l3` | 76 | 1.00 | +0.141 | 0.2312 | nan | [-0.122, +0.386] | defined |
| `B_sibling_density_mean_l2` | 76 | 1.00 | +0.135 | 0.2599 | nan | [-0.143, +0.403] | defined |
| `B_sibling_density_mean_l3` | 55 | 0.72 | -0.017 | 0.9034 | nan | [-0.338, +0.292] | defined |
| `B_sibling_density_median_parent_l2` | 76 | 1.00 | +0.148 | 0.2152 | nan | [-0.140, +0.420] | defined |
| `B_sibling_density_median_parent_l3` | 55 | 0.72 | -0.114 | 0.4141 | nan | [-0.425, +0.199] | defined |
| `B_sibling_density_child_weighted_l2` | 76 | 1.00 | +0.143 | 0.2294 | nan | [-0.137, +0.413] | defined |
| `B_sibling_density_child_weighted_l3` | 55 | 0.72 | -0.080 | 0.5715 | nan | [-0.400, +0.222] | defined |
| `B_sibling_density_item_weighted_l2` | 76 | 1.00 | +0.124 | 0.2933 | nan | [-0.152, +0.393] | defined |
| `B_sibling_density_item_weighted_l3` | 55 | 0.72 | -0.245 | 0.07938 | nan | [-0.521, +0.057] | defined |
| `B_sibling_transitivity_mean_l2` | 76 | 1.00 | +0.156 | 0.1946 | nan | [-0.135, +0.424] | defined |
| `B_sibling_transitivity_mean_l3` | 50 | 0.66 | -0.029 | 0.8382 | nan | [-0.308, +0.252] | defined |
| `B_sibling_transitivity_child_weighted_l2` | 76 | 1.00 | +0.139 | 0.2484 | nan | [-0.151, +0.407] | defined |
| `B_sibling_transitivity_child_weighted_l3` | 50 | 0.66 | -0.014 | 0.9224 | nan | [-0.292, +0.273] | defined |
| `B_sibling_cycle_rank_per_node_mean_l2` | 76 | 1.00 | +0.452 | 0.0002 | nan | [+0.196, +0.652] | defined |
| `B_sibling_cycle_rank_per_node_mean_l3` | 76 | 1.00 | +0.099 | 0.4057 | nan | [-0.177, +0.363] | defined |
| `B_sibling_degree_median_mean_l2` | 76 | 1.00 | +0.449 | 0.0002 | nan | [+0.196, +0.657] | defined |
| `B_sibling_degree_median_mean_l3` | 76 | 1.00 | +0.143 | 0.2298 | nan | [-0.122, +0.402] | defined |
| `B_sibling_degree_mean_mean_l2` | 76 | 1.00 | +0.497 | 0.0002 | nan | [+0.252, +0.700] | defined |
| `B_sibling_degree_mean_mean_l3` | 76 | 1.00 | +0.100 | 0.3999 | nan | [-0.164, +0.358] | defined |
| `B_sibling_degree_p90_mean_l2` | 76 | 1.00 | +0.504 | 0.0002 | nan | [+0.254, +0.702] | defined |
| `B_sibling_degree_p90_mean_l3` | 76 | 1.00 | +0.052 | 0.6619 | nan | [-0.213, +0.313] | defined |
| `B_sibling_forman_mean_l2` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_forman_mean_l3` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_forman_negative_fraction_l2` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_forman_negative_fraction_l3` | 0 | 0.00 | | | | | TOO_FEW |
| `B_sibling_edge_share_l2` | 76 | 1.00 | +0.551 | 0.0002 | nan | [+0.300, +0.765] | defined |
| `B_sibling_edge_share_l3` | 76 | 1.00 | +0.162 | 0.1728 | nan | [-0.103, +0.406] | defined |
| `D_multi_only_density_l1` | 75 | 0.99 | +0.064 | 0.5749 | nan | [-0.216, +0.323] | defined |
| `D_multi_only_density_l2` | 76 | 1.00 | -0.147 | 0.2076 | nan | [-0.404, +0.128] | defined |
| `D_multi_only_density_l3` | 76 | 1.00 | -0.202 | 0.08918 | nan | [-0.421, +0.042] | defined |
| `D_multi_only_transitivity_l1` | 75 | 0.99 | +0.052 | 0.6503 | nan | [-0.222, +0.314] | defined |
| `D_multi_only_transitivity_l2` | 52 | 0.68 | -0.185 | 0.1982 | nan | [-0.472, +0.156] | defined |
| `D_multi_only_transitivity_l3` | 49 | 0.64 | +0.098 | 0.5075 | nan | [-0.251, +0.415] | defined |
| `D_multi_only_cycle_rank_per_node_l1` | 76 | 1.00 | -0.245 | 0.03159 | nan | [-0.496, +0.028] | defined |
| `D_multi_only_cycle_rank_per_node_l2` | 76 | 1.00 | +0.001 | 0.9936 | nan | [-0.267, +0.256] | defined |
| `D_multi_only_cycle_rank_per_node_l3` | 76 | 1.00 | -0.163 | 0.169 | nan | [-0.417, +0.121] | defined |
| `D_multi_only_degree_median_l1` | 76 | 1.00 | -0.248 | 0.02939 | nan | [-0.496, +0.019] | defined |
| `D_multi_only_degree_median_l2` | 76 | 1.00 | +0.021 | 0.8596 | nan | [-0.256, +0.275] | defined |
| `D_multi_only_degree_median_l3` | 76 | 1.00 | -0.151 | 0.2012 | nan | [-0.407, +0.133] | defined |
| `D_multi_only_forman_mean_l1` | 75 | 0.99 | +0.293 | 0.0112 | nan | [+0.036, +0.527] | defined |
| `D_multi_only_forman_mean_l2` | 76 | 1.00 | +0.015 | 0.901 | nan | [-0.245, +0.277] | defined |
| `D_multi_only_forman_mean_l3` | 76 | 1.00 | +0.187 | 0.1134 | nan | [-0.096, +0.446] | defined |
| `D_multi_only_forman_negative_fraction_l1` | 75 | 0.99 | | | | | UNDEFINED_DEGENERATE |
| `D_multi_only_forman_negative_fraction_l2` | 76 | 1.00 | -0.010 | 0.9414 | nan | [-0.223, +0.208] | defined |
| `D_multi_only_forman_negative_fraction_l3` | 76 | 1.00 | -0.157 | 0.1846 | nan | [-0.415, +0.123] | defined |
| `C_density_global_delta_l1` | 75 | 0.99 | -0.016 | 0.8884 | nan | [-0.288, +0.228] | defined |
| `C_density_global_delta_l2` | 76 | 1.00 | -0.070 | 0.5505 | nan | [-0.329, +0.209] | defined |
| `C_density_global_delta_l3` | 76 | 1.00 | +0.127 | 0.2801 | nan | [-0.150, +0.384] | defined |
| `C_density_global_z_l1` | 73 | 0.96 | +0.050 | 0.6819 | nan | [-0.206, +0.288] | defined |
| `C_density_global_z_l2` | 76 | 1.00 | -0.088 | 0.4509 | nan | [-0.316, +0.130] | defined |
| `C_density_global_z_l3` | 76 | 1.00 | -0.123 | 0.2957 | nan | [-0.343, +0.113] | defined |
| `C_density_global_ratio_l1` | 75 | 0.99 | +0.009 | 0.94 | nan | [-0.259, +0.268] | defined |
| `C_density_global_ratio_l2` | 76 | 1.00 | -0.096 | 0.4109 | nan | [-0.361, +0.183] | defined |
| `C_density_global_ratio_l3` | 76 | 1.00 | +0.113 | 0.3389 | nan | [-0.168, +0.372] | defined |
| `C_density_degree_stratified_delta_l1` | 75 | 0.99 | -0.062 | 0.5929 | nan | [-0.334, +0.193] | defined |
| `C_density_degree_stratified_delta_l2` | 76 | 1.00 | -0.007 | 0.9536 | nan | [-0.267, +0.261] | defined |
| `C_density_degree_stratified_delta_l3` | 76 | 1.00 | +0.048 | 0.6875 | nan | [-0.225, +0.311] | defined |
| `C_density_degree_stratified_z_l1` | 67 | 0.88 | -0.287 | 0.017 | nan | [-0.505, -0.017] | defined |
| `C_density_degree_stratified_z_l2` | 76 | 1.00 | -0.354 | 0.002 | nan | [-0.563, -0.094] | defined |
| `C_density_degree_stratified_z_l3` | 76 | 1.00 | -0.285 | 0.014 | nan | [-0.512, -0.023] | defined |
| `C_density_degree_stratified_ratio_l1` | 75 | 0.99 | -0.039 | 0.7421 | nan | [-0.308, +0.211] | defined |
| `C_density_degree_stratified_ratio_l2` | 76 | 1.00 | -0.125 | 0.2877 | nan | [-0.375, +0.144] | defined |
| `C_density_degree_stratified_ratio_l3` | 76 | 1.00 | +0.036 | 0.7598 | nan | [-0.236, +0.299] | defined |
| `C_density_parent_conditioned_delta_l2` | 76 | 1.00 | -0.032 | 0.7772 | nan | [-0.295, +0.228] | defined |
| `C_density_parent_conditioned_delta_l3` | 76 | 1.00 | -0.087 | 0.4565 | nan | [-0.350, +0.190] | defined |
| `C_density_parent_conditioned_z_l2` | 76 | 1.00 | -0.035 | 0.7582 | nan | [-0.251, +0.179] | defined |
| `C_density_parent_conditioned_z_l3` | 71 | 0.93 | -0.288 | 0.0176 | nan | [-0.526, -0.026] | defined |
| `C_density_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.081 | 0.4799 | nan | [-0.334, +0.172] | defined |
| `C_density_parent_conditioned_ratio_l3` | 76 | 1.00 | -0.120 | 0.3043 | nan | [-0.383, +0.154] | defined |
| `C_transitivity_global_delta_l1` | 75 | 0.99 | -0.038 | 0.7401 | nan | [-0.311, +0.213] | defined |
| `C_transitivity_global_delta_l2` | 76 | 1.00 | -0.180 | 0.1214 | nan | [-0.414, +0.069] | defined |
| `C_transitivity_global_delta_l3` | 76 | 1.00 | -0.074 | 0.5225 | nan | [-0.329, +0.194] | defined |
| `C_transitivity_global_z_l1` | 73 | 0.96 | +0.064 | 0.5927 | nan | [-0.189, +0.298] | defined |
| `C_transitivity_global_z_l2` | 76 | 1.00 | -0.194 | 0.09518 | nan | [-0.436, +0.057] | defined |
| `C_transitivity_global_z_l3` | 76 | 1.00 | -0.212 | 0.06519 | nan | [-0.456, +0.037] | defined |
| `C_transitivity_global_ratio_l1` | 75 | 0.99 | -0.022 | 0.8464 | nan | [-0.290, +0.231] | defined |
| `C_transitivity_global_ratio_l2` | 76 | 1.00 | -0.159 | 0.1714 | nan | [-0.400, +0.090] | defined |
| `C_transitivity_global_ratio_l3` | 76 | 1.00 | -0.070 | 0.5457 | nan | [-0.315, +0.198] | defined |
| `C_transitivity_degree_stratified_delta_l1` | 75 | 0.99 | -0.077 | 0.5061 | nan | [-0.349, +0.174] | defined |
| `C_transitivity_degree_stratified_delta_l2` | 76 | 1.00 | -0.230 | 0.04559 | nan | [-0.445, +0.013] | defined |
| `C_transitivity_degree_stratified_delta_l3` | 76 | 1.00 | -0.099 | 0.4069 | nan | [-0.347, +0.171] | defined |
| `C_transitivity_degree_stratified_z_l1` | 67 | 0.88 | -0.292 | 0.0156 | nan | [-0.512, -0.026] | defined |
| `C_transitivity_degree_stratified_z_l2` | 76 | 1.00 | +0.008 | 0.9464 | nan | [-0.248, +0.256] | defined |
| `C_transitivity_degree_stratified_z_l3` | 76 | 1.00 | -0.003 | 0.9812 | nan | [-0.233, +0.220] | defined |
| `C_transitivity_degree_stratified_ratio_l1` | 75 | 0.99 | -0.069 | 0.5503 | nan | [-0.339, +0.182] | defined |
| `C_transitivity_degree_stratified_ratio_l2` | 76 | 1.00 | -0.047 | 0.6893 | nan | [-0.286, +0.182] | defined |
| `C_transitivity_degree_stratified_ratio_l3` | 76 | 1.00 | -0.096 | 0.4223 | nan | [-0.346, +0.175] | defined |
| `C_transitivity_parent_conditioned_delta_l2` | 76 | 1.00 | -0.213 | 0.06519 | nan | [-0.459, +0.041] | defined |
| `C_transitivity_parent_conditioned_delta_l3` | 76 | 1.00 | +0.278 | 0.0158 | nan | [+0.021, +0.500] | defined |
| `C_transitivity_parent_conditioned_z_l2` | 76 | 1.00 | -0.257 | 0.02639 | nan | [-0.493, +0.000] | defined |
| `C_transitivity_parent_conditioned_z_l3` | 69 | 0.91 | +0.347 | 0.003399 | nan | [+0.089, +0.566] | defined |
| `C_transitivity_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.230 | 0.04719 | nan | [-0.473, +0.023] | defined |
| `C_transitivity_parent_conditioned_ratio_l3` | 76 | 1.00 | +0.288 | 0.0118 | nan | [+0.043, +0.510] | defined |
| `C_cycle_rank_per_node_global_delta_l1` | 76 | 1.00 | -0.042 | 0.7241 | nan | [-0.304, +0.219] | defined |
| `C_cycle_rank_per_node_global_delta_l2` | 76 | 1.00 | -0.090 | 0.4397 | nan | [-0.344, +0.193] | defined |
| `C_cycle_rank_per_node_global_delta_l3` | 76 | 1.00 | +0.118 | 0.3189 | nan | [-0.162, +0.376] | defined |
| `C_cycle_rank_per_node_global_z_l1` | 74 | 0.97 | +0.077 | 0.5115 | nan | [-0.174, +0.323] | defined |
| `C_cycle_rank_per_node_global_z_l2` | 76 | 1.00 | -0.088 | 0.4509 | nan | [-0.316, +0.130] | defined |
| `C_cycle_rank_per_node_global_z_l3` | 76 | 1.00 | -0.123 | 0.2957 | nan | [-0.343, +0.113] | defined |
| `C_cycle_rank_per_node_global_ratio_l1` | 75 | 0.99 | +0.008 | 0.946 | nan | [-0.259, +0.267] | defined |
| `C_cycle_rank_per_node_global_ratio_l2` | 76 | 1.00 | -0.098 | 0.4025 | nan | [-0.363, +0.182] | defined |
| `C_cycle_rank_per_node_global_ratio_l3` | 76 | 1.00 | +0.110 | 0.3557 | nan | [-0.169, +0.371] | defined |
| `C_cycle_rank_per_node_degree_stratified_delta_l1` | 76 | 1.00 | -0.097 | 0.4161 | nan | [-0.351, +0.170] | defined |
| `C_cycle_rank_per_node_degree_stratified_delta_l2` | 76 | 1.00 | -0.113 | 0.3305 | nan | [-0.352, +0.158] | defined |
| `C_cycle_rank_per_node_degree_stratified_delta_l3` | 76 | 1.00 | +0.045 | 0.7015 | nan | [-0.224, +0.309] | defined |
| `C_cycle_rank_per_node_degree_stratified_z_l1` | 72 | 0.95 | -0.298 | 0.0118 | nan | [-0.518, -0.031] | defined |
| `C_cycle_rank_per_node_degree_stratified_z_l2` | 76 | 1.00 | -0.354 | 0.002 | nan | [-0.563, -0.094] | defined |
| `C_cycle_rank_per_node_degree_stratified_z_l3` | 76 | 1.00 | -0.285 | 0.014 | nan | [-0.512, -0.023] | defined |
| `C_cycle_rank_per_node_degree_stratified_ratio_l1` | 75 | 0.99 | -0.038 | 0.7473 | nan | [-0.307, +0.214] | defined |
| `C_cycle_rank_per_node_degree_stratified_ratio_l2` | 76 | 1.00 | -0.129 | 0.2783 | nan | [-0.383, +0.144] | defined |
| `C_cycle_rank_per_node_degree_stratified_ratio_l3` | 76 | 1.00 | +0.036 | 0.7628 | nan | [-0.236, +0.298] | defined |
| `C_cycle_rank_per_node_parent_conditioned_delta_l2` | 76 | 1.00 | -0.130 | 0.2621 | nan | [-0.376, +0.135] | defined |
| `C_cycle_rank_per_node_parent_conditioned_delta_l3` | 76 | 1.00 | -0.070 | 0.5607 | nan | [-0.341, +0.206] | defined |
| `C_cycle_rank_per_node_parent_conditioned_z_l2` | 76 | 1.00 | -0.035 | 0.7582 | nan | [-0.251, +0.179] | defined |
| `C_cycle_rank_per_node_parent_conditioned_z_l3` | 68 | 0.89 | -0.283 | 0.0208 | nan | [-0.529, -0.014] | defined |
| `C_cycle_rank_per_node_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.082 | 0.4783 | nan | [-0.335, +0.167] | defined |
| `C_cycle_rank_per_node_parent_conditioned_ratio_l3` | 76 | 1.00 | -0.097 | 0.4111 | nan | [-0.368, +0.181] | defined |
| `C_degree_median_global_delta_l1` | 76 | 1.00 | +0.004 | 0.9702 | nan | [-0.266, +0.271] | defined |
| `C_degree_median_global_delta_l2` | 76 | 1.00 | +0.155 | 0.178 | nan | [-0.102, +0.405] | defined |
| `C_degree_median_global_delta_l3` | 76 | 1.00 | +0.295 | 0.009398 | nan | [+0.041, +0.502] | defined |
| `C_degree_median_global_z_l1` | 25 | 0.33 | -0.128 | 0.5297 | nan | [-0.579, +0.331] | defined |
| `C_degree_median_global_z_l2` | 34 | 0.45 | +0.312 | 0.07638 | nan | [-0.031, +0.596] | defined |
| `C_degree_median_global_z_l3` | 13 | 0.17 | +0.512 | 0.07758 | nan | [-0.100, +0.932] | defined |
| `C_degree_median_global_ratio_l1` | 75 | 0.99 | +0.039 | 0.7415 | nan | [-0.232, +0.299] | defined |
| `C_degree_median_global_ratio_l2` | 76 | 1.00 | +0.142 | 0.2146 | nan | [-0.108, +0.388] | defined |
| `C_degree_median_global_ratio_l3` | 76 | 1.00 | +0.294 | 0.009998 | nan | [+0.041, +0.499] | defined |
| `C_degree_median_degree_stratified_delta_l1` | 76 | 1.00 | -0.041 | 0.7321 | nan | [-0.299, +0.219] | defined |
| `C_degree_median_degree_stratified_delta_l2` | 76 | 1.00 | -0.308 | 0.006799 | nan | [-0.534, -0.030] | defined |
| `C_degree_median_degree_stratified_delta_l3` | 76 | 1.00 | +0.244 | 0.03039 | nan | [-0.109, +0.476] | defined |
| `C_degree_median_degree_stratified_z_l1` | 32 | 0.42 | -0.330 | 0.06419 | nan | [-0.630, +0.013] | defined |
| `C_degree_median_degree_stratified_z_l2` | 30 | 0.39 | -0.461 | 0.011 | nan | [-0.757, -0.029] | defined |
| `C_degree_median_degree_stratified_z_l3` | 5 | 0.07 | | | | | TOO_FEW |
| `C_degree_median_degree_stratified_ratio_l1` | 75 | 0.99 | +0.021 | 0.8642 | nan | [-0.245, +0.285] | defined |
| `C_degree_median_degree_stratified_ratio_l2` | 76 | 1.00 | -0.312 | 0.006199 | nan | [-0.532, -0.041] | defined |
| `C_degree_median_degree_stratified_ratio_l3` | 76 | 1.00 | +0.244 | 0.03139 | nan | [-0.109, +0.474] | defined |
| `C_degree_median_parent_conditioned_delta_l2` | 76 | 1.00 | -0.049 | 0.6831 | nan | [-0.279, +0.197] | defined |
| `C_degree_median_parent_conditioned_delta_l3` | 76 | 1.00 | +0.243 | 0.03439 | nan | [-0.034, +0.453] | defined |
| `C_degree_median_parent_conditioned_z_l2` | 35 | 0.46 | -0.060 | 0.7297 | nan | [-0.383, +0.276] | defined |
| `C_degree_median_parent_conditioned_z_l3` | 7 | 0.09 | | | | | TOO_FEW |
| `C_degree_median_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.019 | 0.8742 | nan | [-0.244, +0.220] | defined |
| `C_degree_median_parent_conditioned_ratio_l3` | 76 | 1.00 | +0.243 | 0.03499 | nan | [-0.034, +0.451] | defined |
| `C_forman_mean_global_delta_l1` | 75 | 0.99 | +0.014 | 0.8994 | nan | [-0.237, +0.288] | defined |
| `C_forman_mean_global_delta_l2` | 76 | 1.00 | +0.274 | 0.018 | nan | [+0.005, +0.511] | defined |
| `C_forman_mean_global_delta_l3` | 76 | 1.00 | +0.158 | 0.1864 | nan | [-0.101, +0.402] | defined |
| `C_forman_mean_global_z_l1` | 73 | 0.96 | -0.060 | 0.6193 | nan | [-0.300, +0.198] | defined |
| `C_forman_mean_global_z_l2` | 76 | 1.00 | +0.283 | 0.0142 | nan | [+0.017, +0.522] | defined |
| `C_forman_mean_global_z_l3` | 76 | 1.00 | +0.159 | 0.1876 | nan | [-0.099, +0.398] | defined |
| `C_forman_mean_global_ratio_l1` | 75 | 0.99 | -0.008 | 0.9436 | nan | [-0.278, +0.242] | defined |
| `C_forman_mean_global_ratio_l2` | 76 | 1.00 | -0.286 | 0.014 | nan | [-0.523, -0.021] | defined |
| `C_forman_mean_global_ratio_l3` | 76 | 1.00 | -0.152 | 0.205 | nan | [-0.396, +0.107] | defined |
| `C_forman_mean_degree_stratified_delta_l1` | 75 | 0.99 | +0.066 | 0.5687 | nan | [-0.190, +0.338] | defined |
| `C_forman_mean_degree_stratified_delta_l2` | 76 | 1.00 | +0.056 | 0.6359 | nan | [-0.217, +0.312] | defined |
| `C_forman_mean_degree_stratified_delta_l3` | 76 | 1.00 | -0.140 | 0.2406 | nan | [-0.388, +0.121] | defined |
| `C_forman_mean_degree_stratified_z_l1` | 67 | 0.88 | +0.291 | 0.0168 | nan | [+0.024, +0.505] | defined |
| `C_forman_mean_degree_stratified_z_l2` | 76 | 1.00 | +0.129 | 0.2791 | nan | [-0.154, +0.375] | defined |
| `C_forman_mean_degree_stratified_z_l3` | 76 | 1.00 | +0.026 | 0.8292 | nan | [-0.233, +0.271] | defined |
| `C_forman_mean_degree_stratified_ratio_l1` | 75 | 0.99 | -0.054 | 0.6413 | nan | [-0.323, +0.198] | defined |
| `C_forman_mean_degree_stratified_ratio_l2` | 76 | 1.00 | -0.064 | 0.5833 | nan | [-0.320, +0.206] | defined |
| `C_forman_mean_degree_stratified_ratio_l3` | 76 | 1.00 | +0.137 | 0.2523 | nan | [-0.129, +0.385] | defined |
| `C_forman_mean_parent_conditioned_delta_l2` | 76 | 1.00 | +0.235 | 0.04219 | nan | [-0.023, +0.470] | defined |
| `C_forman_mean_parent_conditioned_delta_l3` | 76 | 1.00 | -0.264 | 0.02539 | nan | [-0.498, +0.002] | defined |
| `C_forman_mean_parent_conditioned_z_l2` | 76 | 1.00 | +0.296 | 0.008998 | nan | [+0.036, +0.520] | defined |
| `C_forman_mean_parent_conditioned_z_l3` | 66 | 0.87 | -0.372 | 0.0024 | nan | [-0.594, -0.098] | defined |
| `C_forman_mean_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.222 | 0.05439 | nan | [-0.464, +0.039] | defined |
| `C_forman_mean_parent_conditioned_ratio_l3` | 76 | 1.00 | +0.266 | 0.0234 | nan | [-0.000, +0.500] | defined |
| `C_triangles_per_node_mean_global_delta_l1` | 76 | 1.00 | -0.070 | 0.5587 | nan | [-0.331, +0.191] | defined |
| `C_triangles_per_node_mean_global_delta_l2` | 76 | 1.00 | -0.223 | 0.04899 | nan | [-0.459, +0.044] | defined |
| `C_triangles_per_node_mean_global_delta_l3` | 76 | 1.00 | -0.262 | 0.0214 | nan | [-0.514, +0.002] | defined |
| `C_triangles_per_node_mean_global_z_l1` | 73 | 0.96 | +0.068 | 0.5761 | nan | [-0.186, +0.303] | defined |
| `C_triangles_per_node_mean_global_z_l2` | 76 | 1.00 | -0.200 | 0.08618 | nan | [-0.462, +0.075] | defined |
| `C_triangles_per_node_mean_global_z_l3` | 76 | 1.00 | -0.285 | 0.0136 | nan | [-0.523, -0.042] | defined |
| `C_triangles_per_node_mean_global_ratio_l1` | 75 | 0.99 | -0.007 | 0.9478 | nan | [-0.278, +0.242] | defined |
| `C_triangles_per_node_mean_global_ratio_l2` | 76 | 1.00 | -0.237 | 0.03999 | nan | [-0.478, +0.036] | defined |
| `C_triangles_per_node_mean_global_ratio_l3` | 76 | 1.00 | -0.261 | 0.0206 | nan | [-0.508, -0.001] | defined |
| `C_triangles_per_node_mean_degree_stratified_delta_l1` | 76 | 1.00 | -0.116 | 0.3249 | nan | [-0.374, +0.149] | defined |
| `C_triangles_per_node_mean_degree_stratified_delta_l2` | 76 | 1.00 | -0.014 | 0.9032 | nan | [-0.274, +0.249] | defined |
| `C_triangles_per_node_mean_degree_stratified_delta_l3` | 76 | 1.00 | -0.161 | 0.174 | nan | [-0.406, +0.094] | defined |
| `C_triangles_per_node_mean_degree_stratified_z_l1` | 67 | 0.88 | -0.280 | 0.0204 | nan | [-0.498, -0.015] | defined |
| `C_triangles_per_node_mean_degree_stratified_z_l2` | 76 | 1.00 | +0.037 | 0.7485 | nan | [-0.238, +0.330] | defined |
| `C_triangles_per_node_mean_degree_stratified_z_l3` | 76 | 1.00 | -0.135 | 0.2593 | nan | [-0.384, +0.113] | defined |
| `C_triangles_per_node_mean_degree_stratified_ratio_l1` | 75 | 0.99 | -0.053 | 0.6507 | nan | [-0.323, +0.198] | defined |
| `C_triangles_per_node_mean_degree_stratified_ratio_l2` | 76 | 1.00 | -0.026 | 0.8212 | nan | [-0.296, +0.253] | defined |
| `C_triangles_per_node_mean_degree_stratified_ratio_l3` | 76 | 1.00 | -0.157 | 0.1848 | nan | [-0.399, +0.098] | defined |
| `C_triangles_per_node_mean_parent_conditioned_delta_l2` | 76 | 1.00 | -0.230 | 0.03859 | nan | [-0.480, +0.031] | defined |
| `C_triangles_per_node_mean_parent_conditioned_delta_l3` | 76 | 1.00 | +0.456 | 0.0002 | nan | [+0.225, +0.645] | defined |
| `C_triangles_per_node_mean_parent_conditioned_z_l2` | 76 | 1.00 | -0.237 | 0.03539 | nan | [-0.486, +0.023] | defined |
| `C_triangles_per_node_mean_parent_conditioned_z_l3` | 66 | 0.87 | +0.426 | 0.0005999 | nan | [+0.178, +0.639] | defined |
| `C_triangles_per_node_mean_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.229 | 0.04039 | nan | [-0.479, +0.025] | defined |
| `C_triangles_per_node_mean_parent_conditioned_ratio_l3` | 76 | 1.00 | +0.446 | 0.0002 | nan | [+0.213, +0.639] | defined |
| `C_local_transitivity_mean_global_delta_l1` | 75 | 0.99 | -0.051 | 0.6549 | nan | [-0.326, +0.201] | defined |
| `C_local_transitivity_mean_global_delta_l2` | 76 | 1.00 | -0.238 | 0.03939 | nan | [-0.449, -0.014] | defined |
| `C_local_transitivity_mean_global_delta_l3` | 76 | 1.00 | +0.018 | 0.8758 | nan | [-0.264, +0.297] | defined |
| `C_local_transitivity_mean_global_z_l1` | 73 | 0.96 | +0.068 | 0.5757 | nan | [-0.186, +0.305] | defined |
| `C_local_transitivity_mean_global_z_l2` | 76 | 1.00 | -0.111 | 0.3421 | nan | [-0.338, +0.122] | defined |
| `C_local_transitivity_mean_global_z_l3` | 76 | 1.00 | +0.058 | 0.6255 | nan | [-0.190, +0.305] | defined |
| `C_local_transitivity_mean_global_ratio_l1` | 75 | 0.99 | -0.053 | 0.6479 | nan | [-0.328, +0.200] | defined |
| `C_local_transitivity_mean_global_ratio_l2` | 76 | 1.00 | -0.143 | 0.218 | nan | [-0.364, +0.092] | defined |
| `C_local_transitivity_mean_global_ratio_l3` | 76 | 1.00 | +0.026 | 0.8168 | nan | [-0.256, +0.301] | defined |
| `C_local_transitivity_mean_degree_stratified_delta_l1` | 75 | 0.99 | -0.088 | 0.4409 | nan | [-0.358, +0.171] | defined |
| `C_local_transitivity_mean_degree_stratified_delta_l2` | 76 | 1.00 | +0.034 | 0.7672 | nan | [-0.193, +0.251] | defined |
| `C_local_transitivity_mean_degree_stratified_delta_l3` | 76 | 1.00 | +0.074 | 0.5259 | nan | [-0.212, +0.356] | defined |
| `C_local_transitivity_mean_degree_stratified_z_l1` | 67 | 0.88 | -0.302 | 0.0118 | nan | [-0.522, -0.034] | defined |
| `C_local_transitivity_mean_degree_stratified_z_l2` | 76 | 1.00 | +0.144 | 0.2232 | nan | [-0.086, +0.343] | defined |
| `C_local_transitivity_mean_degree_stratified_z_l3` | 76 | 1.00 | +0.159 | 0.1716 | nan | [-0.119, +0.426] | defined |
| `C_local_transitivity_mean_degree_stratified_ratio_l1` | 75 | 0.99 | -0.080 | 0.4849 | nan | [-0.352, +0.172] | defined |
| `C_local_transitivity_mean_degree_stratified_ratio_l2` | 76 | 1.00 | +0.147 | 0.2254 | nan | [-0.077, +0.356] | defined |
| `C_local_transitivity_mean_degree_stratified_ratio_l3` | 76 | 1.00 | +0.096 | 0.4091 | nan | [-0.193, +0.376] | defined |
| `C_local_transitivity_mean_parent_conditioned_delta_l2` | 76 | 1.00 | -0.337 | 0.002999 | nan | [-0.552, -0.094] | defined |
| `C_local_transitivity_mean_parent_conditioned_delta_l3` | 76 | 1.00 | +0.111 | 0.3417 | nan | [-0.180, +0.365] | defined |
| `C_local_transitivity_mean_parent_conditioned_z_l2` | 76 | 1.00 | -0.361 | 0.0012 | nan | [-0.564, -0.139] | defined |
| `C_local_transitivity_mean_parent_conditioned_z_l3` | 70 | 0.92 | +0.160 | 0.1914 | nan | [-0.108, +0.400] | defined |
| `C_local_transitivity_mean_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.306 | 0.006599 | nan | [-0.523, -0.058] | defined |
| `C_local_transitivity_mean_parent_conditioned_ratio_l3` | 76 | 1.00 | +0.107 | 0.3597 | nan | [-0.181, +0.360] | defined |
| `C_sibling_density_mean_global_delta_l2` | 76 | 1.00 | +0.296 | 0.0108 | nan | [+0.013, +0.532] | defined |
| `C_sibling_density_mean_global_delta_l3` | 49 | 0.64 | +0.238 | 0.1012 | nan | [-0.088, +0.524] | defined |
| `C_sibling_density_mean_global_z_l2` | 76 | 1.00 | +0.392 | 0.0003999 | nan | [+0.151, +0.588] | defined |
| `C_sibling_density_mean_global_z_l3` | 49 | 0.64 | +0.304 | 0.03759 | nan | [-0.032, +0.600] | defined |
| `C_sibling_density_mean_global_ratio_l2` | 76 | 1.00 | +0.355 | 0.0012 | nan | [+0.097, +0.586] | defined |
| `C_sibling_density_mean_global_ratio_l3` | 49 | 0.64 | +0.412 | 0.003799 | nan | [+0.107, +0.647] | defined |
| `C_sibling_density_mean_degree_stratified_delta_l2` | 76 | 1.00 | +0.354 | 0.0022 | nan | [+0.078, +0.587] | defined |
| `C_sibling_density_mean_degree_stratified_delta_l3` | 50 | 0.66 | +0.080 | 0.5853 | nan | [-0.233, +0.391] | defined |
| `C_sibling_density_mean_degree_stratified_z_l2` | 76 | 1.00 | +0.361 | 0.0016 | nan | [+0.125, +0.558] | defined |
| `C_sibling_density_mean_degree_stratified_z_l3` | 50 | 0.66 | +0.197 | 0.1758 | nan | [-0.147, +0.515] | defined |
| `C_sibling_density_mean_degree_stratified_ratio_l2` | 76 | 1.00 | +0.384 | 0.0005999 | nan | [+0.122, +0.615] | defined |
| `C_sibling_density_mean_degree_stratified_ratio_l3` | 50 | 0.66 | +0.262 | 0.07299 | nan | [-0.074, +0.556] | defined |
| `C_sibling_density_mean_parent_conditioned_delta_l2` | 76 | 1.00 | -0.110 | 0.3513 | nan | [-0.370, +0.150] | defined |
| `C_sibling_density_mean_parent_conditioned_delta_l3` | 55 | 0.72 | +0.089 | 0.5117 | nan | [-0.205, +0.368] | defined |
| `C_sibling_density_mean_parent_conditioned_z_l2` | 76 | 1.00 | -0.187 | 0.1134 | nan | [-0.382, +0.033] | defined |
| `C_sibling_density_mean_parent_conditioned_z_l3` | 51 | 0.67 | -0.342 | 0.0114 | nan | [-0.590, -0.034] | defined |
| `C_sibling_density_mean_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.018 | 0.8698 | nan | [-0.268, +0.232] | defined |
| `C_sibling_density_mean_parent_conditioned_ratio_l3` | 55 | 0.72 | -0.022 | 0.8722 | nan | [-0.330, +0.283] | defined |
| `C_sibling_transitivity_mean_global_delta_l2` | 76 | 1.00 | +0.378 | 0.0016 | nan | [+0.105, +0.601] | defined |
| `C_sibling_transitivity_mean_global_delta_l3` | 14 | 0.18 | +0.604 | 0.02999 | nan | [+0.163, +0.862] | defined |
| `C_sibling_transitivity_mean_global_z_l2` | 76 | 1.00 | +0.518 | 0.0002 | nan | [+0.294, +0.687] | defined |
| `C_sibling_transitivity_mean_global_z_l3` | 14 | 0.18 | +0.679 | 0.0108 | nan | [+0.164, +0.937] | defined |
| `C_sibling_transitivity_mean_global_ratio_l2` | 76 | 1.00 | +0.282 | 0.0132 | nan | [+0.024, +0.520] | defined |
| `C_sibling_transitivity_mean_global_ratio_l3` | 14 | 0.18 | +0.688 | 0.007798 | nan | [+0.267, +0.875] | defined |
| `C_sibling_transitivity_mean_degree_stratified_delta_l2` | 76 | 1.00 | +0.304 | 0.008998 | nan | [+0.020, +0.537] | defined |
| `C_sibling_transitivity_mean_degree_stratified_delta_l3` | 12 | 0.16 | +0.594 | 0.04359 | nan | [+0.065, +0.871] | defined |
| `C_sibling_transitivity_mean_degree_stratified_z_l2` | 76 | 1.00 | +0.426 | 0.0005999 | nan | [+0.179, +0.630] | defined |
| `C_sibling_transitivity_mean_degree_stratified_z_l3` | 12 | 0.16 | +0.448 | 0.1472 | nan | [-0.183, +0.914] | defined |
| `C_sibling_transitivity_mean_degree_stratified_ratio_l2` | 76 | 1.00 | +0.183 | 0.1166 | nan | [-0.091, +0.428] | defined |
| `C_sibling_transitivity_mean_degree_stratified_ratio_l3` | 12 | 0.16 | +0.685 | 0.0162 | nan | [+0.209, +0.921] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_delta_l2` | 76 | 1.00 | -0.330 | 0.004199 | nan | [-0.564, -0.084] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_delta_l3` | 50 | 0.66 | -0.205 | 0.149 | nan | [-0.499, +0.081] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_z_l2` | 76 | 1.00 | -0.331 | 0.003799 | nan | [-0.555, -0.083] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_z_l3` | 49 | 0.64 | -0.314 | 0.03259 | nan | [-0.573, -0.036] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_ratio_l2` | 76 | 1.00 | -0.335 | 0.004199 | nan | [-0.567, -0.090] | defined |
| `C_sibling_transitivity_mean_parent_conditioned_ratio_l3` | 49 | 0.64 | -0.197 | 0.175 | nan | [-0.478, +0.102] | defined |

## Paired bootstrap of |rho| against the declared baseline

| candidate | n | rho | baseline rho | delta |rho| | 95% CI |
|---|---|---|---|---|---|
| `A_local_transitivity_mean_l1` | 106 | +0.134 | +0.182 | -0.048 | [-0.091, +0.042] |
| `A_local_transitivity_mean_l2` | 108 | -0.458 | -0.465 | -0.007 | [-0.030, +0.015] |
| `A_local_transitivity_mean_l3` | 108 | -0.450 | -0.438 | +0.012 | [-0.074, +0.142] |
| `A_local_transitivity_median_l1` | 106 | +0.048 | +0.182 | -0.134 | [-0.238, +0.123] |
| `A_local_transitivity_median_l2` | 108 | -0.468 | -0.465 | +0.003 | [-0.015, +0.023] |
| `A_local_transitivity_median_l3` | 108 | -0.472 | -0.438 | +0.033 | [-0.041, +0.158] |
| `A_local_transitivity_p10_l1` | 106 | +0.172 | +0.182 | -0.011 | [-0.031, +0.013] |
| `A_local_transitivity_p10_l2` | 108 | -0.513 | -0.465 | +0.048 | [+0.002, +0.103] |
| `A_local_transitivity_p10_l3` | 108 | -0.541 | -0.438 | +0.102 | [-0.076, +0.294] |
| `A_local_transitivity_p90_l1` | 106 | -0.314 | +0.182 | +0.131 | [-0.240, +0.436] |
| `A_local_transitivity_p90_l2` | 108 | -0.443 | -0.465 | -0.022 | [-0.081, +0.028] |
| `A_local_transitivity_p90_l3` | 108 | -0.385 | -0.438 | -0.053 | [-0.155, +0.087] |
| `A_local_transitivity_iqr_l1` | 106 | -0.179 | +0.182 | -0.004 | [-0.050, +0.043] |
| `A_local_transitivity_iqr_l2` | 108 | -0.191 | -0.465 | -0.274 | [-0.449, -0.103] |
| `A_local_transitivity_iqr_l3` | 108 | -0.329 | -0.438 | -0.110 | [-0.268, +0.059] |
| `A_triangles_per_node_mean_l1` | 106 | +0.165 | +0.182 | -0.017 | [-0.285, +0.261] |
| `A_triangles_per_node_mean_l2` | 108 | -0.316 | -0.465 | -0.149 | [-0.292, -0.032] |
| `A_triangles_per_node_mean_l3` | 108 | -0.494 | -0.438 | +0.056 | [-0.003, +0.167] |
| `A_ego_density_mean_l1` | 107 | +0.075 | +0.133 | -0.058 | [-0.135, +0.071] |
| `A_ego_density_mean_l2` | 108 | -0.484 | -0.451 | +0.033 | [-0.132, +0.194] |
| `A_ego_density_mean_l3` | 108 | -0.264 | -0.479 | -0.215 | [-0.460, +0.032] |
| `A_ego_density_median_l1` | 107 | +0.016 | +0.133 | -0.116 | [-0.217, +0.137] |
| `A_ego_density_median_l2` | 108 | -0.483 | -0.451 | +0.032 | [-0.146, +0.202] |
| `A_ego_density_median_l3` | 108 | -0.135 | -0.479 | -0.345 | [-0.565, -0.066] |
| `A_ego_cycle_surplus_mean_l1` | 108 | +0.210 | +0.194 | +0.016 | [-0.012, +0.034] |
| `A_ego_cycle_surplus_mean_l2` | 108 | -0.316 | -0.235 | +0.081 | [+0.016, +0.160] |
| `A_ego_cycle_surplus_mean_l3` | 108 | -0.494 | -0.478 | +0.017 | [-0.007, +0.044] |
| `A_cycle_rank_per_edge_l1` | 107 | +0.174 | +0.172 | +0.001 | [-0.002, +0.005] |
| `A_cycle_rank_per_edge_l2` | 108 | -0.235 | -0.235 | +0.001 | [+0.000, +0.002] |
| `A_cycle_rank_per_edge_l3` | 108 | -0.478 | -0.478 | +0.000 | [+0.000, +0.000] |
| `A_cycle_surplus_per_component_mean_l1` | 107 | +0.122 | +0.172 | -0.051 | [-0.087, +0.057] |
| `A_cycle_surplus_per_component_mean_l2` | 108 | +0.261 | -0.235 | +0.027 | [-0.189, +0.245] |
| `A_cycle_surplus_per_component_mean_l3` | 108 | +0.095 | -0.478 | -0.383 | [-0.517, -0.177] |
| `A_degree_mean_l1` | 108 | +0.194 | +0.193 | +0.001 | [-0.016, +0.017] |
| `A_degree_mean_l2` | 108 | -0.235 | -0.267 | -0.032 | [-0.132, +0.044] |
| `A_degree_mean_l3` | 108 | -0.478 | -0.562 | -0.084 | [-0.159, -0.018] |
| `A_degree_p10_l1` | 108 | +0.322 | +0.193 | +0.129 | [+0.054, +0.190] |
| `A_degree_p10_l2` | 108 | -0.040 | -0.267 | -0.227 | [-0.344, +0.086] |
| `A_degree_p10_l3` | 108 | -0.391 | -0.562 | -0.171 | [-0.411, +0.037] |
| `A_degree_iqr_l1` | 108 | -0.134 | +0.193 | -0.060 | [-0.326, +0.252] |
| `A_degree_iqr_l2` | 108 | -0.140 | -0.267 | -0.127 | [-0.288, +0.041] |
| `A_degree_iqr_l3` | 108 | -0.420 | -0.562 | -0.142 | [-0.281, -0.038] |
| `A_degree_cv_l1` | 107 | -0.111 | +0.172 | -0.061 | [-0.300, +0.214] |
| `A_degree_cv_l2` | 108 | +0.259 | -0.267 | -0.009 | [-0.167, +0.166] |
| `A_degree_cv_l3` | 108 | +0.265 | -0.562 | -0.297 | [-0.468, -0.154] |
| `A_degree_p90_over_mean_l1` | 107 | -0.164 | +0.086 | +0.078 | [-0.217, +0.317] |
| `A_degree_p90_over_mean_l2` | 108 | -0.176 | -0.200 | -0.024 | [-0.179, +0.123] |
| `A_degree_p90_over_mean_l3` | 108 | -0.346 | -0.476 | -0.130 | [-0.287, -0.016] |
| `A_degree_median_over_mean_l1` | 107 | +0.202 | +0.172 | +0.031 | [-0.230, +0.268] |
| `A_degree_median_over_mean_l2` | 108 | +0.196 | -0.267 | -0.071 | [-0.342, +0.213] |
| `A_degree_median_over_mean_l3` | 108 | +0.049 | -0.562 | -0.513 | [-0.650, -0.245] |
| `A_forman_p10_l1` | 107 | -0.110 | -0.161 | -0.051 | [-0.084, +0.060] |
| `A_forman_p10_l2` | 108 | +0.157 | +0.170 | -0.013 | [-0.072, +0.051] |
| `A_forman_p10_l3` | 108 | +0.363 | +0.420 | -0.057 | [-0.190, +0.035] |
| `A_forman_p90_l1` | 107 | -0.213 | -0.161 | +0.052 | [-0.041, +0.093] |
| `A_forman_p90_l2` | 108 | +0.270 | +0.170 | +0.100 | [-0.036, +0.248] |
| `A_forman_p90_l3` | 108 | +0.472 | +0.420 | +0.052 | [-0.042, +0.181] |
| `A_forman_node_incident_mean_l1` | 107 | -0.164 | -0.161 | +0.003 | [-0.008, +0.014] |
| `A_forman_node_incident_mean_l2` | 108 | +0.179 | +0.170 | +0.009 | [-0.041, +0.085] |
| `A_forman_node_incident_mean_l3` | 108 | +0.442 | +0.420 | +0.022 | [-0.041, +0.113] |
| `A_forman_node_incident_median_l1` | 107 | -0.160 | -0.161 | -0.001 | [-0.011, +0.009] |
| `A_forman_node_incident_median_l2` | 108 | +0.199 | +0.170 | +0.029 | [-0.031, +0.105] |
| `A_forman_node_incident_median_l3` | 108 | +0.480 | +0.420 | +0.060 | [-0.022, +0.177] |
| `B_sibling_density_mean_l2` | 108 | -0.145 | -0.451 | -0.306 | [-0.415, -0.166] |
| `B_sibling_density_mean_l3` | 87 | +0.332 | -0.587 | -0.256 | [-0.438, -0.079] |
| `B_sibling_density_median_parent_l2` | 108 | -0.134 | -0.451 | -0.317 | [-0.427, -0.164] |
| `B_sibling_density_median_parent_l3` | 87 | +0.233 | -0.587 | -0.354 | [-0.545, -0.162] |
| `B_sibling_density_child_weighted_l2` | 108 | -0.156 | -0.451 | -0.295 | [-0.401, -0.168] |
| `B_sibling_density_child_weighted_l3` | 87 | +0.304 | -0.587 | -0.284 | [-0.477, -0.093] |
| `B_sibling_density_item_weighted_l2` | 108 | -0.190 | -0.451 | -0.261 | [-0.366, -0.154] |
| `B_sibling_density_item_weighted_l3` | 87 | +0.163 | -0.587 | -0.425 | [-0.631, -0.177] |
| `B_sibling_transitivity_mean_l2` | 108 | -0.180 | -0.465 | -0.285 | [-0.399, -0.163] |
| `B_sibling_transitivity_mean_l3` | 81 | +0.155 | -0.620 | -0.464 | [-0.749, -0.070] |
| `B_sibling_transitivity_child_weighted_l2` | 108 | -0.235 | -0.465 | -0.230 | [-0.331, -0.130] |
| `B_sibling_transitivity_child_weighted_l3` | 81 | +0.128 | -0.620 | -0.492 | [-0.748, -0.095] |
| `B_sibling_cycle_rank_per_node_mean_l2` | 108 | +0.271 | -0.235 | +0.036 | [-0.313, +0.370] |
| `B_sibling_cycle_rank_per_node_mean_l3` | 108 | -0.081 | -0.478 | -0.397 | [-0.516, -0.176] |
| `B_sibling_degree_median_mean_l2` | 108 | +0.273 | -0.267 | +0.006 | [-0.358, +0.366] |
| `B_sibling_degree_median_mean_l3` | 108 | +0.083 | -0.562 | -0.479 | [-0.663, -0.124] |
| `B_sibling_degree_mean_mean_l2` | 108 | +0.364 | -0.267 | +0.097 | [-0.269, +0.450] |
| `B_sibling_degree_mean_mean_l3` | 108 | -0.091 | -0.562 | -0.471 | [-0.594, -0.269] |
| `B_sibling_degree_p90_mean_l2` | 108 | +0.433 | -0.200 | +0.233 | [-0.109, +0.513] |
| `B_sibling_degree_p90_mean_l3` | 108 | -0.159 | -0.476 | -0.317 | [-0.459, -0.165] |
| `B_sibling_edge_share_l2` | 108 | +0.421 | -0.451 | -0.030 | [-0.372, +0.308] |
| `B_sibling_edge_share_l3` | 108 | -0.022 | -0.479 | -0.458 | [-0.561, -0.139] |
| `D_multi_only_density_l1` | 107 | +0.140 | +0.133 | +0.008 | [-0.013, +0.023] |
| `D_multi_only_density_l2` | 108 | -0.487 | -0.451 | +0.036 | [-0.006, +0.089] |
| `D_multi_only_density_l3` | 108 | -0.427 | -0.479 | -0.052 | [-0.246, +0.123] |
| `D_multi_only_transitivity_l1` | 106 | +0.179 | +0.182 | -0.003 | [-0.011, +0.002] |
| `D_multi_only_transitivity_l2` | 84 | -0.583 | -0.561 | +0.022 | [-0.072, +0.121] |
| `D_multi_only_transitivity_l3` | 81 | -0.149 | -0.681 | -0.532 | [-0.711, -0.303] |
| `D_multi_only_cycle_rank_per_node_l1` | 108 | +0.198 | +0.194 | +0.004 | [-0.004, +0.013] |
| `D_multi_only_cycle_rank_per_node_l2` | 108 | -0.308 | -0.235 | +0.073 | [+0.023, +0.128] |
| `D_multi_only_cycle_rank_per_node_l3` | 108 | -0.497 | -0.478 | +0.019 | [-0.014, +0.058] |
| `D_multi_only_degree_median_l1` | 108 | +0.191 | +0.193 | -0.003 | [-0.008, +0.002] |
| `D_multi_only_degree_median_l2` | 108 | -0.309 | -0.267 | +0.042 | [-0.016, +0.098] |
| `D_multi_only_degree_median_l3` | 108 | -0.476 | -0.562 | -0.086 | [-0.166, -0.014] |
| `D_multi_only_forman_mean_l1` | 107 | -0.161 | -0.161 | +0.000 | [-0.001, +0.003] |
| `D_multi_only_forman_mean_l2` | 108 | +0.289 | +0.170 | +0.119 | [+0.029, +0.201] |
| `D_multi_only_forman_mean_l3` | 108 | +0.521 | +0.420 | +0.101 | [+0.012, +0.226] |
| `D_multi_only_forman_negative_fraction_l2` | 108 | -0.067 | -0.211 | -0.144 | [-0.293, +0.047] |
| `D_multi_only_forman_negative_fraction_l3` | 108 | -0.467 | -0.218 | +0.249 | [+0.002, +0.484] |
| `C_triangles_per_node_mean_global_delta_l1` | 106 | -0.297 | +0.182 | +0.115 | [-0.257, +0.423] |
| `C_triangles_per_node_mean_global_delta_l2` | 108 | +0.162 | -0.465 | -0.303 | [-0.453, -0.127] |
| `C_triangles_per_node_mean_global_delta_l3` | 108 | +0.015 | -0.438 | -0.423 | [-0.554, -0.091] |
| `C_triangles_per_node_mean_global_z_l1` | 100 | -0.125 | +0.336 | -0.212 | [-0.454, +0.103] |
| `C_triangles_per_node_mean_global_z_l2` | 107 | +0.094 | -0.450 | -0.356 | [-0.454, -0.161] |
| `C_triangles_per_node_mean_global_z_l3` | 108 | +0.118 | -0.438 | -0.320 | [-0.512, -0.061] |
| `C_triangles_per_node_mean_global_ratio_l1` | 106 | -0.068 | +0.182 | -0.114 | [-0.320, +0.223] |
| `C_triangles_per_node_mean_global_ratio_l2` | 108 | +0.106 | -0.465 | -0.359 | [-0.494, -0.157] |
| `C_triangles_per_node_mean_global_ratio_l3` | 108 | +0.008 | -0.438 | -0.430 | [-0.555, -0.094] |
| `C_triangles_per_node_mean_degree_stratified_delta_l1` | 106 | -0.389 | +0.182 | +0.207 | [-0.150, +0.499] |
| `C_triangles_per_node_mean_degree_stratified_delta_l2` | 108 | +0.174 | -0.465 | -0.291 | [-0.432, -0.135] |
| `C_triangles_per_node_mean_degree_stratified_delta_l3` | 108 | +0.120 | -0.438 | -0.318 | [-0.507, -0.084] |
| `C_triangles_per_node_mean_degree_stratified_z_l1` | 93 | -0.466 | +0.273 | +0.193 | [-0.080, +0.442] |
| `C_triangles_per_node_mean_degree_stratified_z_l2` | 107 | +0.153 | -0.450 | -0.297 | [-0.421, -0.147] |
| `C_triangles_per_node_mean_degree_stratified_z_l3` | 108 | +0.157 | -0.438 | -0.281 | [-0.456, -0.071] |
| `C_triangles_per_node_mean_degree_stratified_ratio_l1` | 106 | -0.314 | +0.182 | +0.132 | [-0.257, +0.439] |
| `C_triangles_per_node_mean_degree_stratified_ratio_l2` | 108 | +0.140 | -0.465 | -0.325 | [-0.458, -0.154] |
| `C_triangles_per_node_mean_degree_stratified_ratio_l3` | 108 | +0.134 | -0.438 | -0.304 | [-0.498, -0.076] |
| `C_triangles_per_node_mean_parent_conditioned_delta_l2` | 108 | +0.220 | -0.465 | -0.245 | [-0.424, -0.063] |
| `C_triangles_per_node_mean_parent_conditioned_delta_l3` | 108 | +0.308 | -0.438 | -0.130 | [-0.431, +0.168] |
| `C_triangles_per_node_mean_parent_conditioned_z_l2` | 107 | +0.221 | -0.450 | -0.229 | [-0.394, -0.052] |
| `C_triangles_per_node_mean_parent_conditioned_z_l3` | 98 | +0.331 | -0.488 | -0.157 | [-0.442, +0.135] |
| `C_triangles_per_node_mean_parent_conditioned_ratio_l2` | 108 | +0.218 | -0.465 | -0.247 | [-0.429, -0.059] |
| `C_triangles_per_node_mean_parent_conditioned_ratio_l3` | 108 | +0.319 | -0.438 | -0.120 | [-0.420, +0.172] |
| `C_local_transitivity_mean_global_delta_l1` | 106 | -0.306 | +0.182 | +0.124 | [-0.248, +0.430] |
| `C_local_transitivity_mean_global_delta_l2` | 108 | -0.005 | -0.465 | -0.460 | [-0.572, -0.125] |
| `C_local_transitivity_mean_global_delta_l3` | 108 | -0.055 | -0.438 | -0.383 | [-0.543, -0.096] |
| `C_local_transitivity_mean_global_z_l1` | 100 | -0.133 | +0.336 | -0.203 | [-0.453, +0.101] |
| `C_local_transitivity_mean_global_z_l2` | 107 | +0.285 | -0.450 | -0.164 | [-0.351, +0.029] |
| `C_local_transitivity_mean_global_z_l3` | 108 | +0.106 | -0.438 | -0.332 | [-0.544, -0.032] |
| `C_local_transitivity_mean_global_ratio_l1` | 106 | -0.294 | +0.182 | +0.112 | [-0.262, +0.421] |
| `C_local_transitivity_mean_global_ratio_l2` | 108 | +0.242 | -0.465 | -0.223 | [-0.410, -0.016] |
| `C_local_transitivity_mean_global_ratio_l3` | 108 | -0.053 | -0.438 | -0.385 | [-0.544, -0.098] |
| `C_local_transitivity_mean_degree_stratified_delta_l1` | 106 | -0.393 | +0.182 | +0.211 | [-0.150, +0.505] |
| `C_local_transitivity_mean_degree_stratified_delta_l2` | 108 | +0.325 | -0.465 | -0.140 | [-0.341, +0.073] |
| `C_local_transitivity_mean_degree_stratified_delta_l3` | 108 | -0.025 | -0.438 | -0.413 | [-0.546, -0.105] |
| `C_local_transitivity_mean_degree_stratified_z_l1` | 94 | -0.421 | +0.291 | +0.131 | [-0.165, +0.414] |
| `C_local_transitivity_mean_degree_stratified_z_l2` | 107 | +0.446 | -0.450 | -0.004 | [-0.203, +0.197] |
| `C_local_transitivity_mean_degree_stratified_z_l3` | 108 | +0.228 | -0.438 | -0.210 | [-0.507, +0.101] |
| `C_local_transitivity_mean_degree_stratified_ratio_l1` | 106 | -0.382 | +0.182 | +0.200 | [-0.169, +0.496] |
| `C_local_transitivity_mean_degree_stratified_ratio_l2` | 108 | +0.450 | -0.465 | -0.015 | [-0.211, +0.187] |
| `C_local_transitivity_mean_degree_stratified_ratio_l3` | 108 | +0.015 | -0.438 | -0.423 | [-0.554, -0.077] |
| `C_local_transitivity_mean_parent_conditioned_delta_l2` | 108 | -0.006 | -0.465 | -0.459 | [-0.561, -0.113] |
| `C_local_transitivity_mean_parent_conditioned_delta_l3` | 108 | -0.136 | -0.438 | -0.302 | [-0.522, -0.021] |
| `C_local_transitivity_mean_parent_conditioned_z_l2` | 107 | -0.004 | -0.450 | -0.446 | [-0.543, -0.110] |
| `C_local_transitivity_mean_parent_conditioned_z_l3` | 102 | -0.027 | -0.468 | -0.440 | [-0.593, -0.127] |
| `C_local_transitivity_mean_parent_conditioned_ratio_l2` | 108 | +0.021 | -0.465 | -0.444 | [-0.548, -0.133] |
| `C_local_transitivity_mean_parent_conditioned_ratio_l3` | 108 | -0.122 | -0.438 | -0.316 | [-0.526, -0.031] |
| `C_sibling_density_mean_global_delta_l2` | 108 | +0.228 | -0.451 | -0.223 | [-0.538, +0.122] |
| `C_sibling_density_mean_global_delta_l3` | 81 | +0.667 | -0.733 | -0.066 | [-0.182, +0.048] |
| `C_sibling_density_mean_global_z_l2` | 106 | +0.688 | -0.419 | +0.269 | [+0.117, +0.423] |
| `C_sibling_density_mean_global_z_l3` | 80 | -0.067 | -0.727 | -0.661 | [-0.775, -0.400] |
| `C_sibling_density_mean_global_ratio_l2` | 108 | +0.723 | -0.451 | +0.272 | [+0.159, +0.397] |
| `C_sibling_density_mean_global_ratio_l3` | 81 | +0.178 | -0.733 | -0.555 | [-0.790, -0.244] |
| `C_sibling_density_mean_degree_stratified_delta_l2` | 108 | +0.259 | -0.451 | -0.192 | [-0.518, +0.149] |
| `C_sibling_density_mean_degree_stratified_delta_l3` | 82 | +0.477 | -0.679 | -0.202 | [-0.430, +0.032] |
| `C_sibling_density_mean_degree_stratified_z_l2` | 106 | +0.618 | -0.419 | +0.198 | [+0.042, +0.355] |
| `C_sibling_density_mean_degree_stratified_z_l3` | 81 | -0.165 | -0.672 | -0.508 | [-0.706, -0.282] |
| `C_sibling_density_mean_degree_stratified_ratio_l2` | 108 | +0.722 | -0.451 | +0.271 | [+0.158, +0.396] |
| `C_sibling_density_mean_degree_stratified_ratio_l3` | 82 | +0.046 | -0.679 | -0.633 | [-0.771, -0.291] |
| `C_sibling_density_mean_parent_conditioned_delta_l2` | 108 | +0.023 | -0.451 | -0.428 | [-0.533, -0.100] |
| `C_sibling_density_mean_parent_conditioned_delta_l3` | 87 | +0.080 | -0.587 | -0.507 | [-0.672, -0.253] |
| `C_sibling_density_mean_parent_conditioned_z_l2` | 106 | -0.346 | -0.419 | -0.073 | [-0.335, +0.196] |
| `C_sibling_density_mean_parent_conditioned_z_l3` | 82 | -0.288 | -0.620 | -0.332 | [-0.664, +0.007] |
| `C_sibling_density_mean_parent_conditioned_ratio_l2` | 108 | -0.116 | -0.451 | -0.335 | [-0.558, -0.007] |
| `C_sibling_density_mean_parent_conditioned_ratio_l3` | 87 | +0.255 | -0.587 | -0.332 | [-0.518, -0.162] |
| `C_sibling_transitivity_mean_global_delta_l2` | 108 | +0.467 | -0.465 | +0.002 | [-0.290, +0.285] |
| `C_sibling_transitivity_mean_global_delta_l3` | 34 | +0.592 | -0.530 | +0.062 | [-0.320, +0.489] |
| `C_sibling_transitivity_mean_global_z_l2` | 106 | +0.655 | -0.434 | +0.221 | [-0.032, +0.475] |
| `C_sibling_transitivity_mean_global_z_l3` | 34 | +0.373 | -0.530 | -0.157 | [-0.607, +0.320] |
| `C_sibling_transitivity_mean_global_ratio_l2` | 108 | +0.666 | -0.465 | +0.201 | [+0.079, +0.337] |
| `C_sibling_transitivity_mean_global_ratio_l3` | 34 | +0.651 | -0.530 | +0.120 | [-0.151, +0.458] |
| `C_sibling_transitivity_mean_degree_stratified_delta_l2` | 108 | +0.419 | -0.465 | -0.046 | [-0.331, +0.238] |
| `C_sibling_transitivity_mean_degree_stratified_delta_l3` | 32 | +0.550 | -0.493 | +0.056 | [-0.385, +0.531] |
| `C_sibling_transitivity_mean_degree_stratified_z_l2` | 106 | +0.505 | -0.434 | +0.071 | [-0.202, +0.350] |
| `C_sibling_transitivity_mean_degree_stratified_z_l3` | 32 | +0.341 | -0.493 | -0.153 | [-0.647, +0.418] |
| `C_sibling_transitivity_mean_degree_stratified_ratio_l2` | 108 | +0.593 | -0.465 | +0.128 | [-0.042, +0.300] |
| `C_sibling_transitivity_mean_degree_stratified_ratio_l3` | 32 | +0.735 | -0.493 | +0.242 | [-0.123, +0.641] |
| `C_sibling_transitivity_mean_parent_conditioned_delta_l2` | 108 | -0.250 | -0.465 | -0.215 | [-0.539, +0.128] |
| `C_sibling_transitivity_mean_parent_conditioned_delta_l3` | 81 | -0.174 | -0.620 | -0.446 | [-0.730, -0.120] |
| `C_sibling_transitivity_mean_parent_conditioned_z_l2` | 106 | -0.349 | -0.434 | -0.085 | [-0.419, +0.254] |
| `C_sibling_transitivity_mean_parent_conditioned_z_l3` | 80 | -0.210 | -0.674 | -0.465 | [-0.724, -0.157] |
| `C_sibling_transitivity_mean_parent_conditioned_ratio_l2` | 108 | -0.320 | -0.465 | -0.145 | [-0.451, +0.169] |
| `C_sibling_transitivity_mean_parent_conditioned_ratio_l3` | 80 | -0.148 | -0.674 | -0.526 | [-0.749, -0.263] |

`delta |rho|` is the paired resampling difference (5000 draws, same rows, same resamples); a positive point estimate whose CI spans zero is not an improvement.

## Selection verdict

**GAIN** - the rule was frozen before the scan: a gain requires a valid geometry definition, coverage >= 0.95, the paired |rho| CI above zero and a direction that is stable within the 76 arms.

Gains passing that rule: 3. The smallest Holm-adjusted p in the whole 304-candidate family is 0.061; for these gains it is 0.061, i.e. **borderline after multiplicity correction** and not significant at 0.05. The verdict follows the frozen rule; read it as **borderline**: under a stricter rule that also demanded Holm < 0.05 the answer would be NO STABLE GAIN, and an independent SID family is required either way.

Top |rho| candidates:

| candidate | rho | delta |rho| | Holm p | within76 rho | stable |
|---|---|---|---|---|---|
| `C_sibling_transitivity_mean_degree_stratified_ratio_l3` | +0.735 | +0.242 | 0.061 | +0.685 | yes |
| `C_sibling_density_mean_global_ratio_l2` | +0.723 | +0.272 | 0.061 | +0.355 | no |
| `C_sibling_density_mean_degree_stratified_ratio_l2` | +0.722 | +0.271 | 0.061 | +0.384 | yes |
| `C_sibling_density_mean_global_z_l2` | +0.688 | +0.269 | 0.061 | +0.392 | yes |
| `C_sibling_density_mean_global_delta_l3` | +0.667 | -0.066 | 0.061 | +0.238 | no |

## Nulls, exactness, runtime

* C: {'replicates': 20, 'designs': ['global', 'degree_stratified', 'parent_conditioned'], 'level1_designs': ['global', 'degree_stratified'], 'parent_conditioned': 'L2/L3 only, each parent keeps its item count and child-count histogram'}.
* Every null series keeps its declared design: the global shuffle preserves the label multiset, the degree-stratified shuffle preserves it within each log2-degree bucket, and the parent-conditioned shuffle preserves each parent's item count and child-count histogram. z is reported only when the null sd is positive; the null Monte Carlo uncertainty is `null_mc_se` and is never used as a p-value.
* Runtime: 724.74 s/SID median, max 1033.94 s, phases {'quotient': 456.75, 'sibling': 721.93, 'multi_only': 44.97, 'nulls': 71124.87}; null applications None; peak RSS 952.2 MB.
* No cap is applied to items, groups, quotient nodes/edges or sampled groups (asserted by the 320-child test).
* #134's true-geodesic ORC is **not** imported here; the previous clipped ORC correlation is not used as evidence for any variant.

* Strict same-commit arms (n=73), the gains: `C_sibling_density_mean_global_z_l2` rho +0.407, `C_sibling_density_mean_degree_stratified_z_l2` rho +0.388, `C_sibling_density_mean_degree_stratified_ratio_l2` rho +0.422.
* The strongest gains are the **null-relative sibling density at L2**: the raw sibling density is not associated (rho -0.145, p 0.13), but the ratio/z against a size-preserving global shuffle is (rho +0.72/+0.69), so the normalisation, not the raw within-parent density, carries the association. The parent-conditioned null of the same feature is **negative** (-0.35 at L2), so the gain comes from the global size-preserving null, **not** from within-parent structure: this is not a hierarchy-preserving effect.

## Not claimed

No causal geometry-to-Recall effect, no interaction-strength weighting, no direct curvature assignment to a learned model `-c_l`, and no confirmation beyond this exploratory 108-arm scan.
