# What each pair tier contains

182 audited pairs. `_iterNN` is folded into its parent series.

## A_verified

73 pairs, 2 series; the largest series is 99% of the tier; recalls [0.0423, 0.0634].

| series | pairs |
|---|---|
| `curvature_RQ-VAE` | 72 |
| `r4_08` | 1 |

| feature | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| `z_curv_L1` | 70 | +0.150 | 0.212 | [-0.117, +0.389] |
| `z_curv_L2` | 73 | +0.129 | 0.278 | [-0.128, +0.365] |
| `z_curv_L3` | 73 | -0.044 | 0.706 | [-0.274, +0.184] |
| `z_curv_L1_minus_L3` | 70 | +0.141 | 0.239 | [-0.128, +0.379] |

## B_same_commit_and_code_path

86 pairs, 5 series; the largest series is 94% of the tier; recalls [0.0423, 0.0634].

| series | pairs |
|---|---|
| `curvature_RQ-VAE` | 81 |
| `TIGER_FORK_RQ-VAE` | 2 |
| `r4_08` | 1 |
| `TIGER_RQ-VAE` | 1 |
| `TIGER_BEHAVIOR_RQ-VAE` | 1 |

| feature | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| `z_curv_L1` | 82 | +0.053 | 0.629 | [-0.192, +0.292] |
| `z_curv_L2` | 86 | +0.074 | 0.501 | [-0.144, +0.298] |
| `z_curv_L3` | 86 | +0.029 | 0.800 | [-0.197, +0.243] |
| `z_curv_L1_minus_L3` | 82 | +0.051 | 0.644 | [-0.197, +0.289] |

## C_code_path_only

178 pairs, 40 series; the largest series is 58% of the tier; recalls [0.0306, 0.0643].

| series | pairs |
|---|---|
| `curvature_RQ-VAE` | 103 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE` | 20 |
| `TIGER_FORK_RQ-VAE` | 11 |
| `TIGER_HYPERBOLIC_RQ-VAE` | 7 |
| `TIGER_BRad-RQ` | 2 |
| `TIGER_c106_directed_codeword_margin_geometry` | 1 |
| `TIGER_behavior_residual_geometry_transport` | 1 |
| `TIGER_bootstrap_code_stability` | 1 |

| feature | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| `z_curv_L1` | 167 | -0.091 | 0.241 | [-0.232, +0.066] |
| `z_curv_L2` | 176 | +0.161 | 0.036 | [+0.009, +0.311] |
| `z_curv_L3` | 178 | +0.140 | 0.077 | [-0.007, +0.284] |
| `z_curv_L1_minus_L3` | 167 | -0.095 | 0.225 | [-0.237, +0.060] |

