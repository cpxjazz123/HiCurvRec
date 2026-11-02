# Issue #136 - AF3 curvature and the original-graph within/cross edge gap

108 content-hash-matched rows; excluded digest mismatches: 2; cohorts {'all': 108, 'curvature_RQ-VAE': 76, 'other': 32}. Family: 528 tests (66 metric-levels x 4 outcomes x 2 coefficient types), Holm over the whole family, 20000 permutations per test.

AF3 is the frozen triangle-only convention `F + 3T`, with degree and triangles always taken in the graph the edge belongs to. No ORC is recomputed, no Stage2/3 artefact is touched, and R@20/NDCG@20 are `NOT_EVALUATED`.

## Intra-prefix induced graphs

## Intra, L1

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `intra_af3_mean:test_recall@5:L1` | 108 | +0.385 | 0.0035 | -0.307 | 0.0011 | 0.113 | defined |
| `intra_af3_mean:test_recall@10:L1` | 108 | +0.403 | 0.0014 | -0.321 | 0.0008 | 0.086 | defined |
| `intra_af3_mean:test_ndcg@5:L1` | 108 | +0.385 | 0.00255 | -0.289 | 0.00265 | 0.227 | defined |
| `intra_af3_mean:test_ndcg@10:L1` | 108 | +0.396 | 0.00185 | -0.308 | 0.0013 | 0.133 | defined |
| `intra_af3_median:test_recall@5:L1` | 108 | +0.342 | 0.00615 | -0.521 | 5e-05 | 0.013 | defined |
| `intra_af3_median:test_recall@10:L1` | 108 | +0.363 | 0.003 | -0.529 | 5e-05 | 0.013 | defined |
| `intra_af3_median:test_ndcg@5:L1` | 108 | +0.340 | 0.00735 | -0.517 | 5e-05 | 0.013 | defined |
| `intra_af3_median:test_ndcg@10:L1` | 108 | +0.354 | 0.00505 | -0.531 | 5e-05 | 0.013 | defined |
| `intra_af3_p10:test_recall@5:L1` | 108 | +0.363 | 0.0027 | -0.157 | 0.1054 | 0.235 | defined |
| `intra_af3_p10:test_recall@10:L1` | 108 | +0.373 | 0.00165 | -0.170 | 0.07585 | 0.163 | defined |
| `intra_af3_p10:test_ndcg@5:L1` | 108 | +0.364 | 0.00345 | -0.135 | 0.1591 | 0.272 | defined |
| `intra_af3_p10:test_ndcg@10:L1` | 108 | +0.371 | 0.0021 | -0.152 | 0.1163 | 0.204 | defined |
| `intra_af3_p90:test_recall@5:L1` | 108 | +0.335 | 0.0084 | -0.581 | 5e-05 | 0.013 | defined |
| `intra_af3_p90:test_recall@10:L1` | 108 | +0.359 | 0.0049 | -0.586 | 5e-05 | 0.013 | defined |
| `intra_af3_p90:test_ndcg@5:L1` | 108 | +0.334 | 0.0096 | -0.584 | 5e-05 | 0.013 | defined |
| `intra_af3_p90:test_ndcg@10:L1` | 108 | +0.349 | 0.0061 | -0.590 | 5e-05 | 0.013 | defined |
| `intra_af3_negative_fraction:test_recall@5:L1` | 108 | +0.562 | 5e-05 | +0.676 | 5e-05 | 0.013 | defined |
| `intra_af3_negative_fraction:test_recall@10:L1` | 108 | +0.573 | 5e-05 | +0.670 | 5e-05 | 0.013 | defined |
| `intra_af3_negative_fraction:test_ndcg@5:L1` | 108 | +0.563 | 5e-05 | +0.690 | 5e-05 | 0.013 | defined |
| `intra_af3_negative_fraction:test_ndcg@10:L1` | 108 | +0.570 | 5e-05 | +0.693 | 5e-05 | 0.013 | defined |
| `intra_af3_edge_weighted:test_recall@5:L1` | 108 | +0.550 | 5e-05 | +0.014 | 0.8842 | 0.013 | defined |
| `intra_af3_edge_weighted:test_recall@10:L1` | 108 | +0.555 | 0.0001 | +0.002 | 0.9798 | 0.015 | defined |
| `intra_af3_edge_weighted:test_ndcg@5:L1` | 108 | +0.553 | 5e-05 | +0.029 | 0.7621 | 0.013 | defined |
| `intra_af3_edge_weighted:test_ndcg@10:L1` | 108 | +0.558 | 5e-05 | +0.015 | 0.8832 | 0.013 | defined |
| `intra_forman_mean:test_recall@5:L1` | 108 | +0.386 | 0.0025 | -0.339 | 0.00035 | 0.042 | defined |
| `intra_forman_mean:test_recall@10:L1` | 108 | +0.402 | 0.00165 | -0.345 | 0.0002 | 0.027 | defined |
| `intra_forman_mean:test_ndcg@5:L1` | 108 | +0.386 | 0.0026 | -0.327 | 0.0008 | 0.086 | defined |
| `intra_forman_mean:test_ndcg@10:L1` | 108 | +0.396 | 0.00195 | -0.340 | 0.00045 | 0.052 | defined |

## Intra, L2

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `intra_af3_mean:test_recall@5:L2` | 108 | +0.350 | 0.0042 | +0.185 | 0.05635 | 0.298 | defined |
| `intra_af3_mean:test_recall@10:L2` | 108 | +0.343 | 0.00435 | +0.224 | 0.01865 | 0.304 | defined |
| `intra_af3_mean:test_ndcg@5:L2` | 108 | +0.355 | 0.00505 | +0.170 | 0.07655 | 0.340 | defined |
| `intra_af3_mean:test_ndcg@10:L2` | 108 | +0.352 | 0.00235 | +0.187 | 0.05205 | 0.219 | defined |
| `intra_af3_median:test_recall@5:L2` | 108 | +0.418 | 0.0043 | +0.427 | 5e-05 | 0.013 | defined |
| `intra_af3_median:test_recall@10:L2` | 108 | +0.429 | 0.00195 | +0.432 | 5e-05 | 0.013 | defined |
| `intra_af3_median:test_ndcg@5:L2` | 108 | +0.418 | 0.0038 | +0.429 | 5e-05 | 0.013 | defined |
| `intra_af3_median:test_ndcg@10:L2` | 108 | +0.426 | 0.0033 | +0.428 | 5e-05 | 0.013 | defined |
| `intra_af3_p10:test_recall@5:L2` | 108 | +0.297 | 0.01255 | +0.167 | 0.08585 | 0.616 | defined |
| `intra_af3_p10:test_recall@10:L2` | 108 | +0.286 | 0.0126 | +0.205 | 0.03355 | 0.616 | defined |
| `intra_af3_p10:test_ndcg@5:L2` | 108 | +0.302 | 0.0112 | +0.146 | 0.1346 | 0.605 | defined |
| `intra_af3_p10:test_ndcg@10:L2` | 108 | +0.296 | 0.0124 | +0.163 | 0.09295 | 0.616 | defined |
| `intra_af3_p90:test_recall@5:L2` | 108 | +0.347 | 0.01145 | +0.305 | 0.00035 | 0.042 | defined |
| `intra_af3_p90:test_recall@10:L2` | 108 | +0.353 | 0.0115 | +0.299 | 0.0005 | 0.056 | defined |
| `intra_af3_p90:test_ndcg@5:L2` | 108 | +0.347 | 0.01095 | +0.307 | 0.0003 | 0.037 | defined |
| `intra_af3_p90:test_ndcg@10:L2` | 108 | +0.352 | 0.01105 | +0.303 | 0.0005 | 0.056 | defined |
| `intra_af3_negative_fraction:test_recall@5:L2` | 108 | -0.443 | 0.0001 | -0.166 | 0.08755 | 0.015 | defined |
| `intra_af3_negative_fraction:test_recall@10:L2` | 108 | -0.462 | 0.00015 | -0.205 | 0.0341 | 0.021 | defined |
| `intra_af3_negative_fraction:test_ndcg@5:L2` | 108 | -0.448 | 0.0002 | -0.152 | 0.1159 | 0.027 | defined |
| `intra_af3_negative_fraction:test_ndcg@10:L2` | 108 | -0.459 | 0.0002 | -0.168 | 0.08175 | 0.027 | defined |
| `intra_af3_edge_weighted:test_recall@5:L2` | 108 | +0.372 | 0.0028 | +0.156 | 0.1059 | 0.241 | defined |
| `intra_af3_edge_weighted:test_recall@10:L2` | 108 | +0.359 | 0.00255 | +0.193 | 0.0458 | 0.227 | defined |
| `intra_af3_edge_weighted:test_ndcg@5:L2` | 108 | +0.381 | 0.0023 | +0.143 | 0.1418 | 0.216 | defined |
| `intra_af3_edge_weighted:test_ndcg@10:L2` | 108 | +0.374 | 0.00245 | +0.158 | 0.105 | 0.225 | defined |
| `intra_forman_mean:test_recall@5:L2` | 108 | +0.363 | 0.0035 | +0.135 | 0.1612 | 0.272 | defined |
| `intra_forman_mean:test_recall@10:L2` | 108 | +0.355 | 0.00285 | +0.172 | 0.07635 | 0.242 | defined |
| `intra_forman_mean:test_ndcg@5:L2` | 108 | +0.368 | 0.00305 | +0.118 | 0.2259 | 0.250 | defined |
| `intra_forman_mean:test_ndcg@10:L2` | 108 | +0.365 | 0.0025 | +0.134 | 0.1674 | 0.225 | defined |

## Intra, L3

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `intra_af3_mean:test_recall@5:L3` | 108 | +0.229 | 0.0311 | +0.220 | 0.0219 | 0.701 | defined |
| `intra_af3_mean:test_recall@10:L3` | 108 | +0.215 | 0.0372 | +0.269 | 0.005 | 0.340 | defined |
| `intra_af3_mean:test_ndcg@5:L3` | 108 | +0.239 | 0.0275 | +0.194 | 0.04235 | 0.742 | defined |
| `intra_af3_mean:test_ndcg@10:L3` | 108 | +0.230 | 0.0279 | +0.222 | 0.02395 | 0.701 | defined |
| `intra_af3_median:test_recall@5:L3` | 108 | +0.267 | 0.0128 | +0.312 | 0.00035 | 0.042 | defined |
| `intra_af3_median:test_recall@10:L3` | 108 | +0.250 | 0.01755 | +0.300 | 0.0009 | 0.094 | defined |
| `intra_af3_median:test_ndcg@5:L3` | 108 | +0.278 | 0.01165 | +0.313 | 0.00055 | 0.060 | defined |
| `intra_af3_median:test_ndcg@10:L3` | 108 | +0.268 | 0.01305 | +0.311 | 0.0006 | 0.065 | defined |
| `intra_af3_p10:test_recall@5:L3` | 108 | +0.202 | 0.04435 | +0.193 | 0.0454 | 0.823 | defined |
| `intra_af3_p10:test_recall@10:L3` | 108 | +0.190 | 0.04975 | +0.240 | 0.01325 | 0.616 | defined |
| `intra_af3_p10:test_ndcg@5:L3` | 108 | +0.210 | 0.0384 | +0.179 | 0.06605 | 0.806 | defined |
| `intra_af3_p10:test_ndcg@10:L3` | 108 | +0.203 | 0.0399 | +0.207 | 0.03215 | 0.742 | defined |
| `intra_af3_p90:test_recall@5:L3` | 108 | | | | | | UNDEFINED_DEGENERATE |
| `intra_af3_p90:test_recall@10:L3` | 108 | | | | | | UNDEFINED_DEGENERATE |
| `intra_af3_p90:test_ndcg@5:L3` | 108 | | | | | | UNDEFINED_DEGENERATE |
| `intra_af3_p90:test_ndcg@10:L3` | 108 | | | | | | UNDEFINED_DEGENERATE |
| `intra_af3_negative_fraction:test_recall@5:L3` | 108 | -0.301 | 0.00585 | -0.222 | 0.02275 | 0.374 | defined |
| `intra_af3_negative_fraction:test_recall@10:L3` | 108 | -0.301 | 0.00615 | -0.241 | 0.0115 | 0.387 | defined |
| `intra_af3_negative_fraction:test_ndcg@5:L3` | 108 | -0.313 | 0.00395 | -0.223 | 0.0209 | 0.288 | defined |
| `intra_af3_negative_fraction:test_ndcg@10:L3` | 108 | -0.313 | 0.00515 | -0.229 | 0.01625 | 0.340 | defined |
| `intra_af3_edge_weighted:test_recall@5:L3` | 108 | +0.221 | 0.0329 | +0.235 | 0.0148 | 0.651 | defined |
| `intra_af3_edge_weighted:test_recall@10:L3` | 108 | +0.199 | 0.04465 | +0.279 | 0.00395 | 0.288 | defined |
| `intra_af3_edge_weighted:test_ndcg@5:L3` | 108 | +0.236 | 0.0297 | +0.208 | 0.0325 | 0.742 | defined |
| `intra_af3_edge_weighted:test_ndcg@10:L3` | 108 | +0.222 | 0.03365 | +0.234 | 0.016 | 0.672 | defined |
| `intra_forman_mean:test_recall@5:L3` | 108 | +0.271 | 0.0166 | +0.137 | 0.1525 | 0.672 | defined |
| `intra_forman_mean:test_recall@10:L3` | 108 | +0.255 | 0.0208 | +0.174 | 0.07095 | 0.691 | defined |
| `intra_forman_mean:test_ndcg@5:L3` | 108 | +0.280 | 0.01525 | +0.116 | 0.2354 | 0.656 | defined |
| `intra_forman_mean:test_ndcg@10:L3` | 108 | +0.271 | 0.016 | +0.135 | 0.1608 | 0.672 | defined |

## Inter-prefix quotient

## Inter quotient, L1

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `inter_af3_mean:test_recall@5:L1` | 107 | +0.377 | 0.0001 | +0.313 | 0.00165 | 0.015 | defined |
| `inter_af3_mean:test_recall@10:L1` | 107 | +0.394 | 0.0001 | +0.292 | 0.0025 | 0.015 | defined |
| `inter_af3_mean:test_ndcg@5:L1` | 107 | +0.393 | 0.00015 | +0.340 | 0.00065 | 0.021 | defined |
| `inter_af3_mean:test_ndcg@10:L1` | 107 | +0.400 | 0.0001 | +0.321 | 0.0008 | 0.015 | defined |
| `inter_af3_median:test_recall@5:L1` | 107 | +0.384 | 5e-05 | +0.299 | 0.0015 | 0.013 | defined |
| `inter_af3_median:test_recall@10:L1` | 107 | +0.400 | 5e-05 | +0.277 | 0.00425 | 0.013 | defined |
| `inter_af3_median:test_ndcg@5:L1` | 107 | +0.399 | 0.0001 | +0.328 | 0.00045 | 0.015 | defined |
| `inter_af3_median:test_ndcg@10:L1` | 107 | +0.406 | 5e-05 | +0.308 | 0.0012 | 0.013 | defined |
| `inter_af3_p10:test_recall@5:L1` | 107 | +0.305 | 0.0012 | +0.430 | 5e-05 | 0.013 | defined |
| `inter_af3_p10:test_recall@10:L1` | 107 | +0.319 | 0.00075 | +0.400 | 5e-05 | 0.013 | defined |
| `inter_af3_p10:test_ndcg@5:L1` | 107 | +0.319 | 0.0013 | +0.454 | 5e-05 | 0.013 | defined |
| `inter_af3_p10:test_ndcg@10:L1` | 107 | +0.324 | 0.00115 | +0.433 | 5e-05 | 0.013 | defined |
| `inter_af3_p90:test_recall@5:L1` | 107 | +0.395 | 5e-05 | +0.205 | 0.03555 | 0.013 | defined |
| `inter_af3_p90:test_recall@10:L1` | 107 | +0.415 | 0.0001 | +0.191 | 0.0457 | 0.015 | defined |
| `inter_af3_p90:test_ndcg@5:L1` | 107 | +0.408 | 0.0001 | +0.238 | 0.0135 | 0.015 | defined |
| `inter_af3_p90:test_ndcg@10:L1` | 107 | +0.418 | 5e-05 | +0.221 | 0.0227 | 0.013 | defined |
| `inter_af3_negative_fraction:test_recall@5:L1` | 107 | -0.124 | 0.1902 | -0.118 | 0.2218 | 1.000 | defined |
| `inter_af3_negative_fraction:test_recall@10:L1` | 107 | -0.143 | 0.1376 | -0.083 | 0.3975 | 1.000 | defined |
| `inter_af3_negative_fraction:test_ndcg@5:L1` | 107 | -0.135 | 0.1519 | -0.145 | 0.1353 | 1.000 | defined |
| `inter_af3_negative_fraction:test_ndcg@10:L1` | 107 | -0.143 | 0.1333 | -0.126 | 0.1979 | 1.000 | defined |
| `inter_forman_mean:test_recall@5:L1` | 107 | -0.403 | 5e-05 | -0.172 | 0.07575 | 0.013 | defined |
| `inter_forman_mean:test_recall@10:L1` | 107 | -0.423 | 0.0001 | -0.161 | 0.09365 | 0.015 | defined |
| `inter_forman_mean:test_ndcg@5:L1` | 107 | -0.412 | 0.0001 | -0.200 | 0.03855 | 0.015 | defined |
| `inter_forman_mean:test_ndcg@10:L1` | 107 | -0.423 | 5e-05 | -0.184 | 0.0587 | 0.013 | defined |
| `inter_density:test_recall@5:L1` | 107 | +0.074 | 0.4465 | +0.170 | 0.0818 | 1.000 | defined |
| `inter_density:test_recall@10:L1` | 107 | +0.088 | 0.3636 | +0.133 | 0.1728 | 1.000 | defined |
| `inter_density:test_ndcg@5:L1` | 107 | +0.087 | 0.365 | +0.196 | 0.0433 | 0.823 | defined |
| `inter_density:test_ndcg@10:L1` | 107 | +0.091 | 0.3457 | +0.175 | 0.0709 | 1.000 | defined |
| `inter_transitivity:test_recall@5:L1` | 106 | +0.091 | 0.3441 | +0.227 | 0.018 | 0.680 | defined |
| `inter_transitivity:test_recall@10:L1` | 106 | +0.099 | 0.3069 | +0.182 | 0.05975 | 0.956 | defined |
| `inter_transitivity:test_ndcg@5:L1` | 106 | +0.101 | 0.2932 | +0.249 | 0.00905 | 0.519 | defined |
| `inter_transitivity:test_ndcg@10:L1` | 106 | +0.102 | 0.2961 | +0.225 | 0.02225 | 0.701 | defined |

## Inter quotient, L2

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `inter_af3_mean:test_recall@5:L2` | 108 | -0.389 | 0.00025 | -0.033 | 0.7247 | 0.031 | defined |
| `inter_af3_mean:test_recall@10:L2` | 108 | -0.360 | 0.00025 | +0.006 | 0.9488 | 0.031 | defined |
| `inter_af3_mean:test_ndcg@5:L2` | 108 | -0.397 | 0.00015 | -0.062 | 0.5221 | 0.021 | defined |
| `inter_af3_mean:test_ndcg@10:L2` | 108 | -0.382 | 0.0002 | -0.042 | 0.6668 | 0.027 | defined |
| `inter_af3_median:test_recall@5:L2` | 108 | -0.245 | 0.00985 | +0.016 | 0.869 | 0.552 | defined |
| `inter_af3_median:test_recall@10:L2` | 108 | -0.205 | 0.03475 | +0.067 | 0.4907 | 0.764 | defined |
| `inter_af3_median:test_ndcg@5:L2` | 108 | -0.253 | 0.0089 | -0.015 | 0.8786 | 0.519 | defined |
| `inter_af3_median:test_ndcg@10:L2` | 108 | -0.232 | 0.01625 | +0.013 | 0.8932 | 0.672 | defined |
| `inter_af3_p10:test_recall@5:L2` | 108 | -0.451 | 5e-05 | +0.047 | 0.6287 | 0.013 | defined |
| `inter_af3_p10:test_recall@10:L2` | 108 | -0.435 | 5e-05 | +0.060 | 0.534 | 0.013 | defined |
| `inter_af3_p10:test_ndcg@5:L2` | 108 | -0.459 | 0.0001 | +0.015 | 0.8783 | 0.015 | defined |
| `inter_af3_p10:test_ndcg@10:L2` | 108 | -0.451 | 5e-05 | +0.026 | 0.7884 | 0.013 | defined |
| `inter_af3_p90:test_recall@5:L2` | 108 | -0.265 | 0.007 | -0.039 | 0.6852 | 0.427 | defined |
| `inter_af3_p90:test_recall@10:L2` | 108 | -0.226 | 0.0192 | +0.010 | 0.9122 | 0.691 | defined |
| `inter_af3_p90:test_ndcg@5:L2` | 108 | -0.268 | 0.0062 | -0.068 | 0.4835 | 0.387 | defined |
| `inter_af3_p90:test_ndcg@10:L2` | 108 | -0.248 | 0.0105 | -0.041 | 0.6775 | 0.577 | defined |
| `inter_af3_negative_fraction:test_recall@5:L2` | 108 | +0.653 | 5e-05 | +0.232 | 0.01565 | 0.013 | defined |
| `inter_af3_negative_fraction:test_recall@10:L2` | 108 | +0.640 | 5e-05 | +0.221 | 0.0236 | 0.013 | defined |
| `inter_af3_negative_fraction:test_ndcg@5:L2` | 108 | +0.655 | 5e-05 | +0.261 | 0.0059 | 0.013 | defined |
| `inter_af3_negative_fraction:test_ndcg@10:L2` | 108 | +0.651 | 5e-05 | +0.251 | 0.00895 | 0.013 | defined |
| `inter_forman_mean:test_recall@5:L2` | 108 | -0.046 | 0.6379 | +0.117 | 0.2338 | 1.000 | defined |
| `inter_forman_mean:test_recall@10:L2` | 108 | +0.008 | 0.9327 | +0.170 | 0.07855 | 1.000 | defined |
| `inter_forman_mean:test_ndcg@5:L2` | 108 | -0.053 | 0.5864 | +0.087 | 0.3729 | 1.000 | defined |
| `inter_forman_mean:test_ndcg@10:L2` | 108 | -0.023 | 0.8152 | +0.113 | 0.2396 | 1.000 | defined |
| `inter_density:test_recall@5:L2` | 108 | -0.687 | 5e-05 | -0.404 | 5e-05 | 0.013 | defined |
| `inter_density:test_recall@10:L2` | 108 | -0.675 | 5e-05 | -0.451 | 5e-05 | 0.013 | defined |
| `inter_density:test_ndcg@5:L2` | 108 | -0.690 | 5e-05 | -0.376 | 0.00025 | 0.013 | defined |
| `inter_density:test_ndcg@10:L2` | 108 | -0.687 | 5e-05 | -0.399 | 0.0001 | 0.013 | defined |
| `inter_transitivity:test_recall@5:L2` | 108 | -0.696 | 5e-05 | -0.415 | 5e-05 | 0.013 | defined |
| `inter_transitivity:test_recall@10:L2` | 108 | -0.705 | 5e-05 | -0.465 | 5e-05 | 0.013 | defined |
| `inter_transitivity:test_ndcg@5:L2` | 108 | -0.699 | 5e-05 | -0.386 | 5e-05 | 0.013 | defined |
| `inter_transitivity:test_ndcg@10:L2` | 108 | -0.706 | 5e-05 | -0.411 | 5e-05 | 0.013 | defined |

## Inter quotient, L3

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `inter_af3_mean:test_recall@5:L3` | 108 | -0.008 | 0.923 | +0.227 | 0.0179 | 0.680 | defined |
| `inter_af3_mean:test_recall@10:L3` | 108 | +0.039 | 0.6575 | +0.282 | 0.0034 | 0.272 | defined |
| `inter_af3_mean:test_ndcg@5:L3` | 108 | -0.027 | 0.7693 | +0.191 | 0.0473 | 0.823 | defined |
| `inter_af3_mean:test_ndcg@10:L3` | 108 | +0.003 | 0.9762 | +0.218 | 0.023 | 0.701 | defined |
| `inter_af3_median:test_recall@5:L3` | 108 | +0.074 | 0.4177 | +0.282 | 0.0034 | 0.272 | defined |
| `inter_af3_median:test_recall@10:L3` | 108 | +0.132 | 0.1638 | +0.345 | 0.00025 | 0.031 | defined |
| `inter_af3_median:test_ndcg@5:L3` | 108 | +0.063 | 0.4879 | +0.250 | 0.0084 | 0.504 | defined |
| `inter_af3_median:test_ndcg@10:L3` | 108 | +0.096 | 0.304 | +0.283 | 0.0035 | 0.272 | defined |
| `inter_af3_p10:test_recall@5:L3` | 108 | -0.222 | 0.02885 | +0.225 | 0.01945 | 0.691 | defined |
| `inter_af3_p10:test_recall@10:L3` | 108 | -0.176 | 0.0697 | +0.267 | 0.00505 | 0.340 | defined |
| `inter_af3_p10:test_ndcg@5:L3` | 108 | -0.242 | 0.02025 | +0.199 | 0.0393 | 0.691 | defined |
| `inter_af3_p10:test_ndcg@10:L3` | 108 | -0.214 | 0.03295 | +0.217 | 0.023 | 0.701 | defined |
| `inter_af3_p90:test_recall@5:L3` | 108 | -0.007 | 0.9364 | +0.270 | 0.0038 | 0.281 | defined |
| `inter_af3_p90:test_recall@10:L3` | 108 | +0.049 | 0.5788 | +0.330 | 0.0004 | 0.047 | defined |
| `inter_af3_p90:test_ndcg@5:L3` | 108 | -0.019 | 0.8204 | +0.239 | 0.01185 | 0.616 | defined |
| `inter_af3_p90:test_ndcg@10:L3` | 108 | +0.013 | 0.8793 | +0.269 | 0.00455 | 0.314 | defined |
| `inter_af3_negative_fraction:test_recall@5:L3` | 108 | +0.533 | 0.0008 | +0.077 | 0.431 | 0.086 | defined |
| `inter_af3_negative_fraction:test_recall@10:L3` | 108 | +0.494 | 0.00135 | +0.074 | 0.4502 | 0.136 | defined |
| `inter_af3_negative_fraction:test_ndcg@5:L3` | 108 | +0.549 | 0.00135 | +0.062 | 0.5269 | 0.136 | defined |
| `inter_af3_negative_fraction:test_ndcg@10:L3` | 108 | +0.527 | 0.001 | +0.064 | 0.5095 | 0.104 | defined |
| `inter_forman_mean:test_recall@5:L3` | 108 | +0.533 | 5e-05 | +0.364 | 0.0001 | 0.013 | defined |
| `inter_forman_mean:test_recall@10:L3` | 108 | +0.571 | 5e-05 | +0.420 | 5e-05 | 0.013 | defined |
| `inter_forman_mean:test_ndcg@5:L3` | 108 | +0.529 | 5e-05 | +0.329 | 0.00065 | 0.013 | defined |
| `inter_forman_mean:test_ndcg@10:L3` | 108 | +0.553 | 5e-05 | +0.357 | 0.0003 | 0.013 | defined |
| `inter_density:test_recall@5:L3` | 108 | -0.591 | 0.0001 | -0.430 | 5e-05 | 0.013 | defined |
| `inter_density:test_recall@10:L3` | 108 | -0.556 | 5e-05 | -0.479 | 5e-05 | 0.013 | defined |
| `inter_density:test_ndcg@5:L3` | 108 | -0.608 | 0.0001 | -0.399 | 5e-05 | 0.013 | defined |
| `inter_density:test_ndcg@10:L3` | 108 | -0.588 | 0.0003 | -0.425 | 5e-05 | 0.013 | defined |
| `inter_transitivity:test_recall@5:L3` | 108 | -0.649 | 5e-05 | -0.386 | 5e-05 | 0.013 | defined |
| `inter_transitivity:test_recall@10:L3` | 108 | -0.631 | 5e-05 | -0.438 | 5e-05 | 0.013 | defined |
| `inter_transitivity:test_ndcg@5:L3` | 108 | -0.665 | 5e-05 | -0.355 | 0.00035 | 0.013 | defined |
| `inter_transitivity:test_ndcg@10:L3` | 108 | -0.655 | 5e-05 | -0.383 | 0.0001 | 0.013 | defined |

## Original-graph within/cross edge gap

## Gap, L1

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `gap_af3:test_recall@5:L1` | 107 | +0.823 | 5e-05 | +0.788 | 5e-05 | 0.013 | defined |
| `gap_af3:test_recall@10:L1` | 107 | +0.809 | 5e-05 | +0.760 | 5e-05 | 0.013 | defined |
| `gap_af3:test_ndcg@5:L1` | 107 | +0.832 | 5e-05 | +0.814 | 5e-05 | 0.013 | defined |
| `gap_af3:test_ndcg@10:L1` | 107 | +0.826 | 5e-05 | +0.802 | 5e-05 | 0.013 | defined |
| `gap_forman:test_recall@5:L1` | 107 | +0.817 | 5e-05 | +0.780 | 5e-05 | 0.013 | defined |
| `gap_forman:test_recall@10:L1` | 107 | +0.801 | 5e-05 | +0.750 | 5e-05 | 0.013 | defined |
| `gap_forman:test_ndcg@5:L1` | 107 | +0.826 | 5e-05 | +0.805 | 5e-05 | 0.013 | defined |
| `gap_forman:test_ndcg@10:L1` | 107 | +0.819 | 5e-05 | +0.793 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_recall@5:L1` | 107 | +0.791 | 5e-05 | +0.740 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_recall@10:L1` | 107 | +0.778 | 5e-05 | +0.713 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_ndcg@5:L1` | 107 | +0.802 | 5e-05 | +0.772 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_ndcg@10:L1` | 107 | +0.796 | 5e-05 | +0.759 | 5e-05 | 0.013 | defined |
| `gap_cross_mean:test_recall@5:L1` | 107 | -0.731 | 5e-05 | -0.612 | 5e-05 | 0.013 | defined |
| `gap_cross_mean:test_recall@10:L1` | 107 | -0.715 | 5e-05 | -0.583 | 5e-05 | 0.013 | defined |
| `gap_cross_mean:test_ndcg@5:L1` | 107 | -0.732 | 5e-05 | -0.638 | 5e-05 | 0.013 | defined |
| `gap_cross_mean:test_ndcg@10:L1` | 107 | -0.726 | 5e-05 | -0.631 | 5e-05 | 0.013 | defined |
| `edge_retention:test_recall@5:L1` | 108 | -0.349 | 0.00305 | +0.211 | 0.02675 | 0.250 | defined |
| `edge_retention:test_recall@10:L1` | 108 | -0.361 | 0.0018 | +0.209 | 0.0311 | 0.176 | defined |
| `edge_retention:test_ndcg@5:L1` | 108 | -0.351 | 0.00365 | +0.210 | 0.0303 | 0.274 | defined |
| `edge_retention:test_ndcg@10:L1` | 108 | -0.359 | 0.0029 | +0.221 | 0.02215 | 0.242 | defined |
| `gap_af3_z_global:test_recall@5:L1` | 107 | +0.726 | 5e-05 | +0.574 | 5e-05 | 0.013 | defined |
| `gap_af3_z_global:test_recall@10:L1` | 107 | +0.706 | 5e-05 | +0.542 | 5e-05 | 0.013 | defined |
| `gap_af3_z_global:test_ndcg@5:L1` | 107 | +0.739 | 5e-05 | +0.601 | 5e-05 | 0.013 | defined |
| `gap_af3_z_global:test_ndcg@10:L1` | 107 | +0.729 | 5e-05 | +0.582 | 5e-05 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_recall@5:L1` | 107 | +0.564 | 5e-05 | +0.348 | 0.0003 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_recall@10:L1` | 107 | +0.552 | 5e-05 | +0.347 | 0.00045 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_ndcg@5:L1` | 107 | +0.559 | 5e-05 | +0.349 | 0.0004 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_ndcg@10:L1` | 107 | +0.556 | 5e-05 | +0.363 | 0.0002 | 0.013 | defined |

## Gap, L2

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `gap_af3:test_recall@5:L2` | 108 | +0.771 | 5e-05 | +0.651 | 5e-05 | 0.013 | defined |
| `gap_af3:test_recall@10:L2` | 108 | +0.773 | 5e-05 | +0.685 | 5e-05 | 0.013 | defined |
| `gap_af3:test_ndcg@5:L2` | 108 | +0.770 | 5e-05 | +0.616 | 5e-05 | 0.013 | defined |
| `gap_af3:test_ndcg@10:L2` | 108 | +0.775 | 5e-05 | +0.634 | 5e-05 | 0.013 | defined |
| `gap_forman:test_recall@5:L2` | 108 | +0.763 | 5e-05 | +0.647 | 5e-05 | 0.013 | defined |
| `gap_forman:test_recall@10:L2` | 108 | +0.762 | 5e-05 | +0.679 | 5e-05 | 0.013 | defined |
| `gap_forman:test_ndcg@5:L2` | 108 | +0.763 | 5e-05 | +0.612 | 5e-05 | 0.013 | defined |
| `gap_forman:test_ndcg@10:L2` | 108 | +0.765 | 5e-05 | +0.629 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_recall@5:L2` | 108 | +0.767 | 5e-05 | +0.641 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_recall@10:L2` | 108 | +0.770 | 5e-05 | +0.674 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_ndcg@5:L2` | 108 | +0.765 | 5e-05 | +0.606 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_ndcg@10:L2` | 108 | +0.771 | 5e-05 | +0.624 | 5e-05 | 0.013 | defined |
| `gap_cross_mean:test_recall@5:L2` | 108 | -0.355 | 0.0021 | -0.179 | 0.06485 | 0.204 | defined |
| `gap_cross_mean:test_recall@10:L2` | 108 | -0.325 | 0.00285 | -0.141 | 0.1453 | 0.242 | defined |
| `gap_cross_mean:test_ndcg@5:L2` | 108 | -0.360 | 0.0021 | -0.197 | 0.04045 | 0.204 | defined |
| `gap_cross_mean:test_ndcg@10:L2` | 108 | -0.344 | 0.00245 | -0.182 | 0.0618 | 0.225 | defined |
| `edge_retention:test_recall@5:L2` | 108 | -0.292 | 0.0129 | -0.107 | 0.2732 | 0.616 | defined |
| `edge_retention:test_recall@10:L2` | 108 | -0.282 | 0.01155 | -0.144 | 0.1415 | 0.612 | defined |
| `edge_retention:test_ndcg@5:L2` | 108 | -0.300 | 0.0121 | -0.090 | 0.3537 | 0.616 | defined |
| `edge_retention:test_ndcg@10:L2` | 108 | -0.294 | 0.01195 | -0.106 | 0.2765 | 0.616 | defined |
| `gap_af3_z_global:test_recall@5:L2` | 82 | +0.559 | 0.00015 | +0.062 | 0.5843 | 0.021 | defined |
| `gap_af3_z_global:test_recall@10:L2` | 82 | +0.524 | 0.0002 | +0.017 | 0.8835 | 0.027 | defined |
| `gap_af3_z_global:test_ndcg@5:L2` | 82 | +0.560 | 0.00035 | +0.052 | 0.6401 | 0.042 | defined |
| `gap_af3_z_global:test_ndcg@10:L2` | 82 | +0.543 | 0.0005 | +0.031 | 0.7864 | 0.056 | defined |
| `gap_af3_z_degree_stratified:test_recall@5:L2` | 83 | -0.281 | 0.01415 | -0.410 | 0.0002 | 0.027 | defined |
| `gap_af3_z_degree_stratified:test_recall@10:L2` | 83 | -0.298 | 0.0079 | -0.439 | 5e-05 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_ndcg@5:L2` | 83 | -0.291 | 0.01085 | -0.421 | 5e-05 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_ndcg@10:L2` | 83 | -0.299 | 0.0086 | -0.431 | 0.0002 | 0.027 | defined |

## Gap, L3

| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |
|---|---|---|---|---|---|---|---|
| `gap_af3:test_recall@5:L3` | 108 | +0.612 | 5e-05 | +0.492 | 5e-05 | 0.013 | defined |
| `gap_af3:test_recall@10:L3` | 108 | +0.600 | 5e-05 | +0.524 | 5e-05 | 0.013 | defined |
| `gap_af3:test_ndcg@5:L3` | 108 | +0.608 | 5e-05 | +0.452 | 5e-05 | 0.013 | defined |
| `gap_af3:test_ndcg@10:L3` | 108 | +0.605 | 5e-05 | +0.470 | 5e-05 | 0.013 | defined |
| `gap_forman:test_recall@5:L3` | 108 | +0.582 | 0.00015 | +0.485 | 5e-05 | 0.013 | defined |
| `gap_forman:test_recall@10:L3` | 108 | +0.568 | 0.00015 | +0.516 | 5e-05 | 0.013 | defined |
| `gap_forman:test_ndcg@5:L3` | 108 | +0.578 | 0.0003 | +0.444 | 0.0001 | 0.015 | defined |
| `gap_forman:test_ndcg@10:L3` | 108 | +0.574 | 0.00015 | +0.462 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_recall@5:L3` | 108 | +0.612 | 0.00015 | +0.490 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_recall@10:L3` | 108 | +0.599 | 5e-05 | +0.522 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_ndcg@5:L3` | 108 | +0.608 | 0.0001 | +0.450 | 5e-05 | 0.013 | defined |
| `gap_within_mean:test_ndcg@10:L3` | 108 | +0.605 | 5e-05 | +0.469 | 5e-05 | 0.013 | defined |
| `gap_cross_mean:test_recall@5:L3` | 108 | -0.063 | 0.4323 | +0.030 | 0.7581 | 1.000 | defined |
| `gap_cross_mean:test_recall@10:L3` | 108 | -0.059 | 0.4677 | +0.075 | 0.4468 | 1.000 | defined |
| `gap_cross_mean:test_ndcg@5:L3` | 108 | -0.066 | 0.4124 | +0.010 | 0.9143 | 1.000 | defined |
| `gap_cross_mean:test_ndcg@10:L3` | 108 | -0.063 | 0.4338 | +0.033 | 0.7383 | 1.000 | defined |
| `edge_retention:test_recall@5:L3` | 108 | -0.204 | 0.04335 | -0.209 | 0.0292 | 0.742 | defined |
| `edge_retention:test_recall@10:L3` | 108 | -0.183 | 0.05525 | -0.252 | 0.0088 | 0.519 | defined |
| `edge_retention:test_ndcg@5:L3` | 108 | -0.214 | 0.03845 | -0.190 | 0.0486 | 0.806 | defined |
| `edge_retention:test_ndcg@10:L3` | 108 | -0.202 | 0.0455 | -0.211 | 0.0282 | 0.742 | defined |
| `gap_af3_z_global:test_recall@5:L3` | 81 | +0.473 | 0.0042 | -0.387 | 0.0004 | 0.047 | defined |
| `gap_af3_z_global:test_recall@10:L3` | 81 | +0.435 | 0.0052 | -0.420 | 0.0001 | 0.015 | defined |
| `gap_af3_z_global:test_ndcg@5:L3` | 81 | +0.483 | 0.0046 | -0.389 | 0.00045 | 0.052 | defined |
| `gap_af3_z_global:test_ndcg@10:L3` | 81 | +0.462 | 0.00435 | -0.408 | 0.00025 | 0.031 | defined |
| `gap_af3_z_degree_stratified:test_recall@5:L3` | 81 | -0.487 | 0.0003 | -0.589 | 5e-05 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_recall@10:L3` | 81 | -0.501 | 0.00015 | -0.597 | 5e-05 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_ndcg@5:L3` | 81 | -0.504 | 0.0002 | -0.594 | 5e-05 | 0.013 | defined |
| `gap_af3_z_degree_stratified:test_ndcg@10:L3` | 81 | -0.510 | 0.0003 | -0.593 | 5e-05 | 0.013 | defined |

## Incremental test: gap beyond retention and beyond ordinary Forman

| gap vs outcome | level | control | type | n | partial coefficient | p |
|---|---|---|---|---|---|---|
| `test_recall@5` | 1 | edge_retention | partial_pearson | 107 | +0.804 | 0.0002 |
| `test_recall@5` | 1 | edge_retention | partial_spearman | 107 | +0.780 | 0.0002 |
| `test_recall@5` | 1 | gap_forman | partial_pearson | 107 | +0.260 | 0.0112 |
| `test_recall@5` | 1 | gap_forman | partial_spearman | 107 | +0.632 | 0.0002 |
| `test_recall@5` | 1 | intra_transitivity | partial_pearson | 107 | +0.482 | 0.0002 |
| `test_recall@5` | 1 | intra_transitivity | partial_spearman | 107 | +0.285 | 0.004599 |
| `test_recall@10` | 1 | edge_retention | partial_pearson | 107 | +0.788 | 0.0002 |
| `test_recall@10` | 1 | edge_retention | partial_spearman | 107 | +0.750 | 0.0002 |
| `test_recall@10` | 1 | gap_forman | partial_pearson | 107 | +0.320 | 0.0018 |
| `test_recall@10` | 1 | gap_forman | partial_spearman | 107 | +0.593 | 0.0002 |
| `test_recall@10` | 1 | intra_transitivity | partial_pearson | 107 | +0.429 | 0.0007998 |
| `test_recall@10` | 1 | intra_transitivity | partial_spearman | 107 | +0.224 | 0.0186 |
| `test_ndcg@5` | 1 | edge_retention | partial_pearson | 107 | +0.815 | 0.0002 |
| `test_ndcg@5` | 1 | edge_retention | partial_spearman | 107 | +0.806 | 0.0002 |
| `test_ndcg@5` | 1 | gap_forman | partial_pearson | 107 | +0.264 | 0.008798 |
| `test_ndcg@5` | 1 | gap_forman | partial_spearman | 107 | +0.669 | 0.0002 |
| `test_ndcg@5` | 1 | intra_transitivity | partial_pearson | 107 | +0.502 | 0.0003999 |
| `test_ndcg@5` | 1 | intra_transitivity | partial_spearman | 107 | +0.339 | 0.0003999 |
| `test_ndcg@10` | 1 | edge_retention | partial_pearson | 107 | +0.808 | 0.0002 |
| `test_ndcg@10` | 1 | edge_retention | partial_spearman | 107 | +0.794 | 0.0002 |
| `test_ndcg@10` | 1 | gap_forman | partial_pearson | 107 | +0.300 | 0.003599 |
| `test_ndcg@10` | 1 | gap_forman | partial_spearman | 107 | +0.652 | 0.0002 |
| `test_ndcg@10` | 1 | intra_transitivity | partial_pearson | 107 | +0.476 | 0.0003999 |
| `test_ndcg@10` | 1 | intra_transitivity | partial_spearman | 107 | +0.313 | 0.0012 |
| `test_recall@5` | 2 | edge_retention | partial_pearson | 108 | +0.746 | 0.0002 |
| `test_recall@5` | 2 | edge_retention | partial_spearman | 108 | +0.616 | 0.0002 |
| `test_recall@5` | 2 | gap_forman | partial_pearson | 108 | +0.183 | 0.06439 |
| `test_recall@5` | 2 | gap_forman | partial_spearman | 108 | +0.461 | 0.0002 |
| `test_recall@5` | 2 | inter_density | partial_pearson | 108 | +0.507 | 0.0002 |
| `test_recall@5` | 2 | inter_density | partial_spearman | 108 | +0.577 | 0.0002 |
| `test_recall@10` | 2 | edge_retention | partial_pearson | 108 | +0.751 | 0.0002 |
| `test_recall@10` | 2 | edge_retention | partial_spearman | 108 | +0.655 | 0.0002 |
| `test_recall@10` | 2 | gap_forman | partial_pearson | 108 | +0.256 | 0.009598 |
| `test_recall@10` | 2 | gap_forman | partial_spearman | 108 | +0.507 | 0.0002 |
| `test_recall@10` | 2 | inter_density | partial_pearson | 108 | +0.526 | 0.0002 |
| `test_recall@10` | 2 | inter_density | partial_spearman | 108 | +0.618 | 0.0002 |
| `test_ndcg@5` | 2 | edge_retention | partial_pearson | 108 | +0.743 | 0.0002 |
| `test_ndcg@5` | 2 | edge_retention | partial_spearman | 108 | +0.576 | 0.0002 |
| `test_ndcg@5` | 2 | gap_forman | partial_pearson | 108 | +0.167 | 0.08198 |
| `test_ndcg@5` | 2 | gap_forman | partial_spearman | 108 | +0.412 | 0.0002 |
| `test_ndcg@5` | 2 | inter_density | partial_pearson | 108 | +0.501 | 0.0002 |
| `test_ndcg@5` | 2 | inter_density | partial_spearman | 108 | +0.534 | 0.0002 |
| `test_ndcg@10` | 2 | edge_retention | partial_pearson | 108 | +0.750 | 0.0002 |
| `test_ndcg@10` | 2 | edge_retention | partial_spearman | 108 | +0.597 | 0.0002 |
| `test_ndcg@10` | 2 | gap_forman | partial_pearson | 108 | +0.212 | 0.03059 |
| `test_ndcg@10` | 2 | gap_forman | partial_spearman | 108 | +0.437 | 0.0002 |
| `test_ndcg@10` | 2 | inter_density | partial_pearson | 108 | +0.516 | 0.0002 |
| `test_ndcg@10` | 2 | inter_density | partial_spearman | 108 | +0.556 | 0.0002 |
| `test_recall@5` | 3 | edge_retention | partial_pearson | 108 | +0.597 | 0.0002 |
| `test_recall@5` | 3 | edge_retention | partial_spearman | 108 | +0.453 | 0.0002 |
| `test_recall@5` | 3 | gap_forman | partial_pearson | 108 | +0.467 | 0.0002 |
| `test_recall@5` | 3 | gap_forman | partial_spearman | 108 | +0.407 | 0.0002 |
| `test_recall@5` | 3 | inter_density | partial_pearson | 108 | +0.474 | 0.0002 |
| `test_recall@5` | 3 | inter_density | partial_spearman | 108 | +0.455 | 0.0002 |
| `test_recall@10` | 3 | edge_retention | partial_pearson | 108 | +0.586 | 0.0003999 |
| `test_recall@10` | 3 | edge_retention | partial_spearman | 108 | +0.491 | 0.0002 |
| `test_recall@10` | 3 | gap_forman | partial_pearson | 108 | +0.484 | 0.0002 |
| `test_recall@10` | 3 | gap_forman | partial_spearman | 108 | +0.443 | 0.0002 |
| `test_recall@10` | 3 | inter_density | partial_pearson | 108 | +0.466 | 0.0002 |
| `test_recall@10` | 3 | inter_density | partial_spearman | 108 | +0.490 | 0.0002 |
| `test_ndcg@5` | 3 | edge_retention | partial_pearson | 108 | +0.593 | 0.0003999 |
| `test_ndcg@5` | 3 | edge_retention | partial_spearman | 108 | +0.410 | 0.0002 |
| `test_ndcg@5` | 3 | gap_forman | partial_pearson | 108 | +0.463 | 0.0002 |
| `test_ndcg@5` | 3 | gap_forman | partial_spearman | 108 | +0.363 | 0.0003999 |
| `test_ndcg@5` | 3 | inter_density | partial_pearson | 108 | +0.466 | 0.0002 |
| `test_ndcg@5` | 3 | inter_density | partial_spearman | 108 | +0.412 | 0.0002 |
| `test_ndcg@10` | 3 | edge_retention | partial_pearson | 108 | +0.591 | 0.0003999 |
| `test_ndcg@10` | 3 | edge_retention | partial_spearman | 108 | +0.431 | 0.0002 |
| `test_ndcg@10` | 3 | gap_forman | partial_pearson | 108 | +0.476 | 0.0002 |
| `test_ndcg@10` | 3 | gap_forman | partial_spearman | 108 | +0.384 | 0.0002 |
| `test_ndcg@10` | 3 | inter_density | partial_pearson | 108 | +0.466 | 0.0002 |
| `test_ndcg@10` | 3 | inter_density | partial_spearman | 108 | +0.432 | 0.0002 |

## Metric redundancy

| left | right | level | n | Pearson | Spearman |
|---|---|---|---|---|---|
| `intra_af3_mean` | `intra_transitivity` | 1 | 108 | +0.391 | -0.174 |
| `intra_af3_mean` | `intra_transitivity` | 2 | 82 | +0.386 | +0.446 |
| `intra_af3_mean` | `intra_transitivity` | 3 | 81 | +0.317 | +0.561 |
| `intra_af3_mean` | `intra_conductance` | 1 | 107 | +0.345 | +0.758 |
| `intra_af3_mean` | `intra_conductance` | 2 | 108 | +0.972 | +0.778 |
| `intra_af3_mean` | `intra_conductance` | 3 | 108 | +0.314 | +0.444 |
| `gap_af3` | `edge_retention` | 1 | 107 | -0.390 | +0.161 |
| `gap_af3` | `edge_retention` | 2 | 108 | -0.373 | -0.608 |
| `gap_af3` | `edge_retention` | 3 | 108 | -0.178 | -0.779 |
| `inter_af3_mean` | `inter_density` | 1 | 107 | +0.538 | +0.452 |
| `inter_af3_mean` | `inter_density` | 2 | 108 | +0.751 | -0.425 |
| `inter_af3_mean` | `inter_density` | 3 | 108 | +0.614 | -0.711 |
| `inter_af3_mean` | `inter_transitivity` | 1 | 106 | +0.516 | +0.490 |
| `inter_af3_mean` | `inter_transitivity` | 2 | 108 | +0.643 | -0.432 |
| `inter_af3_mean` | `inter_transitivity` | 3 | 108 | +0.487 | -0.672 |
| `gap_af3` | `orc_mean` | 1 | 107 | -0.224 | +0.359 |
| `gap_af3` | `orc_mean` | 2 | 108 | -0.794 | -0.762 |
| `gap_af3` | `orc_mean` | 3 | 108 | -0.478 | -0.686 |

## Verdict

* (i) Does adding triangles change the curvature association? **YES** - the strongest AF3-vs-outcome entry at any level and the Holm table above decide it; the ordinary-Forman rows are reported beside every AF3 row for the same graph.
* (ii) Does the original-edge gap add beyond basic edge retention? YES: the partial Pearson of gap_af3 vs R@10 controlling edge retention is 0.7881027846589761 / 0.7507349376377254 / 0.5860046571439954 at L1/L2/L3 (p = 2e-4); against the ordinary-Forman gap the AF3 increment is 0.31972057716017827 / 0.2560358350721396 / 0.48427760912149515.
* (iii) Which effects hold within one RQ-VAE series? gap vs R@10 L1: r +0.526, rho +0.522; gap vs R@10 L2: r +0.608, rho +0.457; gap vs R@10 L3: r +0.451, rho +0.253. The intra AF3 mean has a Pearson/Spearman sign conflict (r +0.37, rho -0.53 at L1), so it is outlier-driven and listed as mixed-arm.
* (iv) What is computationally blocked? {'d3_prefix_alignment': 'NOT_AVAILABLE: no D3/edge-retention implementation exists in the repository', 'af3_vs_orc_same_edges': 'NOT_AVAILABLE_BY_DESIGN: #134 stores SID-level ORC aggregates only, not its per-edge sampled values', 'r20_ndcg20': 'NOT_EVALUATED: the saved Stage3 artifacts only contain topk [5, 10]'}.

### AF3 is not a model curvature

AF3 is a chosen graph-curvature convention; it is not the hyperbolic sectional curvature `-c_l`, and a large gap does not prove hyperbolic necessity. The gap is measured on the same original edges under a fixed curvature cache, so its per-SID variation comes only from prefix membership.
