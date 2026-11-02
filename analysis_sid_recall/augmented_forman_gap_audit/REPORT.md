# Issue #137 audit - corrected inference and AF3 negative-edge semantics

108 content-hash-matched rows; cohorts {'all': 108, 'curvature_RQ-VAE': 76, 'other': 32}; provenance {'HASH_PROVEN': 0, 'RECORD_HASH_MATCH': 108, 'MISSING': 0}.

## What was wrong in #136 and is now SUPERSEDED

* **B1**: the permutation p-value resampled outcomes WITH replacement, i.e. it was a bootstrap null with duplicated and missing observations, not the finite-sample permutation null, and its Spearman statistic did not re-rank the resampled sample. The corrected null draws a uniform **bijection** of all n outcome indices for every draw and uses `(1 + #{|stat_perm| >= |stat_obs|}) / (B + 1)`. The bootstrap stays a with-replacement paired resampling and is reported **only** as a CI.
* **B2**: the old single `holm_p` was Holm over `min(p_r, p_rho)` across 260 metric/outcome entries, which is not an adjusted p for either coefficient. The corrected analysis reports `p_pearson`, `p_spearman`, `holm_pearson`, `holm_spearman`, each adjusted within its own column over the full family (528 potential coefficient tests; 528 defined; 0 undefined).
* **B3**: the old `intra_af3_negative_fraction` was the fraction of GROUPS with a negative mean AF3, not the fraction of negative EDGES. Both are published now, with the ordinary-Forman analogues, plus the unweighted mean of the per-group edge fractions (`feature_semantics_audit.csv`).

The archived #136 report is unchanged and its p-values, Holm column and the misleading negative-edge field are **SUPERSEDED**; `new_vs_old_correlations.csv` shows both.

## What is retracted from #136

* The headline "intra AF3 negative-edge fraction at L1 is strongly associated (rho +0.670)" was the **group-mean** fraction. The true negative-edge fraction at L1 is rho +0.197 (Holm 1.000). The two measurements must not be substituted.
* The old p-values came from a bootstrap null (B1) and the old single `holm_p` was not an adjusted p for either coefficient (B2); every adjusted p in #136 is replaced here.
* The claim that the gap adds beyond ordinary Forman, retention and transitivity **at every level** is downgraded: the three-control partial is significant for both coefficient types only at L2; at L1 the Pearson partial is -0.088 (p 0.35) and at L3 the Spearman partial is -0.034 (p 0.73), and the leave-series-out RMSE improves at L1/L2 but worsens at L3.
* The implication that the gap result is null-insensitive is downgraded: the degree-stratified z changes sign at L2/L3. The null-centred gap remains positive at every level, so the raw association is stable but its z-score is null-model dependent.
* What **stands**: the raw `Gap_AF3` association itself (H1) survives the corrected permutation inference, the per-column Holm correction, the within-76 cohort and the removal of Cook-flagged arms.

Pearson and Spearman disagree by sign for several intra AF3 statistics (for example `intra_af3_median` R@10: r +0.605, rho -0.055). The AF3 scale carries the original-graph degree sums, so a few high-degree arms drive the linear fit; the rank coefficient is the outlier-robust statement and the influence table lists the flagged arms.

## Primary family (#136 metric set): 528 defined coefficient tests

### Primary, L1

| metric:outcome | n | Pearson r | p_r | Holm_r | Spearman rho | p_rho | Holm_rho |
|---|---|---|---|---|---|---|---|
| `intra_af3_mean:test_recall@5:L1` | 108 | +0.385 | 0.00295 | 0.360 | -0.307 | 0.00165 | 0.269 |
| `intra_af3_mean:test_recall@10:L1` | 108 | +0.403 | 0.0015 | 0.193 | -0.321 | 0.0009 | 0.151 |
| `intra_af3_mean:test_ndcg@5:L1` | 108 | +0.385 | 0.0042 | 0.437 | -0.289 | 0.0023 | 0.361 |
| `intra_af3_mean:test_ndcg@10:L1` | 108 | +0.396 | 0.0024 | 0.300 | -0.308 | 0.00165 | 0.269 |
| `intra_af3_median:test_recall@5:L1` | 108 | +0.609 | 5e-05 | 0.013 | -0.047 | 0.6295 | 1.000 |
| `intra_af3_median:test_recall@10:L1` | 108 | +0.605 | 5e-05 | 0.013 | -0.055 | 0.5679 | 1.000 |
| `intra_af3_median:test_ndcg@5:L1` | 108 | +0.613 | 5e-05 | 0.013 | -0.035 | 0.7199 | 1.000 |
| `intra_af3_median:test_ndcg@10:L1` | 108 | +0.613 | 5e-05 | 0.013 | -0.049 | 0.6217 | 1.000 |
| `intra_af3_p10:test_recall@5:L1` | 108 | +0.540 | 5e-05 | 0.013 | +0.067 | 0.498 | 1.000 |
| `intra_af3_p10:test_recall@10:L1` | 108 | +0.546 | 5e-05 | 0.013 | +0.057 | 0.5583 | 1.000 |
| `intra_af3_p10:test_ndcg@5:L1` | 108 | +0.542 | 5e-05 | 0.013 | +0.084 | 0.3839 | 1.000 |
| `intra_af3_p10:test_ndcg@10:L1` | 108 | +0.548 | 0.0001 | 0.017 | +0.072 | 0.4639 | 1.000 |
| `intra_af3_p90:test_recall@5:L1` | 108 | +0.340 | 0.00375 | 0.412 | -0.133 | 0.1682 | 1.000 |
| `intra_af3_p90:test_recall@10:L1` | 108 | +0.354 | 0.0023 | 0.290 | -0.125 | 0.2023 | 1.000 |
| `intra_af3_p90:test_ndcg@5:L1` | 108 | +0.342 | 0.0049 | 0.495 | -0.126 | 0.1921 | 1.000 |
| `intra_af3_p90:test_ndcg@10:L1` | 108 | +0.351 | 0.0032 | 0.378 | -0.128 | 0.1851 | 1.000 |
| `intra_negative_edge_fraction:test_recall@5:L1` | 108 | +0.121 | 0.2113 | 1.000 | +0.195 | 0.04345 | 1.000 |
| `intra_negative_edge_fraction:test_recall@10:L1` | 108 | +0.113 | 0.2412 | 1.000 | +0.197 | 0.0379 | 1.000 |
| `intra_negative_edge_fraction:test_ndcg@5:L1` | 108 | +0.113 | 0.2427 | 1.000 | +0.196 | 0.0423 | 1.000 |
| `intra_negative_edge_fraction:test_ndcg@10:L1` | 108 | +0.111 | 0.2512 | 1.000 | +0.207 | 0.0298 | 1.000 |
| `intra_af3_edge_weighted:test_recall@5:L1` | 108 | +0.550 | 5e-05 | 0.013 | +0.014 | 0.8883 | 1.000 |
| `intra_af3_edge_weighted:test_recall@10:L1` | 108 | +0.555 | 5e-05 | 0.013 | +0.002 | 0.981 | 1.000 |
| `intra_af3_edge_weighted:test_ndcg@5:L1` | 108 | +0.553 | 0.0001 | 0.017 | +0.029 | 0.7618 | 1.000 |
| `intra_af3_edge_weighted:test_ndcg@10:L1` | 108 | +0.558 | 5e-05 | 0.013 | +0.015 | 0.883 | 1.000 |
| `intra_forman_mean:test_recall@5:L1` | 108 | +0.386 | 0.0036 | 0.403 | -0.339 | 0.0004 | 0.070 |
| `intra_forman_mean:test_recall@10:L1` | 108 | +0.402 | 0.00155 | 0.198 | -0.345 | 0.0002 | 0.037 |
| `intra_forman_mean:test_ndcg@5:L1` | 108 | +0.386 | 0.00395 | 0.419 | -0.327 | 0.00055 | 0.093 |
| `intra_forman_mean:test_ndcg@10:L1` | 108 | +0.396 | 0.003 | 0.363 | -0.340 | 0.00025 | 0.045 |
| `inter_af3_mean:test_recall@5:L1` | 107 | +0.377 | 0.00015 | 0.024 | +0.313 | 0.0011 | 0.183 |
| `inter_af3_mean:test_recall@10:L1` | 107 | +0.394 | 5e-05 | 0.013 | +0.292 | 0.0015 | 0.247 |
| `inter_af3_mean:test_ndcg@5:L1` | 107 | +0.393 | 5e-05 | 0.013 | +0.340 | 0.00035 | 0.063 |
| `inter_af3_mean:test_ndcg@10:L1` | 107 | +0.400 | 0.0001 | 0.017 | +0.321 | 0.00085 | 0.144 |
| `inter_af3_median:test_recall@5:L1` | 107 | +0.384 | 5e-05 | 0.013 | +0.299 | 0.0016 | 0.262 |
| `inter_af3_median:test_recall@10:L1` | 107 | +0.400 | 5e-05 | 0.013 | +0.277 | 0.0044 | 0.664 |
| `inter_af3_median:test_ndcg@5:L1` | 107 | +0.399 | 5e-05 | 0.013 | +0.328 | 0.00035 | 0.063 |
| `inter_af3_median:test_ndcg@10:L1` | 107 | +0.406 | 5e-05 | 0.013 | +0.308 | 0.00165 | 0.269 |
| `inter_af3_p10:test_recall@5:L1` | 107 | +0.305 | 0.00165 | 0.210 | +0.430 | 5e-05 | 0.013 |
| `inter_af3_p10:test_recall@10:L1` | 107 | +0.319 | 0.00095 | 0.128 | +0.400 | 5e-05 | 0.013 |
| `inter_af3_p10:test_ndcg@5:L1` | 107 | +0.319 | 0.00085 | 0.116 | +0.454 | 5e-05 | 0.013 |
| `inter_af3_p10:test_ndcg@10:L1` | 107 | +0.324 | 0.0005 | 0.072 | +0.433 | 5e-05 | 0.013 |
| `inter_af3_p90:test_recall@5:L1` | 107 | +0.395 | 0.0001 | 0.017 | +0.205 | 0.03555 | 1.000 |
| `inter_af3_p90:test_recall@10:L1` | 107 | +0.415 | 5e-05 | 0.013 | +0.191 | 0.04755 | 1.000 |
| `inter_af3_p90:test_ndcg@5:L1` | 107 | +0.408 | 0.00015 | 0.024 | +0.238 | 0.01395 | 1.000 |
| `inter_af3_p90:test_ndcg@10:L1` | 107 | +0.418 | 5e-05 | 0.013 | +0.221 | 0.02275 | 1.000 |
| `inter_negative_edge_fraction:test_recall@5:L1` | 107 | -0.124 | 0.1934 | 1.000 | -0.118 | 0.2239 | 1.000 |
| `inter_negative_edge_fraction:test_recall@10:L1` | 107 | -0.143 | 0.1339 | 1.000 | -0.083 | 0.3925 | 1.000 |
| `inter_negative_edge_fraction:test_ndcg@5:L1` | 107 | -0.135 | 0.1523 | 1.000 | -0.145 | 0.1373 | 1.000 |
| `inter_negative_edge_fraction:test_ndcg@10:L1` | 107 | -0.143 | 0.1315 | 1.000 | -0.126 | 0.1965 | 1.000 |
| `inter_forman_mean:test_recall@5:L1` | 107 | -0.403 | 0.0001 | 0.017 | -0.172 | 0.0715 | 1.000 |
| `inter_forman_mean:test_recall@10:L1` | 107 | -0.423 | 5e-05 | 0.013 | -0.161 | 0.097 | 1.000 |
| `inter_forman_mean:test_ndcg@5:L1` | 107 | -0.412 | 5e-05 | 0.013 | -0.200 | 0.03765 | 1.000 |
| `inter_forman_mean:test_ndcg@10:L1` | 107 | -0.423 | 0.0001 | 0.017 | -0.184 | 0.05605 | 1.000 |
| `inter_density:test_recall@5:L1` | 107 | +0.074 | 0.4467 | 1.000 | +0.170 | 0.0794 | 1.000 |
| `inter_density:test_recall@10:L1` | 107 | +0.088 | 0.3661 | 1.000 | +0.133 | 0.1719 | 1.000 |
| `inter_density:test_ndcg@5:L1` | 107 | +0.087 | 0.3704 | 1.000 | +0.196 | 0.0441 | 1.000 |
| `inter_density:test_ndcg@10:L1` | 107 | +0.091 | 0.3459 | 1.000 | +0.175 | 0.07185 | 1.000 |
| `inter_transitivity:test_recall@5:L1` | 106 | +0.091 | 0.3499 | 1.000 | +0.227 | 0.0183 | 1.000 |
| `inter_transitivity:test_recall@10:L1` | 106 | +0.099 | 0.3031 | 1.000 | +0.182 | 0.05745 | 1.000 |
| `inter_transitivity:test_ndcg@5:L1` | 106 | +0.101 | 0.296 | 1.000 | +0.249 | 0.01035 | 1.000 |
| `inter_transitivity:test_ndcg@10:L1` | 106 | +0.102 | 0.288 | 1.000 | +0.225 | 0.02115 | 1.000 |
| `gap_gap_af3:test_recall@5:L1` | 107 | +0.823 | 5e-05 | 0.013 | +0.788 | 5e-05 | 0.013 |
| `gap_gap_af3:test_recall@10:L1` | 107 | +0.809 | 5e-05 | 0.013 | +0.760 | 5e-05 | 0.013 |
| `gap_gap_af3:test_ndcg@5:L1` | 107 | +0.832 | 5e-05 | 0.013 | +0.814 | 5e-05 | 0.013 |
| `gap_gap_af3:test_ndcg@10:L1` | 107 | +0.826 | 5e-05 | 0.013 | +0.802 | 5e-05 | 0.013 |
| `gap_gap_forman:test_recall@5:L1` | 107 | +0.817 | 5e-05 | 0.013 | +0.780 | 5e-05 | 0.013 |
| `gap_gap_forman:test_recall@10:L1` | 107 | +0.801 | 5e-05 | 0.013 | +0.750 | 5e-05 | 0.013 |
| `gap_gap_forman:test_ndcg@5:L1` | 107 | +0.826 | 5e-05 | 0.013 | +0.805 | 5e-05 | 0.013 |
| `gap_gap_forman:test_ndcg@10:L1` | 107 | +0.819 | 5e-05 | 0.013 | +0.793 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_recall@5:L1` | 107 | +0.791 | 5e-05 | 0.013 | +0.740 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_recall@10:L1` | 107 | +0.778 | 5e-05 | 0.013 | +0.713 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_ndcg@5:L1` | 107 | +0.802 | 5e-05 | 0.013 | +0.772 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_ndcg@10:L1` | 107 | +0.796 | 5e-05 | 0.013 | +0.759 | 5e-05 | 0.013 |
| `gap_gap_cross_mean:test_recall@5:L1` | 107 | -0.731 | 5e-05 | 0.013 | -0.612 | 5e-05 | 0.013 |
| `gap_gap_cross_mean:test_recall@10:L1` | 107 | -0.715 | 5e-05 | 0.013 | -0.583 | 5e-05 | 0.013 |
| `gap_gap_cross_mean:test_ndcg@5:L1` | 107 | -0.732 | 5e-05 | 0.013 | -0.638 | 5e-05 | 0.013 |
| `gap_gap_cross_mean:test_ndcg@10:L1` | 107 | -0.726 | 5e-05 | 0.013 | -0.631 | 5e-05 | 0.013 |
| `gap_edge_retention:test_recall@5:L1` | 107 | -0.297 | 0.0127 | 1.000 | +0.244 | 0.01075 | 1.000 |
| `gap_edge_retention:test_recall@10:L1` | 107 | -0.298 | 0.0117 | 0.994 | +0.242 | 0.0129 | 1.000 |
| `gap_edge_retention:test_ndcg@5:L1` | 107 | -0.299 | 0.01435 | 1.000 | +0.243 | 0.01215 | 1.000 |
| `gap_edge_retention:test_ndcg@10:L1` | 107 | -0.300 | 0.0124 | 1.000 | +0.254 | 0.00855 | 1.000 |
| `gap_z_global:test_recall@5:L1` | 107 | +0.726 | 5e-05 | 0.013 | +0.574 | 5e-05 | 0.013 |
| `gap_z_global:test_recall@10:L1` | 107 | +0.706 | 5e-05 | 0.013 | +0.542 | 5e-05 | 0.013 |
| `gap_z_global:test_ndcg@5:L1` | 107 | +0.739 | 5e-05 | 0.013 | +0.601 | 5e-05 | 0.013 |
| `gap_z_global:test_ndcg@10:L1` | 107 | +0.729 | 5e-05 | 0.013 | +0.582 | 5e-05 | 0.013 |
| `gap_z_degree_stratified:test_recall@5:L1` | 107 | +0.564 | 5e-05 | 0.013 | +0.348 | 0.0003 | 0.054 |
| `gap_z_degree_stratified:test_recall@10:L1` | 107 | +0.552 | 5e-05 | 0.013 | +0.347 | 0.00035 | 0.063 |
| `gap_z_degree_stratified:test_ndcg@5:L1` | 107 | +0.559 | 5e-05 | 0.013 | +0.349 | 0.00045 | 0.078 |
| `gap_z_degree_stratified:test_ndcg@10:L1` | 107 | +0.556 | 5e-05 | 0.013 | +0.363 | 0.0002 | 0.037 |

### Primary, L2

| metric:outcome | n | Pearson r | p_r | Holm_r | Spearman rho | p_rho | Holm_rho |
|---|---|---|---|---|---|---|---|
| `intra_af3_mean:test_recall@5:L2` | 108 | +0.350 | 0.00535 | 0.519 | +0.185 | 0.0554 | 1.000 |
| `intra_af3_mean:test_recall@10:L2` | 108 | +0.343 | 0.0051 | 0.505 | +0.224 | 0.02005 | 1.000 |
| `intra_af3_mean:test_ndcg@5:L2` | 108 | +0.355 | 0.00525 | 0.514 | +0.170 | 0.0754 | 1.000 |
| `intra_af3_mean:test_ndcg@10:L2` | 108 | +0.352 | 0.00505 | 0.505 | +0.187 | 0.05375 | 1.000 |
| `intra_af3_median:test_recall@5:L2` | 108 | +0.428 | 0.0011 | 0.145 | +0.170 | 0.07855 | 1.000 |
| `intra_af3_median:test_recall@10:L2` | 108 | +0.410 | 0.00105 | 0.140 | +0.188 | 0.052 | 1.000 |
| `intra_af3_median:test_ndcg@5:L2` | 108 | +0.441 | 0.001 | 0.134 | +0.177 | 0.0688 | 1.000 |
| `intra_af3_median:test_ndcg@10:L2` | 108 | +0.430 | 0.0012 | 0.156 | +0.179 | 0.06585 | 1.000 |
| `intra_af3_p10:test_recall@5:L2` | 108 | +0.370 | 0.00335 | 0.389 | +0.135 | 0.1613 | 1.000 |
| `intra_af3_p10:test_recall@10:L2` | 108 | +0.359 | 0.00305 | 0.366 | +0.169 | 0.07995 | 1.000 |
| `intra_af3_p10:test_ndcg@5:L2` | 108 | +0.379 | 0.00365 | 0.405 | +0.118 | 0.2229 | 1.000 |
| `intra_af3_p10:test_ndcg@10:L2` | 108 | +0.373 | 0.00305 | 0.366 | +0.133 | 0.1707 | 1.000 |
| `intra_af3_p90:test_recall@5:L2` | 108 | +0.187 | 0.0529 | 1.000 | +0.286 | 0.00225 | 0.355 |
| `intra_af3_p90:test_recall@10:L2` | 108 | +0.175 | 0.06815 | 1.000 | +0.283 | 0.0024 | 0.374 |
| `intra_af3_p90:test_ndcg@5:L2` | 108 | +0.196 | 0.046 | 1.000 | +0.286 | 0.002 | 0.320 |
| `intra_af3_p90:test_ndcg@10:L2` | 108 | +0.188 | 0.0506 | 1.000 | +0.284 | 0.00205 | 0.326 |
| `intra_negative_edge_fraction:test_recall@5:L2` | 108 | -0.398 | 0.0002 | 0.030 | -0.143 | 0.1392 | 1.000 |
| `intra_negative_edge_fraction:test_recall@10:L2` | 108 | -0.423 | 0.0001 | 0.017 | -0.180 | 0.0592 | 1.000 |
| `intra_negative_edge_fraction:test_ndcg@5:L2` | 108 | -0.406 | 0.00015 | 0.024 | -0.129 | 0.1836 | 1.000 |
| `intra_negative_edge_fraction:test_ndcg@10:L2` | 108 | -0.420 | 0.0001 | 0.017 | -0.145 | 0.136 | 1.000 |
| `intra_af3_edge_weighted:test_recall@5:L2` | 108 | +0.372 | 0.00325 | 0.380 | +0.156 | 0.1055 | 1.000 |
| `intra_af3_edge_weighted:test_recall@10:L2` | 108 | +0.359 | 0.0038 | 0.412 | +0.193 | 0.0469 | 1.000 |
| `intra_af3_edge_weighted:test_ndcg@5:L2` | 108 | +0.381 | 0.00435 | 0.444 | +0.143 | 0.1429 | 1.000 |
| `intra_af3_edge_weighted:test_ndcg@10:L2` | 108 | +0.374 | 0.00375 | 0.412 | +0.158 | 0.1021 | 1.000 |
| `intra_forman_mean:test_recall@5:L2` | 108 | +0.363 | 0.00415 | 0.436 | +0.135 | 0.1625 | 1.000 |
| `intra_forman_mean:test_recall@10:L2` | 108 | +0.355 | 0.00355 | 0.401 | +0.172 | 0.0742 | 1.000 |
| `intra_forman_mean:test_ndcg@5:L2` | 108 | +0.368 | 0.00425 | 0.438 | +0.118 | 0.2211 | 1.000 |
| `intra_forman_mean:test_ndcg@10:L2` | 108 | +0.365 | 0.00385 | 0.412 | +0.134 | 0.1683 | 1.000 |
| `inter_af3_mean:test_recall@5:L2` | 108 | -0.389 | 0.00015 | 0.024 | -0.033 | 0.7328 | 1.000 |
| `inter_af3_mean:test_recall@10:L2` | 108 | -0.360 | 0.0005 | 0.072 | +0.006 | 0.9457 | 1.000 |
| `inter_af3_mean:test_ndcg@5:L2` | 108 | -0.397 | 0.0001 | 0.017 | -0.062 | 0.5188 | 1.000 |
| `inter_af3_mean:test_ndcg@10:L2` | 108 | -0.382 | 0.00025 | 0.038 | -0.042 | 0.6607 | 1.000 |
| `inter_af3_median:test_recall@5:L2` | 108 | -0.245 | 0.0114 | 0.980 | +0.016 | 0.872 | 1.000 |
| `inter_af3_median:test_recall@10:L2` | 108 | -0.205 | 0.0336 | 1.000 | +0.067 | 0.4927 | 1.000 |
| `inter_af3_median:test_ndcg@5:L2` | 108 | -0.253 | 0.00845 | 0.781 | -0.015 | 0.8797 | 1.000 |
| `inter_af3_median:test_ndcg@10:L2` | 108 | -0.232 | 0.01865 | 1.000 | +0.013 | 0.8951 | 1.000 |
| `inter_af3_p10:test_recall@5:L2` | 108 | -0.451 | 0.00015 | 0.024 | +0.047 | 0.6326 | 1.000 |
| `inter_af3_p10:test_recall@10:L2` | 108 | -0.435 | 5e-05 | 0.013 | +0.060 | 0.535 | 1.000 |
| `inter_af3_p10:test_ndcg@5:L2` | 108 | -0.459 | 5e-05 | 0.013 | +0.015 | 0.8746 | 1.000 |
| `inter_af3_p10:test_ndcg@10:L2` | 108 | -0.451 | 5e-05 | 0.013 | +0.026 | 0.7949 | 1.000 |
| `inter_af3_p90:test_recall@5:L2` | 108 | -0.265 | 0.0073 | 0.693 | -0.039 | 0.6818 | 1.000 |
| `inter_af3_p90:test_recall@10:L2` | 108 | -0.226 | 0.02025 | 1.000 | +0.010 | 0.9177 | 1.000 |
| `inter_af3_p90:test_ndcg@5:L2` | 108 | -0.268 | 0.00575 | 0.552 | -0.068 | 0.4805 | 1.000 |
| `inter_af3_p90:test_ndcg@10:L2` | 108 | -0.248 | 0.01245 | 1.000 | -0.041 | 0.6795 | 1.000 |
| `inter_negative_edge_fraction:test_recall@5:L2` | 108 | +0.653 | 5e-05 | 0.013 | +0.232 | 0.0161 | 1.000 |
| `inter_negative_edge_fraction:test_recall@10:L2` | 108 | +0.640 | 5e-05 | 0.013 | +0.221 | 0.02305 | 1.000 |
| `inter_negative_edge_fraction:test_ndcg@5:L2` | 108 | +0.655 | 5e-05 | 0.013 | +0.261 | 0.0054 | 0.792 |
| `inter_negative_edge_fraction:test_ndcg@10:L2` | 108 | +0.651 | 5e-05 | 0.013 | +0.251 | 0.00845 | 1.000 |
| `inter_forman_mean:test_recall@5:L2` | 108 | -0.046 | 0.6409 | 1.000 | +0.117 | 0.2246 | 1.000 |
| `inter_forman_mean:test_recall@10:L2` | 108 | +0.008 | 0.9352 | 1.000 | +0.170 | 0.0791 | 1.000 |
| `inter_forman_mean:test_ndcg@5:L2` | 108 | -0.053 | 0.5844 | 1.000 | +0.087 | 0.3699 | 1.000 |
| `inter_forman_mean:test_ndcg@10:L2` | 108 | -0.023 | 0.8151 | 1.000 | +0.113 | 0.2417 | 1.000 |
| `inter_density:test_recall@5:L2` | 108 | -0.687 | 5e-05 | 0.013 | -0.404 | 5e-05 | 0.013 |
| `inter_density:test_recall@10:L2` | 108 | -0.675 | 5e-05 | 0.013 | -0.451 | 5e-05 | 0.013 |
| `inter_density:test_ndcg@5:L2` | 108 | -0.690 | 5e-05 | 0.013 | -0.376 | 0.0001 | 0.019 |
| `inter_density:test_ndcg@10:L2` | 108 | -0.687 | 5e-05 | 0.013 | -0.399 | 5e-05 | 0.013 |
| `inter_transitivity:test_recall@5:L2` | 108 | -0.696 | 5e-05 | 0.013 | -0.415 | 5e-05 | 0.013 |
| `inter_transitivity:test_recall@10:L2` | 108 | -0.705 | 5e-05 | 0.013 | -0.465 | 5e-05 | 0.013 |
| `inter_transitivity:test_ndcg@5:L2` | 108 | -0.699 | 5e-05 | 0.013 | -0.386 | 5e-05 | 0.013 |
| `inter_transitivity:test_ndcg@10:L2` | 108 | -0.706 | 5e-05 | 0.013 | -0.411 | 5e-05 | 0.013 |
| `gap_gap_af3:test_recall@5:L2` | 108 | +0.771 | 5e-05 | 0.013 | +0.651 | 5e-05 | 0.013 |
| `gap_gap_af3:test_recall@10:L2` | 108 | +0.773 | 5e-05 | 0.013 | +0.685 | 5e-05 | 0.013 |
| `gap_gap_af3:test_ndcg@5:L2` | 108 | +0.770 | 5e-05 | 0.013 | +0.616 | 5e-05 | 0.013 |
| `gap_gap_af3:test_ndcg@10:L2` | 108 | +0.775 | 5e-05 | 0.013 | +0.634 | 5e-05 | 0.013 |
| `gap_gap_forman:test_recall@5:L2` | 108 | +0.763 | 5e-05 | 0.013 | +0.647 | 5e-05 | 0.013 |
| `gap_gap_forman:test_recall@10:L2` | 108 | +0.762 | 5e-05 | 0.013 | +0.679 | 5e-05 | 0.013 |
| `gap_gap_forman:test_ndcg@5:L2` | 108 | +0.763 | 5e-05 | 0.013 | +0.612 | 5e-05 | 0.013 |
| `gap_gap_forman:test_ndcg@10:L2` | 108 | +0.765 | 5e-05 | 0.013 | +0.629 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_recall@5:L2` | 108 | +0.767 | 5e-05 | 0.013 | +0.641 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_recall@10:L2` | 108 | +0.770 | 5e-05 | 0.013 | +0.674 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_ndcg@5:L2` | 108 | +0.765 | 5e-05 | 0.013 | +0.606 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_ndcg@10:L2` | 108 | +0.771 | 5e-05 | 0.013 | +0.624 | 5e-05 | 0.013 |
| `gap_gap_cross_mean:test_recall@5:L2` | 108 | -0.355 | 0.00335 | 0.389 | -0.179 | 0.0642 | 1.000 |
| `gap_gap_cross_mean:test_recall@10:L2` | 108 | -0.325 | 0.0034 | 0.389 | -0.141 | 0.1486 | 1.000 |
| `gap_gap_cross_mean:test_ndcg@5:L2` | 108 | -0.360 | 0.00265 | 0.326 | -0.197 | 0.0419 | 1.000 |
| `gap_gap_cross_mean:test_ndcg@10:L2` | 108 | -0.344 | 0.0025 | 0.310 | -0.182 | 0.0601 | 1.000 |
| `gap_edge_retention:test_recall@5:L2` | 108 | -0.292 | 0.017 | 1.000 | -0.107 | 0.2686 | 1.000 |
| `gap_edge_retention:test_recall@10:L2` | 108 | -0.282 | 0.0156 | 1.000 | -0.144 | 0.1397 | 1.000 |
| `gap_edge_retention:test_ndcg@5:L2` | 108 | -0.300 | 0.017 | 1.000 | -0.090 | 0.3497 | 1.000 |
| `gap_edge_retention:test_ndcg@10:L2` | 108 | -0.294 | 0.0151 | 1.000 | -0.106 | 0.2706 | 1.000 |
| `gap_z_global:test_recall@5:L2` | 82 | +0.559 | 0.00025 | 0.038 | +0.062 | 0.574 | 1.000 |
| `gap_z_global:test_recall@10:L2` | 82 | +0.524 | 0.00055 | 0.078 | +0.017 | 0.8871 | 1.000 |
| `gap_z_global:test_ndcg@5:L2` | 82 | +0.560 | 0.0006 | 0.084 | +0.052 | 0.6408 | 1.000 |
| `gap_z_global:test_ndcg@10:L2` | 82 | +0.543 | 0.0004 | 0.059 | +0.031 | 0.7836 | 1.000 |
| `gap_z_degree_stratified:test_recall@5:L2` | 83 | -0.281 | 0.01365 | 1.000 | -0.410 | 0.0002 | 0.037 |
| `gap_z_degree_stratified:test_recall@10:L2` | 83 | -0.298 | 0.0084 | 0.781 | -0.439 | 5e-05 | 0.013 |
| `gap_z_degree_stratified:test_ndcg@5:L2` | 83 | -0.291 | 0.01075 | 0.935 | -0.421 | 0.0001 | 0.019 |
| `gap_z_degree_stratified:test_ndcg@10:L2` | 83 | -0.299 | 0.0092 | 0.828 | -0.431 | 5e-05 | 0.013 |

### Primary, L3

| metric:outcome | n | Pearson r | p_r | Holm_r | Spearman rho | p_rho | Holm_rho |
|---|---|---|---|---|---|---|---|
| `intra_af3_mean:test_recall@5:L3` | 108 | +0.229 | 0.0305 | 1.000 | +0.220 | 0.02265 | 1.000 |
| `intra_af3_mean:test_recall@10:L3` | 108 | +0.215 | 0.0348 | 1.000 | +0.269 | 0.00535 | 0.792 |
| `intra_af3_mean:test_ndcg@5:L3` | 108 | +0.239 | 0.0277 | 1.000 | +0.194 | 0.04225 | 1.000 |
| `intra_af3_mean:test_ndcg@10:L3` | 108 | +0.230 | 0.0296 | 1.000 | +0.222 | 0.0218 | 1.000 |
| `intra_af3_median:test_recall@5:L3` | 108 | +0.206 | 0.0415 | 1.000 | +0.124 | 0.2059 | 1.000 |
| `intra_af3_median:test_recall@10:L3` | 108 | +0.183 | 0.0561 | 1.000 | +0.142 | 0.1399 | 1.000 |
| `intra_af3_median:test_ndcg@5:L3` | 108 | +0.222 | 0.033 | 1.000 | +0.125 | 0.2049 | 1.000 |
| `intra_af3_median:test_ndcg@10:L3` | 108 | +0.207 | 0.03805 | 1.000 | +0.132 | 0.174 | 1.000 |
| `intra_af3_p10:test_recall@5:L3` | 108 | +0.255 | 0.0224 | 1.000 | +0.153 | 0.1101 | 1.000 |
| `intra_af3_p10:test_recall@10:L3` | 108 | +0.233 | 0.03085 | 1.000 | +0.188 | 0.05165 | 1.000 |
| `intra_af3_p10:test_ndcg@5:L3` | 108 | +0.269 | 0.01695 | 1.000 | +0.128 | 0.1866 | 1.000 |
| `intra_af3_p10:test_ndcg@10:L3` | 108 | +0.255 | 0.02105 | 1.000 | +0.145 | 0.1338 | 1.000 |
| `intra_af3_p90:test_recall@5:L3` | 108 | -0.140 | 0.1114 | 1.000 | +0.442 | 5e-05 | 0.013 |
| `intra_af3_p90:test_recall@10:L3` | 108 | -0.143 | 0.1084 | 1.000 | +0.447 | 5e-05 | 0.013 |
| `intra_af3_p90:test_ndcg@5:L3` | 108 | -0.129 | 0.1291 | 1.000 | +0.436 | 5e-05 | 0.013 |
| `intra_af3_p90:test_ndcg@10:L3` | 108 | -0.134 | 0.1235 | 1.000 | +0.444 | 5e-05 | 0.013 |
| `intra_negative_edge_fraction:test_recall@5:L3` | 108 | -0.355 | 0.00115 | 0.151 | -0.218 | 0.0239 | 1.000 |
| `intra_negative_edge_fraction:test_recall@10:L3` | 108 | -0.361 | 0.00075 | 0.103 | -0.249 | 0.01015 | 1.000 |
| `intra_negative_edge_fraction:test_ndcg@5:L3` | 108 | -0.370 | 0.00085 | 0.116 | -0.203 | 0.03685 | 1.000 |
| `intra_negative_edge_fraction:test_ndcg@10:L3` | 108 | -0.373 | 0.0006 | 0.084 | -0.220 | 0.02085 | 1.000 |
| `intra_af3_edge_weighted:test_recall@5:L3` | 108 | +0.221 | 0.03425 | 1.000 | +0.235 | 0.01435 | 1.000 |
| `intra_af3_edge_weighted:test_recall@10:L3` | 108 | +0.199 | 0.0438 | 1.000 | +0.279 | 0.00345 | 0.524 |
| `intra_af3_edge_weighted:test_ndcg@5:L3` | 108 | +0.236 | 0.0252 | 1.000 | +0.208 | 0.03105 | 1.000 |
| `intra_af3_edge_weighted:test_ndcg@10:L3` | 108 | +0.222 | 0.0318 | 1.000 | +0.234 | 0.01575 | 1.000 |
| `intra_forman_mean:test_recall@5:L3` | 108 | +0.271 | 0.01785 | 1.000 | +0.137 | 0.155 | 1.000 |
| `intra_forman_mean:test_recall@10:L3` | 108 | +0.255 | 0.02195 | 1.000 | +0.174 | 0.07255 | 1.000 |
| `intra_forman_mean:test_ndcg@5:L3` | 108 | +0.280 | 0.01725 | 1.000 | +0.116 | 0.2336 | 1.000 |
| `intra_forman_mean:test_ndcg@10:L3` | 108 | +0.271 | 0.01805 | 1.000 | +0.135 | 0.1583 | 1.000 |
| `inter_af3_mean:test_recall@5:L3` | 108 | -0.008 | 0.9249 | 1.000 | +0.227 | 0.0188 | 1.000 |
| `inter_af3_mean:test_recall@10:L3` | 108 | +0.039 | 0.6564 | 1.000 | +0.282 | 0.00325 | 0.500 |
| `inter_af3_mean:test_ndcg@5:L3` | 108 | -0.027 | 0.7529 | 1.000 | +0.191 | 0.0485 | 1.000 |
| `inter_af3_mean:test_ndcg@10:L3` | 108 | +0.003 | 0.9731 | 1.000 | +0.218 | 0.0245 | 1.000 |
| `inter_af3_median:test_recall@5:L3` | 108 | +0.074 | 0.4129 | 1.000 | +0.282 | 0.0033 | 0.505 |
| `inter_af3_median:test_recall@10:L3` | 108 | +0.132 | 0.1617 | 1.000 | +0.345 | 0.00035 | 0.063 |
| `inter_af3_median:test_ndcg@5:L3` | 108 | +0.063 | 0.4827 | 1.000 | +0.250 | 0.00945 | 1.000 |
| `inter_af3_median:test_ndcg@10:L3` | 108 | +0.096 | 0.2972 | 1.000 | +0.283 | 0.0031 | 0.480 |
| `inter_af3_p10:test_recall@5:L3` | 108 | -0.222 | 0.0293 | 1.000 | +0.225 | 0.01875 | 1.000 |
| `inter_af3_p10:test_recall@10:L3` | 108 | -0.176 | 0.07125 | 1.000 | +0.267 | 0.00525 | 0.782 |
| `inter_af3_p10:test_ndcg@5:L3` | 108 | -0.242 | 0.0206 | 1.000 | +0.199 | 0.03615 | 1.000 |
| `inter_af3_p10:test_ndcg@10:L3` | 108 | -0.214 | 0.03435 | 1.000 | +0.217 | 0.0238 | 1.000 |
| `inter_af3_p90:test_recall@5:L3` | 108 | -0.007 | 0.9368 | 1.000 | +0.270 | 0.00535 | 0.792 |
| `inter_af3_p90:test_recall@10:L3` | 108 | +0.049 | 0.5621 | 1.000 | +0.330 | 0.00045 | 0.078 |
| `inter_af3_p90:test_ndcg@5:L3` | 108 | -0.019 | 0.8202 | 1.000 | +0.239 | 0.014 | 1.000 |
| `inter_af3_p90:test_ndcg@10:L3` | 108 | +0.013 | 0.875 | 1.000 | +0.269 | 0.0048 | 0.720 |
| `inter_negative_edge_fraction:test_recall@5:L3` | 108 | +0.533 | 0.00045 | 0.065 | +0.077 | 0.4321 | 1.000 |
| `inter_negative_edge_fraction:test_recall@10:L3` | 108 | +0.494 | 0.0005 | 0.072 | +0.074 | 0.445 | 1.000 |
| `inter_negative_edge_fraction:test_ndcg@5:L3` | 108 | +0.549 | 0.0004 | 0.059 | +0.062 | 0.5253 | 1.000 |
| `inter_negative_edge_fraction:test_ndcg@10:L3` | 108 | +0.527 | 0.0004 | 0.059 | +0.064 | 0.5061 | 1.000 |
| `inter_forman_mean:test_recall@5:L3` | 108 | +0.533 | 5e-05 | 0.013 | +0.364 | 0.0002 | 0.037 |
| `inter_forman_mean:test_recall@10:L3` | 108 | +0.571 | 5e-05 | 0.013 | +0.420 | 5e-05 | 0.013 |
| `inter_forman_mean:test_ndcg@5:L3` | 108 | +0.529 | 5e-05 | 0.013 | +0.329 | 0.001 | 0.167 |
| `inter_forman_mean:test_ndcg@10:L3` | 108 | +0.553 | 5e-05 | 0.013 | +0.357 | 0.00045 | 0.078 |
| `inter_density:test_recall@5:L3` | 108 | -0.591 | 5e-05 | 0.013 | -0.430 | 5e-05 | 0.013 |
| `inter_density:test_recall@10:L3` | 108 | -0.556 | 5e-05 | 0.013 | -0.479 | 5e-05 | 0.013 |
| `inter_density:test_ndcg@5:L3` | 108 | -0.608 | 5e-05 | 0.013 | -0.399 | 5e-05 | 0.013 |
| `inter_density:test_ndcg@10:L3` | 108 | -0.588 | 5e-05 | 0.013 | -0.425 | 5e-05 | 0.013 |
| `inter_transitivity:test_recall@5:L3` | 108 | -0.649 | 5e-05 | 0.013 | -0.386 | 5e-05 | 0.013 |
| `inter_transitivity:test_recall@10:L3` | 108 | -0.631 | 5e-05 | 0.013 | -0.438 | 5e-05 | 0.013 |
| `inter_transitivity:test_ndcg@5:L3` | 108 | -0.665 | 5e-05 | 0.013 | -0.355 | 0.00015 | 0.028 |
| `inter_transitivity:test_ndcg@10:L3` | 108 | -0.655 | 5e-05 | 0.013 | -0.383 | 0.00015 | 0.028 |
| `gap_gap_af3:test_recall@5:L3` | 108 | +0.612 | 5e-05 | 0.013 | +0.492 | 5e-05 | 0.013 |
| `gap_gap_af3:test_recall@10:L3` | 108 | +0.600 | 5e-05 | 0.013 | +0.524 | 5e-05 | 0.013 |
| `gap_gap_af3:test_ndcg@5:L3` | 108 | +0.608 | 5e-05 | 0.013 | +0.452 | 5e-05 | 0.013 |
| `gap_gap_af3:test_ndcg@10:L3` | 108 | +0.605 | 5e-05 | 0.013 | +0.470 | 5e-05 | 0.013 |
| `gap_gap_forman:test_recall@5:L3` | 108 | +0.582 | 0.00015 | 0.024 | +0.485 | 5e-05 | 0.013 |
| `gap_gap_forman:test_recall@10:L3` | 108 | +0.568 | 5e-05 | 0.013 | +0.516 | 5e-05 | 0.013 |
| `gap_gap_forman:test_ndcg@5:L3` | 108 | +0.578 | 0.00015 | 0.024 | +0.444 | 5e-05 | 0.013 |
| `gap_gap_forman:test_ndcg@10:L3` | 108 | +0.574 | 5e-05 | 0.013 | +0.462 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_recall@5:L3` | 108 | +0.612 | 5e-05 | 0.013 | +0.490 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_recall@10:L3` | 108 | +0.599 | 5e-05 | 0.013 | +0.522 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_ndcg@5:L3` | 108 | +0.608 | 5e-05 | 0.013 | +0.450 | 5e-05 | 0.013 |
| `gap_gap_within_mean:test_ndcg@10:L3` | 108 | +0.605 | 0.0001 | 0.017 | +0.469 | 5e-05 | 0.013 |
| `gap_gap_cross_mean:test_recall@5:L3` | 108 | -0.063 | 0.4237 | 1.000 | +0.030 | 0.764 | 1.000 |
| `gap_gap_cross_mean:test_recall@10:L3` | 108 | -0.059 | 0.4627 | 1.000 | +0.075 | 0.4395 | 1.000 |
| `gap_gap_cross_mean:test_ndcg@5:L3` | 108 | -0.066 | 0.3976 | 1.000 | +0.010 | 0.9143 | 1.000 |
| `gap_gap_cross_mean:test_ndcg@10:L3` | 108 | -0.063 | 0.4208 | 1.000 | +0.033 | 0.7366 | 1.000 |
| `gap_edge_retention:test_recall@5:L3` | 108 | -0.204 | 0.0377 | 1.000 | -0.209 | 0.0301 | 1.000 |
| `gap_edge_retention:test_recall@10:L3` | 108 | -0.183 | 0.05115 | 1.000 | -0.252 | 0.00825 | 1.000 |
| `gap_edge_retention:test_ndcg@5:L3` | 108 | -0.214 | 0.03425 | 1.000 | -0.190 | 0.0494 | 1.000 |
| `gap_edge_retention:test_ndcg@10:L3` | 108 | -0.202 | 0.0397 | 1.000 | -0.211 | 0.0243 | 1.000 |
| `gap_z_global:test_recall@5:L3` | 81 | +0.473 | 0.00735 | 0.693 | -0.387 | 0.0002 | 0.037 |
| `gap_z_global:test_recall@10:L3` | 81 | +0.435 | 0.00995 | 0.876 | -0.420 | 0.0001 | 0.019 |
| `gap_z_global:test_ndcg@5:L3` | 81 | +0.483 | 0.00865 | 0.787 | -0.389 | 0.00035 | 0.063 |
| `gap_z_global:test_ndcg@10:L3` | 81 | +0.462 | 0.0095 | 0.845 | -0.408 | 0.00015 | 0.028 |
| `gap_z_degree_stratified:test_recall@5:L3` | 81 | -0.487 | 0.0003 | 0.045 | -0.589 | 5e-05 | 0.013 |
| `gap_z_degree_stratified:test_recall@10:L3` | 81 | -0.501 | 0.0001 | 0.017 | -0.597 | 5e-05 | 0.013 |
| `gap_z_degree_stratified:test_ndcg@5:L3` | 81 | -0.504 | 0.00015 | 0.024 | -0.594 | 5e-05 | 0.013 |
| `gap_z_degree_stratified:test_ndcg@10:L3` | 81 | -0.510 | 0.0001 | 0.017 | -0.593 | 5e-05 | 0.013 |

## Secondary family (corrected semantics and null variants)

### Secondary, L1

| metric:outcome | n | Pearson r | p_r | Holm_r | Spearman rho | p_rho | Holm_rho |
|---|---|---|---|---|---|---|---|
| `group_mean_negative_fraction:test_recall@5:L1` | 108 | +0.562 | 5e-05 | 0.005 | +0.676 | 5e-05 | 0.005 |
| `group_mean_negative_fraction:test_recall@10:L1` | 108 | +0.573 | 5e-05 | 0.005 | +0.670 | 5e-05 | 0.005 |
| `group_mean_negative_fraction:test_ndcg@5:L1` | 108 | +0.563 | 5e-05 | 0.005 | +0.690 | 5e-05 | 0.005 |
| `group_mean_negative_fraction:test_ndcg@10:L1` | 108 | +0.570 | 5e-05 | 0.005 | +0.693 | 5e-05 | 0.005 |
| `mean_group_negative_edge_fraction:test_recall@5:L1` | 108 | +0.395 | 0.0001 | 0.006 | +0.594 | 5e-05 | 0.005 |
| `mean_group_negative_edge_fraction:test_recall@10:L1` | 108 | +0.394 | 5e-05 | 0.005 | +0.601 | 5e-05 | 0.005 |
| `mean_group_negative_edge_fraction:test_ndcg@5:L1` | 108 | +0.392 | 0.0002 | 0.011 | +0.594 | 5e-05 | 0.005 |
| `mean_group_negative_edge_fraction:test_ndcg@10:L1` | 108 | +0.393 | 5e-05 | 0.005 | +0.607 | 5e-05 | 0.005 |
| `forman_group_mean_negative_fraction:test_recall@5:L1` | 108 | +0.580 | 5e-05 | 0.005 | +0.679 | 5e-05 | 0.005 |
| `forman_group_mean_negative_fraction:test_recall@10:L1` | 108 | +0.591 | 5e-05 | 0.005 | +0.671 | 5e-05 | 0.005 |
| `forman_group_mean_negative_fraction:test_ndcg@5:L1` | 108 | +0.580 | 5e-05 | 0.005 | +0.691 | 5e-05 | 0.005 |
| `forman_group_mean_negative_fraction:test_ndcg@10:L1` | 108 | +0.588 | 5e-05 | 0.005 | +0.693 | 5e-05 | 0.005 |
| `forman_edge_negative_fraction:test_recall@5:L1` | 108 | +0.169 | 0.0805 | 0.644 | +0.231 | 0.017 | 0.816 |
| `forman_edge_negative_fraction:test_recall@10:L1` | 108 | +0.164 | 0.0921 | 0.644 | +0.221 | 0.0225 | 1.000 |
| `forman_edge_negative_fraction:test_ndcg@5:L1` | 108 | +0.162 | 0.0907 | 0.644 | +0.239 | 0.01185 | 0.604 |
| `forman_edge_negative_fraction:test_ndcg@10:L1` | 108 | +0.161 | 0.09845 | 0.644 | +0.244 | 0.0125 | 0.625 |
| `forman_mean_group_negative_edge_fraction:test_recall@5:L1` | 108 | +0.442 | 5e-05 | 0.005 | +0.632 | 5e-05 | 0.005 |
| `forman_mean_group_negative_edge_fraction:test_recall@10:L1` | 108 | +0.442 | 5e-05 | 0.005 | +0.621 | 5e-05 | 0.005 |
| `forman_mean_group_negative_edge_fraction:test_ndcg@5:L1` | 108 | +0.440 | 5e-05 | 0.005 | +0.643 | 5e-05 | 0.005 |
| `forman_mean_group_negative_edge_fraction:test_ndcg@10:L1` | 108 | +0.442 | 5e-05 | 0.005 | +0.646 | 5e-05 | 0.005 |
| `centered_global:test_recall@5:L1` | 107 | +0.808 | 5e-05 | 0.005 | +0.768 | 5e-05 | 0.005 |
| `centered_global:test_recall@10:L1` | 107 | +0.792 | 5e-05 | 0.005 | +0.740 | 5e-05 | 0.005 |
| `centered_global:test_ndcg@5:L1` | 107 | +0.818 | 5e-05 | 0.005 | +0.793 | 5e-05 | 0.005 |
| `centered_global:test_ndcg@10:L1` | 107 | +0.811 | 5e-05 | 0.005 | +0.781 | 5e-05 | 0.005 |
| `centered_degree_stratified:test_recall@5:L1` | 107 | +0.587 | 5e-05 | 0.005 | +0.362 | 0.0001 | 0.006 |
| `centered_degree_stratified:test_recall@10:L1` | 107 | +0.617 | 5e-05 | 0.005 | +0.421 | 5e-05 | 0.005 |
| `centered_degree_stratified:test_ndcg@5:L1` | 107 | +0.580 | 5e-05 | 0.005 | +0.343 | 0.0005 | 0.026 |
| `centered_degree_stratified:test_ndcg@10:L1` | 107 | +0.601 | 5e-05 | 0.005 | +0.377 | 5e-05 | 0.005 |
| `triangle_increment:test_recall@5:L1` | 107 | -0.728 | 5e-05 | 0.005 | -0.625 | 5e-05 | 0.005 |
| `triangle_increment:test_recall@10:L1` | 107 | -0.700 | 5e-05 | 0.005 | -0.572 | 5e-05 | 0.005 |
| `triangle_increment:test_ndcg@5:L1` | 107 | -0.738 | 5e-05 | 0.005 | -0.662 | 5e-05 | 0.005 |
| `triangle_increment:test_ndcg@10:L1` | 107 | -0.724 | 5e-05 | 0.005 | -0.635 | 5e-05 | 0.005 |

### Secondary, L2

| metric:outcome | n | Pearson r | p_r | Holm_r | Spearman rho | p_rho | Holm_rho |
|---|---|---|---|---|---|---|---|
| `group_mean_negative_fraction:test_recall@5:L2` | 108 | -0.443 | 0.0003 | 0.014 | -0.166 | 0.08605 | 1.000 |
| `group_mean_negative_fraction:test_recall@10:L2` | 108 | -0.462 | 0.00025 | 0.013 | -0.205 | 0.03245 | 1.000 |
| `group_mean_negative_fraction:test_ndcg@5:L2` | 108 | -0.448 | 0.00035 | 0.016 | -0.152 | 0.1133 | 1.000 |
| `group_mean_negative_fraction:test_ndcg@10:L2` | 108 | -0.459 | 0.0002 | 0.011 | -0.168 | 0.08265 | 1.000 |
| `mean_group_negative_edge_fraction:test_recall@5:L2` | 108 | -0.458 | 0.00035 | 0.016 | -0.159 | 0.1034 | 1.000 |
| `mean_group_negative_edge_fraction:test_recall@10:L2` | 108 | -0.476 | 0.00015 | 0.009 | -0.197 | 0.0429 | 1.000 |
| `mean_group_negative_edge_fraction:test_ndcg@5:L2` | 108 | -0.463 | 5e-05 | 0.005 | -0.143 | 0.1417 | 1.000 |
| `mean_group_negative_edge_fraction:test_ndcg@10:L2` | 108 | -0.474 | 0.0001 | 0.006 | -0.160 | 0.09475 | 1.000 |
| `forman_group_mean_negative_fraction:test_recall@5:L2` | 108 | -0.389 | 0.00105 | 0.037 | -0.099 | 0.3041 | 1.000 |
| `forman_group_mean_negative_fraction:test_recall@10:L2` | 108 | -0.413 | 0.0002 | 0.011 | -0.138 | 0.1499 | 1.000 |
| `forman_group_mean_negative_fraction:test_ndcg@5:L2` | 108 | -0.392 | 0.00105 | 0.037 | -0.081 | 0.4035 | 1.000 |
| `forman_group_mean_negative_fraction:test_ndcg@10:L2` | 108 | -0.407 | 0.00095 | 0.034 | -0.100 | 0.3063 | 1.000 |
| `forman_edge_negative_fraction:test_recall@5:L2` | 108 | -0.319 | 0.0013 | 0.043 | -0.077 | 0.4244 | 1.000 |
| `forman_edge_negative_fraction:test_recall@10:L2` | 108 | -0.344 | 0.00015 | 0.009 | -0.114 | 0.2426 | 1.000 |
| `forman_edge_negative_fraction:test_ndcg@5:L2` | 108 | -0.323 | 0.00085 | 0.031 | -0.060 | 0.5333 | 1.000 |
| `forman_edge_negative_fraction:test_ndcg@10:L2` | 108 | -0.338 | 0.0006 | 0.023 | -0.076 | 0.4264 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_recall@5:L2` | 108 | -0.412 | 0.00045 | 0.019 | -0.090 | 0.3573 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_recall@10:L2` | 108 | -0.433 | 0.0004 | 0.018 | -0.127 | 0.1877 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_ndcg@5:L2` | 108 | -0.415 | 0.0005 | 0.020 | -0.071 | 0.4668 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_ndcg@10:L2` | 108 | -0.428 | 0.00025 | 0.013 | -0.088 | 0.3647 | 1.000 |
| `centered_global:test_recall@5:L2` | 82 | +0.793 | 5e-05 | 0.005 | +0.734 | 5e-05 | 0.005 |
| `centered_global:test_recall@10:L2` | 82 | +0.787 | 5e-05 | 0.005 | +0.752 | 5e-05 | 0.005 |
| `centered_global:test_ndcg@5:L2` | 82 | +0.799 | 5e-05 | 0.005 | +0.740 | 5e-05 | 0.005 |
| `centered_global:test_ndcg@10:L2` | 82 | +0.798 | 5e-05 | 0.005 | +0.744 | 5e-05 | 0.005 |
| `centered_degree_stratified:test_recall@5:L2` | 83 | +0.290 | 0.0079 | 0.141 | +0.049 | 0.6565 | 1.000 |
| `centered_degree_stratified:test_recall@10:L2` | 83 | +0.307 | 0.00495 | 0.141 | +0.081 | 0.4579 | 1.000 |
| `centered_degree_stratified:test_ndcg@5:L2` | 83 | +0.293 | 0.0064 | 0.141 | +0.049 | 0.656 | 1.000 |
| `centered_degree_stratified:test_ndcg@10:L2` | 83 | +0.303 | 0.00525 | 0.141 | +0.068 | 0.5455 | 1.000 |
| `triangle_increment:test_recall@5:L2` | 108 | -0.639 | 5e-05 | 0.005 | -0.707 | 5e-05 | 0.005 |
| `triangle_increment:test_recall@10:L2` | 108 | -0.617 | 5e-05 | 0.005 | -0.680 | 5e-05 | 0.005 |
| `triangle_increment:test_ndcg@5:L2` | 108 | -0.644 | 5e-05 | 0.005 | -0.720 | 5e-05 | 0.005 |
| `triangle_increment:test_ndcg@10:L2` | 108 | -0.634 | 5e-05 | 0.005 | -0.702 | 5e-05 | 0.005 |

### Secondary, L3

| metric:outcome | n | Pearson r | p_r | Holm_r | Spearman rho | p_rho | Holm_rho |
|---|---|---|---|---|---|---|---|
| `group_mean_negative_fraction:test_recall@5:L3` | 108 | -0.301 | 0.00715 | 0.141 | -0.222 | 0.02305 | 1.000 |
| `group_mean_negative_fraction:test_recall@10:L3` | 108 | -0.301 | 0.0057 | 0.141 | -0.241 | 0.0126 | 0.625 |
| `group_mean_negative_fraction:test_ndcg@5:L3` | 108 | -0.313 | 0.0063 | 0.141 | -0.223 | 0.0219 | 1.000 |
| `group_mean_negative_fraction:test_ndcg@10:L3` | 108 | -0.313 | 0.0051 | 0.141 | -0.229 | 0.01855 | 0.872 |
| `mean_group_negative_edge_fraction:test_recall@5:L3` | 108 | -0.313 | 0.00525 | 0.141 | -0.215 | 0.0249 | 1.000 |
| `mean_group_negative_edge_fraction:test_recall@10:L3` | 108 | -0.311 | 0.00485 | 0.141 | -0.243 | 0.01125 | 0.585 |
| `mean_group_negative_edge_fraction:test_ndcg@5:L3` | 108 | -0.326 | 0.0056 | 0.141 | -0.199 | 0.0396 | 1.000 |
| `mean_group_negative_edge_fraction:test_ndcg@10:L3` | 108 | -0.324 | 0.00395 | 0.122 | -0.214 | 0.02675 | 1.000 |
| `forman_group_mean_negative_fraction:test_recall@5:L3` | 108 | -0.273 | 0.00875 | 0.141 | -0.114 | 0.24 | 1.000 |
| `forman_group_mean_negative_fraction:test_recall@10:L3` | 108 | -0.279 | 0.00845 | 0.141 | -0.146 | 0.1296 | 1.000 |
| `forman_group_mean_negative_fraction:test_ndcg@5:L3` | 108 | -0.283 | 0.0098 | 0.141 | -0.094 | 0.3381 | 1.000 |
| `forman_group_mean_negative_fraction:test_ndcg@10:L3` | 108 | -0.286 | 0.0088 | 0.141 | -0.108 | 0.2603 | 1.000 |
| `forman_edge_negative_fraction:test_recall@5:L3` | 108 | -0.389 | 0.0006 | 0.023 | -0.140 | 0.1459 | 1.000 |
| `forman_edge_negative_fraction:test_recall@10:L3` | 108 | -0.395 | 0.00055 | 0.022 | -0.170 | 0.0803 | 1.000 |
| `forman_edge_negative_fraction:test_ndcg@5:L3` | 108 | -0.400 | 0.0003 | 0.014 | -0.119 | 0.2207 | 1.000 |
| `forman_edge_negative_fraction:test_ndcg@10:L3` | 108 | -0.404 | 0.0004 | 0.018 | -0.135 | 0.1637 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_recall@5:L3` | 108 | -0.292 | 0.00755 | 0.141 | -0.122 | 0.2067 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_recall@10:L3` | 108 | -0.294 | 0.00555 | 0.141 | -0.152 | 0.1217 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_ndcg@5:L3` | 108 | -0.302 | 0.00645 | 0.141 | -0.100 | 0.3096 | 1.000 |
| `forman_mean_group_negative_edge_fraction:test_ndcg@10:L3` | 108 | -0.303 | 0.0046 | 0.138 | -0.116 | 0.2301 | 1.000 |
| `centered_global:test_recall@5:L3` | 81 | +0.586 | 0.00025 | 0.013 | +0.400 | 0.00025 | 0.014 |
| `centered_global:test_recall@10:L3` | 81 | +0.573 | 0.0001 | 0.006 | +0.395 | 0.0004 | 0.022 |
| `centered_global:test_ndcg@5:L3` | 81 | +0.583 | 0.0002 | 0.011 | +0.396 | 0.0004 | 0.022 |
| `centered_global:test_ndcg@10:L3` | 81 | +0.579 | 0.0002 | 0.011 | +0.394 | 0.00045 | 0.024 |
| `centered_degree_stratified:test_recall@5:L3` | 81 | +0.004 | 0.9724 | 1.000 | -0.227 | 0.0416 | 1.000 |
| `centered_degree_stratified:test_recall@10:L3` | 81 | +0.007 | 0.9514 | 1.000 | -0.215 | 0.0558 | 1.000 |
| `centered_degree_stratified:test_ndcg@5:L3` | 81 | +0.010 | 0.9292 | 1.000 | -0.223 | 0.0447 | 1.000 |
| `centered_degree_stratified:test_ndcg@10:L3` | 81 | +0.012 | 0.9165 | 1.000 | -0.213 | 0.05445 | 1.000 |
| `triangle_increment:test_recall@5:L3` | 108 | -0.471 | 0.00535 | 0.141 | -0.556 | 5e-05 | 0.005 |
| `triangle_increment:test_recall@10:L3` | 108 | -0.453 | 0.0023 | 0.074 | -0.520 | 5e-05 | 0.005 |
| `triangle_increment:test_ndcg@5:L3` | 108 | -0.468 | 0.00855 | 0.141 | -0.579 | 5e-05 | 0.005 |
| `triangle_increment:test_ndcg@10:L3` | 108 | -0.461 | 0.00625 | 0.141 | -0.559 | 5e-05 | 0.005 |

## Negative-curvature semantics

| measure | L1 | L2 | L3 |
|---|---|---|---|
| `edge_negative_fraction` | +0.113 / +0.197 | -0.423 / -0.180 | -0.361 / -0.249 |
| `group_mean_negative_fraction` | +0.573 / +0.670 | -0.462 / -0.205 | -0.301 / -0.241 |
| `mean_group_negative_edge_fraction` | +0.394 / +0.601 | -0.476 / -0.197 | -0.311 / -0.243 |
| `forman_edge_negative_fraction` | +0.164 / +0.221 | -0.344 / -0.114 | -0.395 / -0.170 |
| `forman_group_mean_negative_fraction` | +0.591 / +0.671 | -0.413 / -0.138 | -0.279 / -0.146 |

Cells are Pearson / Spearman for R@10. The group-mean and edge-negative fractions are different measurements with different denominators; neither is called the other.

## Controls and models

| level | outcome | r2 controls | r2 + gap | increment | gap coef | LSO folds | RMSE controls | RMSE + gap | max VIF |
|---|---|---|---|---|---|---|---|---|---|
| 1 | test_recall@5 | 0.164 | 0.694 | +0.530 | +5.50e-05 | 1 | 0.004331432544121057 | 0.0026206097816897003 | 2.97 |
| 1 | test_recall@10 | 0.182 | 0.668 | +0.486 | +7.69e-05 | 1 | 0.006392656828261466 | 0.0041075314838969635 | 2.97 |
| 1 | test_ndcg@5 | 0.172 | 0.709 | +0.537 | +3.79e-05 | 1 | 0.0028820704579933273 | 0.001681415927754808 | 2.97 |
| 1 | test_ndcg@10 | 0.182 | 0.698 | +0.516 | +4.50e-05 | 1 | 0.0035349037553831656 | 0.0021429738350605162 | 2.97 |
| 2 | test_recall@5 | 0.567 | 0.647 | +0.080 | +2.61e-05 | 1 | 0.004976468571232643 | 0.004709768516676331 | 2.24 |
| 2 | test_recall@10 | 0.570 | 0.642 | +0.073 | +3.65e-05 | 1 | 0.007506820601131496 | 0.007252456772597138 | 2.24 |
| 2 | test_ndcg@5 | 0.571 | 0.652 | +0.081 | +1.79e-05 | 1 | 0.003343830291689568 | 0.0031384941355182753 | 2.24 |
| 2 | test_ndcg@10 | 0.575 | 0.652 | +0.077 | +2.13e-05 | 1 | 0.004139223849271728 | 0.003939569414552065 | 2.24 |
| 3 | test_recall@5 | 0.579 | 0.623 | +0.044 | +7.43e-06 | 1 | 0.004135963775676809 | 0.0046030265987795395 | 2.93 |
| 3 | test_recall@10 | 0.584 | 0.623 | +0.039 | +1.03e-05 | 1 | 0.0068134112222721635 | 0.007476151091545552 | 2.93 |
| 3 | test_ndcg@5 | 0.589 | 0.630 | +0.040 | +4.87e-06 | 1 | 0.0027849626768093553 | 0.0030901293227003027 | 2.93 |
| 3 | test_ndcg@10 | 0.593 | 0.632 | +0.039 | +5.82e-06 | 1 | 0.003612480048342245 | 0.003983825135802884 | 2.93 |

Partial correlations (both coefficient types) controlling the gap: 

| level | outcome | controls | type | n | partial | p |
|---|---|---|---|---|---|---|
| 1 | test_recall@10 | gap_gap_forman, gap_edge_retention, intra_transitivity | Pearson | 107 | -0.088 | 0.3463 |
| 1 | test_recall@10 | gap_gap_forman, gap_edge_retention, intra_transitivity | Spearman | 107 | +0.232 | 0.0164 |
| 1 | test_recall@10 | gap_gap_forman | Pearson | 107 | +0.320 | 0.0024 |
| 1 | test_recall@10 | gap_gap_forman | Spearman | 107 | +0.350 | 0.0002 |
| 1 | test_recall@10 | gap_edge_retention | Pearson | 107 | +0.788 | 0.0002 |
| 1 | test_recall@10 | gap_edge_retention | Spearman | 107 | +0.752 | 0.0002 |
| 1 | test_recall@10 | intra_transitivity | Pearson | 107 | +0.429 | 0.0003999 |
| 1 | test_recall@10 | intra_transitivity | Spearman | 107 | +0.052 | 0.5939 |
| 1 | test_recall@10 | occupied | Pearson | 107 | +0.775 | 0.0002 |
| 1 | test_recall@10 | occupied | Spearman | 107 | +0.758 | 0.0002 |
| 1 | test_recall@10 | singleton_share | Pearson | 107 | +0.808 | 0.0002 |
| 1 | test_recall@10 | singleton_share | Spearman | 107 | +0.757 | 0.0002 |
| 1 | test_ndcg@10 | gap_gap_forman, gap_edge_retention, intra_transitivity | Pearson | 107 | -0.113 | 0.2342 |
| 1 | test_ndcg@10 | gap_gap_forman, gap_edge_retention, intra_transitivity | Spearman | 107 | +0.247 | 0.0106 |
| 1 | test_ndcg@10 | gap_gap_forman | Pearson | 107 | +0.300 | 0.0024 |
| 1 | test_ndcg@10 | gap_gap_forman | Spearman | 107 | +0.368 | 0.0002 |
| 1 | test_ndcg@10 | gap_edge_retention | Pearson | 107 | +0.808 | 0.0002 |
| 1 | test_ndcg@10 | gap_edge_retention | Spearman | 107 | +0.797 | 0.0002 |
| 1 | test_ndcg@10 | intra_transitivity | Pearson | 107 | +0.476 | 0.0005999 |
| 1 | test_ndcg@10 | intra_transitivity | Spearman | 107 | +0.090 | 0.3635 |
| 1 | test_ndcg@10 | occupied | Pearson | 107 | +0.797 | 0.0002 |
| 1 | test_ndcg@10 | occupied | Spearman | 107 | +0.800 | 0.0002 |
| 1 | test_ndcg@10 | singleton_share | Pearson | 107 | +0.826 | 0.0002 |
| 1 | test_ndcg@10 | singleton_share | Spearman | 107 | +0.800 | 0.0002 |
| 2 | test_recall@10 | gap_gap_forman, gap_edge_retention, inter_density | Pearson | 108 | +0.341 | 0.0012 |
| 2 | test_recall@10 | gap_gap_forman, gap_edge_retention, inter_density | Spearman | 108 | +0.269 | 0.003799 |
| 2 | test_recall@10 | gap_gap_forman | Pearson | 108 | +0.256 | 0.011 |
| 2 | test_recall@10 | gap_gap_forman | Spearman | 108 | +0.238 | 0.0122 |
| 2 | test_recall@10 | gap_edge_retention | Pearson | 108 | +0.751 | 0.0002 |
| 2 | test_recall@10 | gap_edge_retention | Spearman | 108 | +0.760 | 0.0002 |
| 2 | test_recall@10 | inter_density | Pearson | 108 | +0.526 | 0.0002 |
| 2 | test_recall@10 | inter_density | Spearman | 108 | +0.600 | 0.0002 |
| 2 | test_recall@10 | occupied | Pearson | 108 | +0.713 | 0.0002 |
| 2 | test_recall@10 | occupied | Spearman | 108 | +0.600 | 0.0002 |
| 2 | test_recall@10 | singleton_share | Pearson | 108 | +0.714 | 0.0002 |
| 2 | test_recall@10 | singleton_share | Spearman | 108 | +0.621 | 0.0002 |
| 2 | test_ndcg@10 | gap_gap_forman, gap_edge_retention, inter_density | Pearson | 108 | +0.282 | 0.003399 |
| 2 | test_ndcg@10 | gap_gap_forman, gap_edge_retention, inter_density | Spearman | 108 | +0.260 | 0.007998 |
| 2 | test_ndcg@10 | gap_gap_forman | Pearson | 108 | +0.212 | 0.03259 |
| 2 | test_ndcg@10 | gap_gap_forman | Spearman | 108 | +0.224 | 0.021 |
| 2 | test_ndcg@10 | gap_edge_retention | Pearson | 108 | +0.750 | 0.0002 |
| 2 | test_ndcg@10 | gap_edge_retention | Spearman | 108 | +0.722 | 0.0002 |
| 2 | test_ndcg@10 | inter_density | Pearson | 108 | +0.516 | 0.0002 |
| 2 | test_ndcg@10 | inter_density | Spearman | 108 | +0.567 | 0.0002 |
| 2 | test_ndcg@10 | occupied | Pearson | 108 | +0.728 | 0.0002 |
| 2 | test_ndcg@10 | occupied | Spearman | 108 | +0.566 | 0.0002 |
| 2 | test_ndcg@10 | singleton_share | Pearson | 108 | +0.729 | 0.0002 |
| 2 | test_ndcg@10 | singleton_share | Spearman | 108 | +0.588 | 0.0002 |
| 3 | test_recall@10 | gap_gap_forman, gap_edge_retention, inter_density | Pearson | 108 | +0.379 | 0.0002 |
| 3 | test_recall@10 | gap_gap_forman, gap_edge_retention, inter_density | Spearman | 108 | -0.034 | 0.7307 |
| 3 | test_recall@10 | gap_gap_forman | Pearson | 108 | +0.484 | 0.0002 |
| 3 | test_recall@10 | gap_gap_forman | Spearman | 108 | +0.118 | 0.2186 |
| 3 | test_recall@10 | gap_edge_retention | Pearson | 108 | +0.586 | 0.0002 |
| 3 | test_recall@10 | gap_edge_retention | Spearman | 108 | +0.539 | 0.0002 |
| 3 | test_recall@10 | inter_density | Pearson | 108 | +0.466 | 0.0002 |
| 3 | test_recall@10 | inter_density | Spearman | 108 | +0.249 | 0.0106 |
| 3 | test_recall@10 | occupied | Pearson | 108 | +0.445 | 0.0002 |
| 3 | test_recall@10 | occupied | Spearman | 108 | +0.252 | 0.008598 |
| 3 | test_recall@10 | singleton_share | Pearson | 108 | +0.376 | 0.0002 |
| 3 | test_recall@10 | singleton_share | Spearman | 108 | +0.239 | 0.0138 |
| 3 | test_ndcg@10 | gap_gap_forman, gap_edge_retention, inter_density | Pearson | 108 | +0.344 | 0.0002 |
| 3 | test_ndcg@10 | gap_gap_forman, gap_edge_retention, inter_density | Spearman | 108 | -0.006 | 0.9498 |
| 3 | test_ndcg@10 | gap_gap_forman | Pearson | 108 | +0.476 | 0.0002 |
| 3 | test_ndcg@10 | gap_gap_forman | Spearman | 108 | +0.124 | 0.207 |
| 3 | test_ndcg@10 | gap_edge_retention | Pearson | 108 | +0.591 | 0.0002 |
| 3 | test_ndcg@10 | gap_edge_retention | Spearman | 108 | +0.499 | 0.0002 |
| 3 | test_ndcg@10 | inter_density | Pearson | 108 | +0.466 | 0.0002 |
| 3 | test_ndcg@10 | inter_density | Spearman | 108 | +0.227 | 0.0184 |
| 3 | test_ndcg@10 | occupied | Pearson | 108 | +0.453 | 0.0003999 |
| 3 | test_ndcg@10 | occupied | Spearman | 108 | +0.231 | 0.0138 |
| 3 | test_ndcg@10 | singleton_share | Pearson | 108 | +0.383 | 0.0002 |
| 3 | test_ndcg@10 | singleton_share | Spearman | 108 | +0.217 | 0.024 |

## Null sensitivity

| variant | level | n | null mean | null sd | Pearson (R@10) | Spearman (R@10) |
|---|---|---|---|---|---|---|
| `gap_af3` | 1 | 107 |  |  | 0.8088873163816304 | 0.7596104863863378 |
| `gap_af3` | 2 | 108 |  |  | 0.7734128135982523 | 0.6845037585564376 |
| `gap_af3` | 3 | 108 |  |  | 0.599514644128376 | 0.5235104324555238 |
| `gap_af3_centered_global` | 1 | 107 | -0.82 | 10.94 | 0.7922820061441775 | 0.7403925987789193 |
| `gap_af3_centered_global` | 2 | 82 | 3.37 | 51.45 | 0.7868731043914379 | 0.7517237251966361 |
| `gap_af3_centered_global` | 3 | 81 | 7.08 | 143.57 | 0.5732186049281215 | 0.3946882569715376 |
| `gap_af3_z_global` | 1 | 107 | -0.82 | 10.94 | 0.7063924365414985 | 0.5421976219554846 |
| `gap_af3_z_global` | 2 | 82 | 3.37 | 51.45 | 0.5240693336436805 | 0.016684897068489652 |
| `gap_af3_z_global` | 3 | 81 | 7.08 | 143.57 | 0.4348662457135022 | -0.4198693519575625 |
| `gap_af3_z_degree_stratified` | 1 | 107 | -38.63 | 7.12 | 0.5517369268655371 | 0.3472208276186544 |
| `gap_af3_z_degree_stratified` | 2 | 83 | -24.71 | 34.15 | -0.2976100081911356 | -0.43880040936353387 |
| `gap_af3_z_degree_stratified` | 3 | 81 | -15.61 | 99.38 | -0.5009203039342967 | -0.5968484110230192 |

The raw gap, the gap minus its null mean and the z-score are all reported; the sign change under the degree-stratified null is explained by that null randomising degree while keeping the degree strata, so the null mean moves far from zero at L3. No null scheme is selected by its Recall correlation.

## Within the 76 curvature_RQ-VAE arms

See `within76.csv`; the gap at L1/L2/L3 is reported there with n and both coefficients. The 76 rows are one series, not 76 independent mechanisms.

## Influence

* `gap_af3:R@10:L1`: 5 of 107 arms exceed Cook's threshold 0.0374; without them r +0.732 / rho +0.759 (full: r +0.809 / rho +0.760).
* `gap_af3:R@10:L2`: 4 of 108 arms exceed Cook's threshold 0.0370; without them r +0.681 / rho +0.651 (full: r +0.773 / rho +0.685).
* `gap_af3:R@10:L3`: 6 of 108 arms exceed Cook's threshold 0.0370; without them r +0.514 / rho +0.488 (full: r +0.600 / rho +0.524).

## Verdict

* gap association after proper inference: **GAP_ASSOCIATION_ROBUST** - gap Holm_r/Holm_rho at L1/L2/L3: 0.0132/0.0132, 0.0132/0.0132, 0.0132/0.0132; within-76 Pearson +0.526, +0.608, +0.451.
* edge-negative vs group-mean-negative semantics: **MIXED/NULL_SENSITIVE** - edge-negative fraction: L1 r +0.113/rho +0.197 (Holm 1/1), L2 r -0.423/rho -0.180 (Holm 0.0172/1), L3 r -0.361/rho -0.249 (Holm 0.1035/1); group-mean-negative fraction: L1 r +0.573/rho +0.670 (Holm 0.0048/0.0048), L2 r -0.462/rho -0.205 (Holm 0.01275/1), L3 r -0.301/rho -0.241 (Holm 0.1406/0.625); the two denominators measure different objects, and no single negative-curvature claim survives both coefficient types.
* gap beyond Forman, retention and resolution controls: **MIXED/NULL_SENSITIVE** - three-control partial p (Pearson/Spearman) at L1/L2/L3: 0.346/0.0164, 0.0012/0.0038, 0.0002/0.731; R2 increment and LSO RMSE are in model_controls.csv.
* null-model sensitivity: **MIXED/NULL_SENSITIVE** - raw, null-centred and z variants are in null_sensitivity.csv; the degree-stratified z flips sign at L2/L3 because its null mean is strongly negative, while the null-centred gap stays positive.

Family counts: 528 potential coefficient tests; 528 defined; 0 undefined. The issue's 520/8 figure described #136's field set, where four metric/outcome entries were degenerate; the corrected edge-negative fraction is defined for every arm, so no entry is dropped. This is a semantic correction, not a selection.

Blocked: d3_prefix_alignment = NOT_AVAILABLE: no D3 implementation; r20_ndcg20 = NOT_EVALUATED: saved artifacts cover topk 5/10; collision_code_usage = NOT_INCLUDED: no per-SID collision or code-usage artifact exists; occupied prefix count and singleton share are the resolution controls.

## Not claimed

The gap is observational: it does not prove hyperbolic manifold necessity, does not justify a Stage2 curvature loss, and the 76 arms are one series rather than repeated independent mechanisms. No Stage2/Stage3 artifact was modified and no R@20/NDCG@20 was synthesised.
