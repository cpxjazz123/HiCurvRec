# Cluster-level 73-d ORCA: full graph vs internal subgraph

Fixed semantic clusters: KMeans(256) on L2-normalised sentence-T5 embeddings. Method A aggregates the full-graph node ORCA over each cluster. Method B runs ORCA on the union of internal semantic-cluster edges and aggregates that. Vectors are `mean_i log1p(ORCA_i)` normalised to sum 1; JSD and cosine are computed per cluster.

Behaviour graph: **24587** nodes, **315391** edges, **256** semantic clusters.
Behaviour internal edges: **16887 / 315391**; clusters with at least one internal edge: **253 / 256**.

## Method A: full-graph ORCA aggregated per semantic cluster

| arm | valid clusters | JSD weighted | JSD median | cosine weighted | cosine median |
|---|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 256 | 0.0444 | 0.0444 | 0.9303 | 0.9306 |
| mixed_F0 | 256 | 0.0459 | 0.0458 | 0.9279 | 0.9269 |

## Method B: internal-subgraph ORCA per semantic cluster

| arm | q internal edges | clusters with internal edges | valid clusters | JSD weighted | JSD median | cosine weighted | cosine median |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 119494 | 256 | 253 | 0.3398 | 0.3858 | 0.6377 | 0.6031 |
| mixed_F0 | 112522 | 256 | 253 | 0.3367 | 0.3655 | 0.6443 | 0.6313 |

## External structure: boundary ORCA and cross-cluster role matrix

Boundary vectors use the full-graph ORCA of unique external neighbours. `cross role JSD` compares the 9x9 role-pair matrix of cross-cluster edges before/after.

| arm | boundary JSD weighted | boundary JSD median | boundary cosine weighted | cross rate before | cross rate after | cross role JSD |
|---|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 0.0119 | 0.0115 | 0.9776 | 0.9725 | 0.5269 | 0.6807 |
| mixed_F0 | 0.0132 | 0.0125 | 0.9755 | 0.9725 | 0.5609 | 0.8164 |

## How much does dropping cross-cluster edges change the behaviour cluster vector?

Behaviour graph: full-graph aggregation (A) vs internal-only ORCA (B).

- valid clusters: **253 / 256**
- JSD weighted / median: **0.2987 / 0.3266**
- cosine weighted / median: **0.6783 / 0.6758**

This is the direct cost of using only internal subgraphs on the behaviour side.

## Top changed semantic clusters, internal ORCA

| arm | cluster | size | JSD | internal edges before | internal edges after | cross rate before | cross rate after |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 182 | 33 | 0.9556 | 2 | 186 | 0.9951 | 0.3961 |
| pure_euclidean_TIGER | 105 | 39 | 0.9518 | 1 | 149 | 0.9977 | 0.5718 |
| pure_euclidean_TIGER | 212 | 54 | 0.9503 | 2 | 205 | 0.9979 | 0.5988 |
| pure_euclidean_TIGER | 95 | 56 | 0.9465 | 2 | 137 | 0.9977 | 0.8073 |
| pure_euclidean_TIGER | 153 | 44 | 0.9435 | 1 | 135 | 0.9989 | 0.6683 |
| pure_euclidean_TIGER | 87 | 34 | 0.9346 | 1 | 142 | 0.9982 | 0.5463 |
| pure_euclidean_TIGER | 92 | 19 | 0.9283 | 1 | 109 | 0.9977 | 0.2431 |
| pure_euclidean_TIGER | 106 | 17 | 0.9262 | 1 | 82 | 0.9962 | 0.6450 |
| pure_euclidean_TIGER | 198 | 37 | 0.9142 | 2 | 61 | 0.9978 | 0.8310 |
| pure_euclidean_TIGER | 239 | 19 | 0.9128 | 3 | 34 | 0.9935 | 0.8496 |
| mixed_F0 | 198 | 37 | 0.9592 | 2 | 115 | 0.9978 | 0.7202 |
| mixed_F0 | 182 | 33 | 0.9580 | 2 | 170 | 0.9951 | 0.4516 |
| mixed_F0 | 212 | 54 | 0.9504 | 2 | 158 | 0.9979 | 0.7496 |
| mixed_F0 | 105 | 39 | 0.9503 | 1 | 88 | 0.9977 | 0.7864 |
| mixed_F0 | 87 | 34 | 0.9464 | 1 | 131 | 0.9982 | 0.6478 |
| mixed_F0 | 153 | 44 | 0.9379 | 1 | 112 | 0.9989 | 0.7934 |
| mixed_F0 | 88 | 53 | 0.9264 | 1 | 107 | 0.9989 | 0.8033 |
| mixed_F0 | 95 | 56 | 0.9131 | 2 | 94 | 0.9977 | 0.8481 |
| mixed_F0 | 239 | 19 | 0.9043 | 3 | 40 | 0.9935 | 0.8326 |
| mixed_F0 | 202 | 37 | 0.9024 | 3 | 45 | 0.9945 | 0.8846 |
