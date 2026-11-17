# Per-edge Graphlet roles on the quantized KNN graph

q graph: pure-Euclidean TIGER q, K=10, symmetrised union. Triangle/K4/4-cycle/4-path
from PGD micro; K1,3 star from the exact verified counter; PGD local 3-star not used.

q graph edges **186041**; behaviour edges **315391**; retained **3569** (1.132%).

## Multi-role prevalence

| graph | role-count 0 | 1 | 2 | 3 | 4 | 5 | multi-role |
|---|---:|---:|---:|---:|---:|---:|---:|
| q KNN | 0 | 0 | 6911 | 14422 | 31025 | 133683 | 186041 |
| behaviour | 0 | 52 | 8262 | 122202 | 80457 | 104418 | 315339 |

## Top role combinations (raw presence)

| graph | combination | edges |
|---|---|---:|
| q KNN | triangle + K4 + 4-cycle + K1,3 star + 4-path | 133683 |
| q KNN | triangle + K4 + K1,3 star + 4-path | 17371 |
| q KNN | triangle + 4-cycle + K1,3 star + 4-path | 13186 |
| q KNN | triangle + K4 + K1,3 star | 7796 |
| q KNN | triangle + K4 | 6045 |
| q KNN | 4-cycle + K1,3 star + 4-path | 3392 |
| q KNN | triangle + K1,3 star + 4-path | 1507 |
| q KNN | triangle + K4 + 4-path | 1316 |
| q KNN | K1,3 star + 4-path | 866 |
| q KNN | triangle + K4 + 4-cycle + 4-path | 411 |
| behaviour | 4-cycle + K1,3 star + 4-path | 121379 |
| behaviour | triangle + K4 + 4-cycle + K1,3 star + 4-path | 104418 |
| behaviour | triangle + 4-cycle + K1,3 star + 4-path | 80435 |
| behaviour | K1,3 star + 4-path | 8249 |
| behaviour | triangle + K1,3 star + 4-path | 822 |
| behaviour | K1,3 star | 51 |
| behaviour | triangle + K4 + K1,3 star + 4-path | 22 |
| behaviour | triangle + K1,3 star | 13 |
| behaviour | triangle + K4 + K1,3 star | 1 |
| behaviour | 4-path | 1 |

## High-involvement roles (top quartile within dmax bins)

| graph | 0 | 1 | 2 | 3 | 4 | 5 | multi-role |
|---|---:|---:|---:|---:|---:|---:|---:|
| q KNN | 58761 | 43187 | 64929 | 17880 | 894 | 390 | 84093 |
| behaviour | 166206 | 56305 | 30215 | 25574 | 21037 | 16054 | 92880 |

## Retention by number of behaviour-graph roles

| behaviour roles | n | retained | rate |
|---|---:|---:|---:|
| 0 | 0 | 0 | nan |
| 1 | 52 | 1 | 0.0192 |
| 2 | 8262 | 136 | 0.0165 |
| 3 | 122202 | 1332 | 0.0109 |
| 4 | 80457 | 1124 | 0.0140 |
| 5 | 104418 | 976 | 0.0093 |

## Retention by high-involvement role count

| high roles | n | retained | rate |
|---|---:|---:|---:|
| 0 | 166206 | 1626 | 0.0098 |
| 1 | 56305 | 739 | 0.0131 |
| 2 | 30215 | 472 | 0.0156 |
| 3 | 25574 | 328 | 0.0128 |
| 4 | 21037 | 247 | 0.0117 |
| 5 | 16054 | 157 | 0.0098 |

## Retained edges: q vs behaviour counts

| role | behaviour mean | q mean | Spearman | q presence |
|---|---:|---:|---:|---:|
| triangle | 3.656 | 7.513 | -0.018 | 0.988 |
| K4 | 16.374 | 21.592 | -0.028 | 0.939 |
| 4-cycle | 126.999 | 6.347 | +0.019 | 0.711 |
| K1,3 star | 20628.043 | 83.520 | +0.066 | 0.873 |
| 4-path | 5368.021 | 70.364 | +0.075 | 0.826 |

`q_counts_file` contains all q-edge counts; `retained_file` contains the retained
behaviour-edge counts before/after quantization.
