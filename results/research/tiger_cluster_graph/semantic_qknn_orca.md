# Semantic clusters → quantized kNN graph ORCA relations

Semantic clusters: KMeans(256) on L2-normalised sentence-T5 item embeddings. Quantized graphs: global k=10 nearest neighbours of q (Euclidean), symmetrised. ORCA is recomputed on each q-kNN graph with exact 5-node orbits.

Behaviour graph: **24587** nodes, **315391** edges, **256** semantic clusters, multi-role **30.992%**.

## Role preservation (node level)

| arm | q-kNN edges | components | top-role ARI | top-role NMI | same top role | role JSD mean | multi agreement | bridge nodes | bridge role JSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 186044 | 10 | -0.0003 | 0.0002 | 0.4287 | 0.3326 | 0.6548 | 24543 | 0.2180 |
| mixed_F0 | 184398 | 12 | -0.0003 | 0.0002 | 0.4316 | 0.3571 | 0.6826 | 24543 | 0.2621 |

## Semantic-cluster role profiles and role-pair relations

Same semantic clusters are used on the behaviour graph and on each q-kNN graph; only the graph changes. `profile JSD` compares the 9-group ORCA profile of each semantic cluster before/after. `within semantic` / `cross semantic` role-pair JSD compare role-pair distributions inside/outside semantic clusters.

| arm | profile JSD | profile JSD median | top-role match | boundary JSD | multi-role MAE | all role-pair JSD | within semantic JSD | cross semantic JSD | before same | after same |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 0.2331 | 0.2485 | 0.3164 | 0.2775 | 0.2307 | 0.7005 | 0.5749 | 0.7054 | 0.0535 | 0.6423 |
| mixed_F0 | 0.2727 | 0.2840 | 0.3242 | 0.3312 | 0.2937 | 0.7834 | 0.6546 | 0.7920 | 0.0535 | 0.6102 |

## Semantic cluster structure: triangles and cross-cluster edges

`triangles/internal edge` is the number of triangles fully inside a semantic cluster divided by the number of internal behaviour/q edges. `cross rate` is the fraction of incident edges that leave the semantic cluster.

| arm | triangles before | triangles after | triangles/edge Spearman | triangles/edge MAE | cross rate before | cross rate after | cross-rate gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 546438 | 390495 | 0.303 | 41.4457 | 0.9725 | 0.5269 | -0.4567 |
| mixed_F0 | 546438 | 383166 | 0.243 | 41.3648 | 0.9725 | 0.5609 | -0.4235 |

## Semantic clusters vs quantized-vector clusters

| arm | q clusters | q max share | ARI | NMI | forward coverage | profile JSD (semantic→q) |
|---|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 256 | 0.0270 | 0.4655 | 0.7904 | 0.7330 | 0.2597 |
| mixed_F0 | 256 | 0.0168 | 0.4189 | 0.7498 | 0.6320 | 0.2860 |

## Top semantic clusters by size

| arm | cluster | size | top role before | top role after | profile JSD | multi before | multi after | cross rate before | cross rate after |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 211 | 338 | K1_4 | P5 | 0.1663 | 0.343 | 0.163 | 0.9559 | 0.3705 |
| pure_euclidean_TIGER | 19 | 317 | K1_4 | P5 | 0.2374 | 0.233 | 0.050 | 0.9571 | 0.5074 |
| pure_euclidean_TIGER | 241 | 312 | P5 | P5 | 0.1759 | 0.266 | 0.042 | 0.9118 | 0.1758 |
| pure_euclidean_TIGER | 125 | 270 | K1_4 | P5 | 0.2833 | 0.285 | 0.107 | 0.9785 | 0.4686 |
| pure_euclidean_TIGER | 77 | 266 | K1_4 | P5 | 0.2297 | 0.308 | 0.008 | 0.9593 | 0.4154 |
| pure_euclidean_TIGER | 29 | 264 | K1_4 | P5 | 0.2043 | 0.398 | 0.045 | 0.9765 | 0.6753 |
| pure_euclidean_TIGER | 139 | 259 | K1_4 | P5 | 0.2113 | 0.309 | 0.108 | 0.9734 | 0.6051 |
| pure_euclidean_TIGER | 78 | 246 | K1_4 | P5 | 0.3199 | 0.272 | 0.012 | 0.9702 | 0.1204 |
| pure_euclidean_TIGER | 17 | 241 | K1_4 | P5 | 0.2727 | 0.270 | 0.004 | 0.9821 | 0.3582 |
| pure_euclidean_TIGER | 207 | 233 | K1_4 | P5 | 0.1679 | 0.300 | 0.296 | 0.9739 | 0.4462 |
| pure_euclidean_TIGER | 63 | 220 | K1_4 | P5 | 0.2439 | 0.359 | 0.009 | 0.9686 | 0.5002 |
| pure_euclidean_TIGER | 194 | 212 | P5 | P5 | 0.0832 | 0.241 | 0.042 | 0.9111 | 0.3746 |
| mixed_F0 | 211 | 338 | K1_4 | P5 | 0.2318 | 0.343 | 0.000 | 0.9559 | 0.3666 |
| mixed_F0 | 19 | 317 | K1_4 | P5 | 0.2659 | 0.233 | 0.032 | 0.9571 | 0.5968 |
| mixed_F0 | 241 | 312 | P5 | P5 | 0.2169 | 0.266 | 0.003 | 0.9118 | 0.2141 |
| mixed_F0 | 125 | 270 | K1_4 | P5 | 0.3321 | 0.285 | 0.022 | 0.9785 | 0.5552 |
| mixed_F0 | 77 | 266 | K1_4 | P5 | 0.2515 | 0.308 | 0.011 | 0.9593 | 0.5154 |
| mixed_F0 | 29 | 264 | K1_4 | P5 | 0.2307 | 0.398 | 0.000 | 0.9765 | 0.7448 |
| mixed_F0 | 139 | 259 | K1_4 | P5 | 0.2607 | 0.309 | 0.004 | 0.9734 | 0.6795 |
| mixed_F0 | 78 | 246 | K1_4 | P5 | 0.3590 | 0.272 | 0.000 | 0.9702 | 0.1176 |
| mixed_F0 | 17 | 241 | K1_4 | P5 | 0.2577 | 0.270 | 0.129 | 0.9821 | 0.3825 |
| mixed_F0 | 207 | 233 | K1_4 | P5 | 0.2609 | 0.300 | 0.000 | 0.9739 | 0.3815 |
| mixed_F0 | 63 | 220 | K1_4 | P5 | 0.2545 | 0.359 | 0.000 | 0.9686 | 0.5020 |
| mixed_F0 | 194 | 212 | P5 | P5 | 0.1023 | 0.241 | 0.000 | 0.9111 | 0.3738 |

## Top role-pair changes (same semantic clusters)

| arm | left role | right role | edges | before within | after within | before lift | after lift | delta lift |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | K1_4 | K1_4 | 204492 | 0.0432 | 0.6683 | 0.806 | 1.040 | +0.235 |
| pure_euclidean_TIGER | P5 | P5 | 24866 | 0.1075 | 0.6390 | 2.007 | 0.995 | -1.012 |
| pure_euclidean_TIGER | P5 | K1_4 | 86033 | 0.0627 | 0.6058 | 1.170 | 0.943 | -0.227 |
| mixed_F0 | P5 | P5 | 24866 | 0.1075 | 0.6088 | 2.007 | 0.998 | -1.009 |
| mixed_F0 | K1_4 | K1_4 | 204492 | 0.0432 | 0.5478 | 0.806 | 0.898 | +0.092 |
| mixed_F0 | P5 | K1_4 | 86033 | 0.0627 | 0.6545 | 1.170 | 1.073 | -0.098 |
