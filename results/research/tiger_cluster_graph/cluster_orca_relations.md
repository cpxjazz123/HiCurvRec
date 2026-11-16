# Cluster-level ORCA relations before vs after quantisation

Before: Louvain communities of the TRAIN behaviour graph. After: actual L1/L2/L3 SID prefix clusters. No KMeans.

Behaviour graph: **24587** nodes, **315391** edges, **37** Louvain labels: **6** active communities plus **31** isolated nodes. Role JSD / boundary JSD are computed on the active communities only. Multi-role items: **30.992%**.

## Partition agreement and matched-cluster role profiles

`coverage` is the largest fraction of a behaviour community captured by one SID cluster. `role JSD` compares mean ORCA role profiles of matched clusters; `boundary JSD` compares the role profiles of their unique external behaviour neighbours.

| arm | depth | SID clusters | ARI | NMI | forward coverage | forward role JSD | role valid | reverse coverage | reverse role JSD | boundary JSD | boundary valid | multi-role MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | L1 | 249 | 0.0350 | 0.2443 | 0.0879 | 0.0022 | 6 | 0.6649 | 0.0051 | 0.0031 | 6 | 0.0190 |
| pure_euclidean_TIGER | L2 | 7374 | 0.0017 | 0.2422 | 0.0119 | 0.0172 | 6 | 0.7734 | 0.0308 | 0.0161 | 6 | 0.0935 |
| pure_euclidean_TIGER | L3 | 20425 | 0.0001 | 0.2812 | 0.0050 | 0.0082 | 6 | 0.9476 | 0.0803 | 0.0204 | 6 | 0.0889 |
| mixed_F0 | L1 | 79 | 0.0821 | 0.2640 | 0.1687 | 0.0064 | 6 | 0.6443 | 0.0047 | 0.0016 | 6 | 0.0435 |
| mixed_F0 | L2 | 4151 | 0.0046 | 0.2346 | 0.0260 | 0.0010 | 6 | 0.7313 | 0.0190 | 0.0071 | 6 | 0.0172 |
| mixed_F0 | L3 | 18264 | 0.0002 | 0.2759 | 0.0060 | 0.0120 | 6 | 0.9182 | 0.0723 | 0.0133 | 6 | 0.0922 |

## Per-role-pair within-cluster fraction

For each unordered ORCA role pair, the fraction of its behaviour edges whose endpoints share a cluster. `pair MAE` is weighted by the number of edges of that role pair. `pair Spearman` is computed on role pairs with at least 100 edges; there are only three such pairs in these data, so it is not informative and is reported only for completeness.

`pair MAE` is the raw within-fraction difference; `lift MAE` uses the same-fraction lift `within_fraction / global_same_cluster_fraction`, which controls the overall cluster granularity. `pairs_used` is small when only a few ORCA role pairs have enough edges.

| arm | depth | global same before | global same after | pair MAE | lift MAE | pair Spearman | pairs used | within role-pair JSD | cross role-pair JSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | L1 | 0.5979 | 0.0811 | 0.5168 | 0.1564 | 1.000 | 3 | 0.0054 | 0.0045 |
| pure_euclidean_TIGER | L2 | 0.5979 | 0.0080 | 0.5900 | 0.2328 | 1.000 | 3 | 0.0131 | 0.0058 |
| pure_euclidean_TIGER | L3 | 0.5979 | 0.0017 | 0.5963 | 0.1960 | 1.000 | 3 | 0.0121 | 0.0060 |
| mixed_F0 | L1 | 0.5979 | 0.1446 | 0.4534 | 0.1207 | 1.000 | 3 | 0.0030 | 0.0037 |
| mixed_F0 | L2 | 0.5979 | 0.0188 | 0.5792 | 0.2340 | 1.000 | 3 | 0.0143 | 0.0055 |
| mixed_F0 | L3 | 0.5979 | 0.0024 | 0.5955 | 0.3497 | 1.000 | 3 | 0.0323 | 0.0059 |

## Top role-pair changes at L1

| arm | left role | right role | edges | before within | after within | before lift | after lift | delta lift |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | K1_4 | K1_4 | 204492 | 0.5699 | 0.0675 | 0.953 | 0.833 | -0.121 |
| pure_euclidean_TIGER | P5 | P5 | 24866 | 0.8095 | 0.1556 | 1.354 | 1.918 | +0.565 |
| pure_euclidean_TIGER | P5 | K1_4 | 86033 | 0.6034 | 0.0919 | 1.009 | 1.133 | +0.123 |
| mixed_F0 | K1_4 | K1_4 | 204492 | 0.5699 | 0.1244 | 0.953 | 0.860 | -0.093 |
| mixed_F0 | P5 | K1_4 | 86033 | 0.6034 | 0.1625 | 1.009 | 1.124 | +0.115 |
| mixed_F0 | P5 | P5 | 24866 | 0.8095 | 0.2488 | 1.354 | 1.721 | +0.367 |
