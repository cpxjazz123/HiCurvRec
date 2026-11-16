# Behavior Graph → Quantized SID Structural Fidelity

Ground truth: TRAIN behaviour graph; both arms use the same graph and ORCA table.
SID clusters are the actual L1/L2/L3 prefixes from item_sids.json (no KMeans).

Behaviour graph: **24587** nodes, **315391** edges, **37** Louvain communities.
Multi-role items: **30.992%**.

## Behaviour-edge prefix rates

Raw prefix rates are not comparable across arms by themselves: a collapsed codebook makes prefix sharing easier. `lift@k = prefix@k / E_random@k`, where `E_random@k = sum_c (n_c/n)^2` over that arm's own Lk clusters; lift controls for cluster granularity and imbalance.

| arm | prefix@1 | lift@1 | prefix@2 | lift@2 | prefix@3 | lift@3 | mean LCP | LCP 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pure_euclidean_TIGER | 0.0811 | 7.48 | 0.0080 | 16.82 | 0.0017 | 22.64 | 0.091 | [289810, 23063, 1993, 525] |
| mixed_F0 | 0.1446 | 4.85 | 0.0188 | 14.09 | 0.0024 | 24.70 | 0.166 | [269791, 39684, 5151, 765] |

Paired mixed − pure (same edges): prefix@1 +0.0635 ± 0.0010; prefix@2 +0.0108 ± 0.0005; prefix@3 +0.0008 ± 0.0002 (95% CI).

## SID prefix clusters

| arm | depth | clusters | max share | random prefix | within-edge share | within density | density lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | L1 | 249 | 0.029 | 0.0108 | 0.0423 | 0.00784 | 7.51 |
| pure_euclidean_TIGER | L2 | 7374 | 0.003 | 0.0005 | 0.0040 | 0.01920 | 18.40 |
| pure_euclidean_TIGER | L3 | 20425 | 0.001 | 0.0001 | 0.0008 | 0.05286 | 50.66 |
| mixed_F0 | L1 | 79 | 0.059 | 0.0298 | 0.0779 | 0.00507 | 4.86 |
| mixed_F0 | L2 | 4151 | 0.007 | 0.0013 | 0.0095 | 0.01516 | 14.53 |
| mixed_F0 | L3 | 18264 | 0.002 | 0.0001 | 0.0012 | 0.04400 | 42.16 |

## Communities: within vs cross prefix rates

| arm | within prefix@1 | cross prefix@1 | gap | within lift@1 | cross lift@1 | gap lift | within prefix@2 | cross prefix@2 | L1 community purity | normalized purity | L1 NMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 0.1170 | 0.0277 | +0.0892 | 10.79 | 2.56 | +8.23 | 0.0118 | 0.0023 | 0.6649 | 0.5854 | 0.2453 |
| mixed_F0 | 0.2022 | 0.0589 | +0.1433 | 6.78 | 1.98 | +4.81 | 0.0275 | 0.0057 | 0.6443 | 0.5600 | 0.2648 |

## ORCA role alignment

| arm | role purity L1 | normalized role purity L1 | role purity L2 | role NMI L1 | role ARI L1 | multi-role edge prefix@1 | single-role edge prefix@1 | multi-role edge prefix@2 | single-role edge prefix@2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 0.6361 | 0.2619 | 0.7515 | 0.0175 | 0.0016 | 0.0808 | 0.0815 | 0.0077 | 0.0083 |
| mixed_F0 | 0.6250 | 0.2394 | 0.6989 | 0.0169 | 0.0035 | 0.1437 | 0.1458 | 0.0174 | 0.0205 |

## Graphlet instances

Values are the fraction of induced behaviour edges whose endpoints share the prefix; `all@k` is the fraction of instances where all five nodes share the prefix.

| arm | pattern | instances | induced edges | prefix@1 | lift@1 | prefix@2 | lift@2 | prefix@3 | lift@3 | all@1 | all@2 | all@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | K5 | 300 | 3000 | 0.0723 | 6.67 | 0.0033 | 7.02 | 0.0007 | 9.07 | 0.0000 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | double_triangle | 119 | 714 | 0.1541 | 14.21 | 0.0196 | 41.32 | 0.0028 | 38.09 | 0.0252 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | K4_tail | 300 | 2100 | 0.0614 | 5.67 | 0.0029 | 6.02 | 0.0019 | 25.90 | 0.0000 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | K2_3 | 300 | 1800 | 0.0600 | 5.54 | 0.0061 | 12.88 | 0.0011 | 15.11 | 0.0000 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | P5_centre | 300 | 1200 | 0.0867 | 8.00 | 0.0092 | 19.32 | 0.0008 | 11.33 | 0.0067 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | C5 | 300 | 1500 | 0.0787 | 7.26 | 0.0060 | 12.64 | 0.0027 | 36.27 | 0.0000 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | P5_hops | 300 | 1200 | 0.0867 | 8.00 | 0.0092 | 19.32 | 0.0008 | 11.33 | 0.0067 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | K1_4 | 300 | 1200 | 0.0892 | 8.23 | 0.0083 | 17.56 | 0.0017 | 22.67 | 0.0000 | 0.0000 | 0.0000 |
| pure_euclidean_TIGER | T5 | 300 | 1200 | 0.0908 | 8.38 | 0.0125 | 26.34 | 0.0025 | 34.00 | 0.0000 | 0.0000 | 0.0000 |
| mixed_F0 | K5 | 300 | 3000 | 0.0927 | 3.11 | 0.0150 | 11.26 | 0.0007 | 6.79 | 0.0000 | 0.0000 | 0.0000 |
| mixed_F0 | double_triangle | 119 | 714 | 0.2297 | 7.70 | 0.0490 | 36.81 | 0.0070 | 71.32 | 0.0336 | 0.0000 | 0.0000 |
| mixed_F0 | K4_tail | 300 | 2100 | 0.0981 | 3.29 | 0.0110 | 8.22 | 0.0010 | 9.70 | 0.0033 | 0.0000 | 0.0000 |
| mixed_F0 | K2_3 | 300 | 1800 | 0.1222 | 4.10 | 0.0128 | 9.59 | 0.0028 | 28.29 | 0.0100 | 0.0000 | 0.0000 |
| mixed_F0 | P5_centre | 300 | 1200 | 0.1583 | 5.31 | 0.0208 | 15.64 | 0.0000 | 0.00 | 0.0100 | 0.0000 | 0.0000 |
| mixed_F0 | C5 | 300 | 1500 | 0.1460 | 4.90 | 0.0207 | 15.52 | 0.0053 | 54.31 | 0.0033 | 0.0000 | 0.0000 |
| mixed_F0 | P5_hops | 300 | 1200 | 0.1583 | 5.31 | 0.0208 | 15.64 | 0.0000 | 0.00 | 0.0100 | 0.0000 | 0.0000 |
| mixed_F0 | K1_4 | 300 | 1200 | 0.1500 | 5.03 | 0.0208 | 15.64 | 0.0042 | 42.43 | 0.0033 | 0.0000 | 0.0000 |
| mixed_F0 | T5 | 300 | 1200 | 0.1583 | 5.31 | 0.0250 | 18.77 | 0.0050 | 50.92 | 0.0033 | 0.0000 | 0.0000 |

## Top ORCA role pairs by behaviour-edge count

| arm | left role | right role | edges | prefix@1 | prefix@2 | prefix@3 |
|---|---|---|---:|---:|---:|---:|
| pure_euclidean_TIGER | K1_4 | K1_4 | 204492 | 0.0675 | 0.0062 | 0.0013 |
| pure_euclidean_TIGER | P5 | K1_4 | 86033 | 0.0919 | 0.0091 | 0.0018 |
| pure_euclidean_TIGER | P5 | P5 | 24866 | 0.1556 | 0.0189 | 0.0041 |
| mixed_F0 | K1_4 | K1_4 | 204492 | 0.1244 | 0.0145 | 0.0017 |
| mixed_F0 | P5 | K1_4 | 86033 | 0.1625 | 0.0209 | 0.0027 |
| mixed_F0 | P5 | P5 | 24866 | 0.2488 | 0.0463 | 0.0077 |
