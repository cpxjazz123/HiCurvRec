# Why are the recovered TIGER behaviour edges recovered?

TIGER only. Edge-budget-matched global k=10 z/q graphs from `embedding_knn_compare.py`. Each row is a real TRAIN behaviour edge.

Behaviour edges **315391**; z recovered **5178 (1.6418%)**; q recovered **3568 (1.1313%)**.
Matched edge budget: z 199094→186044, q 186044→186044.

## Retention by local structure and community

| type | edges | z recovered | z rate | z lift | q recovered | q rate | q lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| triangle | 185711 | 3130 | 0.01685 | 1.027 | 2123 | 0.01143 | 1.011 |
| non_triangle | 129680 | 2048 | 0.01579 | 0.962 | 1445 | 0.01114 | 0.985 |
| k4 | 104441 | 1446 | 0.01385 | 0.843 | 969 | 0.00928 | 0.820 |
| non_k4 | 210950 | 3732 | 0.01769 | 1.078 | 2599 | 0.01232 | 1.089 |
| common_0 | 129680 | 2048 | 0.01579 | 0.962 | 1445 | 0.01114 | 0.985 |
| common_1 | 54593 | 1051 | 0.01925 | 1.173 | 729 | 0.01335 | 1.180 |
| common_2 | 28927 | 567 | 0.01960 | 1.194 | 402 | 0.01390 | 1.228 |
| common_3plus | 102191 | 1512 | 0.01480 | 0.901 | 992 | 0.00971 | 0.858 |
| within_community | 188588 | 4563 | 0.02420 | 1.474 | 3151 | 0.01671 | 1.477 |
| cross_community | 126803 | 615 | 0.00485 | 0.295 | 417 | 0.00329 | 0.291 |
| multi_edge | 178083 | 2795 | 0.01569 | 0.956 | 1910 | 0.01073 | 0.948 |
| single_edge | 137308 | 2383 | 0.01736 | 1.057 | 1658 | 0.01208 | 1.067 |
| star_like | 355 | 10 | 0.02817 | 1.716 | 8 | 0.02254 | 1.992 |

`non_triangle` is used as the chain-like/acyclic proxy; `star_like` = common neighbours 0, max degree ≥ 5, min degree ≤ 2. These are heuristics, not exact 5-node graphlet classes.

## Degree control

| dmax quintile | dmax range | edges | z rate | z lift | q rate | q lift |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 2–33 | 63833 | 0.03255 | 1.983 | 0.02206 | 1.950 |
| Q2 | 34–72 | 62878 | 0.01985 | 1.209 | 0.01396 | 1.234 |
| Q3 | 73–145 | 62668 | 0.01495 | 0.911 | 0.01056 | 0.934 |
| Q4 | 146–312 | 63007 | 0.00903 | 0.550 | 0.00635 | 0.561 |
| Q5 | 313–2220 | 63005 | 0.00549 | 0.334 | 0.00349 | 0.309 |

## Logistic controls (odds ratios)

| model | term | coef | odds ratio | p |
|---|---|---:|---:|---:|
| z recovery | const | -2.0846 | 0.124 | 0.0000 |
| z recovery | log_du | -0.4263 | 0.653 | 0.0000 |
| z recovery | log_dv | -0.5583 | 0.572 | 0.0000 |
| z recovery | triangle | +0.8329 | 2.300 | 0.0000 |
| z recovery | k4 | +0.6153 | 1.850 | 0.0000 |
| z recovery | within_community | +1.3395 | 3.817 | 0.0000 |
| q retention given z | const | +0.1205 | 1.128 | 0.4710 |
| q retention given z | log_du | -0.0842 | 0.919 | 0.0155 |
| q retention given z | log_dv | -0.1371 | 0.872 | 0.0000 |
| q retention given z | triangle | +0.1420 | 1.153 | 0.0479 |
| q retention given z | k4 | +0.1593 | 1.173 | 0.0708 |
| q retention given z | within_community | +0.1698 | 1.185 | 0.0586 |

## Four fates of a behaviour edge

| fate | edges | share |
|---|---:|---:|
| always | 2107 | 0.0067 |
| q_lost | 3071 | 0.0097 |
| q_added | 1461 | 0.0046 |
| never | 308752 | 0.9789 |

## ORCA role pairs: z recovery

| left role | right role | edges | z recovered | z rate | z lift |
|---|---|---:|---:|---:|---:|
| K1_4 | K1_4 | 204492 | 2587 | 0.01265 | 0.771 |
| P5 | K1_4 | 86033 | 1720 | 0.01999 | 1.218 |
| P5 | P5 | 24866 | 871 | 0.03503 | 2.134 |

## Quantisation selectivity among z-recovered edges

| group | total | retained by q | lost by q | q retention |
|---|---:|---:|---:|---:|
| triangle | 3130 | 1271 | 1859 | 0.4061 |
| non_triangle | 2048 | 836 | 1212 | 0.4082 |
| k4 | 1446 | 573 | 873 | 0.3963 |
| non_k4 | 3732 | 1534 | 2198 | 0.4110 |
| within_community | 4563 | 1883 | 2680 | 0.4127 |
| cross_community | 615 | 224 | 391 | 0.3642 |
| multi_edge | 2795 | 1125 | 1670 | 0.4025 |
| single_edge | 2383 | 982 | 1401 | 0.4121 |
