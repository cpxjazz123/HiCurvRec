# Graphlet Role x relation x SID prefix tables
TRAIN edges 315,391; TEST rows 57,439; T5 checkpoint `results/stage3_T5Train/TIGER_RQ-VAE/ckpt/Amazon_2023_Instruments/Nov-12-2026_21-04-04/HG_Rec_best.pth` (recall@10 0.05545).
## Experiment 1 - relation profile by local structure
| structure | n edges | % of edges | sem cos | title Jaccard | co-occur users | max p(trans) | same community | future hit | future lift | mean LCP | prefix@1 | prefix@2 | prefix@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all edges | 315,391 | 100.00% | 0.726 | 0.044 | 1.51 | 0.108 | 59.79% | 6.39% | 0.044 | 0.091 | 8.11% | 0.80% | 0.17% |
| K5 | 4,825 | 1.53% | 0.749 | 0.051 | 9.08 | 0.011 | 83.15% | 59.25% | 0.334 | 0.074 | 6.80% | 0.50% | 0.12% |
| double_triangle | 706 | 0.22% | 0.745 | 0.066 | 2.07 | 0.123 | 70.96% | 9.07% | 0.065 | 0.178 | 15.58% | 1.98% | 0.28% |
| K4_tail | 5,470 | 1.73% | 0.744 | 0.048 | 7.14 | 0.026 | 74.24% | 44.52% | 0.246 | 0.071 | 6.47% | 0.53% | 0.13% |
| K2_3 | 5,800 | 1.84% | 0.726 | 0.046 | 3.32 | 0.087 | 60.17% | 19.62% | 0.109 | 0.077 | 6.84% | 0.71% | 0.12% |
| P5_centre | 3,975 | 1.26% | 0.726 | 0.044 | 1.31 | 0.134 | 61.43% | 4.81% | 0.036 | 0.095 | 8.30% | 0.91% | 0.28% |
| C5 | 4,951 | 1.57% | 0.730 | 0.045 | 1.93 | 0.095 | 61.36% | 9.47% | 0.057 | 0.093 | 8.40% | 0.75% | 0.10% |
| P5_hops | 3,975 | 1.26% | 0.726 | 0.044 | 1.31 | 0.134 | 61.43% | 4.81% | 0.036 | 0.095 | 8.30% | 0.91% | 0.28% |
| K1_4 | 3,959 | 1.26% | 0.724 | 0.047 | 1.12 | 0.176 | 61.66% | 1.64% | 0.015 | 0.110 | 9.60% | 1.09% | 0.28% |
| T5 | 3,957 | 1.25% | 0.723 | 0.049 | 1.18 | 0.161 | 62.42% | 2.88% | 0.024 | 0.112 | 9.83% | 1.14% | 0.20% |

Matched differences (structure minus the rest, stratified by degree and popularity deciles):

| structure | d sem cos | d title | d co-occur | d p(trans) | d future hit | d LCP | d prefix@1 | d prefix@1 (semantic-controlled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K5 | +0.009 | +0.011 | +3.18 | +0.000 | +15.93pp | +0.019 | +1.73pp | +0.32pp |
| double_triangle | +0.018 | +0.019 | +0.39 | -0.004 | +1.24pp | +0.076 | +6.80pp | +2.87pp |
| K4_tail | +0.007 | +0.007 | +2.21 | +0.001 | +8.45pp | +0.016 | +1.36pp | +0.79pp |
| K2_3 | -0.003 | +0.005 | +0.89 | +0.002 | +4.74pp | +0.004 | +0.34pp | +0.13pp |
| P5_centre | +0.002 | -0.001 | -0.03 | +0.003 | +0.20pp | -0.003 | -0.52pp | -0.41pp |
| C5 | +0.003 | +0.002 | +0.13 | -0.001 | +0.56pp | +0.009 | +0.95pp | +0.38pp |
| P5_hops | +0.002 | -0.001 | -0.03 | +0.003 | +0.20pp | -0.003 | -0.52pp | -0.41pp |
| K1_4 | +0.002 | -0.001 | -0.01 | +0.004 | -0.36pp | -0.001 | -0.08pp | +0.28pp |
| T5 | +0.000 | +0.003 | -0.02 | +0.004 | -0.11pp | +0.004 | +0.40pp | -1.29pp |

### Exact structural bins

| bin | n | future hit | co-occur users | max p(trans) | prefix@1 | prefix@2 | prefix@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no triangle | 129,680 | 0.99% | 1.04 | 0.162 | 7.29% | 0.72% | 0.15% |
| one triangle | 54,593 | 2.02% | 1.13 | 0.114 | 8.95% | 0.96% | 0.23% |
| 2+ triangles | 131,118 | 13.54% | 2.13 | 0.051 | 8.57% | 0.80% | 0.16% |
| different Louvain community | 126,803 | 4.31% | 1.29 | 0.099 | 2.77% | 0.23% | 0.05% |
| same Louvain community | 188,588 | 7.78% | 1.65 | 0.113 | 11.70% | 1.18% | 0.25% |
| non-bridge | 315,340 | 6.39% | 1.51 | 0.108 | 8.11% | 0.80% | 0.17% |
| bridge | 51 | 7.84% | 1.00 | 0.583 | 21.57% | 1.96% | 0.00% |

### Role pairs (undirected, by endpoint ORCA role)

| role pair | n | sem cos | co-occur users | max p(trans) | future hit | prefix@1 | prefix@2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| role0-role5 | 94,373 | 0.722 | 1.34 | 0.099 | 5.40% | 5.85% | 0.50% |
| role5-role5 | 51,112 | 0.718 | 1.08 | 0.133 | 1.17% | 8.10% | 0.89% |
| role3-role5 | 42,668 | 0.723 | 1.11 | 0.115 | 1.77% | 8.77% | 0.88% |
| role0-role0 | 39,774 | 0.736 | 3.46 | 0.033 | 26.00% | 6.70% | 0.50% |
| role0-role3 | 29,896 | 0.725 | 1.33 | 0.083 | 5.72% | 7.42% | 0.53% |
| role3-role3 | 13,465 | 0.738 | 1.31 | 0.091 | 4.03% | 12.58% | 1.23% |
| role0-role4 | 6,686 | 0.729 | 1.31 | 0.082 | 5.15% | 7.70% | 0.63% |
| role4-role5 | 6,370 | 0.717 | 1.08 | 0.117 | 1.05% | 9.23% | 1.19% |
| role2-role5 | 6,285 | 0.722 | 1.06 | 0.233 | 1.05% | 11.28% | 1.26% |
| role2-role3 | 5,538 | 0.748 | 1.23 | 0.194 | 2.55% | 14.70% | 2.13% |
| role2-role2 | 3,303 | 0.764 | 1.17 | 0.272 | 3.39% | 19.83% | 2.91% |
| role1-role3 | 3,123 | 0.746 | 1.28 | 0.194 | 3.14% | 18.25% | 2.34% |

## Experiment 2 - relation -> SID prefix

### Semantic quintiles

| group | n | mean LCP | prefix@1 | prefix@2 | prefix@3 | quantised L2 | q cos | matched d prefix@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 63,079 | 0.004 | 0.37% | 0.00% | 0.00% | 0.260 | 0.433 | -9.91pp |
| 1 | 63,078 | 0.013 | 1.25% | 0.01% | 0.00% | 0.157 | 0.525 | -8.35pp |
| 2 | 63,078 | 0.024 | 2.35% | 0.01% | 0.00% | 0.131 | 0.582 | -7.09pp |
| 3 | 63,078 | 0.053 | 5.16% | 0.10% | 0.00% | 0.110 | 0.663 | -3.51pp |
| 4 | 63,078 | 0.361 | 31.42% | 3.87% | 0.83% | 0.079 | 0.807 | +28.89pp |

### Title-token Jaccard

| group | n | mean LCP | prefix@1 | prefix@2 | prefix@3 | quantised L2 | q cos | matched d prefix@1 | d prefix@1 (sem-controlled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 165,237 | 0.020 | 1.97% | 0.05% | 0.00% | 0.163 | 0.527 | -12.96pp | -4.47pp |
| 1 | 53,187 | 0.042 | 4.10% | 0.11% | 0.01% | 0.140 | 0.612 | -4.56pp | -6.38pp |
| 2 | 76,921 | 0.140 | 13.27% | 0.73% | 0.04% | 0.133 | 0.690 | +6.92pp | +0.77pp |
| 3 | 20,046 | 0.610 | 49.55% | 9.04% | 2.41% | 0.095 | 0.855 | +43.68pp | +32.76pp |

### Co-occurrence

| group | n | mean LCP | prefix@1 | prefix@2 | prefix@3 | quantised L2 | q cos | matched d prefix@1 | d prefix@1 (sem-controlled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 250,282 | 0.076 | 6.92% | 0.61% | 0.11% | 0.151 | 0.590 | -10.51pp | -3.44pp |
| 1 | 49,014 | 0.142 | 12.41% | 1.44% | 0.36% | 0.135 | 0.644 | +7.31pp | +2.26pp |
| 2 | 13,525 | 0.157 | 13.46% | 1.80% | 0.44% | 0.131 | 0.654 | +9.35pp | +2.40pp |
| 3 | 2,570 | 0.156 | 13.62% | 1.60% | 0.43% | 0.131 | 0.649 | +9.00pp | +3.25pp |

### Directed transition strength

| group | n | mean LCP | prefix@1 | prefix@2 | prefix@3 | quantised L2 | q cos | matched d prefix@1 | d prefix@1 (sem-controlled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40,141 | 0.055 | 5.11% | 0.37% | 0.04% | 0.144 | 0.572 | -2.22pp | -0.91pp |
| 1 | 88,458 | 0.078 | 7.10% | 0.57% | 0.10% | 0.144 | 0.597 | -0.31pp | +0.15pp |
| 2 | 112,227 | 0.096 | 8.57% | 0.87% | 0.18% | 0.149 | 0.606 | +0.10pp | +0.21pp |
| 3 | 74,565 | 0.117 | 10.24% | 1.19% | 0.29% | 0.151 | 0.617 | +1.11pp | -0.15pp |

### Triangles per edge

| group | n | mean LCP | prefix@1 | prefix@2 | prefix@3 | quantised L2 | q cos | matched d prefix@1 | d prefix@1 (sem-controlled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 129,680 | 0.082 | 7.29% | 0.72% | 0.15% | 0.154 | 0.590 | -7.66pp | -2.93pp |
| 1 | 54,593 | 0.101 | 8.95% | 0.96% | 0.23% | 0.147 | 0.609 | +0.87pp | +0.32pp |
| 2 | 131,118 | 0.095 | 8.57% | 0.80% | 0.16% | 0.141 | 0.611 | +8.99pp | +3.43pp |

### Graphlet x semantic-high

| structure | high-sem n | high-sem prefix@1 | other n | other prefix@1 |
|---|---:|---:|---:|---:|
| K5 | 1,217 | 24.65% | 3,608 | 0.78% |
| double_triangle | 192 | 47.92% | 514 | 3.50% |
| K4_tail | 1,232 | 25.00% | 4,238 | 1.09% |
| K2_3 | 1,109 | 28.58% | 4,691 | 1.71% |
| P5_centre | 827 | 31.68% | 3,148 | 2.16% |
| C5 | 1,030 | 32.52% | 3,921 | 2.07% |
| P5_hops | 827 | 31.68% | 3,148 | 2.16% |
| K1_4 | 836 | 36.84% | 3,123 | 2.31% |
| T5 | 871 | 34.21% | 3,086 | 2.95% |

## Experiment 3 - T5 prediction by relation

`matched d recall` is the within-cell difference in recall@10: popularity and history-length deciles for every group, plus semantic-similarity and title bins for prefix, co-occurrence, transition, triangle, community, seen-edge and graphlet groups (never for the variable used as the grouping). Parentheses give the standard error and the number of cells with >=5 rows on both sides; `n/a` means no cell had enough support. `adj. d recall` is the OLS-residual difference on the same controls and is more stable, but it is linear.

### Overall and SID prefix

| group | n | recall@10 | MRR@10 | mean log p | mean rest log p | median rank (11=miss) | adj. d recall | adj. d log p | matched d recall | matched d rest log p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all test rows | 57,439 | 5.55% | 0.0228 | -9.180 | -5.157 | 11 | n/a | n/a | n/a | n/a |
| prefix1: different L1 | 52,333 | 4.73% | 0.0193 | -9.330 | -5.127 | 11 | -5.13pp | -0.912 | -0.89pp (±1.66, 66c) | 0.083 |
| prefix1: same L1 | 5,106 | 13.89% | 0.0585 | -7.650 | -5.472 | 11 | +5.13pp | 0.912 | +0.89pp (±1.66, 66c) | -0.083 |
| prefix2: LCP < 2 | 56,912 | 5.44% | 0.0223 | -9.196 | -5.156 | 11 | -5.29pp | -0.424 | +0.00pp (±0.00, 1c) | 0.200 |
| prefix2: same L1+L2 | 527 | 17.08% | 0.0705 | -7.508 | -5.271 | 11 | +5.29pp | 0.424 | +0.00pp (±0.00, 1c) | -0.200 |
| prefix3: LCP < 3 | 57,331 | 5.53% | 0.0227 | -9.184 | -5.158 | 11 | +2.63pp | -0.114 | n/a | n/a |
| prefix3: same L1+L2+L3 | 108 | 12.04% | 0.0338 | -7.384 | -5.046 | 11 | -2.63pp | 0.114 | n/a | n/a |
| lcp: LCP 0 | 52,333 | 4.73% | 0.0193 | -9.330 | -5.127 | 11 | -5.13pp | -0.912 | -0.89pp (±1.66, 66c) | 0.083 |
| lcp: LCP 1 | 4,579 | 13.52% | 0.0571 | -7.666 | -5.495 | 11 | +5.01pp | 0.955 | +1.04pp (±1.85, 65c) | -0.203 |
| lcp: LCP 2 | 419 | 18.38% | 0.0800 | -7.540 | -5.329 | 11 | +7.32pp | 0.503 | +0.00pp (±0.00, 1c) | -0.200 |
| lcp: LCP 3 | 108 | 12.04% | 0.0338 | -7.384 | -5.046 | 11 | -2.63pp | 0.114 | n/a | n/a |

### Relation classes

| group | n | recall@10 | MRR@10 | mean log p | mean target pop | adj. d recall | matched d recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| sem_bin: semantic Q1 | 9,565 | 2.59% | 0.0095 | -9.869 | 94.7 | +0.91pp | -2.64pp (±0.18, 951c) |
| sem_bin: semantic Q2 | 11,120 | 4.14% | 0.0161 | -9.517 | 112.7 | -0.60pp | -2.14pp (±0.19, 978c) |
| sem_bin: semantic Q3 | 11,731 | 4.34% | 0.0176 | -9.408 | 96.8 | -0.84pp | -0.95pp (±0.19, 974c) |
| sem_bin: semantic Q4 | 12,015 | 5.23% | 0.0212 | -9.017 | 103.1 | -1.20pp | -0.51pp (±0.20, 977c) |
| sem_bin: semantic Q5 | 13,008 | 10.30% | 0.0444 | -8.332 | 111.5 | +1.72pp | +5.32pp (±0.23, 969c) |
| title_bin: title Jaccard 0 | 28,534 | 2.98% | 0.0114 | -9.675 | 88.4 | -0.99pp | -3.32pp (±0.99, 175c) |
| title_bin: 0 < J <= .05 | 11,171 | 6.20% | 0.0267 | -9.082 | 118.9 | +0.24pp | +0.83pp (±1.90, 88c) |
| title_bin: .05 < J <= .15 | 13,917 | 8.57% | 0.0354 | -8.577 | 123.8 | +1.09pp | -0.72pp (±1.27, 186c) |
| title_bin: J > .15 | 3,817 | 11.79% | 0.0506 | -7.974 | 107.1 | +0.16pp | -0.60pp (±1.91, 59c) |
| cooc_bin: co-occur 1 user | 52,864 | 3.21% | 0.0115 | -9.442 | 79.5 | -12.29pp | -19.47pp (±2.52, 49c) |
| cooc_bin: 2-3 users | 1,433 | 16.26% | 0.0643 | -6.965 | 200.8 | +2.40pp | -2.74pp (±15.27, 2c) |
| cooc_bin: 4-9 users | 1,598 | 26.35% | 0.1122 | -6.252 | 316.9 | +7.01pp | -10.30pp (±4.30, 22c) |
| cooc_bin: 10+ users | 1,544 | 54.15% | 0.2761 | -5.327 | 638.6 | +24.96pp | +44.44pp (±3.59, 33c) |
| trans_bin: p_trans <= .02 | 55,325 | 4.38% | 0.0165 | -9.290 | 96.6 | -18.65pp | -35.62pp (±4.40, 16c) |
| trans_bin: .02-.06 | 1,278 | 40.14% | 0.2048 | -6.065 | 373.8 | +20.40pp | +30.11pp (±5.48, 10c) |
| trans_bin: .06-.15 | 626 | 33.87% | 0.1890 | -6.437 | 227.5 | +18.40pp | +21.18pp (±21.51, 1c) |
| trans_bin: >.15 | 210 | 16.19% | 0.0751 | -7.531 | 81.1 | +5.21pp | n/a |
| tri_bin: no triangle | 340 | 2.65% | 0.0108 | -8.925 | 32.6 | -1.80pp | n/a |
| tri_bin: one triangle | 270 | 6.30% | 0.0261 | -8.018 | 44.4 | +0.19pp | n/a |
| tri_bin: 2+ triangles | 5,111 | 30.54% | 0.1408 | -6.274 | 372.7 | +11.14pp | +16.24pp (±2.38, 57c) |
| same_community_test: different community | 28,081 | 2.53% | 0.0102 | -9.908 | 93.1 | -3.35pp | -2.18pp (±0.51, 305c) |
| same_community_test: same community | 29,358 | 8.43% | 0.0348 | -8.484 | 114.7 | +3.35pp | +2.18pp (±0.51, 305c) |
| seen_edge: unseen TRAIN edge | 51,718 | 3.09% | 0.0112 | -9.475 | 78.4 | -9.96pp | -16.18pp (±2.38, 57c) |
| seen_edge: seen TRAIN edge | 5,721 | 27.74% | 0.1277 | -6.514 | 337.0 | +9.96pp | +16.18pp (±2.38, 57c) |

### Graphlets (seen TRAIN edges only, compared with the other seen edges)

| structure | n | recall@10 | MRR@10 | mean log p | adj. d recall | matched d recall |
|---|---:|---:|---:|---:|---:|---:|
| seen-edge baseline | 5,721 | 27.74% | 0.1277 | -6.514 | n/a | n/a |
| K5 | 1,387 | 44.34% | 0.2033 | -5.535 | -1.90pp | -6.18pp (±6.19, 14c) |
| K4_tail | 1,180 | 47.37% | 0.2262 | -5.477 | +0.77pp | +4.89pp (±5.21, 21c) |
| K2_3 | 460 | 46.74% | 0.2299 | -5.747 | +5.99pp | +7.93pp (±7.57, 9c) |
| P5_centre | 45 | 24.44% | 0.1395 | -7.128 | +5.69pp | n/a |
| C5 | 159 | 38.36% | 0.1822 | -6.112 | +3.98pp | -15.00pp (±23.95, 1c) |
| P5_hops | 45 | 24.44% | 0.1395 | -7.128 | +5.69pp | n/a |
| T5 | 34 | 20.59% | 0.1179 | -7.580 | -0.32pp | n/a |

### Role pairs on seen TRAIN edges

| role pair | n | recall@10 | MRR@10 | mean log p | adj. d recall | matched d recall |
|---|---:|---:|---:|---:|---:|---:|
| role0-role0 | 3,420 | 35.15% | 0.1644 | -5.936 | -1.87pp | +0.00pp (±30.98, 1c) |
| role0-role5 | 1,009 | 19.62% | 0.0904 | -7.285 | -0.43pp | n/a |
| role0-role3 | 393 | 21.88% | 0.0924 | -6.782 | +6.26pp | n/a |
| role3-role5 | 231 | 9.52% | 0.0458 | -7.783 | +1.04pp | n/a |
| role5-role5 | 190 | 4.21% | 0.0165 | -8.232 | -5.37pp | n/a |
| role3-role3 | 164 | 21.34% | 0.0828 | -6.631 | +12.00pp | n/a |
| role0-role4 | 77 | 22.08% | 0.0970 | -7.212 | +1.84pp | n/a |
| role2-role2 | 45 | 6.67% | 0.0294 | -8.768 | -5.18pp | n/a |
| role2-role3 | 33 | 9.09% | 0.0223 | -7.491 | +4.73pp | n/a |
| role1-role3 | 32 | 15.62% | 0.0451 | -6.998 | +9.28pp | n/a |

### Cross-prefix but still predictable

| prefix | relation | n | recall@10 | MRR@10 | mean log p |
|---|---|---:|---:|---:|---:|
| all | semantic_Q5 | 13,008 | 10.30% | 0.0444 | -8.332 |
| all | cooc_4plus | 3,142 | 40.01% | 0.1928 | -5.798 |
| all | trans_gt_.15 | 210 | 16.19% | 0.0751 | -7.531 |
| all | seen_edge | 5,721 | 27.74% | 0.1277 | -6.514 |
| same_L1 | semantic_Q5 | 4,097 | 14.74% | 0.0612 | -7.609 |
| same_L1 | cooc_4plus | 618 | 45.95% | 0.2232 | -5.625 |
| same_L1 | trans_gt_.15 | 85 | 24.71% | 0.1257 | -6.735 |
| same_L1 | seen_edge | 1,119 | 33.42% | 0.1516 | -6.211 |
| different_L1 | semantic_Q5 | 8,911 | 8.26% | 0.0367 | -8.664 |
| different_L1 | cooc_4plus | 2,524 | 38.55% | 0.1853 | -5.840 |
| different_L1 | trans_gt_.15 | 125 | 10.40% | 0.0407 | -8.073 |
| different_L1 | seen_edge | 4,602 | 26.36% | 0.1218 | -6.588 |

### Is the L1 prefix worth anything inside a relation class?

| relation | n same | recall same | n diff | recall diff | raw gap | matched gap | adj. gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| all rows | 5,106 | 13.89% | 52,333 | 4.73% | +9.15pp | +0.89pp (±1.66, 66c) | +5.13pp |
| semantic_Q5 | 4,097 | 14.74% | 8,911 | 8.26% | +6.48pp | +0.91pp (±1.69, 65c) | +4.72pp |
| cooc_4plus | 618 | 45.95% | 2,524 | 38.55% | +7.40pp | -5.44pp (±7.15, 8c) | +6.60pp |
| trans_gt_.15 | 85 | 24.71% | 125 | 10.40% | +14.31pp | n/a | +9.88pp |
| seen_edge | 1,119 | 33.42% | 4,602 | 26.36% | +7.06pp | -7.62pp (±6.91, 9c) | +7.52pp |

