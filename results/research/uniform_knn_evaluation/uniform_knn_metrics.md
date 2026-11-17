# Experiment 0: quantization loss localization

Offline reconstruction and kNN only. No training, no Stage3.

- items: **24587**; TRAIN behaviour edges: **315391**; K main: **10**
- x: sentence-T5 768-d, cosine; z/q: 32-d, Euclidean; x_hat: decoder(q), cosine
- q source: `model_rq_argmin`; tokens_match_export: `True`

## Distortion

| measure | value |
|---|---:|
| z→q MSE | 0.00014332 |
| x→x_hat MSE | 0.00007337 |
| x→x_hat cosine mean | 0.923090 |

## Duplicates and KNN ties

| stage | unique rows | duplicate rows | duplicate share | tie@K=10 |
|---|---:|---:|---:|---:|
| x | 24565 | 22 | 0.0009 | 0.0019 |
| z | 24565 | 22 | 0.0009 | 0.0013 |
| q | 20659 | 3928 | 0.1598 | 0.4016 |
| xhat | 19415 | 5172 | 0.2104 | 0.3517 |

### Duplicate-vector behaviour edges

| stage | duplicate groups | items | max group | same-vector edges | hit rate | both-in-duplicate-group edges | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| x | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| z | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| q | 2258 | 6186 | 22 | 268 | 0.9655 | 21047 | 0.06673 |
| xhat | 2727 | 7899 | 23 | 346 | 0.9487 | 33048 | 0.10478 |

## Overall behaviour-edge recovery

| stage | kNN edges | behaviour hits | recovery |
|---|---:|---:|---:|
| x | 198415 | 6517.0 | 0.020663 ± 0.000000 |
| z | 196536 | 5094.0 | 0.016151 ± 0.000003 |
| q | 181794 | 2957.4 | 0.009377 ± 0.000029 |
| xhat | 177208 | 2989.5 | 0.009479 ± 0.000014 |

## Conditional on the edges x already recovers

x behaviour edges: **6517** (precision of x kNN graph: **0.0328**)

| path | retained | rate |
|---|---:|---:|
| x → z | 3961.5 | 0.607872 ± 0.000109 |
| x → q (independent) | 1628.4 | 0.249866 ± 0.001052 |
| x → x_hat (independent) | 1705.8 | 0.261739 ± 0.001267 |
| x → z → q | 1408.0 | 0.216050 ± 0.001018 |
| x → z → q → x_hat | 1203.1 | 0.184613 ± 0.001202 |

- z → q conditional on x→z: **0.355421 ± 0.001662**
- q → x_hat conditional on x→z→q: **0.854489 ± 0.003020**

## K sensitivity (one tie-break seed for z/q/x_hat)

| K | x overall | z overall | q overall | x_hat overall | x→q | x→z→q |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.01376 | 0.01047 | 0.00534 | 0.00553 | 0.16725 | 0.12767 |
| 10 | 0.02066 | 0.01615 | 0.00939 | 0.00948 | 0.24965 | 0.21605 |
| 15 | 0.02609 | 0.02086 | 0.01306 | 0.01330 | 0.31472 | 0.28289 |
| 20 | 0.03067 | 0.02493 | 0.01625 | 0.01657 | 0.35998 | 0.33478 |

## Metric sensitivity at K=10

| metric | x overall | z overall | q overall | x_hat overall | q conditional |
|---|---:|---:|---:|---:|---:|
| cosine | 0.02066 | 0.01609 | 0.00932 | 0.00948 | 0.24950 |
| euclidean | 0.02053 | 0.01615 | 0.00939 | 0.00946 | 0.24965 |

## Graphlet-type recovery (K=10)

| type | n | x | z | q | x_hat | q conditional-on-x |
|---|---:|---:|---:|---:|---:|---:|
| triangle | 30437 | 0.06078 | 0.04774 | 0.02602 | 0.02681 | 0.26324 |
| star | 35139 | 0.01033 | 0.00811 | 0.00482 | 0.00504 | 0.21763 |
| chain | 13420 | 0.00745 | 0.00596 | 0.00369 | 0.00382 | 0.20000 |
| cycle | 24935 | 0.02254 | 0.01704 | 0.01062 | 0.01071 | 0.25623 |
| mixed | 42539 | 0.01638 | 0.01289 | 0.00708 | 0.00746 | 0.24964 |
| low | 168921 | 0.01743 | 0.01363 | 0.00817 | 0.00805 | 0.24550 |
