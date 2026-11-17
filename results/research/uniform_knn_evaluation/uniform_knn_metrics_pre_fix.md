# Experiment 0: quantization loss localization

Offline reconstruction and kNN only. No training, no Stage3.

- items: **24587**; TRAIN behaviour edges: **315391**; K main: **10**
- x: sentence-T5 768-d, cosine; z/q: 32-d, Euclidean; x_hat: decoder(q), cosine
- q source: `exported_sid_codebooks`; tokens_match_export: `False`

## Distortion

| measure | value |
|---|---:|
| z→q MSE | 0.00008701 |
| x→x_hat MSE | 0.00008800 |
| x→x_hat cosine mean | 0.910979 |

## Duplicates and KNN ties

| stage | unique rows | duplicate rows | duplicate share | tie@K=10 |
|---|---:|---:|---:|---:|
| x | 24565 | 22 | 0.0009 | 0.0019 |
| z | 24565 | 22 | 0.0009 | 0.0009 |
| q | 21172 | 3415 | 0.1389 | 0.1402 |
| xhat | 21172 | 3415 | 0.1389 | 0.1539 |

### Duplicate-vector behaviour edges

| stage | duplicate groups | items | max group | same-vector edges | hit rate | both-in-duplicate-group edges | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| x | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| z | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| q | 2615 | 6030 | 18 | 150 | 0.9992 | 18887 | 0.05988 |
| xhat | 2615 | 6030 | 18 | 150 | 0.9983 | 18887 | 0.05988 |

## Overall behaviour-edge recovery

| stage | kNN edges | behaviour hits | recovery |
|---|---:|---:|---:|
| x | 198415 | 6517.0 | 0.020663 ± 0.000000 |
| z | 176636 | 1664.1 | 0.005276 ± 0.000001 |
| q | 167831 | 1445.2 | 0.004582 ± 0.000008 |
| xhat | 174195 | 2205.2 | 0.006992 ± 0.000013 |

## Conditional on the edges x already recovers

x behaviour edges: **6517** (precision of x kNN graph: **0.0328**)

| path | retained | rate |
|---|---:|---:|
| x → z | 626.0 | 0.096056 ± 0.000000 |
| x → q (independent) | 542.8 | 0.083282 ± 0.000184 |
| x → x_hat (independent) | 1034.5 | 0.158739 ± 0.000664 |
| x → z → q | 515.0 | 0.079024 ± 0.000188 |
| x → z → q → x_hat | 334.6 | 0.051346 ± 0.000216 |

- z → q conditional on x→z: **0.822684 ± 0.001956**
- q → x_hat conditional on x→z→q: **0.649758 ± 0.002488**

## K sensitivity (one tie-break seed for z/q/x_hat)

| K | x overall | z overall | q overall | x_hat overall | x→q | x→z→q |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.01376 | 0.00366 | 0.00305 | 0.00403 | 0.06583 | 0.06149 |
| 10 | 0.02066 | 0.00528 | 0.00458 | 0.00698 | 0.08347 | 0.07902 |
| 15 | 0.02609 | 0.00654 | 0.00580 | 0.00970 | 0.09207 | 0.08917 |
| 20 | 0.03067 | 0.00751 | 0.00671 | 0.01240 | 0.09759 | 0.09508 |

## Metric sensitivity at K=10

| metric | x overall | z overall | q overall | x_hat overall | q conditional |
|---|---:|---:|---:|---:|---:|
| cosine | 0.02066 | 0.00546 | 0.00472 | 0.00698 | 0.08746 |
| euclidean | 0.02053 | 0.00528 | 0.00458 | 0.00695 | 0.08347 |

## Graphlet-type recovery (K=10)

| type | n | x | z | q | x_hat | q conditional-on-x |
|---|---:|---:|---:|---:|---:|---:|
| triangle | 30437 | 0.06078 | 0.01384 | 0.01249 | 0.01883 | 0.09027 |
| star | 35139 | 0.01033 | 0.00248 | 0.00192 | 0.00381 | 0.04132 |
| chain | 13420 | 0.00745 | 0.00298 | 0.00253 | 0.00335 | 0.11000 |
| cycle | 24935 | 0.02254 | 0.00529 | 0.00450 | 0.00711 | 0.08719 |
| mixed | 42539 | 0.01638 | 0.00411 | 0.00347 | 0.00539 | 0.08034 |
| low | 168921 | 0.01743 | 0.00479 | 0.00417 | 0.00620 | 0.08353 |
