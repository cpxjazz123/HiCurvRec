# Experiment 0: quantization loss localization

Offline reconstruction and kNN only. No training, no Stage3.

- items: **24587**; TRAIN behaviour edges: **315391**; K main: **10**
- x: sentence-T5 768-d, cosine; z/q: 32-d, Euclidean; x_hat: decoder(q), cosine
- q source: `exported_sid_codebooks`; tokens_match_export: `True`

## Distortion

| measure | value |
|---|---:|
| z→q MSE | 0.00014136 |
| x→x_hat MSE | 0.00006584 |
| x→x_hat cosine mean | 0.930249 |

## Duplicates and KNN ties

| stage | unique rows | duplicate rows | duplicate share | tie@K=10 |
|---|---:|---:|---:|---:|
| x | 24565 | 22 | 0.0009 | 0.0019 |
| z | 24565 | 22 | 0.0009 | 0.0009 |
| q | 20084 | 4503 | 0.1831 | 0.2463 |
| xhat | 20084 | 4503 | 0.1831 | 0.2140 |

### Duplicate-vector behaviour edges

| stage | duplicate groups | items | max group | same-vector edges | hit rate | both-in-duplicate-group edges | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| x | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| z | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| q | 2598 | 7101 | 30 | 507 | 0.9559 | 28067 | 0.08899 |
| xhat | 2598 | 7101 | 30 | 507 | 0.9532 | 28067 | 0.08899 |

## Overall behaviour-edge recovery

| stage | kNN edges | behaviour hits | recovery |
|---|---:|---:|---:|
| x | 198415 | 6517.0 | 0.020663 ± 0.000000 |
| z | 198196 | 5358.0 | 0.016988 ± 0.000000 |
| q | 186782 | 3554.8 | 0.011271 ± 0.000026 |
| xhat | 183126 | 3520.5 | 0.011162 ± 0.000020 |

## Conditional on the edges x already recovers

x behaviour edges: **6517** (precision of x kNN graph: **0.0328**)

| path | retained | rate |
|---|---:|---:|
| x → z | 4320.0 | 0.662882 ± 0.000000 |
| x → q (independent) | 2099.1 | 0.322100 ± 0.000906 |
| x → x_hat (independent) | 2134.0 | 0.327451 ± 0.000861 |
| x → z → q | 1855.9 | 0.284774 ± 0.000765 |
| x → z → q → x_hat | 1625.8 | 0.249463 ± 0.000641 |

- z → q conditional on x→z: **0.429601 ± 0.001154**
- q → x_hat conditional on x→z→q: **0.876008 ± 0.003098**

## K sensitivity (one tie-break seed for z/q/x_hat)

| K | x overall | z overall | q overall | x_hat overall | x→q | x→z→q |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.01376 | 0.01108 | 0.00651 | 0.00667 | 0.21252 | 0.16896 |
| 10 | 0.02066 | 0.01699 | 0.01124 | 0.01117 | 0.32177 | 0.28477 |
| 15 | 0.02609 | 0.02206 | 0.01518 | 0.01520 | 0.39604 | 0.36531 |
| 20 | 0.03067 | 0.02628 | 0.01908 | 0.01881 | 0.45742 | 0.43212 |

## Metric sensitivity at K=10

| metric | x overall | z overall | q overall | x_hat overall | q conditional |
|---|---:|---:|---:|---:|---:|
| cosine | 0.02066 | 0.01694 | 0.01136 | 0.01117 | 0.32377 |
| euclidean | 0.02053 | 0.01699 | 0.01124 | 0.01122 | 0.32177 |

## Graphlet-type recovery (K=10)

| type | n | x | z | q | x_hat | q conditional-on-x |
|---|---:|---:|---:|---:|---:|---:|
| triangle | 30437 | 0.06078 | 0.05030 | 0.03343 | 0.03308 | 0.35243 |
| star | 35139 | 0.01033 | 0.00808 | 0.00580 | 0.00551 | 0.24242 |
| chain | 13420 | 0.00745 | 0.00618 | 0.00417 | 0.00326 | 0.25000 |
| cycle | 24935 | 0.02254 | 0.01817 | 0.01141 | 0.01167 | 0.34342 |
| mixed | 42539 | 0.01638 | 0.01323 | 0.00894 | 0.00869 | 0.32999 |
| low | 168921 | 0.01743 | 0.01447 | 0.00955 | 0.00957 | 0.30866 |
