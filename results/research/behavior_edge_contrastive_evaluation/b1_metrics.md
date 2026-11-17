# Experiment 0: quantization loss localization

Offline reconstruction and kNN only. No training, no Stage3.

- items: **24587**; TRAIN behaviour edges: **315391**; K main: **10**
- x: sentence-T5 768-d, cosine; z/q: 32-d, Euclidean; x_hat: decoder(q), cosine
- q source: `exported_sid_codebooks`; tokens_match_export: `False`

## Distortion

| measure | value |
|---|---:|
| z→q MSE | 0.00016736 |
| x→x_hat MSE | 0.00007451 |
| x→x_hat cosine mean | 0.923374 |

## Duplicates and KNN ties

| stage | unique rows | duplicate rows | duplicate share | tie@K=10 |
|---|---:|---:|---:|---:|
| x | 24565 | 22 | 0.0009 | 0.0019 |
| z | 24565 | 22 | 0.0009 | 0.0013 |
| q | 16171 | 8416 | 0.3423 | 0.4475 |
| xhat | 16171 | 8416 | 0.3423 | 0.3866 |

### Duplicate-vector behaviour edges

| stage | duplicate groups | items | max group | same-vector edges | hit rate | both-in-duplicate-group edges | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| x | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| z | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| q | 3417 | 11833 | 68 | 1237 | 0.7874 | 80577 | 0.25548 |
| xhat | 3417 | 11833 | 68 | 1237 | 0.8042 | 80577 | 0.25548 |

## Overall behaviour-edge recovery

| stage | kNN edges | behaviour hits | recovery |
|---|---:|---:|---:|
| x | 198415 | 6517.0 | 0.020663 ± 0.000000 |
| z | 200287 | 5274.2 | 0.016723 ± 0.000001 |
| q | 183635 | 3028.2 | 0.009602 ± 0.000040 |
| xhat | 179300 | 3102.2 | 0.009836 ± 0.000061 |

## Conditional on the edges x already recovers

x behaviour edges: **6517** (precision of x kNN graph: **0.0328**)

| path | retained | rate |
|---|---:|---:|
| x → z | 4135.0 | 0.634494 ± 0.000000 |
| x → q (independent) | 1693.8 | 0.259897 ± 0.001427 |
| x → x_hat (independent) | 1762.5 | 0.270447 ± 0.001881 |
| x → z → q | 1452.0 | 0.222802 ± 0.001584 |
| x → z → q → x_hat | 1227.8 | 0.188392 ± 0.001295 |

- z → q conditional on x→z: **0.351149 ± 0.002496**
- q → x_hat conditional on x→z→q: **0.845579 ± 0.005694**

## K sensitivity (one tie-break seed for z/q/x_hat)

| K | x overall | z overall | q overall | x_hat overall | x→q | x→z→q |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.01376 | 0.01061 | 0.00570 | 0.00592 | 0.17370 | 0.12968 |
| 10 | 0.02066 | 0.01672 | 0.00963 | 0.00988 | 0.26055 | 0.22280 |
| 15 | 0.02609 | 0.02131 | 0.01287 | 0.01345 | 0.31870 | 0.28884 |
| 20 | 0.03067 | 0.02557 | 0.01607 | 0.01651 | 0.37318 | 0.34684 |

## Metric sensitivity at K=10

| metric | x overall | z overall | q overall | x_hat overall | q conditional |
|---|---:|---:|---:|---:|---:|
| cosine | 0.02066 | 0.01671 | 0.00965 | 0.00988 | 0.26300 |
| euclidean | 0.02053 | 0.01672 | 0.00963 | 0.00980 | 0.26055 |

## Graphlet-type recovery (K=10)

| type | n | x | z | q | x_hat | q conditional-on-x |
|---|---:|---:|---:|---:|---:|---:|
| triangle | 30437 | 0.06078 | 0.05083 | 0.02753 | 0.02803 | 0.28432 |
| star | 35139 | 0.01033 | 0.00820 | 0.00506 | 0.00512 | 0.21488 |
| chain | 13420 | 0.00745 | 0.00566 | 0.00386 | 0.00358 | 0.18000 |
| cycle | 24935 | 0.02254 | 0.01861 | 0.01150 | 0.01134 | 0.32206 |
| mixed | 42539 | 0.01638 | 0.01307 | 0.00835 | 0.00833 | 0.29125 |
| low | 168921 | 0.01743 | 0.01387 | 0.00781 | 0.00819 | 0.23497 |
