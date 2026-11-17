# Experiment 0: quantization loss localization

Offline reconstruction and kNN only. No training, no Stage3.

- items: **24587**; TRAIN behaviour edges: **315391**; K main: **10**
- x: sentence-T5 768-d, cosine; z/q: 32-d, Euclidean; x_hat: decoder(q), cosine
- q source: `exported_sid_codebooks`; tokens_match_export: `False`

## Distortion

| measure | value |
|---|---:|
| z→q MSE | 0.00014979 |
| x→x_hat MSE | 0.00006633 |
| x→x_hat cosine mean | 0.930091 |

## Duplicates and KNN ties

| stage | unique rows | duplicate rows | duplicate share | tie@K=10 |
|---|---:|---:|---:|---:|
| x | 24565 | 22 | 0.0009 | 0.0019 |
| z | 24565 | 22 | 0.0009 | 0.0011 |
| q | 20425 | 4162 | 0.1693 | 0.2299 |
| xhat | 20425 | 4162 | 0.1693 | 0.1974 |

### Duplicate-vector behaviour edges

| stage | duplicate groups | items | max group | same-vector edges | hit rate | both-in-duplicate-group edges | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| x | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| z | 22 | 44 | 2 | 1 | 1.0000 | 2 | 0.00001 |
| q | 2361 | 6523 | 24 | 525 | 0.9645 | 25821 | 0.08187 |
| xhat | 2361 | 6523 | 24 | 525 | 0.9643 | 25821 | 0.08187 |

## Overall behaviour-edge recovery

| stage | kNN edges | behaviour hits | recovery |
|---|---:|---:|---:|
| x | 198415 | 6517.0 | 0.020663 ± 0.000000 |
| z | 199095 | 5397.4 | 0.017113 ± 0.000002 |
| q | 185846 | 3557.2 | 0.011279 ± 0.000017 |
| xhat | 181932 | 3520.0 | 0.011161 ± 0.000029 |

## Conditional on the edges x already recovers

x behaviour edges: **6517** (precision of x kNN graph: **0.0328**)

| path | retained | rate |
|---|---:|---:|
| x → z | 4280.0 | 0.656744 ± 0.000000 |
| x → q (independent) | 2129.1 | 0.326703 ± 0.000742 |
| x → x_hat (independent) | 2154.2 | 0.330559 ± 0.000739 |
| x → z → q | 1885.9 | 0.289378 ± 0.000635 |
| x → z → q → x_hat | 1654.8 | 0.253913 ± 0.000789 |

- z → q conditional on x→z: **0.440625 ± 0.000966**
- q → x_hat conditional on x→z→q: **0.877443 ± 0.001570**

## K sensitivity (one tie-break seed for z/q/x_hat)

| K | x overall | z overall | q overall | x_hat overall | x→q | x→z→q |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.01376 | 0.01090 | 0.00657 | 0.00656 | 0.22035 | 0.17282 |
| 10 | 0.02066 | 0.01711 | 0.01127 | 0.01111 | 0.32745 | 0.28938 |
| 15 | 0.02609 | 0.02201 | 0.01539 | 0.01516 | 0.40003 | 0.36969 |
| 20 | 0.03067 | 0.02634 | 0.01889 | 0.01874 | 0.45696 | 0.43036 |

## Metric sensitivity at K=10

| metric | x overall | z overall | q overall | x_hat overall | q conditional |
|---|---:|---:|---:|---:|---:|
| cosine | 0.02066 | 0.01713 | 0.01137 | 0.01111 | 0.32929 |
| euclidean | 0.02053 | 0.01711 | 0.01127 | 0.01108 | 0.32745 |

## Graphlet-type recovery (K=10)

| type | n | x | z | q | x_hat | q conditional-on-x |
|---|---:|---:|---:|---:|---:|---:|
| triangle | 30437 | 0.06078 | 0.05131 | 0.03253 | 0.03205 | 0.35676 |
| star | 35139 | 0.01033 | 0.00891 | 0.00542 | 0.00520 | 0.26997 |
| chain | 13420 | 0.00745 | 0.00581 | 0.00328 | 0.00358 | 0.25000 |
| cycle | 24935 | 0.02254 | 0.01853 | 0.01193 | 0.01193 | 0.33096 |
| mixed | 42539 | 0.01638 | 0.01328 | 0.00920 | 0.00903 | 0.34433 |
| low | 168921 | 0.01743 | 0.01431 | 0.00973 | 0.00966 | 0.31409 |
