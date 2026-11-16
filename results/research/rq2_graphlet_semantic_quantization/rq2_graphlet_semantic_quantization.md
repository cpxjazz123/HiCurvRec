# RQ2: Graphlet × semantic relation × quantization edge recovery

Behaviour graph: **315,391** real TRAIN edges; semantic labels cover **300864** edges (**95.39%**).
Primary Graphlet type: highest top-quartile degree-binned rank among triangle, exact K1,3 star,
PGD 4-path chain, and PGD 4-cycle; `mixed` = near-tie, `low` = below top quartile in all four.
Top-quartile memberships remain multi-label; the primary type is an explicit exclusive summary.

Text kNN: sentence-T5 cosine. q kNN: pure-Euclidean TIGER q, Euclidean.
Recovery = behaviour edge present in the symmetrised KNN union.

## K=10

| graphlet | relation | n | text | q | delta (q-text) |
|---|---|---:|---:|---:|---:|
| triangle | all | 29221 | 0.0615 | 0.0330 | -0.0285 |
| triangle | same_fine | 4547 | 0.2641 | 0.1427 | -0.1214 |
| triangle | same_coarse_diff_fine | 13759 | 0.0335 | 0.0175 | -0.0160 |
| triangle | cross_coarse | 10915 | 0.0124 | 0.0069 | -0.0055 |
| star | all | 33594 | 0.0105 | 0.0056 | -0.0049 |
| star | same_fine | 1569 | 0.1517 | 0.0790 | -0.0727 |
| star | same_coarse_diff_fine | 13489 | 0.0064 | 0.0035 | -0.0030 |
| star | cross_coarse | 18536 | 0.0015 | 0.0009 | -0.0006 |
| chain | all | 12918 | 0.0077 | 0.0034 | -0.0043 |
| chain | same_fine | 437 | 0.1442 | 0.0641 | -0.0801 |
| chain | same_coarse_diff_fine | 4328 | 0.0053 | 0.0023 | -0.0030 |
| chain | cross_coarse | 8153 | 0.0016 | 0.0007 | -0.0009 |
| cycle | all | 23851 | 0.0228 | 0.0120 | -0.0107 |
| cycle | same_fine | 1746 | 0.1987 | 0.1082 | -0.0905 |
| cycle | same_coarse_diff_fine | 11390 | 0.0126 | 0.0067 | -0.0060 |
| cycle | cross_coarse | 10715 | 0.0049 | 0.0021 | -0.0028 |
| mixed | all | 41197 | 0.0163 | 0.0093 | -0.0070 |
| mixed | same_fine | 2580 | 0.1818 | 0.1019 | -0.0798 |
| mixed | same_coarse_diff_fine | 18181 | 0.0091 | 0.0052 | -0.0039 |
| mixed | cross_coarse | 20436 | 0.0019 | 0.0013 | -0.0006 |
| low | all | 160083 | 0.0176 | 0.0099 | -0.0077 |
| low | same_fine | 9996 | 0.1799 | 0.1037 | -0.0761 |
| low | same_coarse_diff_fine | 62678 | 0.0117 | 0.0066 | -0.0051 |
| low | cross_coarse | 87409 | 0.0033 | 0.0015 | -0.0018 |
| all_edges | all | 300864 | 0.0209 | 0.0115 | -0.0094 |
| all_edges | same_fine | 20875 | 0.1972 | 0.1097 | -0.0875 |
| all_edges | same_coarse_diff_fine | 123825 | 0.0131 | 0.0071 | -0.0059 |
| all_edges | cross_coarse | 156164 | 0.0036 | 0.0018 | -0.0018 |

## K stability (overall and graphlet all-edge subsets)

| K | graph | n | text | q | delta |
|---|---|---:|---:|---:|---:|
| 5 | all_edges | 300864 | 0.0139 | 0.0068 | -0.0071 |
| 5 | triangle | 29221 | 0.0419 | 0.0204 | -0.0215 |
| 5 | star | 33594 | 0.0062 | 0.0031 | -0.0032 |
| 5 | chain | 12918 | 0.0050 | 0.0025 | -0.0026 |
| 5 | cycle | 23851 | 0.0156 | 0.0071 | -0.0085 |
| 5 | mixed | 41197 | 0.0110 | 0.0056 | -0.0054 |
| 5 | low | 160083 | 0.0117 | 0.0057 | -0.0059 |
| 10 | all_edges | 300864 | 0.0209 | 0.0115 | -0.0094 |
| 10 | triangle | 29221 | 0.0615 | 0.0330 | -0.0285 |
| 10 | star | 33594 | 0.0105 | 0.0056 | -0.0049 |
| 10 | chain | 12918 | 0.0077 | 0.0034 | -0.0043 |
| 10 | cycle | 23851 | 0.0228 | 0.0120 | -0.0107 |
| 10 | mixed | 41197 | 0.0163 | 0.0093 | -0.0070 |
| 10 | low | 160083 | 0.0176 | 0.0099 | -0.0077 |
| 15 | all_edges | 300864 | 0.0264 | 0.0156 | -0.0108 |
| 15 | triangle | 29221 | 0.0757 | 0.0447 | -0.0310 |
| 15 | star | 33594 | 0.0141 | 0.0081 | -0.0059 |
| 15 | chain | 12918 | 0.0105 | 0.0063 | -0.0041 |
| 15 | cycle | 23851 | 0.0284 | 0.0161 | -0.0123 |
| 15 | mixed | 41197 | 0.0205 | 0.0125 | -0.0079 |
| 15 | low | 160083 | 0.0225 | 0.0132 | -0.0093 |
| 20 | all_edges | 300864 | 0.0310 | 0.0191 | -0.0119 |
| 20 | triangle | 29221 | 0.0875 | 0.0543 | -0.0332 |
| 20 | star | 33594 | 0.0171 | 0.0104 | -0.0068 |
| 20 | chain | 12918 | 0.0130 | 0.0074 | -0.0056 |
| 20 | cycle | 23851 | 0.0326 | 0.0196 | -0.0130 |
| 20 | mixed | 41197 | 0.0245 | 0.0156 | -0.0089 |
| 20 | low | 160083 | 0.0265 | 0.0163 | -0.0102 |

## Cross-coarse behaviour sample

Cross-coarse behaviour edges: **156164**; 30 sampled with titles and recovery flags in `cross_coarse_behaviour_sample.csv`.

This sample is for the next semantic audit; cross-coarse is not automatically complementarity.
