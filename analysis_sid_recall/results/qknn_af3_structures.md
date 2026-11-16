# AF3 / local-topology edge classes vs q-kNN recovery and recall

AF3 = 4 - du - dv + 3*T, on the fixed TRAIN behaviour graph. Dense/bridge/star/chain thresholds are degree/quantile based and may overlap. q is rebuilt from standard 3x(256,32) VQ codebooks; q-kNN is global k=10.

Behaviour edges: **315391**; arms: **48**.

## AF3 distribution by dmax quintile

| bin | dmax range | edges | AF3 mean | AF3 median | p05 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 2–33 | 63833 | -28.518 | -28.000 | -48.000 | -12.000 |
| Q2 | 34–72 | 62878 | -67.928 | -66.000 | -103.000 | -42.000 |
| Q3 | 73–145 | 62668 | -132.430 | -128.000 | -197.000 | -86.000 |
| Q4 | 146–312 | 63007 | -252.877 | -244.000 | -373.000 | -164.000 |
| Q5 | 313–2220 | 63005 | -770.415 | -627.000 | -1679.000 | -360.000 |

## Class definitions and sizes

| class | edges | share | AF3 mean | AF3 median | dmax median | common median | edge-betweenness median | inside share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 22781 | 0.0722 | -273.878 | -328.000 | 314.0 | 4.0 | 1.365e-08 | 0.6971 |
| bridge | 1990 | 0.0063 | -355.391 | -160.000 | 139.0 | 0.0 | 6.115e-08 | 0.0000 |
| star | 755 | 0.0024 | -744.558 | -316.000 | 311.0 | 0.0 | 1.052e-07 | 0.3219 |
| chain | 18805 | 0.0596 | -59.072 | -47.000 | 29.0 | 0.0 | 3.792e-09 | 0.5552 |
| af3_pos | 0 | 0.0000 | NA | NA | NA | NA | NA | NA |
| af3_zero | 1 | 0.0000 | 0.000 | 0.000 | 2.0 | 0.0 | 9.425e-08 | 1.0000 |
| af3_neg | 315390 | 1.0000 | -250.052 | -128.000 | 103.0 | 1.0 | 1.035e-08 | 0.5979 |
| inside | 188588 | 0.5979 | -225.872 | -111.000 | 88.0 | 1.0 | 8.878e-09 | 1.0000 |
| cross | 126803 | 0.4021 | -286.011 | -156.000 | 125.0 | 1.0 | 1.323e-08 | 0.0000 |

Raw AF3 sign counts: positive **0**, zero **1**, negative **315390**. Because almost every edge is negative, raw AF3 sign is not a useful class; the classes above use AF3 rank within the dmax quintile plus local topology.

## Direct and incremental recall relations

Partial controls overall recovery, q-kNN edge count, L3 collision, and inside/cross recovery. ΔR² is in-sample; CV is repeated 5-fold Ridge(alpha=1).

| class | mean rate | direct Spearman | direct Pearson | partial Spearman | partial Pearson | ΔR² rate | p(rate) | CV ΔR² rate | CV ΔR² excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.01676 | +0.6599 | +0.7477 | -0.0637 | -0.0637 | +0.0015 | 0.6847 | -0.0055 | +0.0361 |
| bridge | 0.00179 | +0.0081 | +0.0120 | +0.1606 | +0.1606 | +0.0097 | 0.3037 | -0.0006 | -0.0105 |
| star | 0.00149 | +0.1529 | +0.2545 | +0.1667 | +0.1667 | +0.0104 | 0.2852 | -0.0101 | +0.0824 |
| chain | 0.01019 | +0.6857 | +0.7549 | +0.2337 | +0.2337 | +0.0205 | 0.1315 | +0.0345 | +0.0079 |
| inside | 0.01724 | +0.6767 | +0.7453 | +0.2880 | +0.2880 | +0.0339 | 0.0551 | +0.0022 | -0.0034 |
| cross | 0.00300 | +0.6327 | +0.6405 | -0.2880 | -0.2880 | +0.0339 | 0.0551 | -0.0596 | -0.0034 |
