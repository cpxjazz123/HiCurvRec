# Exact K1,3 edge recovery and Recall@10

K1,3 counts are exact per-edge induced star counts from `k13_exact.cpp`, not PGD's
broken `3-star` column. Primary class: top quartile of exact count within degree-max bins;
matched controls use degree-max + degree-min + semantic bins. `positive` uses every edge
with a nonzero exact count. CV is grouped by model series.

Behavior edges **315391**; arms **48**; K1,3 edges **315390**; top class **78838**.

## Class

| quantity | value |
|---|---:|
| K1,3 positive edges | 315390 |
| K1,3 positive share | 0.999997 |
| count median / max | 5443 / 2424941 |
| top-class count median | 9179 |
| top-class dmax median | 134 |

## Recovery and recall

| quantity | n | mean | direct rho (p) | partial rho (p) | CV ΔR² (p) | wins |
|---|---:|---:|---:|---:|---:|---:|
| K1,3 rate | 48 | 0.009164 | +0.6455 (7.29e-07) | +0.0036 (0.981) | +0.0149 (0.335) | 13/30 |
| K1,3 matched excess | 48 | -0.002987 | -0.6330 (1.38e-06) | -0.0313 (0.833) | -0.0295 (0.95) | 11/30 |

## Sensitivity: collision < 0.90 (44 arms)

| quantity | direct rho | partial rho (p) | CV ΔR² (p) |
|---|---:|---:|---:|
| K1,3 rate | +0.5855 | +0.0677 (0.663) | -0.0381 (1) |
| K1,3 matched excess | -0.5458 | -0.0309 (0.842) | +0.0032 (0.271) |

Coverage for the primary class: mean 0.9919, min 0.9919.
The all-positive subset is nearly the whole graph; its recovery is not a discriminative
class and is reported only for completeness.
