# Robustness: fixed q-kNN budget, high-collision exclusion, larger graphlets, CV

Standard-VQ arms with L3 collision < **0.9**: **44**. All q graphs are k=20 candidates trimmed to **185115** edges, so q edge count is constant.

## Sample-size stability (300 vs 3000 strict instances)

| structure | old edges | new edges | rate Spearman | rate Pearson | rate MAE | old recall rho | new recall rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1_4 | 1200 | 11605 | +0.7547 | +0.9581 | 0.00300 | +0.5689 | +0.5958 |
| P5 | 1199 | 11749 | +0.5488 | +0.9676 | 0.00241 | +0.4666 | +0.5412 |
| T5 | 1195 | 11656 | +0.7470 | +0.9671 | 0.00276 | +0.5895 | +0.5876 |
| C5 | 1495 | 14561 | +0.5985 | +0.9606 | 0.00260 | +0.5004 | +0.4727 |

## Repeated 5-fold CV: incremental recall prediction

Base controls: overall recovery, L3 collision, inside/cross recovery (for graphlet structures). Ridge(alpha=1) after StandardScaler; 20 repeats of 5-fold. ΔR²/ΔMAE are with minus base.

| structure | base R² | +rate R² | ΔR² rate | ΔMAE rate | +excess R² | ΔR² excess | ΔMAE excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1_4 | 0.3119 | 0.3029 | -0.0090 | +0.00002 | 0.2798 | -0.0321 | +0.00008 |
| P5 | 0.3119 | 0.3080 | -0.0038 | +0.00001 | 0.2515 | -0.0603 | +0.00014 |
| T5 | 0.3119 | 0.3141 | +0.0022 | +0.00001 | 0.2729 | -0.0389 | +0.00012 |
| C5 | 0.3119 | 0.3035 | -0.0084 | +0.00001 | 0.2987 | -0.0132 | +0.00002 |
| triangle | 0.3119 | 0.3097 | -0.0022 | +0.00001 | 0.3090 | -0.0029 | +0.00001 |
| K4 | 0.3119 | 0.3062 | -0.0057 | +0.00001 | 0.3073 | -0.0046 | +0.00001 |
| inside | 0.3129 | 0.3119 | -0.0010 | +0.00001 | 0.3114 | -0.0015 | +0.00001 |
| cross | 0.2815 | 0.3119 | +0.0303 | -0.00014 | 0.2790 | -0.0025 | +0.00000 |
| all graphlets | 0.3119 | 0.2883 | -0.0236 | +0.00006 | 0.1542 | -0.1576 | +0.00038 |
