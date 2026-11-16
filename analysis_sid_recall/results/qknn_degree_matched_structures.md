# Degree-matched AF3/local-topology edge classes (no Louvain)

48 standard VQ arms; q rebuilt from codebooks + SID; global k=10 q-kNN. No community label is used. Classes use degree-normalised AF3 and local topology. Each class is compared with non-class edges in the same (dmax,dmin) quintile cell.

Behaviour edges **315391**; AF3 sign: positive **0**, zero **1**, negative **315390**.

## Classes

| class | edges | share | AF3 median | dmax median | dmin median | asym median | common median | betweenness median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 22781 | 0.0722 | -328.0 | 314.0 | 32.0 | 0.820 | 4.0 | 1.365e-08 |
| bridge | 3109 | 0.0099 | -152.0 | 134.0 | 17.0 | 0.814 | 0.0 | 5.735e-08 |
| star | 755 | 0.0024 | -316.0 | 311.0 | 8.0 | 0.983 | 0.0 | 1.052e-07 |
| chain | 18805 | 0.0596 | -47.0 | 29.0 | 22.0 | 0.224 | 0.0 | 3.792e-09 |

## Recall relations

Partial controls overall recovery, q-kNN edge count and L3 collision. `+rate`, `+excess`, `+lift` are CV R² differences over the base controls.

| class | mean rate | direct rho | direct r | partial rho | partial r | ΔR² rate | p(rate) | CV +rate | CV +excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.01676 | +0.6599 | +0.7477 | +0.0426 | +0.2017 | +0.0166 | 0.1839 | +0.0011 | +0.0175 |
| bridge | 0.00424 | +0.6326 | +0.7176 | +0.3376 | +0.3716 | +0.0564 | 0.0120 | +0.1403 | -0.0051 |
| star | 0.00149 | +0.1529 | +0.2545 | +0.0719 | +0.1792 | +0.0131 | 0.2389 | -0.0189 | -0.0173 |
| chain | 0.01019 | +0.6857 | +0.7549 | +0.2460 | +0.1366 | +0.0076 | 0.3708 | +0.0210 | -0.0226 |
