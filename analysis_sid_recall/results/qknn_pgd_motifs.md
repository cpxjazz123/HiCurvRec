# PGD 4-node edge motifs vs q-kNN recovery and recall

PGD micro output: triangle, 2-star, 4-clique, 4-chordal-cycle, 4-tailed-triangle, 4-cycle, 3-star, 4-path. This is **4-node PGD**, not 68-d Edge-ORCA and not K1,4/P5/T5.

Behaviour edges **315391**; arms **48**.

## Motif-high classes (degree-normalised top quartile)

| motif | edges | share | motif median | dmax median |
|---|---:|---:|---:|---:|
| triangle | 67492 | 0.2140 | 7.0 | 140.0 |
| 2-star | 78934 | 0.2503 | 177.0 | 127.0 |
| 4-clique | 53166 | 0.1686 | 18.0 | 197.0 |
| 4-chordal-cycle | 52865 | 0.1676 | 26.0 | 200.0 |
| 4-tailed-triangle | 0 | 0.0000 | nan | nan |
| 4-cycle | 79539 | 0.2522 | 235.0 | 113.0 |
| 3-star | 78796 | 0.2498 | 9610.0 | 134.0 |
| 4-path | 78840 | 0.2500 | 6770.0 | 116.0 |

## Recall relations

Partial controls overall recovery, q-kNN edge count, L3 collision. CV is repeated 5-fold Ridge; excess uses the same (dmax,dmin,semantic) matching cell.

| motif | mean rate | mean excess | direct rho | direct r | partial rho | ΔR² rate | p(rate) | CV +rate | CV +excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| triangle | 0.02167 | +0.01267 | +0.6365 | +0.6800 | -0.2390 | +0.1844 | 0.0000 | +0.0828 | +0.1955 |
| 2-star | 0.00953 | -0.00370 | +0.5555 | +0.6535 | -0.1518 | +0.1366 | 0.0000 | +0.0994 | -0.0390 |
| 4-clique | 0.01560 | +0.00814 | +0.5159 | +0.6072 | -0.2149 | +0.1738 | 0.0000 | +0.2097 | +0.2973 |
| 4-chordal-cycle | 0.01952 | +0.01194 | +0.5351 | +0.6448 | -0.2225 | +0.1794 | 0.0000 | +0.1572 | +0.2402 |
| 4-cycle | 0.01284 | +0.00133 | +0.5609 | +0.6324 | -0.1105 | +0.1508 | 0.0000 | +0.1396 | +0.2823 |
| 3-star | 0.00909 | -0.00306 | +0.6527 | +0.6961 | -0.0141 | +0.1077 | 0.0003 | +0.0231 | -0.0255 |
| 4-path | 0.01018 | -0.00556 | +0.4732 | +0.5690 | -0.2450 | +0.1552 | 0.0000 | +0.2190 | +0.0550 |
