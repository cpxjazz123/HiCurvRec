# Audited PGD motifs vs q-kNN recovery and recall

PGD micro edge motifs; **3-star and 4-tailed-triangle are excluded** because PGD's
`local_4_tailed_tris` is never populated and its 3-star correction therefore over-counts.
A synthetic K3+pendant graph reproduces: tailed=0 on all edges, 3-star=1 on the pendant edge.
Reliable columns used: triangle, 4-clique, 4-chordal-cycle, 4-cycle, 4-path.

CPU kNN now explicitly removes each query's own index. PGD relabeling is inverted and
every behaviour edge must map exactly once. Matching coverage is reported per class.

Behaviour edges **315391**; arms **48**; grouped CV by model series.

## Classes

| motif | edges | share | count median | dmax median |
|---|---:|---:|---:|---:|
| triangle | 67492 | 0.2140 | 7.0 | 140.0 |
| 4-clique | 53166 | 0.1686 | 18.0 | 197.0 |
| 4-chordal-cycle | 52865 | 0.1676 | 26.0 | 200.0 |
| 4-cycle | 79539 | 0.2522 | 235.0 | 113.0 |
| 4-path | 78840 | 0.2500 | 6770.0 | 116.0 |

## Recall relations

Partial controls leave-target-out overall recovery, q-kNN edge count and L3 collision. CV is GroupShuffleSplit by model series, 30 splits, Ridge(alpha=1).

| motif | coverage | mean rate | mean excess | direct rho | direct r | partial rho | partial r | CV +rate | CV +excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| triangle | 1.0000 | 0.02167 | +0.01267 | +0.6365 | +0.6800 | -0.1539 | -0.6183 | +0.1346 | +0.1726 |
| 4-clique | 1.0000 | 0.01560 | +0.00814 | +0.5159 | +0.6072 | -0.1598 | -0.6087 | +0.1979 | +0.2304 |
| 4-chordal-cycle | 1.0000 | 0.01952 | +0.01194 | +0.5351 | +0.6448 | -0.1783 | -0.6171 | +0.1697 | +0.2004 |
| 4-cycle | 0.9987 | 0.01284 | +0.00133 | +0.5609 | +0.6324 | -0.0207 | -0.5577 | +0.1360 | +0.2199 |
| 4-path | 0.9919 | 0.01018 | -0.00556 | +0.4732 | +0.5690 | -0.1712 | -0.5668 | +0.1846 | +0.0417 |
