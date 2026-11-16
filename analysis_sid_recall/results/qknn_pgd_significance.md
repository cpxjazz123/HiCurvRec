# Paired significance and collision sensitivity

Grouped CV (GroupShuffleSplit by model series, 30 splits, Ridge alpha=1). `ΔR²` is feature minus base; p is one-sided Wilcoxon on the paired folds. Significance of a predictive gain is **not** evidence that higher retention raises recall: the partial correlation is reported separately.

## Main (48 arms)

| motif | n | partial rho (p) | partial r (p) | CV ΔR² rate (p) | wins | CV ΔR² excess (p) | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| triangle | 48 | -0.1539 (0.296) | -0.6183 (0.000) | +0.1346 (0.000) | 29/30 | +0.1726 (0.000) | 26/30 |
| 4-clique | 48 | -0.1598 (0.278) | -0.6087 (0.000) | +0.1979 (0.000) | 28/30 | +0.2304 (0.000) | 26/30 |
| 4-chordal-cycle | 48 | -0.1783 (0.225) | -0.6171 (0.000) | +0.1697 (0.000) | 29/30 | +0.2004 (0.000) | 27/30 |
| 4-cycle | 48 | -0.0207 (0.889) | -0.5577 (0.000) | +0.1360 (0.000) | 25/30 | +0.2199 (0.000) | 27/30 |
| 4-path | 48 | -0.1712 (0.245) | -0.5668 (0.000) | +0.1846 (0.000) | 26/30 | +0.0417 (0.214) | 19/30 |

## Sensitivity: collision < 0.90 (44 arms)

| motif | partial rho | CV ΔR² rate (p) | CV ΔR² excess (p) |
|---|---:|---:|---:|
| triangle | -0.1215 | -0.0185 (1.000) | -0.0209 (1.000) |
| 4-clique | -0.1520 | -0.0241 (1.000) | -0.0330 (1.000) |
| 4-chordal-cycle | -0.1893 | -0.0182 (1.000) | -0.0201 (1.000) |
| 4-cycle | +0.0618 | -0.0324 (1.000) | -0.0501 (1.000) |
| 4-path | -0.1914 | -0.0347 (1.000) | +0.0031 (0.017) |
