# 结构保留度 vs 下游 recall（ORCA 4+5-节点轨道 88 维 -> 6 个角色）

配对：**67** 个有 recall 的臂（跳过 0 个），每个臂一条 PARTITION，结构固定、不随臂变化。

## 相关系数（Spearman，2000 次 bootstrap，n = 有效臂数）

| 序列 | n | rho | 95% bootstrap |
|---|---|---|---|
| `robust/centered/role5/prefix@1` | 62 | +0.617 | [+0.378, +0.792] |
| `robust/centered/role3/prefix@1` | 62 | +0.613 | [+0.358, +0.801] |
| `centered/role5/prefix@1` | 67 | +0.573 | [+0.316, +0.764] |
| `centered/role3/prefix@1` | 67 | +0.569 | [+0.309, +0.764] |
| `role3/prefix@1` | 67 | +0.505 | [+0.232, +0.724] |
| `centered/role4/prefix@1` | 67 | +0.458 | [+0.205, +0.659] |
| `robust/contrast/role_spread` | 62 | +0.453 | [+0.147, +0.703] |
| `role5/prefix@1` | 67 | +0.445 | [+0.173, +0.665] |
| `contrast/role_gap` | 67 | +0.434 | [+0.152, +0.686] |
| `contrast/role_spread` | 67 | +0.422 | [+0.142, +0.677] |
| `role3/prefix@2` | 67 | +0.397 | [+0.140, +0.613] |
| `role4/prefix@1` | 67 | +0.392 | [+0.118, +0.612] |
| `role2/prefix@1` | 67 | +0.387 | [+0.113, +0.610] |
| `role5/prefix@2` | 67 | +0.359 | [+0.103, +0.574] |
| `role5/prefix@3` | 67 | +0.321 | [+0.049, +0.588] |
| `role3/prefix@3` | 67 | +0.317 | [+0.044, +0.568] |
| `overall/prefix@1` | 67 | +0.281 | [+0.006, +0.521] |
| `role6/prefix@1` | 67 | +0.259 | [-0.010, +0.496] |
| `role6/prefix@3` | 67 | +0.252 | [-0.046, +0.541] |
| `role2/prefix@2` | 67 | +0.246 | [-0.033, +0.489] |
| `role4/prefix@2` | 67 | +0.245 | [-0.024, +0.488] |
| `role4/prefix@3` | 67 | +0.233 | [-0.046, +0.501] |
| `role2/prefix@3` | 67 | +0.232 | [-0.049, +0.498] |
| `overall/prefix@3` | 67 | +0.228 | [-0.053, +0.500] |
| `overall/prefix@2` | 67 | +0.173 | [-0.099, +0.433] |
| `role6/prefix@2` | 67 | +0.172 | [-0.107, +0.434] |
| `role1/prefix@1` | 67 | +0.143 | [-0.151, +0.411] |
| `role1/prefix@3` | 67 | +0.064 | [-0.223, +0.364] |
| `role1/prefix@2` | 67 | +0.056 | [-0.241, +0.336] |
| `centered/role2/prefix@1` | 67 | -0.107 | [-0.377, +0.171] |
| `robust/centered/role2/prefix@1` | 62 | -0.108 | [-0.379, +0.196] |
| `centered/role6/prefix@1` | 67 | -0.402 | [-0.649, -0.127] |
| `robust/centered/role6/prefix@1` | 62 | -0.403 | [-0.657, -0.100] |
| `centered/role1/prefix@1` | 67 | -0.558 | [-0.758, -0.316] |

## 各角色的平均保留度（跨臂均值，`prefix@1`）

| 角色 | 成员 | 平均度 | 平均聚类 | 平均 core | 平均 prefix@1 | 平均 prefix@3 | 臂数 |
|---|---|---|---|---|---|---|---|
| 1 | 2448 | 58.7 | 0.010 | 13.0 | 0.1688 | 0.0177 | 67 |
| 2 | 2715 | 9.5 | 0.024 | 8.4 | 0.1767 | 0.0158 | 67 |
| 3 | 4489 | 10.2 | 0.023 | 7.4 | 0.2572 | 0.0218 | 67 |
| 4 | 4053 | 26.4 | 0.022 | 15.0 | 0.2092 | 0.0184 | 67 |
| 5 | 2992 | 7.6 | 0.101 | 7.2 | 0.2077 | 0.0162 | 67 |
| 6 | 7859 | 36.4 | 0.061 | 19.2 | 0.1781 | 0.0173 | 67 |

## 边界

- structure is ORCA's node orbits of graphlets up to 5 nodes (88 columns) plus four invariants; the six roles are NMF with k fixed by the 213 script, not selected here
- an edge is attributed to the role of its first endpoint (the history item), the same convention as the 213 audit
- every column compares the same SID position on both sides; no cross-level matching
- 67 pairs are 67 arms under distinct family names: no within-family replication exists, so the interval is over arms, not over runs of one arm
- Spearman on 67 arms with no multiple-comparison correction is exploratory; a row that clears zero is a hypothesis, not a finding
- the robust/ rows drop the arms whose level-1 share exceeds 0.5 (27, 28, 36, 37, 51): a level-1 code that coarse cannot carry structure, and the rule was fixed before any correlation was computed
- report.json disagrees with the archive on ; the ledger and the mirror agree with each other, so the archive is what these numbers use
