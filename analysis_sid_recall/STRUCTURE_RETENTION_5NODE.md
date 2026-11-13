# 结构保留度 vs 下游 recall（ORCA 4+5-节点轨道 88 维 -> 6 个角色）

配对：**67** 个有 recall 的臂（跳过 0 个），每个臂一条 PARTITION，结构固定、不随臂变化。

## 相关系数（Spearman，2000 次 bootstrap，n = 有效臂数）

| 序列 | n | rho | 95% bootstrap |
|---|---|---|---|
| `robust/centered/role4/prefix@1` | 62 | +0.675 | [+0.470, +0.824] |
| `centered/role4/prefix@1` | 67 | +0.648 | [+0.449, +0.801] |
| `robust/centered/role2/prefix@1` | 62 | +0.633 | [+0.385, +0.811] |
| `robust/centered/role1/prefix@1` | 62 | +0.630 | [+0.383, +0.809] |
| `centered/role1/prefix@1` | 67 | +0.602 | [+0.347, +0.787] |
| `centered/role2/prefix@1` | 67 | +0.582 | [+0.319, +0.777] |
| `role1/prefix@1` | 67 | +0.523 | [+0.248, +0.733] |
| `centered/role3/prefix@1` | 67 | +0.515 | [+0.266, +0.710] |
| `role2/prefix@1` | 67 | +0.498 | [+0.229, +0.718] |
| `robust/contrast/role_spread` | 62 | +0.459 | [+0.154, +0.697] |
| `role1/prefix@2` | 67 | +0.437 | [+0.192, +0.647] |
| `contrast/role_gap` | 67 | +0.431 | [+0.152, +0.681] |
| `contrast/role_spread` | 67 | +0.416 | [+0.137, +0.663] |
| `role4/prefix@1` | 67 | +0.414 | [+0.140, +0.635] |
| `role1/prefix@3` | 67 | +0.413 | [+0.148, +0.651] |
| `role3/prefix@1` | 67 | +0.394 | [+0.121, +0.612] |
| `role2/prefix@2` | 67 | +0.368 | [+0.113, +0.583] |
| `role5/prefix@1` | 67 | +0.357 | [+0.089, +0.576] |
| `overall/prefix@1` | 67 | +0.281 | [+0.006, +0.521] |
| `role3/prefix@2` | 67 | +0.269 | [+0.004, +0.503] |
| `role2/prefix@3` | 67 | +0.268 | [-0.008, +0.533] |
| `role4/prefix@2` | 67 | +0.257 | [-0.026, +0.501] |
| `role4/prefix@3` | 67 | +0.257 | [-0.033, +0.529] |
| `role3/prefix@3` | 67 | +0.251 | [-0.035, +0.529] |
| `role5/prefix@3` | 67 | +0.244 | [-0.048, +0.529] |
| `overall/prefix@3` | 67 | +0.228 | [-0.053, +0.500] |
| `role5/prefix@2` | 67 | +0.216 | [-0.059, +0.464] |
| `role0/prefix@1` | 67 | +0.173 | [-0.110, +0.431] |
| `overall/prefix@2` | 67 | +0.173 | [-0.099, +0.433] |
| `role0/prefix@3` | 67 | +0.133 | [-0.154, +0.432] |
| `role0/prefix@2` | 67 | +0.093 | [-0.198, +0.372] |
| `robust/centered/role5/prefix@1` | 62 | -0.424 | [-0.661, -0.137] |
| `centered/role5/prefix@1` | 67 | -0.458 | [-0.678, -0.186] |
| `centered/role0/prefix@1` | 67 | -0.538 | [-0.745, -0.292] |

## 各角色的平均保留度（跨臂均值，`prefix@1`）

| 角色 | 成员 | 平均 prefix@1 | 平均 prefix@3 | 臂数 |
|---|---|---|---|---|
| 0 | 3521 | 0.1761 | 0.0176 | 67 |
| 1 | 1311 | 0.2700 | 0.0214 | 67 |
| 2 | 2643 | 0.2612 | 0.0219 | 67 |
| 3 | 3543 | 0.2093 | 0.0187 | 67 |
| 4 | 657 | 0.1932 | 0.0181 | 67 |
| 5 | 12881 | 0.1783 | 0.0171 | 67 |

## 边界

- structure is ORCA's node orbits of graphlets up to 5 nodes (88 columns) plus four invariants; the six roles are NMF with k fixed by the 213 script, not selected here
- an edge is attributed to the role of its first endpoint (the history item), the same convention as the 213 audit
- every column compares the same SID position on both sides; no cross-level matching
- 67 pairs are 67 arms under distinct family names: no within-family replication exists, so the interval is over arms, not over runs of one arm
- Spearman on 67 arms with no multiple-comparison correction is exploratory; a row that clears zero is a hypothesis, not a finding
- the robust/ rows drop the arms whose level-1 share exceeds 0.5 (27, 28, 36, 37, 51): a level-1 code that coarse cannot carry structure, and the rule was fixed before any correlation was computed
- report.json disagrees with the archive on 51, 52, 53, 54, 55, 56; the ledger and the mirror agree with each other, so the archive is what these numbers use
