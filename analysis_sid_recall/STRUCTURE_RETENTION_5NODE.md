# 结构保留度 vs 下游 recall（ORCA 4+5-节点轨道 88 维 -> 6 个角色）

配对：**68** 个有 recall 的臂（跳过 0 个），每个臂一条 PARTITION，结构固定、不随臂变化。

## 相关系数（Spearman，2000 次 bootstrap，n = 有效臂数）

| 序列 | n | rho | 95% bootstrap |
|---|---|---|---|
| `robust/centered/role1/prefix@1` | 63 | +0.635 | [+0.407, +0.806] |
| `robust/centered/role2/prefix@1` | 63 | +0.618 | [+0.388, +0.801] |
| `centered/role1/prefix@1` | 68 | +0.602 | [+0.353, +0.779] |
| `centered/role2/prefix@1` | 68 | +0.567 | [+0.306, +0.761] |
| `role1/prefix@1` | 68 | +0.528 | [+0.274, +0.724] |
| `centered/role3/prefix@1` | 68 | +0.513 | [+0.269, +0.708] |
| `role2/prefix@1` | 68 | +0.500 | [+0.245, +0.702] |
| `robust/contrast/role_spread` | 63 | +0.454 | [+0.191, +0.690] |
| `contrast/role_gap` | 68 | +0.441 | [+0.166, +0.678] |
| `robust/centered/role4/prefix@1` | 63 | +0.440 | [+0.196, +0.644] |
| `role4/prefix@1` | 68 | +0.420 | [+0.170, +0.625] |
| `role1/prefix@3` | 68 | +0.418 | [+0.164, +0.647] |
| `contrast/role_spread` | 68 | +0.416 | [+0.140, +0.655] |
| `role3/prefix@1` | 68 | +0.391 | [+0.147, +0.595] |
| `role1/any` | 68 | +0.389 | [+0.118, +0.616] |
| `role5/prefix@1` | 68 | +0.384 | [+0.136, +0.586] |
| `centered/role4/prefix@1` | 68 | +0.350 | [+0.103, +0.566] |
| `role2/any` | 68 | +0.343 | [+0.074, +0.574] |
| `overall/prefix@1` | 68 | +0.295 | [+0.033, +0.520] |
| `role3/prefix@3` | 68 | +0.261 | [-0.022, +0.527] |
| `role2/prefix@3` | 68 | +0.257 | [-0.018, +0.508] |
| `role5/prefix@3` | 68 | +0.240 | [-0.043, +0.502] |
| `role3/any` | 68 | +0.233 | [-0.033, +0.474] |
| `role4/any` | 68 | +0.231 | [-0.040, +0.476] |
| `overall/prefix@3` | 68 | +0.227 | [-0.052, +0.489] |
| `role4/prefix@3` | 68 | +0.205 | [-0.077, +0.464] |
| `role0/prefix@1` | 68 | +0.190 | [-0.081, +0.443] |
| `role5/any` | 68 | +0.177 | [-0.095, +0.425] |
| `overall/any` | 68 | +0.159 | [-0.120, +0.419] |
| `role0/prefix@3` | 68 | +0.156 | [-0.132, +0.438] |
| `role0/any` | 68 | +0.092 | [-0.192, +0.372] |
| `robust/centered/role5/prefix@1` | 63 | -0.425 | [-0.656, -0.154] |
| `centered/role5/prefix@1` | 68 | -0.458 | [-0.665, -0.203] |
| `centered/role0/prefix@1` | 68 | -0.554 | [-0.754, -0.296] |

## 各角色的平均保留度（跨臂均值，`prefix@1`）

| 角色 | 成员 | 平均 prefix@1 | 平均 prefix@3 | 臂数 |
|---|---|---|---|---|
| 0 | 3515 | 0.1724 | 0.0173 | 68 |
| 1 | 1386 | 0.2665 | 0.0208 | 68 |
| 2 | 2574 | 0.2585 | 0.0213 | 68 |
| 3 | 3835 | 0.2078 | 0.0184 | 68 |
| 4 | 875 | 0.1826 | 0.0170 | 68 |
| 5 | 12371 | 0.1767 | 0.0169 | 68 |

## 边界

- structure is ORCA's node orbits of graphlets up to 5 nodes (88 columns) plus four invariants; the six roles are NMF with k fixed by the 213 script, not selected here
- an edge is attributed to the role of its first endpoint (the history item), the same convention as the 213 audit
- 'any' counts a token value matched at any level, so a level-1 code of one item may match a level-2 code of the other; prefix@1 and prefix@3 do not
- 68 pairs are 68 arms under distinct family names: no within-family replication exists, so the interval is over arms, not over runs of one arm
- Spearman on 68 arms with no multiple-comparison correction is exploratory; a row that clears zero is a hypothesis, not a finding
- the robust/ rows drop the arms whose level-1 share exceeds 0.5 (27, 28, 36, 37, 51): a level-1 code that coarse cannot carry structure, and the rule was fixed before any correlation was computed
- report.json disagrees with the archive on 51, 52, 53, 54, 55, 56; the ledger and the mirror agree with each other, so the archive is what these numbers use
