# 结构保留度 vs 下游 recall（ORCA 4-节点轨道 -> 6 个角色）

配对：**67** 个有 recall 的臂（跳过 0 个），每个臂一条 PARTITION，结构固定、不随臂变化。

## 相关系数（Spearman，2000 次 bootstrap，n = 有效臂数）

| 序列 | n | rho | 95% bootstrap |
|---|---|---|---|
| `robust/centered/role1/prefix@1` | 62 | +0.636 | [+0.403, +0.804] |
| `robust/centered/role2/prefix@1` | 62 | +0.629 | [+0.378, +0.804] |
| `centered/role1/prefix@1` | 67 | +0.616 | [+0.374, +0.794] |
| `centered/role2/prefix@1` | 67 | +0.579 | [+0.320, +0.774] |
| `robust/contrast/role_spread` | 62 | +0.511 | [+0.227, +0.732] |
| `role1/prefix@1` | 67 | +0.505 | [+0.239, +0.718] |
| `role2/prefix@1` | 67 | +0.501 | [+0.230, +0.721] |
| `contrast/role_spread` | 67 | +0.481 | [+0.214, +0.707] |
| `contrast/role_gap` | 67 | +0.454 | [+0.184, +0.687] |
| `role1/prefix@2` | 67 | +0.412 | [+0.163, +0.627] |
| `role3/prefix@1` | 67 | +0.392 | [+0.121, +0.613] |
| `role2/prefix@2` | 67 | +0.384 | [+0.126, +0.597] |
| `role5/prefix@1` | 67 | +0.378 | [+0.104, +0.601] |
| `role1/prefix@3` | 67 | +0.342 | [+0.069, +0.585] |
| `role2/prefix@3` | 67 | +0.292 | [+0.016, +0.551] |
| `role0/prefix@1` | 67 | +0.290 | [+0.021, +0.520] |
| `overall/prefix@1` | 67 | +0.281 | [+0.006, +0.521] |
| `role5/prefix@2` | 67 | +0.254 | [-0.030, +0.498] |
| `role3/prefix@2` | 67 | +0.253 | [-0.028, +0.499] |
| `role5/prefix@3` | 67 | +0.242 | [-0.038, +0.517] |
| `overall/prefix@3` | 67 | +0.228 | [-0.053, +0.500] |
| `role4/prefix@1` | 67 | +0.226 | [-0.058, +0.482] |
| `centered/role0/prefix@1` | 67 | +0.217 | [-0.060, +0.457] |
| `role0/prefix@3` | 67 | +0.209 | [-0.081, +0.499] |
| `role3/prefix@3` | 67 | +0.200 | [-0.094, +0.483] |
| `role4/prefix@3` | 67 | +0.194 | [-0.090, +0.481] |
| `role0/prefix@2` | 67 | +0.193 | [-0.087, +0.452] |
| `overall/prefix@2` | 67 | +0.173 | [-0.099, +0.433] |
| `role4/prefix@2` | 67 | +0.140 | [-0.134, +0.411] |
| `centered/role3/prefix@1` | 67 | +0.030 | [-0.231, +0.284] |
| `robust/centered/role5/prefix@1` | 62 | -0.326 | [-0.558, -0.035] |
| `centered/role5/prefix@1` | 67 | -0.345 | [-0.574, -0.081] |
| `centered/role4/prefix@1` | 67 | -0.364 | [-0.612, -0.072] |
| `robust/centered/role4/prefix@1` | 62 | -0.388 | [-0.633, -0.092] |

## 各角色的平均保留度（跨臂均值，`prefix@1`）

| 角色 | 成员 | 平均 prefix@1 | 平均 prefix@3 | 臂数 |
|---|---|---|---|---|
| 0 | 3679 | 0.1897 | 0.0169 | 67 |
| 1 | 1144 | 0.2679 | 0.0193 | 67 |
| 2 | 2895 | 0.2596 | 0.0216 | 67 |
| 3 | 2081 | 0.1821 | 0.0178 | 67 |
| 4 | 6917 | 0.1857 | 0.0180 | 67 |
| 5 | 7840 | 0.1725 | 0.0165 | 67 |

## 边界

- structure is ORCA's node orbits of graphlets up to 4 nodes plus four invariants; the six roles are NMF with k fixed by the 213 script, not selected here
- an edge is attributed to the role of its first endpoint (the history item), the same convention as the 213 audit
- every column compares the same SID position on both sides; no cross-level matching
- 67 pairs are 67 arms under distinct family names: no within-family replication exists, so the interval is over arms, not over runs of one arm
- Spearman on 67 arms with no multiple-comparison correction is exploratory; a row that clears zero is a hypothesis, not a finding
- the robust/ rows drop the arms whose level-1 share exceeds 0.5 (27, 28, 36, 37, 51): a level-1 code that coarse cannot carry structure, and the rule was fixed before any correlation was computed
- report.json disagrees with the archive on 51, 52, 53, 54, 55, 56; the ledger and the mirror agree with each other, so the archive is what these numbers use
