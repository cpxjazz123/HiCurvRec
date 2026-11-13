# 结构保留度 vs 下游 recall（ORCA 4+5-节点轨道 88 维 -> 6 个角色）

配对：**68** 个有 recall 的臂（跳过 0 个），每个臂一条 PARTITION，结构固定、不随臂变化。

## 相关系数（Spearman，2000 次 bootstrap，n = 有效臂数）

| 序列 | n | rho | 95% bootstrap |
|---|---|---|---|
| `robust/centered/role4/prefix@1` | 63 | +0.664 | [+0.457, +0.810] |
| `centered/role4/prefix@1` | 68 | +0.638 | [+0.435, +0.789] |
| `robust/centered/role1/prefix@1` | 63 | +0.632 | [+0.406, +0.804] |
| `robust/centered/role2/prefix@1` | 63 | +0.617 | [+0.383, +0.802] |
| `centered/role1/prefix@1` | 68 | +0.601 | [+0.350, +0.779] |
| `centered/role2/prefix@1` | 68 | +0.566 | [+0.301, +0.760] |
| `role1/prefix@1` | 68 | +0.531 | [+0.278, +0.723] |
| `centered/role3/prefix@1` | 68 | +0.522 | [+0.276, +0.716] |
| `role2/prefix@1` | 68 | +0.503 | [+0.246, +0.703] |
| `robust/contrast/role_spread` | 63 | +0.460 | [+0.199, +0.695] |
| `role1/prefix@2` | 68 | +0.449 | [+0.198, +0.652] |
| `contrast/role_gap` | 68 | +0.435 | [+0.164, +0.675] |
| `role1/prefix@3` | 68 | +0.428 | [+0.169, +0.657] |
| `role4/prefix@1` | 68 | +0.425 | [+0.172, +0.630] |
| `contrast/role_spread` | 68 | +0.416 | [+0.139, +0.651] |
| `role3/prefix@1` | 68 | +0.407 | [+0.165, +0.612] |
| `role5/prefix@1` | 68 | +0.371 | [+0.124, +0.574] |
| `role2/prefix@2` | 68 | +0.368 | [+0.109, +0.571] |
| `overall/prefix@1` | 68 | +0.295 | [+0.033, +0.520] |
| `role3/prefix@2` | 68 | +0.288 | [+0.022, +0.516] |
| `role4/prefix@2` | 68 | +0.274 | [-0.008, +0.513] |
| `role2/prefix@3` | 68 | +0.257 | [-0.016, +0.509] |
| `role3/prefix@3` | 68 | +0.252 | [-0.034, +0.519] |
| `role4/prefix@3` | 68 | +0.241 | [-0.036, +0.498] |
| `role5/prefix@3` | 68 | +0.240 | [-0.045, +0.499] |
| `overall/prefix@3` | 68 | +0.227 | [-0.052, +0.489] |
| `role5/prefix@2` | 68 | +0.226 | [-0.048, +0.468] |
| `role0/prefix@1` | 68 | +0.192 | [-0.081, +0.445] |
| `overall/prefix@2` | 68 | +0.186 | [-0.085, +0.444] |
| `role0/prefix@3` | 68 | +0.148 | [-0.137, +0.434] |
| `role0/prefix@2` | 68 | +0.114 | [-0.166, +0.397] |
| `robust/centered/role5/prefix@1` | 63 | -0.431 | [-0.658, -0.157] |
| `centered/role5/prefix@1` | 68 | -0.462 | [-0.668, -0.204] |
| `centered/role0/prefix@1` | 68 | -0.543 | [-0.744, -0.285] |

## 各角色的平均保留度（跨臂均值，`prefix@1`）

| 角色 | 成员 | 平均 prefix@1 | 平均 prefix@3 | 臂数 |
|---|---|---|---|---|
| 0 | 3515 | 0.1746 | 0.0173 | 68 |
| 1 | 1386 | 0.2684 | 0.0212 | 68 |
| 2 | 2574 | 0.2598 | 0.0217 | 68 |
| 3 | 3835 | 0.2078 | 0.0184 | 68 |
| 4 | 875 | 0.1917 | 0.0179 | 68 |
| 5 | 12371 | 0.1769 | 0.0168 | 68 |

## 边界

- structure is ORCA's node orbits of graphlets up to 5 nodes (88 columns) plus four invariants; the six roles are NMF with k fixed by the 213 script, not selected here
- an edge is attributed to the role of its first endpoint (the history item), the same convention as the 213 audit
- every column compares the same SID position on both sides; no cross-level matching
- 68 pairs are 68 arms under distinct family names: no within-family replication exists, so the interval is over arms, not over runs of one arm
- Spearman on 68 arms with no multiple-comparison correction is exploratory; a row that clears zero is a hypothesis, not a finding
- the robust/ rows drop the arms whose level-1 share exceeds 0.5 (27, 28, 36, 37, 51): a level-1 code that coarse cannot carry structure, and the rule was fixed before any correlation was computed
- report.json disagrees with the archive on 51, 52, 53, 54, 55, 56; the ledger and the mirror agree with each other, so the archive is what these numbers use
