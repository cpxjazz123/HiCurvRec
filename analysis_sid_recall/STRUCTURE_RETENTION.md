# 结构保留度 vs 下游 recall（ORCA 4-节点轨道 -> 6 个角色）

配对：**68** 个有 recall 的臂（跳过 0 个），每个臂一条 PARTITION，结构固定、不随臂变化。

## 相关系数（Spearman，2000 次 bootstrap，n = 有效臂数）

| 序列 | n | rho | 95% bootstrap |
|---|---|---|---|
| `robust/centered/role1/prefix@1` | 63 | +0.635 | [+0.407, +0.803] |
| `robust/centered/role2/prefix@1` | 63 | +0.615 | [+0.378, +0.797] |
| `centered/role1/prefix@1` | 68 | +0.612 | [+0.377, +0.780] |
| `centered/role2/prefix@1` | 68 | +0.565 | [+0.301, +0.759] |
| `role1/prefix@1` | 68 | +0.512 | [+0.258, +0.707] |
| `robust/contrast/role_spread` | 63 | +0.510 | [+0.263, +0.725] |
| `role2/prefix@1` | 68 | +0.505 | [+0.247, +0.706] |
| `contrast/role_spread` | 68 | +0.479 | [+0.215, +0.693] |
| `contrast/role_gap` | 68 | +0.455 | [+0.188, +0.681] |
| `role3/prefix@1` | 68 | +0.407 | [+0.152, +0.614] |
| `role5/prefix@1` | 68 | +0.391 | [+0.139, +0.599] |
| `role1/any` | 68 | +0.372 | [+0.105, +0.603] |
| `role1/prefix@3` | 68 | +0.353 | [+0.099, +0.584] |
| `role2/any` | 68 | +0.352 | [+0.078, +0.585] |
| `role0/prefix@1` | 68 | +0.305 | [+0.048, +0.522] |
| `overall/prefix@1` | 68 | +0.295 | [+0.033, +0.520] |
| `role2/prefix@3` | 68 | +0.290 | [+0.017, +0.537] |
| `role5/prefix@3` | 68 | +0.249 | [-0.039, +0.503] |
| `role4/prefix@1` | 68 | +0.243 | [-0.030, +0.481] |
| `role3/any` | 68 | +0.234 | [-0.041, +0.475] |
| `overall/prefix@3` | 68 | +0.227 | [-0.052, +0.489] |
| `role0/prefix@3` | 68 | +0.226 | [-0.066, +0.490] |
| `role5/any` | 68 | +0.207 | [-0.068, +0.452] |
| `role3/prefix@3` | 68 | +0.206 | [-0.082, +0.471] |
| `centered/role0/prefix@1` | 68 | +0.198 | [-0.057, +0.433] |
| `role4/prefix@3` | 68 | +0.196 | [-0.079, +0.467] |
| `role0/any` | 68 | +0.161 | [-0.113, +0.416] |
| `overall/any` | 68 | +0.159 | [-0.120, +0.419] |
| `role4/any` | 68 | +0.142 | [-0.135, +0.406] |
| `centered/role3/prefix@1` | 68 | +0.032 | [-0.245, +0.303] |
| `robust/centered/role5/prefix@1` | 63 | -0.334 | [-0.572, -0.058] |
| `centered/role4/prefix@1` | 68 | -0.340 | [-0.587, -0.033] |
| `centered/role5/prefix@1` | 68 | -0.352 | [-0.586, -0.093] |
| `robust/centered/role4/prefix@1` | 63 | -0.362 | [-0.607, -0.056] |

## 各角色的平均保留度（跨臂均值，`prefix@1`）

| 角色 | 成员 | 平均 prefix@1 | 平均 prefix@3 | 臂数 |
|---|---|---|---|---|
| 0 | 3679 | 0.1882 | 0.0167 | 68 |
| 1 | 1144 | 0.2664 | 0.0191 | 68 |
| 2 | 2895 | 0.2582 | 0.0214 | 68 |
| 3 | 2081 | 0.1806 | 0.0175 | 68 |
| 4 | 6917 | 0.1842 | 0.0177 | 68 |
| 5 | 7840 | 0.1710 | 0.0162 | 68 |

## 边界

- structure is ORCA's node orbits of graphlets up to 4 nodes plus four invariants; the six roles are NMF with k fixed by the 213 script, not selected here
- an edge is attributed to the role of its first endpoint (the history item), the same convention as the 213 audit
- 'any' counts a token value matched at any level, so a level-1 code of one item may match a level-2 code of the other; prefix@1 and prefix@3 do not
- 68 pairs are 68 arms under distinct family names: no within-family replication exists, so the interval is over arms, not over runs of one arm
- Spearman on 68 arms with no multiple-comparison correction is exploratory; a row that clears zero is a hypothesis, not a finding
- the robust/ rows drop the arms whose level-1 share exceeds 0.5 (27, 28, 36, 37, 51): a level-1 code that coarse cannot carry structure, and the rule was fixed before any correlation was computed
- report.json disagrees with the archive on 51, 52, 53, 54, 55, 56; the ledger and the mirror agree with each other, so the archive is what these numbers use
