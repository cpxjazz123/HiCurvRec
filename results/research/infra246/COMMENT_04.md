## 节点 4:TIGER SID 来源校验与基线复算(不训练)

### 来源定案(`results/research/infra246/tiger_audit.json`,只读)

| 产物 | sha256(前16) | 前三码碰撞 | 重复份额 | 判定 |
|---|---|---|---|---|
| `item_sids.json` | `025ad8bd5809ce62` | 0.169276 | 0.265303 | **Stage3 实际读取**(日志 `code_path_sha256=025ad8bd…`)→ 已确认基线 |
| `sids_for_hgrec.npy` | `6f2682c18d20ba88` | 0.110384 | 0.184813 | 与 checkpoint 同属**另一轮**(300 epoch),Stage3 从未读取 |
| `rqvae_best.pth` | `af8efc29a8bcbbe5` | — | — | 自带 `epoch=300, collision=0.11038` → 与 NPY 同轮 |
| 只读重导(新目录 `tiger_reexport/`) | 新文件 | 0.109936 | 0.184244 | 与 NPY 仅 **115 行**不同(集中在第三码与消歧码,并列导致) |

结论:冻结臂目录**混了两轮产物**;checkpoint 无法重建 Stage3 用过的表(重导与 JSON 24,587 行全不同),因此**不得**用该 checkpoint 重导去代表 Stage3 的基线。重建误差(冻结 checkpoint 只读前向):`reconstruction_mse = 9.61e-05`(768 维输入,平方误差单位)。历史 Recall 的绑定:TIGER 的 `test_R@10` 由 Stage3 日志的 `code_path_sha256` 绑定到 JSON;无该记录的臂标 NA,不猜测。

### 更正 v1 的错误结论(重要)
v1 的 INVENTORY 称 §63 记录的 `l3_collision=0.2653`「不可复算」——**这是错的**:该记录的方法学写的是"item-weighted share of repeated three-token SIDs",即**重复商品份额** = 0.265303;而 §89 门禁口径 `1 − distinct/items` = 0.169276,同样正确。两个口径共用 "collision" 一词才是缺陷。已在 `results/research/infra245/INVENTORY.md` §5(b) 加更正注记,并在 v2 `CONTRACT.md` §13 固定两口径。

### 基线复算(`results/research/infra246/tiger_baseline_v2.json`)
用**已确认**的 `item_sids.json` 在统一独立 bank 上复算(prefix@1/2/3、关系强度 Spearman、结构保真与多子图);两个旧表另列,不混用。

| 组 | prefix@1 | prefix@2 | prefix@3 |
|---|---|---|---|
| train_direct | 0.0878 | 0.0092 | 0.0018 |
| train_shared_in | 0.0435 | 0.0028 | 0.0004 |
| train_shared_out | 0.0409 | 0.0027 | 0.0004 |
| train_two_hop | 0.0139 | 0.0003 | 0.0000 |
| train_degree_matched | 0.0136 | 0.0008 | 0.0000 |
| strict_struct_test_direct | 0.0777 | 0.0114 | 0.0004 |

排序与图关系强度**单调一致**(直接边 > 共同前驱/后继 ≫ 两跳 ≈ 度数匹配),说明 SID 前缀共享确实携带图结构信息;这是**描述性**结论,不构成"复杂几何有用"的证据。关系强度 Spearman 用真实冻结图分数(`two_hop_count`、`jaccard_out`、`hop_full_visible`)对前缀共享计算,常数输入返回 NA;Jaccard 与 hop=2 的**定义性**关系不作为科学结论。

下一步:结论对账(头条数字独立重算比对)、提交推送与关闭。
