# INVENTORY v2 — 资产、复用决定与已知错误 (issue #246)

v1 清单保留在 `results/research/infra245/INVENTORY.md`;本文件只写 v2 的增量与更正。
v2 资产全部在 `results/research/infra246/assets/`,manifest 为 `manifest.json`(逐项 sha256 / schema /
shape / dtype / graph_version / legal_use / version / purpose / command / 源哈希 / 码哈希 / 依赖)。

## 1. 新增的生产模块(公共位置,不复制到 issue 目录)

| 模块 | 职责 | 复用决定 |
|---|---|---|
| `config.py` | 版本、划分 seed/比例、cap、种子、资产根;仓库根由 `stage2_RQ-VAE/baseline` 标记定位(被拷贝的包仍指向真实数据) | 所有新代码只从这里取常量 |
| `graph_build.py` | 序列→加权边多重集、真实 item ID 集、自环计数 | 替代各 issue 里自写的边统计 |
| `splits.py` | train 内 80/10/10 用户划分与审计 | 所有泛化资产的唯一划分来源 |
| `isolation.py` | 留出支撑(布尔口径)、无向双向阻断、bank 审计 | 替代 v1 `build_assets.py` 里的加权相减 |
| `bank_features.py` | 逐对图特征(全图无向 hop、有向 hop、子图 hop、两跳计数、Jaccard、Bhattacharyya、见证) | 替代各实验自写的距离计算 |
| `ppr.py` | 稀疏 LU 精确 PPR(悬挂质量归还源点) | 替代 256 探针式 PPR |
| `curvature.py` | 可见图上的精确 Ollivier–Ricci(运输 LP) | 替代把 #236 全量图曲率直接搬到可见图的做法 |
| `graphlet.py`(扩展) | 3 节点 canonical/轨道/族/实例、全量向量化计数、4 节点抽样 | 替代布尔标签与"同一 ego 换采样当不同 graphlet"的做法 |
| `build_assets_v2.py` / `baseline_v2.py` / `tiger_audit.py` / `module_checks.py` | 构建、复算、来源校验、分模块验证 | 可重复执行入口 |

## 2. v1 资产的去向

| v1 资产 | 决定 |
|---|---|
| `results/research/infra245/assets/*` | **保留可追溯,不再更新**;v2 资产不与其混用,manifest 带版本 |
| `results/research/infra245/baseline_tiger.json` | 保留为 v1 口径记录;v2 的复算在 `tiger_baseline_v2.json` |
| v1 的 `build_assets.py` | 保留(历史可复现);新资产一律用 `build_assets_v2.py` |

## 3. 已确认的已知错误(v2 实测)

**(a) 冻结 TIGER 目录混了两轮产物。**
`item_sids.json`(sha `025ad8bd…`,碰撞 0.169276)与 `sids_for_hgrec.npy`+`rqvae_best.pth`
(sha `6f2682c1…` / `af8efc29…`,碰撞 0.110384,checkpoint 自带 `epoch=300`)分属两轮。
Stage3 的日志写明读的是 `item_sids.json`(`code_path_sha256=025ad8bd…`),所以**已确认基线 = JSON**;
checkpoint 无法重建它(重导与 NPY 仅差 115 行 tie),**不得**用该 checkpoint 代表 Stage3 用过的表。
证据:`results/research/infra246/tiger_audit.json`。

**(b) 更正 v1 的错误结论。**
v1 说 §63 记录的 `l3_collision=0.2653`「不可复算」。实测它是**重复商品份额**(重复前三码的商品占
24,587 件中的比例)= 0.265303;门禁口径 `1 − distinct/items` = 0.169276,同样正确。两个口径共用
"collision" 一词才是缺陷。已在 `results/research/infra245/INVENTORY.md` §5(b) 加更正注记,
并在 v2 `CONTRACT.md` §13 固定两口径。

**(c) 几何口径在 v1 不连续。** 球面与双曲分支的 2 倍约定不一致,`k=0` 与曲率族差 2 倍。
v2 统一为 `d(0,expmap0(v)) = 2|v|`、`lim_{k→0} d_k = 2|x−y|`;`k≠0` 的距离值不变。

**(d) 旧 `ego_induced` 的方向判定不可信。** 它用另一次随机采样(固定 seed、只取 8 个)的后继集合判角色,
超出预算的真实后继会被误判。v2 改读 CSR membership。

**(e) 旧 `build_assets.py` 的留出计算会丢方向。** `eval − eval.multiply(visible)` 是加权减法,
可见权重更大时差值为负,再对称求和把真实留出方向抵消。v2 用布尔支撑集差。

**(f) 旧 fork/merge 判据过严。** 要求整个后继邻域两两独立,会漏掉"三后继中两个相连"这类真实 fork。
v2 改为存在性定义,并在契约里区分宽松谓词 / 严格诱导有向实例 / 无向三角。

**(g) 旧 `subgraph_summary` 无方向语义。** 失真(越低越好)被当成相似度处理,pass 用 `>=`、worst 取 min、
p90 取中心均值,三个分母混用。v2 显式 `direction` 并把三个比例与"无有效子图中心"分列。

## 4. 与 Stage3 的关系

Stage3 的 validation/test 用户划分与本轮 train 内三路划分**互不相干**:本轮的 `struct_val`/
`struct_test` 只用于图结构评估,不参与传播、监督或标定;Stage3 的划分文件未被触碰。
两者的用户 ID 交集未被合并计算——issue 只要求"明确区分",并要求 Stage3 标签不参与图构建 ✓。

## 5. 下游必须做的事

新机制只实现自身部分(曲率权重、温度、门控、尺度、λ),逐对结果经 `assets.join_scores_v2` 按
pair id 对齐;读取资产前先 `assets.audit_manifest` + `assets.verify_manifest`,并用
`assets.assert_legal_use` 声明用途(description / generalisation / evaluation)。
