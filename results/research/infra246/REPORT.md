# REPORT — issue #246 (Infra v2: 修复数据隔离、精确有向 Graphlet 实例、结构评估与 TIGER SID 来源校验)

本轮只做 Infra:不训练 Teacher/GNN/RQ-VAE/Stage3,不重训 TIGER,不扫参、不多种子(issue 原文)。
所有历史产物只读;新增内容在 `tools/graph_infra/` 与 `results/research/infra246/`。

## 1. 交付物

| 交付物 | 位置 |
|---|---|
| 共享生产模块(新增 8 个,修复 8 个) | `tools/graph_infra/`(`config`, `graph_build`, `splits`, `isolation`, `bank_features`, `ppr`, `curvature`, `graphlet` 扩展) |
| v2 资产构建与验证入口 | `tools/graph_infra/build_assets_v2.py`、`verify_assets.py --version v2` |
| TIGER 来源校验 / 基线复算 | `tools/graph_infra/tiger_audit.py`、`baseline_v2.py` |
| 分模块验证 | `tools/graph_infra/module_checks.py` → `results/research/infra246/MODULE_CHECKS.json` |
| 验收测试(39 条断言 + 14 个变异) | `tools/graph_infra/tests/test_v2_*.py`、`test_mutation_v2.py` |
| 冻结契约 / 资产清单 / 复用指南 | `results/research/infra246/{CONTRACT,INVENTORY,INFRA,REUSE}.md` |
| v1 结论更正 | `results/research/infra245/INVENTORY.md` §5(b) |

## 2. 修复的缺陷(先红后绿)

首跑 **36 FAIL / 2 PASS**(红),修复后全绿。逐条对应 issue 的 P0 反例:

| # | 缺陷 | 根因 | 修复 | 抓住它的断言 |
|---|---|---|---|---|
| 1 | 留出方向被抵消 | `eval − eval.multiply(visible)` 是**加权**减法,可见权重 3 让差值变负,再对称求和把真实留出方向抵消 | `isolation.held_out_support` 先取布尔支撑集差再对称并集 | `test_held_out_support_uses_boolean_semantics`(已知答案:可见 `(7→8)` 权重 3、留出 `(8→7)` 存在) |
| 2 | ego 角色判错 | `ego_induced` 用**另一次随机采样**(`default_rng(0)`,只取 8 个)的后继集合判方向 | `root_roles` 直接读 CSR membership | `test_direction_reads_real_csr_membership_beyond_the_sample_budget` |
| 3 | 互惠邻居只有一个 mask | 角色来自采样集合而非真实邻接 | in/out 两个独立布尔 mask,互惠=两个都置位 | `test_reciprocal_neighbour_keeps_both_masks` |
| 4 | hop 是遍历深度 | 采样截断时,真实 1-hop 节点被记成 2 | `hop` 按根的真实邻接重算,`traversal_depth` 另存 | `test_hop_is_the_shortest_hop_not_the_traversal_depth`、`test_traversal_depth_and_true_hop_can_differ` |
| 5 | 根索引无保证 | 依赖"中心在第 0 行" | `root_local` 显式;`ego_full` 返回真实中心行 | `test_root_local_is_explicit_when_the_root_has_the_largest_id` |
| 6 | cap 语义不明 | 旧代码 `cap+1` 含根 | 契约写明 cap=**邻居数**,根另计,`cap_includes_root=False` | `test_cap_counts_neighbours_not_the_root` |
| 7 | batch 只补节点 | 缺边/边 mask/特征 | `batch_egos` 返回 `edges/edge_mask/features/root_row` | `test_batch_egos_returns_edges_masks_and_features` |
| 8 | fork/merge 漏判 | 要求**整个邻域两两独立** | 存在性定义(一对不相邻即成立),契约区分宽松谓词/严格诱导实例/无向三角 | `test_fork_relaxed_accepts_two_adjacent_plus_one_isolated_successor` |
| 9 | 无精确 graphlet 分类 | 只有 5 个布尔标签 | 3 节点 canonical(6 置换取最小)+自同构轨道+16 个一一对应族名+实例枚举 | `test_canonicalisation_matches_the_reference_on_all_64_patterns`、`test_orbits_match_the_reference_on_all_64_patterns` |
| 10 | 结构保真指标口径错 | 无 direction、pass 用 `>=`、worst 取 min、p90 取中心均值 | 显式 `direction`;失真 pass=`E<=t`、worst=max、p90=中心内 quantile(.9) 后汇总;三个分母分列 | `test_subgraph_summary_distortion_known_answer`(E=[.3,.4,.8],t=.5 → 2/3、0、.8、.72) |
| 11 | 碰撞/重复单一口径 | 只报一个 collision | 前三码与完整(含消歧码)分别报,熵标 nats/bits | `test_collision_and_duplicate_prefix_vs_full` |
| 12 | SID 无校验 | 无长度/取值/缺失/重复检查 | `metrics.sid_audit` | `test_sid_audit_reports_every_defect` |
| 13 | join 不严格 | 只查 shape 与缺失 | shape/dtype/重复/缺失/额外/非有限/图版本全查,按 pair id 对齐 | `test_join_rejects_every_bad_input` |
| 14 | manifest 无版本与用途 | 无 graph_version/legal_use/依赖哈希 | `entry_v2` + `audit_manifest` + `assert_legal_use`;`verify_manifest` 只校验不写 | `test_legal_use_is_enforced`、`test_verify_does_not_write_and_keeps_the_expectation` |
| 15 | PPR 只是 256 个探针 | 探针冒充全 bank | `ppr.solve_selected` 稀疏 LU 精确求解,覆盖每个候选源,残差上报 | `test_mass_conserves_with_a_dangling_row` |
| 16 | 可见图没有曲率 | #236 曲率是**全量图、无向**的,不能移植 | 可见图上自建固定参考集 + 精确 Ollivier–Ricci(小规模运输 LP) | `test_orc_known_answers`(路径 0、三角 .5、方环 0) |

## 3. 几何口径更正(影响后续所有臂)

v1 的 κ-立体算子在球面与双曲两个分支使用了**不同的 2 倍约定**,且 `k=0` 返回普通欧氏范数:

```
v1: 球面 d(0,exp0(v)) = |v| ; 双曲 = 2|v| ; k=0 分支 = |x−y|   ← 与曲率族相差 2 倍
v2: 球面 = 双曲 = 2|v| ; k=0 = 2|x−y| = lim_{k→0} d_k          ← 连续
```

`k ≠ 0` 的距离数值不变,变的是球面的 exp/log 与 k=0 分支。断言:`d(0,expmap0(v)) = 2|v|`(所有 k)与 `lim_{k→0} d_k = 2|x−y|`;球面注入半径为 `πR`,即 `|v| < πR/2`。

## 4. TIGER SID 来源(只读,无训练)

| 产物 | sha256(前 16) | 碰撞(前三码) | 重复份额 | 结论 |
|---|---|---|---|---|
| `item_sids.json` | `025ad8bd5809ce62` | 0.169276 | 0.265303 | **Stage3 实际读取**(日志 `code_path_sha256=025ad8bd…`),来自 3000 epoch 那一轮 |
| `sids_for_hgrec.npy` | `6f2682c18d20ba88` | 0.110384 | 0.184813 | 与 checkpoint 同属**另一轮**(300 epoch),Stage3 从未读取 |
| `rqvae_best.pth` | `af8efc29a8bcbbe5` | — | — | 自带 `epoch=300, collision=0.11038` → 与 NPY 同轮 |
| 只读重导 | 新目录 `tiger_reexport/` | 0.109936 | 0.184244 | 与 NPY 仅 **115 行**不同,差异集中在第三码与消歧码(并列导致的 tie) |

重建误差(冻结 checkpoint 只读前向):`reconstruction_mse = 9.61e-05`(768 维输入,单位为平方误差)。

**结论**:冻结臂目录里混了两轮产物。已确认的基线是 `item_sids.json`;checkpoint 与 NPY 不能重建它,因此**不得**用该 checkpoint 重导去代表 Stage3 读过的表。历史 Recall 的绑定:Stage3 日志记录了 `code_path_sha256`,故 TIGER 的 `test_R@10` 可绑定到 JSON;其它臂若无该记录则标 NA。

**更正 v1 的错误结论**:v1 的 INVENTORY 说 §63 记录的 `l3_collision=0.2653`「不可复算」。这是错的:该记录的方法学写的是"item-weighted share of repeated three-token SIDs",即**重复商品份额** = 0.265303,而门禁口径 `1 − distinct/items = 0.169276` 同样正确——两者共用 "collision" 一词才是缺陷。已在 v1 INVENTORY 加更正注记。

## 5. v2 冻结资产

`results/research/infra246/assets/manifest.json`:**47 项**,逐项 sha256 + schema/shape/dtype/graph_version/legal_use/version/purpose/command/源哈希/码哈希/依赖,`verify_assets.py --version v2` 复核 **47/47**。

| 组 | 内容 | 规模 |
|---|---|---|
| 图 | 4 个划分 × 正向/反向/无向 + `held_out_support` + `split_users` + `stats.json` | 可见 291,954 边、评估 33,971 边;留出无向对 **62,324**;用户 45,951/5,744/5,744(80/10/10),两两交集 0、并集完整 |
| ego | 完整一跳、预算采样(cap 32)、全量图描述版 | 中心索引显式、hop 与遍历深度分列、in/out/mutual 三个 mask |
| motif | 3 节点**全量**计数 + 抽样实例(3 节点 59,385、4 节点 4,549) | 全量邻接对 28,391,948;覆盖 98.94% 商品 |
| pair bank | 训练 5 组 × 12,000;严格评估 2×5 组(每组 2,726–2,938);宽松评估 2×5 组(每组 60–274);探针 62 | 阻断无向对 85,395;严格组与训练 bank/探针交集 **0**;宽松组与训练集共享 52 对(=「用户隔离但边已见」的定义) |
| 距离 | 可见图 BFS 缓存(2,000 参考根) | 不可达 0.49% |
| 曲率 | **本轮未生成资产**(owner 指示;模块与已知答案测试已交付) | 补齐命令:`python3 tools/graph_infra/build_assets_v2.py curvature` |

隔离审计(`isolation_audit.json`):严格评估 key 与训练 bank 交集 0、与探针交集 0、宽松组与严格组交集 0;跨组多标签(同一对出现在两个关系组)共 19 对,已落盘并声明为「关系类型标签,组内不重复加权」。

## 6. 验证

| 检查 | 结果 |
|---|---|
| v2 验收测试 | 9 个套件,见 `VERDICT.md` 验收表 |
| 变异测试 | 14 个变异,判定=未变异副本先通过 + 变异副本出现指定 `FAIL` 行,崩溃不计 → **14/14** |
| 分模块验证 | `MODULE_CHECKS.json`,每个模块在**真实数据**上重算并给出 verdict |
| 真实资产隔离审计 | `test_v2_assets_real.py`:manifest、逐序列独立复算、用户集合、留出支撑、bank 隔离、ego 与 CSR 一致、motif 复算、距离 mask、曲率报告 |

## 7. 未做与理由

* **不训练**:issue 明令禁止;所有验收指标可由已有产物 + checkpoint 只读前向复算。
* **不重导覆盖历史产物**:重导写到新目录 `tiger_reexport/`。
* 4 节点 Graphlet 为**抽样**(每中心 ≤10 个实例,声明覆盖率),3 节点计数为**全量**(42.2M 邻接对)。
* 可见图曲率为**抽样**(固定参考集 + 每节点 ≤3 条边),缺失节点单列,未做度数保序零模型。
* 关系强度 Spearman 使用真实冻结图分数(`two_hop_count`、`ppr`、`jaccard_out`、`hop_full_visible`)与 SID 前缀共享,常数输入返回 NA;Jaccard 与 hop=2 的**定义性**关系不作为科学结论。
