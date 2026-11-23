# RUN_LOG — issue #246 (Infra v2)

| field | value |
|---|---|
| issue | #246 — Infra v2: 修复数据隔离、精确有向 Graphlet 实例、结构评估与 TIGER SID 来源校验 |
| body sha256 | `b457e25698d34d1188a24e29694c71217bc17135c486c796808fe26a0e5b5362` |
| pre-existing HEAD | `dbe8e354` (issue #245 delivery) |
| interpreter | Stage2/机制测试 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` (CLAUDE.md §7) |
| scope | `tools/graph_infra/` only + `results/research/infra246/`; historical assets read-only |

## 是否需要训练:否

issue 原文：「本轮只做 Infra，不训练 Teacher/GNN/RQ-VAE/Stage3，不重训 TIGER，不扫参、不多种子。允许对现有冻结
checkpoint 做只读推理、重导和指标复算」。§8 再述「不得擅自创建其他 Issue 或启动任何训练」。
验收指标全部可由**已有产物 + 只读前向**复算，因此按 skill §4 判为「分析型 / 数据与契约型」，不进入训练步骤。

## 冻结验收清单(C1–C37)

| ID | issue 要求 | 判据(机械复算) | 阈值 | 数据 / 口径 |
|---|---|---|---|---|
| C1 | 复核原始 item ID、行顺序、源哈希，用实际 ID 集合而非 max+1 | `assets.item_index()` 从 `items.parquet` 读 ID 集合并与 `sentence_t5.npy` 行数、`item_sids.json` 键比较 | 集合相等; 行数相等; 三份源 sha256 记录 | `results/stage0_build_parquet/items.parquet`, `stage1_GeneEmbedding/output/sentence_t5.npy` |
| C2 | 独立逐序列统计核对原图边与计数,不只比总边数; 记录实际自环丢弃数 | `crosscheck.sequence_edges()`(独立实现)与 CSR 逐边比对 | 对称差 = 0; 自环计数与实际一致 | `train.parquet` 全序列 |
| C3 | train 内冻结 传播可见/结构验证/结构测试 三个用户集合,固定 seed 比例,开工前锁定; 保存真实用户清单; 交集为空、并集完整; 三图计数和=完整图 | `splits.audit()` | 两两交集 = 0; 并集 = 全部用户; 边权总和相等 | seed 冻结于 `tools/graph_infra/config.py`; 与 Stage3 val/test 无关 |
| C4 | 完整图只作描述; 所有泛化资产从可见图生成; full-description 与 visible-train 分开; manifest 机器校验合法用途; 禁止跨版本 join | `assets.assert_legal_use(manifest,name,'generalisation')` | 违规调用抛错; manifest 含 `graph_version`+`legal_use` | v2 manifest |
| C5 | 修复 `eval_only = eval - eval.multiply(visible)` 加权抵消 bug,先转布尔/集合差再求双向并集 | `isolation.held_out_support()` 与已知答案 | 可见反向计数>1、留出正向存在时 blocked 必含该无向对 | 合成反例 + 真实图 |
| C6 | 区分 user-disjoint 与 pair-disjoint; 严格评估 bank 的无向规范 key 从训练 bank/梯度探针/训练采样器排除(双向); 用户隔离但已见边的评估另列 | `isolation.audit_banks()` | 严格评估组 ∩ 训练 side = 0; 宽松组单独命名并计数 | v2 banks |
| C7 | 验证/测试候选之间去重或明确独立口径; 逐项审计实际交集 | `isolation.audit_banks()` 的 `pair_overlap` | 严格组两两交集 = 0(或声明重叠数) | v2 banks |
| C8 | 共同邻域/两跳任务有独立标签证据与路径隔离; 未观察≠负例 | `pairs.label_evidence()` 记录每条 pair 的见证节点/路径; 字段 `label_evidence` 非空 | 见证路径独立于被评估图; `negative` 字段不出现 | v2 banks |
| C9 | 冻结 train/val/test 真实 pair ID、组别、合法源图版本; 记录跨组重复/多标签; 不重复加权而不声明 | bank 文件含 `pair_id`, `group`, `graph_version`, `cross_group` 列 | 文件存在且行数一致; 重复计数可复算 | v2 banks |
| C10 | 修复 `ego_induced` 用另一随机采样判定方向; 必须读真实 CSR; 互惠邻居同时有 in/out 两个 mask | `topology.root_roles_csr()` 与 CSR membership 逐项比对 | 不一致 = 0 | 真实图 + 合成反例 |
| C11 | 根节点用明确 `root_local_idx`,不要求 ID 最小或首行 | `ego_*.npz` 的 `root_local` 指向真实 root | 全部块成立 | v2 ego |
| C12 | 严格区分最短 hop 标签与采样遍历深度,若不同分别保留 | ego 含 `hop`(真实最短,按 CSR 重算)与 `traversal_depth` | 两字段独立存在; 至少一个块二者不同(否则说明采样未截断) | v2 ego |
| C13 | 完整一跳/两跳集合与诱导边(可压缩或确定性按需生成)+ 冻结预算采样版; cap 语义(含根?); 截断/覆盖率按邻域算 | `topology.ego_full()` 与 `cap_audit()` | 采样版实际节点数 ≤ cap+1; 覆盖率按各商品真实邻域计算并落盘 | v2 ego |
| C14 | 保留完整边/hop/root/in-out-mutual/空邻域/node-edge mask/偏移; `batch_egos` 输出边/边 mask/特征 | `batch_egos` 返回 `edge_index`,`edge_mask`,`features` | 键存在且形状自洽; mask 与 padding 一致 | 单元测试 |
| C15 | visible 与 full 两版本; 每商品固定采样与图哈希; 不隐式换样 | 两版 ego 文件 + manifest `graph_version`; `sample_seed` 常量 | 两版明文可区分; 采样哈希写入 manifest | v2 ego |
| C16 | 全量检验 root/唯一性/边索引/方向 mask 与 CSR 一致; 抽样按独立邻接核对全部诱导边; 反例覆盖根非最小、高度数采样、互惠、孤立、非树边、方向反转 | `test_ego_full.py` + `test_isolation.py` | 全量不一致 = 0; 抽样不一致 = 0 | v2 ego |
| C17 | 修复 fork/merge 存在性定义:存在一对不相连后继/前驱即成立; 契约区分宽松谓词/严格诱导有向 Graphlet/无向三角 | `graphlet.fork_relaxed()` 三后继两两相邻之外仍成立 | 三后继 2 相邻 1 不相邻 → fork=True | 合成反例 |
| C18 | 可复用 3/4 节点有向 Graphlet 与 Orbit 分类; 精确定义与节点角色; canonical adjacency/isomorphism+automorphism; 小图穷举验证; 历史口径另存 | `graphlet.canonical()` 对 6 个置换取最小; `orbits()` 给出 automorphism 轨道 | 64 个 3 节点有向型全部可分类; 穷举与逐中心枚举一致 | 合成 + 真实小图 |
| C19 | 冻结实际局部子图实例(graphlet_id/node_ids/induced edges/family/orbit/center/graph_version)+ 每商品次数/角色/重叠索引; 四节点抽样声明覆盖; 全量统计与抽样实例分开; 匹配评估 bank; 三后继反例; 角色跨实例不丢 | `motif_instances.npz` + `motif_stats.json` | 实例可逐条复算; 抽样覆盖率落盘; `graphlet_id` 与 canonical 一致 | v2 motifs |
| C20 | 逐对全图可见无向 hop、有向可达/hop、局部子图 hop(不同字段)、不可达 mask | distance npz 每 bank 一行 4 字段 | 行数 = bank 行数; 不可达用 mask 不用 0 | v2 distance |
| C21 | 直接转移频率、共同前驱/后继 Jaccard 与 Bhattacharyya(空=missing)、两跳路径数、多跳 PPR **覆盖全部固定候选**(非 256 探针); 固定 restart/收敛/截断并报误差与覆盖; 度数匹配逐 pair 验证; K-hop 组区分 | `distance` npz 的 `ppr_*` 行数 = bank 行数; `residual` 与 `coverage` 落盘 | 残差 < 1e-6; 覆盖 = 全部唯一 source | v2 distance |
| C22 | 有向 pair 顺序与无向 key 分字段 | bank 含 `pair_directed` 与 `pair_undirected` | 两列均存在且方向信息不丢 | v2 banks |
| C23 | 可见图曲率: 合法固定参考集 + 测地三元组代理; 记录参考数/抽样/单位/分布/缺失; 不用评估标签; 几何算子边界复测 | `curvature_visible.json` + `test_geometry_bounds.py` | 参考数 > 0; 单位与抽样覆盖落盘; 近对跖/κ→0/边界断言通过 | v2 curvature |
| C24 | `subgraph_summary` 显式 direction; 失真 pass=E<=t、worst=max、p90=中心内 quantile 后聚合; 三个比例分开; 无有效子图中心与不可达分母另列; 已知答案 E=[.3,.4,.8],t=.5 → 2/3,0,.8,.72 | `test_metrics.py::test_subgraph_summary_distortion_known_answer` | 四个数字精确匹配 | 单元 |
| C25 | 逐 pair/子图/中心可复算记录; 按结构/度数/子图数分层 | 记录文件 + `metrics.stratified` | 文件存在且可重算 | v2 |
| C26 | 修复 prefix@k 与碰撞/重复(前三码 vs 完整含消歧码分开); 校验 SID 长度/取值/行映射/缺失/重复; entropy 标 nats/bits | `metrics.sid_audit()` | 非法长度/取值/缺失/重复均被报出; entropy 带单位 | TIGER 重导 SID |
| C27 | 关系强度 Spearman 用真实冻结图分数; 常数 NA; ties 正确; 不把 Jaccard hop=2 定义关系当科学结论 | `metrics.spearman` + bank 内 `relation_strength` 列 | 常数 NA; ties 手算一致; 文档写明定义性关系 | v2 |
| C28 | `join_scores`/manifest 严格校验 shape/dtype/重复(含 bank)/缺失/额外/非有限/图版本/依赖哈希/源码版本; 允许按 ID 重排并测试 | `assets.join_scores_v2()` | 每类非法输入各抛一次; 重排后逐对一致 | 单元 |
| C29 | manifest 路径可移植; 校验失败不得自动重建覆盖; 重建写新版本或 staging | `assets.verify_manifest()` 失败抛错且不写文件; 路径为仓库相对 | 相对路径 = 全部; 失败后文件哈希不变 | 单元 |
| C30 | NPY schema 核查(padding/保留 token/层偏移/消歧/ID 映射/顺序),不盲比前三列; 查历史 loader/config/实际读取/哈希/checkpoint 对应 | `tiger_audit.json` 的 `npy_schema` 段 | 每列语义有证据; 行映射可复算 | TIGER 产物 |
| C31 | 从冻结 checkpoint 只读重导到新目录; 验证 checkpoint→SID→Stage3 读取一致、映射、量化重建; 报告两旧文件各自碰撞/不同三码/重复率 | `tiger_audit.json` + `reexport/` | 重导 SID 与 checkpoint 逐项一致; 两旧文件指标分列 | 新目录 |
| C32 | 无法绑定 SID 哈希的历史 Recall 明确 NA; 保留唯一合法基线 manifest 或分开命名不一致产物 | `tiger_audit.json` 的 `recall_binding` | 每条 recall 有 `bound|NA` 与理由 | 历史 Stage3 |
| C33 | 用确认 SID 在统一独立 bank 复算 prefix@1/2/3、关系 Spearman、结构保真与多子图; 只读前向复算 z/q/recon MSE; 保留逐对记录 | `tiger_baseline_v2.json` | 每个指标有值与口径; 逐对记录可复算 | v2 |
| C34 | 共享模块更新而非私有复制; v2 manifest 含数据源/配置/代码/schema/shape/dtype/hash/用途/版本/命令/依赖; 区分已完成与未来依赖模型项 | manifest v2 schema 校验 | 必填字段齐全 | v2 |
| C35 | 先写失败测试再修复; 真实数据全量与隔离; 独立复算不读聚合; 每个 P0 反例覆盖; 正向通过 + 变异被指定断言捕获 | 测试红→绿日志 + MUTATIONS.md | 每个 P0 至少 1 个反例; 变异全捕获 | 本目录 |
| C36 | 核验实际未来使用入口读取 v2 而非 v1; 提交可复现构建/验证入口与 REUSE.md | `verify_assets.py --version v2` + `build_v2.py` | 命令可执行且校验 v2 | `tools/graph_infra/` |
| C37 | 交付 CONTRACT/INFRA/REPORT/VERDICT + 完整验收表; 关键节点评论; 不建新 Issue、不训练 | 文件存在; 每条 C 有 PASS 证据 + `path:sha256` | 无 NAS/FAIL 遗留 | 本目录 |

## 关键节点评论记录

| # | 节点 | 评论文件 |
|---|---|---|
| 1 | 冻结清单 + 是否训练判定 | `COMMENT_01.md` |
| 2 | 反例先行(红) | `COMMENT_02.md` |
| 3 | 生产修复 + 变异 + 模块验证 | `COMMENT_03.md` |
| 4 | v2 资产冻结 + 隔离审计 | `COMMENT_04.md` |
| 5 | TIGER 来源校验 + 基线复算 | `COMMENT_05.md` |
| 6 | 结论对账 / 关闭 | `COMMENT_06.md` |
