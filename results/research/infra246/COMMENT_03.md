## 节点 3:v2 资产冻结与隔离审计

### 冻结内容(`results/research/infra246/assets/manifest.json`,逐项 sha256 + schema/shape/dtype/graph_version/legal_use/version/purpose/command/源哈希/码哈希/依赖)

| 组 | 内容 | 关键量 |
|---|---|---|
| 图 | 正向/反向/无向 × full_description、visible、struct_val、struct_test,加 `held_out_support` 与 `split_users` | 可见图 291,954 边、评估图 33,971 边、留出支撑 **63,200** 条无向对;用户 51,695/5,744 两两交集空、并集全 |
| ego | `ego_full_1hop_visible`(完整)、`ego_sampled_visible`(cap 32)、`ego_sampled_full_description`(仅描述) | 中心索引显式、hop 标签与遍历深度分列、in/out/mutual 三个 mask |
| motif | `motif_counts.npz`(3 节点**全量**计数)、`motif_instances_visible.npz`(2,000 中心抽样 59,385 实例)、`motif_instances_4node_visible.npz`(500 中心 4,549 实例,声明为抽样) | 全量邻接对 28,391,948,98.94% 商品至少有一个实例;chain 12.1M、out_star 6.06M、in_star 5.94M、ff_triangle 435,582、fb_triangle 151,647 |
| pair bank | 5 组训练 bank + 5 组严格评估 bank + 5 组宽松评估 bank + 梯度探针 | 每组 12,000 对;镜像留出对按**无向双向**剔除(见各组 `dropped_by_held_out_blocking`);每对带全图无向 hop、有向 hop、子图 hop、两跳计数、Jaccard、Bhattacharyya、见证节点、**PPR** |
| 距离/传播 | 可见图 BFS 缓存(2,000 参考根)与 PPR(全部候选源一次求解,容差 1e-6,残差写进 manifest) | PPR 覆盖全部 bank 源,不再是 256 探针 |
| 曲率 | 可见图精确 Ollivier–Ricci(小规模运输 LP,邻域上限 64/侧,记录为近似) | 参考节点 2,000、抽样边、缺失节点单列、单位无量纲 |

### 隔离审计(`results/research/infra246/isolation_audit.json`)
- **user-disjoint**:`struct_val`/`struct_test` 用户不进可见图(集合级断言)。
- **pair-disjoint(严格)**:严格评估 bank 的无向 key 与训练 bank 的交集 = **0**,与梯度探针的交集 = **0**,与另一个评估划分的交集单独报告。
- **宽松组单列**:user-isolated 但边已见的对单独成组,不与严格组混用。
- 跨组多标签计数落盘(关系类型是标签而非互斥划分,组内不重复加权)。

### 分模块验证(`results/research/infra246/MODULE_CHECKS.json`)
12 个模块在**真实数据**上逐项重算并给出 verdict:graph_build(独立逐序列复算一致)、splits(并集/交集/边权相加)、isolation(留出支撑与训练 bank 零阻塞)、topology(role/hop/边错误均为 0)、graphlet(抽样中心与暴力枚举一致)、pairs(逐组校验 + 度数匹配违规 0)、bank_features(BFS 复核)、ppr(残差)、curvature(分布与缺失)、metrics(已知答案 + TIGER SID 审计)、assets(manifest 与 join 的七类非法输入)、geometry(四个 k 的往返恒等式)。

### 变异测试
`results/research/infra246/MUTATIONS_V2.md`:**14 个变异全部被指定断言捕获**(数值 5 / 结构 6 / 路由 3),判定为「未变异副本先通过 + 变异副本出现指定 `FAIL` 行」,崩溃不计。

下一步:TIGER 来源校验与基线复算(节点 4),随后结论对账与关闭。
