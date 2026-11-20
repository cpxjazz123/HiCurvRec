# 冻结的多角色行为图结构 Teacher(Issue #229)

日期:2026-11-21
版本:`structure-teacher-v1`,build SHA `ba66a98a0865681be05353ca9ca8dc301da3ca2a08806f182b6f6fe5a9ea6607`
实现:`structure_teacher.py`;契约:`teacher/structure/teacher_structure_contract.json`;
机器可读:`structure_teacher_report.json`;测试:`tests/test_structure_teacher.py`(10 项)。

## 0. 交付物

| 产物 | 内容 | SHA-256(前 12) |
|---|---|---|
| `teacher/structure/structure_targets_v1.npz` | 每商品监督目标表(24,587 行) | `c0962cd20f69` |
| `teacher/structure/teacher_structure_v1.pth` | 冻结 Teacher:图编码器 + 载体头(`768 → relation 128 + orbit 376`) | `ec465a7d81e7` |
| `teacher/structure/teacher_structure_contract.json` | 输入边界、目标语义、损失、能力判决 | `b970eaf2d8f8` |
| `teacher/structure/teacher_structure_manifest.json` | 上述文件的大小与 SHA | — |
| `structure_teacher_report.json` | 全部数字与本报告的 Done 对照 | — |

目标表的每一行对应 Stage1/Stage2 商品宇宙中的同一商品 id(`item_alignment` 写进契约)。

## 1. 输入边界(硬约束)

| 约束 | 实测 |
|---|---|
| RQ-VAE Encoder 输入是否改变 | **否**;仍只有 768 维 Stage1 商品向量 |
| Teacher 推理是否需要邻接矩阵 | **否**;冻结后只用商品 id 查目标表,或让载体头吃 768 维向量 |
| 是否用于 SID 推理 | **否**;只作为 RQ-VAE **训练损失**的监督 |
| 图在何时被使用 | 仅在构建目标表时读取 TRAIN-only 消息图 |
| 三层 SID 协议 | 未改动 |

`orbit_loss` / `relation_ranking_loss` / `relation_distill_loss` 的签名里没有任何
`adjacency` / `graph` / `edge` 参数(有测试断言),监督路径无法偷偷读图。

## 2. 监督目标的语义与统计

| 字段 | 形状 | 语义 |
|---|---|---|
| `orbit_target` | (24,587, 376) | 有向 graphlet orbit 分布;同一商品可同时有多个角色 |
| `relation` | (24,587, 128) | 商品级关系向量;其两两余弦是**商品间结构关系**目标 |
| `coverage` | (24,587,) | 观测图是否支持 orbit 监督(1/0) |
| `confidence` | (24,587,) | 覆盖 × 对数度数证据;0 表示图无法支持 |
| `positive_pairs` / `negative_pairs` | (32,637, 2) / (32,637,) | 关系排序损失的正例对与度数匹配负例 |

| 指标 | 实测 |
|---|---|
| 覆盖商品 | **12,266 / 24,587 = 49.89%** |
| 每覆盖商品平均角色数 | **2.08** |
| 拥有 >1 个角色的商品比例 | **46.11%** |
| 覆盖商品的平均置信度 | 0.944 |
| 置信度为 0(无图证据)的商品 | 12,321(明确标记,不捏造标签) |
| 关系向量两两余弦(1,024 抽样) | 均值 0.200;近重复(>0.999)比例 **0.000** |
| 两个独立 seed 的关系排序一致性 | **Spearman 0.967**,符号一致率 0.9385 |

## 3. 能力验收

### 3.1 识别(Recognition)—— 成立,但必须按口径读

| 任务 | 数字 | 来源/口径 |
|---|---:|---|
| 给真实诱导邻接的 subtype macro-F1 | **0.7121** | #221 held-out 产物,`directed_graphlet_role/head_pretrained_directed.json` |
| 给真实诱导邻接的 orbit macro-F1 | **0.6101** | 同上 |
| 不给邻接的 subtype macro-F1 | 0.0206 | #221 冻结 z 上的预测器 |
| 不给邻接的 orbit macro-F1 | 0.3440 | 同上 |
| 冻结 C 节点态 → orbit 线性探针 CE | 3.5488 | 常数基线 3.5163,**比常数差** |
| 冻结 D(打乱图)节点态 → orbit 探针 CE | 3.5800 | 同上 |

结论:**“给子图邻接识别 Type/Orbit”成立**;**“仅凭商品表示识别 orbit”不成立**。
后者的失败被明确记录,不用前者的 0.71 替代。

### 3.2 预测(Prediction)—— 边存在性成立,方向未成立

严格无泄漏测试对(46,586 正例 + 度数匹配负例,目标对及反向均不在消息图中):

| 变体 | test AUC(seed 226/227) | 与 C 的配对 bootstrap |
|---|---:|---|
| A 语义投影(768→128 + 同 head) | 0.7108 / 0.7089 | C−A **+0.0395 [0.0372, 0.0418]** |
| C 残差 GNN(真图) | **0.7473 / 0.7475** | — |
| D 度数保持重接线 | 0.7141 / 0.7120 | C−D **+0.0343 [0.0322, 0.0363]** |
| E 同架构无消息 | 0.7201 / 0.7191 | C−E +0.0271 / +0.0283 |
| 40 次真实更新的 #227 架构 | 0.5039 | 优化不足的直接对照 |

方向任务(单边 held-out 对,两类平衡):A 0.5002/0.5000、B 0.5079/0.5099、
C 0.5111/0.5076、D 0.4984/0.4979、E 0.4970/0.5007;真实多数类 **0.5016**,
预测频数上限 0.53–0.61。**没有任何变体超过多数类 + 0.02 ⇒ 方向能力声明为未成立。**

### 3.3 商品级可承载 —— 关系目标成立,orbit 目标成立但“可被固定表示承载”未成立

- **关系目标**:冻结载体(768 → 128)在**未见商品**上的余弦 **0.8465**;两个独立 seed 的
  关系排序 Spearman **0.967**;商品间目标差异明显(近重复 0.000)。
- **orbit 目标**:目标本身是多角色、可复核、非退化的(2.08 角色/商品);
  但 768 维语义到 orbit 的载体 CE **3.4956**,常数基线 **3.4913** ⇒ **不可由固定 Stage1 表示承载**。
  该限制已写进契约与 `structure_teacher_report.json` 的 `not_established`。

### 3.4 监督可交付 —— 成立

- 目标表是冻结的、按商品 id 对齐的、可从 SHA 复核的。
- 训练损失已实现并被测试覆盖:`orbit_loss`(仅覆盖商品)、`relation_ranking_loss`
  (正例对 vs 度数匹配负例)、`relation_distill_loss`(匹配 Teacher 的两两几何)。
- RQ-VAE 不需要邻接、不需要 Teacher hidden、不需要改推理协议。

## 4. 与失败相关的事实(不得省略)

1. 方向未成立;不把常数预测或 0.511 当作方向建模。
2. orbit 不能被 768 维语义的线性载体复现;这限制了“结构信号已进入固定表示”的强度,
   但目标表与关系目标不受影响。
3. 本 Issue **未训练** RQ-VAE / Stage2 / Stage3,因此没有任何 SID 碰撞率或 Recall 声明。
4. 40-step 对照在机会水平,说明 #227 的“图无信号”结论只适用于那 40 次更新。

## 5. Done 对照

| #229 要求 | 状态 | 证据 |
|---|---|---|
| 识别成立(多类 graphlet + 实例 orbit,多角色) | **PASS** | §3.1 + 2.08 角色/商品、46.1% 多角色 |
| 预测成立(至少一项无真值结构的任务,稳定超过语义与拓扑对照) | **PASS** | §3.2,C−A/C−D 双 seed CI 均 > 0,效应 0.034–0.040 |
| 商品级可承载(多角色 + 商品间关系 + 区分性/稳定性/覆盖) | **PASS** | §3.3,关系载体 0.8465、Spearman 0.967、覆盖 49.89% |
| 监督可交付(冻结 Teacher → 非退化目标;RQ-VAE 无额外输入) | **PASS** | §3.4,契约 + 损失 + 测试 |
| 证据可信(无泄漏/伪负例/错位;对照、重复性、失败记录) | **PASS** | §2 + #228 §2 的 0 违规 + §4 |
| 产物完整(权重/契约/版本校验/报告) | **PASS** | §0,manifest 的 SHA 全部可复核 |
| 方向能力 | **未成立,已声明** | §3.2 |
| SID / Recall 改善 | **不在本 Issue 范围** | 契约禁止提前宣称 |

## 6. 复现

```bash
# 构建(读取 TRAIN 图,训练冻结目标;约 1 分钟,GPU)
python3.9 structure_teacher.py            # build_structure_teacher(device=cuda:0, seed=226)
# 契约回归
python3.9 -m pytest tests/test_structure_teacher.py -q
```

前置:Stage1 `sentence_t5.npy`、TRAIN 有向图 `behavior_graph_directed/graph.npz`、
以及 #228 的 `teacher/opt/C_residual_edge_main_seed226.pth`(SHA 记在契约的 `build` 块)。
