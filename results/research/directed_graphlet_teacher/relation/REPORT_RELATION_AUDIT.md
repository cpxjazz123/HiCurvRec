# 现有 relation 目标能否作为结构监督？—— 数值审计（v2，含一次撤回）

**问题**：不重新生成 Teacher，先检查现有 128 维 `relation`（C_residual GNN 的商品级状态）是否
(1) 稳定、(2) 比纯语义 768 维多出真实结构信息、(3) 能被只看 768 维的离散 SID 保留。

**产物**：`audit_relation.json`（全部数值 + 输入 sha256）、`audit_relation.py`（只读，不训练）、
`audit_negative_validation.py`（独立重算负例分数）、`../tests/test_relation_audit.py`（19 条度量自检）。

## 撤回声明（v1 → v2）

v1 在负例一侧算错了：代码拿**正边分数数组**按**负例商品 id** 取值

```python
value = pair_cosine(relation, edge)        # 每条正边一个分数
auc = auc_score(value, value[negative])    # 错：value[negative] 是别的正边的分数
```

正确的对比是同一锚点下 `cos(h_anchor, h_positive)` 对 `cos(h_anchor, h_negative)`。因此 v1 的
relation / semantics / residual 余弦 AUC 及其区间**全部作废**，不能作为“结构信息有无”的证据。
v1 的 `Jaccard = 0.270` 也不该读成“27% 近邻重合”：Jaccard 以并集为分母，对应的重合比例是
`j/(1+j)`。v2 直接报告重合比例。

v2 的每条表示都用**同一批三元组**打分：负例不再按商品 id 索引，`usable_triplets` 同时要求
锚点、正例、负例都覆盖；负例分数另有独立实现（`gather + einsum`，不经被审计的辅助函数）逐元素复算，
`negative_score_cross_check` 里记录两次实现的最大绝对差与“旧读法”的偏差量。

## 0. 身份与来源

| 项 | 值 |
|---|---|
| relation 目标 | `teacher/structure/v3/structure_targets_v3.npz`，sha256 `7af2d165…`，(24587,128)，已 L2 归一 |
| 生成路径 | `C_residual_edge_main_seed226.pth` → `variant.encode(features, adjacency)`，需图 |
| 复核 | 重算 seed226 状态与表内 relation 余弦 = **1.000** |
| `teacher_structure_v3.pth` | 是**不含图的 768→目标 carrier**，meta 指向 seed226 checkpoint（sha 一致）；graph/split sha 与契约一致 |
| relation 覆盖 | 24094 / 24587（`coverage_neighbour`）；own-orbit 覆盖仅 12266 |
| 三元组 | 46586 个 held-out 边对；三方都覆盖的可用三元组见 `inputs.triplets_usable` |

## 1. 稳定性

| 指标（seed226 vs seed227） | 值 |
|---|---|
| 原始逐商品余弦 | 0.071（整体旋转所致，此值不说明稳定性） |
| Procrustes 对齐后余弦 | **0.962**（中位 0.967） |
| top-10 近邻 Jaccard | 0.270 |
| top-10 **重合比例**（共享/10） | 见 `stability.top10_overlap_fraction_mean` |

读法：逐点坐标在同一几何的旋转下稳定；近邻集合的绝对重合率由重合比例给出，而不是 Jaccard。

## 2. 是否比语义多出结构信息

见 `pair_auc`（同一三元组上的 anchor–positive vs anchor–negative 余弦 AUC，含 bootstrap 区间）、
`paired_delta_vs_relation`（同一行上的配对 AUC 差），以及 `graph_precision.points`（2000 探针、
precision@10，`distractors_per_item` 是评估切分里每个购买项对应一个干扰项所隐含的非结构基线：
约 1/平均度 的商品会被抽中，10 次近邻里“碰到一个结构邻居”本身就有约 `1-(1-p)^10` 的量级）。

`ridge` 一并给出：训练内 R²、池化留出 R²、**逐商品留出 R² 的中位数**与超过 0.5 的商品比例。
池化 R² 高可能只由少数主方向撑起；恢复整体方差与恢复边判别所需信息不是一回事，因此这里的
逐商品统计只是**容量必要性**的参考，不构成“768 足够承载 Teacher 分数”的推断。

## 3. 只看 768 维的离散 SID 能否保留它

`sid_retention` 逐步报告：`prefix_auc`（与 §2 同一批三元组上的前缀长度 AUC）、
`first_code_purity_relation_top` 与 `first_code_purity_semantics_top`（各自 top-10 近邻落在同一首码的比例）、
`first_code_chance`（该首码在本臂码表中的占比）。两个纯度并排才有意义：前者度量“SID 是否保留
relation 近邻”，后者度量“SID 保留语义近邻”。前者低于后者时，需要先看 §2：若 relation 余弦本身
不指向真实边，这个差不是损失。

## 4. 裁决（v2 的口径）

- **可以说**：在同一批 45869 组三元组上，relation 余弦把锚点–正例与锚点–负例分开的 AUC 是
  **0.739**，语义 768 维是 **0.610**，去掉 768 可线性预测部分后的残差仍有 **0.566**；
  配对 bootstrap 的差是 −0.130 与 −0.174（均不含 0，方向一致）。两个 seed 各自独立测都是 0.739。
  也就是说：**语义余弦本身几乎没有正负间隔，relation 余弦有；它的剩余部分也仍有间隔。**
- **可以说**：同锚点内部（691 个有 ≥10 条 held-out 边的锚点）relation 0.692、语义 0.606，
  不是少数大度商品平均出来的。
- **不可以说**：v1 的“0.496 / 0.503 / 0.500 = 接近随机”“kNN 监督就是噪声”。这些数字来自把负例
  当数组下标读取的错算，已在 JSON 的 `withdrawn` 块中撤回。
- **方向对照的读数**：负例本身也是锚点的 held-out 邻居时（42676 组），在两个空间各自打分后
  relation 的 forward/reverse 胜率 0.732/0.731（swing −0.004，95% [−0.008, +0.001]），
  语义 0.641/0.635（swing +0.001，[−0.000, +0.003]），残差 0.564/0.570（swing −0.008，
  [−0.013, −0.003]）。前反向几乎相同且都不等于 0.5：**该子集上不存在可分离的方向证据**，
  两跳边同样靠近，所以这条对照**不能**把 relation 的优势归因于“图特有的结构”或排除“相似度延伸”。
  要区分二者需要换实验（只留负例与锚点无共同邻居的三元组，或改变负例采样分布）。
- **结论口径**：relation 余弦**含真实结构信号**（对真实 held-out 边有 0.739 的判别力，显著高于语义
  与其残差），但“它比语义多的部分是图特有的还是相似度的延伸”仍然**未判定**。

## 复现

```
cd results/research/directed_graphlet_teacher
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 relation/audit_relation.py
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest tests/test_relation_audit.py -q
```
