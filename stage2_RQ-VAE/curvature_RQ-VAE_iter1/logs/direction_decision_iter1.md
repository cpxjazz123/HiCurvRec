# iter1 方向裁决（Agent B direction-judge）

## 1. 裁决边界与输入

本裁决只依据以下项目内材料，不调用外部搜索或外部论文资料：

- `logs/lit_search_iter1.md`：Agent A 的文献检索结果与 P1/P2/P3 候选；
- `curvature_RQ-VAE.py`：当前训练实现；
- `curvature_config.py`：当前迭代配置与上一轮结果。

本文件只做方向评判，不修改训练代码，不修改 Stage3。

## 2. 上一轮失败根因

已知事实如下：

1. 当前实现已经使用全局 cyclic curvature：
   `c(t) = 0.3 + (1.0 - 0.3) * |sin(pi*t/50000)|`。
2. 训练与 SID 导出链路完成，raw SID 为 3 层，最终 collision extension 后为 4 token，且全表唯一；因此上一轮不能归因于 SID 导出失败或碰撞未消除。
3. 上一轮 Stage3 `test_R@10=0.05356987412733508`，低于 0.065 目标。
4. 因而更可信的失败根因是：全局、随时间变化的曲率只改变了距离几何和最近码字分配，却没有保证离散 assignment 保留推荐所需的 item-neighborhood 语义。SID 统计上的改善不等价于生成推荐下游改善。

裁决重点是优先修复“几何距离到离散码字边界”的失配，而不是再单纯扩大曲率振幅或叠加一个无关的 Stage1 正则项。

## 3. 评分规则

每项 1--5 分：

- **(a) 根因修复**：5 = 能直接针对上述失配；1 = 与根因只有间接关系。
- **(b) 2023+ 实证**：5 = Agent A 材料中有直接且任务高度匹配的实证；1 = 无材料支持。任务不匹配时即使论文有结果也不满分。
- **(c) cyclic 兼容性**：5 = 可直接复用现有 `c(t)`，不引入第二个曲率控制器；1 = 与现有 schedule 有明显不可辨识或分布冲突。
- **(d) Stage1 ceiling 风险**：此项按“风险低”计分，5 = 风险低，1 = 高风险；低分同时必须写出 WARNING。历史摘要显示纯 Stage1 曲率机制曾多次 downstream regression，因此不能把此项低分隐藏在总分中。
- **(e) 曲率相关性合规**：5 = 明确命中机制池的曲率/双曲/流形类别，且 Agent A 材料中的标题或摘要含相关几何关键词；1 = 主要是非曲率优化。

## 4. P1/P2/P3 评分表

| 候选 | 单一机制定义 | (a) 根因修复 | (b) 2023+ 实证（仅据 lit_search） | (c) 与 cyclic `c(t)` | (d) Stage1 ceiling 风险 | (e) 曲率相关性 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| **P1** | **双曲 MLR/决策边界式量化**：用双曲空间中的可学习多类决策边界参与 code assignment，而不是只按最近双曲距离分配 | **4/5**。直接学习 assignment boundary，可修复当前全局距离驱动的码字分配偏置；不能保证推荐语义完全恢复 | **3/5**。`HyperVQ`（2024）以 codebook collapse 为动机，报告判别性能和潜表示解耦改善；但没有 RQ-VAE-SID 或生成式推荐下游实证 | **3/5**。若 logits 使用每一步实际的现有 `c(t)`，可与 schedule 共用同一几何；若 MLR 在固定曲率下学习，则会出现训练/推理边界失配 | **2/5**。**WARNING：这是 Stage1 端的纯双曲 assignment/几何变更，历史上纯 Stage1 曲率变更有明显 downstream ceiling 与 regression 风险** | **5/5**。明确命中 hyperbolic/manifold geometry 与 distance-based assignment | **唯一推荐候选** |
| **P2** | **残差码本层内均衡与层间去冗余**：控制各层 code usage，减少 residual codebook 重复编码 | **3/5**。可改善 collapse/冗余，但不能直接修复“SID 统计改善而推荐语义下降”的核心失配 | **3/5**。`ERVQ`（2024）报告 codebook utilization 和层间多样性改善；`LETTER`/`CoST` 的推荐证据涉及语义或协同信号，但不等于纯均衡机制的直接证明 | **2/5**。强均衡可能破坏有用的非均匀粗粒度结构，额外梯度还可能与 cyclic 曲率竞争 | **2/5**。**WARNING：仍是 Stage1 码本优化改变，且容易把表面 usage 指标当成下游语义改善** | **1/5**。纯均衡/去冗余本身不命中曲率类别；若改为 CoST 式协同目标，又会引入非曲率 novelty 和协同信号 | **不合规，不列为 BACKUP** |
| **P3** | **按层级、residual 难度或 item 复杂度自适应曲率/量化粒度**：由 residual norm、局部密度或 assignment entropy 条件控制 `c` 或温度 | **4/5**。可解除全局曲率对所有 item/layer 一刀切造成的几何分配失配 | **4/5**。`Curvature-Adaptive Meta-Learning`（2023）在乘积流形任务上验证任务相关曲率更新；`Adaptive Discrete Communication Bottlenecks`（2023）验证复杂度相关的动态离散粒度；但都不是 RQ-VAE 生成推荐 | **2/5**。在已有 global `c(t)` 上再加 item/layer difficulty controller，会形成 difficulty→curvature→assignment→difficulty 的反馈与非平稳闭环 | **1/5**。**WARNING：高风险 Stage1 纯曲率变更，且比 P1 更容易改变训练/最终 SID 的条件分布；历史 downstream ceiling 风险最高** | **5/5**。明确命中 dynamic/per-layer curvature、manifold 与 curvature-adaptive 类别 | **BACKUP** |

### P1 的根因修复句

当前距离量化把随 `c(t)` 变化的几何距离直接当作离散语义边界；P1 让边界参数从数据中学习，并在每一步使用同一个现有 `c(t)`，因此直接针对“距离分配合理但推荐语义不保留”的失配。

### P2 的合规判定

P2 若只做 usage balancing，不属于曲率相关机制；若加入 `CoST`/`LETTER` 式协同或对比目标，则需要引入协同信号，并且不再是单一曲率 novelty。按本轮硬约束，P2 不能作为最终方向。

### P3 的 BACKUP 限制

P3 的文献支持和曲率相关性较强，但它实际上在现有 cyclic controller 之外再引入数据或层级 controller。只有在 P1 无法实现或梯度检查失败时，才可作为下一方向重新设计；本轮不同时实现 P1 与 P3。

## 5. 机制池合规快速审计

| 机制池方向 | 本轮判断 | 原因 |
|---|---|---|
| cyclic `c(t)` | 已用，不能作为本轮 novelty | 当前实现已经是 `c_min=0.3`、`c_max=1.0`、`period=50000` |
| layer-wise learned curvature | 包含在 P3 | 有 2023 曲率自适应实证，但与现有 global cyclic controller 冲突较大 |
| Lorentz / projective hyperbolic | 不推荐 | `lit_search_iter1.md` 没有给出与 RQ-VAE 生成推荐直接匹配的实证；属于高风险 Stage1 几何替换 |
| adaptive margin `alpha/c_l` | 不推荐 | 检索记录没有该具体机制的 2023+ 直接实证，且会新增 loss/目标尺度变化 |
| negative `c` | 不推荐 | 检索记录没有负曲率切换为球面分支的直接实证，数值与语义风险均未被材料覆盖 |
| mixed-curvature product manifold | 仅有间接支持 | `HiHPQ` 与曲率自适应材料可说明乘积流形可用，但没有推荐下游证据；本轮不引入额外流形分支 |
| temperature by `1/c` | 不推荐 | 会再造一个 assignment 温度控制器，和现有 cyclic schedule 发生不可辨识耦合；检索记录没有直接实证 |
| Riemannian Adam | 不推荐 | `Robust Hyperbolic Learning`（2024）支持稳定性/优化效率，但不直接修复离散 assignment 与推荐语义失配，也仍属 Stage1 端高风险变更 |
| Sinkhorn `epsilon=epsilon0/c` | 不推荐 | 检索记录没有该 c-dependent epsilon 的直接实证；主要改善分布/usage，不能直接证明下游 item-neighborhood 被保留 |

## 6. 唯一推荐

**推荐 P1：双曲 MLR/决策边界式量化（只实现这一项 novelty）。**

实施边界必须固定为：

1. 在现有双曲 residual quantizer 中，用当前训练步的 `c(t)` 计算 MLR assignment logits，替代“仅按最近双曲距离”的 assignment；不另加第二个曲率 schedule。
2. 不叠加 layer-wise learned curvature、adaptive margin、temperature、Sinkhorn epsilon、Riemannian Adam、diversity 或 contrastive loss。
3. 不修改 Stage3，不引入协同信号；推荐本身不要求 Stage3 改动，满足本轮合规性。
4. 在任何 GPU 训练前，必须先完成项目规定的单 ckpt、单 batch 梯度通路检查，确认 MLR 参数和总 loss 都有非零梯度。

推荐理由（不超过三行）：P1 直接改 assignment boundary，针对上一轮“距离几何变化但离散推荐语义失配”的根因；`HyperVQ`（2024）提供了 codebook-collapse/判别边界方向的 2023+ 后实证；它可以复用现有 `c(t)`，比 P3 少一个曲率控制器，也不需要 Stage3 协同信号。

## 7. 最终 WARNING 与裁决

**WARNING（必须保留）：P1 仍是 Stage1 端的纯双曲几何/assignment 变更，不能假设 SID usage 或碰撞指标改善就会带来 `test_R@10` 改善。历史摘要中纯 Stage1 曲率机制曾发生 downstream regression，因此 P1 只是本轮最合规、最直接的单机制候选，不是成功保证。**

最终裁决：**P1 = GO（仅允许进入实现与梯度通路检查）；P3 = BACKUP；P2 = 不合规；其余机制本轮不选。**

本裁决唯一推荐 P1，不允许把 P1 与任何其他 novelty 叠加。 
