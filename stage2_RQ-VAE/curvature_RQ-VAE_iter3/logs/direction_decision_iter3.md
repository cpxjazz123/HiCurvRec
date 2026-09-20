# iter3 方向评判

## 评判边界

依据仅限 Agent A 的 `lit_search_iter3.md`、当前机制池及 iter2 失败事实：dead-code online clustering 虽通过梯度检查并完整运行 100k，但训练期未触发；raw unique 为 `23243/24587`；Stage3 150 epoch 的 `test_R@10=0.053186859102700254`，较正式 baseline `0.05356987412733508` 下降约 `0.7147%`，未达到 `0.065`。以下评分是方向评判，不表示任何候选已经实测有效。

评分含义：1–5 分，分数越高表示越符合该项要求；总分满分 25 分。

## 评分表

| 候选 | (a) 根因修复路径 | (b) 2023+ 论文实证匹配度 | (c) 与现有几何/量化栈兼容性 | (d) Stage1 ceiling / downstream regression 风险 | (e) 曲率合规且未重复 | 总分 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| P1：双曲乘积流形上的因子化码本与几何一致量化 | 4 | 5 | 3 | 2 | 5 | **19** | **BACKUP** |
| P2：残差 Semantic-ID 路径稳定化与 anti-hourglass 分配 | 5 | 5 | 3 | 2 | 1 | **16** | **BACKUP** |
| P3：不依赖 dead-code 触发的稠密 STE/codebook 更新与重参数化 | 5 | 5 | 4 | 3 | 4 | **21** | **唯一推荐** |

### P1 评语（BACKUP）

- **(a) 4**：不再等待 dead-code 条件，而是用因子化的双曲码本约束局部决策边界，目标是减少 residual token 翻转及级联传播；但对 iter2 下游回归的直接修复链条仍属间接路径。
- **(b) 5**：Agent A 提供的 HiHPQ（2024）、HyperVQ（2024）和 HVQ-VAE（2025）均直接覆盖双曲乘积/双曲量化/流形码本方向。
- **(c) 3**：原则上可保留 cyclic `c(t)`、Poincaré distance、Sinkhorn 与 M2/M3 residual/transport，但产品因子与动态曲率、现有 assignment 接口之间需要重新对齐。
- **(d) 2**：**WARNING：这是高影响的 Stage2 纯曲率/码本结构变更，Stage1 未变，存在 Stage1 ceiling 与 downstream SID regression 风险。**
- **(e) 5**：明确命中 hyperbolic、product-manifold、geometric/manifold 类别，且机制池及 iter2 事实未显示该因子化码本机制已被尝试。

### P2 评语（BACKUP）

- **(a) 5**：直接针对“dead-code 未触发但中间路径仍可能 hourglass/条件分配失衡”的根因，以路径级条件分配和邻域一致性替代零使用码字触发。
- **(b) 5**：Agent A 提供的 Breaking the Hourglass（2024）、CoST（2024）和 LETTER（2024）均给出 2023 年之后的相关实证依据。
- **(c) 3**：可叠加在现有 residual、Sinkhorn 和 cyclic 流程上，但需要可用的邻域/路径信号，并可能改变现有量化目标与 residual 分配平衡。
- **(d) 2**：**WARNING：Stage1 不变，新增路径/邻域目标仍可能重排 SID，故有明显 downstream regression 风险；不能由 raw unique 改善推断推荐指标会改善。**
- **(e) 1**：候选核心是 Semantic-ID 路径和邻域分配正则，Agent A 给出的定义没有把 manifold、hyperbolic、Riemannian、curvature 或 distance metric 作为机制本体；因此不满足本轮曲率合规要求，即使其可与曲率量化器并用。

### P3 评语（唯一推荐）

- **(a) 5**：把修复点从“等待未使用码字出现”改为始终作用于活跃/近活跃码的稠密更新或重参数化，直接覆盖 iter2 的未触发事实及潜在 gradient sparsity / encoder-codebook mismatch。
- **(b) 5**：Agent A 提供的 Straightening Out the STE（2023）、Online Clustered Codebook（2023）和 SimVQ（2024）均与 STE、码本更新稀疏或重参数化有直接实证关联。
- **(c) 4**：Agent A 明确指出该更新拓扑与现有 cyclic `c(t)`、Poincaré distance、Sinkhorn、M2/M3 residual/transport 正交；仍需确保稠密更新后的码本参数保持合法双曲坐标并避免破坏现有 assignment 接口。
- **(d) 3**：相比整体替换流形，影响面较窄，但仍是 Stage2 码本更新规则变更；**WARNING：Stage1 未变，SID 边界与下游推荐仍可能回归，不能据此承诺超过硬目标。**
- **(e) 4**：候选明确允许在双曲坐标/流形码本参数化中实现，命中 hyperbolic、manifold、geometric 类别；同时不是 iter2 已执行的 dead-code online clustering，也未列为机制池已尝试项。实现时必须保留这一曲率约束，不能退化为普通欧氏 STE。

## 最终结论

**唯一推荐：P3：不依赖 dead-code 触发的稠密 STE/codebook 更新与重参数化。**

推荐理由：它直接绕开 iter2 未触发的 dead-code 分支，并针对码本梯度稀疏/分布错配；相比重构整个双曲表示，和现有 cyclic、Poincaré、Sinkhorn、M2/M3 拓扑的接口冲突较少；但仍必须把双曲流形参数化作为实现约束，而不是退化成普通欧氏更新。

P1、P2 均标记 **BACKUP**。以上仅为候选方向评判，未宣称任何候选已经实测有效。

**GO 仅限实现与梯度检查；训练前必须执行常驻 `scripts/grad_check.py`；若引入 loss/参数，必须检查非零梯度和 `detach` 截断。**
