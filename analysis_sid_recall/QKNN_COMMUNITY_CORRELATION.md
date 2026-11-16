# q-kNN community recovery vs Stage3 recall

日期：2026-11-16  
实验：在 **48 个标准 3×(256,32) VQ 码本**的 TIGER arm 上，从 checkpoint 码本 + 导出 SID 重建 32 维 q，建全局 k=10 q-kNN 图，分别计算社区内部和跨社区真实行为边的恢复率，与 `test_recall@10` 做相关分析。

## 1. 数据与口径

- 行为图：TRAIN 用户 session 连续商品对，24,587 商品，315,391 边。
- Louvain（seed 42）：37 个 label，其中 **6 个活跃社区**（31 个孤立商品各成 singleton）。
- 社区内行为边：**188,588**；跨社区：**126,803**。
- 每个 arm 的 q：checkpoint 的 `rq.vq_layers.{0,1,2}.embed.weight`（256×32）按导出 SID 三元组求和。
- q-kNN：每个商品全局最近 10 个 q 邻居，无向化、去重。
- 恢复率：q-kNN 边集合与真实行为边集合的交集，分别除以社区内/跨社区行为边总数。
- q-kNN 边数范围：**137,395–242,842**。
- Recall 范围：**0.030606–0.064764**。
- 排除：mixed/hyperbolic/非标准码本 arm，共 26 个；它们需要各自的 geodesic/readout 解码，不混入标准 VQ 结果。

## 2. 主相关结果

n=48。`partial` 控制 **q-kNN 边数** 与 **L3 collision**。

| 指标 | Spearman | p | Pearson | p | partial Spearman | partial Pearson |
|---|---:|---:|---:|---:|---:|---:|
| 社区内部恢复率 | **+0.6767** | 0.0002 | **+0.7453** | 0.0002 | **+0.6411** | **+0.6222** |
| 跨社区恢复率 | +0.6327 | 0.0002 | +0.6405 | 0.0002 | +0.5633 | +0.4590 |
| 内部−跨社区 gap | +0.6698 | 0.0002 | +0.7494 | 0.0002 | +0.6282 | +0.6308 |

在全样本上：

- **社区内部恢复率的相关性更高**：Spearman 0.677 vs 0.633，Pearson 0.745 vs 0.640；
- partial 后更明显：0.641/0.622 vs 0.563/0.459；
- gap 也强：0.670/0.749。

## 3. 排除高 collision 异常臂后的敏感性

L3 collision ≥ 0.90 的 arm 会同时具有极高 collision 和异常低的恢复率；排除后 n=44：

| 指标 | Spearman | Pearson | partial Spearman | partial Pearson |
|---|---:|---:|---:|---:|
| 社区内部恢复率 | +0.5982 | +0.7522 | +0.6093 | +0.7763 |
| 跨社区恢复率 | **+0.6559** | +0.7207 | **+0.6665** | +0.7238 |
| 内部−跨社区 gap | +0.5883 | +0.7511 | +0.5944 | +0.7764 |

**这是关键的反转：排除高 collision arm 后，跨社区的 Spearman/partial Spearman 反而略高于社区内部。**  
社区内部的 Pearson 仍然略高（0.752 vs 0.721），但差距很小。

因此：

1. **两种恢复率都与 recall 强相关**，不是只有社区内部有关。
2. 全样本上“社区内部更高”，但其中一部分来自高 collision 离群臂；排除后差异消失，甚至跨社区略高。
3. **不能据此断言“社区内部保留比跨社区保留更重要”**；至少当前 48 个标准 VQ 臂不支持稳定的显著差异。

## 4. 解释

- q-kNN 边更多的 arm 通常整体行为边恢复率更高，也更可能获得较高 recall；社区内/跨社区恢复率都部分反映了这种全局 q 质量。
- 控制 q-kNN 边数和 L3 collision 后，两种恢复率仍保持较强相关，说明它们都包含真实信号；但两者的差异对异常臂敏感。
- 社区内部恢复率的 Pearson 更稳定一些（0.745→0.752 排除异常臂后几乎不变）；跨社区恢复率的 Spearman 更稳定一些（0.633→0.656）。没有一方在所有口径下胜出。
- 若要判断“社区内优先”的机制，需要**固定 q-kNN 边数**、或做同边数重采样，并排除 collision/codebook 极端的臂；当前跨臂相关性不足以支持因果结论。

## 5. 限制

- 只覆盖 **48 个标准 VQ 码本**的 TIGER arm；26 个 mixed/hyperbolic/非标准码本 arm 未纳入，因此不能代表全部 74 个归档模型。
- q 是从 **导出 SID + checkpoint 码本**重建的；对标准 VQ 是直接求和，对非标准结构不适用。
- q-kNN 边数未在 arm 间强制匹配，只在 partial correlation 中控制；边数差异可能仍影响结果。
- 社区划分固定为当前行为图的 6 个活跃 Louvain 社区；换社区算法会改变绝对恢复率。
- 跨臂相关是观察性关联，不是因果；不能仅凭相关决定机制。
- 48 个 arm 中有 4 个 collision ≥0.90 的异常臂，敏感性结果应优先看。
- 没有 Stage3 重训；recall 来自归档结果。

## 6. 产物

- `analysis_sid_recall/qknn_community_correlation.py`
- `analysis_sid_recall/results/qknn_community_correlation.json`
- `analysis_sid_recall/results/qknn_community_correlation.md`
