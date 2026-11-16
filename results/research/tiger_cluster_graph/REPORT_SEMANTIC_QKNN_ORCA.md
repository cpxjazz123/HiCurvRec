# 商品语义 Cluster → 量化 kNN 图 ORCA 关系

日期：2026-11-16  
回答的问题：**用商品语义 Embedding 聚类；从量化向量 q 建立全局 kNN 图；分别在这张图和原始行为图上计算 ORCA，比较两者的结构角色关系。**

注意：这里的量化图是**新的全局 kNN 图**，不是上一轮 SID 前缀 cluster，也不是 `build_graphs.py` 的 within-cluster 图。

## 1. 方法

### 1.1 商品语义 Cluster

- 输入：`stage1_GeneEmbedding/output/sentence_t5.npy`，sentence-T5，768 维；
- 先做 L2 归一化；
- KMeans，`k=256`，`seed=42`，`n_init=5`；
- 256 个语义 cluster，最大 cluster 只占 **1.37%**，没有坍缩。

### 1.2 原始行为图

- TRAIN 用户最长 session 的连续商品对，无向、去重；
- **24,587** 商品，**315,391** 条行为边；
- ORCA：GPU 精确 5-node，73 维 orbit；聚合为 9 个角色组；
- 多重角色商品：**30.99%**。

### 1.3 量化后 kNN 图

对 pure TIGER 和 mixed F0 分别取量化向量 \(q_i\)：

- \(q\) 维度：32；
- 距离：**q 空间的欧氏距离**，两个 arm 使用同一口径；
- 对每个商品取全局最近的 **k=10** 个商品；
- 有向 kNN 边对称化并去重，得到 \(G_q\)；
- 在 \(G_q\) 上重新跑 ORCA 5-node，得到 73 维新角色。

结果：

| arm | q-kNN 边数 | 连通分量 | q-vector KMeans(256) 最大簇 |
|---|---:|---:|---:|
| `pure_euclidean_TIGER` | 186,044 | 10 | 2.70% |
| `mixed_F0` | 184,398 | 12 | 1.68% |

### 1.4 量化向量 Cluster

对 \(q\) 也做 KMeans(256) 得到 \(C_q\)，用于与商品语义 Cluster \(C_s\) 比较。这样表格中的 “Cluster” 两边都有明确定义：

- 原始侧：商品语义聚类 \(C_s\)；
- 量化侧：量化向量聚类 \(C_q\)。

## 2. 最重要的现象：量化 kNN 图是“语义近邻图”，不是行为图

| 量 | 原始行为图 | pure q-kNN | mixed q-kNN |
|---|---:|---:|---:|
| 边两端属于同一**语义 cluster** 的比例 | **0.0535** | **0.6423** | **0.6102** |
| 跨语义 cluster 的边占比 | **0.9725** | 0.5269 | 0.5609 |
| 三角数 | 546,438 | 390,495 | 383,166 |

原始行为边几乎全部跨语义 cluster（97.25%），而量化 kNN 图有 **61%–64%** 的边落在语义 cluster 内部。  
这说明：**q 近邻图表达的是“语义相似”，不是“购买行为连接”**。边定义变了，ORCA 角色当然也会变。

## 3. Node-level ORCA 角色保留

分别对每个商品比较原始行为图角色和 q-kNN 图角色：

| arm | top-role ARI | top-role NMI | top role 相同 | 9 组 role JSD 均值 | multi-role 一致率 | bridge nodes | bridge role JSD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pure_euclidean_TIGER` | -0.0003 | 0.0002 | 0.4287 | 0.3326 | 0.6548 | 24,543 | 0.2180 |
| `mixed_F0` | -0.0003 | 0.0002 | 0.4316 | 0.3571 | 0.6826 | 24,543 | 0.2621 |

关键：

1. **top-role ARI 基本为 0，NMI 也基本为 0**：新图的 ORCA 角色与原始行为图角色几乎无关。
2. top role 相同只有 **42.9% / 43.2%**；考虑到角色分布集中在少数组，这个数字不构成“保留”。
3. 每个商品的 9 组 role 分布 JSD 均值 **0.333 / 0.357**，是明显变化。
4. **多重角色从 30.99% 降到 9.0%（pure）/ 1.7%（mixed）**：q-kNN 图把角色多样性压掉了。
5. 24,543 个商品在原始行为图中都有跨语义 cluster 的边；这些 bridge 商品的角色分布也发生明显变化（JSD **0.218 / 0.262**）。

## 4. 语义 Cluster 级 ORCA 关系

同一个语义 cluster 在原始图和 q-kNN 图中使用同一批商品，只改变边。

| arm | cluster role profile JSD | profile JSD 中位数 | top role 相同 | boundary role JSD | multi-role MAE | all role-pair JSD | within-semantic JSD | cross-semantic JSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pure_euclidean_TIGER` | **0.2331** | 0.2485 | 0.3164 | **0.2775** | **0.2307** | **0.7005** | **0.5749** | **0.7054** |
| `mixed_F0` | 0.2727 | 0.2840 | 0.3242 | 0.3312 | 0.2937 | 0.7834 | 0.6546 | 0.7920 |

解释：

- 语义 cluster 的平均 ORCA 角色 profile 变化很大（JSD 0.23–0.27）；只有约 32% 的 cluster 保持同一个 top role。
- cluster 的**外部邻居角色**也明显改变（boundary JSD 0.28–0.33）。
- 同一语义 cluster 内部、外部的 role-pair 分布都与原始图相差很大（JSD 0.58–0.79）。

### Top semantic clusters

前几个最大的语义 cluster 表现出系统性的角色漂移：

| arm | cluster | size | top role 原始 | top role q-kNN | profile JSD | multi 原始 | multi q-kNN | cross rate 原始 | cross rate q-kNN |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| pure | 211 | 338 | K1_4 | P5 | 0.166 | 0.343 | 0.163 | 0.956 | 0.371 |
| pure | 19 | 317 | K1_4 | P5 | 0.237 | 0.233 | 0.050 | 0.957 | 0.507 |
| pure | 241 | 312 | P5 | P5 | 0.176 | 0.266 | 0.042 | 0.912 | 0.176 |
| pure | 125 | 270 | K1_4 | P5 | 0.283 | 0.285 | 0.107 | 0.979 | 0.469 |
| mixed | 211 | 338 | K1_4 | P5 | 0.232 | 0.343 | 0.000 | 0.956 | 0.367 |
| mixed | 19 | 317 | K1_4 | P5 | 0.266 | 0.233 | 0.032 | 0.957 | 0.597 |
| mixed | 241 | 312 | P5 | P5 | 0.217 | 0.266 | 0.003 | 0.912 | 0.214 |
| mixed | 125 | 270 | K1_4 | P5 | 0.332 | 0.285 | 0.022 | 0.979 | 0.555 |

最明显的模式是 **K1_4 → P5**：量化 kNN 图把原本以 K1_4 为主的语义 cluster 推向 P5；mixed 更严重，很多 cluster 的 multi-role 比例直接变成 0。

## 5. Role-to-role 连接

原始图中，`P5–P5` 是相对最“语义 cluster 内部”的角色对；q-kNN 图几乎抹平了这种差异：

| 角色对 | 原始 within | 原始 lift | pure q-kNN within | pure lift | mixed q-kNN within | mixed lift |
|---|---:|---:|---:|---:|---:|---:|
| K1_4–K1_4 | 0.0432 | 0.806 | 0.6683 | 1.040 | 0.5478 | 0.898 |
| P5–K1_4 | 0.0627 | 1.170 | 0.6058 | 0.943 | 0.6545 | 1.073 |
| P5–P5 | 0.1075 | **2.007** | 0.6390 | **0.995** | 0.6088 | **0.998** |

- 原始图中 `P5–P5` 的 lift 是 **2.007**，明显高于其他角色对；
- 在 q-kNN 图中，`P5–P5` 的 lift 降到 **0.995 / 0.998**，与全图同簇率几乎相同；
- 也就是说，q-kNN 图把“角色对特异的内部连接”变成了“几乎所有角色对都同等程度地留在语义 cluster 内”。
- 这不是 ORCA 角色关系的保留，而是结构性抹平。

## 6. 三角形与跨 Cluster 边

| arm | 三角数原始 | 三角数 q-kNN | 三角/内部边 Spearman | 三角/内部边 MAE | cross rate 原始 | cross rate q-kNN | cross-rate gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure | 546,438 | 390,495 | 0.303 | 41.45 | 0.9725 | 0.5269 | -0.4567 |
| mixed | 546,438 | 383,166 | 0.243 | 41.36 | 0.9725 | 0.5609 | -0.4235 |

- 原始行为图的边几乎全部跨语义 cluster（cross rate 0.9725）；q-kNN 图把大量边关进语义 cluster，cross rate 降到 **0.527 / 0.561**。
- 语义 cluster 内的三角形密度相关性只有 **0.303 / 0.243**；q-kNN 图改变了每个 cluster 的局部三角结构。
- 三角数是图规模的函数，不能直接与 raw 值比较；这里用“三角数 / 内部边”和对语义 cluster 的 MAE 表示局部结构差异。

## 7. 商品语义 Cluster vs 量化向量 Cluster

| arm | q cluster 数 | q 最大簇 | ARI | NMI | 覆盖率 | semantic→q profile JSD |
|---|---:|---:|---:|---:|---:|---:|
| pure | 256 | 2.70% | 0.4655 | 0.7904 | 0.7330 | 0.2597 |
| mixed | 256 | 1.68% | 0.4189 | 0.7498 | 0.6320 | 0.2860 |

- 量化向量 cluster 与商品语义 cluster 有较强对应（NMI 0.79 / 0.75），说明 q 保留了语义信息；
- 但 q cluster 与语义 cluster 并非同一划分，且 mixed 的对应明显更弱（NMI 0.750 vs 0.790，覆盖率 0.632 vs 0.733）。

## 8. 结论

1. **量化 kNN 图不保留原始行为图的 ORCA 角色关系**：
   - top-role ARI ≈ 0，NMI ≈ 0.0002；
   - role JSD 均值 0.33 / 0.36；
   - multi-role 从 30.99% 降到 9.0% / 1.7%；
   - bridge 角色变化 JSD 0.218 / 0.262。
2. **量化 kNN 图是语义近邻图，不是行为图**：
   - 原始行为边 97.25% 跨语义 cluster；
   - q-kNN 边 61%–64% 落在语义 cluster 内部；
   - cross rate 从 0.9725 降到 0.527 / 0.561。
3. **role-to-role 结构被抹平**：
   - `P5–P5` 的原始 lift 2.007，q-kNN 只有 0.995 / 0.998；
   - 各个角色对在 q-kNN 图中几乎同等程度地落在语义 cluster 内。
4. **pure 比 mixed 更接近原始行为 ORCA，但两者都失败**：
   - pure 的 profile JSD、boundary JSD、role-pair JSD、三角结构相关性、semantic cluster NMI 都更好；
   - mixed 的 multi-role 几乎完全消失（1.7%），role JSD 更大；
   - 但 pure 的 top-role ARI 也≈0，不能称为“保留”。
5. 因此，**“用语义 cluster 解释结构角色，再用 q 近邻图重算 ORCA”这条路径本身不成立**：
   - q 近邻图主要按语义相似度连接商品；
   - 原始行为图的结构角色来自购买转移，不是语义近邻；
   - 仅靠量化向量距离重连边，会把行为图换成另一张图，ORCA 角色也随之改变。
   若要保留行为结构，必须把行为边/角色信号显式注入量化或图构造过程，而不是只从 q 向量做 kNN 重连。

## 9. 限制

- kNN 只测了 **k=10**，未做 k 敏感性扫描；不同 k 会改变边密度和 ORCA 角色。
- 两个 arm 都使用 q 空间欧氏距离；没有使用 mixed 模型的曲率距离。这里比较的是“量化向量的近邻图”，不是“混合几何下的测地近邻图”。
- 语义 cluster 用 KMeans(256) 在 sentence-T5 上得到；换 cluster 数或算法会改变 cluster 级绝对数值。
- 原始行为图有 31 个孤立商品；q-kNN 图所有商品都有边，因此图中的 31 个孤立点只在原始侧出现。
- 三角数是 raw count；q 图和行为图边数不同，已额外报告“三角/内部边”与 MAE，但两者仍不是同一密度的图。
- 没有 Stage3；本报告只回答“图结构与 ORCA 角色关系是否保留”，不回答 recall。

## 10. 产物

- `semantic_qknn_orca.py`：语义 KMeans、q kNN 构图、ORCA 重算、角色关系对比。
- `semantic_qknn_orca.json`：全部指标、role matrix、top semantic clusters。
- `semantic_qknn_orca.md`：自动生成的表格。
- q-kNN 的 ORCA 输入/输出在 `results/orca_build/runs/semantic_qknn_*`。
