# 原始 Sentence-T5 文本 embedding 的 kNN K 扫描

日期：2026-11-17  
输入：`stage1_GeneEmbedding/output/sentence_t5.npy`，24,587 × 768，原始 TIGER 文本 embedding。  
构图：L2 归一化后按 **cosine** 取全局 top-K；每个商品排除自身；K ∈ {5,10,15,20}；有向 kNN 边取**无向并集**（去重），以便计算无向 Graphlet。

## 1. 为什么不用商品类别做粗/细粒度

`results/stage0_build_parquet/items.parquet` 的 `categories` 字段全部为 `None`；仓库中也没有可用的原始类别表。因此本报告不伪造类别标签，而用同一文本 embedding 上的 KMeans 作为语义粒度代理：

- **粗粒度**：KMeans，`k=16`；
- **细粒度**：KMeans，`k=256`（复用已有 `semantic_clusters.npz` 的 256 簇标签）；
- 两者独立拟合，不是嵌套层级；
- 一致率 = 一条无向边两端属于同一簇的比例；
- 随机基线 = `sum_c (n_c / N)^2`；lift = 一致率 / 随机基线。

这只回答“kNN 边在语义空间里是否局部”，不能替代真实商品类目。

## 2. Graphlet 统计

Graphlet 为 **connected induced 4-node graphlets**；同时给出 3-node 的三角形占（三角形 / (三角形 + 2-star)）。

| graph | undirected edges | mean degree | triangles | 3-node triangle share | K4 | diamond | C4 | paw | K1,3 | P4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| behavior reference | 315,391 | 25.66 | 546,438 | 0.0133 | 0.0001 | 0.0020 | 0.0016 | 0.0467 | 0.6247 | 0.3248 |
| kNN K=5 | 98,547 | 8.02 | 83,210 | 0.1159 | 0.0049 | 0.0256 | 0.0064 | 0.1687 | 0.2725 | 0.5220 |
| kNN K=10 | 198,415 | 16.14 | 382,051 | 0.1286 | 0.0073 | 0.0317 | 0.0068 | 0.1877 | 0.2536 | 0.5129 |
| kNN K=15 | 298,627 | 24.29 | 904,515 | 0.1335 | 0.0083 | 0.0344 | 0.0071 | 0.1947 | 0.2485 | 0.5071 |
| kNN K=20 | 399,072 | 32.46 | 1,654,443 | 0.1366 | 0.0089 | 0.0360 | 0.0072 | 0.1987 | 0.2442 | 0.5050 |

PGD 的 global 4-node 计数已用随机图与逐子集暴力枚举对照，六类全部一致；PGD 的 **local 3-star** 缺陷不影响这里的 global 统计。

### 变化方向

从 K=5 到 K=20：

- K4 份额 **+83%**（0.0049 → 0.0089）；
- diamond **+41%**（0.0256 → 0.0360）；
- paw（triangle + pendant）**+18%**（0.1687 → 0.1987）；
- C4 **+12%**（0.0064 → 0.0072）；
- K1,3 **−10%**（0.2725 → 0.2442）；
- P4 **−3%**（0.5220 → 0.5050）；
- 3-node triangle share **+18%**（0.1159 → 0.1366）。

也就是说，增大 K 会把 kNN 图从“星形/路径为主”推向更多三角形和稠密四节点模式。即使 K=20，K1,3 + P4 仍占 **74.9%** 的 connected 4-node graphlet；行为图是 K1,3 62.5% + P4 32.5% = **94.9%**。份额方向相似，但 kNN 的星形份额明显更低、稠密模式更高，两者仍是不同的图。

## 3. 细/粗粒度语义一致率

| graph | mutual directed | coarse rate | coarse lift | fine rate | fine lift | edge cosine mean |
|---|---:|---:|---:|---:|---:|---:|
| behavior reference | — | 0.2437 | 3.230 | 0.0535 | 10.071 | 0.7257 |
| kNN K=5 | 0.3968 | 0.7964 | 10.556 | 0.7178 | 135.013 | 0.8993 |
| kNN K=10 | 0.3860 | 0.7806 | 10.346 | 0.6799 | 127.886 | 0.8886 |
| kNN K=15 | 0.3806 | 0.7699 | 10.204 | 0.6511 | 122.470 | 0.8821 |
| kNN K=20 | 0.3769 | 0.7622 | 10.102 | 0.6259 | 117.717 | 0.8773 |

从 K=5 到 K=20：

- coarse 一致率 **0.796 → 0.762**（相对 −4.3%）；
- fine 一致率 **0.718 → 0.626**（相对 −12.8%）；
- fine lift **135 → 118**，仍远高于行为图的 10.1；
- edge cosine **0.899 → 0.877**；
- reciprocal directed fraction **0.397 → 0.377**。

结论：

1. K 越大，kNN 边越多、越不局部，细粒度语义一致率下降明显；粗粒度下降较小。
2. 即使 K=20，kNN 边仍有 **62.6%** 同细粒度簇、**76.2%** 同粗粒度簇；行为图只有 **5.4% / 24.4%**。
3. 因此 K=20 的 mean degree（32.5）已经接近行为图（25.7），但**边的内容仍然完全不同**：kNN 是语义近邻图，行为图是购买转移图。
4. Graphlet 变化与语义变化方向一致：K 增大后局部三角形/稠密结构增加、星形减少，但语义一致性下降；这正说明“结构密度”与“语义局部性”不是同一个量。

## 4. 建议的 K

- 若目标是**细粒度语义邻域**：K=5 最纯（fine 0.718，lift 135）。
- 若目标是**接近行为图平均度数**：K=15（24.3）比 K=20 更接近 25.7，同时 fine 0.651；K=20 更密、更多三角形，但语义更散。
- 若目标是**保留稠密局部结构**：K=20 的 K4/diamond/paw 份额最高，但这是密度带来的，不应解读为“更像行为图”。
- 单看本扫描，**K=10 或 K=15** 是“语义纯度 vs 图覆盖”的折中点；没有证据表明某个 K 能让文本 kNN 变成行为图的替代品。

## 5. 限制

- 原始类别不可用；粗/细粒度是 KMeans 代理，换簇数会改变绝对值，lift 比绝对值更可比较。
- KNN 图是无向并集；互惠率单独报告。没有用 mutual-only 图。
- Graphlet 只统计 connected induced 4-node； disconnected 4-node 子图（数量占绝对多数）未纳入份额。
- KNN 的 K 同时改变边数和度数，Graphlet 份额已条件化在“connected 4-node”内，但仍不是密度无关的因果量。
- 仅 TIGER 原始 sentence-T5 embedding；未比较 32 维 encoder z、量化 q 或其他编码器。

## 6. 产物

- `results/research/tiger_cluster_graph/embedding_knn_k_sweep.py`
- `results/research/embedding_knn_k_sweep/embedding_knn_k_sweep.json`
- `results/research/embedding_knn_k_sweep/embedding_knn_k_sweep.md`
- `results/research/embedding_knn_k_sweep/pgd_selftest.pgd.log`
- 各 K 的 PGD global 计数：`knn_k5.pgd.log` … `knn_k20.pgd.log`
