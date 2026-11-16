# 单一语义层级树假设检验（修正版）

日期：2026-11-17  
范围：只做离线诊断；不训练 RQ-VAE/混合曲率，不运行 Stage3。

## 0. 本轮修正

上一版有两个方法问题，本版已修正：

1. **理想树并列处理**：同一细类商品距离全部为 1，原实现按 item_id 取前 K，会把少数低 ID 商品变成人为 hub，放大 K1,3。现在对每个距离层内的候选**均匀随机抽取**，使用 **20 个种子**的并列集成；报告 Graphlet 均值与标准差。
2. **四点/超度量距离**：不再用非度量的 `1 - cosine`，改为 **L2 归一化向量的欧氏距离**：
   \[
   d(i,j)=\sqrt{2-2\cos(i,j)}
   \]
   再计算四点条件与超度量 residual。

## 1. 标签来源

本地 `items.parquet` 的 `categories` 为空。使用原始 Amazon Reviews 2023：

- `raw/meta_categories/meta_Musical_Instruments.jsonl`
- 下载 SHA-256：`96f40b5440918f07e3e08e84b7a6874a06a5e32a9988998ca9520cfd1b51a12c`
- 归一化 title 精确匹配；多个不同 category path 的 title 丢弃
- 保留 **23,697 / 24,587（96.4%）**
- coarse：category path 第 2 层，**16 类**
- fine：完整 category path，**567 个叶路径**
- 标签与 embedding 独立，不是 KMeans
- 标签 SHA-256 见 `label_checksums.txt`

Amazon 类目本身有噪声，所以结论按“元数据层级”解释，不把它当作完美人工真值。

## 2. 理想层级树

\[
d(i,j)=
\begin{cases}
1 & \text{same fine path}\\
2 & \text{same coarse, different fine}\\
3 & \text{cross coarse}
\end{cases}
\]

同距离候选区间内均匀随机取 K；20 个种子的集成结果如下。

## 3. 实验一：K 增大时新邻居

文本 kNN：

| ranks | 同一 fine | 同一 coarse、不同 fine | 跨 coarse |
|---|---:|---:|---:|
| 1–5 | 0.6333 | 0.2558 | 0.1109 |
| 6–10 | 0.5649 | 0.2979 | 0.1373 |
| 11–15 | 0.5245 | 0.3190 | 0.1565 |
| 16–20 | 0.4983 | 0.3321 | 0.1696 |

理想树（20 seeds，band rates 与种子无关）：

| ranks | 同一 fine | 同一 coarse、不同 fine | 跨 coarse |
|---|---:|---:|---:|
| 1–5 | 0.9883 | 0.0117 | 0.0001 |
| 6–10 | 0.9651 | 0.0348 | 0.0001 |
| 11–15 | 0.9437 | 0.0561 | 0.0001 |
| 16–20 | 0.9249 | 0.0749 | 0.0001 |

层级顺序违反：

| graph | 跨 coarse 早于同一 coarse | 不同 fine 早于同一 fine |
|---|---:|---:|
| text | **0.1611** | **0.2690** |
| ideal tree | 0.0000 | 0.0000 |

- 最早 5 个邻居已有 **11.1%** 跨 coarse；理想树为 0.01%。
- K=20 最后一档跨 coarse 达 **17.0%**，同一 fine 降到 **49.8%**。
- 16.1% 的商品先看到跨 coarse 邻居；26.9% 先看到不同 fine 邻居；理想树均为 0。

## 4. Graphlet：修正并列后的结果

K=10：

| graph | K1,3 | P4 | paw | diamond | K4 | C4 |
|---|---:|---:|---:|---:|---:|---:|
| text K=10 | 0.2543 | 0.5107 | 0.1887 | 0.0320 | 0.0073 | 0.0069 |
| ideal tree K=10 (20 seeds) | 0.2155 | 0.6633 | 0.0844 | 0.0136 | 0.0023 | 0.0210 |

K=5/15/20 的文本 Graphlet 与理想树均值见 `semantic_tree_hypothesis.md`。修正后：

- 理想树不再被低 ID hub 人工压成 96% K1,3；其 K1,3 约 **20–23%**，P4 约 **62–70%**。
- 文本图与理想树的差异仍存在，但更具体：文本的 **paw** 与稠密 motif 更高，理想树的 **P4** 更高；差异不再是“链状 vs 星状”的假象。
- 因此 Graphlet 只能作为辅助证据，不能单独承担“非树”的结论。

## 5. 四点条件与超度量（欧氏距离）

对 200,000 个无重复四元组/三元组：

| test | text Euclidean mean | >0.05 | ideal tree |
|---|---:|---:|---:|
| four-point | 0.0291 | **17.4%** | 0 |
| ultrametric | 0.0605 | **50.6%** | 0 |

- 改用欧氏距离后，偏离幅度比 `1 - cosine` 小，但仍明显；
- 超度量 residual 中位数约 0.051，说明大量三元组不满足层级树约束；
- 理想树精确为 0（构造保证）。

这仍然是“single tree 不充分”的证据，但比上一版弱，不能夸大。

## 6. 跨 coarse top-5 近邻抽查

文本 kNN 前 5 名中，跨 coarse 的 item-neighbor 对共 **13,142** 个。随机抽 **100** 对人工检查：

| 人工关系 | 数量 |
|---|---:|
| 同一商品类型但被分在不同 coarse 分支 | 42 |
| 同一使用场景/同类设备 | 32 |
| 功能互补/配件关系 | 24 |
| 明显无关或类目噪声 | 2 |

示例：

- 同一商品跨分支：Turntable ↔ Turntable、Mixer ↔ Mixer、Headphones ↔ DJ Headphones、Saxophone ↔ Saxophone。
- 互补/配件：麦克风 ↔ 麦克风线/防喷罩、吉他 ↔ 吉他弦/琴包、鼓 ↔ 鼓皮/鼓架、效果器 ↔ 电源。
- 同一使用场景：效果器 ↔ 音箱、音频接口 ↔ 前置放大器、DJ 控制器 ↔ 音频接口。
- 明显无关：`Food Grade Grease for Stand Mixer` ↔ `Quiklok Mixer Stand`；`Music Clips` ↔ `Bass Bridge Tailpiece`。

100 对的 cosine 均值 **0.883**，中位数 0.897；随机样本中 **98/100** 被判为语义相关（同类、同场景或互补），只有 2 对明显无关。这说明跨 coarse 近邻主要不是随机 embedding 错误，而是：

- Amazon 类目把同一产品类型拆到不同分支；
- 功能互补（麦克风线、琴弦、鼓皮、电源）；
- 同一使用场景（效果器/音箱、接口/话放）。

**因此单一类别树不足以描述这些近邻关系**，但这是小样本人工审计，不能替代完整语义真值。

## 7. 结论

修正后仍可得到：

1. 文本 embedding 的近邻不严格按单一层级树展开：11.1% 的首 5 邻居跨 coarse；16.1% / 26.9% 的顺序违反。
2. 归一化欧氏距离下，17.4% 的四点组、50.6% 的三角组超过 5% residual；树不是精确描述。
3. 100 对跨 coarse 近邻中 98 对语义相关，主要来自类目拆分、互补配件和共享场景；不是随机错配。
4. 修正并列后，理想树的 Graphlet 不再人为星状化；Graphlet 差异变小，因此**不能再把“链状 vs 星状”当作主要证据**。

这些结果支持“商品关系存在多个组织模式”，但**仍不能证明混合曲率是必需的**：

- 只测了 text embedding，不涉及行为边、q 向量或 Recall@10；
- 类目层级有噪声；
- 单树被拒绝不等于任何混合曲率模型都会提升。

下一步应做实验三（行为边 × q-kNN × Recall），而不是直接训练新模型。

## 8. 产物

- `results/research/tiger_cluster_graph/semantic_tree_hypothesis.py`
- `semantic_tree_labels_metadata.csv` / `semantic_tree_labels.csv`
- `label_validation_samples.csv`
- `cross_coarse_top5_pairs.csv`（原始 100 对）
- `cross_coarse_top5_pairs_manual.csv`（人工标注 100 对）
- `semantic_tree_hypothesis.json`
- `semantic_tree_hypothesis.md`
- `label_checksums.txt`
