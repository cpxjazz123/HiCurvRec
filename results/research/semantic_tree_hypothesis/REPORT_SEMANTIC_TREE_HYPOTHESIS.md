# 单一语义层级树假设检验

日期：2026-11-17  
范围：只做离线诊断；不训练 RQ-VAE/混合曲率，不运行 Stage3。

## 1. 语义标签来源

本地 `results/stage0_build_parquet/items.parquet` 的 `categories` 字段为空。为了得到**不依赖 embedding** 的独立层级，使用原始 Amazon Reviews 2023 元数据：

- 文件：`raw/meta_categories/meta_Musical_Instruments.jsonl`
- 来源：`McAuley-Lab/Amazon-Reviews-2023`（Hugging Face）
- 下载文件 SHA-256：`96f40b5440918f07e3e08e84b7a6874a06a5e32a9988998ca9520cfd1b51a12c`
- 映射：item title 归一化后精确匹配 `parent_asin` 条目的 title；匹配到多个不同 category path 的 title 直接丢弃
- 保留：23,697 / 24,587（96.4%）
- 丢弃：776 个 title 未匹配；59 个 ambiguous title；非语义促销/outlet 类 coarse 类别
- 粗粒度：category path 第 2 层（16 类）
- 细粒度：完整 category path（567 个叶节点；同一叶名可能出现在不同 coarse 分支，因此使用完整路径）
- 标签文件：`semantic_tree_labels.csv`；SHA-256 `98125cf941cba6362b4a86f9baea76984d69861ddaf0b303e69fe30fc2d614f6`

这不是 KMeans 从 embedding 反推的标签；它是 Amazon 原始类目，和 embedding 独立。类目本身仍可能有 Amazon 标注噪声，因此结论按“元数据层级”而不是绝对人工真值解释。

## 2. 理想层级树

在保留的商品子集上，定义树距离：

\[
d(i,j)=
\begin{cases}
1 & \text{same fine path}\\
2 & \text{same coarse, different fine}\\
3 & \text{cross coarse}
\end{cases}
\]

用与 text kNN 完全相同的 k=5/10/15/20 规则构造理想树 kNN；距离相同按 item_id 排序。

## 3. 实验一：K 增大时新邻居如何进入

对每个 item，统计 rank band 内新邻居的关系：

| ranks | 同一 fine | 同一 coarse, 不同 fine | 跨 coarse |
|---|---:|---:|---:|
| 1–5 | 0.6333 | 0.2558 | 0.1109 |
| 6–10 | 0.5649 | 0.2979 | 0.1373 |
| 11–15 | 0.5245 | 0.3190 | 0.1565 |
| 16–20 | 0.4983 | 0.3321 | 0.1696 |

理想树：

| ranks | 同一 fine | 同一 coarse, 不同 fine | 跨 coarse |
|---|---:|---:|---:|
| 1–5 | 0.9883 | 0.0117 | 0.0001 |
| 6–10 | 0.9651 | 0.0348 | 0.0001 |
| 11–15 | 0.9437 | 0.0561 | 0.0001 |
| 16–20 | 0.9249 | 0.0749 | 0.0001 |

结论：

- 文本 kNN 的邻居**没有严格按细→粗层级扩展**。最早 5 个邻居只有 63.3% 同 fine；**11.1% 已跨 coarse**。
- 到 16–20 档，同 fine 降到 49.8%，跨 coarse 升到 17.0%。
- 理想树几乎不产生跨 coarse 邻居（≤0.01%），同 fine 始终 ≥92%。
- 同一商场类目下，文本 embedding 的局部邻域仍然主要是语义局部的，但混合了大量**跨大类**的近邻。

## 4. 层级顺序违反

对每个 item 找第一条 `same_coarse`、第一条 `same_fine`、第一条 `cross_coarse`：

| graph | 跨 coarse 出现在同一 coarse 之前 | 不同 fine 出现在同 fine 之前 | first cross n | first cross median | first cross ≤5 |
|---|---:|---:|---:|---:|---:|
| text | **0.1611** | **0.2690** | 11,804 | 6 | 0.4818 |
| ideal tree | 0.0000 | 0.0000 | 3 | 3 | 1.0000 |

解释：

- **16.1%** 的可比商品在看到一个同 coarse 商品之前，先看到了跨 coarse 商品。
- **26.9%** 在看到一个同 fine 商品之前，先看到了同 coarse 但不同 fine 的商品。
- 这两类顺序在理想层级树中都是 0。
- 文本 embedding 的近邻排序不是一棵单一层级树的深度优先/广度优先展开。

## 5. 实验二：Graphlet 与树度量偏离

### 5.1 Graphlet 分布

| graph | K1,3 | P4 | paw | diamond | K4 | C4 |
|---|---:|---:|---:|---:|---:|---:|
| text k=5 | 0.2738 | 0.5198 | 0.1693 | 0.0257 | 0.0049 | 0.0065 |
| text k=10 | 0.2543 | 0.5107 | 0.1887 | 0.0320 | 0.0073 | 0.0069 |
| text k=15 | 0.2490 | 0.5049 | 0.1958 | 0.0348 | 0.0084 | 0.0071 |
| text k=20 | 0.2448 | 0.5026 | 0.1999 | 0.0364 | 0.0090 | 0.0073 |
| ideal tree k=5 | 0.9834 | 0.0001 | 0.0004 | 0.0160 | 0.0002 | 0.0000 |
| ideal tree k=10 | 0.9622 | 0.0005 | 0.0022 | 0.0342 | 0.0009 | 0.0000 |
| ideal tree k=15 | 0.9411 | 0.0017 | 0.0056 | 0.0493 | 0.0020 | 0.0002 |
| ideal tree k=20 | 0.9212 | 0.0039 | 0.0103 | 0.0605 | 0.0033 | 0.0008 |

文本 kNN 以 **P4 + K1,3 + paw** 为主；理想树以 **K1,3** 为绝对主导。随着 K 增大，文本图的 paw/diamond/K4 份额上升，但没有收敛到理想树的形状。

### 5.2 四点条件与超度量

对 200,000 个无重复四元组/三元组，定义相对 residual：

\[
\text{residual}=\frac{S_1-S_2}{S_1}
\]

其中 \(S_1\ge S_2\ge S_3\) 是四个点的三个两两距离和（四点条件），或三个点排序后的三条边长（超度量）。

| test | data | mean | median | p95 | >0.05 | >0.10 |
|---|---|---:|---:|---:|---:|---:|
| four-point | text cosine | 0.0542 | 0.0459 | 0.1329 | **0.4628** | 0.1436 |
| four-point | ideal tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ultrametric | text cosine | 0.1152 | 0.0990 | 0.2765 | **0.7351** | 0.4957 |
| ultrametric | ideal tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

结论：

- **46.3%** 的文本四点组 residual 超过 5%，**14.4%** 超过 10%。
- **73.5%** 的文本三元组违反超度量不等式的 5% 容差；平均 residual 0.115。
- 层级树类的 residual 精确为 0（这是构造保证，不是拟合结果）。
- 当前 cosine 距离明显不是单一层级树距离。

## 6. 总结

在 Amazon 原始类别层级下，文本 embedding 的 kNN 关系**不能**被一棵单一语义树充分解释：

1. 最早 5 个邻居中已有 **11.1%** 跨 coarse；理想树为 0.01%。
2. **16.1%** 的商品先看到跨 coarse 邻居，**26.9%** 先看到不同 fine 邻居；理想树均为 0。
3. 四点条件与超度量检验均明显偏离：46.3% / 73.5% 超过 5% residual；理想树为 0。
4. 文本图的 P4/paw 份额远高于树图，K1,3 份额显著低于树图。

因此，**“商品语义关系是一棵单一层级树”这一前提没有得到支持**。这支持继续研究多模式/非单树结构，但**不等于已经证明混合曲率是必需的**：

- 这里只检验了 text embedding 的 cosine 距离；
- 没有把行为边、RQ-VAE 量化、Stage3 recall 纳入；
- Amazon 类目层级本身有标注噪声；
- 单一树被拒绝，不代表任意混合曲率模型都会更好。

实验三（行为边 × q-kNN × Recall）和混合曲率训练暂不启动。

## 7. 产物

- `results/research/tiger_cluster_graph/semantic_tree_hypothesis.py`
- `semantic_tree_labels_metadata.csv`（原始 title→category 映射；23,719 条）
- `semantic_tree_labels.csv`（过滤后的 23,697 条分析标签）
- `label_validation_samples.csv`（每 coarse/fine 标签的抽样，供人工核对）
- `semantic_tree_hypothesis.json`
- `semantic_tree_hypothesis.md`
- `label_checksums.txt`（元数据、原始标签、过滤标签的 SHA-256）
