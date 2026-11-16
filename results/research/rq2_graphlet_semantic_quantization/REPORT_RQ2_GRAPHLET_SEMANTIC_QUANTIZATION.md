# RQ2：Graphlet × 语义关系 × 量化前后真实边恢复

日期：2026-11-17  
范围：纯离线评估；不训练混合曲率，不增加行为监督，不运行 Stage3。

## 1. 设置

### 行为边

- 原始 TRAIN 行为图：**315,391** 条无向真实边。
- Amazon 语义标签覆盖 **300,864** 条（**95.39%**）；其余边没有可靠类别，不纳入分组。

### Graphlet 类型

对每条行为边计算：

- triangle：PGD `triangle > 0`
- star：精确逐边 K1,3 计数 `k13 > 0`
- chain：PGD `4-path > 0`
- cycle：PGD `4-cycle > 0`

这些存在性在行为图中几乎普遍，不能区分边。因此本报告使用**主 Graphlet 类型**：

1. 在 `dmax` 五分位内，分别对四类计数取秩；
2. 取秩 >= 0.75 的最高一类；
3. 两个最高秩差 <0.03 记为 `mixed`；
4. 四类都低于 0.75 记为 `low`。

类型计数（全部行为边）：

| type | edges |
|---|---:|
| triangle | 30,437 |
| star | 35,139 |
| chain | 13,420 |
| cycle | 24,935 |
| mixed | 42,539 |
| low | 168,921 |

`low`/`mixed` 共 211,460 条，说明 Graphlet 类型只对约三分之一行为边给出强标签；后面的 Graphlet 分组结论只适用于这部分。

### 向量与 kNN

- 文本：sentence-T5 768 维，L2 归一化，cosine kNN；
- 量化：纯欧氏 TIGER 的 32 维 q（`tiger_vectors.npz`），欧氏 kNN；
- 每个 item 取全局 top-K，对称化并去重；K=10 主实验，K=5/15/20 检查稳定性。

## 2. K=10 主结果

Recovery = 行为边出现在 kNN 无向并集中的比例。

| graphlet | relation | n | text | q | delta (q−text) |
|---|---|---:|---:|---:|---:|
| triangle | all | 29,221 | 0.0615 | 0.0330 | −0.0285 |
| triangle | same_fine | 4,547 | 0.2641 | 0.1427 | −0.1214 |
| triangle | same_coarse_diff_fine | 13,759 | 0.0335 | 0.0175 | −0.0160 |
| triangle | cross_coarse | 10,915 | 0.0124 | 0.0069 | −0.0055 |
| star | all | 33,594 | 0.0105 | 0.0056 | −0.0049 |
| star | same_fine | 1,569 | 0.1517 | 0.0790 | −0.0727 |
| star | same_coarse_diff_fine | 13,489 | 0.0064 | 0.0035 | −0.0030 |
| star | cross_coarse | 18,536 | 0.0015 | 0.0009 | −0.0006 |
| chain | all | 12,918 | 0.0077 | 0.0034 | −0.0043 |
| chain | same_fine | 437 | 0.1442 | 0.0641 | −0.0801 |
| chain | same_coarse_diff_fine | 4,328 | 0.0053 | 0.0023 | −0.0030 |
| chain | cross_coarse | 8,153 | 0.0016 | 0.0007 | −0.0009 |
| cycle | all | 23,851 | 0.0228 | 0.0120 | −0.0107 |
| cycle | same_fine | 1,746 | 0.1987 | 0.1082 | −0.0905 |
| cycle | same_coarse_diff_fine | 11,390 | 0.0126 | 0.0067 | −0.0060 |
| cycle | cross_coarse | 10,715 | 0.0049 | 0.0021 | −0.0028 |
| mixed | all | 41,197 | 0.0163 | 0.0093 | −0.0070 |
| low | all | 160,083 | 0.0176 | 0.0099 | −0.0077 |
| all_edges | all | 300,864 | 0.0209 | 0.0115 | −0.0094 |
| all_edges | same_fine | 20,875 | 0.1972 | 0.1097 | −0.0875 |
| all_edges | same_coarse_diff_fine | 123,825 | 0.0131 | 0.0071 | −0.0059 |
| all_edges | cross_coarse | 156,164 | 0.0036 | 0.0018 | −0.0018 |

### 观察

1. **语义关系主导**：
   - 同 fine 的 text recovery **19.72%**，cross coarse 只有 **0.36%**；
   - 量化后分别是 10.97% / 0.18%。
   - 语义关系越远，文本和 q 都越难恢复。

2. **三角形边恢复最高**：
   - triangle all：text 6.15%，q 3.30%；
   - star all：1.05% / 0.56%；
   - chain all：0.77% / 0.34%；
   - cycle all：2.28% / 1.20%。
   - triangle 内 same-fine 比例 15.6%，star 只有 4.7%，chain 3.4%，cycle 7.3%；这解释了 triangle 的较高总体恢复率的一部分。

3. **同一语义关系内，Graphlet 类型仍有差异（Case C）**：
   - same-fine：triangle 26.4%，star 15.2%，chain 14.4%，cycle 19.9%；
   - cross-coarse：triangle 1.24%，star 0.15%，chain 0.16%，cycle 0.49%；
   - 所以“是否同 fine”不足以解释所有恢复差异，Graphlet 类型本身携带信息。

4. **量化损失**：
   - 总体 K=10：text 2.09% → q 1.15%，绝对 **−0.94 pp**，相对 **−45%**；
   - triangle −46% 相对，star −47%，chain −56%，cycle −47%；
   - **没有某一类被量化特别保护**；各类相对损失接近，绝对损失随基础恢复率变化。

## 3. K 稳定性

| K | all text | all q | delta | all relative |
|---|---:|---:|---:|---:|
| 5 | 0.0139 | 0.0068 | −0.0071 | −51% |
| 10 | 0.0209 | 0.0115 | −0.0094 | −45% |
| 15 | 0.0264 | 0.0156 | −0.0108 | −41% |
| 20 | 0.0310 | 0.0191 | −0.0119 | −38% |

按 Graphlet 的 all-edge 恢复率：

| K | triangle text/q | star text/q | chain text/q | cycle text/q |
|---|---|---|---|---|
| 5 | 0.0419 / 0.0204 | 0.0062 / 0.0031 | 0.0050 / 0.0025 | 0.0156 / 0.0071 |
| 10 | 0.0615 / 0.0330 | 0.0105 / 0.0056 | 0.0077 / 0.0034 | 0.0228 / 0.0120 |
| 15 | 0.0757 / 0.0447 | 0.0141 / 0.0081 | 0.0105 / 0.0063 | 0.0284 / 0.0161 |
| 20 | 0.0875 / 0.0543 | 0.0171 / 0.0104 | 0.0130 / 0.0074 | 0.0326 / 0.0196 |

结论稳定：triangle > cycle > star/chain；q 始终低于 text；量化的相对损失随 K 增大略减小，但绝对差距增大。

## 4. 对三种情形的判断

### Case A：三角形恢复高是否因为语义相似？

**部分成立。**

- triangle 中 same-fine 比例高于 star/chain/cycle；
- 即使只看 same-fine，triangle 的 text recovery（26.4%）仍高于 star（15.2%）、chain（14.4%）、cycle（19.9%）。
- 所以 triangle 的优势既来自语义组成，也来自其局部结构本身。

### Case B：星状/链状边量化前可恢复，量化后丢失？

**没有出现“大量 text 可恢复但 q 系统性丢失”的星状/链状专有现象。**

- star cross text 0.15%，q 0.09%；chain cross 0.16% / 0.07%；
- 这些边在文本 kNN 中本来就几乎不可恢复；
- q 的损失存在，但不是从高恢复率跌到零，而是低基数上的相对下降。
- 更重要的是：**q 对所有 Graphlet 类型都损失约 40–56% 相对恢复率**，没有只打击 star/chain。

### Case C：同语义关系下 Graphlet 损失不同？

**成立。**

- same-fine：triangle 26.4% vs chain 14.4%（text）；
- cross-coarse：triangle 1.24% vs star 0.15%（text）；
- 量化后排序基本保持，但绝对损失不同；
- 说明“是否同类”不足以解释恢复，Graphlet 结构有额外信息。

## 5. 与我们研究目标的关系

本实验能确定：

- 纯欧氏 TIGER q 对所有语义关系、所有主要 Graphlet 类型都损失约 **40–56%** 的相对真实边恢复率；
- 同 fine 的边恢复最高，cross coarse 最低；
- triangle 边恢复最高，star/chain 最低；这一排序在 K=5/10/15/20 稳定；
- 没有证据表明 q 特别保护某一类结构，也没有证据表明 star/chain 的跨类边在文本中已经可恢复却被 q 特别抹掉。

它**不能**直接证明混合曲率会改善这些损失：

- 只比较了文本与纯欧氏 q，没有双曲/球面/混合曲率量化器；
- 没有 Stage3 Recall@10；
- Graphlet 主类型只覆盖约三分之一行为边，`low` 与 `mixed` 仍占多数；
- 语义标签是 Amazon 类目，跨类不等于功能互补。

## 6. 产物

- `results/research/tiger_cluster_graph/rq2_graphlet_semantic_quantization.py`
- `results/research/rq2_graphlet_semantic_quantization/rq2_graphlet_semantic_quantization.json`
- `results/research/rq2_graphlet_semantic_quantization/rq2_graphlet_semantic_quantization.md`
- `results/research/rq2_graphlet_semantic_quantization/cross_coarse_behaviour_sample.csv`
- 量化 q：`results/research/tiger_cluster_graph/tiger_vectors.npz`（纯欧氏 TIGER）
- Graphlet：`pgd_behavior.micro` + `k13_edges.csv`
- 语义标签：`semantic_tree_labels.csv`
