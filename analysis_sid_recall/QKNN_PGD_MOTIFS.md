# PGD 4-node edge motifs vs q-kNN recovery and Recall@10

> **2026-11-16 审计更新：本文结论已被取代。** PGD 的 `4-tailed-triangle` 从未赋值，
> 依赖它的 `3-star` 也错误；此外高碰撞臂主导了正 CV。修正后的统计与结论见
> `analysis_sid_recall/QKNN_PGD_AUDIT.md`；本文只保留为缺陷与历史记录，不得引用其相关性结论。

日期：2026-11-16  
对象：48 个标准 `3×(256,32)` VQ 码本 TIGER arm；q 从 checkpoint 码本 + 导出 SID 重建，k=10 全局 q-KNN。  
工具：PGD（Parallel Graphlet Decomposition），`--micro` 输出每条边的 4-node motif 计数。

## 1. PGD 运行

- PGD：`nkahmed/PGD`，本地编译；
- 输入：315,391 条行为边；
- `--workers 8`；
- PGD 本体运行约 **1.5 秒**；
- 输出列：

```
triangle, 2-star, 4-clique, 4-chordal-cycle,
4-tailed-triangle, 4-cycle, 3-star, 4-path
```

注意：**PGD 是 4-node edge motif，不是 68 维 Edge-ORCA**；没有 K1,4、P5、T5 等 5-node edge orbit。

PGD 会对顶点重新编号；已按输入首次出现顺序恢复 original ID 映射，315,391 条边全部对齐。

## 2. Motif-high 类

对每个 motif，在 dmax 五分位内取 **top quartile**，得到可重叠的 motif-high 边集。  
每个类与相同 `(dmax,dmin,semantic-similarity)` cell 中的非该类边匹配，计算：

- `rate`：q-kNN 恢复率；
- `excess = rate − matched`；
- `lift = rate / matched`。

| motif | edges | share | motif median | dmax median |
|---|---:|---:|---:|---:|
| triangle | 67,492 | 21.4% | 7 | 140 |
| 2-star | 78,934 | 25.0% | 177 | 127 |
| 4-clique | 53,166 | 16.9% | 18 | 197 |
| 4-chordal-cycle | 52,865 | 16.8% | 26 | 200 |
| 4-tailed-triangle | 0 | 0 | — | — |
| 4-cycle | 79,539 | 25.2% | 235 | 113 |
| 3-star | 78,796 | 25.0% | 9,610 | 134 |
| 4-path | 78,840 | 25.0% | 6,770 | 116 |

`4-tailed-triangle` 在 PGD 输出中全为 0，无法形成类。

## 3. 与 Recall@10 的关系

Partial 控制总体恢复率、q-KNN 边数、L3 collision。  
CV 是重复 5-fold Ridge(alpha=1)，`+rate`/`+excess` 是相对 base controls 的 R² 增量。

| motif-high | mean rate | mean excess | direct rho | direct r | partial rho | ΔR² rate | p(rate) | CV +rate | CV +excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| triangle | 0.02167 | **+0.01267** | +0.637 | +0.680 | -0.239 | +0.184 | 4.4e-7 | +0.083 | **+0.195** |
| 2-star | 0.00953 | -0.00370 | +0.555 | +0.654 | -0.152 | +0.137 | 3.2e-5 | +0.099 | -0.039 |
| 4-clique | 0.01560 | **+0.00814** | +0.516 | +0.607 | -0.215 | +0.174 | 1.2e-6 | +0.210 | **+0.297** |
| 4-chordal-cycle | 0.01952 | **+0.01194** | +0.535 | +0.645 | -0.222 | +0.179 | 7.1e-7 | +0.157 | **+0.240** |
| 4-cycle | 0.01284 | +0.00133 | +0.561 | +0.632 | -0.111 | +0.151 | 9.7e-6 | +0.140 | **+0.282** |
| 3-star | 0.00909 | -0.00306 | +0.653 | +0.696 | -0.014 | +0.108 | 3.1e-4 | +0.023 | -0.026 |
| 4-path | 0.01018 | -0.00556 | +0.473 | +0.569 | -0.245 | +0.155 | 6.6e-6 | +0.219 | +0.055 |

CV base R²：**0.1418**。

## 4. 解读

### 4.1 PGD 本身

- 运行很快：315k 边、8 线程约 **1.5 秒**；
- 能稳定输出 triangle / 4-clique / 4-cycle / 4-chordal-cycle / 3-star / 4-path 等 4-node 计数；
- 不是 68 维 Edge-ORCA，不能替代 5-node K1,4/P5/T5；
- `4-tailed-triangle` 在这张图上为空。

### 4.2 稠密 4-node motifs 信号更强

**正 degree+semantic matched excess：**

- triangle：+0.0127；
- 4-chordal-cycle：+0.0119；
- 4-clique：+0.0081；
- 4-cycle：+0.0013（较弱）。

它们的 CV 增量也最大（excess 约 +0.20 到 +0.30），说明“高稠密 motif 边恢复得比同度数、同语义相似度的边更好”的模型，Recall 也更高。

**负 excess：**

- 2-star：-0.0037；
- 3-star：-0.0031；
- 4-path：-0.0056。

也就是说，单纯“高星形/高路径计数”并不带来正信号；这类类的 raw correlation 高，但匹配后退掉，和之前 degree-matched 的结论一致。

### 4.3 这不是 Edge-ORCA 的替代

PGD 只能证明 **4-node edge motif 可用**，不能回答：

- 5-node K1,4 星形；
- P5 链状；
- T5 分叉；
- 它们对应的 68 维 edge orbit。

要回答这些，仍需要完整的 5-node Edge-ORCA（CPU 很慢，GPU 目前没有现成 kernel）。

## 5. 结论

1. **PGD 是一个可行的快速第一层筛选工具**：315k 边 1.5 秒，能给出逐边 4-node motif；
2. **4-clique / 4-chordal-cycle / triangle / 4-cycle 的 matched excess 为正**，且 CV 增量明显；
3. **高 2-star / 3-star / 4-path 没有正 matched excess**；
4. 因此，PGD 结果支持“稠密 4-node 局部结构更值得关注”，但**不能据此推广到 K1,4/P5/T5**；
5. 下一步若必须区分 5-node 星/链/分叉，还是需要解决 68 维 Edge-ORCA 的 GPU/CPU 计算问题。

## 6. 限制

- PGD 是 4-node，不是 68 维 Edge-ORCA；不能替代原计划；
- 48 个标准 VQ arm；mixed/hyperbolic 未纳入；
- motif-high 类可重叠，不是互斥结构分类；
- PGD 内部 relabel 后按输入首次出现顺序恢复了 original ID；跨版本 PGD 需重新验证；
- CV base R² 约 0.14，fold 方差大；PGD motif 之间存在强共线性；
- 跨臂观察性，不是因果；没有 Stage3 重训。

## 7. 产物

- `analysis_sid_recall/qknn_pgd_motifs.py`
- `analysis_sid_recall/results/qknn_pgd_motifs.json`
- `analysis_sid_recall/results/qknn_pgd_motifs.md`
- PGD 外部源码曾置于 `tools/research/third_party/PGD`（未入库，owner 2026-11-22 指令删除）；上游为 Ahmed et al., ICDM 2015 的 PGD，需要复算时从上游重新获取。PGD 输出仍保留在 `results/research/tiger_cluster_graph/pgd_behavior.micro`。
