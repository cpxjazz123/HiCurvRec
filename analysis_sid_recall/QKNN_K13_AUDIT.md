# Exact K1,3 edge counts, recovery and Recall@10

日期：2026-11-16  
替代对象：`analysis_sid_recall/QKNN_PGD_MOTIFS.md` 中的 PGD `3-star` 列。  
数据：训练集行为图，315,391 条无向边；48 个标准 `3×(256,32)` VQ 码本 arm；q 从码本 + 导出 SID 重建，k=10 全局 q-KNN。

## 1. K1,3 计数修复

新增 `analysis_sid_recall/k13_exact.cpp`，逐边精确计数 induced K1,3，不使用 PGD 的 `3-star`。

对边 `(u,v)`，以 `u` 为中心的 K1,3 满足：
- `v` 是叶子；
- 另外两个叶子 `a,b ∈ N(u) \\ N[v]`；
- `a,b` 互不相邻。

因此
\[
k_{u,v} = \binom{|S|}{2} - e(S),\qquad S=N(u)\setminus N[v].
\]
每条 K1,3 恰好有一个中心；边 `(u,v)` 的总参与数：
\[
k_{uv}=k_{u,v}+k_{v,u}.
\]

实现要点：
- 每个 S 用稀疏 64-bit bitset 表示；
- 小 S 枚举点对；大 S 用 `popcount(adj[x] & S)` 计算 `e(S)`；
- 315,391 条边全图运行约 **2.5 秒**（4 线程）。

验证：
- `--selftest` 覆盖 K1,3、K1,4、K4、triangle+pendant、C4，并通过 30 个随机小图的逐边暴力对照；
- 真实图随机抽取 40 条边、另写 Python 暴力枚举点对核对，全部一致；
- 输出总计 `24,009,146,973` 条边参与计数，`8,003,048,991` 个 K1,3。

与 PGD `3-star` 的关系：
- PGD 值始终不小于精确值；均值高约 1,898（约 2.5%），99% 的边不同；
- PGD 的 top-quartile 类与精确 top-quartile 类 Jaccard = **0.9679**；
- 原因是 PGD 用 `C(A,2)` 后没有减去 `local_4_tailed_tris`；它把“三角形 + pendant”也算进了 3-star。因此旧 `3-star` 不能作为精确 K1,3。

## 2. 结果：真实 K1,3 边恢复率

主类定义与旧分析一致：在 `dmax` 五分位内，K1,3 计数位于前 25% 的边。  
匹配对照：`dmax × dmin × semantic` 五分层格；覆盖率为 0.9919。

| 指标 | 48 臂均值 | 中位数 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| K1,3 top 类恢复率 | **0.009164** | 0.008238 | 0.001002 | 0.024049 |
| K1,3 top 类匹配 excess | **-0.002987** | -0.002225 | -0.012272 | +0.002088 |
| 全图恢复率 | 0.011514 | 0.010309 | 0.001160 | 0.032401 |
| 所有 K1,3-positive 边恢复率 | 0.011514 | 0.010309 | 0.001160 | 0.032401 |

解释：

- **K1,3 富集边反而比同度数/同语义匹配的普通边恢复得更少**：平均 `-0.00299`，即相对低约 26%。
- 全图 99.9997% 的边都至少参与一个 K1,3，因此“所有 K1,3 边”与全图相同，不能作为区分性类别。
- top 类中 K1,3 计数中位数为 9,179，`dmax` 中位数为 134；这是高度节点附近的稠密星结构，不是普通局部三角形。

## 3. Recall 相关性

| 指标 | n | direct ρ (p) | partial ρ (p) | CV ΔR² (p) | wins |
|---|---:|---:|---:|---:|---:|
| K1,3 rate | 48 | **+0.6455** (7.3e-7) | +0.0036 (0.981) | +0.0149 (0.335) | 13/30 |
| K1,3 matched excess | 48 | **-0.6330** (1.4e-6) | -0.0313 (0.833) | -0.0295 (0.950) | 11/30 |

控制变量：leave-target-out overall 恢复率、q-edge 数、L3 collision；CV 为按模型系列分组的 30 次 `GroupShuffleSplit`，Ridge(alpha=1)。

`collision < 0.90` 的 44 臂：

| 指标 | direct ρ | partial ρ (p) | CV ΔR² (p) |
|---|---:|---:|---:|
| K1,3 rate | +0.5855 | +0.0677 (0.663) | -0.0381 (1.000) |
| K1,3 matched excess | -0.5458 | -0.0309 (0.842) | +0.0032 (0.271) |

结论：

1. **原始秩相关看起来很强（+0.646）**，但它与 q-edge 数、整体恢复率和碰撞率共变；加入这些控制后 partial ρ ≈ 0。
2. **分组 CV 没有正向预测增量**（rate ΔR²=+0.015，p=0.335；excess ΔR²=-0.030），去掉高碰撞臂后 rate CV 为负。
3. 因此 exact K1,3 **不构成“K1,3 保留越好，Recall@10 越高”的证据**；原始正相关是码本/图密度差异的混合物，而不是结构保真度信号。
4. 匹配 excess 为负，说明在同等度数、语义条件下，K1,3 富集边并没有被 SID 更好地保留。

## 4. 结论

- PGD 的 `3-star` 有约 2.5% 的逐边高估，且 top 类与精确类并非完全一致；之前的 3-star 数值不能作为精确 Graphlet 指标。
- 修复后的 exact K1,3 在 48 臂上：**top 类恢复率 0.00916，低于全图 0.01151，匹配 excess -0.00299**。
- 它与 Recall@10 的**原始相关强，控制后消失，分组 CV 不显著**；没有可用的 Recall 预测证据。
- 若继续做“结构 × SID 前缀”分析，应把 K1,3 作为诊断特征，而不是机制目标；更不能把 PGD 3-star 的正相关当作结论。

## 5. 产物

| 文件 | 内容 |
|---|---|
| `analysis_sid_recall/k13_exact.cpp` | 精确逐边 K1,3 计数器与自测 |
| `analysis_sid_recall/qknn_k13_audited.py` | 恢复率、匹配 excess、相关性、分组 CV |
| `results/research/tiger_cluster_graph/k13_edges.csv` | 每条边的精确 K1,3 计数 |
| `results/research/tiger_cluster_graph/k13_summary.txt` | 总边数、总星数、最大边计数 |
| `analysis_sid_recall/results/qknn_k13_audited.json/.md` | 48 臂结果 |
