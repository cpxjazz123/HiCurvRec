# Cluster-level 73-d ORCA: full graph vs internal subgraph

日期：2026-11-16  
固定同一套商品语义 Cluster，分别回答：

- **方法 A**：在完整图上算每个商品的 ORCA，再按 Cluster 聚合 73 维；
- **方法 B**：只保留 Cluster 内部边，在内部子图上重新跑 ORCA，再聚合 73 维；
- **外部结构**：保留跨 Cluster 边、边界商品、以及跨 Cluster 的 Role-to-Role 矩阵。

## 1. 设置

### 商品语义 Cluster

- sentence-T5 768 维，L2 归一化；
- KMeans(256)，seed 42；
- 256 个语义 cluster，最大簇 **1.37%**；
- 所有方法共用这套 cluster。

### 两张图

| 图 | 边定义 |
|---|---|
| 原始行为图 | TRAIN 用户 session 连续商品对，无向去重；315,391 边 |
| pure q-kNN | 32 维量化向量 q 的全局 10-NN，对称化；186,044 边 |
| mixed q-kNN | 同上；184,398 边 |

### 73 维 Cluster 向量

对图中每个商品 \(i\) 得到 73 维 ORCA \(o_i\)，Cluster \(C\) 的向量为：

\[
v_C = \operatorname{normalize}\left(
\frac{1}{|C|}\sum_{i\in C}\log(1+o_i)
\right)
\]

归一化后用于逐 Cluster 的 **JSD** 和 **cosine**。同时记录 Cluster 大小、内部边数、跨 Cluster 边比例，避免把不同规模/密度的 Cluster 直接比较。

方法 B 的内部图是所有 Cluster 内部边的并集；由于不同 Cluster 之间没有边，ORCA 可以在一个不连通图上一次跑完，等价于对每个 Cluster 的内部子图分别跑 ORCA。

## 2. 方法 A：全图 ORCA 聚合

| arm | valid clusters | JSD weighted | JSD median | cosine weighted | cosine median |
|---|---:|---:|---:|---:|---:|
| pure TIGER | 256 | **0.0444** | 0.0444 | **0.9303** | 0.9306 |
| mixed F0 | 256 | 0.0459 | 0.0458 | 0.9279 | 0.9269 |

**全图 73 维 Orbit 的 Cluster 平均非常稳定**：JSD 只有 0.044–0.046，cosine 0.928–0.930。  
但这**不能**推出商品级角色被保留：上一轮已经看到 top-role ARI≈0、NMI≈0.0002。Cluster 平均把大量节点级变化平均掉了，所以方法 A 对节点角色变化不敏感。

## 3. 方法 B：Cluster 内部子图 ORCA

| arm | q 内部边 | 有内部边的 Cluster | valid clusters | JSD weighted | JSD median | cosine weighted | cosine median |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure TIGER | 119,494 | 256 | 253 | 0.3398 | 0.3858 | 0.6377 | 0.6031 |
| mixed F0 | 112,522 | 256 | 253 | **0.3367** | **0.3655** | **0.6443** | **0.6313** |

- 原始行为图的内部边只有 **16,887 / 315,391（5.35%）**；只有 **253 / 256** 个 Cluster 至少有一条内部边。
- q-kNN 图的内部边大幅增加：**119,494 / 112,522**，所有 256 个 Cluster 都有内部边。
- 内部 ORCA 的 Cluster 向量 JSD 约 **0.34**，远高于方法 A 的 0.044；内部结构特征明显改变。
- 这个比较受内部边数量影响：q 图的内部边约是行为图的 **6.7–7.1 倍**；JSD 是在归一化 73 维向量上算的，但 ORCA 轨道计数本身仍依赖图密度。

### 为什么内部 ORCA 变化这么大

Top changed 的例子：

| arm | cluster | size | JSD | 内部边 原始 | 内部边 q | cross 原始 | cross q |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure | 182 | 33 | 0.956 | **2** | **186** | 0.995 | 0.396 |
| pure | 105 | 39 | 0.952 | 1 | 149 | 0.998 | 0.572 |
| pure | 212 | 54 | 0.950 | 2 | 205 | 0.998 | 0.599 |
| mixed | 198 | 37 | 0.959 | 2 | 115 | 0.998 | 0.720 |
| mixed | 182 | 33 | 0.958 | 2 | 170 | 0.995 | 0.452 |
| mixed | 212 | 54 | 0.950 | 2 | 158 | 0.998 | 0.750 |

原始图里这些 Cluster 内部几乎只有 1–2 条边，q-kNN 却给它们加入 80–200 条内部边；内部 ORCA 从接近“空图”变成一个小型近邻图，JSD 自然接近 1。

## 4. 只保留内部子图会损失什么

在原始行为图上直接比较方法 A 与方法 B（同一批 Cluster，同一张图，只去掉跨 Cluster 边）：

- valid clusters：**253 / 256**
- JSD weighted / median：**0.2987 / 0.3266**
- cosine weighted / median：**0.6783 / 0.6758**

这是内部子图方法在行为图上的直接代价：**只保留内部边，Cluster 的 73 维结构表示就已经改变约 0.30 JSD**。  
因此方法 B 必须同时报告外部结构，否则无法代表 Cluster 在整张图中的角色。

## 5. 外部结构：boundary 与跨 Cluster Role-to-Role

### Boundary 角色

Boundary 向量取每个 Cluster 的**唯一外部邻居**的全图 ORCA，聚合方式同为 log1p + 归一化。

| arm | boundary JSD weighted | boundary JSD median | boundary cosine weighted | boundary cosine median |
|---|---:|---:|---:|---:|
| pure | **0.0119** | 0.0115 | **0.9776** | 0.9785 |
| mixed F0 | 0.0132 | 0.0125 | 0.9755 | 0.9767 |

- 外部邻居的 73 维角色分布**非常稳定**（JSD 约 0.012–0.013）。
- 但这只是边界商品的**平均角色组成**；不能说明跨 Cluster 的 Role-to-Role 连接被保留。

### 跨 Cluster Role-to-Role 矩阵

| arm | cross rate 原始 | cross rate q | cross role matrix JSD |
|---|---:|---:|---:|
| pure | 0.9725 | 0.5269 | **0.6807** |
| mixed F0 | 0.9725 | 0.5609 | 0.8164 |

- 原始行为边 97.25% 跨 Cluster；q 图只有 52.7%/56.1% 跨 Cluster。
- 跨 Cluster 的 9×9 role-pair 矩阵 JSD **0.681 / 0.816**，说明角色之间的跨 Cluster 连接方式改变很大；mixed 更严重。
- Boundary 平均角色稳定、但跨 Cluster role-pair 矩阵不稳定：**“边界有什么角色”与“这些角色如何连接”是两件事**。

## 6. 结论

1. **方法 A 的 Cluster 平均 73 维 ORCA 很稳定，但这是聚合掩蔽效应。**  
   全图聚合 JSD 0.044，不能反驳节点级 top-role ARI≈0；Cluster 平均对节点角色变化不敏感。

2. **方法 B 的内部结构不保留。**  
   Cluster 内部 ORCA 的 JSD 约 0.34–0.39（中位数）；q-kNN 图给原本内部几乎空的 Cluster 加入了大量内部边，内部子图从稀疏连接变成近邻图。

3. **只保留内部边本身就会改变行为图的结构表示。**  
   在原始行为图上，方法 A 与方法 B 的 JSD 已经约 0.30；这说明外部连接不能被丢掉。

4. **外部 boundary 角色平均稳定，但跨 Cluster role-pair 连接不稳定。**  
   boundary JSD 约 0.012，但 cross role matrix JSD 0.68/0.82；桥接关系没有被保留。

5. **pure 与 mixed 没有一致赢家。**
   - 方法 A：pure 略好（JSD 0.0444 vs 0.0459，cosine 0.930 vs 0.928）；
   - 方法 B：mixed 略好（JSD 0.3367 vs 0.3398，cosine 0.644 vs 0.638）；
   - boundary：pure 略好；
   - 跨 Cluster role matrix：pure 明显更好（0.681 vs 0.816）；
   - 两者都远未保留原始行为图结构。

6. **最重要的方法学结论**：  
   - 只用 Cluster 内部 ORCA 会丢掉跨 Cluster 结构（行为图上 JSD 0.30）；
   - 只用全图平均 73 维会把节点角色变化平均掉（JSD 0.044 但 node ARI≈0）；
   - 必须同时报告：方法 A、方法 B、boundary、cross role matrix、以及 Cluster 大小/内部边密度，不能只报一个 73 维平均向量。

## 7. 限制

- 语义 Cluster 固定为 KMeans(256)；换 cluster 数会改变内部边数和绝对 JSD。
- 方法 B 的 ORCA 是在内部边并集上计算的；等价于各 Cluster 内部子图分别计算，但没有跨 Cluster 边。
- 原始行为图只有 16,887 条内部边；253/256 个 Cluster 可用；3 个 Cluster 没有内部边，方法 B 跳过。
- q 图内部边 112k–119k，远多于行为图；JSD 已归一化，但仍受图密度影响，不能把 0.34 与 0.30 直接当成同一尺度的“损伤”。
- k=10、q 空间欧氏距离；没有 k 敏感性扫描，没有曲率距离；没有 Stage3。

## 8. 产物

- `cluster_internal_orca.py`：方法 A/B、boundary、cross role matrix 的全部计算。
- `cluster_internal_orca.json`：逐 Cluster 指标、top changed、矩阵。
- `cluster_internal_orca.md`：自动生成的表格。
- 内部图的 ORCA 输入/输出在 `results/orca_build/runs/cluster_internal_*`。
