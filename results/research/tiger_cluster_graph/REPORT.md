# TIGER 量化前后：同簇 + 近邻图的结构保留对比

日期：2026-11-16  
数据：TRAIN 行为图作为标准答案；TEST 不参与建图  
输入：TIGER canonical checkpoint `results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth`  
脚本与数值：`results/research/tiger_cluster_graph/`

本报告回答：

> 对原始表示做 cluster + 近邻连边，和对量化后的表示做同一套 cluster + 近邻连边，哪一张图更好地保留了 TRAIN 行为图中的局部结构？

---

## 1. 方法

### 1.1 行为图

- TRAIN 用户会话的相邻购买对去重，得到 **315,391 条无向边、24,587 个商品**。
- 这是标准答案；它不参与任何 cluster 或图构建。

### 1.2 TIGER 表示

- `z`：TIGER encoder 的量化前输出，32 维。
- `q`：量化后读出，32 维；由导出的 `item_sids.json` 的 3 个 code 加上 checkpoint 的三个码本向量重建。
- 说明：`model.rq(z, infer_use_sk=False)` 的 token 与导出 SID 不一致（导出路径使用了不同的最终分配/Sinkhorn），因此 `q` 使用**实际导出的 SID**，不使用另一套 argmin 结果。

### 1.3 同一套 cluster + 近邻规则

对 `z` 和 `q` 分别执行：

1. KMeans，`k=256`，`seed=42`，`n_init=10`（与 TIGER 一级码本容量一致）。
2. 在每个 KMeans 簇内，为每个商品连接最近的 `m` 个同簇商品。
3. 对称化（union），得到无向图。
4. 用二分搜索选 `m`，使边数尽量接近行为图的 315,391 条。

两边使用**完全相同的规则、相同 k、相同边预算**；`m` 也恰好都为 17。

### 1.4 指标

- 行为边 recall / precision / Jaccard。
- degree / clustering / core 与行为图的 Spearman 秩相关。
- 从行为图抽样 9 类 5 节点 graphlet 实例（每类最多 500 个；双三角 119 个）：
  - **edge share**：该实例的行为边中有多少比例出现在表示图中；
  - **exact**：表示图在同样 5 个节点上的 induced edge 集是否与行为图完全相同。
- ORCA 4 节点 orbit 计数（15 维），比较 z/q 与行为图的逐 orbit Spearman 相关、profile cosine。

---

## 2. 图级结果

| 指标 | G_z（量化前） | G_q（量化后） | q / z |
|---|---:|---:|---:|
| 边数 | 321,260 | 309,677 | 0.964 |
| 行为边 recall | **2.0819%** | **1.6399%** | 0.788 |
| precision | **2.0438%** | **1.6701%** | 0.817 |
| Jaccard | **1.0421%** | **0.8343%** | 0.800 |
| degree ρ | **0.03670** | **0.01873** | 0.510 |
| clustering ρ | **0.02294** | **0.01607** | 0.700 |
| core ρ | **0.02204** | **0.02161** | 0.980 |

- 量化后图在**边保留、precision、Jaccard、degree 秩相关**上都更差；degree 秩相关几乎减半。
- 两边的绝对边保留都只有 1.6%–2.1%：KMeans 簇是语义簇，行为边大量跨簇；这张图不是行为图的近似复制，而是“同簇 + 近邻”的受控表示图。比较的是 z 与 q 的相对损失，不是绝对质量。

### 2.1 簇一致性

| 指标 | 值 |
|---|---:|
| KMeans 簇数 | 256 vs 256 |
| ARI(z clusters, q clusters) | **0.7298** |
| NMI | **0.8866** |

- **量化几乎保留了 KMeans 簇归属**：ARI 0.73、NMI 0.89。
- 但图的**边和局部结构**明显下降。这说明“簇中心/粗粒度分组稳定”不等于“簇内近邻关系稳定”。

---

## 3. Graphlet 结构保留

| 结构 | 实例数 | z edge share | q edge share | q/z | z exact | q exact |
|---|---:|---:|---:|---:|---:|---:|
| K5 | 500 | **1.24%** | 0.80% | 0.645 | 0 | 0 |
| double_triangle | 119 | **4.62%** | 3.50% | 0.758 | 0 | 0 |
| K4_tail | 500 | **1.09%** | 0.80% | 0.734 | 0 | 0 |
| K2_3 | 500 | 1.37% | 1.37% | 1.000 | 0 | 0 |
| P5_centre | 500 | **2.35%** | 1.55% | 0.660 | 0 | 0 |
| C5 | 500 | **1.72%** | 1.44% | 0.837 | 0 | 0 |
| P5_hops | 500 | **2.35%** | 1.55% | 0.660 | 0 | 0 |
| K1_4 | 500 | **2.60%** | 2.05% | 0.788 | 0 | 0 |
| T5 | 500 | **2.65%** | 2.05% | 0.774 | 0 | 0 |

- **z 在 8/9 类结构上更好，K2_3 持平，q 没有任何一类更好。**
- 量化损失对 **K5、K4_tail、P5** 最大（q/z 约 0.65–0.73）；对 **C5、星形、T5** 较小（0.77–0.84）；对 K2_3 没有影响。
- `exact` 两边都是 0：cluster 内 kNN 会在同一簇里产生大量行为图没有的边，严格 induced 子图匹配过于苛刻。**edge share 是这轮的主指标**，exact 只说明两张图都不是行为图的精确子图重建。

---

## 4. ORCA 4 节点 orbit

| 比较 | median Spearman | mean | p10 | p90 | profile cosine |
|---|---:|---:|---:|---:|---:|
| z vs behaviour | 0.0058 | 0.0126 | −0.0106 | 0.0362 | 0.8403 |
| q vs behaviour | 0.0139 | 0.0132 | −0.0035 | 0.0257 | 0.8375 |
| z vs q | **0.5479** | **0.5705** | 0.5133 | 0.6360 | **0.9813** |

- 对行为图的 4-node orbit，z 和 q 都几乎不相关（median ρ 0.006 / 0.014，mean 0.013）；profile cosine 都在 0.84 左右。
- **q 与 z 彼此非常接近**（median ρ 0.548、cosine 0.981）：量化保留了“节点 orbit 画像”的大部分，但这两张图都没有复现行为图的 orbit 结构。
- ORCA 的 4-node orbit 主要被度数/三角形数量驱动；它看不到 cluster 内近邻边被替换造成的局部损失。因此 ORCA 与 graphlet edge share 给出不同侧面，不能互相替代。
- **5-node ORCA 未完成**：串行 ORCA 在 31.5 万边的行为图上跑了 6 分钟仍无输出（99% CPU、RSS 2.7G），为避免超时/伪造，报告只使用精确 4-node orbit。5-node 需要 GPU/并行 ORCA 另行补。

---

## 5. 结论

1. **量化保留了粗粒度簇归属，但损失了局部图结构。** ARI 0.73/NMI 0.89 说明“同簇”关系很稳定；但行为边 recall 从 2.08% 降到 1.64%（−21% 相对），degree 秩相关从 0.037 降到 0.019。
2. **在 9 类 graphlet 中，z 全面优于或等于 q。** 量化对 K5/K4_tail/路径类损失最大，对 K2_3 无损。这与“量化主要破坏簇内精细近邻，而不是粗簇”一致。
3. **ORCA 4-node orbit 与 graphlet edge share 结论不同**：orbit 画像在 z/q 之间高度相似（0.98 cosine），但两者对行为图的 orbit 相关都接近 0。不要用 orbit 画像代替 induced edge 结构保留。
4. **这张图的绝对行为边保留很低是方法性质**：KMeans 簇是语义簇，行为边跨簇；本实验只做 z vs q 的受控比较。不能据此说 TIGER 的 SID 没有行为信息，也不能据此预测 Stage3 Recall。

---

## 6. 限制

1. `k=256` 与 `m=17` 是固定选择，没有扫参；两边相同，但其他 k/m 可能改变绝对保留率。
2. KMeans 簇在 z/q 上分别拟合；q 的簇中心可能落在重复量化点上，簇大小分布不完全相同。
3. z 图 321,260 边、q 图 309,677 边，目标 315,391；离散 m 使边数无法精确相等，差异 3.7%。
4. q 使用导出 SID 的码本和，不使用 `model.rq` 的 argmin 路径；这是实际部署口径，但和训练时 Sinkhorn 分配的内部路径不完全相同。
5. graphlet 实例最多 500/类（双三角 119），不是全量枚举；edge share 是抽样估计。
6. 5-node ORCA 未完成；只有 4-node orbit。没有 ORCA 角色表/NMF 重拟合。
7. 行为图来自 TRAIN；没有用 TEST 验证，也没有与 Stage3 Recall 关联。

---

## 7. 复现

```bash
# 1) TIGER z/q/SID
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  results/research/tiger_cluster_graph/build_vectors.py

# 2) KMeans + cluster 内 kNN 图
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  results/research/tiger_cluster_graph/build_graphs.py

# 3) graphlet 结构保留
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  results/research/tiger_cluster_graph/structure_retention.py

# 4) ORCA 4-node orbit（ORCA 源码/二进制在 results/orca_build/，已忽略）
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  results/research/tiger_cluster_graph/orca_structure.py
```

产物：

| 文件 | 内容 |
|---|---|
| `tiger_vectors_meta.json` | TIGER checkpoint/SID SHA、z/q 形状、q 来源 |
| `cluster_graphs_meta.json` | K、m、边数、recall/precision/Jaccard、degree ρ、ARI/NMI |
| `structure_retention.json/.md` | 9 类 graphlet 的 edge share/exact |
| `orca_structure.json/.md` | 4-node ORCA orbit 对比与 5-node 跳过原因 |
