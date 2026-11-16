# Behavior Graph → Quantized SID Structural Fidelity

日期：2026-11-16  
主评估对象：**真实行为图 vs 最终 SID 前缀**；不再使用 KMeans Cluster 作为主评估。  
前一版 KMeans 分析：`REPORT_CLUSTER_ORCA.md`（保留为辅助证据，结论由本报告取代）。

## 1. 数据与方法

### 1.1 Ground truth

- 行为图：TRAIN 中每个用户最长 session 的连续商品对，无向、去重。
- 节点：**24,587**
- 行为边：**315,391**
- 社区：行为图上的 Louvain（seed 42），**37** 个社区。
- 结构角色：GPU ORCA 5-node，每个商品 **73 维 orbit 分布**，聚合为 9 个 orbit-group 角色；多重角色定义为至少两个角色组得分 ≥ 0.25。多重角色商品占 **30.992%**。

### 1.2 被比较的 SID

| 名称 | SID 文件 | 说明 |
|---|---|---|
| `SID_E` | `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` | 纯欧氏 TIGER，canonical SID |
| `SID_M` | `results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json` | 混合曲率 F0（E16×H16） |

两个 SID 都直接对同一张行为图比较，不做 z/q 之间的比较，不做 KMeans。

### 1.3 前缀 Cluster 与指标

对每个商品取 SID 前三码，定义：

- `L1`：第一个码；
- `L2`：前两个码；
- `L3`：前三个码。

SID 的 Cluster 就是这些**实际前缀**的等价类，不是 KMeans 簇。

主要指标：

1. **行为边 prefix@k**：行为边两端共享 Lk 前缀的比例。
2. **prefix lift@k**：  
   \[
   \operatorname{lift}_k=\frac{P(\text{行为边共享 }Lk)}{E_{\text{random},k}},
   \qquad
   E_{\text{random},k}=\sum_c \left(\frac{n_c}{n}\right)^2
   \]
   其中 \(n_c\) 是该 SID 表自己的 Lk 簇大小。  
   **必须看 lift，不能只看 raw prefix**：把一级码本压到很少的码，会机械地提高 raw prefix。
3. **within-edge density lift**：  
   \[
   \frac{\text{簇内行为边数}/\sum_c \binom{n_c}{2}}
        {2E/(n(n-1))}
   \]
   即 SID Cluster 内部行为边密度相对全图密度的倍数。
4. **社区分离**：分别计算社区内边、跨社区边的 prefix 率与 lift；`gap lift = within lift − cross lift`。
5. **角色对齐**：L1/L2 Cluster 对 ORCA 角色的 purity、归一化 purity、NMI/ARI。
6. **Graphlet 保真度**：在行为图中采样 K5、double triangle、K4_tail、K2_3、P5、C5、K1_4、T5 的实例，统计每个实例内部被 SID 前缀保留下来的边比例，并按同样的随机基线算 lift。
   - 每个 pattern 最多 300 个实例；double triangle 只有 119 个。
   - `P5_centre` 与 `P5_hops` 是同一路径的两种标注，样本相同，表中重复列出，不当作独立证据。
7. **多重角色边**：至少一个端点属于多重角色商品的边，比较其 prefix 率与单角色边。

---

## 2. 主结果

### 2.1 原始 prefix 率会误导

| arm | L1 簇数 | L2 簇数 | L3 簇数 | raw prefix@1 | raw prefix@2 | raw prefix@3 |
|---|---:|---:|---:|---:|---:|---:|
| `SID_E` | 249 | 7,374 | 20,425 | 0.0811 | 0.0080 | 0.00166 |
| `SID_M` | 79 | 4,151 | 18,264 | **0.1446** | **0.0188** | **0.00243** |

混合 F0 的 raw prefix **全部更高**，但它的 L1 只有 79 个码（纯欧氏 249 个），最大 L1 簇占 5.89%（纯欧氏 2.92%）。这正是码本坍缩，不是保真度提高。

按各自随机基线归一化后：

| metric | `SID_E` | `SID_M` | 更高 |
|---|---:|---:|---|
| prefix lift@1 | **7.48** | 4.85 | E |
| prefix lift@2 | **16.82** | 14.09 | E |
| prefix lift@3 | 22.64 | **24.70** | M（仅 L3） |
| within-density lift, L1 | **7.51** | 4.86 | E |
| within-density lift, L2 | **18.40** | 14.53 | E |
| within-density lift, L3 | **50.66** | 42.16 | E |

结论：**L1/L2 上纯欧氏 SID 的结构保真度更高；混合 F0 只在 L3 的 edge-prefix lift 上略高。** 混合 F0 的 raw prefix 优势来自一级码本从 249 个码压到 79 个码。

### 2.2 社区内部 vs 跨社区

| arm | within prefix@1 | cross prefix@1 | raw gap | within lift@1 | cross lift@1 | **gap lift@1** |
|---|---:|---:|---:|---:|---:|---:|
| `SID_E` | 0.1170 | 0.0277 | +0.0892 | **10.79** | 2.56 | **+8.23** |
| `SID_M` | 0.2022 | 0.0589 | +0.1433 | 6.78 | 1.98 | +4.81 |

| arm | L1 community purity | normalized purity | L1 NMI |
|---|---:|---:|---:|
| `SID_E` | 0.6649 | **0.5854** | 0.2453 |
| `SID_M` | 0.6443 | 0.5600 | **0.2648** |

- 混合 F0 的 raw gap 更大，同样是因为 within/cross 两侧的 raw prefix 都被整体抬高。
- 按 lift 看，纯欧氏对“社区内边 vs 跨社区边”的区分更强：**8.23 vs 4.81**。
- L1 community normalized purity 纯欧氏略高（0.585 vs 0.560）；NMI 混合略高（0.265 vs 0.245），但两者都不高，不能据此说混合 SID 更好地表达了社区。

### 2.3 ORCA 角色对齐

| arm | role purity L1 | normalized purity L1 | role purity L2 | normalized purity L2 | role NMI L1 |
|---|---:|---:|---:|---:|---:|
| `SID_E` | 0.6361 | **0.2619** | 0.7515 | **0.4960** | **0.0175** |
| `SID_M` | 0.6250 | 0.2394 | 0.6989 | 0.3893 | 0.0169 |

- 纯欧氏在 L1/L2 的角色纯度上都更高，尤其是 L2 的归一化纯度 **0.496 vs 0.389**。
- 角色 NMI 都只有 0.017 左右，说明 SID 前缀与 ORCA 角色几乎只是弱相关；这不是强角色编码。
- 不能把“角色分布稳定”误读为“SID 表达了角色”：角色分布本身是商品固定属性，SID 只是在弱对齐。

### 2.4 多重角色

| arm | 多重角色边 prefix@1 | 单角色边 prefix@1 | 多重角色边 prefix@2 | 单角色边 prefix@2 |
|---|---:|---:|---:|---:|
| `SID_E` | 0.0808 | 0.0815 | 0.0077 | 0.0083 |
| `SID_M` | 0.1437 | 0.1458 | 0.0174 | 0.0205 |

- 两个 arm 中，多重角色相关边的 prefix 率都**不高于**单角色边；没有任何证据表明混合 F0 对多重角色关系更好。
- 多重角色边的数量很大（178,083 条），但当前 SID 前缀没有表现出专门的多角色结构表达。

### 2.5 Graphlet 结构保真度

每个 pattern 的 lift 如下（L1 / L2 / L3）：

| pattern | `SID_E` L1 / L2 / L3 | `SID_M` L1 / L2 / L3 | L1 更高 |
|---|---|---|---:|
| K5 | **6.67 / 7.02 / 9.07** | 3.11 / 11.26 / 6.79 | E |
| double_triangle | **14.21 / 41.32** / 38.09 | 7.70 / 36.81 / **71.32** | E |
| K4_tail | **5.67** / 6.02 / **25.90** | 3.29 / **8.22** / 9.70 | E |
| K2_3 | **5.54 / 12.88** / 15.11 | 4.10 / 9.59 / **28.29** | E |
| P5 | **8.00 / 19.32** / 11.33 | 5.31 / 15.64 / 0.00 | E |
| C5 | **7.26** / 12.64 / 36.27 | 4.90 / **15.52** / **54.31** | E |
| K1_4 | **8.23 / 17.56** / 22.67 | 5.03 / 15.64 / **42.43** | E |
| T5 | **8.38 / 26.34** / 34.00 | 5.31 / 18.77 / **50.92** | E |

- **L1：纯欧氏在全部 pattern 上都更高**（6.67–14.21 vs 3.11–7.70）。
- L2：纯欧氏在 5/8 个 pattern 上更高；混合只在 K5、K4_tail、C5 上更高。
- L3：混合在 double_triangle、K2_3、C5、K1_4、T5 上更高，但 L3 共享边数很少、样本小，且混合 L3 碰撞更高；不能据此说混合整体更好。
- 因此 graphlet 结果与总体一致：**混合 F0 在粗粒度 L1 上明显丢失结构，局部 L3 的高 lift 不构成整体优势。**

---

## 3. `F_E` 与 `F_M` 应该怎么读

不把不同深度、不同结构的指标压成一个任意加权标量。按分量报告：

\[
F_E=(\text{L1 lift}=7.48,\ \text{L2 lift}=16.82,\ \text{L3 lift}=22.64;\ 
\text{community gap lift}=8.23;\ 
\text{role norm. purity L1/L2}=0.262/0.496)
\]

\[
F_M=(\text{L1 lift}=4.85,\ \text{L2 lift}=14.09,\ \text{L3 lift}=24.70;\ 
\text{community gap lift}=4.81;\ 
\text{role norm. purity L1/L2}=0.239/0.389)
\]

**结论：**

1. `SID_E` 在 L1、L2、社区分离、角色对齐、within-density lift 上全面优于 `SID_M`。
2. `SID_M` 只在 L3 edge-prefix lift 上略高（24.70 vs 22.64），且这一优势不伴随更好的 L1/L2、社区或角色对齐。
3. `SID_M` 的 raw prefix 更高，是 **L1/L2 码本坍缩** 的假象；它的 L1 只有 79 个码、最大簇 5.89%，纯欧氏是 249 个码、最大簇 2.92%。
4. 所以，**混合曲率 F0 并没有让 SID 更好地保留购买行为图的结构关系**；在这个具体配置上，结论是负面的。

---

## 4. 对机制设计的直接含义

1. **不要用 raw prefix rate 评估任何结构感知量化机制**。必须同时报告 `lift`、簇数、最大簇占比和 within-density lift；否则，坍缩的一级码本会得到虚假的高分。
2. **结构保真度的主战场在 L1/L2**。行为边端点在 L1/L2 上共享前缀的 lift，纯欧氏 7.48/16.82，混合 4.85/14.09；机制若只改善 L3，不足以弥补粗粒度结构的损失。
3. **社区内关系与跨社区桥接必须分开评估**。纯欧氏给人的社区内/跨社区区分更强（gap lift 8.23 vs 4.81）；这比“所有行为边都更共享前缀”更接近我们要的结构。
4. **ORCA 角色对齐目前很弱**（NMI 约 0.017）。如果机制声称保留多重角色，必须直接证明角色纯度/L2 对齐提高，而不是只证明行为边 prefix 提高。
5. **多重角色不是自动被 SID 表达的**。两个 arm 的多重角色边 prefix 都不高于单角色边；下一步若以多重角色为目标，需要显式把角色组合写进量化分配，而不是指望曲率自动产生。

---

## 5. 限制

- 只有一个混合配置 **F0**；结论限定为“F0 这一具体混合曲率基线”，不推广为“所有混合曲率机制都无效”。
- 没有 Stage3；本报告只测 SID 对行为图的结构保真度，不测 next-item recall。
- Louvain 社区是描述性的，不是唯一 ground truth；换了社区算法，绝对数值会变。
- Graphlet 实例是采样（300，double triangle 119），不是全图枚举；P5 两种标注是同一路径。
- `lift` 使用每个 arm 自己的 L1/L2/L3 簇大小作为随机基线；这控制了码本大小和簇不平衡，但不是对行为图生成过程的因果模型。
- ORCA GDV 是商品在行为图中的固定属性；本报告只衡量 SID 前缀与它的对齐，不证明改变 SID 就一定能提升 Stage3。

## 6. 产物

- `sid_structural_fidelity.py`：完整计算；先运行 `sample_graphlets.py` 生成 `graphlet_instances.npz`（seed 42，约 5 分钟）。
- `sid_structural_fidelity.json`：逐 arm、逐 pattern、逐深度的全部指标。
- `sid_structural_fidelity.md`：完整表格（含 raw、lift、簇统计、角色对、graphlet all-prefix）。
- `sample_graphlets.py`：行为图 graphlet 实例缓存脚本。
- 前一版 `REPORT_CLUSTER_ORCA.md`：KMeans 量化前后分析；保留但不再是主评估。
