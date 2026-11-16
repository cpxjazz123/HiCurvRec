> 注意：本报告是 KMeans 量化前后分析，不是“行为图 → SID”保真度。主评估已改为 `REPORT_SID_FIDELITY.md`；本报告保留为量化前后 cluster 稳定性的辅助证据。

# Cluster-Level ORCA Structural Fidelity：纯欧氏 TIGER vs 混合曲率 F0

日期：2026-11-16  
行为图 ORCA：73 维 node orbit（GPU 精确 5-node，`behaviour_gpu_5.out`）  
纯欧氏：`results/stage2_RQ-VAE/TIGER_RQ-VAE`（canonical SID，collision 0.1693）  
混合曲率：`results/stage2_RQ-VAE/TIGER_mixed_rq_F0`（E16×H16，curvature 1.0，collision 0.2572）  
脚本：`cluster_orca_fidelity.py`  
数值：`cluster_orca_fidelity.json`；表：`cluster_orca_fidelity.md`

---

## 1. 方法

对每个 arm 独立执行：

1. 在 **量化前 `z`** 上拟合 KMeans，`k=256`，`seed=42`，`n_init=10`。
2. 用**同一套聚类中心**给 `z` 和量化后 `q` 分配 cluster：`labels_z = argmin ||z-c||`，`labels_q = argmin ||q-c||`。
3. 行为图只运行一次 ORCA，得到每个商品的 73 维 GDV；不重新构图。
4. 按 cluster 聚合：
   - 73 维 orbit 概率分布的 cluster 均值；
   - 9 个 orbit-group 的结构角色分布（K5、K4_tail、K2_3、C5、T5、P5、K1_4、K4、triangle）；
   - 行为边是否同簇、簇内行为边占比、同簇行为边在 z→q 后是否保留。
5. 统计多重角色商品：归一化角色向量中至少两个 orbit-group 得分 ≥0.25。

说明：ORCA GDV 是**每个商品在行为图中的固定属性**；量化不会改变商品自己的 GDV，只会改变它被分到哪个 cluster。因此 cluster 角色分布变化只来自 cluster 成员变化。

---

## 2. 总体结果

| 指标 | 纯欧氏 TIGER | 混合曲率 F0 |
|---|---:|---:|
| z→q item switch rate | **10.65%** | 16.19% |
| ARI(z cluster, q cluster) | **0.839** | 0.732 |
| NMI | **0.908** | 0.865 |
| GDV JSD（按 cluster 加权） | **0.000185** | 0.000245 |
| GDV agreement = 1−JSD | **0.999815** | 0.999755 |
| GDV JSD max cluster | **0.00808** | 0.01138 |
| Role JSD（9 组，按 cluster 加权） | **0.000151** | 0.000177 |
| Role agreement | **0.999849** | 0.999823 |
| Role JSD max cluster | **0.00935** | 0.01132 |
| 行为边同簇率（z） | 7.996% | 7.204% |
| 行为边同簇率（q） | 8.042% | 7.241% |
| 同簇行为边 z→q 保留率 | **89.94%** | 80.996% |
| 簇内行为边占比（z） | 4.165% | 3.737% |
| 簇内行为边占比（q） | 4.190% | 3.756% |
| 多重角色商品占比 | 30.99% | 30.99% |
| 多重角色 item switch | 11.10% | 16.36% |
| 单角色 item switch | 10.45% | 16.11% |

### 读法

1. **两种量化的 cluster 级 ORCA 角色分布都极其稳定**：Role JSD 加权只有 0.00015–0.00018，最大单簇也只有 0.009–0.011。即使 10%–16% 的商品换了 cluster，簇的角色组成几乎不变。
2. **这不等于“量化无损”**：同簇行为边保留率纯欧氏 89.94%，混合曲率 80.996%；后者有约 19% 的原本同簇行为边在量化后不再同簇。角色分布是平滑的聚合量，掩盖了个体关系变化。
3. **混合曲率 F0 没有比纯欧氏更好地保留 cluster 结构**：switch 更高（16.2% vs 10.7%）、ARI 更低（0.732 vs 0.839）、行为边保留率更低（81.0% vs 89.9%）、簇内行为边占比更低（3.74% vs 4.17%）。Role/GDV JSD 也略高。以这次 F0 配置，假设“混合曲率保留更多结构角色”**没有得到支持**。
4. **多重角色商品更容易换簇，但差异不大**：纯欧氏 11.10% vs 10.45%，混合 16.36% vs 16.11%。多重角色商品的额外风险存在，但不是主要失真来源。

---

## 3. 逐 Cluster 检查（不只平均值）

`cluster_orca_fidelity.json` 保存了每个保留 cluster 的：
- `size_z`、`size_q`、`overlap`
- `gdv_jsd`、`role_jsd`
- `within_share_z/q`
- `role_z` / `role_q`（9 维角色分布）
- `top_role`

### 纯欧氏：Role JSD 最大的 cluster

| cluster | size z | size q | overlap | GDV JSD | Role JSD | within z | within q | top role |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 193 | 19 | 12 | 0.632 | 0.0079 | 0.0093 | 3.98% | 3.27% | P5 |
| 204 | 11 | 11 | 0.727 | 0.0081 | 0.0093 | 0.88% | 0.62% | P5 |
| 27 | 13 | 14 | 0.846 | 0.0065 | 0.0062 | 0.00% | 0.00% | P5 |
| 223 | 15 | 14 | 0.800 | 0.0039 | 0.0060 | 1.15% | 0.60% | P5 |
| 43 | 23 | 14 | 0.565 | 0.0046 | 0.0058 | 1.05% | 0.70% | K1_4 |

### 混合曲率：Role JSD 最大的 cluster

| cluster | size z | size q | overlap | GDV JSD | Role JSD | within z | within q | top role |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 253 | 10 | 13 | 0.900 | 0.0114 | 0.0113 | 0.37% | 0.66% | K1_4 |
| 212 | 17 | 12 | 0.588 | 0.0081 | 0.0093 | 0.35% | 0.00% | K1_4 |
| 17 | 15 | 10 | 0.467 | 0.0091 | 0.0053 | 0.71% | 0.47% | K1_4 |
| 222 | 11 | 12 | 0.818 | 0.0064 | 0.0048 | 0.78% | 0.53% | P5 |
| 106 | 38 | 29 | 0.605 | 0.0073 | 0.0045 | 0.14% | 0.15% | K1_4 |

变化最大的都是 10–38 个商品的小簇；大簇的 JSD 更小。没有任何一个 cluster 出现“角色组成整体翻转”的大幅变化。

---

## 4. 对假设的判定

**假设**：混合曲率量化比纯欧氏量化保留更多不同的 graphlet 角色，并减少行为关系失真。

**本次 F0 数据**：

- Role/GDV 分布：两者都几乎不变；混合的 JSD 略高，不是更低。
- 行为关系：纯欧氏保留 89.94% 的同簇行为边，混合保留 80.996%。
- Cluster 稳定性：纯欧氏 ARI 0.839，混合 0.732。
- 多重角色商品：混合的换簇率更高（16.36% vs 11.10%）。

**判定**：以 `TIGER_mixed_rq_F0` 这个混合曲率配置，**没有证据支持它比纯欧氏更好地保留 cluster-level ORCA 结构**；相反，它在 cluster 稳定性与行为边保留上更差。不能把这个负结果解释为“混合曲率一定无效”——这里只测了 F0 一个训练结果，且混合臂 collision 0.257 明显高于纯欧氏的 0.169；需要其他混合曲率臂/多次训练才能区分机制与训练噪声。

---

## 5. 限制

1. **GDV 是固定的**：量化只改变 cluster 归属，不改变商品自己的 ORCA 向量。Cluster 角色分布是聚合量，可能掩盖个体角色变化；本报告补充了多重角色商品与逐簇表，但仍不是完整的个体级角色检验。
2. **Cluster 中心只在 z 上拟合**：这符合“同一套中心”的要求，但 KMeans 对 z 的划分本身可能偏向纯欧氏几何；混合臂的 z 与 q 在同一个中心下比较，不能完全排除中心选择对 mixed 的不利。
3. **Role 组有重叠**：K5/K4_tail/K2_3 等 orbit 组共享 orbit 索引，归一化后的 9 维角色分布是“参与度”而非互斥类别；多重角色阈值 0.25 是人为固定。
4. **行为边同簇率绝对很低**（约 7%–8%）：KMeans 是语义簇，行为边大量跨簇；这里比较的是 z/q 的相对保留，不是“SID 已经能重建行为图”。
5. **混合臂只测了 F0**：没有扫描其他混合配置，也没有多种子；不能下机制级结论。
6. **q 使用导出 SID + 码本重建**：`model.rq` 的 argmin token 与导出 SID 不一致，实际量化口径以导出 SID 为准。
7. **没有 Stage3 Recall 关联**：这是离线结构保真度分析，不是推荐效果。

---

## 6. 复现

```bash
# 纯欧氏 z/q（已有）
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  results/research/tiger_cluster_graph/build_vectors.py

# 混合 F0 z/q
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  results/research/tiger_cluster_graph/mixed_vectors.py

# Cluster-level ORCA fidelity
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  results/research/tiger_cluster_graph/cluster_orca_fidelity.py
```

产物：

| 文件 | 内容 |
|---|---|
| `cluster_orca_fidelity.json` | 两个 arm 的全部 cluster 指标、9 维 role_z/role_q、top role |
| `cluster_orca_fidelity.md` | 总体表、multi-role 表、各 arm top shifted clusters |
| `orbit_groups.json` | 73 个 ORCA orbit 到 9 个结构组的映射 |
| `mixed_vectors_meta.json` | F0 checkpoint/SID SHA、z/q 来源 |
