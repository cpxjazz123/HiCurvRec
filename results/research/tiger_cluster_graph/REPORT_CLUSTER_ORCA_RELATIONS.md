# Behavior-graph clusters vs SID clusters: ORCA relations before and after quantisation

日期：2026-11-16  
回答的问题：**行为图先聚成 cluster，量化后的 SID 也聚成 cluster；比较两套 cluster 上的 ORCA 关系。**  
不再用 KMeans 作为主评估；条目级行为图→SID 保真度见 `REPORT_SID_FIDELITY.md`。

## 1. 两套 cluster

### Before：行为图 cluster

在 TRAIN 行为图（24,587 商品、315,391 条无向边）上跑 Louvain（seed 42）：

- 37 个 Louvain label 中，**6 个是真正含边的社区**；
- 另外 **31 个是孤立商品**（没有行为边），各自成 singleton；
- 6 个活跃社区大小：**5,856 / 5,543 / 5,540 / 3,244 / 2,309 / 2,064**。

因此 cluster 级 ORCA 比较只使用 6 个活跃社区；孤立点不参与行为边，也不参与角色 profile 的 JSD。表中 `role valid` / `boundary valid` 都是 6。

### After：量化 SID cluster

每个商品取 SID 前三码，按 **实际前缀** 聚成：

- `L1`：第一个码；
- `L2`：前两个码；
- `L3`：前三个码。

比较两个 arm：

| 名称 | SID 文件 |
|---|---|
| `SID_E` | `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` |
| `SID_M` | `results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json` |

## 2. 比较哪些 ORCA 关系

1. **Cluster role profile**：每个 cluster 内商品的平均 9 组 ORCA 角色分布；行为社区与 SID cluster 按最大重叠配对，算 JSD。
2. **Boundary role profile**：cluster 的**外部唯一邻居**的角色分布；比较配对 cluster 的 JSD。
3. **Role-pair within-cluster fraction**：对每一个 ORCA 角色对 \((r,s)\)，统计该角色对的行为边中，两端落入同一 cluster 的比例。
   - Before：同一行为社区；
   - After：同一 SID 前缀 cluster。
4. **Role-pair lift**：
   \[
   \operatorname{lift}_{r,s}
   =\frac{P(\text{同一 cluster}\mid \text{角色对 }r,s)}
          {P(\text{同一 cluster})}
   \]
   这控制了整体同簇率。行为图的整体同簇率是 **0.598**，SID 的 L1 同簇率只有 **0.081（纯欧氏）/ 0.145（混合）**；不看 lift 会把“cluster 变多”误读成“关系丢失”。
5. **Within/cross role-pair distribution**：同簇边、跨簇边各自的角色对分布，比较 before/after 的 JSD。
6. **Multi-role 组成**：配对 cluster 的多重角色商品比例差异，按 cluster 大小加权。

注意：行为图只有 **3 个角色对** 的行为边数达到 100（K1_4–K1_4、P5–K1_4、P5–P5），Role-pair Spearman 只有 3 个点，**不作为结论**；主指标是加权 lift MAE 与 JSD。

## 3. Before 的 ORCA cluster 关系

行为社区本身有很强的角色对聚集：

| 角色对 | 行为边数 | 同一行为社区比例 | 相对全图同簇率的 lift |
|---|---:|---:|---:|
| K1_4 – K1_4 | 204,492 | 0.5699 | 0.953 |
| P5 – K1_4 | 86,033 | 0.6034 | 1.009 |
| P5 – P5 | 24,866 | 0.8095 | 1.354 |
| 全部行为边 | 315,391 | **0.5979** | 1.000 |

也就是说，行为图上的 ORCA 关系**不是**“所有角色对都同等程度地留在同一社区”：

- `P5–P5` 最偏向同社区（80.95%，lift 1.354）；
- `P5–K1_4` 接近平均（60.34%，lift 1.009）；
- `K1_4–K1_4` 反而略低于平均（56.99%，lift 0.953）。

这是 before 的基准，不是每个 SID 都自动应该复现的绝对目标；但它给出了“哪些角色关系更像社区内部关系”的参照。

## 4. After：L1 的 ORCA 关系

L1 prefix cluster 的总体同簇率远低于行为社区，但用 lift 后可以比较相对结构：

| arm | L1 簇数 | 全图同簇率 | K1_4–K1_4 lift | P5–K1_4 lift | P5–P5 lift | 加权 lift MAE |
|---|---:|---:|---:|---:|---:|---:|
| Before 行为社区 | 6 | 0.5979 | 0.953 | 1.009 | 1.354 | — |
| `SID_E` 纯欧氏 | 249 | 0.0811 | 0.833 | 1.133 | 1.918 | 0.156 |
| `SID_M` 混合 F0 | 79 | 0.1446 | 0.860 | 1.124 | 1.721 | **0.121** |

解释：

1. 两套 SID 都保留了 before 的**角色对顺序**：`P5–P5 > P5–K1_4 > K1_4–K1_4`。
2. 两者都把 `P5–P5` 的相对同簇性**放大**（1.35 → 1.92 / 1.72），把 `K1_4–K1_4` **压低**（0.95 → 0.83 / 0.86）。
3. 在 L1 上，混合 F0 的 lift MAE 更小（0.121 vs 0.156），主要因为它的 `P5–P5` 为 1.72，比纯欧氏的 1.92 更接近行为基准 1.354；`K1_4–K1_4` 和 `P5–K1_4` 也略接近。
4. 但混合 F0 的 L1 只有 79 个码，纯欧氏 249 个；它的整体同簇率 0.145 也更高。lift 已归一化整体同簇率，但**簇的大小/覆盖仍不同**，所以这个 L1 优势不能直接解释为“混合结构更好”。

### Cluster profile 与 boundary

| arm | L1 簇数 | 行为→SID 覆盖 | forward role JSD | 反向覆盖 | reverse role JSD | boundary JSD | 多重角色 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SID_E` | 249 | 0.088 | **0.0022** | 0.665 | 0.0051 | 0.0031 | **0.019** |
| `SID_M` | 79 | 0.169 | 0.0064 | 0.644 | **0.0047** | **0.0016** | 0.044 |

- 行为→SID 的覆盖都很低（8.8% / 16.9%）：一个行为社区通常被拆到很多 SID L1 簇里，**SID L1 不是行为社区的替代物**。
- 反向覆盖高（0.665 / 0.644），因为一个 SID L1 簇通常能被一个大的行为社区覆盖。
- Role profile JSD 很小（0.002–0.006），但 ORCA 角色分布本身集中在少数角色组（K1_4 / P5），小 JSD 不能证明 cluster 语义匹配；它只说明平均角色组成差异不大。
- Boundary JSD 也很小；混合略低，但不足以单独作为“混合更好”的证据。

### L1 的 within/cross 角色对分布

| arm | within role-pair JSD vs before | cross role-pair JSD vs before |
|---|---:|---:|
| `SID_E` | 0.0054 | 0.0045 |
| `SID_M` | **0.0030** | **0.0037** |

混合 F0 的 within/cross 角色对分布在 L1 上更接近 before，但差异都很小（≤0.005）；这与它更大的 L1 簇、更高的覆盖一致。

## 5. L2 与 L3

### L2

| arm | L2 簇数 | ARI | NMI | forward role JSD | reverse role JSD | boundary JSD | lift MAE | within JSD | cross JSD | multi MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SID_E` | 7,374 | 0.0017 | 0.2422 | 0.0172 | 0.0308 | 0.0161 | 0.233 | **0.0131** | 0.0058 | 0.0935 |
| `SID_M` | 4,151 | 0.0046 | 0.2346 | **0.0010** | **0.0190** | **0.0071** | 0.234 | 0.0143 | 0.0055 | **0.0172** |

L2 上：

- 混合的 cluster role profile 和 boundary 更接近行为社区；
- 两者的 role-pair lift MAE 几乎相同（0.233 vs 0.234）；
- 纯欧氏的 within 角色对分布略更接近（0.0131 vs 0.0143）；
- 混合的多重角色组成更接近（0.017 vs 0.094）。

### L3

| arm | L3 簇数 | ARI | NMI | forward role JSD | reverse role JSD | boundary JSD | lift MAE | within JSD | cross JSD | multi MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SID_E` | 20,425 | 0.0001 | 0.2812 | **0.0082** | 0.0803 | 0.0204 | **0.196** | **0.0121** | 0.0060 | **0.0889** |
| `SID_M` | 18,264 | 0.0002 | 0.2759 | 0.0120 | **0.0723** | **0.0133** | 0.350 | 0.0323 | 0.0059 | 0.0922 |

L3 上：

- 纯欧氏的 role-pair lift MAE 明显更低（0.196 vs 0.350），within 角色对 JSD 也更低（0.0121 vs 0.0323）；
- 混合的 reverse role JSD、boundary JSD 略好；
- 两者 L3 ARI 都接近 0；L3 前缀与行为社区几乎不重合。

## 6. 结论

### 6.1 对“前后 cluster 的 ORCA 关系”的直接回答

1. **行为图的 cluster 级 ORCA 关系是有结构的**：`P5–P5` 明显比 `K1_4–K1_4` 更偏向同一社区（lift 1.354 vs 0.953）。
2. **SID 保留了角色对的相对顺序，但压缩了同簇关系**：
   - 两套 SID 的 L1 lift 排序都是 `P5–P5 > P5–K1_4 > K1_4–K1_4`，与 before 一致；
   - 但 `P5–P5` 被放大，`K1_4–K1_4` 被压低；SID L1 不是行为社区的细化，而是另一种划分。
3. **行为社区与 SID cluster 的整体一致性很弱**：ARI 0.000–0.082，NMI 0.23–0.28；SID 前缀不是行为社区标签。
4. **L1/L2 上混合 F0 的部分 cluster 级指标更接近行为社区**（ARI/NMI、boundary JSD、L1 role-pair lift），但 **L3 上纯欧氏的 role-pair lift 和 within 角色对分布明显更好**。
5. 因此，**cluster 级 ORCA 关系没有出现“混合一致更好”**；结果随深度和指标方向变化。

### 6.2 与上一轮 item-level 结果的关系

上一轮 `REPORT_SID_FIDELITY.md` 的条目级比较是：

- L1/L2 行为边 prefix lift：纯欧氏 7.48/16.82 vs 混合 4.85/14.09；
- 社区 gap lift：纯欧氏 8.23 vs 混合 4.81；
- within-density lift：纯欧氏更高；
- 混合仅在 L3 prefix lift 略高（24.70 vs 22.64）。

本轮 cluster 级比较显示：

- 混合的 **粗粒度 L1/L2 cluster** 更容易与 6 个行为社区匹配（覆盖更高、ARI/NMI 略高、boundary JSD 略低），这在某种程度上是“簇少而大”的结果；
- 纯欧氏的 **条目级边关系** 和 **L3 角色对关系** 更好；
- 两者不是同一件事：**cluster 对齐变好，不等于 SID 对行为边的保真度变好**。

所以不能把“混合 L1 的 ARI/NMI 更高”直接解释为混合机制成功；它和 L1 码本从 249 压缩到 79、覆盖从 8.8% 升到 16.9% 同时发生。

## 7. 限制

- 行为图只有 **6 个活跃社区**；31 个孤立商品形成 singleton，已排除在 role/boundary JSD 之外。6 个社区上的 JSD 不足以做细粒度统计推断。
- 行为社区是 Louvain 的描述性划分，不是唯一 ground truth；换社区算法会改变绝对数值。
- 只有 3 个 ORCA 角色对达到 100 条边；`pair Spearman` 不是有效证据。
- Role profile JSD 很小，部分因为 ORCA 角色分布集中在 K1_4/P5；小 JSD 不代表 cluster 语义等价。
- L1/L2/L3 的 SID 簇数差异很大；lift 控制了整体同簇率，但没有完全消除簇大小/覆盖差异。要严格区分“机制”与“码本容量”，下一步应做**同簇数或同覆盖的匹配比较**。
- 本报告仍然只讨论 F0 这一具体混合配置，不推广到所有混合曲率机制；没有 Stage3。

## 8. 产物

- `cluster_orca_relations.py`：计算脚本。
- `cluster_orca_relations.json`：全部 arm/depth 的 ARI/NMI、coverage、role JSD、boundary JSD、role-pair lift、within/cross JSD、multi-role MAE。
- `cluster_orca_relations.md`：自动生成的完整表格。
- 上一轮 `REPORT_SID_FIDELITY.md`：条目级行为图→SID 保真度，仍是主评估；本报告是 cluster 级补充。
