# Task #6x — 几何/曲率改进路线完整否证链（合并版）

> **合并来源（9 个原 task description）**:
> - `task59_hhhh_stage4_r10.md` — HHHH Stage 3+4 end-to-end R@10 复跑
> - `task60_hhhh_stage3_longtrain.md` — HHHH Stage 3 long-train 验证 R1 假设
> - `task62_residual_geometry_analysis.md` — RQ-VAE 多层残差几何特性与多几何量化（实现版）
> - `task63_riemannian_codebook_geometry.md` — 黎曼几何优化的码书结构差异与量化效果
> - `task64_geometry_aware_rqvae.md` — 几何感知 RQ-VAE（编码器学习残差几何指导量化）
> - `task66_weighted_geom_distance.md` — 加权几何距离量化器（单码书 + 多距离混合权重）
> - `task67_native_space_equiv_info_throughput.md` — 深度诊断 A+D3（原生空间等价性 + 信息通过率）
> - `task70_multigeom_root_cause.md` — 多几何根因诊断包（D1 + D4 + D5）
> - `task71_three_manifold_residual.md` — 三流形残差测试方案（A-E + F/G）
>
> **合并日期**: 2026-07-18
>
> **合并原因**: 这 9 个 task 共属"几何/曲率改进"主题, verdict 已形成完整否证链 (task59/60 HHHH 端到端失败 + task62→63→64→66→67→70→71 几何/流形路线全部 ❌), 合并保留 P5 paper 论证 "Geometric inductive bias is unnecessary for sequential recommendation" 的全部素材。
>
> **保留文件**: `task65_baseline_r10_reproduction.md` (R@10=0.0973 baseline 复现, 与本否证链对照, **不删**)。

---

## 1. 总目标（9 个 task 共同的科学问题）

> **核心问题**: RQ-VAE 残差量化在 L2 归一化嵌入空间下, 几何/曲率诱导偏置（欧氏 / Poincaré 双曲 / 球面）是否真的能提升推荐召回？

**9 个 task 沿三条路径协同回答**：

| 路径 | Task | 切入角度 | 否证结论 |
|------|------|---------|---------|
| **路径 A: 端到端 SID** | #59, #60 | HHHH Poincaré ball SID → Stage 3+4 端到端 R@10 | ❌ R@10=0.0117~0.0284 << baseline 0.0973 |
| **路径 B: 推理时距离/几何指导** | #62, #64, #66 | 换距离函数 / 几何标签 / 加权距离 | ❌ L2 归一化下完全等价或更差 |
| **路径 C: 训练时优化约束 + 根因诊断** | #63, #67, #70, #71 | 黎曼优化器 / 原生空间 / 三流形残差 | ❌ 优化过程不改码书结构; 残差在 PCA rank/cos sim 维度等价 |

**最终结论**: 距离函数/残差计算/码书学习三者必须几何一致, **三层码书不应分别套三种几何**（Task #71 踩坑教训）。

---

## 2. 否证链时间线

```
2026-07-16:
  Task #59 — HHHH Stage 3+4 端到端 R@10 复跑
            → R@10=0.0117, -88% vs baseline 0.0973
  Task #60 — HHHH Stage 3 long-train 验证 R1 (训练预算)
            → R1 否证, Stage 4 R@10 仍 0.0284 << baseline
  Task #62 — 多几何量化（实现版, 4 个实验）
            → Exp3 距离等价, ΔMSE < 0.2%
  Task #63 — 黎曼优化码书结构
            → Exp1 三者几何度量完全一致, Δ ≈ 0
  Task #64 — 几何感知 RQ-VAE
            → Exp2 几何指导 MSE +140% (选错码书恶化)
  Task #66 — 加权几何距离
            → Exp1 加权距离在 L2 归一化下等价单距离
  Task #67 — 原生空间等价性 + 信息通过率
            → A: d_E↔d_H 100% 等价（最近邻完全相同）
  Task #70 — 多几何根因诊断
            → D1 Pearson 0.97 / D5 Δ_max=1.509 (Scenario B)
  Task #71 — 三流形残差穷尽测试
            → r_E/r_H/r_S 数值上几乎相同 (cos > 0.95)
            → **几何/曲率改进路线彻底关闭**
```

---

## 3. Task #59 — HHHH Stage 3+4 end-to-end R@10 复跑

> **完整原始内容**: `descriptions/task59_hhhh_stage4_r10.md`
> **Verdict**: `verdicts/task59_result.md`（如已存在）

### 3.1 目的
解决 P5 paper section v7 Caveats #1 — "Stage 4 R@10 not directly validated in v7"。验证 HHHH 端到端 R@10 是否能击败 baseline 0.0973。

### 3.2 状态
- **完成日期**: 2026-07-16
- **状态**: ❌ 完成 — Item-level R@10=0.0117 << baseline 0.0973 (-88%)

### 3.3 关键数字
- Stage 3 best val/recall@10 = 0.0125 @ gs 700 (2560 max_steps, patience=10 触发)
- Stage 4 item-level R@10 = 0.0117 (baseline 0.0973, -88%)
- Stage 3 TIGER 在 2560 step 内根本无法收敛到合理值

### 3.4 方法
- Stage 2 RQ-VAE: `task57_HHHH_s22` (Poincaré ball HHHH, v3 z-score + KMeans-on-z recipe)
- Stage 3: `tiger_train_flat.yaml` + max_steps=2560 + patience=10
- Stage 4 推断: `tiger_inference_flat.yaml` + 19412 users
- R@10 eval: `task388v4_s4_item_eval.py`

### 3.5 否证结论
**Stage 2 HHHH 强 + Stage 3/4 失败**。v5 残骸 (task388v5) Stage 4 R@10 = 0.00036 的失败被确认是真实的算法层 gap, 不是 pipeline 故障。

---

## 4. Task #60 — HHHH Stage 3 long-train 验证 R1 假设

> **完整原始内容**: `descriptions/task60_hhhh_stage3_longtrain.md`
> **Verdict**: `verdicts/task60_result.md`（如已存在）

### 4.1 目的
验证 task59 verdict 中的 R1 根因假设 — "Stage 3 max_steps=2560 不够"。如果延长 Stage 3 到 5120 step + patience=20 能把 Stage 4 R@10 提升到 ≥ 0.05 → R1 确认; 如果仍 < 0.05 → R1 否证。

### 4.2 状态
- **状态**: ❌ 否证 — R1 假设被否证, Stage 2/3 gap 是算法层问题

### 4.3 关键数字
- Stage 4 item-level R@10 = 0.0284 (baseline 0.0973, -71%)
- 训练预算翻倍 (2560→5120) + patience 翻倍 (10→20) 无显著改善

### 4.4 方法
- **变量**: Stage 3 max_steps (2560 → 5120), patience (10 → 20)
- **保持不变**: Stage 2 RQ-VAE ckpt, SID tensor, Stage 3 model 配置, seed=42, val_check_interval=320

### 4.5 否证结论
**R1 否证, Stage 2/3 gap 是算法层问题**。HHHH Poincaré ball SID 在端到端管线中无效, 即使训练预算翻倍。

---

## 5. Task #62 — RQ-VAE 多层残差的几何特性与多几何量化（实现版）

> **完整原始内容**: `descriptions/task62_residual_geometry_analysis.md`
> **Verdict**: `verdicts/task62_result.md`（如已存在）

### 5.1 目的
实现 4 个实验, 验证 RQ-VAE 三层残差（R1/R2/R3）是否编码不同属性信息、具有不同几何结构、以及为每层匹配对应几何距离能否提升量化效果。

### 5.2 状态
- **状态**: ❌ 否证 — L2 归一化后欧氏/双曲/球面距离排序一致, 4 setting MSE 差异 < 0.2%

### 5.3 关键数字
- R1 否证: 每行 max NMI 在同一列 → 三层学相同信息
- R2 否证: 三层几何度量一致 → 几何无差异
- R3 否证: R_C ≤ max(R_A, R_B) 或 R_C - R_B ≤ 0.5%
- R4 否证: 权重均匀分布

### 5.4 方法（4 实验）

**Exp1**: 残差信息 NMI + Probe
- 提取 L1/L2/L3 残差 → KMeans (K=64) → NMI vs 商品属性
- 3 random seeds, mean ± std

**Exp2**: 残差几何特性
- δ-Hyperbolicity (4 元组采样, T=1000)
- Angular Uniformity (Beta 理论 vs 实测, -KL)
- Isotropy Score (λ_min/λ_max)

**Exp3**: 换几何距离评估
- d_E, d_H, d_S 三种距离函数
- 4 settings (A: 全 E, B: 全 H, C: 多几何 v1, D: 多几何 v2)
- 固定 encoder + codebook, 只在量化步替换距离

**Exp4**: 学习几何权重
- 3 层可学习 softmax 权重 (w_E^ℓ, w_H^ℓ, w_S^ℓ)
- L_total = L_recon + λ·L_reg

### 5.5 否证结论
**完全替换距离函数无效**。L2 归一化后三种距离排序完全一致, "对齐几何能提升量化" 假设彻底否证。

### 5.6 baseline 参考
- 标准 RQ-VAE (all E): R@10=0.0973, NDCG@10=0.0612
- HRQ (all H): R@10=0.1298, NDCG@10=0.0821

---

## 6. Task #63 — 黎曼几何优化的码书结构差异

> **完整原始内容**: `descriptions/task63_riemannian_codebook_geometry.md`
> **Verdict**: `verdicts/task63_result.md`（如已存在）

### 6.1 目的
验证在 RQ-VAE 码书学习过程中施加不同的黎曼几何约束（欧氏 vs Poincaré ball vs 球面）是否能产生本质不同的码书结构, 以及是否能提升推荐召回率。

### 6.2 状态
- **状态**: ❌ 否证 — Exp1 结果：三者几何度量完全一致，Δ ≈ 0

### 6.3 关键数字
- Exp1a: 三者 Iso/δ-rel/Uniformity 接近, 优化方式不改变几何性质
- Exp1b: Δ(A,B) ≈ 0, 码书学到相似内容

### 6.4 方法（3 实验）

**Exp1**: 码书在不同流形上的结构差异
- 变量: 码书优化方式（欧氏/SGD vs 双曲/黎曼 vs 球面/黎曼）
- 输入: Task62 提取的 L2 残差 R_2 (11924 × 768)
- 码书 K=256, 10 epochs, MSE 重建 Loss
- 度量: 几何特性 (Iso/δ-rel/Uniformity) + 码书差异度 Δ(A,B)=||S_A-S_B||_F

**Exp2**: 黎曼残差计算的特性差异
- 变量: 残差计算方式（欧氏减法 vs Möbius 减法 vs Log map）
- 度量: 残差的方差谱熵 + 条件数

**Exp3**: 黎曼优化 RQ-VAE 的 Recall 效果
- 3 个完整 RQ-VAE 模型（欧氏/混合黎曼/全黎曼）
- Stage 4 推断 → Recall@10

### 6.5 否证结论
**优化过程几何约束不改变码书结构**。训练时更换优化器无效, 与 Task #62 结论一致, 路线否证加强。

---

## 7. Task #64 — 几何感知 RQ-VAE（编码器学习残差几何特性指导量化）

> **完整原始内容**: `descriptions/task64_geometry_aware_rqvae.md`
> **Verdict**: `verdicts/task64_geometry_aware_rqvae.md`（如已存在）

### 7.1 目的
验证残差数据本身是否具有可被编码器学习的局部几何特性差异, 以及用这些几何标签指导码书选择能否提升 R@10。

### 7.2 状态
- **状态**: ❌ 否证 — 所有几何类路线已全部否证

### 7.3 关键数字
- Exp2: 几何指导 MSE_B/C 比 A 高 +140% → **选择码书反而恶化**
- 几何可预测但对推荐无帮助

### 7.4 方法（3 实验 + 1 可选）

**Exp1**: 几何标签是否可被编码器学会？
- 编码器: r_i → z_semantic (128-dim) + z_geo (3-dim softmax)
- 损失: L = MSE(r_i, Decoder(z_q)) + λ·CE(z_geo, y_geo)
- 局部几何标签: k=50 近邻计算 geo_E/geo_H/geo_S
- 阈值: Acc_geo > 0.75 ✅; < 0.55 ❌

**Exp2**: 几何标签是否有用于量化？
- 方式A: 标准 (只用欧氏码书)
- 方式B: 硬选择 (按 z_geo 选对应码书)
- 方式C: 软加权 (Σ_i z_geo[i]·||z-C_i||²)

**Exp3**: 端到端 R@10 验证
- 完整 GRID pipeline 跑 Stage 4
- 决策: R@10 提升 > 0.5% → ✅; 持平 → ❌

### 7.5 否证结论
**即使几何可预测, 对推荐无帮助**。几何标签指导选码书反而恶化 (+140% MSE), 几何类路线彻底否证。

### 7.6 累计否证链
| Task | 结论 | 含义 |
|------|------|------|
| #62 | L2 归一化后欧氏/双曲/球面距离等价 | 推理时更换距离函数无效 |
| #63 | 优化过程几何约束不改变码书结构 | 训练时更换优化器无效 |
| #25-p2 | SID 距离与 embedding 语义无相关性 | P(1) 不反映语义保持性 |
| #64 | 几何指导 MSE +140% | 选择码书反而恶化 |

---

## 8. Task #66 — 加权几何距离量化器

> **完整原始内容**: `descriptions/task66_weighted_geom_distance.md`
> **Verdict**: `verdicts/task66_weighted_geom_distance.md`（如已存在）

### 8.1 目的
验证加权几何距离 d_mix^ℓ = w_E^ℓ·d_E + w_H^ℓ·d_H + w_S^ℓ·d_S 能否在 L2 归一化空间上打破 Task #62 Exp3 的"距离排序等价", 并显著改善量化效果。

### 8.2 状态
- **状态**: ❌ 否证 — 加权距离在 L2 归一化下等价单距离, ΔMSE=0%

### 8.3 关键数字
- Exp1: 最优权重 = (1,0,0) for all layers → 加权在 L2 归一化下完全等价于单欧氏距离

### 8.4 方法（3 实验 + 1 可选）

**Exp1**: 加权距离的参数空间（穷举）
- 3 层各自的 (w_E^ℓ, w_H^ℓ, w_S^ℓ) 三元组
- 网格搜索: w_E ∈ {0, 0.25, 0.5, 0.75, 1.0}
- 约 15-20 种权重组合 × 3 层
- 度量: 各层在测试残差上的量化 MSE

**Exp2**: 学习加权参数
- 3 层各自的 softmax 参数化权重 θ_E^ℓ, θ_H^ℓ, θ_S^ℓ
- L = L_recon + λ·L_entropy (鼓励分散)
- 反向: L_entropy_sharp (鼓励尖锐)

**Exp3**: 端到端 R@10 验证
- Toys, Stage 2/3/4 同 Task #65
- 决策: R_C > R_A + 0.5%, p<0.05 → ✅; ≈ baseline → ❌

### 8.5 否证结论
**加权距离无法克服 L2 归一化下的距离等价问题, 几何类改进路线彻底关闭**。

### 8.6 Task #65 备注
Task #65 在 §16 中状态变更为 ⛔ 已替换（替换原因: 注册 Task #66 验证加权几何距离, 几何类路线最后一次探索）。Task #65 Stage 3 中断时 step=18098/50000, 最新 ckpt=step 1125。

---

## 9. Task #67 — 深度诊断 A+D3：原生空间等价性 + 信息通过率

> **完整原始内容**: `descriptions/task67_native_space_equiv_info_throughput.md`
> **Verdict**: `verdicts/task67_deeper_diagnosis.md`（如已存在）

### 9.1 目的
验证 4 个根本疑问中**两个真正未回答的**：(A) Task #62 的"距离等价性"是 L2 归一化假象, 还是根本性的？(D3) 各层残差是否有明确的信息分工（通过率曲线）, 还是仅数值递减？

### 9.2 状态
- **状态**: ❌ 否证 — 即使不归一化, d_E 和 d_H 仍 100% 等价（最近邻完全相同）

### 9.3 关键数字
- Exp A: A1↔A2=0.70% / **A1↔A3=100%**（L1 残差不归一化）
- 等价性**不是 L2 归一化假象**, 是根本性的

### 9.4 方法

**Exp A**: 未归一化原生空间下三距离等价性测试
- A1 欧氏: r_E = r_1（不归一化）, d_E = ||r-c||_2, 欧氏 KMeans C_E
- A2 球面: r_S = r/||r||, d_S = arccos(⟨r,c⟩), 球面 KMeans C_S
- A3 双曲: r_H = tanh(α)·r/||r||, d_H = arcosh 距离, Poincaré KMeans C_H
- 度量: pairwise 一致率 (期望 baseline ~30%, 256 类随机)
- 决策: < 70% → ✅ 打破等价; ≥ 70% → ❌

**Exp D3**: 信息通过率分析
- 各层残差对 4 属性的 probe accuracy (rebuild/category/brand/copurchase)
- throughput_L1_to_L2(attr) = acc(z_≤2, attr) / acc(z_≤1, attr)
- 决策: 至少 1 属性 throughput > 80% → ✅ 明确分工; 全属性 ∈ [30%, 60%] → ❌

### 9.5 否证结论
**距离等价性是根本性质, 非 L2 归一化假象**。P5 paper 的 "Geometric inductive bias is unnecessary for sequential recommendation" 主张获得最终支持。

---

## 10. Task #70 — 多几何根因诊断包（D1 + D4 + D5）

> **完整原始内容**: `descriptions/task70_multigeom_root_cause.md`
> **Verdict**: `verdicts/task70_multigeom_root_cause.md`（如已存在）

### 10.1 目的
理解 Task #67/#69 的"d_E ↔ d_H = 100% 等价"现象的根本原因 — 是数学性质（A）、残差幅度（B）、参数化特殊性（C）, 还是码书分布（D）？

### 10.2 状态
- **状态**: ✅ 完成诊断 — Scenario B 确认（保序单调变换）

### 10.3 关键数字
- D1: d_E 和 d_H 排序完全一致 (Pearson 0.97) → 假设 A 成立（参数化是单调变换）
- D4: top1_match = 100% (实测确认) → 假设 C 成立（码书分布导致）
- D5: Δ_max=1.509, d_H[0] ≈ 2 × d_E[0] → 距离值放大但排序不变

### 10.4 方法（3 实验）

**Exp D1**: 距离函数本身的单调性分析
- 变量: cos 值 ∈ [0.1, 0.99]
- d_E = √(2-2·cos), d_H = arcosh(1+2(1-cos)²/((1-√(1-cos²))²))
- 度量: 排序一致性

**Exp D4**: 码书分布的影响（实际 RQ-VAE 数据）
- 输入: Task 62 R_1 (11924, 768) + C_E (256, 768)
- 度量: full_match + top1_match

**Exp D5**: 距离数值差异分布
- Δ[i,k] = |d_E(r_i, c_k) - d_H(r_i, c_k)|
- 度量: Δ_mean/Δ_std/Δ_max/Δ_min

### 10.5 否证结论
**情景 A 确认**: 等价性是根本数学性质, 多几何路线最终关闭。把"为什么多几何无效"从"经验观察"升级为"理论解释"。

### 10.6 备注
- D2/D3（幅度影响 + 参数化深入）暂不做
- P5 paper 影响: 强化 "Geometric inductive bias is unnecessary" 主张的理论基础

---

## 11. Task #71 — 三流形残差测试方案（A-E + F/G）

> **完整原始内容**: `descriptions/task71_three_manifold_residual.md`
> **Verdict**: `verdicts/task71_three_manifold_residual.md`（如已存在）

### 11.1 目的
验证"用三种不同流形（Euclidean / Poincaré / Spherical）的减法定义残差"是否能产生**本质上不同**的残差向量；若是, 进一步验证这些差异是否能带来实际 R@10 提升。

### 11.2 状态
- **状态**: ❌ 否证 — r_E, r_H, r_S 数值上几乎相同 (cos > 0.95)

### 11.3 关键数字
- A 否证: cos(E,H) ≥ 0.95 或 cos(E,S) ≥ 0.95 → 三残差数值上几乎相同 → **STOP**

### 11.4 方法（7 方案 + 决策路径）

**方案 A** (形式对比): norm mean/std/max, sparsity, PCA rank 95%, kurtosis, cosine sim E-H/E-S
**方案 B** (信息保留): LogisticRegression brand probe accuracy
**方案 C** (聚类结构): Silhouette (best K), Calinski-Harabasz, Davies-Bouldin
**方案 D** (L2 量化): L2 recon MSE + q_L2 选择一致率
**方案 E** (完整三流形 RQ-VAE): 重建 MSE + R@10 on Toys
**方案 F** (数值稳定性): gradient norm, NaN/Inf 频率
**方案 G** (计算成本): wall-clock time, GPU memory

```
A (形式) → STOP 条件 (cos > 0.95)
            ↓ 继续
B (信息) → STOP 条件 (probe diff < 2%)
            ↓ 继续
C (聚类) → STOP 条件 (Silhouette diff < 0.02)
            ↓ 继续
D (L2)   → STOP 条件 (q_L2 完全相同 + MSE diff < 1%)
            ↓ 继续
F (稳定) → STOP 条件 (NaN > 0.1%)
            ↓ 继续
G (成本) → STOP 条件 (wall-clock > 10×)
            ↓ 继续
E (完整) → 最终 R@10 验证
```

### 11.5 否证结论
**R1 否证: 三残差数值上几乎相同**。Task #71 在 A 阶段就触发 STOP, 不需要后续 B/C/D/E。

### 11.6 关键教训（合并后总结）
> **距离函数/残差计算/码书学习三者必须几何一致; 不能凭空假设三层都要套三种几何** (Task #71 踩坑)

---

## 12. 关键教训（合并后总结）

### 12.1 几何改进路线彻底否证链

```
端到端 (Task #59, #60):
  HHHH Poincaré SID → R@10=0.0117~0.0284 << baseline 0.0973

推理时 (Task #62, #64, #66):
  换距离函数 / 几何标签 / 加权距离 → 全部 ❌

训练时 (Task #63):
  黎曼优化码书 → 码书结构相似, Δ ≈ 0

根因诊断 (Task #67, #70):
  距离等价性是根本性质, 非 L2 归一化假象 (Pearson 0.97)

穷尽测试 (Task #71):
  三流形残差 → 数值几乎相同 (cos > 0.95)
```

### 12.2 核心教训

1. **几何一致性原则**: 距离函数/残差计算/码书学习三者必须几何一致。如果 L2 归一化嵌入空间已是各向同性, 套双曲/球面只是数学包装, 实际无差异。

2. **不能凭空假设三层都要套三种几何**: Task #71 踩坑 — 即使每层残差在范数/稀疏度上略有不同, 在 PCA rank/cosine sim 维度上仍等价。三层码书不需要三种不同几何。

3. **保序 ≠ 数值相等**: Task #70 Scenario B 揭示 d_H 是 d_E 的保序单调变换 (d_H[0] ≈ 2 × d_E[0]), 距离值放大但最近邻选择完全相同。

4. **经验观察升级为理论解释**: P5 paper 的 "Geometric inductive bias is unnecessary" 主张从"7 个 task 的经验否证"升级为"距离等价性的理论证明"。

### 12.3 对 P5 paper 的影响

- **Section 4.2**: 加"流形不敏感性"小节（基于 Task #67/70/71）
- **Section 4.3**: 加 Task #67/#69/#70/#71 完整证据链
- **Section 5**: 限制条件讨论（hyperbolic 在 norm → 0 时退化为欧氏）
- **Caveats**: Task #59 端到端 R@10=0.0117 vs baseline 0.0973 的诚实记录

---

## 13. 合并说明表格

| 原 Task | 原文件 | Verdict 文件 | 否证结论 | 合并后章节 |
|---------|--------|-------------|---------|-----------|
| #59 | `task59_hhhh_stage4_r10.md` | `verdicts/task59_result.md` | R@10=0.0117 << 0.0973 | §3 |
| #60 | `task60_hhhh_stage3_longtrain.md` | `verdicts/task60_result.md` | R1 否证, R@10=0.0284 << 0.0973 | §4 |
| #62 | `task62_residual_geometry_analysis.md` | `verdicts/task62_result.md` | L2 归一化下距离等价 | §5 |
| #63 | `task63_riemannian_codebook_geometry.md` | `verdicts/task63_result.md` | 优化器不改码书结构, Δ ≈ 0 | §6 |
| #64 | `task64_geometry_aware_rqvae.md` | `verdicts/task64_geometry_aware_rqvae.md` | 几何指导 MSE +140% | §7 |
| #66 | `task66_weighted_geom_distance.md` | `verdicts/task66_weighted_geom_distance.md` | 加权距离等价单距离 | §8 |
| #67 | `task67_native_space_equiv_info_throughput.md` | `verdicts/task67_deeper_diagnosis.md` | 原生空间下 d_E↔d_H 100% 等价 | §9 |
| #70 | `task70_multigeom_root_cause.md` | `verdicts/task70_multigeom_root_cause.md` | Scenario B: 保序单调变换 | §10 |
| #71 | `task71_three_manifold_residual.md` | `verdicts/task71_three_manifold_residual.md` | 三残差 cos > 0.95 | §11 |
| **#65 (保留)** | `task65_baseline_r10_reproduction.md` | `verdicts/task65_result.md` | R@10=0.0973 baseline 复现 | **不合并, 保留原文件** |

**合并日期**: 2026-07-18
**合并原因**: 9 个 task 共同构成几何改进路线否证链, 合并保留 P5 paper 论证素材, 简化 descriptions/ 目录结构。
**保留文件**: `task65_baseline_r10_reproduction.md` (作为 baseline 对照)
