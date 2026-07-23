# Task #64 — G2: 内在几何场诊断 → 局部率分配 (Type III 完整结构, 主论文候选)

> **任务目的**: 验证 G2-H1/H2: 量化误差与下游 per-item 召回失败在 embedding 空间中不均匀分布, 且可由三个非参数化内在几何量 (LID, Ollivier-Ricci, Gromov δ-双曲性) 预测; G2-H3: 按几何场做变率分配 (variable-rate SID) 优于均匀分配
> **执行日期**: (待启动, 优先级 3/5)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #58-#61 Campaign + G1 (Task #63). 这是 PM-RQ 的**倒置**:
- PM-RQ 把常曲率流形结构强加给数据 (V1-V3 falsified)
- G2 反过来从数据非参数地测几何, 再让码率跟着几何走
- **不假设任何全局流形**

**核心假设**:
- **G2-H1 (现象)**: 量化误差与下游 per-item 召回失败在 embedding 空间中**不均匀分布**, 可由三个非参数化内在几何量预测
- **G2-H2 (机制)**: 固定码率的均匀量化器隐含假设内在维度恒定; 真实数据的 $d_{\mathrm{loc}}(x)$ 是一个场, 高维区域按 Zador 律吃掉超线性的码率
- **G2-H3 (解法)**: 按几何场做变率分配 (variable-rate SID) 优于均匀分配

---

## 2. 实验设计

**变量**: 量化方式 (固定均匀 → 几何驱动变率)
**保持不变**:
- Stage 1 embedding (复用 Task #59/#60/#61 多源)
- seed=42, K=256
- Stage 3 TIGER 配置

**三个几何场估计公式**:
- (a) 逐点 LID (Levina–Bickel MLE): $\hat{d}_k(x) = \big[ \frac{1}{k-1} \sum_{j=1}^{k-1} \log \frac{r_k(x)}{r_j(x)} \big]^{-1}$
- (b) kNN 图 Ollivier-Ricci 曲率: $\kappa_{\mathrm{OR}}(x,y) = 1 - W_1(m_x^\alpha, m_y^\alpha)/d(x,y)$
- (c) 局部 Gromov δ-双曲性 (四元组采样)

**启动命令**:
```bash
# D0: LID vs ε_i Spearman (1 份 embedding, 1-2 天, 纯 CPU + sklearn)
python3 scripts/task64_lid_spearman.py \
    --embeddings logs/task59_s1/.../merged_predictions_tensor.pt \
    --rq_recon logs/task61_s2_infer/pickle/cluster_ids.pt \
    --centers logs/task61_s2_infer/pickle/cluster_centers_layer{0,1,2}.npy \
    --out_json verdicts/task64_lid_spearman.json

# P1: 全诊断矩阵 (3 场 × 多源 embedding)
python3 scripts/task64_full_diagnostic.py \
    --embeddings [Task #58, #59, #60, #61, #87 路径列表] \
    --rq_outputs [对应 RQ 中心点列表] \
    --out_csv verdicts/task64_full_diagnostic.csv

# P2: 几何驱动变率 SID (区域容量版, 改动小)
python3 scripts/task64_variable_rate_kmeans.py \
    --embedding logs/task59_s1/.../merged_predictions_tensor.pt \
    --lid_csv verdicts/task64_full_diagnostic.csv \
    --output_dir logs/task64_s2/pickle \
    --K_total 256 --num_hierarchies 3 --seed 42

# P3: 端到端 TIGER
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task64_s2/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task64_s3_train

# Stage 4 + 评估 (与 Task #58 同)
```

---

## 3. 决策触发 (vs 提案 D0 GO 条件)

| D0 Spearman | 决策 |
|-------------|------|
| \|ρ\| > 0.3 且 p < 0.01 (item 级 n≈12K, 统计力充足) | ✅ GO → P1 全诊断矩阵 + P2 变率原型 + P3 TIGER |
| \|ρ\| 在 0.15-0.3 之间 | ⚠️ PARTIAL → 调 k (近邻数) 到 {10, 20, 50} 三档报稳健性 |
| \|ρ\| < 0.15 (无相关) | ❌ NO-GO → 关闭 G2 几何场主线, 仅作为辅助诊断 |
| P3 R@5 ≥ 0.0977 (Task #61) | ✅✅ 变率 SID 端到端胜利 |
| P3 R@5 < 0.0977 | ⚠️ 诊断成立但解法未带来下游收益, 留作 verdict 反例 |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| P0 文献核查 (1 周) | 1 天 | 无 |
| **D0** LID Spearman (1 份 embedding) | **1-2 天** | 无 (sklearn CPU) |
| P1 全诊断矩阵 (3 场 × 5 源) | 3 天 | 无 (kNN 图, Ollivier-Ricci 慢, 局部 δ-双曲性四元组采样) |
| P2 变率 SID 原型 (区域容量版) | 5 天 | cuda:0 (~30 min) |
| P3 Stage 3 TIGER | 3 天 | cuda:1 (~120 min) |
| P3 Stage 4 + 评估 | 5 min + 1 min | cuda:1 |
| **总计** | **~14 天** | **~5 GPU-hour** |

---

## 5. 风险与缓解

**风险 1**: Ollivier-Ricci 曲率计算复杂度高 ($W_1$ on kNN graphs, O(k^3) per edge) → 用 NetworkX + 近似算法, 或 POT 库的 Sinkhorn 替代精确 W_1
**风险 2**: 局部 δ-双曲性需要大量四元组采样 (n choose 4 太大) → 子采样 + bootstrap
**风险 3**: 变率 SID 需要修改 Stage 3 模型接受可变长度序列 → 若改动大, 先做"区域容量版" (固定长度, 但每区域 K 不同)
**风险 4**: G2-H2 验证需要现成 RQ 运行的逐 item 误差 (Task #61 已存档) + per-item 下游指标 (战线四 TIGER 运行日志, 需先确认可访问)

---

## 6. 完成度跟踪

- [ ] P0 文献核查 (检索词 `local intrinsic dimensionality quantization error`, `variable rate semantic ID`, `curvature-aware vector quantization`, `Ollivier-Ricci recommendation embedding`)
- [ ] D0 写 task64_lid_spearman.py + 跑 Spearman
- [ ] D0 GO/NO-GO 判定 (|ρ| > 0.3 且 p < 0.01)
- [ ] P1 全诊断矩阵 (3 场 × 5 源)
- [ ] P2 变率 SID 原型
- [ ] P3 Stage 3 TIGER 训练
- [ ] P3 Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 verdict + 更新 loop.md §16

---

## 7. 重叠审计

- ✅ 与 D0 旧诊断 (norm_cv, subspace cosine) 不同 — 子空间级 vs G2 item 级场
- ✅ 与战线一 QMP 不同 — QMP 数据集级标量 vs G2 空间上的场
- ✅ 与已杀 "anisotropic collapse" 方向不同 — 全局各向异性 vs 局部场
- ✅ PM-RQ 教训被吸收而非重复: 不再假设参数化几何
- ✅ 与幸存 idea "ambiguity-aware redistribution" 天然交叉验证 (几何驱动 vs 歧义驱动)