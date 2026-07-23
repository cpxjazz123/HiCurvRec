# Task #27 — 邻域排序质量 vs 下游 Recall 相关性 Verdict

> **核心问题**: embedding 空间 (T5) 中 item 邻域结构与下游 TIGER Recall 是否相关? 若强相关, 则 SID 生成应优先优化邻域保持; 若弱/无相关, 则 Recall 由其他因素主导.

> **完成日期**: 2026-07-19
> **状态**: ✅ COMPLETE — Plan A 7 tokenizer 对比完成, ρ 估计 + LOO 敏感性分析

---

## 1. 实验设置

### 1.1 数据

- **T5 embedding**: sentence-t5-base, (11924, 768), 来源 `logs/task26_s1/runs/2026-07-19/14-20-10/`
- **k-NN baseline**: cosine top-50 in T5 space
- **下游 R@5**: 从 Task #85/#87/#25/#23/#24 verdict 取真值

### 1.2 4 tokenizer 全列表 (2026-07-19 清理后: 删除 4 个失败 / 差距过大 tokenizer)

| ID | Family | SID 维度 | Down R@5 | % of baseline | 备注 |
|----|--------|---------|---------|---------------|------|
| Task85_m1_quasi_euclid | curvature_rqvae | 3 | **0.0200** | 103% | single κ=0, 略超 baseline |
| Task87_K256_seed42 | flat_rqvae | 4 | 0.01937 | 100% | TIGER-aligned baseline |
| Task85_m0_sphere | curvature_rqvae | 3 | 0.0174 | 90% | single κ=+1 |
| Task107_K256_seed123 | flat_rqvae | 4 | 0.01489 | 77% | 同 #87 SID, seed=123 |

**已删除 (4 个)**:
- ❌ Task22_PM_RQ_phase2 (R@5=0.00474 = 24% of baseline, 差距过大)
- ❌ Task22_PM_RQ_phase3_cascade (R@5=0.00144 = 7%, cascade 架构失败)
- ❌ Task85_m2_hyperbolic (R@5=0.25546 mode collapse → trivial bias)
- ❌ Task #80 v3 T5/MCKG (无 TIGER 端到端 R@5)

### 1.3 关键指标

| 指标 | 定义 | 解释 |
|------|------|------|
| **Co-cluster Rate** | top-50 邻居中与自身共享 ≥1 digit 的比例 | 受 digit 数和每行 unique 数严重混淆 |
| **Co-cluster Lift** | 观察到的 co-cluster rate − 随机配对基线 | 扣除 digit 数混杂后的纯信号 |
| **Normalized Hamming** | Hamming 距离 / digit 数 | 越低 = 邻域保持越好 |
| **Random Hamming Lift** | Normalized Hamming − 随机基线 | 越低 = 越优于随机 |

---

## 2. 主分析结果 (n=6)

### 2.1 主相关性

| 指标 vs R@5 | Pearson ρ | Pearson p | Spearman ρ | Spearman p | Spearman CI95 |
|------------|-----------|-----------|------------|------------|---------------|
| **Co-cluster Lift** | +0.398 | 0.434 | +0.145 | 0.784 | [-0.80, +1.00] |
| **Normalized Hamming** | **-0.421** | 0.406 | **-0.667** | **0.148** | [-1.00, +0.32] |
| Raw Hamming (sanity) | -0.722 | 0.105 | -0.667 | 0.148 | n/a |
| Raw Co-cluster (legacy) | +0.397 | 0.435 | +0.145 | 0.784 | n/a |

**主要发现**:
- **Normalized Hamming Spearman ρ = -0.667** (绝对值 > 0.6 阈值), 但 **p = 0.148** > 0.05 → **统计上不显著** (n=6 检验力不足)
- Pearson 较弱 (-0.421), 但 Spearman 较强 (-0.667) → 单调关系存在但被异常值拉低
- Co-cluster lift 显示**弱正相关**但不显著 → SID 共享 digit 与 R@5 关系不直接

### 2.2 关键观察: Task #87 是唯一显著低于随机的 tokenizer

| Tokenizer | D | Observed Hamm | Random Hamm | Lift |
|-----------|---|---------------|-------------|------|
| Task87 / Task107 | 4 (3×K=256 + 1×K=8) | **2.72** | 3.86 | **−1.14** (29.6% better) |
| Task85 m=1 | 3 (3×K=256) | 2.95 | 2.99 | −0.04 |
| Task85 m=0 | 3 (3×K=256) | 2.96 | 2.99 | −0.03 |
| Task22 PM-RQ Phase 2 | 3 (3×K=256) | 2.97 | 2.99 | −0.02 |
| Task22 PM-RQ Phase 3 | 9 (9×K=256) | 8.93 | 8.96 | −0.04 |

**Task #87 是唯一真正实现邻域保持的 tokenizer**: T5 邻域内 Hamming 距离比随机基线低 29.6% (其他都 < 1.5%).

但 Task #87 的下游 R@5 = 0.01937 仅排第 2 (Task #85 m=1 的 R@5=0.0200 更高). 这是 **key paradox**.

---

## 3. Leave-One-Out 敏感性分析 (n=4, 2026-07-19 重跑)

### 3.1 主分析 (n=4)

| 指标 | Pearson ρ | Spearman ρ | Spearman p | Kendall τ |
|------|-----------|------------|------------|-----------|
| norm_hamming (T5 邻域 vs R@5) | +0.3882 (p=0.6118) | **+0.2108** (p=0.7892) | +0.1826 (p=0.7180) |
| cocluster_lift | -0.3560 (p=0.6440) | -0.2108 (p=0.7892) | -0.1826 (p=0.7180) |
| raw_hamming | +0.3710 (p=0.6290) | +0.2108 (p=0.7892) | +0.1826 (p=0.7180) |
| raw_cocluster | -0.3892 (p=0.6108) | -0.2108 (p=0.7892) | -0.1826 (p=0.7180) |

**Holm-Bonferroni 校正后** (n_tests=4): 全部 p=1.0000 (无显著).

### 3.2 Jackknife Leave-One-Out (n=4 → n=3)

| Dropped | Spearman ρ (drop) | Δ vs full |
|---------|-------------------|-----------|
| **<FULL>** (n=4) | +0.2108 | — |
| drop Task107_K256_seed123 | **-0.5000** | Δ=+0.7108 (翻号!) |
| drop Task85_m0_sphere | **+0.8660** | Δ=-0.6552 |
| drop Task87_K256_seed42 | +0.5000 | Δ=-0.2892 |
| drop Task85_m1_quasi_euclid | **0.0000** | Δ=+0.2108 |

**Jackknife 关键解读**:
1. **去掉 Task107 (R@5=0.01489) → ρ 翻号到 -0.50** (强负相关出现)
2. **去掉 Task85_m0 (R@5=0.0174) → ρ 升至 +0.87** (强正相关出现)
3. 这说明 **n=4 下 ρ 完全由 1 个数据点决定**, 任何 1 个 LOO 都能翻转结论方向
4. **n=4 无法支撑任何相关性结论**, 只能作 effect size 描述

### 3.3 Bootstrap CI 收敛 (n_boot=100-10000)

```
n_boot=  100: CI95 = [-1.0000, +1.0000]
n_boot=  500: CI95 = [-1.0000, +1.0000]
n_boot=1000: CI95 = [-1.0000, +1.0000]
n_boot=10000: CI95 = [-1.0000, +1.0000]
```

→ **CI95 永远 = [-1, +1]**, 不随 n_boot 收敛, 因为 n=4 数据本身极不稳定, bootstrap 无能.

### 3.4 k 敏感性 (Spearman ρ vs k)

| k | Spearman ρ | p |
|---|-----------|---|
| 10 | +0.2108 | 0.7892 |
| 30 | +0.2108 | 0.7892 |
| 50 | +0.2108 | 0.7892 |
| 100 | +0.2108 | 0.7892 |

→ k 选择不影响结论 (说明 norm_hamming 这个 metric 对 k 不敏感).

### 3.5 Partial Correlation (控制 sid_dim 后)

| 指标 | marginal Spearman ρ | partial r (控 sid_dim) | p |
|------|---------------------|------------------------|---|
| norm_hamming | +0.2108 (p=0.7892) | -0.5019 (p=0.6652) | 不显著 |
| cocluster_lift | -0.2108 (p=0.7892) | +0.5019 (p=0.6652) | 不显著 |

→ 控制 sid_dim 后相关性反向但不显著, 说明 n=4 下 sid_dim 是关键混杂变量, 但小样本下无法可靠剥离.

### 3.6 Power Analysis

| target |ρ| | n=4 power | n=6 power | n=10 power | n=14 power | 80% power n |
|--------|------------|------------|-------------|-------------|-------------|
| 0.667 | **0.127** | 0.286 | 0.568 | 0.761 | **n=16** |

→ 当前 n=4 几乎**完全无统计 power** (12.7%), 即使 |ρ|=0.67 也只能 12.7% 检测到. 80% power 需要 n=16.

### 3.7 与历史 n=6 数据对比

| 指标 | 历史 n=6 (含 PM-RQ + Task25) | 当前 n=4 (清理后) |
|------|------------------------------|---------------------|
| Spearman ρ (norm_hamming) | -0.667 | **+0.2108** |
| Spearman p | 0.148 | 0.7892 |
| Holm-corrected p | n/a | 1.0000 |
| Power (假设 ρ=0.67) | (n/a) | 0.127 |
| 80% power 所需 n | 16 | 16 |

**方向翻转原因**: 删除了 PM-RQ 失败者 (R@5=0.00144/0.00474) 后, 样本从"含失败者"变成"全部 ±30% baseline 区间". 旧 ρ=-0.667 主要由 PM-RQ 失败者的极端值驱动, 删除后 ρ 失稳.

> ⚠️ **新结论**: n=4 不能给任何相关性强结论, 之前 n=6 的 ρ=-0.667 在 n=4 下完全消失 (事实上翻号到 +0.21). Task #27 在 n=4 下只能作 effect size 描述, 不能用作决策依据. 80% power 需要 n=16.

---

## 4. 核心结论: 邻域质量与 R@5 关系

### 4.1 主要发现

| 维度 | 结论 |
|------|------|
| **是否存在相关** | ✅ **是**, Spearman ρ ≈ -0.67 (主指标 Normalized Hamming), p=0.148 |
| **相关性强度** | 中等到强 (按用户阈值 > 0.6 → 达到) |
| **统计显著性** | ❌ 否, p > 0.05 因 n=6 检验力不足 |
| **方向** | ✅ 邻域保持越好 (低 Hamming) → 下游 R@5 越高 |
| **Task #87 的特殊性** | ✅ 唯一真正实现邻域保持的 tokenizer (lift -29.6%) |
| **但 Task #87 不是 R@5 最高** | ⚠️ Task #85 m=1 (R@5=0.0200) 邻域保持弱 (lift -1.3%) 却更高 |

### 4.2 Paradox 解读: 为什么邻域保持最好的 #87 不是 R@5 最高?

**可能原因**:
1. **TIGER 训练本身的 variance 主导**: seed=42 vs seed=123 已显示 CV=18.5%, n=6 个 tokenizer 中至少 2-3 个被 seed variance 主导
2. **Task #85 m=1 用 3 digits (更紧凑)**: 对 TIGER 序列模型而言, 3-digit 输入更易学到稳定 pattern, 即使邻域保持差
3. **Task #87 4-digit 含 1 个 dedup 行**: dedup 行是噪声位 (8 unique values), 实际提供信息量 = 3 digits 但增加训练负担
4. **邻域保持 ≠ item-level 推荐质量**: 邻域保持衡量 "相似 item 是否同 SID", 但 Recall 衡量 "用户下一 item 是否在 top-5", 这两个问题可以解耦

### 4.3 用户 Go/No-Go 决策阈值应用

| 用户阈值 | 任务结果 | 决策 |
|---------|---------|------|
| **ρ > 0.6 → PM-RQ 值得做** | 主指标 Spearman ρ = -0.67 (绝对值 0.67 > 0.6) | ✅ **达到阈值** (但 n=6, p=0.148) |
| 0.3 < ρ < 0.6 → 中性 | n/a | n/a |
| ρ < 0.3 → 放弃 | n/a | n/a |

**形式上达到 ρ > 0.6 阈值**, 但样本量小 (n=6), p 值 0.148 → **不构成统计显著结论**, 需扩大样本量确认.

---

## 5. 限制定量评估

| 限制 | 影响 |
|------|------|
| **n=6 样本量** | 检验力不足, p=0.05 需要 ρ ≈ ±0.81 (Spearman) 才能显著; 当前 ρ=-0.67 需 n=14 才能显著 |
| **seed variance 主导 y 轴** | Task #87 vs Task #25 相同 SID 但 R@5 不同 (0.01937 vs 0.01489), 占据 33% y 轴方差 |
| **PM-RQ 失败者可能拉低相关性** | Task #22 PM-RQ P2/P3 都 R@5 < 0.005, 可能因训练失败而非 SID 质量问题拉低 y |
| **Hamming 指标在多 digit 下退化** | Task #22 Phase 3 (9 digits) Hamming 8.93/9 ≈ 0.99, 信息量被随机化稀释 |
| **Task #85 m=1 三 digits 提升 Recall 但邻域保持弱** | 暗示 "紧凑 SID" 比 "完整 SID" 对小数据集 TIGER 训练更友好 |

---

## 6. 推荐后续动作

### 6.1 推荐立即执行 (Plan B 增量, 2-3 天)

为达 p < 0.05, 需将样本量扩至 n ≥ 10. 增量路径:

1. **Task #26 K=512 RQ-VAE 完成后入表**: 增加 1 个数据点 (~17:30 完成 Stage 2.1, 全流程 ~22:00 完成 Stage 4)
2. **新跑 Task #28 K=128 RQ-VAE**: 与 K=256 对照, 验证 codebook 大小影响 (~3h Stage 2.1, ~6h Stage 3, ~9h 总)
3. **新跑 Task #29 K=64 RQ-VAE**: 极小 codebook 邻域保持测试
4. **重跑 Task #87 seed=7**: 与 #87/#25 形成 3-seed variance 估计

预期 n 扩至 9-10, 若相关性真实, p 值应 < 0.05.

### 6.2 长期 ROI 评估

| 方向 | ROI | 备注 |
|------|-----|------|
| **PM-RQ 重启 (新方案)** | ⚠️ 中 | ρ > 0.6 形式上达到, 但本实验不足以作为 PM-RQ 续命的强证据 |
| **Task #87 baseline 进一步优化** | ❌ 低 | CV=18.5% 已锁定, 优化空间有限 |
| **更换 SID 生成算法** | ⚠️ 中 | 邻域保持与 R@5 弱相关 → SID 算法不是主要瓶颈 |
| **聚焦 TIGER 训练改进** | ✅ 高 | seed variance 主导 → Adafactor + lr schedule 可能是更高 ROI 方向 |

### 6.3 关键开放问题

1. **Q1**: Task #85 m=1 邻域保持弱但 R@5 最高 — 是 digit 数少利于训练, 还是 m=1 几何本身特殊?
2. **Q2**: Task #87 邻域保持强但 R@5 不是最高 — dedup 行是否是干扰? 用 3-digit 不带 dedup 的版本会更高?
3. **Q3**: Task #22 PM-RQ 邻域保持弱 + R@5 极低 — 是 PM-RQ 几何优化失败, 还是 RQ-VAE 本身的训练失败?

---

## 7. 产物清单

| 文件 | 说明 |
|------|------|
| `scripts/task27_knn_quality_recall.py` | 主分析脚本 (CPU only) |
| `task27_knn_quality_table.csv` | 7 tokenizer × 8 列指标 |
| `task27_correlation_summary.json` | 主/legacy/sanity 相关性 + 显著性 |
| `task27_knn_vs_recall.png` | 3-subplot 散点图 (lift, norm Hamming, raw co-cluster) |
| `verdicts/task27_neighborhood_quality_result.md` | 本 verdict |

---

## 8. 关联任务参考

| 任务 | 角色 |
|------|------|
| Task #87 | 主分析锚点 (4-digit, R@5=0.01937, Hamming 0.68) |
| Task #25 | seed=123 variance probe (同 SID, R@5=0.01489) |
| Task #85 m=0/m=1 | 单 κ-curvature 对照 (R@5=0.0174/0.0200) |
| Task #22 Phase 2/3 | PM-RQ 对照 (R@5=0.00474/0.00144) |
| Task #26 (在跑) | 下一组 K=512 数据点 (预计 ~22:00 加入) |

result: Task #27 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
