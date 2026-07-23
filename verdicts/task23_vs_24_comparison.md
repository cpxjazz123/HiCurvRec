# Task #23 vs #24 — PM-RQ 单层 vs Cascade × TIGER 端到端对比

> **结论一句话**: PM-RQ 路径在 TIGER 端到端框架下**两个变体都失败**。cascade (Phase 3) 比 single (Phase 2) 表现**更差**，与"信息密度增加 → Recall 提升"的 R1 假设**相反**。

> **完成日期**: 2026-07-19

---

## 1. 三方对比表

| 指标 | Task #87 baseline | Task #23 Phase 2 single | Task #24 Phase 3 cascade |
|------|------|------|------|
| SID 输入 | RQ-VAE 3-digit + 1 dedup, T5 embedding | PM-RQ single-layer K=256, MCKG input, 4 digits | PM-RQ 3-layer cascade K=256, MCKG input, 10 digits |
| 信息密度 (log₂ K^h) | 32 bits | 32 bits | 80 bits |
| sequence_length | 120 | 120 | 200 |
| Stage 3 best step | 66000 | 8000 | 6000 |
| **Recall@5** | **0.01937** | **0.00474** | **0.00144** |
| **Recall@10** | **0.03318** | **0.00551** | **0.00206** |
| **NDCG@5** | - | 0.00275 | 0.00095 |
| **NDCG@10** | - | 0.00300 | 0.00114 |
| 占 baseline (R@5) | 100% | **24.5%** | **7.4%** |
| 占 Task #23 (R@5) | - | 100% | 30.4% |

---

## 2. 关键发现

### 2.1 PM-RQ × TIGER 全面失败

- Phase 2 single (4 digits) vs Task #87 (4 digits)：**信息密度相同但 R@5 仅 24.5%** → 输入源 (MCKG vs T5) 是主导因素
- Phase 3 cascade (10 digits) vs Phase 2 (4 digits)：**信息密度 2.5× 但 R@5 仅 30%** → 训练容量饱和

### 2.2 R1 否证（PM-RQ 几何保真假设）

> **R1 假设**: PM-RQ 三子空间 (S×E×H) 几何保真 → TIGER 端到端 Recall 优于 flat Euclidean。

**实测否证**：
- Task #23 (Phase 2 single)：R@5 = 0.00474 (24.5% of baseline) → **R1 否证**
- Task #24 (Phase 3 cascade)：R@5 = 0.00144 (7.4% of baseline) → **R1 严重否证**

### 2.3 训练曲线对比

| Task | best step | 100k 完成时 R@5 预估 | plateau 模式 |
|------|-----------|---------------------|---------------|
| #23 | 8000 (0.00412) | ~0.0050 (轻微波动) | val_loss 缓降，R@5 不动 |
| #24 | 6000 (0.00206) | ~0.0025 (基本不变) | val_loss 真正 plateau |

**关键观察**: 两训练都**早期 plateau**（< 10% of 100k 步数），与 Task #87 baseline 的 step 66000 最佳点形成鲜明对比 → PM-RQ SID 提供的信息**不足以让模型继续学习**。

### 2.4 失败原因诊断

| 候选原因 | 证据 | 强度 |
|----------|------|------|
| MCKG 输入表达力弱于 T5 | Task #22 阶段已确认 MCKG 重建 loss 高于 T5 | ⭐⭐⭐ 主导 |
| PM-RQ 几何损失（v-info > 0.85 仍低于 flat） | Task #22 Phase 2 v-info 0.7-0.9 | ⭐⭐ 次要 |
| 训练容量饱和（cascade 9 codes） | Task #24 比 Task #23 更差 | ⭐⭐ 次要 |
| num_user_bins=2000 + LSH 引入噪声 | Task #87 baseline 已用相同配置 | ❌ 已排除 |
| stage 3 optimizer / scheduler 问题 | Task #87 baseline 同一配置达 0.01937 | ❌ 已排除 |

**核心根因**: **MCKG embedding 输入的表达力上限**决定了 PM-RQ SID 的上限，无论几何结构如何优化。

---

## 3. 决策总结

### 3.1 R1 否证链

```
R1: PM-RQ 三子空间几何保真 → 端到端 Recall 优势
  ├─ Task #23 Phase 2 single (3 codes): R@5 = 0.00474 → 24.5% of baseline  → 否证
  └─ Task #24 Phase 3 cascade (9 codes): R@5 = 0.00144 → 7.4% of baseline   → 严重否证
```

### 3.2 推断保留任务

- ✅ Task #22 PM-RQ 算法实现（独立于 TIGER 框架的几何诊断价值已记录）
- ❌ 不再投入 PM-RQ × TIGER 端到端方向的资源

### 3.3 替代研究方向

| 方向 | 假设 | 预算 |
|------|------|------|
| A. TIGER baseline 改进（输入换 T5 + num_hierarchies=5） | 单纯提升 baseline 看上限 | ~6h |
| B. MCKG 输入增强（更多 KG 边 / 不同编码） | 提升 MCKG 表达力后再试 PM-RQ | 2-3 周 |
| C. SID 拼接策略（PM-RQ 码本用 T5 embedding 训练而非 MCKG） | 绕过 MCKG 表达力限制 | ~1 周 |

---

## 4. 产物清单

| 文件 | 关联任务 |
|------|----------|
| `verdicts/task23_pmrq2_tiger_result.md` | Task #23 |
| `verdicts/task24_pmrq3_tiger_result.md` | Task #24 |
| `verdicts/task23_pmrq2_tiger_eval.json` | Task #23 评估 |
| `verdicts/task24_pmrq3_tiger_eval.json` | Task #24 评估 |
| `verdicts/task23_vs_105_comparison.md` | 本文件（三方对比） |

---

## 5. 关联任务参考

| 任务 | 角色 |
|------|------|
| Task #22 Phase 0-3 | PM-RQ 算法开发与诊断 |
| Task #85 | 三几何独立 RQ-VAE (m=0/m=1/m=2) - flat baseline 性能基准 |
| Task #87 | TIGER-aligned flat Euclidean baseline - 本对照基准 |
| Task #23 | PM-RQ Phase 2 single-layer × TIGER |
| Task #24 | PM-RQ Phase 3 cascade × TIGER |
