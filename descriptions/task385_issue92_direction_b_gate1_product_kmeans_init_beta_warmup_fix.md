# Task #385 / Issue #92 [方向B Gate1] product 路径 kmeans_init + β=0 first epoch 修复验证

**日期**: 2026-07-31
**触发**: Issue #92 [方向B Gate1] product 路径 kmeans_init 与 β warmup 修复验证 — R16 + R22 强制立即开工
**前置**: Issue #89 Gate 1 诊断 PASS (commit b5fb4db, component-level collapse @ step 1)
**任务**: 1. R18 4 维度对比 vs Issue #89; 2. kmeans_init + β=0 first epoch 修复 Product3CompHRQVAE; 3. 验证 step1 agree3 < 100%; 4. Gate 1 PASS 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #92 vs Issue #89)

| 维度 | Issue #89 (三分量 product 分离诊断 PASS) | Issue #92 (product 路径 kmeans_init + β warmup 修复) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | per-component trace 3 选 1 判定 (诊断) | kmeans_init + β=0 first epoch 修复 + 验证 component argmin agreement (修复) | ❌ 不同 |
| **D2 实施核心** | baseline recipe 10 epoch + per-component trace (无修复) | baseline recipe + kmeans_init=True + β=0 first epoch (修复) | ❌ 不同 |
| **D3 失败机制假设** | component-level collapse @ step 1, 三 component 都指向同一最近点 | kmeans_init 用真实数据点初始化 + β=0 first epoch 让码字不被推到中心, 三个 component 应该指向不同码字 | ❌ 不同 |
| **D4 引用文献** | arXiv:2307.04514 + DOI 10.1109/ICDM51629.2021.00021 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做修复实证 (R18 强制)**.

## 2. Issue #92 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **保留三分量 product 结构** | learned hyperbolic + fixed hyperbolic + Euclidean + learnable mixing logits | Issue #92 spec 框架合规 |
| **kmeans_init + β=0 first epoch** | 用真实数据点初始化 codebook, first epoch β=0 | Issue #92 spec Gate1 |
| **不得删除三分量结构** | 必须保持三个 component score 在 assignment 前组合 | Issue #92 spec 框架合规 |
| **报告** | step1/step50/epoch0/最后 epoch 的 per-layer/per-component trace: agree3, mixed top-k margin, mixing weights, theta/kappa, codebook pairwise dist, util, collision, grad norm | Issue #92 spec Gate1 |
| **PASS** | step1 agree3 不再为 100%; mixing weights 仍非单分量 >0.98; L0/L1/L2 util 最终 ≥90%; collision ≤0.20; 无 NaN/Inf | Issue #92 spec Gate1 PASS |
| **FAIL** | component argmin 仍 step1 全一致; mixing detach; 任一层 util <90%; 或只改 product 结构不修 codebook/update 根因 | Issue #92 spec Gate1 FAIL |

## 3. Issue #92 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 修复验证)** | ⏳ 进行中 | kmeans_init + β=0 first epoch + per-component step1 验证 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per Issue #92 spec | Gate 1 PASS 后才允许 |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per Issue #92 spec | Gate 1+2 PASS 之后 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per Issue #92 spec | Gate 1+2+3 PASS 之后 |

---

## 4. 实施策略 (R11.5 自主决策)

- 复用 task382 Product3CompHRQVAE wrapper + kmeans_init=True
- β schedule: epoch 0 用 β=0 (只走 recon), epoch 1+ 用 β=0.25
- 50 epoch 训练 + per-step per-component trace 跟 task382 同样格式
- GPU 1 立即启动 (R7 + R19 激进, 4 卡空闲)

---

result: Issue #92 [方向B Gate1 product 路径 kmeans_init + β=0 first epoch 修复验证] R22 + R19 立即开工 (GPU 1). 3 步: 1) Product3CompHRQVAE + kmeans_init=True + β warmup; 2) per-component trace 验证 step1 agree3 < 100%; 3) Gate 1 验证 + commit + push + comment(含 hash) → close. ⏳ 进行中.