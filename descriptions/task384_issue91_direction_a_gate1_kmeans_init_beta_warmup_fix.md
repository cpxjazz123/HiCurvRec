# Task #384 / Issue #91 [方向A Gate1] kmeans_init + β=0 first epoch 修复 step1 坍缩

**日期**: 2026-07-31
**触发**: Issue #91 [方向A Gate1] kmeans_init 与 β warmup 修复验证 — R16 + R22 强制立即开工
**前置**: Issue #88 Gate 1 诊断 PASS (commit d421beb, 最早坍缩 step 1, 第一次 gradient step 推动)
**任务**: 1. R18 4 维度对比 vs Issue #88; 2. kmeans_init=True + β=0 first epoch 修复训练; 3. 验证 step1 不再坍缩; 4. Gate 1 PASS 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #91 vs Issue #88)

| 维度 | Issue #88 (坍缩根因 trace 诊断 PASS) | Issue #91 (kmeans_init + β warmup 修复验证) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | per-step trace 定位最早坍缩点 (诊断) | kmeans_init + β=0 first epoch 修复 + 验证 step1 不再坍缩 (修复) | ❌ 不同 |
| **D2 实施核心** | baseline recipe 10 epoch + per-step trace (无修复) | baseline recipe + kmeans_init=True + β=0 first epoch (修复) | ❌ 不同 |
| **D3 失败机制假设** | commitment loss + codebook loss 第一次 gradient 推动坍缩 | kmeans_init 用真实数据点初始化 + β=0 first epoch 跳过 commitment loss, 让码字不被推到数据几何中心 | ❌ 不同 |
| **D4 引用文献** | arXiv:2405.13979v4 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做修复实证 (R18 强制)**.

## 2. Issue #91 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **kmeans_init 替代 random init** | 用真实数据点做 kmeans 初始化 codebook | Issue #91 spec Gate1 |
| **β=0 first epoch** | 第一 epoch commitment loss weight = 0 (只走 recon loss) | Issue #91 spec Gate1 |
| **保留三层 learnable curvature** | L0 K64, L1 K128, L2 K256, 三层 learnable theta_l | Issue #91 spec 框架合规 |
| **报告** | step1/step50/epoch0/最后 epoch 的 L0/L1/L2 util, max_load, collision, codebook norm, pairwise dist, top-k margin, soft entropy, theta/kappa, grad_codebook | Issue #91 spec Gate1 |
| **PASS** | step1 不出现单码字 >50% load; epoch0 后 L0/L1/L2 util 明显高于 #88 对照; 最终三层 util ≥90%; collision_rate ≤0.20; 无 NaN/Inf | Issue #91 spec Gate1 PASS |
| **FAIL** | step1 仍立即中心坍缩; 任一层 util <90%; 或修复未作用于 #88 第一次 gradient 更新点 | Issue #91 spec Gate1 FAIL |

## 3. Issue #91 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 修复验证)** | ⏳ 进行中 | kmeans_init + β=0 first epoch + step1 验证 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per Issue #91 spec | Gate 1 PASS 后才允许 |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per Issue #91 spec | Gate 1+2 PASS 之后 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per Issue #91 spec | Gate 1+2+3 PASS 之后 |

---

## 4. 实施策略 (R11.5 自主决策)

- 复用 task378 FreeCurvHRQVAE 配置 + kmeans_init=True (替代 baseline recipe 的 kmeans_init=False)
- β schedule: epoch 0 用 β=0 (只走 recon), epoch 1+ 用 β=0.25 (标准 commitment)
- 50 epoch 训练 + per-step trace 跟 task381 同样格式
- GPU 0 立即启动 (R7 + R19 激进, 4 卡空闲)

---

result: Issue #91 [方向A Gate1 kmeans_init + β=0 first epoch 修复 step1 坍缩] R22 + R19 立即开工 (GPU 0). 3 步: 1) kmeans_init=True + β warmup; 2) per-step trace 验证 step1; 3) Gate 1 验证 + commit + push + comment(含 hash) → close. ⏳ 进行中.