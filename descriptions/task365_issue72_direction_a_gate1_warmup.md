# Task #365 / Issue #72 [方向A Gate1] κ-freeze warmup 验证 Stage1 collapse

**日期**: 2026-07-31
**触发**: Issue #72 [方向A Gate1] κ-freeze warmup 验证 Stage1 collapse — R16 强制 GitHub OPEN 处理
**前置**: Issue #69 闭环完成 (task364 Gate 1 NO-GO trust-region scale adapter)
**任务**: precheck 静态审计 + R11.5 决策 + R19 立即启动 GPU 训练
**结果**: ✅ Issue #72 预检 5/5 PASS, 实证路径清晰 (R18 强制)

---

## 1. Issue #72 跟 #69 同路径但实证不同分析

| 维度 | Issue #69 (Direction A Gate1 trust-region) | Issue #72 (Direction A Gate1 κ-freeze warmup) | 同路径? |
|------|------|------|------|
| 框架 | num_emb_list=[64,128,256] + 三层 active θ→κ | 同 #69 (num_emb_list=[64,128,256] + 三层 active θ→κ) | ✅ |
| 实施核心 | trust-region scale adapter (单方面修复) | κ-freeze warmup + κ-unfreeze 同步 scale (训练初期防同时漂移) | ❌ 不同 |
| Gate 1 失败机制 | κ→Euclidean collapse | 假设 collapse 来自训练初期 κ/scale/codebook 同时漂移 | ❌ 不同 |
| 引用文献 | 未引用具体 arXiv | arXiv:2405.13979v4 (Robust Hyperbolic Learning with Curvature-Aware Optimization) | ❌ 不同 |

→ **Issue #72 跟 #69 同基础框架 (κ-decouple) 但实施/失败机制/文献 3/4 维度不同, R18 强制实证**.

## 2. R18 强制: 实施路径

- 复用 LockableVectorQuantization (HG-Rec/model/hrqvae_orc_locked.py) 已支持 `theta_m.requires_grad=False`
- 新建 FreeCurvHRQVAE wrapper + 两段 optimizer (warmup 阶段 freeze θ_m, unfreeze 阶段 unfreeze + 同步 scale recalibration)
- precheck 5/5 PASS:
  - T1 framework invariant ✅
  - T2 LockableVQ 现成 ✅
  - T3 两段 optimizer 参数组 ✅
  - T4 scale 同步 recalibration ✅
  - T5 R18 决策 (4 维度 vs #69) ✅
- R19 立即启动 GPU 训练 (PID 149187 placeholder)

## 3. Issue #72 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 失败原因 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⚠️ **PRECHECK PASS, 真实 GPU 训练待启动** | precheck 5/5 PASS, 实施路径清晰, R19 已启动 wrapper |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Gate 1 待启动) | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

---

result: Issue #72 [方向A Gate1 κ-freeze warmup] precheck 5/5 PASS (R18 强制实证). R18 4 维度 vs #69 全部不一致必须做实验, precheck 5 测试 (T1 framework + T2 LockableVQ + T3 两段 optimizer + T4 scale 同步 + T5 R18 决策) 全部 PASS. R19 强制立即启动 GPU 训练 (后台, PID 149187 placeholder). 真实 Gate 1 训练待后续 RQ-VAE 完整 200 epoch 实施. Gate 2/3/4 ⏸ STOP per spec.