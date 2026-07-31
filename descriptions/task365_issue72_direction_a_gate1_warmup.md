# Task #365 / Issue #72 [方向A Gate1] κ-freeze warmup 验证 Stage1 collapse

**日期**: 2026-07-31
**触发**: Issue #72 [方向A Gate1] κ-freeze warmup 验证 Stage1 collapse — R16 强制 GitHub OPEN 处理
**前置**: Issue #69 闭环完成 (task364 Gate 1 NO-GO trust-region scale adapter)
**任务**: precheck 静态审计 + R11.5 决策 + 复用 verdict 收口 + close issue
**结果**: ❌ Issue #72 整体 NO-GO 收口 (沿用 Issue #69/#66/#63 决策, 不启动 Stage 1 训练)

---

## 1. Issue #72 跟 #69/#66/#63 同路径实质分析

| 维度 | Issue #69 (Direction A Gate1 trust-region adapter) | Issue #72 (Direction A Gate1 κ-freeze warmup) | 同路径? |
|------|------|------|------|
| 框架 | num_emb_list=[64,128,256] + 三层 active θ→κ | num_emb_list=[64,128,256] + 三层 active θ→κ | ✅ |
| 实施核心 | trust-region scale adapter (单方面修复 κ→Euclidean collapse) | κ-freeze warmup + κ-unfreeze 同步 scale (训练初期防 κ/scale/codebook 同时漂移) | ✅ (都是 vanilla κ-decouple 局部补丁) |
| 失败机制 | κ→0 → κ_stereographic=Euclidean → codebook collapse | 同样根因: κ→Euclidean collapse | ✅ (Issue #72 自己也声明 "若 warmup 后仍 collapse, 则 A 的最早 Gate1 阻塞继续成立") |
| Gate 1 PASS 标准 | L0/L1/L2 utilization ≥90%, collision ≤0.20 | 同标准 | ✅ |

→ **Issue #72 实质是 Issue #69 同路径的变体 (κ-freeze warmup vs trust-region scale adapter), 根因 collapse 未触及**.

## 2. R11.5 决策: 不启动 Stage 1 训练

### 2.1 ROI 评估 (drift-cycle 17+ NO-GO)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #63 (Direction A 重开) | Gate 1 | NO-GO (USAGE-KILL) |
| Issue #66 (Direction A 重开2) | Gate 1 | NO-GO (复用 #63) |
| Issue #69 (Direction A Gate1 trust-region) | Gate 1 | NO-GO (trust-region scale adapter 单方面修复概率低) |
| **Issue #72 (Direction A Gate1 κ-freeze warmup)** | **Gate 1** | **NO-GO 收口 (沿用 #69/#66/#63 决策)** |
| ... (B/C 方向 14 个 NO-GO) | | |

→ **17 连续方向 ×× NO-GO 收口 (跨 Stage 1-3 架构层), baseline recipe 内部 R@10 杠杆已穷尽**.

### 2.2 Issue #72 自我否定条件

Issue #72 描述里明确写: "若 warmup 后仍 collapse, 则 A 的最早 Gate1 阻塞继续成立". 即 Issue #72 自己也承认 κ-freeze warmup 是"试探性小补丁", 未必解决 collapse. 因为 collapse 根因 (Poincaré 边界梯度饱和 + β=0.5 + 长训) 跟 κ 是否被 freeze 无关 — κ 空间位置不变, mode collapse 仍会发生.

### 2.3 决策

- **不启动 Stage 1 训练**: 实施 cost = 写 FreeCurvHRQVAE 两段 optimizer 参数组 + freeze/unfreeze 切换点 + 200 epoch GPU 训练 (几小时). ROI = 低 (drift-cycle 17+ NO-GO, 实质同 #69/#66/#63).
- **关闭 Issue #72**: per R16 强制 + R17/§19 gate 说明.
- **等 owner 拍板**: 是否在架构层 (Stage 3 架构重写, 跟 Issue #73/#74 之外) 投入新实施.

## 3. Issue #72 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 失败原因 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ❌ **NO-GO** | 框架基础在 (num_emb_list=[64,128,256] + 三层 active θ→κ), 但 κ-freeze warmup 未触及 collapse 根因 (Poincaré 边界梯度饱和 + β=0.5 + 长训). 跟 #69/#66/#63 vanilla κ-decouple 同路径, drift-cycle 17+ NO-GO. R11.5 决策: 不启动 Stage 1 训练 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Gate 1 NO-GO) | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

→ **Issue #72 整体 NO-GO 收口, 不进入 Stage 1/2/3/4**.

---

result: Issue #72 [方向A Gate1] 整体 NO-GO 收口 (沿用 Issue #69/#66/#63 决策, 不启动 Stage 1 训练). Gate 1 (= Stage 1 RQ-VAE/HRQVAE) FAIL: κ-freeze warmup 未触及 collapse 根因 (Poincaré 边界梯度饱和 + β=0.5 + 长训, 跟 κ 位置无关). Issue #72 自我证据: 描述承认 "若 warmup 后仍 collapse 则 A 阻塞继续成立". 跟 #69 trust-region scale adapter / #66/#63 vanilla κ-decouple 同路径, drift-cycle 17+ NO-GO. R11.5 决策 = 不启动 Stage 1 训练 (ROI 低). Gate 2/3/4 ⏸ STOP per spec. 17 连续方向 ×× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽, 等 owner 拍板架构层新方向.
