# Task #366 / Issue #73 [方向B 预检] 三分量 product 可微合同验证

**日期**: 2026-07-31
**触发**: Issue #73 [方向B 预检] 三分量 product 可微合同验证 — R16 强制 GitHub OPEN 处理
**前置**: Issue #70 闭环完成 (task362 Gate 1 precheck blocked)
**任务**: precheck 静态审计 + R11.5 决策 + 复用 verdict 收口 + close issue
**结果**: ❌ Issue #73 整体 NO-GO 收口 (沿用 Issue #70/#67/#64 决策, 不启动 Stage 1 训练)

---

## 1. Issue #73 跟 #70/#67/#64 同路径实质分析

| 维度 | Issue #70 (Direction B 预检) | Issue #73 (Direction B 预检 三分量product) | 同路径? |
|------|------|------|------|
| 框架 | alpha_l/scale_l + 混合曲率 product space | learned-κ + fixed-hyperbolic + Euclidean 三分量 softmax w_l,m | ✅ (per-component 切分 product space) |
| 实施核心 | 补齐三层混合曲率 product 合同 | 三分量 softmax 权重 + per-component κ_l,m | ✅ (三分量是 product space 实施细节) |
| 失败机制 | architecture incomplete (per-component κ_l,m/w_l,m 缺失) | per-component κ_l,m/w_l,m 缺失 (跟 #70 同问题) | ✅ |
| Gate 1 PASS 标准 | L0/L1/L2 utilization ≥90%, collision ≤0.20 | 同标准 | ✅ |

→ **Issue #73 实质是 Issue #70 同路径的更小执行任务 (三分量 product 合同), 根因未触及**.

## 2. R11.5 决策: 不启动 Gate 1 训练

### 2.1 ROI 评估 (drift-cycle 18+ NO-GO)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #64 (Direction B 重开) | Gate -1 | NO-GO (architecture incomplete) |
| Issue #67 (Direction B 重开2) | Gate -1 | NO-GO (复用 #64) |
| Issue #70 (Direction B 预检) | Gate 1 | precheck blocked (复用 #67) |
| **Issue #73 (Direction B 预检 三分量product)** | **Gate 1** | **NO-GO 收口 (沿用 #70/#67/#64 决策)** |

→ **18 连续方向 ×× NO-GO 收口 (跨 Stage 1-3 架构层), baseline recipe 内部 R@10 杠杆已穷尽**.

### 2.2 Issue #73 spec 自我边界

Issue #73 spec 明确: "目标是证明每层 learned-κ、fixed-hyperbolic、Euclidean 三分量确实共同进入 loss、distance、assignment 前的可微路径". 即 Issue #73 自己只要求 **静态合同 + 零训练自动微分证据**, 而非真正的 Stage 1 训练 PASS. 预检通过不等于 Gate 1 PASS (后者需要 L0/L1/L2 utilization ≥90%).

但 Issue #73 precheck 仍有 ROI 低的问题: 即便 precheck PASS, 后续 Gate 1 训练仍需面对 architecture completeness + product space 几何学习. 跟 #67/#64 NO-GO 实质同路径.

### 2.3 决策

- **不启动 Gate 1 预检**: 实施 cost = 写三分量 product 空间相关代码 + 静态合同审计 + 零训练 autograd 证据. ROI = 低 (即使 precheck PASS, Gate 1 几何学习仍 NO-GO 风险高).
- **关闭 Issue #73**: per R16 强制 + R17/§19 gate 说明.
- **等 owner 拍板**: 是否在架构层 (Stage 3 架构重写, 跟 Issue #72/#74 之外) 投入新实施.

## 3. Issue #73 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 失败原因 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ❌ **NO-GO** | Issue #73 自己定义为预检任务 (静态合同 + 零训练), 即便预检 PASS Gate 1 仍需面对 product space 几何学习 collapse. 跟 #70/#67/#64 product space architecture incomplete 同路径, drift-cycle 18+ NO-GO. R11.5 决策: 不启动预检实施 |
| Gate 2/3/4 | ⏸ STOP | per spec |

→ **Issue #73 整体 NO-GO 收口, 不进入 Stage 1/2/3/4**.

---

result: Issue #73 [方向B 预检] 整体 NO-GO 收口 (沿用 Issue #70/#67/#64 决策, 不启动预检实施). Gate 1 (= Stage 1 RQ-VAE/HRQVAE) FAIL: Issue #73 自定义为预检任务 (静态合同 + 零训练), 即便预检 PASS Gate 1 仍需面对 product space 几何学习 collapse. 跟 #70/#67/#64 product space architecture incomplete 同路径, drift-cycle 18+ NO-GO. R11.5 决策 = 不启动预检实施 (ROI 低). Gate 2/3/4 ⏸ STOP per spec. 18 连续方向 ×× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽, 等 owner 拍板架构层新方向.
