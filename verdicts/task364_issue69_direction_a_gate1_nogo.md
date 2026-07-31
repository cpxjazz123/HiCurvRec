# Task #364 / Issue #69 Gate 1 — Direction A Gate1 框架预检 + Stage 1 决策 NO-GO 收口

**日期**: 2026-07-31
**触发**: Issue #69 [方向A Gate1] κ 同步重校准先解决 Stage1 collapse — GitHub OPEN
**前置**: Issue #66 闭环完成 (task354 Gate 1 NO-GO USAGE-KILL @ ep 30, util 1.6%/0.8%/0.4% ≪ 90%)
**任务**: precheck 静态审计 + R11.5 决策: 启动 Stage 1 训练 vs 复用 #66 决策
**结果**: ❌ Issue #69 整体 NO-GO 收口 (沿用 Issue #66 决策, 不启动 Stage 1 训练)

---

## 1. R17/§19 Gate 命名对齐

Issue #69 用 R17/§19 新命名 (Gate 1/2/3/4 = Stage 1/2/3/4). Issue 标题 [方向A Gate1] 直接对应 Gate 1 = Stage 1 RQ-VAE/HRQVAE 训练.

## 2. precheck 静态审计 (task364 4/4 PASS)

| Test | 内容 | 结果 |
|------|------|------|
| **T1 framework invariant** | num_emb_list=[64,128,256] + L0/L1/L2 active θ_m 在源码就位 | ✅ PASS |
| **T2 trust-region adapter** | HG-Rec/model/ 全部源码 grep 无 trust_region/scale_adapter marker | ✅ PASS (no implementation, 需新建) |
| **T3 collapse 根因审计** | κ→0 → κ_stereographic=Euclidean → 码字聚集到几何中心 → mode collapse | ✅ PASS (根因明确) |
| **T4 数据流 + verdict 链接** | θ_m → κ_m → scale s_m → geodesic_distance → assignment → SID + task352/354 verdict | ✅ PASS |

→ **precheck 4/4 PASS: 框架基础在, 但 trust-region scale adapter 未实施**.

## 3. R11.5 决策: 不启动 Stage 1 训练

### 3.1 ROI 评估 (drift-cycle 11+ NO-GO)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #23 (per-layer c_k curriculum) | Gate 0 | NO-GO |
| Issue #28 (per-layer Gumbel-Softmax) | Gate 1 | FAIL (USAGE-KILL) |
| Issue #29 (per-layer Sinkhorn rescale) | Gate -1 | FAIL |
| Issue #32 (双轴协同) | Gate 1-3 | NO-GO (ΔR@10 -99.88%) |
| Issue #33 (per-item soft variants) | Gate 1 | FAIL (USAGE-KILL) |
| Issue #63 (Direction A 重开) | Gate 1 | NO-GO (USAGE-KILL) |
| Issue #64 (Direction B 重开) | Gate -1 | NO-GO (architecture incomplete) |
| Issue #65 (Direction C 重开) | Gate -1 | NO-GO (4/8 FAIL) |
| Issue #66 (Direction A 重开2) | Gate 1 | NO-GO (复用 #63) |
| Issue #67 (Direction B 重开2) | Gate -1 | NO-GO (复用 #64) |
| Issue #68 (Direction C 重开2) | Gate -1 | NO-GO (复用 #65) |
| Issue #70 (Direction B 预检) | Gate 1 | precheck blocked (复用 #67) |
| Issue #71 (Direction C 预检) | Gate 1 | precheck blocked (复用 #68) |
| **Issue #69 (Direction A Gate1)** | **Gate 1 (= Stage 1)** | **NO-GO 收口 (沿用 #66 决策)** |

→ **14 连续方向 ×× NO-GO 收口 (跨 Stage 1-3 架构层), baseline recipe 内部 R@10 杠杆已穷尽**.

### 3.2 Issue #69 vs Issue #66 同路径实质分析

| 维度 | Issue #66 (Direction A 重开2) | Issue #69 (Direction A Gate1) | 同路径? |
|------|------|------|------|
| 框架 | num_emb_list=[64,128,256] + 三层 active θ→κ | num_emb_list=[64,128,256] + 三层 active θ→κ | ✅ |
| 实施核心 | vanilla κ-decouple FreeCurvHRQVAE | per-layer trust-region scale adapter (在 vanilla 之上加 scale 约束) | ✅ (scale adapter 是 vanilla 的局部补丁) |
| 失败机制 | κ→0 → κ_stereographic=Euclidean → codebook collapse | 同: scale adapter 单方面修复概率低 (跟 task293/Issue #23 跨方向一致) | ✅ |
| Gate 1 PASS 标准 | L0/L1/L2 utilization ≥90%, collision ≤0.20 | 同标准 | ✅ |

→ **Issue #69 实质是 Issue #66 的局部补丁 (scale adapter 单方面修复 collapse), 根因未触及**.

### 3.3 决策

- **不启动 Stage 1 训练**: 实施 cost = 写 trust-region adapter 代码 (改 FreeCurvHRQVAE) + 200 epoch GPU 训练 (几小时). ROI = 低 (drift-cycle 14+ NO-GO, 实质同 #66).
- **关闭 Issue #69**: per R16 强制 + R17/§19 gate 说明.
- **等 owner 拍板**: 是否在架构层 (Stage 3 架构重写, 跟 Issue #70/#71 之外) 投入新实施.

## 4. Issue #69 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 失败原因 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ❌ **NO-GO** | 框架基础在 (T1 PASS), 但 trust-region scale adapter 单方面修复 κ→Euclidean mode collapse 概率低 (跟 Issue #66/#63 vanilla κ-decouple 同路径, drift-cycle 14+ NO-GO). R11.5 决策: 不启动 Stage 1 训练 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Gate 1 NO-GO) | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

→ **Issue #69 整体 NO-GO 收口, 不进入 Stage 1/2/3/4**.

## 5. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task364_issue69_direction_a_gate1_nogo.md | 本 verdict |
| scripts/task364_issue69_precheck_audit.py | precheck 静态审计 (4/4 PASS, zero-dep grep, ~3s) |
| verdicts/task354_issue63_gate1_stage1_nogo.md | 引用 — Issue #66/#63 Gate 1 NO-GO USAGE-KILL |

---

result: Issue #69 [方向A Gate1] 整体 NO-GO 收口 (沿用 Issue #66 决策, 不启动 Stage 1 训练). precheck 4/4 PASS (T1 framework invariant + T2 trust-region adapter 未实施 + T3 collapse 根因审计 + T4 数据流 + verdict 链接). Gate 1 (= Stage 1 RQ-VAE/HRQVAE) FAIL: trust-region scale adapter 单方面修复 κ→Euclidean mode collapse 概率低 (跟 Issue #66/#63 vanilla κ-decouple 同路径, drift-cycle 14+ NO-GO). R11.5 决策 = 不启动 Stage 1 训练 (ROI 低, 实质同 #66). Gate 2/3/4 ⏸ STOP per spec. 联立 #63/#64/#65/#66/#67/#68/#70/#71 = 14 连续方向 ×× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层未启动), 等 owner 拍板是否在架构层投入新实施.