# Task #362 / Issue #70 Gate 1 — Direction B 预检 blocked 复用 #67 闭环整体 NO-GO 收口

**日期**: 2026-07-31
**触发**: Issue #70 [方向B 预检] 补齐三层混合曲率 product 合同 — GitHub OPEN
**前置**: Issue #67 闭环完成 (task355 Gate -1 NO-GO architecture incomplete)
**任务**: 验证 Issue #70 spec 跟 Issue #67 spec 一致性 + 复用 verdict 收口
**结果**: ❌ Issue #70 整体 NO-GO 收口 (沿用 Issue #67 决策, precheck blocked)

---

## 1. R17/§19 Gate 命名对齐

Issue #70 用 R17/§19 新命名 (Gate 1/2/3/4 = Stage 1/2/3/4). spec 描述 "预检未 PASS 前禁止执行 Gate1". 即 "precheck blocked" 状态 = Gate 1 之前的预检 FAIL, 不允许进入 Gate 1 Stage 1 训练.

Issue #67 (重开2) 用历史命名 (Gate -1/0/1/2/3), Gate -1 architecture incomplete FAIL = 当前 Issue #70 的 precheck blocked.

## 2. Issue #70 spec vs Issue #67 spec 对比

| 维度 | Issue #67 (重开2) | Issue #70 (预检) | 一致 |
|------|------|------|------|
| 框架 | num_emb_list=[64,128,256], L0/L1/L2 learnable θ→κ + 三分量 (learned/hyper/euc) | num_emb_list=[64,128,256], 三层 learnable θ→κ + fixed-hyperbolic + Euclidean 三分量 | ✅ |
| 实施 | 三分量 product score + 逐层 mixing logits/weights | 三分量 product score + 受约束 mixing logits (e.g. w_l=softmax(logits_l)) + 不得 detach | ✅ |
| Gate 1 = Stage 1 | 三层 utilization ≥90%, collision ≤0.20, θ + mixing 非边界变化 | L0/L1/L2 utilization ≥90%, collision_rate ≤0.20, θ + mixing 至少一项非边界变化 | ✅ |
| Gate 2 = Stage 2 | 4-digit SID unique ≥9500/9922, 逐层 utilization 偏差 ≤5pp | 4-digit SID unique ≥9500/9922, 逐层 utilization 偏差 ≤5pp | ✅ |
| Gate 3 = Stage 3 | T5 消费 SID, 不得用 T5 修 Stage 1/2 collapse | T5 消费 SID, 不得用 T5 弥补 Stage 1/2 collapse | ✅ |
| Gate 4 = Stage 4 | Test R@5/10/20, NDCG@5/10/20, R@10 > 0.1020 | Test R@5/10/20, NDCG@5/10/20, R@10 > 0.1020 | ✅ |

→ **Issue #70 spec 跟 Issue #67 spec 实质相同, 区别仅在文献补充 + precheck blocked 命名**.

## 3. Issue #67 闭环结果 (引证)

| Gate (历史) | R17/§19 对应 | Task | Commit | 决策 + 失败原因 |
|------|------|------|--------|------|
| Gate -1 architecture incomplete | **precheck blocked (= Issue #70 Gate 1 之前预检 FAIL)** | task355 | a968c23 | ❌ NO-GO: T1 architecture incomplete FAIL (κ_l_m/w_l_m 缺失, 最近实现 α_l+scale_l 满足 7/8) |
| Gate 0+ | - | - | - | ⏸ STOP per spec |

→ **Issue #67 precheck blocked 已 NO-GO 收口, Issue #70 沿用同结论**.

## 4. Issue #70 precheck blocked = Gate 1 之前预检 FAIL

**失败原因**: 三层混合曲率 product space 架构未完整实施 per-component `kappa_l,m` + softmax `w_l,m`. 最近实现仅含 α_l/scale_l (Gate -1 T1 architecture incomplete FAIL 7/8).

**Gate 1 Stage 1 训练**: ⏸ STOP per Issue #70 spec ("预检未 PASS 前禁止执行 Gate1")

## 5. Issue #70 整体决策

| R17 Gate | 决策 | 失败原因 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ❌ **precheck blocked NO-GO** | 三层混合曲率 product space 架构未完整实施 (κ_l_m/w_l_m 缺失). 前置预检 FAIL, 禁止进 Stage 1 训练 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Gate 1 precheck FAIL) | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

→ **Issue #70 整体 NO-GO 收口, 不进入 Stage 1/2/3/4**.

## 6. Drift-cycle 联立分析 (6+ NO-GO 同方向)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #64 (Direction B 重开) | precheck | NO-GO architecture incomplete |
| Issue #67 (Direction B 重开2) | precheck | NO-GO (复用 #64) |
| **Issue #70 (Direction B 预检)** | **Gate 1 之前 precheck** | **NO-GO (复用 #67)** |

→ **3 连续 Direction B precheck NO-GO 收口**.

## 7. R11.5 决策

- Issue #70 整体 NO-GO 收口 (沿用 Issue #67 决策, 不重跑)
- 跟 Issue #71 联立 = Direction B/C 预检 + Issue #69 Gate 1 Stage 1 三 issue 同步处理
- 不重新启动 Issue #70 (architecture incomplete, 实施范围属 owner critical decision)
- 关闭 Issue #70 per R16 强制 (完成 → close)

## 8. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task362_issue70_direction_b_precheck_nogo.md | 本 verdict |
| verdicts/task355_issue64_gate_minus1_nogo.md | 引用 — precheck architecture incomplete |
| verdicts/task360_issue67_direction_b_reopen2_nogo.md | 引用 — 复用 #67 决策 |

---

result: Issue #70 [方向B 预检] 整体 NO-GO 收口 (复用 Issue #67 决策). Gate 1 (= Stage 1 RQ-VAE/HRQVAE) ❌ precheck blocked NO-GO: 三层混合曲率 product space 架构未完整实施 (κ_l_m/w_l_m 缺失). Gate 2/3/4 ⏸ STOP per spec. 跟 Issue #71 联立 = Direction B/C 预检 ×× NO-GO, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层未启动), 等 owner 拍板混合曲率 product space 完整实施范围.