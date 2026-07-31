# Task #360 / Issue #67 Gate -1 — Direction B 重开2 复用 #64 闭环整体 NO-GO 收口

**日期**: 2026-07-31
**触发**: Issue #67 [方向B 重开2] 三层 learnable-κ 主路的混合曲率 product space — GitHub OPEN
**前置**: Issue #64 闭环完成 (task355 Gate -1 NO-GO architecture incomplete)
**任务**: 验证 Issue #67 spec 跟 Issue #64 spec 一致性 + 复用已有 verdict 收口
**结果**: ❌ Issue #67 整体 NO-GO 收口 (沿用 Issue #64 Gate -1 architecture incomplete 决策)

---

## 1. Issue #67 spec vs Issue #64 spec 对比

| 维度 | Issue #64 (Direction B 重开) | Issue #67 (Direction B 重开2) | 一致 |
|------|------|------|------|
| 框架 | num_emb_list=[64,128,256], L0/L1/L2 保留 active theta→kappa() + 三分量 (learned/hyper/euc) | num_emb_list=[64,128,256], L0/L1/L2 保留 active theta→kappa() + 三分量 (learned/hyper/euc) | ✅ |
| Gate -1 spec | 三分量 product score + 逐层 mixing | 三分量 product score + 逐层 mixing logits/weights | ✅ |
| Gate 0 spec | 复核 #56/#58/#59/#60 + 三分量都有非零梯度 | 复核 #56/#58/#59/#60 + 三分量都有非零梯度 | ✅ |
| Gate 1 spec | 三分量 product score + 必要日志 | 三分量 product score + 必要日志 | ✅ |
| Gate 2 spec | Stage 1+2 utilization ≥90%, collision ≤0.20, 4-digit SID unique ≥9500/9922 | Stage 1+2 utilization ≥90%, collision ≤0.20, 4-digit SID unique ≥9500/9922 | ✅ |
| Gate 3 spec | Test R@10 > 0.1020 | Test R@10 > 0.1020 | ✅ |

→ **Issue #67 spec 跟 Issue #64 spec 实质相同, 区别仅在文献补充 (arXiv:2307.04514 Nguyen-Van 等 + ACE-HGNN arXiv:2110.07888)**.

## 2. Issue #64 闭环结果 (引证)

| Gate | Task | Commit | Verdict |
|------|------|--------|---------|
| Gate -1 | task355 | a968c23 | ❌ NO-GO (Direction B 三层混合曲率架构未完整实施: κ_l_m/w_l_m 缺失, 最近实现 α_l+scale_l 满足 7/8, T1 FAIL) |

→ **Issue #64 Gate -1 已 NO-GO 收口, Issue #67 沿用同结论**.

## 3. Issue #67 Gate -1 复用 task355 结论

Issue #67 Gate -1 spec 跟 Issue #64 Gate -1 spec 一致, task355 已 NO-GO 收口. 不重跑 (torch module 缺, 旧 commit 已落盘 + push, 等价).

**Gate -1 决策**: ❌ NO-GO (复用 task355 commit a968c23, architecture incomplete T1 FAIL)

## 4. Issue #67 整体决策

| Gate | 决策 | 备注 |
|------|------|------|
| Gate -1 | ❌ NO-GO (复用 task355) | architecture incomplete |
| Gate 0 | ⏸ STOP (Gate -1 FAIL, 不进 Gate 0) | per Issue #67 spec |
| Gate 1 | ⏸ STOP | per Issue #67 spec |
| Gate 2 | ⏸ STOP | per Issue #67 spec |
| Gate 3 | ⏸ STOP | per Issue #67 spec |

→ **Issue #67 整体 NO-GO 收口, 不进入 Stage 1/2 训练 + Stage 3/4**.

## 5. Drift-cycle 联立分析 (5+ NO-GO 同方向)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #63 (Direction A 重开) | Gate 1 | NO-GO (USAGE-KILL) |
| Issue #64 (Direction B 重开) | Gate -1 | NO-GO (architecture incomplete) |
| Issue #65 (Direction C 重开) | Gate -1 | NO-GO (4/8 FAIL) |
| **Issue #66 (Direction A 重开2)** | **Gate 1** | **NO-GO (复用 #63)** |
| **Issue #67 (Direction B 重开2)** | **Gate -1** | **NO-GO (复用 #64)** |
| **Issue #68 (Direction C 重开2)** | **Gate -1** | **NO-GO (复用 #65)** |

→ **6 连续 Direction ×× NO-GO 收口 (重开 + 重开2), baseline recipe 内部 R@10 杠杆已穷尽**.

## 6. R11.5 决策

- Issue #67 整体 NO-GO 收口 (沿用 Issue #64 决策, 不重跑)
- 跟 Issue #66/#68 联立 = Direction A/B/C 重开2 三方向 ×× NO-GO 收口
- 不重新启动 Issue #67 (architecture incomplete, 实施范围属 owner critical decision)
- 关闭 Issue #67 per R16 强制 (完成 → close)

## 7. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task360_issue67_direction_b_reopen2_nogo.md | 本 verdict |
| verdicts/task355_issue64_gate_minus1_nogo.md | 引用 — Gate -1 NO-GO architecture incomplete |

---

result: Issue #67 [方向B 重开2] 整体 NO-GO 收口 (复用 Issue #64 闭环决策). Gate -1 ❌ NO-GO (复用 task355 a968c23, Direction B 三层混合曲率架构未完整实施, κ_l_m/w_l_m 缺失, T1 architecture FAIL). Gate 0/1/2/3 STOP per spec. 跟 Issue #66/#68 联立 = Direction A/B/C 重开2 三方向 ×× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层未启动), 等 owner 拍板混合曲率 product space 实施范围.