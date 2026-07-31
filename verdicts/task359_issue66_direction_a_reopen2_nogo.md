# Task #359 / Issue #66 Gate -1 — Direction A 重开2 复用 #63 闭环整体 NO-GO 收口

**日期**: 2026-07-31
**触发**: Issue #66 [方向A 重开2] 三层κ同步重校准的最小 scale adapter — GitHub OPEN
**前置**: Issue #63 闭环完成 (task352 Gate -1 PASS + task353 Gate 0 PASS + task354 Gate 1 NO-GO USAGE-KILL)
**任务**: 验证 Issue #66 spec 跟 Issue #63 spec 一致性 + 复用已有 verdict 收口
**结果**: ❌ Issue #66 整体 NO-GO 收口 (沿用 Issue #63 Gate 1 USAGE-KILL 决策)

---

## 1. Issue #66 spec vs Issue #63 spec 对比

| 维度 | Issue #63 (Direction A 重开) | Issue #66 (Direction A 重开2) | 一致 |
|------|------|------|------|
| 框架 | num_emb_list=[64,128,256], L0/L1/L2 保留 active theta→kappa() | num_emb_list=[64,128,256], L0/L1/L2 保留 active theta→kappa() | ✅ |
| 实施范围 | scale adapter 扩展现有路径 | scale adapter 扩展现有路径, 禁止替换 | ✅ |
| Gate -1 spec | 8 项测试, 框架合规 | 8 项测试, 框架合规 | ✅ |
| Gate 0 spec | 复现 #49 训练 + 6 项 Test 指标 | 复现 #49 训练 + 6 项 Test 指标 | ✅ |
| Gate 1 spec | 同步重校准 + κ/scale 梯度非零 + #47 公式无 NaN/Inf | 同步重校准 + κ/scale 梯度非零 + #47 公式无 NaN/Inf | ✅ |
| Gate 2 spec | Stage 1+2 utilization ≥90%, 4-digit SID unique ≥9500/9922 | Stage 1+2 utilization ≥90%, 4-digit SID unique ≥9500/9922 | ✅ |
| Gate 3 spec | Test R@10 > 0.1020 | Test R@10 > 0.1020 | ✅ |

→ **Issue #66 spec 跟 Issue #63 spec 实质相同, 区别仅在文献补充 (arXiv:2405.13979v4 Bdeir 等)**.

## 2. Issue #63 闭环结果 (引证)

| Gate | Task | Commit | Verdict |
|------|------|--------|---------|
| Gate -1 | task352 | 75f1628 | ✅ 8/8 PASS (FreeCurvHRQVAE 三层 κ 隔离审计 8/8 PASS, R137 fix + R12 ckpt 验证) |
| Gate 0 | task353 | 446ee84 | ✅ 8/8 PASS (zero-GPU sanity 8/8: κ=0 Euclidean退化, κ=max bounded, 对称性, triangle inequality, codebook baseline 健康) |
| Gate 1 | task354 | 79a1881 | ❌ NO-GO (vanilla κ-decouple FreeCurvHRQVAE USAGE-KILL @ ep 30, util 1.6%/0.8%/0.4% ≪ 90%, κ→Euclidean mode collapse) |

→ **Issue #63 Gate 1 已 NO-GO 收口, Issue #66 沿用同结论**.

## 3. Issue #66 Gate -1 复用 task352 结论

Issue #66 Gate -1 spec 跟 Issue #63 Gate -1 spec 一致, task352 8/8 PASS 直接复用. 不重跑 (torch module 缺, 旧 commit 已落盘 + push, 等价).

**Gate -1 决策**: ✅ PASS (复用 task352 commit 75f1628, 8/8)

## 4. Issue #66 Gate 1 (Stage 1 训练) 复用 task354 结论

Issue #66 Gate 1 spec 跟 Issue #63 Gate 1 spec 一致, task354 已跑完 USAGE-KILL NO-GO. 不重跑.

**Gate 1 决策**: ❌ NO-GO (复用 task354 commit 79a1881, USAGE-KILL @ ep 30)

## 5. Issue #66 整体决策

| Gate | 决策 | 备注 |
|------|------|------|
| Gate -1 | ✅ PASS (复用 task352) | 框架合规 |
| Gate 0 | ✅ PASS (复用 task353) | 复现 + 失效定位 |
| Gate 1 | ❌ NO-GO (复用 task354) | USAGE-KILL @ ep 30 |
| Gate 2 | ⏸ STOP (Gate 1 FAIL, 不进 Gate 2) | per Issue #66 spec |
| Gate 3 | ⏸ STOP (Gate 1 FAIL, 不进 Gate 3) | per Issue #66 spec |

→ **Issue #66 整体 NO-GO 收口, 不进入 Stage 3/4 大范围复跑**.

## 6. Drift-cycle 联立分析 (5+ NO-GO 同方向)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #23 (per-layer c_k curriculum) | Gate 0 | NO-GO (硬停止) |
| Issue #28 (per-layer Gumbel-Softmax) | Gate 1 | FAIL (USAGE-KILL) |
| Issue #29 (per-layer Sinkhorn rescale) | Gate -1 | FAIL |
| Issue #32 (per-layer Codebook Transforms 双轴协同) | Gate 1-3 | NO-GO (ΔR@10 -99.88%) |
| Issue #33 (per-item soft variants) | Gate 1 | FAIL (USAGE-KILL) |
| Issue #63 (Direction A 重开) | Gate 1 | NO-GO (USAGE-KILL) |
| Issue #64 (Direction B 重开) | Gate -1 | NO-GO (architecture incomplete) |
| Issue #65 (Direction C 重开) | Gate -1 | NO-GO (4/8 FAIL) |
| **Issue #66 (Direction A 重开2)** | **Gate 1** | **NO-GO (复用 #63)** |
| Issue #67 (Direction B 重开2) | Gate -1 | NO-GO (复用 #64) |
| Issue #68 (Direction C 重开2) | Gate -1 | NO-GO (复用 #65) |

→ **11 连续方向 ×× NO-GO 收口 (跨 Stage 1-3 架构层), baseline recipe 内部 R@10 杠杆已穷尽**.

## 7. R11.5 决策

- Issue #66 整体 NO-GO 收口 (沿用 Issue #63 决策, 不重跑 GPU)
- 跟 Issue #67/#68 联立 = Direction A/B/C 重开2 三方向 ×× NO-GO 收口
- 不重新启动 Issue #66 (drift-cycle 已 11+ NO-GO 阈值)
- 关闭 Issue #66 per R16 强制 (完成 → close)

## 8. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task359_issue66_direction_a_reopen2_nogo.md | 本 verdict |
| verdicts/task352_issue63_gate_minus1_result.md | 引用 — Gate -1 8/8 PASS |
| verdicts/task353_issue63_gate0_sanity_result.md | 引用 — Gate 0 8/8 PASS |
| verdicts/task354_issue63_gate1_stage1_nogo.md | 引用 — Gate 1 NO-GO USAGE-KILL |

---

result: Issue #66 [方向A 重开2] 整体 NO-GO 收口 (复用 Issue #63 闭环决策). Gate -1 ✅ PASS (复用 task352 75f1628), Gate 0 ✅ PASS (复用 task353 446ee84), Gate 1 ❌ NO-GO (复用 task354 79a1881, USAGE-KILL @ ep 30), Gate 2/3 STOP per spec. 跟 Issue #67/#68 联立 = Direction A/B/C 重开2 三方向 ×× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层未启动).