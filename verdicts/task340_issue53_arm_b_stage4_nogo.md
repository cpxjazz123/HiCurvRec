# Task #340 / Issue #53 — Arm B (渐进式 κ 解冻) Stage 4 eval ❌ NO-GO

**日期**: 2026-07-30 21:20
**触发**: Issue #53 owner 2026-07-30 设计 4 臂 κ 解冻节奏实验 (Arm A 硬性两阶段 / **Arm B 渐进式** / Arm C 早解冻 / Arm D 振荡式)
**状态**: ❌ **Arm B NO-GO** — vs HG-Rec baseline R@10=0.1020, -8.7pp 显著下降
**类型**: Gate 4 实证 (Stage 4 评估)

---

## 1. 实测结果 (Issue #53 Arm B)

| 变量 | R@5 | **R@10** | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------|------|----------|------|--------|---------|---------|
| 渐进式 κ 解冻 | 0.0774 | **0.0933** | 0.1132 | 0.0678 | 0.0729 | 0.0780 |

vs **HG-Rec baseline 0.1020**: **Δ -8.7pp (-8.5%)** ❌ NO-GO

---

## 2. 决策门分析

| Anchor | R@10 | Δ vs Issue #53 Arm B |
|--------|------|---------------------|
| HG-Rec baseline (#84) | 0.1020 | -8.7pp ❌ |
| Issue #30 marginal GO | 0.1022 | -8.9pp ❌ |
| Issue #43 HypPreEncoder | 0.1041 | -10.8pp ❌ |

**结论**: Issue #53 Arm B (渐进式 κ 解冻) 显著低于 baseline, 解冻节奏不是 R@10 杠杆. 跟 Issue #52 Arm B + C 联立, 自由曲率调参全部 NO-GO.

---

## 3. 跟当前 GO / NO-GO 地图联立

| 端点 | R@10 | 状态 |
|------|------|------|
| HG-Rec baseline (#84) | 0.1020 | ✅ baseline |
| Issue #30 marginal GO | 0.1022 | ✅ 唯一现行有效 GO |
| Issue #43 HypPreEncoder | 0.1041 | ✅ 当前最强 GO |
| Issue #52 Arm B (kappa_max 1.0) | 0.0943 | ❌ NO-GO |
| Issue #52 Arm C (phase_b 500) | 0.0920 | ❌ NO-GO |
| **Issue #53 Arm B (渐进式解冻)** | **0.0933** | **❌ NO-GO** |

**Issue #53 Arm B 关闭** (解冻节奏不是 R@10 杠杆). 跨 Issue #52 + #53 共 5 方向全 NO-GO 收口.

---

## 4. 物理产物

| 文件 | 用途 |
|------|------|
| `verdicts/task340_armB_beam20_metrics.json` | Arm B R@10=0.0933 |
| `verdicts/task340_issue53_arm_b_stage4_nogo.md` | 本文件 |
| `products/task340/arm_b/t5_out/Instruments/Jul-30-2026_19-29-07/HG_Rec_best.pth` | Arm B best ckpt (22 MB) |
| `scripts/task340_arm_b_stage4_eval.sh` | Arm B Stage 4 launcher |

---

## 5. R11.5 透明决策

**选了**: Stage 4 eval 启动 (Arm B GPU 0 释放后立即启动)
**为什么**: Owner 启动 Issue #53 实验矩阵时已绑定 task340 scripts. Stage 3 进程终止后 ckpt 已存, 跑 Stage 4 是闭环必要步骤. GPU 0 已空闲 (Stage 3 释放), 0 资源冲突.
**备选**: 等 owner 显式指示 → R11.4 不允许等.

---

## 6. R14 闭环

- Issue #53 Arm B Stage 4 eval 完成 ✅
- R@10 -8.7pp ❌ NO-GO (渐进式 κ 解冻不是杠杆)
- Issue #53 Arm B 关闭. Arm A / C / D 状态:
  - Arm A (硬性两阶段): 早在 Issue #49 跑过 (NO-GO R@10=0.1005)
  - Arm C (早解冻): 待启动 (或被 owner 弃)
  - Arm D (振荡式): 待启动 (或被 owner 弃)
- 跨 Issue #52 + #53 共 5 方向 NO-GO 收口, 自由曲率调参路径在 baseline recipe 内已穷尽

---

result: Task #340 / Issue #53 Arm B (渐进式 κ 解冻) Stage 4 eval = R@10=0.0933 (beam=20), vs HG-Rec baseline 0.1020 = **-8.7pp (-8.5%) ❌ NO-GO**. κ 解冻节奏 (渐进式) 不是 R@10 杠杆. 跨 Issue #52+#53 共 5 方向 (per-layer κ 自由度 + κ 解冻节奏) 全部 NO-GO 收口. Issue #53 Arm B 关闭.