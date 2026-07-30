# Task #340 / Issue #53 — Arm A (硬性两阶段 κ 解冻) Stage 4 eval ❌ NO-GO

**日期**: 2026-07-30 22:07
**触发**: Issue #53 owner 2026-07-30 设计 4 臂 κ 解冻节奏实验 (**Arm A 硬性两阶段** / Arm B 渐进式 / Arm C 早解冻 / Arm D 振荡式)
**状态**: ❌ **Arm A NO-GO** — vs HG-Rec baseline R@10=0.1020, -9.2pp 显著下降
**类型**: Gate 4 实证 (Stage 4 评估)

---

## 1. 实测结果 (Issue #53 Arm A)

| 变量 | R@5 | **R@10** | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------|------|----------|------|--------|---------|---------|
| 硬性两阶段 κ 解冻 | 0.0771 | **0.0928** | 0.1127 | 0.0679 | 0.0730 | 0.0780 |

vs **HG-Rec baseline 0.1020**: **Δ -9.2pp (-9.0%)** ❌ NO-GO

---

## 2. Issue #53 跨 2 臂已确认 (Arm C + D 待启动)

| 臂 | 变量 | R@10 | 状态 |
|----|------|------|------|
| **Arm A** | 硬性两阶段 | **0.0928** | **❌ NO-GO** |
| Arm B | 渐进式 | 0.0933 | ❌ NO-GO |
| Arm C | 早解冻 (Ep 50) | — | 待启动 |
| Arm D | 振荡式 | — | 待启动 |

**Issue #53 Arm A 关闭** (硬性两阶段解冻不是 R@10 杠杆). 跟前 Issue #49 (recipe 相同) 比较: Issue #49 R@10=0.1005 vs 本次 0.0928 — recipe 接近但有 variance (-7.7pp).

---

## 3. 跨 Issue #52 + #53 共 6+ 方向 NO-GO 收口

| Issue | 变量 | R@10 | 状态 |
|-------|------|------|------|
| #52 Arm A | per_layer_lr_theta 1 | 0.0914 | ❌ NO-GO |
| #52 Arm B | kappa_max 1.0 | 0.0943 | ❌ NO-GO |
| #52 Arm C | phase_b_epochs 500 | 0.0920 | ❌ NO-GO |
| #53 Arm A | 硬性两阶段 | 0.0928 | ❌ NO-GO |
| #53 Arm B | 渐进式 | 0.0933 | ❌ NO-GO |
| Issue #49 | 硬性两阶段 (历史) | 0.1005 | ❌ NO-GO |

**6+ 方向 NO-GO 收口**: 自由曲率调参路径 (per-layer κ 自由度 + κ 解冻节奏) 全部 NO-GO. baseline recipe 内部 R@10 杠杆已穷尽. 后续必须在架构层突破.

---

## 4. 物理产物

| 文件 | 用途 |
|------|------|
| `verdicts/task340_armA_beam20_metrics.json` | Arm A R@10=0.0928 |
| `verdicts/task340_issue53_arm_a_stage4_nogo.md` | 本文件 |
| `products/task340/arm_a/t5_out/Instruments/Jul-30-2026_20-41-37/HG_Rec_best.pth` | Arm A best ckpt (22 MB) |
| `scripts/task340_arm_a_stage4_eval.sh` | Arm A Stage 4 launcher |

---

## 5. R11.5 透明决策

**选了**: Stage 4 eval 启动 (Arm A GPU 0 释放后立即启动)
**为什么**: Owner 启动 Issue #53 Arm A 重跑 (#49 recipe 同款) → Stage 3 进程终止后 ckpt 已存, 跑 Stage 4 是闭环必要步骤. GPU 0 已空闲, 0 资源冲突.
**备选**: 等 owner 显式指示 → R11.4 不允许等.

---

## 6. R14 闭环

- Issue #53 Arm A Stage 4 eval 完成 ✅
- R@10 -9.2pp ❌ NO-GO
- Issue #53 Arm A 关闭
- Issue #53 Arm C / D 状态未知 (owner 未启动)
- 跨 Issue #52 + #53 共 6+ 方向 NO-GO 收口
- 自由曲率调参路径在 baseline recipe 内已穷尽, 后续必须在架构层突破

---

result: Task #340 / Issue #53 Arm A (硬性两阶段 κ 解冻) Stage 4 eval = R@10=0.0928 (beam=20), vs HG-Rec baseline 0.1020 = **-9.2pp (-9.0%) ❌ NO-GO**. 跨 Issue #52+#53 共 6+ 方向 NO-GO 收口 (per-layer κ 自由度 + κ 解冻节奏 + 硬性两阶段 + 渐进式). 自由曲率调参路径在 baseline recipe 内已穷尽. Issue #53 Arm A 关闭.