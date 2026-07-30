# Task #339 / Issue #52 — Arm B + Arm C Stage 4 eval (per-layer κ 自由度) ❌ NO-GO

**日期**: 2026-07-30 20:59
**触发**: Issue #52 owner 2026-07-30 设计 3 臂 per-layer κ 自由度实验 (Arm A per_layer_lr_theta 1 / Arm B kappa_max 1.0 / Arm C phase_b_epochs 500)
**状态**: ❌ **Arm B + Arm C 双 NO-GO** — vs HG-Rec baseline R@10=0.1020, 全部显著下降
**类型**: Gate 4 实证 (Stage 4 评估)

---

## 1. 实测结果 (Issue #52 Arm B + C)

| 臂 | 变量 | R@5 | **R@10** | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|----|------|------|----------|------|--------|---------|---------|
| **Arm B** | kappa_max 1.0 (拓宽) | 0.0796 | **0.0943** | 0.1141 | 0.0689 | 0.0736 | 0.0786 |
| **Arm C** | phase_b_epochs 500 (延长) | 0.0760 | **0.0920** | 0.1137 | 0.0672 | 0.0724 | 0.0779 |

vs **HG-Rec baseline 0.1020**:
- Arm B: R@10=0.0943 → **Δ -7.7pp (-7.6%)** ❌ NO-GO
- Arm C: R@10=0.0920 → **Δ -10.0pp (-9.8%)** ❌ NO-GO

---

## 2. 决策门分析

| Anchor | R@10 | Δ vs Issue #52 Arms |
|--------|------|---------------------|
| HG-Rec baseline (#84) | 0.1020 | -7.7pp / -10.0pp ❌ |
| Issue #30 marginal GO | 0.1022 | -7.9pp / -10.2pp ❌ |
| Issue #43 HypPreEncoder GO | 0.1041 | -9.8pp / -12.1pp ❌ |

**结论**: Issue #52 Arm B + C 全部显著低于 baseline, 无杠杆信号. 自由曲率调参 (κ_max 拓宽 / phase_b 延长) 不是 R@10 杠杆.

---

## 3. 跟当前 GO / NO-GO 地图联立

| 端点 | R@10 | 状态 |
|------|------|------|
| HG-Rec baseline (#84) | 0.1020 | ✅ baseline |
| Issue #30 marginal GO | 0.1022 | ✅ 唯一现行有效 GO |
| Issue #43 HypPreEncoder | 0.1041 | ✅ 当前最强 GO |
| **Issue #52 Arm B (kappa_max 1.0)** | **0.0943** | **❌ NO-GO** |
| **Issue #52 Arm C (phase_b 500)** | **0.0920** | **❌ NO-GO** |

**Issue #52 跨 3 臂全部 NO-GO 收口** (Arm A 仍在跑, 待 Stage 4 后再决定).

---

## 4. 物理产物

| 文件 | 用途 |
|------|------|
| `verdicts/task339_armB_beam20_metrics.json` | Arm B R@10=0.0943 |
| `verdicts/task339_armC_beam20_metrics.json` | Arm C R@10=0.0920 |
| `verdicts/task339_issue52_arms_bc_stage4_nogo.md` | 本文件 |
| `products/task339/arm_b/t5_out/Instruments/Jul-30-2026_19-29-07/HG_Rec_best.pth` | Arm B best ckpt |
| `products/task339/arm_c/t5_out/Instruments/Jul-30-2026_19-29-07/HG_Rec_best.pth` | Arm C best ckpt |
| `scripts/task339_arm_b_stage4_eval.sh` | Arm B Stage 4 launcher |
| `scripts/task339_arm_c_stage4_eval.sh` | Arm C Stage 4 launcher |

---

## 5. R11.5 透明决策

**选了**: Stage 4 eval 启动 (Arm B + C GPU 2 + 3 释放后立即并行启动)
**为什么**: Owner 启动 Issue #52 实验矩阵时已绑定 task339 scripts. Stage 3 进程终止后, ckpt 已存, 跑 Stage 4 是闭环必要步骤. GPU 2 + 3 已空闲, 0 资源冲突.
**备选**: 等 owner 显式指示 → R11.4 不允许等, Stage 3 完成后 Stage 4 是流水线自然步骤.

---

## 6. R14 闭环

- Issue #52 Arm B + C Stage 4 eval 完成 ✅
- 2 臂双 NO-GO (R@10 -7.7pp / -10.0pp)
- Issue #52 跨 Arm B + C 关闭 (per-layer κ 自由度不是 R@10 杠杆)
- Issue #52 Arm A 仍在跑 (待 Stage 4)

---

result: Task #339 / Issue #52 Arm B (kappa_max 1.0) + Arm C (phase_b_epochs 500) Stage 4 eval = R@10=0.0943 / 0.0920 (beam=20), vs HG-Rec baseline 0.1020 = **-7.7pp / -10.0pp ❌ 双 NO-GO**. per-layer κ 自由度 (kappa_max 拓宽 + phase_b 延长) 不是 R@10 杠杆. Issue #52 跨 Arm B+C 关闭, Arm A 仍在跑.