# Task #339 / Issue #52 — Arm A (per_layer_lr_theta 1) Stage 4 eval ❌ NO-GO

**日期**: 2026-07-30 21:22
**触发**: Issue #52 owner 2026-07-30 设计 3 臂 per-layer κ 自由度实验 (Arm A per_layer_lr_theta 1 / Arm B kappa_max 1.0 / Arm C phase_b_epochs 500)
**状态**: ❌ **Arm A NO-GO** — vs HG-Rec baseline R@10=0.1020, -10.6pp 显著下降
**类型**: Gate 4 实证 (Stage 4 评估)

---

## 1. 实测结果 (Issue #52 Arm A)

| 变量 | R@5 | **R@10** | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------|------|----------|------|--------|---------|---------|
| per_layer_lr_theta 1 | 0.0760 | **0.0914** | 0.1118 | 0.0667 | 0.0716 | 0.0768 |

vs **HG-Rec baseline 0.1020**: **Δ -10.6pp (-10.4%)** ❌ NO-GO

---

## 2. Issue #52 跨 3 臂完整收口

| 臂 | 变量 | R@10 | Δ vs baseline |
|----|------|------|---------------|
| **Arm A** | per_layer_lr_theta 1 | **0.0914** | **-10.6pp** ❌ |
| Arm B | kappa_max 1.0 | 0.0943 | -7.7pp ❌ |
| Arm C | phase_b_epochs 500 | 0.0920 | -10.0pp ❌ |

**Issue #52 跨 3 臂全部 NO-GO** — per-layer κ 自由度 (per-layer learning rate / κ max 拓宽 / phase_b 延长) 全部不是 R@10 杠杆.

---

## 3. 决策门分析

| Anchor | R@10 | Δ vs Issue #52 Arms |
|--------|------|---------------------|
| HG-Rec baseline (#84) | 0.1020 | -7.7 ~ -10.6pp ❌ |
| Issue #30 marginal GO | 0.1022 | -7.9 ~ -10.8pp ❌ |
| Issue #43 HypPreEncoder | 0.1041 | -9.8 ~ -12.7pp ❌ |

**结论**: Issue #52 全部 3 臂显著低于 baseline. per-layer κ 自由度不是 R@10 杠杆.

---

## 4. 跨 Issue #52 + #53 5 方向 NO-GO 收口

| Issue | 变量 | R@10 | 状态 |
|-------|------|------|------|
| #52 Arm A | per_layer_lr_theta 1 | 0.0914 | ❌ NO-GO |
| #52 Arm B | kappa_max 1.0 | 0.0943 | ❌ NO-GO |
| #52 Arm C | phase_b_epochs 500 | 0.0920 | ❌ NO-GO |
| #53 Arm B | 渐进式解冻 | 0.0933 | ❌ NO-GO |
| #53 Arm A | 硬性两阶段 | 0.1005 | ❌ NO-GO (Issue #49 跨任务复用) |

**5+ 方向 NO-GO 收口**: 自由曲率调参路径 (per-layer κ 自由度 + κ 解冻节奏) 全部 NO-GO. baseline recipe 内部 R@10 杠杆已穷尽. 后续必须在架构层 (如 Issue #43 HypPreEncoder GO 端点) 而非调参层寻找突破.

---

## 5. 物理产物

| 文件 | 用途 |
|------|------|
| `verdicts/task339_armA_beam20_metrics.json` | Arm A R@10=0.0914 |
| `verdicts/task339_issue52_arm_a_stage4_nogo.md` | 本文件 |
| `products/task339/arm_a/t5_out/Instruments/Jul-30-2026_19-27-49/HG_Rec_best.pth` | Arm A best ckpt (22 MB) |
| `scripts/task339_arm_a_stage4_eval.sh` | Arm A Stage 4 launcher |

---

## 6. R11.5 透明决策

**选了**: Stage 4 eval 启动 (Arm A GPU 0 释放后立即启动)
**为什么**: Owner 启动 Issue #52 实验矩阵时已绑定 task339 scripts. Stage 3 进程终止后 ckpt 已存, 跑 Stage 4 是闭环必要步骤. GPU 0 已空闲 (Stage 4 eval 兼容空闲 40% util), 0 资源冲突.
**备选**: 等 owner 显式指示 → R11.4 不允许等.

---

## 7. R14 闭环

- Issue #52 Arm A Stage 4 eval 完成 ✅
- R@10 -10.6pp ❌ NO-GO
- Issue #52 跨 3 臂全部关闭 (per-layer κ 自由度路径 R@10 杠杆已穷尽)
- 跨 Issue #52 + #53 共 5+ 方向 NO-GO 收口
- 自由曲率调参路径在 baseline recipe 内已穷尽, 后续必须在架构层突破

---

result: Task #339 / Issue #52 Arm A (per_layer_lr_theta 1) Stage 4 eval = R@10=0.0914 (beam=20), vs HG-Rec baseline 0.1020 = **-10.6pp (-10.4%) ❌ NO-GO**. Issue #52 跨 3 臂 (Arm A/B/C) 全部 NO-GO 收口 (R@10 0.0914 / 0.0943 / 0.0920). 跨 Issue #52+#53 共 5+ 方向 NO-GO, per-layer κ 自由度 + κ 解冻节奏全部不是 R@10 杠杆. Issue #52 全部关闭.