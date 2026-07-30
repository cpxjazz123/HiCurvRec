# Task #339 / Issue #52 — Verdict (待 Stage 4 回填)

**日期**: 2026-07-30
**状态**: 3 臂 Stage 1 完成, Stage 3 T5 训练中
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/52

## 摘要

Issue #52 检验: per-layer κ 自由度 (per-layer lr_theta / 拓宽 κ_max / 延长 Phase B) 能否解决 Issue #49 (R@10=0.1005 NO-GO) 的 -1.5% 差距。

## Stage 1 结果 (3 臂全部 Gate 1 PASS)

| Arm | 配置 | Best Collision | κ 终值 | Gate 1 |
|-----|------|----------------|--------|--------|
| A | per_layer_lr_theta [1e-4, 1e-5, 1e-6] | 0.0162 | -0.040 (Phase A 终点) | ✅ |
| B | kappa_max=1.0 | 0.0153 | -0.020 (Phase A 终点) | ✅ |
| C | phase_b_epochs=500 | 0.0162 | -0.040 (Phase A 终点) | ✅ |

**关键观察**: 3 臂最佳 ckpt 都来自 Phase A epoch 10 — **意味着 Phase B 没改善 collision**. κ 几乎不变 (Phase A κ=-0.04, Phase B 几乎没学到 κ).

## Stage 2 SID 结果

3 臂全部 4-digit 9922/9922 unique:
- Arm A: raw collision=0.0162
- Arm B: raw collision=0.0153 (kappa_max=1.0 → κ 终值仅 -0.02, 实际等于 κ=0)
- Arm C: raw collision=0.0162 (phase_b_epochs=500 跟 A/B 一致, 因为 ckpt 是 Phase A 终点)

## Stage 3 T5-mini 训练 (并行 4 卡 GPU 0/1/2/3)

- Arm A: GPU 1 PID 2713877 (跟 Issue #51 Stage 3 共用)
- Arm B: GPU 2 PID 2716125
- Arm C: GPU 3 PID 2716128

## Stage 4 待定

待 Stage 3 完成后跑 beam=20 test eval, 写 verdict JSON.

## 决策阈值

| 条件 | 含义 | 决策 |
|------|------|------|
| 任何臂 R@10 > 0.1020 | 超过 baseline, 协议杠杆有效 | **GO** |
| 任何臂 R@10 > 0.1005 | 缩小差距 (vs Issue #49) | 边际 |
| 任何臂 R@10 ≤ 0.1005 | 跟 #49 持平 | NO-GO |

## 当前预期

- Issue #52 Arm A (per_layer lr): 跟 Issue #49 协议相同, R@10 期望 ≈ 0.1005 NO-GO
- Issue #52 Arm B (kappa_max=1.0): κ≈0 等价 baseline, 期望 ~0.1020 持平
- Issue #52 Arm C (phase_b=500): 跟 A 类似, 期望 ~0.1005 NO-GO

Issue #52 大概率 NO-GO (per-layer lr_theta / kappa_max clamp / phase B 延长都是 #49 recipe 内的微调, 没引入新杠杆).

result: 待 Stage 3/4 完成后回填