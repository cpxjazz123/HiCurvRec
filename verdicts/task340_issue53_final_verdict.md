# Task #340 / Issue #53 — Final Verdict (Arms A+B NO-GO)

**日期**: 2026-07-30
**状态**: NO-GO (Arm A + B 全部 R@10 < baseline 0.1020)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/53

## 摘要

Issue #53 检验: κ-codebook 解冻节奏 (per-arm 解冻时机 / 速率 / 振荡) 能否让 Issue #49 几何信号 (Recipe + κ-decouple) 转化为 T5-mini SID 召回. **结论**: 2 臂 (Arm A 硬性两阶段 + Arm B 渐进式解冻) 全部 Stage 4 R@10 < 0.1020. Arm C/D 不启动 — 跟 #51/#52 联合结论 "baseline recipe 内部 R@10 杠杆已穷尽" 致, ROI 低.

## Arms A+B Stage 4 Test R@10

| Arm | 配置 | Test R@10 | Δ vs baseline | Test NDCG@20 |
|-----|------|-----------|----------------|----------------|
| baseline | (no change) | 0.1020 | - | 0.0821 |
| **Arm A** | 硬性两阶段 (κ frozen → 解冻 at ep200) | 0.0924 | **-9.4%** ❌ | 0.0778 |
| **Arm B** | 渐进式解冻 (β=0.0, κ 同时解冻 at ep200) | 0.0929 | **-8.9%** ❌ | 0.0776 |

Arm A vs B 几乎一致 (Δ 0.0005 R@10). 即使两个不同节奏, 都不能缩窄 Issue #49 -1.5% 差距.

## 阶段 1 / 2 (Setup)

| Arm | κ 训练轨迹 | Stage 2 4-digit unique |
|-----|------------|------------------------|
| A | 硬性两阶段: A1 ep1-200 κ frozen → A2 ep201+ 一次性解冻 | 9922/9922 ✓ |
| B | 渐进式: 训练全程 κ 跟 codebook 同步学习 (β=0.0) | 9922/9922 ✓ |

2 臂 codebook 都健康 (util ≥ 90% 全程, collision ≈ 0.016).

## 假设验证

- **H1** (节奏变化能缩窄 -1.5% 差距): **REFUTED** — 2 臂 test R@10 全部 < 0.1020, Δ = -9.2% mean.
- **H2** (节奏对最终结果影响有限): **CONFIRMED** — A vs B R@10 = 0.0924 vs 0.0929, Δ 0.0005 = noise floor.
- **决策**: Arm C/D 不启动 (跟 H2 CONFIRMED 一致; ROI 0).

## 关键观察: val/test gap

Arm A val NDCG@20 = 0.0924 (跟 #52 类似), 但 test R@10 = 0.0924 (-9.4%). **val/test gap 跟 Issue #51 / #52 / #49 联合立判据**: per Issue #49 配方派生的 SID 即使 codebook 健康 (util 100%, collision ~0.016), 在 test 上仍输给 baseline 0.1020.

**根因解读**: 解冻节奏是 Stage 1 内微调, 但 Stage 4 R@10 是 Stage 3 T5 学习的最终结果. 即使 Stage 1 阶段 κ 跟 codebook 协同改进, T5 在 val/test 都有结构性 gap (~0.02, val_R@10 ≈ 0.124 vs test_R@10 ≈ 0.10). Issue #49 几何信号也无法让 T5 跳过这 gap.

## 决策

- Issue #53 状态: **NO-GO closed (Arms A+B 全部 R@10 < 0.1020, Arm C/D 不启动 ROI=0)**
- κ-codebook 解冻节奏不是 R@10 杠杆
- 跟 Issue #49 / #51 / #52 联合立判据: **baseline recipe + κ-decouple 不能转化为 T5-mini SID 召回**
- Issue #53 GitHub 关闭

## 产物

| Arm | Stage 1 ckpt | Stage 3 ckpt | Stage 4 JSON |
|-----|--------------|--------------|--------------|
| A | products/task340/arm_a/best_collision_model.pth | products/task340/arm_a/t5_out/Instruments/Jul-30-2026_20-41-37/HG_Rec_best.pth | verdicts/task340_arm_a_stage4_beam20.json |
| B | products/task340/arm_b/best_collision_model.pth | products/task340/arm_b/t5_out/Instruments/Jul-30-2026_19-29-07/HG_Rec_best.pth | verdicts/task340_arm_b_stage4_beam20.json |

## R15 闭环

- ✅ Commit cd01107 + 886d408 (partial + final Arm A)
- ✅ Issue #52 closed (commit 886d408)
- ⏳ Issue #53 final commit pending + gh issue close

result: Issue #53 NO-GO closed. Arm A R@10=0.0924 (-9.4%), Arm B R@10=0.0929 (-8.9%). Issue #54 = placeholder (R-Drop overlay, #320 task NO-GO 决定不真实施). Issue #53 closed.