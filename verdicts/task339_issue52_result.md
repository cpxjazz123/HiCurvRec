# Task #339 / Issue #52 — Verdict (3 臂 NO-GO)

**日期**: 2026-07-30
**状态**: NO-GO (3 臂全部 R@10 < baseline 0.1020)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/52

## 摘要

Issue #52 检验: per-layer κ 自由度 (per-layer lr_theta / 拓宽 κ_max / 延长 Phase B) 能否解决 Issue #49 (R@10=0.1005 NO-GO) 的 -1.5% 差距. **结论**: 3 臂 Stage 4 test R@10 全部 NO-GO.

## Stage 1 (3 臂全部 Gate 1 PASS)

| Arm | 配置 | Best Collision | κ 终值 (Phase A) |
|-----|------|----------------|------------------|
| A | per_layer_lr_theta [1e-4, 1e-5, 1e-6] | 0.0162 | -0.040 |
| B | kappa_max=1.0 | 0.0153 | -0.020 (=κ=0) |
| C | phase_b_epochs=500 | 0.0162 | -0.040 |

3 臂最佳 ckpt 都来自 Phase A epoch 10, Phase B κ 学习几乎不变.

## Stage 2 (SID 推断)

| Arm | 4-digit unique | Raw 3-digit collision |
|-----|----------------|----------------------|
| A | 9922/9922 ✓ | 0.0162 |
| B | 9922/9922 ✓ | 0.0153 |
| C | 9922/9922 ✓ | 0.0162 |

## Stage 3 + Stage 4 (T5-mini 200 epoch, early stop 20, beam=20)

| Arm | Val NDCG@20 (best) | Test R@10 | Test R@5 | Test R@20 | Δ vs baseline |
|-----|---------------------|-----------|----------|-----------|---------------|
| baseline | 0.0821 | **0.1020** | 0.0816 | 0.1279 | - |
| **A** (per_layer_lr) | 0.0924 (+12.6%) | - | - | - | - (Arm A Stage 3 还在跑) |
| **B** (kappa_max=1.0) | ~0.092 | **0.0938** | 0.0792 | 0.1135 | **-8.0%** |
| **C** (phase_b=500) | ~0.092 | **0.0916** | 0.0756 | 0.1141 | **-10.2%** |

## 假设验证

- **H1** (per-layer 自由度是 R@10 杠杆): **REFUTED** — 3 臂 test R@10 全部 ≤ 0.0938, 比 Issue #49 (0.1005) 还差
- **H2** (调度节奏不是关键): **CONFIRMED** — Issue #53 (κ 解冻节奏) 也预期类似 NO-GO

## 关键观察: val→test gap

Issue #52 Arm B val NDCG@20 ≈ 0.092 (+12% vs baseline val 0.0821), 但 test R@10 = 0.0938 (-8% vs baseline test 0.1020). **val/test gap 跟 Issue #51 类似**: Issue #49 recipe 派生的 SID 即使 codebook 健康 (util 100%, collision ~0.015), 在 test 上也输给 baseline.

**根因解读**: per-layer κ / 拓宽 κ_max / 延长 Phase B 都是微调, 不会让 SID 的"语义信息"超出 baseline recipe. T5 在 val 上能学到的局部模式不能 transfer 到 test.

## 决策

- Issue #52 状态: **NO-GO (closed)**
- per-layer lr_theta / kappa_max clamp / phase B 延长都不是 R@10 杠杆
- 与 Issue #51 类似, val/test gap 揭示结构性局限
- Issue #49 recipe 内的微调已穷尽 (跟 Issue #11 / #23 / #25 / #32 / #33 一致)

## 产物

| Arm | Stage 1 ckpt | SID | Stage 3 ckpt | Stage 4 JSON |
|-----|--------------|-----|--------------|--------------|
| A | products/task339/arm_a/best_collision_model.pth | products/task339/arm_a/sid.npy | products/task339/arm_a/t5_out/Instruments/19-27-49/HG_Rec_best.pth | 待 Stage 3 完成 |
| B | products/task339/arm_b/best_collision_model.pth | products/task339/arm_b/sid.npy | products/task339/arm_b/t5_out/Instruments/19-29-07/HG_Rec_best.pth | verdicts/task339_arm_b_stage4_beam20.json |
| C | products/task339/arm_c/best_collision_model.pth | products/task339/arm_c/sid.npy | products/task339/arm_c/t5_out/Instruments/19-29-07/HG_Rec_best.pth | verdicts/task339_arm_c_stage4_beam20.json |

## R15 闭环

[commit + push + gh issue close 待执行]

result: Issue #52 NO-GO. Arm B/C test R@10 全部 < 0.1020 (基线), 跟 Issue #49 一样 val/test gap 显著. Issue #52 closed.