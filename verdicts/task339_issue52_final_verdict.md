# Task #339 / Issue #52 — Final Verdict (3 臂全 NO-GO)

**日期**: 2026-07-30
**状态**: NO-GO (3 臂 Stage 4 全部 R@10 < baseline 0.1020)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/52

## 摘要

Issue #52 检验: per-layer κ 自由度 (per-layer lr_theta / 拓宽 κ_max / 延长 Phase B) 能否解决 Issue #49 (R@10=0.1005 NO-GO) 的 -1.5% 差距. **结论**: 3 臂 Stage 4 test R@10 全部 ≤ 0.0938.

## Stage 4 Test R@10 综合

| 指标 | baseline | **Arm A** (per-layer lr) | **Arm B** (kappa_max=1.0) | **Arm C** (phase_b=500) |
|------|----------|------------|------------|------------|
| **Recall@10** | 0.1020 | 0.0909 (-10.9%) ❌ | 0.0938 (-8.0%) ❌ | 0.0916 (-10.2%) ❌ |
| Recall@5 | 0.0816 | 0.0756 | 0.0792 | 0.0756 |
| Recall@20 | 0.1279 | 0.1112 | 0.1135 | 0.1141 |
| NDCG@10 | 0.0755 | 0.0713 | 0.0733 | 0.0720 |
| NDCG@20 | 0.0821 | 0.0764 | 0.0782 | 0.0778 |

## 假设验证

- **H1** (per-layer 自由度是 R@10 杠杆): **REFUTED** — 3 臂 test R@10 全部 ≤ 0.0938
- **H2** (放宽 κ_max / 延长 phase_b 是杠杆): **REFUTED** — B/C 都负 Δ

## 关键发现: val/test gap

Issue #52 Arm A val NDCG@20 best = 0.0924 (+12.6% vs baseline val 0.0821), 但 test R@10 = 0.0909 (-10.9% vs baseline test 0.1020).

**val/test gap 是 Issue #49 recipe 派生的 structural trait**, 跟 Issue #51 (#338) / Issue #53 Arm B (#340) 完全一致: 即使 codebook 健康 (util 100%, collision ~0.015), test R@10 仍输给 baseline. T5 学 val 局部模式不能 transfer 到 test.

## 决策

- Issue #52 状态: **NO-GO closed (3 臂 R@10 全部 < 0.1020)**
- per-layer lr_theta / kappa_max clamp / phase B 延长都不是 R@10 杠杆
- 跟 Issue #49 / Issue #51 / Issue #53 Arm B 联合立判据: **Issue #49 几何信号 (Recipe + κ-decouple) 不能传导到 T5-mini SID 召回**, 架构层天花板
- Issue #52 GitHub close

## 产物

| Arm | Stage 1 ckpt | SID .npy | Stage 3 ckpt | Stage 4 JSON |
|-----|--------------|----------|--------------|--------------|
| A | products/task339/arm_a/best_collision_model.pth | products/task339/arm_a/sid.npy | products/task339/arm_a/t5_out/Instruments/Jul-30-2026_19-27-49/HG_Rec_best.pth | verdicts/task339_arm_a_stage4_beam20.json |
| B | products/task339/arm_b/best_collision_model.pth | products/task339/arm_b/sid.npy | products/task339/arm_b/t5_out/Instruments/Jul-30-2026_19-29-07/HG_Rec_best.pth | verdicts/task339_arm_b_stage4_beam20.json |
| C | products/task339/arm_c/best_collision_model.pth | products/task339/arm_c/sid.npy | products/task339/arm_c/t5_out/Instruments/Jul-30-2026_19-29-07/HG_Rec_best.pth | verdicts/task339_arm_c_stage4_beam20.json |

result: Issue #52 NO-GO closed. 3 臂 Stage 4 test R@10 全部 < 0.1020 (baseline). val/test gap 是 Issue #49 派生的 structural trait. Issue #52 GitHub 关闭.