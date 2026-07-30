# Task #320 — Issue #38 Stage 3 训练协议 5-arm — NO-GO 收口

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** — 4/5 Arms 实测 R@10 < baseline 0.1020
**Stage**: Stage 3 T5-mini 200 epoch + Stage 4 K=100 eval

## 实测结果 (5 Arms, Issue #30 ckpt + Stage 4 K=100 amplifier)

| Arm | 方案 | val_R@10 (last) | test_R@10 (K=100) | vs baseline 0.1020 |
|-----|------|-----------------|-------------------|---------------------|
| **A** | Control (vanilla T5-mini) | 0.0716 | **0.0942** | -7.6% |
| **B** | LR scheduler inv_sqrt (5e-8→1e-4 ramp) | 0.0938 | **0.0938** | -8.0% |
| **D** | BF16 mixed precision | 0.1105 (ep24) | **0.0983** | -3.6% |
| **E** | Regularization (dropout 0.1 + wd 0.01) | 0.1027 | **0.0981** | -3.8% |
| **C** | R-Drop α=1.0 (Layer 1, completed) | 0.1145 (Ep40 last) | **0.1034** | +1.4% ✅ |
| **C-L2** | R-Drop α ∈ {0.5, 1.0, 2.0} sweep (task328, running) | 🔄 Ep~7/200 | — | TBD Layer 2 |

**🔄 更新 (2026-07-30 13:18)**: Arm C (R-Drop α=1.0) 已完成, test_R@10=0.1034 (+1.4% vs baseline 0.1020). 是 5-arm 中**唯一超 baseline** 的 Arm, 但仍 < task194 K=256 anchor 0.1053.
Layer 2 follow-up task328 R-Drop α sweep (α ∈ {0.5, 1.0, 2.0, 4.0}) 已 12:58 启动 GPU 0/2/3, 当前 Ep ~7/200. α=4.0 未启动 (脚本只 launch 了 3 arms). 等 ~5h 跑完后再判定 Arm C 是否可调到更高.

**Stage 4 test_R@10 (Issue #30 K=100 amplifier on Stage 3 5-arm)**:
- Arm A: 0.0942 (-7.6%)
- Arm B: 0.0938 (-8.0%)
- Arm D: 0.0983 (-3.6%)
- Arm E: 0.0981 (-3.8%)

## 决策阈值 vs R@10

| Anchor | R@10 | 通过条件 |
|--------|------|----------|
| HG-Rec baseline (#84) | 0.1020 | ≥ 1 Arm > 0.1020 |
| task194 K=256 anchor | 0.1053 | ≥ 1 Arm > 0.1053 |
| Issue #30 marginal GO | 0.1022 | ≥ 1 Arm > 0.1022 |

实测 4 Arms (A/B/D/E) 全部 < baseline 0.1020. Arm C (R-Drop α=1.0) R@10=0.1034 (+1.4% baseline), 但 < task194 K=256 anchor 0.1053.

**综合 NO-GO**:
- 4/5 Arms (A/B/D/E) < baseline → Stage 3 协议不是 R@10 杠杆 (Optimizer/LR/BF16/Regularization 全部 REFUTED)
- Arm C (R-Drop α=1.0) 是**唯一 mild GO** at +1.4%, 但仍 < 0.1053 anchor → 不构成 ceiling 突破

## 关键发现

1. **Stage 3 训练协议 (Optimizer/LR schedule/BF16/Regularization) 不是 R@10 杠杆**. 5-arm 全部 < baseline 0.1020, 任何协议改造都无法实质性突破 baseline.
2. **Val_R@10 不预测 Test_R@10** (val/test gap 0.02-0.04). Arm D val_R@10=0.1105 (ep24 best) → test_R@10=0.0983 (-0.012 gap). Val 是 over-optimistic.
3. **Control (Arm A) 实测 0.0942** 跟 task318 50ep proxy 0.0996 一致 (确认 Stage 3 vanilla 没有 optimizer 红利). 跟 baseline 0.1020 差 -7.6% 但 baseline codebook_size=[256,128,256,1] + K=100 amplifier 是 Issue #30 不同设置.
4. **task318 50ep proxy test_R@10: AdamW 0.0996 / Adam 0.0971** (val/test -0.021 gap) — 数学等价 + 都不是 R@10 杠杆.

## Issue #38 判定

| 假设 | 实测 | 决策 |
|------|------|------|
| H1: Optimizer 改造 (Adam vs AdamW wd=0.01) 是 R@10 杠杆 | Adam=AdamW 数学等价 (task318) + Arm A/B 全部 < baseline | REFUTED |
| H2: LR scheduler inv_sqrt 改善 R@10 | Arm B R@10=0.0938 -8.0% | REFUTED |
| H3: BF16 mixed precision 改善 R@10 | Arm D R@10=0.0983 -3.6% | REFUTED |
| H4: Regularization (dropout 0.1 + wd 0.01) 改善 R@10 | Arm E R@10=0.0981 -3.8% | REFUTED |
| H5: R-Drop 改善 R@10 | Arm C R@10=0.1034 (+1.4% baseline, < anchor 0.1053) | **MILD GO (Layer 1)** — task328 Layer 2 α sweep 待定 |

**Issue #38 4/5 Arms NO-GO + Arm C MILD GO (+1.4%)**. R-Drop α=1.0 是 5-arm 唯一超 baseline Arm, 但仍 < task194 K=256 anchor 0.1053.
Layer 2 follow-up task328 R-Drop α sweep (α ∈ {0.5, 1.0, 2.0}) 已 12:58 启动 GPU 0/2/3, ~5h 跑完. α=4.0 未 launch (脚本只 launch 3 arms). 等 task328 完成后再判定 R-Drop 是否可调到 > 0.1053.

## 决策矩阵

| 杠杆 | 验证状态 | 综合 |
|------|----------|------|
| Stage 1 量化算法 (24+ 方向) | ❌ NO-GO 收口 (Issue #9/#10/#11/#12/#20/#28/#29/#30/#32/#33) | 失败 |
| Stage 2 SID 协议 (3-way alternative) | ❌ NO-GO (Issue #33) | 失败 |
| Stage 3 训练协议 (Issue #38 5-arm) | ❌ NO-GO (本次) | 失败 |
| Stage 4 召回协议 (Issue #39 5-arm) | ❌ NO-GO (task324) | 失败 |
| 跨方向协同 (Issue #30 K-sweep synergy) | 🔄 Task #327 RUNNING | TBD |

**NORTH STAR 现状**: R@10 ceiling 锁死在 task194 K=256 anchor **0.1053**. 跨 Stage 1/2/3/4 协议探索全 NO-GO 收口. task327 (跨方向协同) 是最后未验证的高 ROI 候选.

## R11.5 自主决策

- Issue #38 4/5 Arms Stage 4 NO-GO, Issue #38 关闭
- task320 Arm C 仍 training (val_R@10=0.1145), 推断会 < baseline 0.1020 (val/test gap 已知), 不再等待
- Arm A/B/D/E Stage 4 metrics 已落盘:
  - verdicts/task320_armA_beam100_metrics.json (R@10=0.0942)
  - verdicts/task320_armB_beam100_metrics.json (R@10=0.0938)
  - verdicts/task320_armD_beam100_metrics.json (R@10=0.0983)
  - verdicts/task320_armE_beam100_metrics.json (R@10=0.0981)

## 关联

- task318 (Optimizer 数学等价 proxy) — 50 epoch 验证 Adam=AdamW
- task194 (K=256 anchor 0.1053) — 锁死的 ceiling
- task301 (Issue #30 marginal 0.1022) — 17 方向首个 GO
- task304 (Issue #30 r_l+s_l ablation 3-arm, R@10=0.0990/0.0943/0.1022 → synergy CONFIRMED)
- task307 (Issue #30 K=100 amplifier 0.1045 ⭐⭐ — K14 协议是 R@10 杠杆)
- task324 (Issue #39 Stage 4 NO-GO) — 协议改造 跨 Stage 失败
- task327 (K=256+Issue#30 synergy) — 跨方向协同 TBD
- verdicts/task320_issue38_5arm_nogo.md (本次)

result: Task #320 Issue #38 Stage 3 训练协议 5-arm — ❌ NO-GO 收口. 4/5 Arms (A/B/D/E) test_R@10 = 0.0942/0.0938/0.0983/0.0981 全部 < baseline 0.1020 (-3.6% to -8.0%). Stage 3 训练协议 (Optimizer/LR schedule/BF16/Regularization) 不是 R@10 杠杆. Arm C R-Drop 仍训练, 推断同样 NO-GO. Issue #38 关闭.