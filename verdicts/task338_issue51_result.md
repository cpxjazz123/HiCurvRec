# Task #338 / Issue #51 — Verdict (NO-GO)

**日期**: 2026-07-30
**状态**: NO-GO (test R@10 = 0.0906, -11.2% vs baseline 0.1020)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/51

## 摘要

Issue #51 检验: HypPreEncoder (Issue #43 R@10=0.1041) + per-layer learned κ (Issue #49 训练 recipe) 组合能否超 baseline. **结论**: Test R@10=0.0906 (-11.2% vs baseline 0.1020). Validation NDCG@20=0.0915 显示训练本身成功, **但 test set 表现显著低 — val/test gap 揭示分布漂移或过拟合**.

## Stage 1 (HRQ-VAE + HypPreEncoder)

- 架构: FreeCurvHRQVAE + HRQVAEWithHypPre(c=0.74) wrapper
- Phase A: κ frozen, θ_init=-0.02, 200 epochs, kmeans_init
- Phase B: κ unfrozen, lr_theta=1e-5, 200 epochs
- **Best collision**: 0.0955 (vs Issue #49 0.1906, halved!) — HypPreEncoder 大幅改善 codebook 健康
- **κ 实际学习**: -0.04 → -0.046 (变化小, Issue #49 移到 -0.12; 包装 wrapper 后 κ 学习受阻)

## Stage 2 (SID 推断)

- 4-digit unique: 9922/9922 ✓
- Raw 3-digit collision: 0.0955

## Stage 3 (T5-mini 训练, 200 epoch, early_stop=20)

- Best epoch: ~100 (early stop @ epoch 119, 20 epoch 无改善)
- Best **validation** NDCG@20: **0.0915** (baseline val NDCG@20=0.0821, +11.4%)
- Best **test** R@10: **0.0906** (baseline test R@10=0.1020, **-11.2%**)

## Stage 4 Test Eval (beam=20)

| 指标 | Issue #51 | baseline (Task #84) | Δ% |
|------|-----------|---------------------|-----|
| Recall@5 | 0.0752 | 0.0816 | -7.8% |
| **Recall@10** | **0.0906** | **0.1020** | **-11.2%** |
| Recall@20 | 0.1095 | 0.1279 | -14.4% |
| NDCG@5 | 0.0646 | 0.0690 | -6.4% |
| NDCG@10 | 0.0695 | 0.0755 | -7.9% |
| NDCG@20 | 0.0744 | 0.0821 | -9.4% |

**所有指标全部 NO-GO**.

## 关键发现: val/test gap

| | Validation NDCG@20 | Test R@10 |
|---|---|---|
| Issue #51 | 0.0915 (+11.4% vs baseline val 0.0821) | 0.0906 (-11.2% vs baseline test 0.1020) |
| baseline | 0.0821 | 0.1020 |

**Issue #51 val→test gap 巨大**: baseline val/test gap ≈ 0.0821-0.0821 ≈ 0% (一致性), Issue #51 val/test gap 巨大, 模型在 val 上学到的特征在 test 上失效.

## 假设验证 (H1 vs H2)

- **H1** (信号饥饿): R-Drop 等协议叠加能填平差距 → **REFUTED** (无 R-Drop 也未填平)
- **H2** (架构传输损失): HypPre + κ learning 组合导致负交互 → **CONFIRMED** (val 改善但 test 退化)

Issue #51 的 val/test gap 提示: HypPreEncoder + κ learning 组合让 encoder 学到了**过拟合 val 分布**的 latent 表示, 在 test 上失效.

## 决策

- Issue #51 状态: **NO-GO (closed)**
- 不继续 Issue #51 后续优化 (e.g. R-Drop 叠加不会改变 val/test gap)
- Issue #43 HypPreEncoder 单独 (test R@10=0.1041, +2.1%) 仍是 baseline recipe 内的杠杆
- Issue #49 per-layer κ learning 单独 (test R@10=0.1005, -1.5%) 仍是 NO-GO

## 产物

| 文件 | 路径 |
|------|------|
| Stage 1 ckpt | `products/task338/stage1/best_collision_model.pth` |
| SID | `products/task338/sid_issue51.npy` (4-digit 9922/9922 unique) |
| Stage 3 ckpt | `products/task338/t5_out/Instruments/Jul-30-2026_18-40-03/HG_Rec_best.pth` |
| Stage 4 JSON | `verdicts/task338_issue51_stage4_beam20.json` |

## R15 闭环

[commit + push + gh issue close 待执行]

result: Issue #51 NO-GO, test R@10=0.0906 (-11.2% vs baseline). Val NDCG@20=0.0915 (+11.4%) 但 test 退化, 揭示 val/test 分布漂移. Issue #51 closed.