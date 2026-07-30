# Task #347 — Issue #59 Stage 1 重训练 #55 Fixed (GPU 3)

## 任务

基于 Issue #58 审计 + #59 修复, 重跑 Issue #55 Stage 1 训练.

## Recipe (跟 #157 完全一致)

| 参数 | 值 |
|------|-----|
| vq_class | `FreeCurvVectorQuantizationMixedCurvWithScaleFixed` (Issue #59 修复) |
| kappa_fixed | 1.0 (跟 HG-Rec baseline 一致) |
| scale_init | 1.0 |
| codebook optimizer | RiemannianAdamWFixed ((1-κ‖x‖²)²/4) |
| other optimizer | AdamW |
| num_emb_list | [64, 128, 256] |
| e_dim | 32 |
| lr | 1e-3 |
| beta | 0.25 |
| num_epochs | 1000 |
| batch_size | 1024 |
| seed | 42 |

## 关键修复 (vs #157)

| Bug | #157 (broken) | #478 (fixed) |
|-----|---------------|---------------|
| α_l grad | None (dead param) | 8.34 (active) |
| scale_l grad | None (dead param) | -0.45 (active) |
| Riemannian formula | (1+‖x‖²)² (错) | (1-κ‖x‖²)²/4 (正) |
| Optimizer 应用范围 | 仅 codebook | codebook + α_l_raw + scale_l (都流形参数) |

## 验证指标

- ✅ α_l 不再全程停在 0.5 → 应该漂移
- ✅ scale_l 不再全程停在 1.0 → 应该漂移
- ✅ Stage 1 loss > 0.01 (vs #157 collapse 0.0002)
- ✅ Stage 2 SID 3-digit unique > 50% (vs #157 collapse 0.01%)

## 决策

| 修复后结果 | 结论 |
|-----------|------|
| α_l+scale_l 漂移 + loss > 0.01 + SID > 50% | Issue #55 重开 GO, 进入 Stage 2/3/4 |
| α_l+scale_l 仍静止 / 仍 collapse | Issue #55 维持 NO-GO |

## 关联产物

- 修复代码: `HG-Rec/model/hrqvae_issue55_56_fixed.py`
- Launcher: `scripts/task478_issue59_stage1_retrain_55_fixed.py`
- 关联 verdict: `verdicts/task169_issue59_bug_fix_sanity_pass.md`
