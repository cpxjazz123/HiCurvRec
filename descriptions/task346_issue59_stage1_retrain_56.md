# Task #346 — Issue #59 Stage 1 重训练 #56 Fixed (GPU 0)

## 任务

基于 Issue #58 审计 + #59 修复 (`hrqvae_issue55_56_fixed.py` 5/5 sanity PASS), 重跑 Issue #56 Stage 1 训练.

## Recipe (跟 #156 完全一致)

| 参数 | 值 |
|------|-----|
| vq_class | `FreeCurvVectorQuantizationMixedCurvFixed` (Issue #59 修复) |
| kappa_fixed | 0.74 (Ollivier mean from #70) |
| num_emb_list | [64, 128, 256] |
| e_dim | 32 |
| lr | 1e-3 |
| beta | 0.25 |
| num_epochs | 1000 |
| batch_size | 1024 |
| seed | 42 |
| device | cuda |

## 关键修复 (vs #156)

| Bug | #156 (broken) | #477 (fixed) |
|-----|---------------|---------------|
| α_l grad | None (dead param) | 8.34 (active) |
| commitment loss path | 用 fixed kappa_m() | 用 mixed_curv_dist(α) |
| Riemannian formula | 没用 optimizer | N/A (用 AdamW) |

## 验证指标

- ✅ α_l 不再全程停在 0.5 → 应该漂移
- ✅ Stage 1 loss > 0.01 (vs #156 collapse 0.0002)
- ✅ Stage 2 SID 3-digit unique > 50% (vs #156 collapse 0.01%)

## 决策

| 修复后结果 | 结论 |
|-----------|------|
| α_l 漂移 + loss > 0.01 + SID > 50% | Issue #56 重开 GO, 进入 Stage 2/3/4 |
| α_l 仍静止 / 仍 collapse | Issue #56 维持 NO-GO, 方向本身失败 |

## 关联产物

- 修复代码: `HG-Rec/model/hrqvae_issue55_56_fixed.py`
- Launcher: `scripts/task477_issue59_stage1_retrain_56_fixed.py`
- 关联 verdict: `verdicts/task169_issue59_bug_fix_sanity_pass.md`
- 关联 issue: #58 (审计), #59 (修复+重跑)
