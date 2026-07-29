# Task #309 — T5-mini → T5-small 升级 + beam=50 (Issue #38 Arm α 候选)

**日期**: 2026-07-30
**状态**: **NO-GO** (R@10 = 0.0979, vs baseline #84 = 0.1020, -4.0%)
**Stage**: Stage 4 eval 完成

## Settings
- T5-small (60M params, d_model=512, d_ff=2048, num_heads=8, num_layers=6, num_decoder_layers=6)
- SID code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy` (Issue #30 GO config)
- Beam size: 50 (K14 optimal)
- 200 epoch Stage 3 training early-stopped at epoch 41 (HG_Rec_best.pth saved)

## Metric (test_*, 97 batches)
- Recall@5 = 0.0799
- Recall@10 = **0.0979** (-4.0% vs baseline 0.1020)
- Recall@20 = 0.1246
- NDCG@5 = 0.0680
- NDCG@10 = 0.0737
- NDCG@20 = 0.0804

## Verdict
**T5-mini (9.18M, d_model=128) → T5-small (60M, d_model=512) upgrade with beam=50 yields R@10=0.0979 (-4.0% vs baseline 0.1020).** Smaller architecture (T5-mini) appears better calibrated to the SID task.

This rules out Issue #38 Arm α (model scale-up) as a R@10 lever. The underlying reason is likely overfitting: 60M params on 9.9K items × ~256 batch with 200 epoch budget is overparameterized. Even with K14 beam=50, performance regresses.

## Confirmed (positive)
- K14 beam=50 protocol still applicable to T5-small (no crash)
- Stage 3+4 pipeline cleanly handles T5-small architecture

## Next steps
- Arm α → CLOSED (R11.5 NO-GO confirmed)
- Issue #38 Arm β/γ/ε (other orthogonal levers) still valid candidates
- task312 (s_l isolation) + task313 (r_l isolation) Stage 3 in progress, will test Arm γ candidates

## R11.3 transparency
- Chose T5-small (not T5-base 220M) because (a) T5-mini→T5-small is single order magnitude step, (b) T5-base would 36× params vs T5-mini = severe overfitting risk
- Beam=50 chosen because K14 (Stage 4 inference protocol) found it optimal across all previous experiments

## Anchor reference
| Config | R@10 | Δ vs baseline |
|--------|------|---------------|
| baseline #84 (T5-mini, beam=20) | 0.1020 | 0 |
| **task309 (T5-small, beam=50)** | **0.0979** | **-4.0%** |
| task194_k0256 (T5-mini, beam=20) | 0.1053 | +3.3% (anchor) |
| Issue #30 GO (T5-mini, beam=20) | 0.1022 | +0.2% |
| Issue #35 (Task #301) (T5-mini, beam=50) | 0.1022 | +0.2% |
