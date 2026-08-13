# Issue #152 — Stage2 Prefix-conditioned Curvature B arm 独立运行 Verdict

## 总结

| 指标 | A arm (baseline, 4卡) | B arm (本任务, 4卡) | Δ |
|---|---|---|---|
| Stage2 final c_global | [0.609, 0.601, 0.608] | [0.614, 1.878, 1.895] | L1/L2 大幅分化 |
| Stage2 final δ_abs_mean | [0, 0, 0] | [0, 1.453, 1.446] | 路由器实际学习 delta |
| Stage2 SID 4-digit unique | 9187/9922 | 9262/9922 | ≈ |
| Stage3 best valid R@10 | **0.1341** (epoch 174) | **0.1343** (epoch 169) | +0.0002 |
| **Stage4 test R@10** | **0.1033** | **0.1027** | **-0.0006** |
| **Stage4 NDCG@10** | 0.0745 | 0.0756 | +0.0011 |
| 95% paired bootstrap CI | — | — | [-0.0029, +0.0016] (跨 0) |
| McNemar only_A / only_B | — | — | 403 / 388, p=0.619 |

## 判定: **INVALID**

B arm (prefix-conditioned curvature routing) 在独立任务、4 卡 DDP 完整运行时:
- Stage2 路由器正常工作 (L1/L2 c 分化到 1.88, δ≈1.45, 与 Issue #147 treatment 逐位一致)
- Stage3 valid R@10 几乎持平 (0.1343 vs 0.1341)
- **Stage4 test R@10 反而略低** (0.1027 vs 0.1033, Δ=-0.0006)
- paired CI [-0.0029, +0.0016] 跨 0, McNemar p=0.619 → 统计不显著

### 与 Issue #147 对比 (关键)

| 运行 | A test R@10 | B test R@10 | Δ | CI | McNemar p |
|---|---|---|---|---|---|
| #147 run1 (2卡/arm) | 0.1094 | 0.1113 | +0.0018 | [-0.0007, +0.0042] | 0.145 |
| #147 run2 (2卡/arm) | 0.1046 | 0.1068 | +0.0022 | [-0.0001, +0.0046] | 0.067 |
| **#152 (4卡 DDP 独立)** | 0.1033 | 0.1027 | **-0.0006** | [-0.0029, +0.0016] | 0.619 |

三次独立运行: B 均值方向在 +0.0018 / +0.0022 / **-0.0006** 之间翻转, paired CI 全部跨 0 → 差异属随机噪声, B arm 相对 A arm 无统计显著提升。与 Issue #147 INVALID 判定一致, 且本次以更严格的 4 卡 DDP 独立运行确认。

### Stage2 数值复现 (B 路径机制健康)
- B final_cs_global = [0.614, 1.878, 1.895] (#147 treatment: [0.607, 1.875, 1.892])
- B δ_abs_mean_final = [0, 1.453, 1.446] (#147: [0, 1.456, 1.451])
- SID unique 4-digit 9262, 无 NaN/Inf, 无边界饱和

## 产物
- `stage1/item_emb.parquet` (SHA=1a6dd2ac, 与 A 一致)
- `stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, verdict.json, train_log.jsonl, _TRAINING_PID}`
- `stage2/eval/{raw_predictions_stage4_full.parquet, eval_test.json}`
- `paired_prediction_analysis/paired_prediction_analysis.json`
- `issue152_verdict.json`

## 结论
B arm 独立完整运行确认: prefix-conditioned curvature routing 机制可实现 (Gate 2 数值健康), 但对最终推荐无统计显著增益 (paired CI 跨 0 且方向翻转)。不进入 baseline, Issue #152 关闭归档。
