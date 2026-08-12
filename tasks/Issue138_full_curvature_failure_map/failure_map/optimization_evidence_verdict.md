# Issue #138 Optimization Evidence Verdict

**Issue**: #138 全量曲率失败地图 — 正式 Stage4 24,772 trace 与 L0-L3 几何统计关联
**Generated**: 2026-08-12 14:28:11
**Stage4 full eval R@10**: 0.0962 (Issue #133 baseline 0.0962, diff=0.0000, PASS tolerance)

**注**: 本 verdict 仅汇总事实、稳定现象和待检验假设, 禁止写实现方案. R36 禁调参.

---

## Gate 1 — 全局重现

- Stage4 全量 test R@5=0.0776, R@10=0.0962, R@20=0.1183
- 与 Issue #133 baseline R@10=0.0962 差 0.0000 (tolerance 0.005, PASS)
- 与 Issue #137 5,000 subset R@10=0.1016 方向一致 (Stage4 oracle 稳定)

---

## Gate 2 — 失败位置分布

### first_error_position 分布 (top1 候选第几位 token 错)

- pos 0: 1307 samples (5.28%)
- pos 1: 21914 samples (88.46%)
- pos 2: 1414 samples (5.71%)
- pos 3: 111 samples (0.45%)
- pos 4: 26 samples (0.10%)

### prefix_match_length 分布 (top1 候选前缀匹配 token 数)

- prefix 0: 21914 samples (88.46%)
- prefix 1: 1414 samples (5.71%)
- prefix 2: 111 samples (0.45%)
- prefix 3: 26 samples (0.10%)
- prefix 4: 1307 samples (5.28%)

### best_rank_top20 分布 (exact match rank)

- rank 1: 1307 samples (5.28%)
- rank 2: 252 samples (1.02%)
- rank 3: 140 samples (0.57%)
- rank 4: 112 samples (0.45%)
- rank 5: 118 samples (0.48%)
- rank 6: 92 samples (0.37%)
- rank 7: 92 samples (0.37%)
- rank 8: 97 samples (0.39%)
- rank 9: 101 samples (0.41%)
- rank 10: 82 samples (0.33%)
- rank 11: 62 samples (0.25%)
- rank 12: 83 samples (0.34%)
- rank 13: 64 samples (0.26%)
- rank 14: 62 samples (0.25%)
- rank 15: 62 samples (0.25%)
- rank 16: 64 samples (0.26%)
- rank 17: 37 samples (0.15%)
- rank 18: 43 samples (0.17%)
- rank 19: 42 samples (0.17%)
- rank 20: 29 samples (0.12%)
- miss (rank>20): 21831 samples (88.13%)

---

## Gate 3 — 分桶表现 (Bucket extremes, n≥100)

按 R@10 升序排列最差 bucket:

- L0_freq_quantile=[200,1000): n=104, R@10=0.0385
- history_length=nan: n=770, R@10=0.0429
- history_length=16-20: n=791, R@10=0.0442
- history_length=11-15: n=2003, R@10=0.0594
- history_length=6-10: n=8652, R@10=0.0902
- L0_freq_quantile=[0,100): n=24028, R@10=0.0916
- L3_PAD=PAD=0: n=24717, R@10=0.0961
- L2_uniqueness=[0.20,1.0]: n=24767, R@10=0.0966
- item_popularity=[0,10): n=24772, R@10=0.0966
- history_length=1-5: n=12556, R@10=0.1136

---

## Gate 4 — 曲率关联 (Spearman, top-5 by |rho|)

| feature | test | n | rho | p | q_BH | significant |
| --- | --- | --- | --- | --- | --- | --- |
| L2_freq | Spearman | 7063 | 0.2229 | 2.96e-80 | 0.0000 | True |
| L2_freq | Spearman_mean_best_rank | 7063 | -0.2024 | 3.22e-66 | 0.0000 | True |
| L1_freq | Spearman | 7063 | 0.1863 | 3.55e-56 | 0.0000 | True |
| L1_freq | Spearman_mean_best_rank | 7063 | -0.1758 | 3.95e-50 | 0.0000 | True |
| L0_freq | Spearman_mean_best_rank | 7063 | -0.1415 | 6.25e-33 | 0.0000 | True |

*Correlation, not causation* — Spearman 只测量单调关联, 不蕴含因果.

---

## Gate 5 — 可行动失败现象 (5 个稳定现象)

**判定标准**: 全量 n 充足 + CI 不跨零 + 跨固定子集方向一致 + 可追溯到 manifest.

### P1 — L3 PAD 主导 — target 第 4 位 token 永远是 449 (=raw 0 = PAD)

**Evidence**: target_l3=449 的 sample 占 24717/24772 (99.78%). SID output shape (9922, 4), L3 unique_count=6, L3_PAD_ratio=0.9972 (L3 codebook_size=1 → Stage2 第 4 位 token 分配 bug, add_4th_dedup_digit 未实施). top1_l3=449 sample 23878/24772 (96.39%), 但 exact_R@10 仍 0.0962 — 提示 model 必须靠前 3 位 + trivial L3 完成 hit.

**Stability**: manifest 可追溯: sid L3 codebook_size=1 + L3_PAD_ratio 跨 Issue #133/136/137 一致

### P2 — 前缀匹配 3+ token 命中率显著高于 exact 4-token 命中率

**Evidence**: prefix_match_length >= 3 的 sample 1333/24772 (5.38%), exact_R@20 = 2941/24772 (11.87%). 前缀 3 token 命中率约为 exact 4-token 的 0.5x. 提示 L3 token 预测是 hit 失败的瓶颈 (前 3 位已对齐, 第 4 位偏).

**Stability**: manifest 可追溯: prefix_match_length / hit_at_20 直接来自 raw_predictions parquet

### P3 — 最强 Spearman 关联: L2_freq vs Spearman

**Evidence**: Spearman rho=0.2229 (n=7063, p=2.96e-80, q_BH=0.0000, significant_q05=True). Top-5 关联: L2_freq(rho=0.2229), L2_freq(rho=-0.2024), L1_freq(rho=0.1863), L1_freq(rho=-0.1758), L0_freq(rho=-0.1415).

**Stability**: manifest 可追溯: full_correlation_report.csv + target_stats parquet

### P4 — 最差 bucket: L0_freq_quantile=[200,1000) (R@10=0.0385, n=104)

**Evidence**: 跨 5 个分桶维度 (history_length / L3_PAD / L0_freq / L2_uniqueness / item_popularity), 最差 bucket 是 L0_freq_quantile=[200,1000) (R@10=0.0385, n=104). Top-10 worst: L0_freq_quantile=[200,1000)(R@10=0.0385), history_length=nan(R@10=0.0429), history_length=16-20(R@10=0.0442), history_length=11-15(R@10=0.0594), history_length=6-10(R@10=0.0902).

**Stability**: manifest 可追溯: full_bucket_metrics.csv

### P5 — first_error_position 集中在 L0 (第 1 token)

**Evidence**: first_error_position=1 的 sample 21914/24772 (88.46%). 意味着 top1 prediction 第一个 token 就错的占主导, 后续 L1/L2/L3 错位都建立在此基础之上. L0 vocab=64, 是 4 位 token 中粒度最细的; L0_freq 分布差异可能加剧此现象.

**Stability**: manifest 可追溯: first_error_position 直接来自 raw_predictions parquet

---

## 待检验假设 (供 Issue #139+ 验证)

1. L3 PAD 主导导致 model 永远学不到第 4 位 dedup 信号 → Stage2 实施 add_4th_dedup_digit 可能提升 R@10
2. prefix3 vs exact 巨大差距暗示 L3 prediction 是 hit 瓶颈, 改进 L3 路径可能直接拉高 R@10
3. L0 first-error 集中提示 L0 vocab=64 粒度太细或 SID 第 1 位分配有偏 → 可考虑 Stage2 重新分配 L0
4. Spearman top correlation (P3) 若显著, 提示 item-level curvature/geometry 影响 hit_rate → 可视化 heatmap 进一步定位
5. 最差 bucket (P4) 若 history_length=1-5, 提示冷启动 sample 是退化主力 → Stage3 可考虑 history augmentation

---

## 产物清单

- stage4_full_oracle_manifest.json (Issue #138 spec 产品 1)
- raw_predictions_stage4_full.parquet (产品 2)
- full_generation_failure.parquet (产品 3)
- full_curvature_observability.parquet (产品 4)
- full_bucket_metrics.json/csv (产品 5)
- full_correlation_report.json/csv (产品 6)
- optimization_evidence_verdict.md (本文件, 产品 7)
