# Issue #138 Optimization Evidence Verdict

**Issue**: #138 全量曲率失败地图 — 正式 Stage4 24,772 trace 与 L0-L3 几何统计关联
**Generated**: 2026-08-12 15:36:40
**Stage4 full eval R@10**: 0.0968 (Issue #133 baseline 0.0962, diff=0.0006, PASS tolerance)

**注**: 本 verdict 仅汇总事实、稳定现象和待检验假设, 禁止写实现方案. R36 禁调参.

---

## Gate 1 — 全局重现

- Stage4 全量 test R@5=0.0773, R@10=0.0968, R@20=0.1201
- 与 Issue #133 baseline R@10=0.0962 差 0.0006 (tolerance 0.005, PASS)
- 与 Issue #137 5,000 subset R@10=0.1016 方向一致 (Stage4 oracle 稳定)

---

## Gate 2 — 失败位置分布

### first_error_position 分布 (top1 候选第几位 token 错)

- pos 0: 1360 samples (5.49%)
- pos 1: 21891 samples (88.37%)
- pos 2: 1424 samples (5.75%)
- pos 3: 69 samples (0.28%)
- pos 4: 28 samples (0.11%)

### prefix_match_length 分布 (top1 候选前缀匹配 token 数)

- prefix 0: 21891 samples (88.37%)
- prefix 1: 1424 samples (5.75%)
- prefix 2: 69 samples (0.28%)
- prefix 3: 28 samples (0.11%)
- prefix 4: 1360 samples (5.49%)

### best_rank_top20 分布 (exact match rank)

- rank 1: 1360 samples (5.49%)
- rank 2: 201 samples (0.81%)
- rank 3: 147 samples (0.59%)
- rank 4: 119 samples (0.48%)
- rank 5: 95 samples (0.38%)
- rank 6: 103 samples (0.42%)
- rank 7: 91 samples (0.37%)
- rank 8: 97 samples (0.39%)
- rank 9: 87 samples (0.35%)
- rank 10: 106 samples (0.43%)
- rank 11: 75 samples (0.30%)
- rank 12: 76 samples (0.31%)
- rank 13: 52 samples (0.21%)
- rank 14: 59 samples (0.24%)
- rank 15: 44 samples (0.18%)
- rank 16: 70 samples (0.28%)
- rank 17: 68 samples (0.27%)
- rank 18: 50 samples (0.20%)
- rank 19: 44 samples (0.18%)
- rank 20: 43 samples (0.17%)
- miss (rank>20): 21785 samples (87.94%)

---

## Gate 3 — 分桶表现 (Bucket extremes, n≥100)

按 R@10 升序排列最差 bucket:

- L0_freq_quantile=[200,1000): n=104, R@10=0.0385
- history_length=16-20: n=791, R@10=0.0455
- history_length=nan: n=770, R@10=0.0468
- history_length=11-15: n=2003, R@10=0.0629
- history_length=6-10: n=8652, R@10=0.0904
- L0_freq_quantile=[0,100): n=24028, R@10=0.0918
- L3_PAD=PAD=0: n=24717, R@10=0.0966
- L2_uniqueness=[0.20,1.0]: n=24767, R@10=0.0971
- item_popularity=[0,10): n=24772, R@10=0.0971
- history_length=1-5: n=12556, R@10=0.1136

---

## Gate 4 — 曲率关联 (Spearman, top-5 by |rho|)

| feature | test | n | rho | p | q_BH | significant |
| --- | --- | --- | --- | --- | --- | --- |
| L2_freq | Spearman | 7063 | 0.222 | 1.49e-79 | 0.0000 | True |
| L2_freq | Spearman_mean_best_rank | 7063 | -0.1966 | 1.84e-62 | 0.0000 | True |
| L1_freq | Spearman | 7063 | 0.1866 | 2.27e-56 | 0.0000 | True |
| L1_freq | Spearman_mean_best_rank | 7063 | -0.1685 | 3.94e-46 | 0.0000 | True |
| L0_freq | Spearman | 7063 | 0.1442 | 4.08e-34 | 0.0000 | True |

*Correlation, not causation* — Spearman 只测量单调关联, 不蕴含因果.

---

## Gate 5 — 可行动失败现象 (5 个稳定现象)

**判定标准**: 全量 n 充足 + CI 不跨零 + 跨固定子集方向一致 + 可追溯到 manifest.

### P1 — L3 PAD 主导 — target 第 4 位 token 永远是 449 (=raw 0 = PAD)

**Evidence**: target_l3=449 的 sample 占 24717/24772 (99.78%). SID output shape (9922, 4), L3 unique_count=6, L3_PAD_ratio=0.9972 (L3 codebook_size=1 → Stage2 第 4 位 token 分配 bug, add_4th_dedup_digit 未实施). top1_l3=449 sample 24052/24772 (97.09%), 但 exact_R@10 仍 0.0962 — 提示 model 必须靠前 3 位 + trivial L3 完成 hit.

**Stability**: manifest 可追溯: sid L3 codebook_size=1 + L3_PAD_ratio 跨 Issue #133/136/137 一致

### P2 — 前缀匹配 3+ token 命中率显著高于 exact 4-token 命中率

**Evidence**: prefix_match_length >= 3 的 sample 1388/24772 (5.60%), exact_R@20 = 2987/24772 (12.06%). 前缀 3 token 命中率约为 exact 4-token 的 0.5x. 提示 L3 token 预测是 hit 失败的瓶颈 (前 3 位已对齐, 第 4 位偏).

**Stability**: manifest 可追溯: prefix_match_length / hit_at_20 直接来自 raw_predictions parquet

### P3 — 最强 Spearman 关联: L2_freq vs Spearman

**Evidence**: Spearman rho=0.222 (n=7063, p=1.49e-79, q_BH=0.0000, significant_q05=True). Top-5 关联: L2_freq(rho=0.222), L2_freq(rho=-0.1966), L1_freq(rho=0.1866), L1_freq(rho=-0.1685), L0_freq(rho=0.1442).

**Stability**: manifest 可追溯: full_correlation_report.csv + target_stats parquet

### P4 — 最差 bucket: L0_freq_quantile=[200,1000) (R@10=0.0385, n=104)

**Evidence**: 跨 5 个分桶维度 (history_length / L3_PAD / L0_freq / L2_uniqueness / item_popularity), 最差 bucket 是 L0_freq_quantile=[200,1000) (R@10=0.0385, n=104). Top-10 worst: L0_freq_quantile=[200,1000)(R@10=0.0385), history_length=16-20(R@10=0.0455), history_length=nan(R@10=0.0468), history_length=11-15(R@10=0.0629), history_length=6-10(R@10=0.0904).

**Stability**: manifest 可追溯: full_bucket_metrics.csv

### P5 — first_error_position 集中在 L0 (第 1 token)

**Evidence**: first_error_position=1 的 sample 21891/24772 (88.37%). 意味着 top1 prediction 第一个 token 就错的占主导, 后续 L1/L2/L3 错位都建立在此基础之上. L0 vocab=64, 是 4 位 token 中粒度最细的; L0_freq 分布差异可能加剧此现象.

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
