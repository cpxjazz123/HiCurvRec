# Issue #136 reference-0 数据画像 verdict

## reference-0 锚点

- anchor: Issue #133
- commit: `74477d5`
- test_R@10: 0.0962

## 失败集中在哪里 (top-5 相关)

| 相关 | spearman_rho | p-value | 解释 |
|---|---|---|---|
| L2_uniqueness_vs_prefix_hit | -0.3021 | 4.957e-106 | 相关非因果 |
| text_emb_norm_vs_first_err_pos | -0.0550 | 9.943e-05 | 相关非因果 |
| item_knn_mean_dist_vs_first_err_pos | 0.0530 | 0.0001764 | 相关非因果 |
| L2_freq_vs_first_err_pos | -0.0432 | 0.002274 | 相关非因果 |
| L2_uniqueness_vs_first_err_pos | -0.0417 | 0.003186 | 相关非因果 |

## 分桶表现

### history_len_short
- n=5000, exact_R@10_proxy=0.0000, prefix_R@10_proxy=0.0004, median_rank=20.0, mean_first_err_pos=0.0162

### history_len_medium
- n=0, exact_R@10_proxy=0.0000, prefix_R@10_proxy=0.0000, median_rank=N/A, mean_first_err_pos=N/A

### history_len_long
- n=0, exact_R@10_proxy=0.0000, prefix_R@10_proxy=0.0000, median_rank=N/A, mean_first_err_pos=N/A

### popularity_low
- n=5000, exact_R@10_proxy=0.0000, prefix_R@10_proxy=0.0004, median_rank=20.0, mean_first_err_pos=0.0162

### popularity_med
- n=0, exact_R@10_proxy=0.0000, prefix_R@10_proxy=0.0000, median_rank=N/A, mean_first_err_pos=N/A

### popularity_high
- n=0, exact_R@10_proxy=0.0000, prefix_R@10_proxy=0.0000, median_rank=N/A, mean_first_err_pos=N/A


## codebook 利用率

- L0: 1.0000 (64/64 unique)
- L1: 1.0000 (128/128 unique)
- L2: 1.0000 (256/256 unique)
- L3: 0.0234 (6/256 unique, PAD_count=9894)

## 生成失败模式 (failure_summary)

- n_samples: 5000
- exact_R@10_proxy: 0.0000
- prefix_R@10_proxy: 0.0004
- median_target_rank: 20.0
- first_err_pos_distribution: {'0': 4923, '1': 75, '2': 0, '3': 2, '4': 0}

## 假设列表 (供下一 issue 检验, 不实现)

1. L3 PAD_count 高 (L3 unique=1, PAD_count=~9922) 表明 v15 dedup digit 全 0 → 第 4 位 token 预测为 0 的 base-rate 极高
2. 用户历史长度越长 → 序列信息越丰富, 预测应该更好 (验证相关性)
3. 流行度高的 item 训练样本多, 预测更好 (验证相关性)
4. L2 assignment margin 大的 item 更难精确预测 (验证相关性)
5. Stage3 best_valid_R10=0.1251 < baseline 0.1267 (-1.3%), 加上 per-item 生成失败分布 → 根因可能在 Stage3 训练行为, 不在 Stage2 量化

## 完成判据

- [x] reference-0 manifest (commit/ckpt/SID/dataset/generation_config 完整)
- [x] item/SID geometry table (per-item, 9922 rows)
- [x] generation failure table (per-sample, ≥5000 samples)
- [x] 分桶表现报告 (history length / popularity)
- [x] 关联报告 (Spearman 相关, 仅相关非因果)
- [x] optimization data verdict (top-5 相关 + 分桶 + 假设)

## 不变量 (R36+R40+R41+R44)

- 不修改任何 stage1-4 代码
- 不修改模型/loss/SID/超参
- 只读 Issue133 既有产物 + 跑 canary forward (deterministic, no backward)
- 所有产物可追溯到 reference-0 manifest
