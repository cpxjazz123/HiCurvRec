# Task #279 — K-sweep 扩展 K=512 + K=1024

## 背景

Task #278 (2026-07-29) 批量 Stage 4 eval 揭示 K-sweep 趋势:

| K (L0) | R@10 | vs baseline 0.1020 |
|--------|------|--------------------|
| 32 | 0.1006 | -1.4% |
| 64 | 0.1041 | +2.1% |
| 128 | 0.1027 | +0.6% |
| **256** | **0.1053** | **+3.3%** ⭐ 最佳 |

**Insight**: L0 codebook size 越大 R@10 越好 (单调上升除 K=128 微跌). 验证 K=512 / K=1024 是否进一步突破 0.1053 是高 ROI 候选 (R10 backlog D2).

## 任务范围

1. **Stage 1 RQ-VAE retrain**: 跑 `train_hrqvae.py` with `--num_emb_list K 128 256` (K ∈ {512, 1024}) + epoch=1000 (Paper Table 6 对齐)
2. **Stage 2 Sinkhorn codebook**: 跑 `task194_stage2_codebook.py` 生成 `_t5_rqvae_k0512.npy` + `_t5_rqvae_k1024.npy` (max_sinkhorn_iters=30)
3. **Stage 3 T5-mini training**: 跑 `task194_stage3_only.sh` with codebook_size=[K, 128, 256, 1]
4. **Stage 4 eval**: 跑 `task278_batch_stage4_eval.py` (v3 with CLI args) with --codebook_size "K,128,256,1"
5. **Verdict**: 综合 K-sweep 4-arm (K=32/64/128/256/512/1024) R@10 趋势

## 关键决策点 (R11.5 + 用户 override "do by yourself")

- **自主执行**: loop.md §16 D2 backlog + R10 主动推进 + 用户 override "不允许等用户拍板" → 直接 launch
- **GPU 并行**: K=512 跑 GPU 0, K=1024 跑 GPU 1 (互不抢卡). 估约 3-4 小时总耗时
- **batch_size=1024**: K=512/K=1024 + kmeans_init=True 需要 batch_size ≥ K (避免 sklearn `n_samples < n_clusters` ValueError). task194 K=256 用 batch_size=256 OK (256=256), K=512/1024 必须 ≥ K
- **不加新架构**: 纯 K-sweep 扩展, 复用 task194 框架 (Stage 1+2+3+4 全套), 不写新代码
- **可中断**: 任何 K 失败可单独补跑

## Stage 4 期望 (vs baseline 0.1020)

| K | 假设 R@10 | 备注 |
|---|-----------|------|
| 512 | 0.105-0.110 | 持续上升趋势验证 |
| 1024 | 0.100-0.108 | 码本过大可能过拟合, 反而降 |

## 已知约束 (R11.5)

1. **batch_size ≥ K**: kmeans_init=True 在 batch_size<K 时报 `n_samples < n_clusters`. 必须 batch_size=1024 for K=512/1024 (task194 K=256 用 batch_size=256 OK)
2. **vocab_size ≥ sum(K)+1**: `item2code` 用 cumulative offset — K=1024 L2 最大 offset = 1024+128+256+1 = 1409. 必须 vocab_size ≥ 1410 否则 `indexSelectLargeIndex` assertion. K=512 vocab_size=1025 仍够 (max offset = 897)
3. **DISABLE_USAGE_KILL=1**: K=512/1024 在 epoch 30 时 L0 利用率 < 20% (USAGE-KILL) 提前 abort. 禁用 kill 让训练继续 (500 epoch 应能恢复利用率)

## 物理产物

```
descriptions/task279_k_sweep_k512_k1024.md  (本文件)
scripts/task279_k_sweep_dispatch.sh  (K=512 + K=1024 并行 launcher)
products/task279/hrqvae_k0512/  (K=512 RQ-VAE ckpt)
products/task279/hrqvae_k1024/  (K=1024 RQ-VAE ckpt)
products/task279/t5mini_k0512/  (K=512 T5-mini ckpt)
products/task279/t5mini_k1024/  (K=1024 T5-mini ckpt)
HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0512.npy
HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k1024.npy
verdicts/task279_k0512_test_metrics.json
verdicts/task279_k1024_test_metrics.json
verdicts/task279_k_sweep_extension_result.md
logs/task279/stage1_k0512.log
logs/task279/stage1_k1024.log
logs/task279/stage4_k0512_eval.out
logs/task279/stage4_k1024_eval.out
```

result: Task #279 — K-sweep 扩展 K=512 + K=1024. 目的: 验证 Task #194 K-sweep 趋势 (K=256 R@10=0.1053 最佳) 是否在 K=512/1024 继续上升. Stage 1 RQ-VAE retrain (1000 epoch, paper-aligned) + Stage 2 Sinkhorn + Stage 3 T5-mini + Stage 4 eval. 2 卡并行估约 3-4 小时总耗时. 复用 task194 框架 + task278 Stage 4 v3 driver.