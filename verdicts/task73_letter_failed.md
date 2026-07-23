# Task #73 — LETTER tokenizer DEFERRED (failed to converge)

> **Date**: 2026-07-22 14:56
> **PID killed**: 1401434 + children (4 processes)
> **Status**: ❌ **FAILED to converge** — cf_loss stuck at 168.84

## 现象

跑 LETTER RQ-VAE 在我们的数据 (Musical_Instruments + sentence-t5-768d + 32d SASRec cf embedding from PCA):

| Epoch | train_loss | recon_loss | cf_loss | collision_rate |
|-------|-----------|-----------|---------|----------------|
| 0 | 1.7089 | 0.0064 | 168.85 | (init) |
| 1 | 1.7071 | 0.0049 | 168.84 | (init) |
| 50 | 1.7070 | 0.0048 | 168.84 | 0.9999 |
| 100 | 1.7070 | 0.0048 | 168.84 | (eval next at 150) |
| 125 | 1.7070 | 0.0048 | 168.84 | (no further) |

cf_loss 168.84 从 epoch 0 到 125 完全不变, gradient 不通, model output 固定.

collision_rate=0.9999 (epoch 50) — 几乎所有 item SID 冲突, RQ-VAE 没学到有意义的 codebook.

## 根因分析

LETTER cf_loss = `compute_contrastive_loss(query=text_encoder_out, semantic=cf_embedding)`.
- query: 768-d text encoder 输出
- semantic: 32-d SASRec embedding (PCA 投影自 128d)

维度不匹配 (768 vs 32) + 32d PCA 投影可能不与 768d 文本分布对得上. LETTER 原 paper 用 32d SASRec 配 4096d LLaMA-TD, 我们用 32d SASRec 配 768d sentence-T5 — 信息容量差距 5×, cf_loss 平衡点完全变了.

## 已尝试的修复

1. ✅ `n_jobs=10 → 1` (joblib 多进程错误)
2. ✅ `n_init=10 → 3, max_iter=10 → 5` (加速 vq_init)
3. ✅ KMeansConstrained 包安装

## 决策

- cf_loss 完全 stuck, 跑多少 epoch 都不会收敛
- 跑 125 epochs 20+ 分钟, 没意义 → 14:56 决定 kill
- 释放 GPU 给其它任务

## 替代产物

- 其他子任务产物 SID tensors 已就绪 (TIGER text + TIGER-SAS)
- LETTER repo upstream 本身可能有兼容性问题 (torch 1.13 设计)

## 未来修复方向 (供参考, 不在本任务执行)

1. 替换 cf_emb 输入: 用 sentence-t5 投影到 32d 而非 SASRec PCA
2. 增大 cf_loss 权重 (alpha=0.1 而非 0.01)
3. 调整 e_dim + layers 使 encoder 输出维度匹配 cf_emb
4. 或用 upstream 提供的 LLaMA-TD 嵌入 (4096-d) 以与原 paper 对齐

## 状态清理

- products/task73/letter_tokenizer/ 保留作为失败 run 产物 (best_collision_model.pth 阶段性 0.99 collision)
- logs/task73_letter_tokenizer.log 保留
- 不计入成功 baseline
