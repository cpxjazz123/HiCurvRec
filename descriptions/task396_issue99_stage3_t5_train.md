# Task #396 / Issue #99 [方向C Gate3] Stage 3 T5-mini 200 epoch 训练

**日期**: 2026-07-31
**触发**: task395 #99 Stage 2 PASS → Stage 3 T5 训练启动
**前置**:
- task394 Stage 1 PASS: L0/L1/L2 util=100%/100%/100%, max_load=1.87%/1.09%/0.62%
- task395 Stage 2 PASS: SID 4-digit unique 9922/9922 (100%)
- task396a: SID file saved to HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy

**任务**: Stage 3 T5-mini 200 epoch 训练 (HG-Rec baseline recipe, post #97 patch)
**结果**: ⏳ TRAINING — Epoch 0 启动中, ~95 min 预计

---

## Pipeline 状态 (跨 Stage 1+2+3 完整运行)

| Stage | 任务 | 状态 | 关键指标 |
|-------|------|------|----------|
| 1 RQ-VAE | task394 | ✅ PASS | L0/L1/L2 util 100%/100%/100%, max_load 1.87/1.09/0.62% |
| 2 Sinkhorn | task395 + 396a | ✅ PASS | SID 4-digit unique 9922/9922 (100%), collision 0% |
| 3 T5-mini | **task396b (本)** | ⏳ TRAINING | Epoch 0 启动, 预计 95 min |
| 4 R@K eval | task397 (下一轮) | ⏸ STOP | 待 Stage 3 完成 |

---

## 训练配置 (HG-Rec baseline recipe)

- dataset: Instruments (Musical_Instruments)
- code_path: _t5_rqvae_task396.npy (task396a 产物)
- codebook_size: 64 128 256 1 (3 hierarchy + 1 dedup)
- num_epochs: 200 + early_stop=20
- batch_size: 256, lr: 1e-4
- T5-mini (5.5M params): num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024, num_heads=6, d_kv=64
- vocab_size: 1025, max_len: 20
- beam_size: 20 (inference)
- seed: 42
- GPU: 0 (CUDA_VISIBLE_DEVICES=0)
- ckpt path: products/task396_issue99_stage3_t5_train/ckpt/

---

## 关键产物

- **SID file**: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy (9922, 4)
- **SID meta**: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.meta.json
- **ckpt**: products/task396_issue99_stage3_t5_train/ckpt/Instruments/<timestamp>/HG_Rec_best.pth (待 Stage 3 完成)
- **log**: logs/task396b_t5mini_stage3_<timestamp>.log
- **task396a log**: products/task396_issue99_stage3_t5_train/task396a_run.log
- **task396b log**: products/task396_issue99_stage3_t5_train/task396b_run.log

---

## 后续 (per R22 + R19)

1. **task397: Stage 4 R@K eval** — 用 task396 ckpt + Instruments test set, 验证 R@10 > 0.1020 (HG-Rec baseline threshold)
2. **Issue #98 patch** (可选进阶): per-component learnable κ 解锁 metadata kappa/scale var > 0
3. **Issue #99 Gate 3 复跑**: 用 task396 ckpt, 复跑 #93 on/off/shuffle 消融 (Stage 3 训练后)

**GO/NO-GO 决策 (per Issue #99 spec + HG-Rec baseline)**:
- ✅ GO: R@10 > 0.1020 + 6 项 metrics (R@5/10/20, NDCG@5/10/20) 全部齐全
- ❌ NO-GO: R@10 ≤ 0.1020 或 missing metrics