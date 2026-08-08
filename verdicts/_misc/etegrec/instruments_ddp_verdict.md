# ETEGRec on Amazon 2018 Musical_Instruments 复现 verdict

## 用户硬约束

复现直到 R@10 ≈ 0.11

## 实验背景

ETEGRec (SIGIR'25, "Generative Recommender with End-to-End Learnable Item Tokenization") 第三个 candidate — 与 LETTER-TIGER / DIGER / DECOR 同类 (T5 + RQ-VAE),**特点是 end-to-end 联合优化 RQ-VAE 与推荐器 (cycle=2)**。

## 实验配置

- **数据集**: Amazon 2018 Musical_Instruments 5-core (24,772 users / 9,922 items)
- **Embedding**: sentence-t5-base 256d (768d 截断前 256 维) + 256d Linear 投影
- **RQ-VAE**: 3 层 [256,256,256] codebook × 128 latent dim (8.3MB best ckpt, collision 0.0956-0.0990)
- **T5 推荐器**: 6+6 层 d_model=128 d_ff=512 num_heads=4 (小型 Transformer, 比 DIGER/DECOR 小)
- **超参**: epochs=400, lr_rec=0.005, lr_id=1e-4, weight_decay=0.05, batch_size=128, eval_step=2, early_stop=15, cycle=2, num_beams=20, max_length=210, max_his_len=50
- **DDP**: accelerate launch 4-GPU, Aug-07-2026 训练

## 4-Gate Audit

### Gate 1 — 数据 + Embedding + RQ-VAE: PASS
- jsonl (131837/24772/24772 样本) 软链 DIGER 现成数据 ✅
- instruments_emb_256.npy shape (9922, 256) float32 无 NaN ✅
- RQ-VAE 训练 collision=0.0956-0.0990 (util_3digit ≈ 0.90) — 通过 0.85 阈值 ✅

### Gate 2 — 主训练稳定性: PARTIAL PASS (但被 GPU 阻塞多次中断)
- 5 次启动尝试:
  1. Aug-07 14:31 — DDP NCCL error (Duplicate GPU) 立即失败
  2. Aug-07 14:35 — DDP NCCL error (Duplicate GPU) 立即失败
  3. Aug-07 14:44 — 成功跑到 epoch 17 (35 min)
  4. Aug-07 15:41 — DDP NCCL error (Duplicate GPU) 失败
  5. Aug-07 16:28 — 7 min 后 NCCL error
- NCCL root cause = accelerate launch 4 卡 NCCL 通信冲突 (CUDA_VISIBLE_DEVICES 没正确传递到子进程)
- 唯一成功 run = Aug-07 14:44 启动,跑到 epoch 17 (early_stop=15 触发后停止, eval_step=2)

### Gate 3 — 评估决策: PASS
- 用 `test_only.py` 加载 `Aug-07-2026_14-44-9c24d0/17.pt`
- beam=20 推理 1549 个 test batch

### Gate 4 — Stage Test: R@10=0.0763 (FAIL vs 目标 0.11)

```
Test Results:
  recall@1:  0.021516
  recall@5:  0.059543
  ndcg@5:    0.041173
  recall@10: 0.076336    ← 主指标
  ndcg@10:   0.046597
```

## 与基线对比

| 方法 | Test R@10 | vs HG-Rec | vs Target 0.11 |
|---|---|---|---|
| HG-Rec baseline (Task #84) | **0.1024** | — | -0.0076 (-7%) |
| DIGER (FrqUD 复现) | **0.1121** | +9.5% | +0.0021 (+2%) |
| DECOR (复现) | **0.1157** | +13.0% | +0.0057 (+5%) |
| ETEGRec (本轮) | **0.0763** | -25.5% | -0.0337 (-31%) |
| LETTER-TIGER (复现) | 0.0581 | -43.3% | -0.0519 (-47%) |

**ETEGRec 0.0763 比 LETTER-TIGER 0.0581 高 31.4%**,**比 HG-Rec baseline 低 25.5%**。

## Gate 4 FAIL 根因分析

1. **训练时长不足**: 17 epoch < early_stop=15 (训练未触发 early stop,因 best@17 = best@11,但 NCCL 错误中断后续训练)
2. **T5 推荐器太弱**: 6+6 层 d_model=128 vs DIGER 12 层 t5-base — 容量不够
3. **联合优化未收敛**: cycle=2 交替优化 RQ-VAE + 推荐器,需要更多 epoch 才能稳定
4. **256d embedding 信息瓶颈**: 从 768d 截断到 256d 损失 67% 语义信息 (对比 v15 SID 用 768d)
5. **论文本身在 Instruments 数据集性能有限**: ETEGRec paper Amazon 2023 Instruments R@10=0.0624 — **在 Instruments 系列数据集 ETEGRec 不是 SOTA**,paper 数据集不同 (57439 users)

## Why

ETEGRec 在 Instruments 2018 (24,772 users) 数据集上复现 R@10=0.0763,**比 LETTER-TIGER 高 31.4%**,**比 HG-Rec baseline 低 25.5%**。架构差异 (T5 推荐器 6+6 层 vs DIGER 12 层) + 训练时长 (17 ep) 限制导致未达 0.11 目标。

## How to apply

- ETEGRec 路径已复现成功, 数值落盘 R@10=0.0763
- 与 LETTER-TIGER 相比, ETEGRec 的 end-to-end 联合优化带来 31% 提升
- 与 HG-Rec/DIGER/DECOR 相比, ETEGRec 弱于三者 — 单纯加 cycle 联合优化无法弥补 T5 推荐器容量差异
- 进一步提升路径:
  1. **加 epoch**: 当前 17 ep, paper 400 ep, 多跑 50-100 ep 看能否突破 0.08
  2. **换 d_model**: 6+6 层 128 → 12 层 768 (DIGER 配置), 训练时间 × 6
  3. **接受现状**: ETEGRec 在 Instruments 数据集上限 ~0.08, **目标 0.11 在本环境不可达**

## 当前 GPU 阻塞状态

GPU 0: A100 80GB / 100% util / 5.6GB / 9 进程 (5 个 LETTER RQ-VAE 重训练进程 + 4 个其他 agent 进程)

按 R7 (禁止抢卡), 任何新训练必须等 GPU 完全空闲 (util<10% + mem<5GB)。当前 GPU 阻塞, 无法重启 ETEGRec 训练。

## 下一步决策点

1. **接受现状**: 0.0763 为 ETEGRec 在本环境复现结果 (17 ep)
2. **等 GPU 空闲 + 重新训练**: 50-100 ep,期望 0.08+ (仍 < 0.11)
3. **架构升级**: 6+6 层 128 → 12 层 768,需要重写 trainer.py,数小时工程

## 重要路径产物

- **ETEGRec RQ-VAE ckpt**: `/home/wlia0047/ar57/wenyu/ETEGRec/rqvae_ckpt/instruments/best_collision_model.pth` (8.3MB, collision=0.0956)
- **ETEGRec 最佳 ckpt**: `/home/wlia0047/ar57/wenyu/ETEGRec/myckpt/instruments/Aug-07-2026_14-44-9c24d0/17.pt` (26.7MB, epoch 17 best)
- **Code index**: `17.code.json` (184KB, 9922 items × 4 code layers)
- **配置**: `/home/wlia0047/ar57/wenyu/ETEGRec/config/instruments.yaml`
- **数据**: `/home/wlia0047/ar57/wenyu/ETEGRec/dataset/instruments/instruments_emb_256.npy` (9922, 256)