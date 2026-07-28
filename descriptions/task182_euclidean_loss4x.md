# Task #182 — 欧式 RQ-VAE + 量化 loss ×4

> **任务目的**: 在 100% 官方对齐 baseline (Phase 0.6) 基础上, 跑一个纯欧式版本, 但把量化 loss (codebook_loss) 乘 4. 验证欧式版本是否也能达到类似 R@10, 以及 loss 乘 4 对码字利用的影响.

> **完成日期**: 2026-07-25 (启动中)
> **状态**: 🟢 Stage 1 在跑

---

## 1. 实验设计

**变量** (vs task181 Phase 0.6):
1. `--euclidean_qloss` — 使用 MSE 替代 Poincaré distance² 作为量化 loss
2. `--loss_mult_codebook 4` — codebook_loss 乘以 4 (loss = commitment + β × 4 × codebook_loss)
3. `--loss_type mse` — 重建 loss 用 MSE 而非 Poincaré

**保持不变** (与 task181 一致):
- `--beta 1.0` (β on codebook, 官方 commit 0bcfd0f 默认)
- `--batch_size 1024` (官方默认)
- `--epochs 1000` (paper Table 6)
- `--num_emb_list 64 128 256` (paper Table 6)
- `--sk_epsilons 0.0 0.0 0.0` (argmin, Sinkhorn OFF)
- `--kmeans_init True --kmeans_iters 1000`
- `--lr 1e-3 --learner AdamW --weight_decay 0`
- `--layers 512 256 128 64 --e_dim 32`
- 数据集: Musical_Instruments (`HG-Rec/dataset/Instruments/`, 9922 items)

## 2. 决策触发

| 指标 | 期望 | vs HG-Rec baseline (0.1020) |
|------|------|-----------------------------|
| Stage 1 collision_rate | ≤ 15% (健康) | — |
| Stage 3 R@10 | ≥ 0.1020 | GO if ≥ baseline |
| Stage 3 R@10 | ≥ 0.1315 | 达到 paper 值 |
| Stage 3 R@10 | < 0.0900 | 欧式退化, NO-GO |

## 3. 流水线

### Stage 1 — RQ-VAE 训练

```bash
bash scripts/task182_stage1_train_euclidean.sh
```

### Stage 2 — Codebook inference (待 Stage 1 完成后)

输出: `dataset/Instruments/Instruments_t5_rqvae_euclidean_loss4x.npy`

### Stage 3 — T5-small (复用 task84 训练脚本)

```bash
python3 scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_euclidean_loss4x.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 --d_model 128 --d_ff 1024 \
    --num_heads 6 --d_kv 64 --vocab_size 1025 --max_len 20 \
    --device cuda:2 --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task182/t5small_euclidean_loss4x/ \
    --seed 42 --early_stop 20 --beam_size 20 --infer_size 96
```

## 4. 完成度跟踪

- [x] 框架修改 (utils.py / hrqvae.py / train_hrqvae.py): `euclidean_qloss` + `loss_mult_codebook`
- [x] py_compile 三文件
- [ ] Stage 1 RQ-VAE 训练 (1000 epoch, GPU ?)
- [ ] Stage 2 codebook inference
- [ ] Stage 3 T5-small 训练
- [ ] Stage 4 Test Eval
- [ ] Verdict
