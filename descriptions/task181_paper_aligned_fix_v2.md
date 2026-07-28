# Task #181 — 100% 官方对齐 baseline (Phase 0.6)

> **任务目的**: 用户 2026-07-25 指令"100% 对齐官方配置". 还原 `0bcfd0f` commit 的 loss 公式 (commitment+β·codebook, 两者 Poincaré distance² on raw vectors, logmap0 映回切空间). 删除 Phase 0.5 引入的 MSE commitment + norm penalty.

> **完成日期**: 2026-07-25 (in progress)
> **状态**: 🟢 在跑 (Stage 1, epoch 75+, collision ~9.3%)

---

## 1. 背景

**用户 2026-07-25 两轮诊断结论**:

1. **第一轮**: task181 (50 epoch Phase 0 fix) 看似健康 (collision 5.39%, recon_loss 8.92 在降). 但 task178/task180 (200 epoch 同 recipe) 暴露:
   - collision_rate 85.24% / 81.70%
   - recon_loss 卡死在 1489.5494 (9 epoch 字面相等)
   - ‖x‖_E q95 = 1.0000 (边界饱和)
2. **第二轮**: 用户指出 task180 数字自相矛盾 (mean ‖x‖_E=0.899 + q95=1.0000 + λ=18822 → 是被 boundary clamp ceiling 拉高).
3. **根因**: Poincaré distance boundary saturation + β=0.5 + 200 epoch 长训 → 码字全推 boundary (‖x‖=1-1e-6) → encoder 卡死 → recon_loss 不降.
4. **Sinkhorn at Stage 2 是 band-aid**: 强制 balanced assignment 让坍缩 codebook 输出 9922/9922 unique SID, 掩盖底层几何坍缩.

**用户 2026-07-25 第二轮指令**: "马上启动这个新的 fix 版本".

**解读**: 用户接受"在 paper 对齐 + mode collapse 修复"双管齐下的新 fix 版本, 不再等更彻底的 Phase 0 重设计. R11.3 自主决策 = **paper 对齐 defaults (上一轮已修) + commitment MSE + norm 正则** 同步启训.

---

## 2. 实验设计 (vs task178 + task84)

**变量**: HVectorQuantization.forward 的 loss 公式 + paper 对齐 defaults
**保持不变**:
- HG-Rec/model/hrqvae.py HRQVAE 主结构
- 数据集: Musical_Instruments (`HG-Rec/dataset/Instruments/`, 9922 items)
- Stage 3 T5 架构 (除 num_layers 4 vs 6)
- seed=42 (跟 task84 一致)
- Stage 2 Sinkhorn (本次 sk_epsilons=[0,0,0], 走 argmin)

**本次 fix 内容** (`HG-Rec/model/utils.py` HVectorQuantization.forward line 337-356):

```python
# Phase 0.6 (Task #181, 2026-07-25): 100% 对齐官方 (0bcfd0f commit) —
# 还原官方 loss = Poincaré distance² on raw Euclidean vectors +
# logmap0 映回切空间 + β on codebook. 删除 Phase 0.5 的 MSE commitment
# 和 norm penalty (它们在双曲空间里几何不一致).
x_q = codebook.index_select(0, indices)

commitment_loss = torch.mean(
    poincare_distance(x_q.detach(), latent, self.c) ** 2
)
codebook_loss = torch.mean(
    poincare_distance(x_q, latent.detach(), self.c) ** 2
)

loss = commitment_loss + self.beta * codebook_loss

x_q = logmap0(x_q, self.c)
latent = logmap0(latent, self.c)
x_q = x + (x_q - x).detach()
```

**启动命令** (Stage 1 RQ-VAE 训练, GPU 1 空闲):

```bash
bash scripts/task181_paper_aligned_fix_stage1_train.sh
```

完整 launcher 内容见 §6. 关键配置:
- `--epochs 1000` (paper Table 6)
- `--batch_size 1024` (官方 commit `0bcfd0f` 默认)
- `--num_emb_list 64 128 256` (paper Table 6)
- `--beta 1.0` (官方 commit `0bcfd0f` 默认)
- `--loss_type poincare` (paper Eq (8))
- `--sk_epsilons 0.0 0.0 0.0` (paper 走 argmin, Sinkhorn OFF)
- `--lr 1e-3 --learner AdamW --weight_decay 0` (paper Table 6)
- `--warmup_epochs 20` (上游开源版默认)

---

## 3. 决策触发 (vs task84 baseline R@10=0.1020)

| Stage 1 collision_rate | Stage 1 recon_loss 趋势 | Stage 3 R@10 | 决策 |
|------------------------|------------------------|---------------|------|
| **≤ 30%** (健康) | 单调下降到 ≤ 1.0 | **≥ 0.1058** (≥ phonism) | ✅ GO — paper recipe 真实有效, 进入后续 κ-Stereo 重做评估 |
| ≤ 30% | 单调下降到 ≤ 1.0 | 0.0900 - 0.1058 | 🟡 PARTIAL — paper recipe 改善 baseline 但未超 phonism, κ-Stereo NO-GO 结论仍大概率成立 |
| ≤ 30% | 单调下降到 ≤ 1.0 | < 0.0900 | 🔴 DEGRADE — paper recipe 比 task84 (broken baseline) 更差, 说明 original task84 有特殊过拟合路径 |
| **> 30%** (mode collapse) | 卡死不变 | 任意 | ⛔ FAIL — 官方 loss 在 9922 Musical_Instruments 上仍塌, 说明官方 recipe 不适用于此数据集规模 |

**辅助验证** (在每 100 epoch 末抽样):
- `log_hyperbolic_norm_stats()` 输出 ‖x‖_E mean / q95 / conformal_mean
- 期望: ‖x‖_E q95 ≤ 0.6 (即 c‖x‖² q95 ≤ 0.36, 离 boundary 还有 64% 余量)
- 期望: conformal_mean ∈ [2.0, 5.0] (κ=1 实际起作用, 而不是 saturate 到 200000)

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 RQ-VAE 1000 epoch (batch 1024) | ~1 h |
| Stage 2 Codebook inference (argmin + 4th-digit dedup) | ~5 min |
| Stage 3 T5-small num_layers=4 (跟 task84 同 recipe, GR epoch 200) | ~1.5-3 h |
| Stage 4 Test Eval | ~5 min |
| 总计 | ~5-7 h |

(注: Stage 3 200 epoch + early_stop=20 实际命中 epoch 估计 50-100, 比 Stage 1 短)

---

## 5. 风险与缓解

| 风险 | 缓解 |
|------|------|
| norm_penalty=0.05 系数可能过大, 压垮主 loss | 启动后 50 epoch 观察 `loss` 总值 vs `codebook_loss` + `commitment_loss` 之和; 若 `norm_penalty` 占比 > 30%, 减半到 0.025 |
| commitment MSE 让 T5 学到的 SID 序列跟原 paper 不一致 (paper Eq (8) 是双端 Poincaré) | Stage 3 R@10 跟 task84 (0.1020) + paper Table 7 (0.1315) 比对, 不强求一致 |
| 1000 epoch 训练时间超出预算, 用户决策要 kill | R12 force-save: 每 epoch 末保存 best_loss ckpt, kill 后可恢复 |
| GPU 1 被 #180 Stage 3 抢占 | R7 检查: 启训前再 nvidia-smi, GPU 1 必须 0% util 0 MB |
| F.mse_loss 没 import | 在 utils.py 顶部加 `import torch.nn.functional as F` |

---

## 6. 启动命令模板

### Stage 1 — RQ-VAE 训练

`scripts/task181_paper_aligned_fix_stage1_train.sh`:

```bash
#!/bin/bash
set -e
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task181

python3 train_hrqvae.py \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --beta 1.0 \
    --loss_type poincare \
    --sk_epsilons 0.0 0.0 0.0 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --weight_decay 0 \
    --warmup_epochs 20 \
    --lr_scheduler_type linear \
    --num_workers 4 \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2 \
    --device cuda:1 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/logs/task181/stage1_train.out
```

### Stage 2 — Codebook inference (待 Stage 1 完成后)

```bash
python3 scripts/task181_paper_aligned_fix_stage2_codebook.py \
    --ckpt /home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2/<TS>/HRQVAE_best.pth \
    --output /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_paper_fix.npy \
    --sk_epsilons 0.0 0.0 0.0
```

### Stage 3 — T5-small (num_layers=4)

```bash
python3 scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_paper_fix.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 4 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:1 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_paper_fix/<TS> \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task181 \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96
```

### Stage 4 — Test Eval

复用 Task #174 v3 config dict pattern:
```bash
python3 scripts/task174_stage4_eval.py \
    --config_dict '{...paper_fix specific...}' \
    --ckpt_path /home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_paper_fix/<TS>/HG_Rec_best.pth
```

---

## 7. 完成度跟踪

- [x] Phase 0.6 — 100% 官方对齐 (还原 0bcfd0f loss + β=1.0 + logmap0)
- [ ] py_compile utils.py + train_hrqvae.py + train_HG-Rec.py
- [ ] Stage 1 RQ-VAE 训练 (1000 epoch, GPU 1, batch_size 256)
- [ ] Stage 1 ckpt 健康度验证 (collision ≤ 30%, recon_loss 单调下降)
- [ ] Stage 2 codebook inference (argmin + 4th-digit dedup)
- [ ] Stage 3 T5-small num_layers=4 训练
- [ ] Stage 4 Test Eval (R@10/NDCG)
- [ ] Verdict 撰写 + §16 cleanup
- [ ] 更新 CLAUDE.md baseline (若 R@10 > 0.1020, 替换数字)