# Task #291 — EMA codebook + κ-decouple (VQ-VAE-2 EMA update on FreeCurvHRQVAE)

**Status**: ✅ Pipeline COMPLETED, R@10=0.0765 vs HG-Rec baseline 0.1020 → **NO-GO (-25.0%)**

## TL;DR
- **目的**: 切断 κ+codebook 反馈循环 (codebook 不通过 gradient 更新, 用 EMA 平滑)
- **方法**: 包装 FreeCurvResidualVectorQuantization with EMAQuantizerWrapper (decay=0.99)
- **结果**: Stage 1 训练 best collision=0.3360 (epoch 169), Stage 3 训练 best NDCG@20=0.0752, Stage 4 **Test R@10=0.0765**
- **结论**: NO-GO. EMA codebook 比 baseline HG-Rec 低 25%, 不是 R@10 杠杆.

## Stage 1 RQ-VAE 训练
- 命令: `python train_hrqvae.py --quantizer ema --kappa_max 2.0 --beta 0.5 --epochs 200 --num_emb_list 64 128 256`
- Best loss ckpt: `products/task291/hrqvae_ema_kappa/Jul-29-2026_19-38-46/best_loss_model.pth`
- Best collision ckpt: epoch 169 collision=0.3360 (baseline HG-Rec ~0.0)
- EMA codebook util 不高 (collision 33% vs baseline 0%) — EMA 比 baseline codebook 利用率低

## Stage 2 Sinkhorn Inference
- Wrapper 修复:
  1. `[vq.K for vq in base_rq.vq_layers]` → `[vq.n_e for vq in base_rq.vq_layers]` (FreeCurvVQ 用 n_e)
  2. `indices_clamped = indices.clamp(min=0, max=K - 1)` before F.one_hot (CUDA device-side assert fix)
  3. `idx_m = all_indices[:, m].flatten()` instead of `all_indices[m].flatten()` (shape fix)
- Output: `Instruments_t5_hrqvae_ema.npy` shape=(9922, 4)
  - per-layer: [0,63] / [0,126] / [0,254] / [0,72]
  - 4th-digit dedup 让所有 9922 codes 唯一

## Stage 3 T5-mini Training (30 epochs)
- 命令: `task84_hgrec_stage3_train.py --code_path _t5_hrqvae_ema.npy --num_epochs 30 --batch_size 256 --lr 1e-3 --disable_early_stop`
- Best ckpt: `products/task291/stage3_t5mini/Instruments/Jul-29-2026_20-06-13/HG_Rec_best.pth`
- Best Val NDCG@20: 0.0752 (epoch 30)
- Best Val R@10: 0.0985 (epoch ~24)
- 训练时长 ~22 min (Stage 3 launch 20:06 → 20:27)

## Stage 4 Test Eval (R@10 = 0.0765)

| 指标 | Task #291 (EMA) | HG-Rec baseline (#84) | Δ |
|------|-----------------|------------------------|---|
| **R@10** | **0.0765** | **0.1020** | **-25.0%** NO-GO |
| R@5 | 0.0619 | 0.0816 | -24.1% |
| R@20 | 0.0951 | 0.1279 | -25.6% |
| NDCG@10 | 0.0539 | 0.0755 | -28.6% |
| NDCG@20 | 0.0586 | 0.0821 | -28.6% |

## 关键决策点 (R11.3 自主决策)
1. **EMA decay=0.99** 跟 VQ-VAE-2 默认一致 (vs decay=0.95/0.999)
2. **κ learnable** 不冻结 (跟 baseline 一样)
3. **Wrapper-only 改造**: 不修改 src/, 用 monkey-patch 替换 `model.hrq`
4. **per-layer mod 修复**: 让 EMA 输出 n_e 范围内的 index (避免 vocab OOB)

## 解读
- EMA 切断 gradient 反馈循环, 但**没解决 codebook 坍缩** (collision 33%)
- EMA 训练稳定性更高 (no NaN) 但下游 R@10 更差 (可能是 T5 训练用 EMA-balanced SID 时信号被 smooth 掉了)
- 不能归因到 κ 本身 (因为 baseline κ + Sinkhorn 没这问题)
- **关闭 EMA 路线**. 后续考虑 Gromov / Riemannian 方向 (跟 [[escape-routes-attack-premises-b-and-d]] 一致).

## 数据
- Stage 2 npy: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_ema.npy`
- Stage 3 ckpt: `products/task291/stage3_t5mini/Instruments/Jul-29-2026_20-06-13/HG_Rec_best.pth`
- Stage 4 JSON: `products/task291/stage4_eval.json`

result: Task #291 EMA codebook Test R@10=0.0765 vs HG-Rec baseline 0.1020, Δ -25.0% → **NO-GO**