# Task #292 — Restoration (EMA + dead code revival) + κ-decouple

**Status**: ✅ Pipeline COMPLETED, R@10=0.0799 vs HG-Rec baseline 0.1020 → **NO-GO (-21.7%)**

## TL;DR
- **目的**: EMA + 自适应 dead code revival (Yu et al. 2022 Restoration)
- **方法**: 包装 FreeCurvResidualVectorQuantization with RestorationQuantizerWrapper (decay=0.99, revive_threshold=1, revive_ratio=0.1, revive_freq_epochs=5)
- **结果**: Stage 1 best collision=0.3234 (epoch 164), Stage 3 best NDCG@20=0.0745, Stage 4 **Test R@10=0.0799**
- **结论**: NO-GO. Restoration 比 baseline HG-Rec 低 22%, **比 EMA (Task #291) 略好 +4.5%**, 但仍 NO-GO.

## Stage 1 RQ-VAE 训练
- 命令: `python train_hrqvae.py --quantizer restoration --kappa_max 2.0 --beta 0.5 --epochs 200`
- Best loss ckpt: `products/task292/hrqvae_restoration_kappa/Jul-29-2026_19-38-46/best_loss_model.pth`
- Best collision ckpt: epoch 164 collision=0.3234 (略好于 Task #291 EMA 0.3360)
- Revival 没明显降低 collision (revive_threshold=1 触发但 util 仍 ~67%)

## Stage 2 Sinkhorn Inference
- 复用 Task #291 EMA wrapper 修复
- Output: `Instruments_t5_hrqvae_restoration.npy` shape=(9922, 4)
  - per-layer: [0,63] / [0,126] / [0,255] / [0,99]
  - 4th-digit dedup 让所有 9922 codes 唯一

## Stage 3 T5-mini Training (30 epochs)
- Best ckpt: `products/task292/stage3_t5mini/Instruments/Jul-29-2026_20-06-15/HG_Rec_best.pth`
- Best Val NDCG@20: 0.0745 (epoch 30)
- Best Val R@10: 0.0975
- 训练时长 ~22 min

## Stage 4 Test Eval (R@10 = 0.0799)

| 指标 | Task #292 (Restoration) | HG-Rec baseline (#84) | Δ |
|------|--------------------------|------------------------|---|
| **R@10** | **0.0799** | **0.1020** | **-21.7%** NO-GO |
| R@5 | 0.0616 | 0.0816 | -24.5% |
| R@20 | 0.0998 | 0.1279 | -22.0% |
| NDCG@10 | 0.0544 | 0.0755 | -28.0% |
| NDCG@20 | 0.0594 | 0.0821 | -27.6% |

## 关键决策点 (R11.3 自主决策)
1. **revive_threshold=1** (跟 Yu et al. 2022 默认): cluster_size<1 → 视为 dead, 替换
2. **revive_ratio=0.1**: 每次复活 10% 的 dead codes
3. **revive_freq_epochs=5**: 每 5 epoch 检查一次 (避免频繁破坏学习)
4. **EMA decay=0.99** (跟 EMA wrapper 一致)

## 解读
- **Restoration 略好于纯 EMA**: +4.5% R@10. dead code revival 有微弱帮助, 但杯水车薪
- **跟 [[free-curv-codebook-collapse]] 一致**: κ+codebook 坍缩是架构根本问题, EMA/Restoration 都解不了
- **Sinkhorn + 4th-digit dedup** Stage 2 解决了 collision 让 SID 唯一, 但**信息已经丢失** (用 ~67% codes), T5 训练基础比 baseline 100% codes 差
- **关闭 Restoration 路线**. 后续 Gromov / Riemannian / Per-Codeword κ 路线 (跟 [[escape-routes-attack-premises-b-and-d]] 一致)

## 数据
- Stage 2 npy: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_restoration.npy`
- Stage 3 ckpt: `products/task292/stage3_t5mini/Instruments/Jul-29-2026_20-06-15/HG_Rec_best.pth`
- Stage 4 JSON: `products/task292/stage4_eval.json`

result: Task #292 Restoration Test R@10=0.0799 vs HG-Rec baseline 0.1020, Δ -21.7% → **NO-GO** (但比 EMA 略好 +4.5%)