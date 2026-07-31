# Task #402 / task401 ckpt 下游 Stage 2+3+4 验证训练时长杠杆下游效应

**日期**: 2026-07-31
**触发**: task401 GO 收口 (Issue #108 Gate 1 PASS, L0/L1/L2=100/100/100 @ epoch 30 early stop), 验证"训练时长 ≤ 30 epoch 是真杠杆"是否在 Stage 3/4 也成立
**对照**: task396b (Issue #99 Stage 1 ckpt + HG-Rec baseline 200 epoch Stage 1) Stage 3/4

---

## 任务结构

### Stage 2 Sinkhorn (从 task401 Stage 1 ckpt 生成 4-digit SID)
- 输入: `products/task401_issue108_best_epoch_verify/ckpt/issue108_ckpt.pt`
- 输出: `HG-Rec/dataset/Instruments/_t5_rqvae_task402.npy` (9922 × 4)
- GPU: 0 (轻量, 用 task396b 闲置时刻)

### Stage 3 T5-mini 200 epoch
- 输入: `_t5_rqvae_task402.npy` (task401 30 epoch ckpt 派生的 SID)
- 训练: 跟 task84 baseline 同 (HG-Rec Stage 3 recipe, num_layers=6, num_decoder_layers=4, d_model=128, codebook_size=64/128/256/1, num_epochs=200)
- GPU: 1 (R7 空闲)
- 估计: ~90 min

### Stage 4 R@K eval
- 输入: task402 Stage 3 best ckpt
- 评估: R@5/10/20, NDCG@5/10/20
- GPU: 1 (跟 Stage 3 串行)
- 估计: ~30 min

---

## 决策标准

- **GO**: R@10 > 0.1020 (HG-Rec baseline)  OR  R@10 > task396b (如果 task396b 已完成)
- **NO-GO**: R@10 ≤ 0.1020

---

## 关键产物

- SID: `HG-Rec/dataset/Instruments/_t5_rqvae_task402.npy`
- Stage 3 ckpt: `products/task402_stage3_t5_train/ckpt/`
- Stage 4 metrics: `products/task402_stage4_rk_eval/metrics.json`
- verdict: `verdicts/task402_task401_ckpt_downstream_v2.md`

---

## 后续 (per R22 + R19)

1. task402 Stage 2 Sinkhorn (~1 min, GPU 0)
2. task402 Stage 3 T5-mini (~90 min, GPU 1)
3. task402 Stage 4 R@K eval (~30 min, GPU 1)
4. 跨方向联立:
   - 若 task402 R@10 > 0.1020 → 训练时长杠杆下游也成立, 28 方向 + 1 实证
   - 若 task402 R@10 ≤ 0.1020 → 训练时长杠杆仅 Stage 1 真, 下游 T5 仍需要 HG-Rec c=1.0 baseline recipe
5. Issue #104 / #105 闭环 (等 Stage 4 结果)