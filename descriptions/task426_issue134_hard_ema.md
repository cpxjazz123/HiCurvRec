# Task #426 / Issue #134 [方向A Gate1] hard-count EMA 双曲码本更新防单码字坍缩

## 目标

Issue #131/#132/#127/#128 全部 NO-GO 共同根因 = 任何 soft weighted posterior (ball projection / per-item posterior / Sinkhorn transport) 都坍缩到单码字 (max_load=1.0)。Issue #134 提出新机制: forward SID 仍由稳定双曲 cost 的 hard argmin 产生; codebook 更新使用 **detach** 的 hard counts + geodesic/切空间 EMA, 不让单样本 soft posterior 反向拉向全局几何中心; 加 count regularizer 约束 hard counts 使用分布。

## 实施

- 脚本: `scripts/task426_issue134_hard_ema.py`
- HardEMAHRQVAE 类: 稳定双曲 cost + hard argmin SID + geodesic_ema_update (tangent centroid) + count_regularizer (negative entropy)
- STE: `z_q_st = z_e + (z_q_hard - z_e).detach()` (梯度走 encoder, 不走 codebook)
- EMA 在每个 batch 后用 `no_grad` 应用 (detached from encoder gradient)
- 5-step audit per Issue #134 §Gate1 1:
  1. hard argmin assignment 存在
  2. EMA/geodesic update 不参与 encoder 梯度偷渡
  3. count regularizer 对 hard counts 有效
  4. κ/scale/codebook 同步残差有限
  5. loss/κ grad/codebook norm 无 NaN/Inf

## 预期产物

- `products/task426_issue134_hard_ema/verdict.json`
- `verdicts/task426_issue134_gate1_*.md`
- 30 epoch 训练 + 5-step audit 全部落盘

## Gate 1 决策

- PASS: 30 epoch 内 (a) L0/L1/L2 util ≥90% (b) max_load <5% (c) 5-step audit PASS
- FAIL: USAGE-KILL @ any epoch (util/load 阈值不足) 或 audit FAIL

## 关联

- Issue #134: hard-count EMA + geodesic codebook update + count regularizer
- 联立 #131/#132 (Sinkhorn NO-GO) → #134 (hard assignment + EMA) 是直接对策
- R7: 启动前 nvidia-smi 选空闲 GPU (默认 cuda:0, 4×L40S 全空闲)
- R12: 训练 ckpt 强制保存到 products/task426/
- R17 + R20: commit + verdict 必须详细 4 Gate 回答