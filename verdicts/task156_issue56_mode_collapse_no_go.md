# Issue #56 (新方向B: 混合曲率乘积空间) — Mode Collapse NO-GO (2026-07-31)

## Issue 摘要

[新方向B] 混合曲率乘积空间替代单一可学习κ —— 对照 arXiv:2307.04514/ACE-HGNN.

Gate 0 实现: `FreeCurvVectorQuantizationMixedCurv(FreeCurvVectorQuantization)`, 用 α_l 参数化层内混合曲率距离:
```
d_mixed(x, y, α, κ) = α · d_hyp(x, y, κ) + (1-α) · d_euc(x, y)
```
α_l = sigmoid(α_l_raw), 期望网络学出最优 hyp/euc 比例 per layer.

## 关键发现 (Stage 1 + Stage 2)

### Stage 1 训练 (Task #156, GPU 2, 1000 epoch)
- Recipe: num_emb_list=[64,128,256], e_dim=32, κ=0.74, α_init=0.5, β=0.25, lr=1e-3
- ckpt epoch=860, loss=**0.000205** (近完美重建)
- **但 α_l 全程静止 = 0.5** (gradient ≈ 0), 既不向 hyp 也不向 euc 漂移

### Stage 2 推断 (Task #160)
- 生成的 SID 形状 (9922, 4), 但 **3-digit unique = 1/9922 (0.01%)**
- 所有 9922 items 映射到同一码字链: codeword 23 → 47 → 210
- 4-digit unique = 100% (dedup counter 区分), 但前 3 位完全坍缩
- Loss 0.0002 + collapse 矛盾 → encoder 把所有 input 编码到同一 latent 点

### Stage 3 训练 (Task #161) — 用户决定终止
- 在 GPU 0 启动后约 12 分钟, epoch ~14
- Best NDCG@20 仅 0.0736 (vs baseline HG-Rec 0.0821, -10%)
- 用户判定早期 mode collapse 信号已足够 NO-GO, 立即 KILL 进程

## 根因诊断

混合曲率距离公式在 RQ-VAE 训练中**几何梯度信号消失**:
1. α_l sigmoid 参数化使梯度流过 sigmoid 接近 saturation 时消失
2. κ=0.74 时 d_hyp 和 d_euc 数值差异小, α_l 无明确优化方向
3. Encoder 倾向 trivial solution (所有 z → 同一点), 重建 loss 仍低但语义丢失

这跟 task178/task180 Phase 0 mode collapse 现象同源 — κ-stereographic 几何梯度在 RQ-VAE 训练中系统性失效.

## 综合结论

**Issue #56 NO-GO**:
- ❌ Stage 1: α_l 静止 (几何参数未激活)
- ❌ Stage 2: 3-digit unique 0.01% (mode collapse)
- ❌ Stage 3 早期: NDCG@20 = 0.0736 (-10% vs baseline)
- 终止于 epoch 14/200 (owner 决策)

## 关联产物

- Stage 1 ckpt: `products/task156/ckpt/Instruments/best_loss_model.pth` (13.8 MB, epoch 860)
- Stage 2 SID: `products/task160/sid/Instruments_t5_hrqvae_issue56.npy` (shape (9922,4), 3-digit 唯一)
- Stage 3 启动: `scripts/task161_issue56_stage3_train.sh` (vocab_size=10380 覆盖 dedup counter)
- 模型代码: `HG-Rec/model/hrqvae_issue55_56.py` (FreeCurvVectorQuantizationMixedCurv)
- 数学 sanity: 5/5 PASS (sanity 本身正确, 训练动态失效)

## 后续方向建议 (供 R10 backlog)

混合曲率路径在 RQ-VAE 训练中失效, 但混合曲率概念本身在 inference 阶段可能仍有用. 后续可探索:
1. α_l **不在 Stage 1 训, 在 Stage 3 T5 推理时用** (避免 Stage 1 几何梯度消失)
2. α_l 作为**事后重加权** SID 距离, 不参与训练
3. 换 Per-Codeword κ (task220 路径), 已是 NO-GO 但记录在案

---
result: Issue #56 混合曲率乘积空间方向 NO-GO. Stage 1 α_l 静止, Stage 2 SID 3-digit unique 0.01% (mode collapse), Stage 3 早期终止. Issue closed.
