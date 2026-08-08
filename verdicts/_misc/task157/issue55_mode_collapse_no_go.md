---
task: 157
type: verdict
issue: 55
status: "NO-GO"
created: 2026-08-02
tags:
  - direction-c
up: "[[index]]"
---
# Issue #55 (新方向A: 曲率感知优化器) — Mode Collapse NO-GO (2026-07-31)

## Issue 摘要

[新方向A] 曲率感知优化器(Curvature-Aware Optimizer)修复几何不一致 —— 对照 arXiv:2405.13979.

Gate 0 实现: `RiemannianAdamW` + `FreeCurvVectorQuantizationMixedCurvWithScale`, 包含:
1. **RiemannianAdamW**: AdamW + Riemannian 梯度投影 `(1/(1+‖x‖²)²)` + Poincaré retraction
2. **α_l**: 混合曲率比例 (跟 Issue #56 共享)
3. **scale_l**: per-layer scale 参数, 每次 κ 更新后重新校准码字范数

## 关键发现 (Stage 1 + Stage 2)

### Stage 1 训练 (Task #157, GPU 3, 1000 epoch)
- Recipe: num_emb_list=[64,128,256], e_dim=32, κ=1.0, α_init=0.5, scale_init=1.0, β=0.25, lr=1e-3
- ckpt epoch=850, loss=**0.000206** (近完美重建)
- **但 α_l 全程静止 = 0.5** (gradient ≈ 0)
- **scale_l 全程静止 = 1.0** (gradient ≈ 0)

### Stage 2 推断 (Task #162)
- 生成的 SID 形状 (9922, 4), 但 **3-digit unique = 1/9922 (0.01%)**
- 所有 9922 items 映射到同一码字链: codeword 23 → 47 → 215
- 4-digit unique = 100% (dedup counter 区分)

### Stage 3 训练 (Task #163) — 用户决定终止
- 在 GPU 3 启动后约 12 分钟, epoch ~14
- Best NDCG@20 仅 0.0750 (vs baseline HG-Rec 0.0821, -8.6%)
- 用户判定早期 mode collapse 信号已足够 NO-GO, 立即 KILL 进程

## 根因诊断

双重几何参数 (α_l + scale_l) 在 RQ-VAE 训练中**全部静止**:
1. **α_l**: 同 Issue #56, 几何梯度消失
2. **scale_l**: Riemannian 梯度投影 + Poincaré retraction 在 norm ≈ 1 的边界附近 saturate
3. **RiemannianAdamW**: 投影公式 `(1/(1+‖x‖²)²)` 在 ‖x‖ ≪ 1 时接近 1 (无信号), 在 ‖x‖ → 1 时接近 0 (梯度消失)

这跟 task178/task180 模式坍缩同源, 但增加了 Riemannian 优化器组件后**没有改善**, 反而加剧.

## 综合结论

**Issue #55 NO-GO**:
- ❌ Stage 1: α_l=0.5 静止, scale_l=1.0 静止 (双参数都失效)
- ❌ Stage 2: 3-digit unique 0.01% (mode collapse, 跟 #56 同)
- ❌ Stage 3 早期: NDCG@20 = 0.0750 (-8.6% vs baseline)
- 终止于 epoch 14/200 (owner 决策)

## 关联产物

- Stage 1 ckpt: `products/task157/ckpt/Instruments/best_loss_model.pth` (13.8 MB, epoch 850)
- Stage 2 SID: `products/task162/sid/Instruments_t5_hrqvae_issue55.npy` (shape (9922,4), 3-digit 唯一)
- Stage 3 启动: `scripts/task163_issue55_stage3_train.sh` (vocab_size=10380 覆盖 dedup counter)
- 模型代码: `HG-Rec/model/hrqvae_issue55_56.py` (RiemannianAdamW + FreeCurvVectorQuantizationMixedCurvWithScale)
- 数学 sanity: 5/5 PASS (Riemannian 投影 + 缩放公式本身正确, 训练动态失效)

## 后续方向建议 (供 R10 backlog)

Riemannian 优化器在 RQ-VAE 上无效. 后续可探索:
1. 在更大 κ (例如 κ=10) 配合 **warm-start 范数** (‖x‖ > 0.5) 测试 RiemannianAdamW 是否激活
2. 改用 **Lorentz manifold** + 配套 Riemannian SGD (对比 Poincaré ball + retraction)
3. 放弃优化器层修复, 直接改 Stage 1 损失函数 (例如加 κ-aware commitment loss)

---
result: Issue #55 曲率感知优化器方向 NO-GO. Stage 1 双参数静止, Stage 2 SID 3-digit unique 0.01% (mode collapse), Stage 3 早期终止. Issue closed.
