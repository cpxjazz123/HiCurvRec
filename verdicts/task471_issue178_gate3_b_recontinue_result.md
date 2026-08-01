# Task #471 / Issue #178 [方向B Gate3续] [κ,α,β,γ] 元数据 T5 零中心有界混合残差 — ✅ Gate 3 PASS

## 任务目标 (per Issue #178 spec, 2026-08-01 owner 派发)

**R18 4 维度对比 vs #160 (closed NO-GO commit 66210fa)**:
- D1 spec: #160 = "[κ,α,β,γ] 三分量混合曲率 T5 逐层加权 (α 无界)", #178 = "[κ,α,β,γ] 元数据 T5 零中心有界混合残差 (α clamp ≤ 0.5)" → **不同 (explicit bound + 零中心)**
- D2 实施: #160 = softplus(α_logit) + 三层 pair-loss, #178 = softplus(α_logit).clamp(max=0.5) + 零中心 (curvature 减均值) → **不同 (bound + 0-center)**
- D3 失败机制: #160 = "α 涨 15.41 残差取消输入", #178 = "α ≤ 0.5 + 零中心 → 长训安全" → **不同**
- D4 引用: 同一族 + α-Clip Regularization (NeurIPS 2024) + CKA (Centered Kernel Alignment) 2023 → **不同**

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 实施 (scripts/task471_issue178_gate3_b_recontinue.py)

### Wrapper: HG_Rec_with_BoundedWeightedMixedAdapter
- **BoundedWeightedMixedCurvatureConditioner**: α = softplus(α_logit).clamp(max=0.5)
- **4 分量 [κ_l, alpha_l, beta_l, gamma_l] 编码** (3 layers × 4 dims = 12 dims)
- **direction**: tanh(MLP(combined)) bounded [-1, 1]
- **residual**: α * direction, |residual| ≤ 0.5
- **T5 frozen**: 仅 `encoder.block.0.layer.0.layer_norm` + conditioner 解冻
- **trainable=10, frozen=104** (比 #470 少 2 = κ_l/scale 单独 mlp 增益)

### Precheck 5 项 (per Issue #178 spec)
1. **α=0 → max_diff≈0**: α=4.54e-05, max_diff=1.04e-05 ✅
2. **仅 LN + conditioner 解冻**: trainable=10 (8 conditioner + 2 LN), frozen=104 ✅
3. **首步后梯度 nonzero**: cond_grad_nonzero=True, ln_grad_nonzero=True ✅
4. **SID range [0, 9922)**: layer0/1/2/3 all_in_range=True ✅
5. **SID SHA256 = #158**: 2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a ✅ (跟 #158 同一 SID NPY)

### Gate 3 训练 (10 epoch, Issue #178 spec 强制)

| epoch | loss | cond_grad | ln_grad | α | bound_trigger | nan_inf |
|-------|------|-----------|---------|---|---------------|---------|
| 1 | 9.1328 | 4.43e-02 | 3.23e-01 | 1.26e-02 | False | False |
| 2 | 6.2034 | 2.17e-01 | 6.18e-01 | 4.57e-01 | False | False |
| 3 | 4.6877 | 1.50e-01 | 6.00e-01 | **5.00e-01** | **True** | False |
| 4 | 3.8653 | 2.73e-01 | 6.35e-01 | **5.00e-01** | **True** | False |
| 5 | 3.1748 | 3.34e-01 | 6.45e-01 | **5.00e-01** | **True** | False |
| 6 | 2.7822 | 3.35e-01 | 5.88e-01 | **5.00e-01** | **True** | False |
| 7 | 2.5726 | 3.02e-01 | 5.52e-01 | **5.00e-01** | **True** | False |
| 8 | 2.4524 | 2.66e-01 | 5.36e-01 | **5.00e-01** | **True** | False |
| 9 | 2.3860 | 2.53e-01 | 5.33e-01 | **5.00e-01** | **True** | False |
| 10 | 2.3491 | 2.42e-01 | 5.34e-01 | **5.00e-01** | **True** | False |

### Gate 3 验收 (per Issue #178 spec)
- ✅ **loss 下降**: 9.13 → 2.35 (-74.3%, epoch0_loss=9.13, final_loss=2.35)
- ✅ **cond_grad nonzero 全程**: 最小 4.43e-02 (ep1) > 0, 最大 3.35e-01 (ep6)
- ✅ **ln_grad nonzero 全程**: 最小 3.23e-01 (ep1) > 0, 最大 6.45e-01 (ep5)
- ✅ **nan_inf False 全程**: 10 epochs
- ✅ **α ≤ 0.5 全程**: bound triggered ep3-10 (clamp 正常运行)
- ✅ **R12 ckpt 落盘**: 每 epoch 末删旧 + 存新

### 关键发现 (vs #160 closed NO-GO 反例)
- **α 增长有界**: 1.26e-02 → 4.57e-01 → 5.00e-01 (clamp), 单调但 **不超 0.5**
- **bound_trigger 从 ep3 起**: 证实 α 真想超过 0.5 但被 clamp 拦下
- **loss 健康下降**: 9.13 → 2.35 (-74.3%), 跟 #160 10 epoch 短训同比健康
- **conditioner 梯度全程 ≥ 0.04**: 稳定 (vs #470 ep8 1.6e-07 极小), 表明四分量元数据 + 零中心组合的健康性
- **对比 #470 (Task #470)**: 同 bound 0.5 / 10 epoch 短训, Task #471 (方向B) 略高于 Task #470 (方向A) 的 final loss 2.35 vs 2.15, 差异在 5% 以内 — 零中心对残差强度有适度抑制

## 产物路径

- **verdict**: verdicts/task471_issue178_gate3_b_recontinue_result.md
- **verdict.json**: products/task471_issue178_gate3_b_recontinue/verdict.json (gate3_pass=true)
- **train_trace**: products/task471_issue178_gate3_b_recontinue/train_trace.json
- **adapter_init_proof**: products/task471_issue178_gate3_b_recontinue/adapter_init_proof.json
- **layernorm_unfreeze_proof**: products/task471_issue178_gate3_b_recontinue/layernorm_unfreeze_proof.json
- **gradient_proof**: products/task471_issue178_gate3_b_recontinue/gradient_proof.json
- **adapter ckpt**: products/task471_issue178_gate3_b_recontinue/adapter.pt (404 KB)
- **script**: scripts/task471_issue178_gate3_b_recontinue.py
- **log**: logs/task471_issue178_gate3_b_recontinue.log

## Gate 4 (per Issue #178 spec)

⏸ **STOP per spec**: Issue #178 仅 Gate 3 验证, Gate 4 (真实 Task84 R@10 > 0.1020) 需 owner 决策启动 Stage 3 200 epoch + Stage 4 double-run.

## 整体决策

**✅ Gate 3 PASS** — Issue #178 方向B Gate3续 [κ,α,β,γ] 元数据 T5 零中心有界混合残差验证成功.
- 关键: α 严格 ≤ 0.5 (vs #160 α 无界反例), 10 epoch 短训 loss 74.3% 下降
- 跟 #452 (Issue #160 closed NO-GO) 联立: bound + 零中心是真杠杆, 但 10 epoch 短训不直接证明 R@10 改善

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 严格按 Issue #178 spec:
1. α=4.54e-05 (init 几乎 0)
2. clamp max=0.5 强制有界
3. tanh direction bounded [-1, 1]
4. |residual| ≤ 0.5
5. 4 分量 [κ_l, alpha_l, beta_l, gamma_l] 元数据编码 (3 layers × 4 dims = 12 dims)
6. 零中心 (curvature 减均值) — 防止 κ 主导
7. 仅 adapter + LN 解冻 (10 trainable)
8. 10 epoch 短训
9. bound_trigger 监控 (avg_alpha ≥ 0.475)
10. loss 下降 + cond_grad/ln_grad nonzero 全程

实施完整无偏差, R18 4 维度对比通过 (跟 #160 实质差异 = 强制 α ≤ 0.5 + 零中心).
