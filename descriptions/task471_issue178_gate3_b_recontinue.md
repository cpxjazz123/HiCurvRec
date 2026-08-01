# Task #471 / Issue #178 [方向B Gate3续] 加权混合曲率元数据 T5 零中心有界混合残差验证

## 任务摘要

Issue #178 owner 2026-08-01 派发方向B Gate3续: 重新实施 [κ_l, alpha_l, beta_l, gamma_l] 三分量加权混合曲率元数据 T5 零中心有界混合残差. 跟前 Task #452/#160 (#160 closed commit 66210fa) 的关键差异 = **α 必须有界 (clamp max=0.5)**, 防止 #160 中 α 从 0.12 → 15.41 单调暴涨的反例 (15.41 × MLP direction ≈ 残差取消输入)。方向一致 (T5 frozen + 仅 LN + conditioner 解冻 + 四分量元数据)，但必须用 bound 防止 α 失控.

R18 4 维度对比 vs #160 (closed commit 66210fa):
- **D1 spec**: #160 是 "[κ,alpha,beta,γ] 三分量 → 混合曲率元数据 T5 逐层加权几何适配 (α 无界 softplus)", #178 是 "[κ,alpha,beta,γ] 三分量 → 混合曲率元数据 T5 零中心有界混合残差 (α clamp ≤ 0.5)" → 不同 (explicit bound + 零中心)
- **D2 实施**: #160 是 softplus(α_logit) + 三层独立 pair-loss, #178 必须 softplus(α_logit).clamp(max=0.5) + 零中心 (curvature 编码减均值) + 添加 bound_trigger 监控 → 不同
- **D3 失败机制**: #160 是 "α 增长无界 → 200 epoch 长训时 α 达 e^50 量级混合残差取消输入", #178 是 "α 增长有界 ≤ 0.5 → 长训安全 + 零中心防止 residual 偏向 κ 主导" → 不同
- **D4 引用**: 同一族 + 新增 α-Clip Regularization (NeurIPS 2024) + Centered Kernel Alignment (CKA) 2023 → 不同

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/best_loss_model.pth` (sha256=59a38fa3...)
- **SID NPY**: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy` SHA256=2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a (跟 #157 + #158 一致, 同一 Stage 2 产出)
- **T5 ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth` (Task #84 frozen)
- **wrapper**: `HG_Rec_with_BoundedWeightedMixedAdapter`
  - BoundedWeightedMixedCurvatureConditioner: α = softplus(α_logit).clamp(max=0.5) 强制有界
  - 4 分量 [κ_l, alpha_l, beta_l, gamma_l] 编码 (3 layers × 4 dims = 12 dims) + 零中心 (减均值)
  - SID token projection + 3-layer MLP direction (tanh bounded)
  - T5 主干冻结, 仅 `encoder.block.0.layer.0.layer_norm` + conditioner 解冻
- **GPU**: GPU 2 (R7 满足, GPU 0 task450, GPU 1 task470, GPU 3 free)

## Gate 顺序 (per Issue #178 spec)

- **precheck**: 5 项 — ① α=0 → max_diff≈0; ② 仅 LN + conditioner 解冻; ③ 首步后梯度 nonzero; ④ SID range [0, 9922); ⑤ SID SHA256 = #158 ✅
- **Gate 3**: 10 epoch 短训, loss 下降 + cond_grad nonzero + ln_grad nonzero + α ≤ 0.5 全程 → **核心 Gate**
- **Gate 4**: 仅 Gate 3 PASS 后, 真实 Task84 R@10 > 0.1020 → ⏸ STOP (本任务仅 Gate 3)

## R23 强制

val_R@10=0 跨 ≥2 epoch / α 超 0.5 跨 ≥2 epoch / cond_grad=0 跨 ≥3 epoch / NaN/Inf → kill + NO-GO

## R12 强制

每 epoch 末强制存 ckpt (删旧 + 存新) + PID 文件

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 严格按 Issue #178 spec:
1. α_logit init = -10.0 (softplus(-10) ≈ 4.5e-5, 几乎为 0)
2. α = F.softplus(α_logit).clamp(max=0.5) 强制有界
3. direction = tanh(MLP(combined)) bounded [-1, 1]
4. residual = α * direction, 整体有界 |residual| ≤ 0.5
5. curvature 元数据 减均值 (零中心)— 防止 κ 主导
6. 仅解冻 adapter + encoder.block.0.layer.0.layer_norm
7. LR_conditioner = 1e-3, LR_LN = 1e-4
8. 10 epoch 短训, 监控 α 是否触发 bound (avg_alpha ≥ 0.475)
9. 三分量 [κ_l, alpha_l, beta_l, gamma_l] = [1, 0, 0, 0] (init 跟 #158 Gate 2 一致)
10. loss 下降 + cond_grad/ln_grad nonzero 全程 + nan_inf False

## 产物路径

- **verdict**: `verdicts/task471_issue178_gate3_b_recontinue_result.md`
- **verdict.json**: `products/task471_issue178_gate3_b_recontinue/verdict.json`
- **train_trace**: `products/task471_issue178_gate3_b_recontinue/train_trace.json`
- **adapter_init_proof**: `products/task471_issue178_gate3_b_recontinue/adapter_init_proof.json`
- **layernorm_unfreeze_proof**: `products/task471_issue178_gate3_b_recontinue/layernorm_unfreeze_proof.json`
- **gradient_proof**: `products/task471_issue178_gate3_b_recontinue/gradient_proof.json`
- **adapter ckpt**: `products/task471_issue178_gate3_b_recontinue/adapter.pt`
- **script**: `scripts/task471_issue178_gate3_b_recontinue.py`
- **log**: `logs/task471_issue178_gate3_b_recontinue.log`
