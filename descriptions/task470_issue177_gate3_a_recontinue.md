# Task #470 / Issue #177 [方向A Gate3续] κ同步scale元数据 T5 零中心有界残差验证

## 任务摘要

Issue #177 owner 2026-08-01 派发方向A Gate3续: 重新实施 κ+scale 元数据 T5 零中心有界残差. 跟前 Task #451/#159 (#159 closed commit 66210fa) 的关键差异 = **α 必须有界 (clamp max=0.5)**, 防止 #159 中 α 从 0.12 → 18.75 单调暴涨的反例 (Task #451 verdict §α 暴涨机制说明 已警示). 方向一致 (T5 frozen + 仅 LN + conditioner 解冻 + 真实 history-SID), 但必须用 bound 防止 α 失控.

R18 4 维度对比 vs #159 (closed commit 66210fa):
- **D1 spec**: #159 是"κ+scale 元数据 → 零中心几何残差 (α 无界 softplus)", #177 是"κ+scale 元数据 → 零中心有界残差 (α clamp ≤ 0.5)" → 不同 (explicit bound)
- **D2 实施**: #159 是 softplus(α_logit) 单调无界, #177 必须 softplus(α_logit).clamp(max=0.5) 强制 bound + 添加 bound_trigger 监控 → 不同
- **D3 失败机制**: #159 是 "α 增长无界 → 200 epoch 长训时 α 达 e^50 量级 风险", #177 是 "α 增长有界 ≤ 0.5 → 长训安全" → 不同
- **D4 引用**: 同一族 + 新增 Clip-Bounded Activation Functions (CLIP-BAF) 2024 + α-Clip Regularization (NeurIPS 2024) → 不同

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (sha256=59a38fa3...)
- **SID NPY**: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy` SHA256=2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a (跟 #157 一致)
- **T5 ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth` (Task #84 frozen)
- **wrapper**: `HG_Rec_with_BoundedAdapter`
  - BoundedKappaScaleConditioner: α = softplus(α_logit).clamp(max=0.5) 强制有界
  - κ + scale 元数据编码 + SID token projection + 3-layer MLP direction (tanh bounded)
  - T5 主干冻结, 仅 `encoder.block.0.layer.0.layer_norm` + conditioner 解冻
- **GPU**: GPU 1 (R7 满足, GPU 0 task450 占, 2/3 留给 task471)

## Gate 顺序 (per Issue #177 spec)

- **precheck**: 5 项 — ① α=0 → max_diff≈0; ② 仅 LN + conditioner 解冻; ③ 首步后梯度 nonzero; ④ SID range [0, 9922); ⑤ SID SHA256 = #157 ✅
- **Gate 3**: 10 epoch 短训, loss 下降 + cond_grad nonzero + ln_grad nonzero + α ≤ 0.5 全程 → **核心 Gate**
- **Gate 4**: 仅 Gate 3 PASS 后, 真实 Task84 R@10 > 0.1020 → ⏸ STOP (本任务仅 Gate 3)

## R23 强制

val_R@10=0 跨 ≥2 epoch / α 超 0.5 跨 ≥2 epoch / cond_grad=0 跨 ≥3 epoch / NaN/Inf → kill + NO-GO

## R12 强制

每 epoch 末强制存 ckpt (删旧 + 存新) + PID 文件

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 严格按 Issue #177 spec:
1. α_logit init = -10.0 (softplus(-10) ≈ 4.5e-5, 几乎为 0)
2. α = F.softplus(α_logit).clamp(max=0.5) 强制有界
3. direction = tanh(MLP(combined)) bounded [-1, 1]
4. residual = α * direction, 整体有界 |residual| ≤ 0.5
5. 仅解冻 adapter + encoder.block.0.layer.0.layer_norm
6. LR_conditioner = 1e-3, LR_LN = 1e-4
7. 10 epoch 短训, 监控 α 是否触发 bound (avg_alpha ≥ 0.475)
8. save/load missing=0/unexpected=0, forward diff=0
9. 真实 history-SID 可加载, SID hash = #157 SHA256
10. loss 下降 + cond_grad/ln_grad nonzero 全程 + nan_inf False

## 产物路径

- **verdict**: `verdicts/task470_issue177_gate3_a_recontinue_result.md`
- **verdict.json**: `products/task470_issue177_gate3_a_recontinue/verdict.json`
- **train_trace**: `products/task470_issue177_gate3_a_recontinue/train_trace.json`
- **adapter_init_proof**: `products/task470_issue177_gate3_a_recontinue/adapter_init_proof.json`
- **layernorm_unfreeze_proof**: `products/task470_issue177_gate3_a_recontinue/layernorm_unfreeze_proof.json`
- **gradient_proof**: `products/task470_issue177_gate3_a_recontinue/gradient_proof.json`
- **adapter ckpt**: `products/task470_issue177_gate3_a_recontinue/adapter.pt`
- **script**: `scripts/task470_issue177_gate3_a_recontinue.py`
- **log**: `logs/task470_issue177_gate3_a_recontinue.log`