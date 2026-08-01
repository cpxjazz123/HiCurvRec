# Task #466 / Issue #173 [方向A Gate2] 三层 κ forward-path 梯度与重校准审计

## 任务摘要

Issue #173 owner 2026-08-01 08:09 派发: 方向A Gate2 真实 forward-path κ 梯度审计. 跟 #171 (closed NO-GO ffa9f0a) 的关键差异 = κ_logit_l 必须**显式接入 forward 计算图** (distance + codebook scale + 统一 #47 公式), 并加**反向梯度探针** + 有限差分验证. #171 失败根因 = κ_logit_l 是 unused parameter, 只在 post-forward synchronize_kappa() 调用 → forward 无梯度 → κ_grad=0 → 三层同步 0.693.

R18 4 维度对比 vs #171 (closed NO-GO ffa9f0a):
- **D1 spec**: #171 是"零中心 + 有界残差 + κ 同步重校准", #173 是"κ 必须真接 forward + 梯度探针 + 有限差分" → 不同
- **D2 实施**: #171 task464 wrapper, #173 必须改 wrapper 让 kappa_l 乘到 distance + codebook_scale, register_hook 捕获梯度 → 不同
- **D3 失败机制**: #171 是 unused parameter, #173 必须验证 forward path 真的连接 → 不同
- **D4 引用**: 同一族 + 新增 CrossRef DOI:10.1080/01621459.2026.2635077 (Hyperbolic Network Latent Space Model with Learnable Curvature 2026) + DOI:10.1016/j.neunet.2026.109172 (Task-Geometry Decoupling) → 不同

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth`
- **数据集**: Musical_Instruments (SID `Instruments_t5_hrqvae_poincare.npy` (9922, 4))
- **wrapper**: `HG_Rec_with_ThreeLayerKappaForwardPathAdapter`
  - 三层独立 learnable κ_l (L0 K=64, L1 K=128, L2 K=256)
  - kappa_l = softplus(kappa_logit_l) per layer
  - **forward path**: `distance_with_kappa = eucl_distance * kappa_l`, `codebook_with_scale = codebook * kappa_l.unsqueeze(-1)`
  - register_hook 捕获 kappa_l 梯度
  - 有限差分: `fd_grad = (loss(κ+ε) - loss(κ-ε)) / (2ε)`
  - 加 stage2 同步重校准 (per-layer kappa_l 改变后 codebook 重新 normalize)
- **GPU**: GPU 1 (R7 满足, GPU 0 task450 占, 1/2/3 空闲)

## Gate 顺序 (per Issue #173 spec)

- **precheck**: 保留三层独立 learnable κ + K64/K128/K256 ✅
- **Gate 1**: 真实 Stage1 κ/scale metadata + 三层 hash → ✅ PASS 预期
- **Gate 2**: 确认每层 κ 真实进入 forward, 梯度非零且无爆炸, codebook/距离同步重校准, SID 流一致 → **核心 Gate**
  - 失败信号 (per spec): κ_grad=0 或三层同步漂移 → 立即 NO-GO
- **Gate 3**: 仅 Gate 2 PASS 后接 Stage 3 T5 → ⏸ STOP (本任务仅 Gate 1+2)
- **Gate 4**: 仅 Gate 3 PASS 后做 Stage 4 R@K eval → ⏸ STOP (本任务仅 Gate 1+2)

## R23 强制

val_R@10=0 跨 ≥2 epoch → kill + NO-GO

## R12 强制

每 epoch 末强制存 ckpt (删旧 + 存新) + PID 文件

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 复用 task462 模板 (真实 Stage 1 export) + 修改 wrapper 让 kappa_l 真的进入 forward:
1. distance_with_kappa = torch.cdist(embeddings, codebook * kappa_l.unsqueeze(-1))  ← κ 真的乘到 codebook
2. forward_diff = (original_distance - distance_with_kappa).abs().mean() ← 验证 κ 真影响 forward
3. kappa_l.register_hook(lambda grad: grad_storage.append(grad.clone())) ← 梯度探针
4. 有限差分: ε=1e-3, loss(κ+ε) vs loss(κ-ε) → fd_grad 与 autograd grad 对比 (相对误差 < 1e-2)

## 产物路径

- **verdict**: `verdicts/task466_issue173_three_layer_kappa_forward_path_gradient_audit_result.md`
- **verdict.json**: `products/task466_issue173_stage2_three_layer_kappa_forward_path_gradient_audit/verdict.json`
- **stage1_proof**: `products/task466_issue173_stage2_three_layer_kappa_forward_path_gradient_audit/stage1_export_proof.json`
- **real_metadata**: `products/task466_issue173_stage2_three_layer_kappa_forward_path_gradient_audit/real_metadata.npy`
- **adapter ckpt**: `products/task466_issue173_stage2_three_layer_kappa_forward_path_gradient_audit/adapter.pt`
- **train_trace**: `products/task466_issue173_stage2_three_layer_kappa_forward_path_gradient_audit/train_trace.json`
- **script**: `scripts/task466_issue173_stage2_three_layer_kappa_forward_path_gradient_audit.py`
- **log**: `logs/task466_issue173_stage2_three_layer_kappa_forward_path_gradient_audit.log`