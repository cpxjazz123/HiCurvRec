# Task #467 / Issue #174 [方向B Gate2] 逐层 κ/mixing forward-path 梯度审计

## 任务摘要

Issue #174 owner 2026-08-01 08:10 派发: 方向B Gate2 真实 forward-path κ + mixing 梯度审计. 跟 #172 (closed NO-GO ffa9f0a) 的关键差异 = mixing_logit_l + kappa_logit_l 都必须**显式接入 forward 计算图** (距离 + 代码本 + SID), 加**逐层梯度与有限差分探针**. #172 失败根因 = mixing_logit_l + kappa_logit_l 都是 unused parameter, 只在 post-forward get_mixing_l() / synchronize_kappa() 调用 → forward 无梯度 → κ_grad=0 + mix_grad=0 → mixing 三层均匀 1/3 + κ 三层同步 0.693.

R18 4 维度对比 vs #172 (closed NO-GO ffa9f0a):
- **D1 spec**: #172 是"逐层 softmax 混合权重 + 零中心范数受限残差", #174 是"mixing + κ 必须真接 forward + 逐层梯度探针 + 有限差分" → 不同
- **D2 实施**: #172 task465 wrapper, #174 必须改 wrapper 让 mixing_l (per layer, 3 components) + kappa_l 真接 forward, register_hook 捕获梯度 → 不同
- **D3 失败机制**: #172 是 unused parameter (mixing + κ), #174 必须验证 forward path 真的连接 + 三层 mixing 不同 → 不同
- **D4 引用**: 同一族 + 新增 CrossRef DOI:10.1007/978-981-95-4367-0_28 (Breaking the Heterophily Mixing Barrier in Graph Learning: A Mixed-Curvature Product Manifold Approach 2026) → 不同

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth`
- **数据集**: Musical_Instruments (SID `Instruments_t5_hrqvae_poincare.npy` (9922, 4))
- **wrapper**: `HG_Rec_with_PerLayerMixingKappaForwardPathAdapter`
  - 三层独立 learnable κ_l (L0 K=64, L1 K=128, L2 K=256)
  - 三层独立 mixing_logit_l (n_layers=3, n_components=3: 固定双曲 + 固定欧氏 + learnable-κ)
  - mixing_l = softmax(mixing_logit_l, dim=-1) per layer (3 weights sum to 1)
  - **forward path**: 
    - distance_hyp_per_layer = torch.cdist(embeddings, codebook)  ← 固定双曲
    - distance_eucl_per_layer = torch.cdist(embeddings, codebook)  ← 固定欧氏
    - distance_learnable_kappa = torch.cdist(embeddings, codebook * kappa_l.unsqueeze(-1))  ← learnable-κ
    - mixing_distance = mixing_l[:, 0] * distance_hyp + mixing_l[:, 1] * distance_eucl + mixing_l[:, 2] * distance_learnable_kappa
  - register_hook 捕获 mixing_logit_l + kappa_logit_l 梯度
  - 有限差分: `fd_grad = (loss(mixing+ε) - loss(mixing-ε)) / (2ε)`
- **GPU**: GPU 2 (R7 满足, GPU 0 task450 占, 1 留给 task466, 3 空闲)

## Gate 顺序 (per Issue #174 spec)

- **precheck**: 保留三层独立 learnable κ + 每层 K64/K128/K256 + 混合组件 ✅
- **Gate 1**: 真实 Stage1 三层三分量 metadata + hash → ✅ PASS 预期
- **Gate 2**: 逐层 κ/mixing 真正影响 forward, 梯度/有限差分非零且稳定, 分量范数和 SID 流可复现 → **核心 Gate**
  - 失败信号 (per spec): 全零梯度或三层强制相同 → 立即 NO-GO
- **Gate 3**: 仅 Gate 2 PASS 后接 Stage 3 T5 → ⏸ STOP (本任务仅 Gate 1+2)
- **Gate 4**: 仅 Gate 3 PASS 后做 Stage 4 R@K eval → ⏸ STOP (本任务仅 Gate 1+2)

## R23 强制

val_R@10=0 跨 ≥2 epoch → kill + NO-GO

## R12 强制

每 epoch 末强制存 ckpt (删旧 + 存新) + PID 文件

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 复用 task463 模板 (真实 Stage 1 三分量 export) + 修改 wrapper 让 mixing_l + kappa_l 真的进入 forward:
1. distance_mixed = sum_l mixing_l[:, c] * distance_component[c]  ← mixing 真影响 forward
2. forward_diff = (original_distance - distance_mixed).abs().mean() ← 验证 mixing 真影响
3. mixing_logit_l.register_hook + kappa_logit_l.register_hook  ← 双向梯度探针
4. 有限差分 ε=1e-3, 验证 mixing 三个分量都非零 (三层 mixing 应该有差异)
5. 三层 mixing 应该不同 (init 0 时 softmax 全 1/3, 但训练后必须有 per-layer 差异)

## 产物路径

- **verdict**: `verdicts/task467_issue174_per_layer_mixing_kappa_forward_path_gradient_audit_result.md`
- **verdict.json**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/verdict.json`
- **stage1_proof**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/stage1_export_proof.json`
- **real_metadata**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/real_metadata.npy`
- **adapter ckpt**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/adapter.pt`
- **train_trace**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/train_trace.json`
- **script**: `scripts/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit.py`
- **log**: `logs/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit.log`