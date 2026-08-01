# Task #469 / Issue #176 [方向B Gate2] mixing 权重 + κ 经代码本量化距离直接进入 VQ 损失的 forward-path 修复

## 任务摘要

Issue #176 owner 2026-08-01 08:50 派发: 方向B Gate2 真实 VQ-commitment-loss 路径 mixing+κ 修复. 跟前 #174 (closed NO-GO a8a5c32) 的关键差异 = mixing_logit_l + kappa_logit_l 都必须**直接代入 RQ-VAE quantization 的距离函数本身**, 该 distance_l = mix_l · [hyp_component(κ_l), eucl_component, learnable-κ] 直接进入 vq_loss/commitment_loss (不是并行 diagnostic tensor). #174 失败根因 = forward_diff 仅作为独立 tensor 返回, 未加入 outputs.loss → backward 不传播 → κ_grad=0 + mix_grad=0.

R18 4 维度对比 vs #174 (closed NO-GO a8a5c32):
- **D1 spec**: #174 是"mixing+κ forward_diff 真接 forward 但不接 loss", #176 是"mixing_l · distance_components 直接进入 VQ commitment loss (loss-relevant path)" → 不同
- **D2 实施**: #174 是改 Stage 3 adapter 加 forward_diff, #176 必须改 Stage 2 RQ-VAE quantization forward 让 mixing+κ 三分量距离进入 commitment_loss/codebook_loss 主路径 → 不同
- **D3 失败机制**: #174 是 "旁路 metric tensor", #176 是 "VQ commitment loss 真实 loss-relevant 路径" → 不同
- **D4 引用**: 同一族 + 新增 Mixed-Curvature 2026 DOI:10.1007/978-981-95-4367-0_28 + Task-Geometry Decoupling Neural Networks 2026 → 不同

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (sha256=59a38fa3...)
- **Stage 1 输入**: `HG-Rec/dataset/Instruments/item_emb.parquet` (9922 items × 768d)
- **wrapper**: `MixingKModulatedHRQVAE`
  - 三层独立 learnable kappa_logit_l (n_layers=3, init=log(e-1)≈0.5413 → softplus≈1.0)
  - 三层独立 mixing_logit_l (n_layers=3, n_components=3: hyp + eucl + learnable-κ)
  - mixing_l = softmax(mixing_logit_l, dim=-1) per layer (3 weights sum to 1)
  - 修改 utils.py HVectorQuantization: `d = mixing_l · [hyp_d(κ_l), eucl_d, learnable_kappa_d]` (替换固定 self.c=1)
  - 同一 mixing_l + κ_l 用于 codebook assignment + commitment_loss + codebook_loss
  - register_hook 双向 on mixing_logit_l + kappa_logit_l
- **GPU**: GPU 2 (R7 满足)

## Gate 顺序 (per Issue #176 spec)

- **precheck**: 三层独立 learnable κ + 每层固定双曲分量/固定欧氏分量/可学习 κ + 可学习 mixing 权重, K64/K128/K256 维持不变 ✅
- **Gate 1**: 真实 Stage 1 三层三分量 metadata + hash → ✅ PASS 预期
- **Gate 2**: 证明 mix_l/κ_l 梯度来自 vq_loss 本身 (非旁路 tensor), 三层独立更新, codebook 同步重校准 → **核心 Gate**
  - 失败信号 (per spec): 全零梯度或三层同步 → 立即 NO-GO
- **Gate 3**: 仅 Gate 2 PASS 后接 Stage 3 T5 几何适配 → ⏸ STOP (本任务仅 Gate 1+2)
- **Gate 4**: 仅 Gate 3 PASS 后做一次可复现单seed Task84 Stage4, 报告 R@5/10/20 NDCG@5/10/20 → ⏸ STOP (本任务仅 Gate 1+2)

## R23 强制

val_R@10=0 跨 ≥2 epoch / mix_grad=0 / κ_grad=0 / 三层 mixing/κ 同步 → kill + NO-GO

## R12 强制

每 epoch 末强制存 ckpt (删旧 + 存新) + PID 文件

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 修改 utils.py HVectorQuantization:
1. 添加 kappa_logit (init=log(e-1)≈0.5413) + mixing_logit (init=[0,0,0], softmax=[1/3,1/3,1/3])
2. `kappa_l = F.softplus(kappa_logit)` + `mixing_l = F.softmax(mixing_logit, dim=-1)`
3. 三分量距离:
   - `d_hyp = poincare_distance(x_exp, cb_exp, kappa_l)` (固定双曲, κ)
   - `d_eucl = euclidean_distance(x_exp, cb_exp)` (固定欧氏, 无 κ)
   - `d_learnable_kappa = poincare_distance(x_exp, cb_exp * kappa_l, 1.0)` (learnable-κ)
4. `d = mixing_l[:, 0] * d_hyp + mixing_l[:, 1] * d_eucl + mixing_l[:, 2] * d_learnable_kappa`
5. `commitment_loss = d.detach()**2 + d**2` 风格 (跟原始一致)
6. `codebook_loss = (mixing weighted).mean()**2`
7. register_hook 双向 on mixing_logit_l + kappa_logit_l
8. 有限差分 ε=1e-3 验证 dL/dmixing + dL/dκ 正确
9. 从 baseline ckpt 短训 (5 epoch), 验证 mix_grad ≠ 0 + κ_grad ≠ 0 + 三层 mixing/κ 不同步

## 产物路径

- **verdict**: `verdicts/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix_result.md`
- **verdict.json**: `products/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix/verdict.json`
- **stage1_proof**: `products/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix/stage1_export_proof.json`
- **real_metadata**: `products/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix/real_three_component_metadata.npy`
- **adapter ckpt**: `products/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix/adapter.pt`
- **train_trace**: `products/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix/train_trace.json`
- **script**: `scripts/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix.py`
- **log**: `logs/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix.log`