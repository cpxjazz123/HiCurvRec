# Task #468 / Issue #175 [方向A Gate2] κ 经代码本量化距离直接进入 VQ 损失的 forward-path 修复

## 任务摘要

Issue #175 owner 2026-08-01 08:50 派发: 方向A Gate2 真实 VQ-commitment-loss 路径 κ 修复. 跟前 #173 (closed NO-GO a8a5c32) 的关键差异 = κ_l 必须**直接代入 RQ-VAE quantization 的距离函数本身**, 该距离张量本身进入 vq_loss/commitment_loss 计算图 (不是并行 diagnostic tensor). #173 失败根因 = forward_diff 仅作为独立 tensor 返回, 未加入 outputs.loss → backward 不传播 → κ_grad=0.

R18 4 维度对比 vs #173 (closed NO-GO a8a5c32):
- **D1 spec**: #173 是"forward_diff 真接 forward 但不接 loss", #175 是"distance 直接进入 VQ commitment loss (loss-relevant path)" → 不同
- **D2 实施**: #173 是改 Stage 3 adapter 加 forward_diff, #175 必须改 Stage 2 RQ-VAE quantization forward (utils.py:239 + :260-261) 让 κ 进入 commitment_loss/codebook_loss 主路径 → 不同
- **D3 失败机制**: #173 是 "旁路 metric tensor", #175 是 "VQ commitment loss 真实 loss-relevant 路径" → 不同
- **D4 引用**: 同一族 + 新增 DOI:10.24963/ijcai.2023/497 (Curvature-aware Isometric Embedding View, IJCAI 2023) + Neural Networks 2026 + Mixed-Curvature 2026 → 不同

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (sha256=59a38fa3...)
- **Stage 1 输入**: `HG-Rec/dataset/Instruments/item_emb.parquet` (9922 items × 768d)
- **wrapper**: `KModulatedHRQVAE`
  - 三层独立 learnable kappa_logit_l (n_layers=3, init=log(e-1)≈0.5413 → softplus≈1.0)
  - 修改 utils.py HVectorQuantization: `poincare_distance(x_exp, cb_exp, kappa_l)` (替换固定 self.c=1)
  - 同一 κ_l 用于 codebook assignment (line 239) + commitment_loss + codebook_loss (line 260-261)
  - register_hook on kappa_logit_l
- **GPU**: GPU 1 (R7 满足, GPU 0 task450 占, 2/3 留给 task469)

## Gate 顺序 (per Issue #175 spec)

- **precheck**: 三层独立 learnable κ + K64/K128/K256 维持不变 ✅
- **Gate 1**: 真实 Stage 1 κ/scale metadata + 三层 hash → ✅ PASS 预期
- **Gate 2**: 证明 κ_l 梯度来自 vq_loss 本身 (非旁路 tensor), 三层 κ 独立更新, codebook 同步重校准 → **核心 Gate**
  - 失败信号 (per spec): 全零梯度或三层同步 → 立即 NO-GO
- **Gate 3**: 仅 Gate 2 PASS 后接 Stage 3 T5 几何适配 → ⏸ STOP (本任务仅 Gate 1+2)
- **Gate 4**: 仅 Gate 3 PASS 后做一次可复现单seed Task84 Stage4, 报告 R@5/10/20 NDCG@5/10/20 → ⏸ STOP (本任务仅 Gate 1+2)

## R23 强制

val_R@10=0 跨 ≥2 epoch / κ_grad=0 / 三层 κ 同步 → kill + NO-GO

## R12 强制

每 epoch 末强制存 ckpt (删旧 + 存新) + PID 文件

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 修改 utils.py HVectorQuantization:
1. 添加 kappa_logit (init=log(e-1)≈0.5413, softplus→1.0 接近 baseline c=1)
2. `self.kappa = F.softplus(self.kappa_logit)` (替换 self.c=1)
3. `d = poincare_distance(x_exp, cb_exp, self.kappa)` (line 239)
4. `commitment_loss = poincare_distance(x_q.detach(), latent, self.kappa)**2` (line 260)
5. `codebook_loss = poincare_distance(x_q, latent.detach(), self.kappa)**2` (line 261)
6. register_hook on kappa_logit 捕获梯度
7. 有限差分 ε=1e-3 验证 dL/dκ 正确
8. 从 baseline ckpt 短训 (5 epoch), 验证 κ_grad ≠ 0 + 三层 κ 不同步

## 产物路径

- **verdict**: `verdicts/task468_issue175_stage2_kappa_vq_loss_forward_path_fix_result.md`
- **verdict.json**: `products/task468_issue175_stage2_kappa_vq_loss_forward_path_fix/verdict.json`
- **stage1_proof**: `products/task468_issue175_stage2_kappa_vq_loss_forward_path_fix/stage1_export_proof.json`
- **real_metadata**: `products/task468_issue175_stage2_kappa_vq_loss_forward_path_fix/real_metadata.npy`
- **adapter ckpt**: `products/task468_issue175_stage2_kappa_vq_loss_forward_path_fix/adapter.pt`
- **train_trace**: `products/task468_issue175_stage2_kappa_vq_loss_forward_path_fix/train_trace.json`
- **script**: `scripts/task468_issue175_stage2_kappa_vq_loss_forward_path_fix.py`
- **log**: `logs/task468_issue175_stage2_kappa_vq_loss_forward_path_fix.log`