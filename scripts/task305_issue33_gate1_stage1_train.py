#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #305 / Issue #33 — Stage 1 训练 wrapper (per-item soft-assign on #30 GO 配置)

背景 (Issue #33 body §Gate 1):
  Stage 1 100 epoch 训练 (per-layer r_l + s_l + per-item soft-assign):
    - r_l = [0.1, 1.0, 10.0]   (L0 紧凑 / L1 中等 / L2 宽松, 沿用 #30 GO)
    - R_l = I identity        (per-layer rotation, 留接口)
    - s_l = [2.0, 2.0, 2.0]   (per-layer scale factor, 沿用 #30 GO)
    - per-item soft-assign commitment loss: τ_l = 1.0 (baseline default)
    - κ 不 frozen (baseline 默认)

通过条件 (Issue #33 §Gate 1):
  (a) L0 utilization ≥ 90% at any evaluation step ≥ ep50
  (b) L1 utilization ≥ 90% at any evaluation step ≥ ep50
  (c) L2 utilization ≥ 90% at any evaluation step ≥ ep50
  (d) collision_rate ≤ 0.20
  (e) ‖x‖_E ∈ [0.7, 0.95] (健康 norm 区, 沿用 #30 GO s_l=2 实现)

实施方式 (R11.4 + R11.5 + R12):
  - 不修改 HG-Rec/model/ 上游源码
  - 沿用 task301 wrapper 模式: apply_per_layer_codebook_transforms (Stage 1 训练前应用 e_i^l → (s_l · r_l) · R_l · e_i^l)
  - **新增**: 在构造后 monkey-patch HVectorQuantization.forward 注入 per-item soft-assign commitment:
      commitment_loss = mean_b (Σ_k softmax(-d/τ_l) · d_k²)  (per-item 软分布期望)
      替代 baseline mean of d² at argmin only (硬分配)
  - Stage 2 Sinkhorn 推断沿用 baseline 路径 (transformed weight 通过 get_codebook() 一致暴露)
  - R12: trainer save_limit=1, 删除旧 ckpt 保存新 ckpt
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae import HRQVAE
from model.hrqvae_trainer import Trainer
from model.utils import EmbDataset, poincare_distance, expmap0, logmap0, proj_to_ball, sinkhorn_algorithm


# ============================================================================
# Per-Layer Codebook Transforms (Issue #30 wrapper pattern)
# ============================================================================

def build_identity_rotation_list(num_emb_list, e_dim):
    return [torch.eye(e_dim) for _ in num_emb_list]


def apply_per_layer_codebook_transforms(model: HRQVAE,
                                          radius_list: List[float],
                                          rotation_list: List[torch.Tensor],
                                          scale_list: List[float]):
    """对每层 HVectorQuantization.embeddings.weight 应用 per-layer 几何变换 (切空间)."""
    n_layers = len(model.num_emb_list)
    assert len(radius_list) == n_layers
    assert len(rotation_list) == n_layers
    assert len(scale_list) == n_layers

    for li, q in enumerate(model.hrq.vq_layers):
        r = radius_list[li]
        R = rotation_list[li]
        s = scale_list[li]
        e_dim = q.embeddings.weight.shape[-1]
        device = q.embeddings.weight.device
        dtype = q.embeddings.weight.dtype

        eff = ((s * r) * R).to(device=device, dtype=dtype)

        with torch.no_grad():
            q.embeddings.weight.data = q.embeddings.weight.data @ eff.t()

        print(f'[Issue #33] Layer {li}: r_l={r}, s_l={s}, eff_norm={eff.norm().item():.4f}, '
              f'weight_norm_after={q.embeddings.weight.norm().item():.4f}')


# ============================================================================
# Per-Item Soft-Assign Patch (Issue #33 核心 — 注入 commit loss on softmax 概率分布)
# ============================================================================

def apply_per_item_soft_assign_commitment(model: HRQVAE,
                                            tau_list: List[float],
                                            per_item_softassign_enabled: bool = True):
    """monkey-patch 每层 HVectorQuantization.forward 注入 per-item soft-assign commitment.

    注入后, 训练全程 forward 路径:
      1. 应用 per-layer codebook transform (W_eff = W @ eff.T, 沿用 Issue #30)
      2. 复用 baseline distance 计算 (Poincaré, sigmoid 等)
      3. argmin/Sinkhorn 沿用 baseline (硬分配, Stage 2 Sinkhorn 推断使用)
      4. commitment_loss 替换为 per-item soft expectation:
            commitment_loss = mean_b (Σ_k softmax(-d/τ_l) · d_k²)
         替代 baseline mean of d² at argmin only
      5. 恢复原始 weight (临时 transform)

    注意: Stage 2 Sinkhorn 推断时仍用 argmin hard-assign (Stage 2 解码时不用此 patch).
    """
    n_layers = len(model.hrq.vq_layers)
    assert len(tau_list) == n_layers

    print(f'[Issue #33] Apply per-item soft-assign commitment: τ_l = {tau_list}, enabled = {per_item_softassign_enabled}')

    for li, q in enumerate(model.hrq.vq_layers):
        _li = li
        _tau = tau_list[_li]
        _soft_enabled = per_item_softassign_enabled
        _beta = q.beta
        _c = q.c

        def make_dual_patched_forward(orig_forward, beta, c, tau, soft_enabled):
            def dual_patched(_self, x, use_sk=True):
                # 1. 应用 per-layer codebook transform
                original_weight = _self.embeddings.weight.data.clone()
                # 在 task301 apply_per_layer_codebook_transforms 中已永久修改 weight.data
                # 这里无需重复 (已生效), 但保留代码以备双保险 — 实测时验证
                try:
                    # 2. 复算距离 (沿用 baseline)
                    latent = x.view(-1, _self.e_dim)
                    codebook = _self.embeddings.weight
                    latent_h = proj_to_ball(expmap0(latent, c), c)
                    codebook_h = proj_to_ball(expmap0(codebook, c), c)
                    B = latent_h.shape[0]
                    K = codebook_h.shape[0]
                    x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
                    cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
                    d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (B, K)

                    if not use_sk or _self.sk_eps <= 0:
                        indices = torch.argmin(d, dim=-1)
                    else:
                        max_d = d.max()
                        min_d = d.min()
                        middle = (max_d + min_d) / 2
                        amplitude = max_d - middle + 1e-10
                        d_centered = ((d - middle) / amplitude).double()
                        Q = sinkhorn_algorithm(d_centered, _self.sk_eps, _self.sk_iters)
                        if torch.isnan(Q).any() or torch.isinf(Q).any():
                            raise ValueError("Sinkhorn algorithm produced NaN or Inf values.")
                        indices = torch.argmax(Q, dim=-1)

                    x_q = codebook.index_select(0, indices)

                    # 3. per-item soft-assign commitment loss (Issue #33)
                    if soft_enabled:
                        # per-item softmax over -d/τ → 概率分布 → 期望 d²
                        d_for_softmax = d / tau  # (B, K)
                        log_neg_d = -d_for_softmax
                        per_item_log_probs = F.log_softmax(log_neg_d, dim=-1)  # (B, K)
                        per_item_probs = per_item_log_probs.exp()  # (B, K)
                        d_sq = d * d  # (B, K)
                        per_item_commitment = (per_item_probs * d_sq).sum(dim=-1)  # (B,)
                        commitment_loss = per_item_commitment.mean()
                    else:
                        # baseline hard argmin commitment (fallback)
                        commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, c) ** 2)

                    # 4. codebook_loss unchanged (argmin)
                    codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), c) ** 2)
                    loss = commitment_loss + beta * codebook_loss

                    x_q = logmap0(x_q, c)
                    latent = logmap0(latent, c)
                    x_q = x + (x_q - x).detach()
                    indices = indices.view(x.shape[:-1])
                    return x_q, loss, indices
                finally:
                    _self.embeddings.weight.data = original_weight
            return dual_patched

        # 替换 vq_layers[li].forward (持久 monkey-patch, 影响所有训练 step)
        q.forward = make_dual_patched_forward(q.forward, _beta, _c, _tau, _soft_enabled).__get__(q, type(q))
        print(f'[Issue #33] Layer {li}: monkey-patch forward applied with τ={_tau}, soft={_soft_enabled}')


# ============================================================================
# Args + Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Task #305 / Issue #33 Stage 1 training (per-item soft-assign on #30 GO)')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100, help='Issue #33 Gate 1 = 100 epoch')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--learner', type=str, default='AdamW')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--data_path', type=str, default=f'{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--loss_type', type=str, default='poincare')
    parser.add_argument('--kmeans_init', type=bool, default=True)
    parser.add_argument('--kmeans_iters', type=int, default=1000)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument('--sk_iters', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:1', help='Issue #33 申请 GPU 1')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--save_limit', type=int, default=1, help='R12: 只保留最新')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/Instruments_issue33_peritem_softassign')
    parser.add_argument('--seed', type=int, default=42)

    # Issue #33 专属参数
    parser.add_argument('--radius_list', type=float, nargs='+', default=[0.1, 1.0, 10.0],
                        help='per-layer 半径缩放. 沿用 #30 GO = [0.1, 1.0, 10.0].')
    parser.add_argument('--scale_list', type=float, nargs='+', default=[2.0, 2.0, 2.0],
                        help='per-layer scale factor. 沿用 #30 GO = [2.0, 2.0, 2.0].')
    parser.add_argument('--per_item_tau_list', type=float, nargs='+', default=[1.0, 1.0, 1.0],
                        help='per-layer 温度. 默认 [1.0, 1.0, 1.0] (baseline default).')
    parser.add_argument('--per_item_softassign_enabled', type=int, default=1,
                        help='1=enable per-item soft commitment (Issue #33 default), 0=disable (回退到 #30 hard argmin)')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert len(args.radius_list) == len(args.num_emb_list)
    assert len(args.scale_list) == len(args.num_emb_list)
    assert len(args.per_item_tau_list) == len(args.num_emb_list)

    print(f'[Issue #33] radius_list = {args.radius_list}')
    print(f'[Issue #33] scale_list = {args.scale_list}')
    print(f'[Issue #33] per_item_tau_list = {args.per_item_tau_list}')
    print(f'[Issue #33] per_item_softassign_enabled = {args.per_item_softassign_enabled}')

    data = EmbDataset(args.data_path)
    print(f'[Issue #33] Dataset: {len(data)} items, dim={data.dim}')

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    # Layer 1: 应用 per-layer Codebook Transforms (Issue #30)
    rotation_list = build_identity_rotation_list(args.num_emb_list, args.e_dim)
    print(f'[Issue #33] Apply per-layer codebook transforms: r_l={args.radius_list}, s_l={args.scale_list}')
    apply_per_layer_codebook_transforms(
        model=model,
        radius_list=args.radius_list,
        rotation_list=rotation_list,
        scale_list=args.scale_list,
    )

    # Layer 2: 应用 per-item soft-assign commitment loss (Issue #33 创新)
    apply_per_item_soft_assign_commitment(
        model=model,
        tau_list=args.per_item_tau_list,
        per_item_softassign_enabled=bool(args.per_item_softassign_enabled),
    )

    # 沿用 baseline Trainer (R12 save_limit=1)
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f'[Issue #33] Best Loss = {best_loss}, Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()
