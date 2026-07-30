#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #327 / Issue #30 — Stage 1 Stage 1 训练 wrapper

背景 (Issue #30 body Gate 1):
  Stage 1 100 epoch 训练 (per-layer r_l + R_l + s_l + per-layer c_k range):
    - r_l = [0.1, 1.0, 10.0]   (L0 紧凑 / L1 中等 / L2 宽松)
    - R_l = I identity        (per-layer 异构 rotation, 后续可调)
    - s_l = [2.0, 2.0, 2.0]   (per-layer 异构 scale factor)
    - per-layer c_k range = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] (task242 Arm A)
    - κ 不 frozen (baseline 默认)

通过条件 (Issue #30 §Gate 1):
  (a) L0 utilization ≥ 90% at any evaluation step ≥ ep50
  (b) L1 utilization ≥ 90% at any evaluation step ≥ ep50
  (c) L2 utilization ≥ 90% at any evaluation step ≥ ep50
  (d) collision_rate ≤ 0.20

实施方式 (R11.4 + R11.5):
  - 不修改 HG-Rec/model/ 上游源码
  - 在 baseline train_hrqvae.py 训练前, 对每层 HVectorQuantization.embeddings.weight 应用
    per-layer codebook transform: e_i^l → (s_l · r_l) · R_l · e_i^l  (切空间)
  - 优化器直接更新变换后的 weight (Stage 1 全程使用变换后的码本)
  - Stage 2 Sinkhorn 推断沿用 baseline 路径 (transformed weight 通过 get_codebook() 一致暴露)
"""
import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae import HRQVAE
from model.hrqvae_trainer import Trainer
from model.utils import EmbDataset


def build_identity_rotation_list(num_emb_list, e_dim):
    return [torch.eye(e_dim) for _ in num_emb_list]


def apply_per_layer_codebook_transforms(model: HRQVAE,
                                          radius_list: List[float],
                                          rotation_list: List[torch.Tensor],
                                          scale_list: List[float]):
    """对每层 HVectorQuantization.embeddings.weight 应用 per-layer 几何变换.

    变换公式 (切空间, Issue #30 body §3):
      e_i^l → (s_l · r_l) · R_l · e_i^l

    Args:
      model: HRQVAE 实例
      radius_list: per-layer 半径缩放, len = num_layers
      rotation_list: per-layer rotation matrix (e_dim × e_dim), len = num_layers
      scale_list: per-layer scale factor, len = num_layers
    """
    n_layers = len(model.num_emb_list)
    assert len(radius_list) == n_layers, f"radius_list len {len(radius_list)} != {n_layers}"
    assert len(rotation_list) == n_layers, f"rotation_list len {len(rotation_list)} != {n_layers}"
    assert len(scale_list) == n_layers, f"scale_list len {len(scale_list)} != {n_layers}"

    for li, q in enumerate(model.hrq.vq_layers):
        r = radius_list[li]
        R = rotation_list[li]
        s = scale_list[li]
        e_dim = q.embeddings.weight.shape[-1]
        device = q.embeddings.weight.device
        dtype = q.embeddings.weight.dtype

        # eff = (s · r) · R (e_dim × e_dim), 应用 W_eff = W @ eff.T
        eff = ((s * r) * R).to(device=device, dtype=dtype)

        with torch.no_grad():
            q.embeddings.weight.data = q.embeddings.weight.data @ eff.t()

        print(f'[Issue #30] Layer {li}: r_l={r}, s_l={s}, R_l=I (eye), eff_norm={eff.norm().item():.4f}, '
              f'weight_norm_after={q.embeddings.weight.norm().item():.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Task #327 / Issue #30 train_hrqvae_codebook_transforms')

    # baseline args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100, help='Issue #30 Gate 1 = 100 epoch')
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
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--loss_mult_codebook', type=float, default=1.0)
    parser.add_argument('--radii', type=float, nargs='+', default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--save_limit', type=int, default=1, help='R12: 只保留最新')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/Instruments_codebook_transforms')
    parser.add_argument('--seed', type=int, default=42)

    # Issue #30 专属参数
    parser.add_argument('--radius_list', type=float, nargs='+', default=[0.1, 1.0, 10.0],
                        help='per-layer 半径缩放. 默认 [0.1, 1.0, 10.0].')
    parser.add_argument('--scale_list', type=float, nargs='+', default=[2.0, 2.0, 2.0],
                        help='per-layer scale factor. 默认 [2.0, 2.0, 2.0].')
    parser.add_argument('--c_k_range_list', type=str, default='1.0:5.0,0.5:20.0,0.5:20.0',
                        help='per-layer c_k range (Issue #30 §Gate 1). 格式 "low1:high1,...".')
    parser.add_argument('--c_k_seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    c_k_range_list = [tuple(map(float, r.split(':'))) for r in args.c_k_range_list.split(',')]
    assert len(c_k_range_list) == len(args.num_emb_list), \
        f"c_k_range_list len {len(c_k_range_list)} != num_emb_list len {len(args.num_emb_list)}"
    assert len(args.radius_list) == len(args.num_emb_list), \
        f"radius_list len {len(args.radius_list)} != num_emb_list len {len(args.num_emb_list)}"
    assert len(args.scale_list) == len(args.num_emb_list), \
        f"scale_list len {len(args.scale_list)} != num_emb_list len {len(args.num_emb_list)}"

    print(f'[Issue #30] radius_list = {args.radius_list}')
    print(f'[Issue #30] scale_list = {args.scale_list}')
    print(f'[Issue #30] c_k_range_list = {c_k_range_list}')

    # baseline HRQVAE 构造
    data = EmbDataset(args.data_path)
    print(f'[Issue #30] Dataset: {len(data)} items, dim={data.dim}')

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

    # 应用 per-layer codebook transforms (Issue #30 核心)
    rotation_list = build_identity_rotation_list(args.num_emb_list, args.e_dim)
    print(f'[Issue #30] Apply per-layer codebook transforms: r_l={args.radius_list}, s_l={args.scale_list}, R_l=I')
    apply_per_layer_codebook_transforms(
        model=model,
        radius_list=args.radius_list,
        rotation_list=rotation_list,
        scale_list=args.scale_list,
    )

    # baseline Trainer 沿用, 加 R12 save_limit=1
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f'[Issue #30] Best Loss = {best_loss}, Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()
