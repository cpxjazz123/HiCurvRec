#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #350 / Issue #62 Gate 1 — Stage 1 Arm C 训练 wrapper
(#30 + #43 联合, baseline K=[64,128,256] + Sinkhorn ON, 1000 epoch batch=1024 lr=1e-3)

背景 (Issue #62 §Gate 1 Arm C):
  Arm C = #30 per-layer Codebook Transforms + #43 HypPreEncoder (NO K0=256, NO Sinkhorn OFF)
  K=[64,128,256] baseline, Sinkhorn ON (sk_eps=[0.003,0.003,0.003])
  1000 epoch batch=1024 + lr=1e-3 AdamW

实施方式 (R11.4 + R11.5):
  - 不修改 HG-Rec/model/ 上游源码
  - Issue #30 per-layer transforms: e_i^l → (s_l · r_l) · R_l · e_i^l
    r_l=[0.1, 1.0, 10.0], R_l=I, s_l=[2.0, 2.0, 2.0]
  - Issue #43 HypPreEncoder 预量化双曲感知映射 (c=0.74)
  - K_list = [64, 128, 256] (Issue #30 baseline K, 不是 K0=256)
  - Sinkhorn ON (sk_eps=[0.003, 0.003, 0.003])
  - per R12: save_limit=1 + 每个 epoch 强制保存 best_loss_model.pth
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
from scripts.task334_issue43_gate2a_hyp_pre_encoder import HRQVAEWithHypPre


def build_identity_rotation_list(num_emb_list, e_dim):
    return [torch.eye(e_dim) for _ in num_emb_list]


def apply_per_layer_codebook_transforms(model: HRQVAE,
                                          radius_list: List[float],
                                          rotation_list: List[torch.Tensor],
                                          scale_list: List[float]):
    """Issue #30: per-layer codebook 几何变换."""
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=f'{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--kmeans_init', type=bool, default=True)
    parser.add_argument('--kmeans_iters', type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.003, 0.003, 0.003])
    parser.add_argument('--sk_iters', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--save_limit', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str,
                        default=f'{REPO}/products/task350/hrqvae_issue62_gate1_armc')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learner', type=str, default='AdamW')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_step', type=int, default=5)

    parser.add_argument('--hyp_c', type=float, default=0.74)
    parser.add_argument('--use_hyp_pre_encoder', type=bool, default=True)
    parser.add_argument('--radius_list', type=float, nargs='+', default=[0.1, 1.0, 10.0])
    parser.add_argument('--scale_list', type=float, nargs='+', default=[2.0, 2.0, 2.0])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=1000)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert len(args.radius_list) == len(args.num_emb_list), \
        f"radius_list len {len(args.radius_list)} != num_emb_list len {len(args.num_emb_list)}"
    assert len(args.scale_list) == len(args.num_emb_list), \
        f"scale_list len {len(args.scale_list)} != num_emb_list len {len(args.num_emb_list)}"

    print(f'[Issue #62 Arm C] radius_list = {args.radius_list}')
    print(f'[Issue #62 Arm C] scale_list = {args.scale_list}')
    print(f'[Issue #62 Arm C] hyp_c = {args.hyp_c}')
    print(f'[Issue #62 Arm C] num_emb_list = {args.num_emb_list} (Issue #30 baseline K)')
    print(f'[Issue #62 Arm C] sk_epsilons = {args.sk_epsilons} (Sinkhorn ON)')
    print(f'[Issue #62 Arm C] lr = {args.lr}, batch_size = {args.batch_size}, num_epochs = {args.num_epochs}')

    data = EmbDataset(args.data_path)
    print(f'[Issue #62 Arm C] Dataset: {len(data)} items, dim={data.dim}')

    base = HRQVAE(
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

    if args.use_hyp_pre_encoder:
        model = HRQVAEWithHypPre(base, c=args.hyp_c, enabled=True)
        print(f'[Issue #43 Gate 2a] HypPreEncoder ENABLED (c={args.hyp_c})')
    else:
        model = base
        print(f'[Issue #43 Gate 2a] HypPreEncoder DISABLED')

    rotation_list = build_identity_rotation_list(args.num_emb_list, args.e_dim)
    print(f'[Issue #30] Apply per-layer codebook transforms: r_l={args.radius_list}, s_l={args.scale_list}, R_l=I')
    apply_per_layer_codebook_transforms(
        model=base,
        radius_list=args.radius_list,
        rotation_list=rotation_list,
        scale_list=args.scale_list,
    )

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f'[Issue #62 Arm C] Best Loss = {best_loss}, Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()