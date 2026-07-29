#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #304 / D6 ablation Arm B — Stage 1 训练 wrapper (s_l only)

背景 (Task #304 D6 ablation):
  Arm B 设计: s_l alone (r_l=baseline) + per-layer c_k range
    - r_l = [1.0, 1.0, 1.0]   (baseline 取消 Issue #30 [0.1,1,10])
    - R_l = I identity
    - s_l = [2.0, 2.0, 2.0]   (跟 Issue #30 相同)
    - per-layer c_k range = [(1,5), (0.5,20), (0.5,20)] (跟 Issue #30 相同)

通过条件 (Task #304 Gate 1):
  (a) L0 utilization ≥ 90% at any evaluation step ≥ ep50
  (b) L1 utilization ≥ 90% at any evaluation step ≥ ep50
  (c) L2 utilization ≥ 90% at any evaluation step ≥ ep50
  (d) collision_rate ≤ 0.20
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

        print(f'[Task #304 Arm B] Layer {li}: r_l={r}, s_l={s}, eff_norm={eff.norm().item():.4f}, '
              f'weight_norm_after={q.embeddings.weight.norm().item():.4f}')


def apply_per_layer_c_k_range(model: HRQVAE, c_k_range_list: List[tuple], seed: int = 42):
    rng = np.random.RandomState(seed)
    for li, q in enumerate(model.hrq.vq_layers):
        cmin, cmax = c_k_range_list[li]
        c = float(rng.uniform(cmin, cmax))
        q.c = c
        print(f'[Task #304 Arm B] Layer {li}: c_k={c:.4f} (sampled from [{cmin}, {cmax}])')


def parse_args():
    parser = argparse.ArgumentParser(description='Task #304 / D6 Arm B Stage 1 training')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
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
    parser.add_argument('--save_limit', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/Instruments_d6_arm_b')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--radius_list', type=float, nargs='+', default=[1.0, 1.0, 1.0],
                        help='per-layer 半径缩放. Arm B = baseline (取消 Issue #30 [0.1,1,10])')
    parser.add_argument('--scale_list', type=float, nargs='+', default=[2.0, 2.0, 2.0],
                        help='per-layer scale factor. Arm B = Issue #30 相同')
    parser.add_argument('--c_k_range_list', type=str, default='1.0:5.0,0.5:20.0,0.5:20.0',
                        help='per-layer c_k range (跟 Issue #30 相同)')
    parser.add_argument('--c_k_seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    c_k_range_list = [tuple(map(float, r.split(':'))) for r in args.c_k_range_list.split(',')]
    assert len(c_k_range_list) == len(args.num_emb_list)
    assert len(args.radius_list) == len(args.num_emb_list)
    assert len(args.scale_list) == len(args.num_emb_list)

    print(f'[Task #304 Arm B] radius_list = {args.radius_list}')
    print(f'[Task #304 Arm B] scale_list = {args.scale_list}')
    print(f'[Task #304 Arm B] c_k_range_list = {c_k_range_list}')

    data = EmbDataset(args.data_path)
    print(f'[Task #304 Arm B] Dataset: {len(data)} items, dim={data.dim}')

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

    rotation_list = build_identity_rotation_list(args.num_emb_list, args.e_dim)
    print(f'[Task #304 Arm B] Apply per-layer codebook transforms: r_l={args.radius_list}, s_l={args.scale_list}')
    apply_per_layer_codebook_transforms(
        model=model,
        radius_list=args.radius_list,
        rotation_list=rotation_list,
        scale_list=args.scale_list,
    )

    print(f'[Task #304 Arm B] Apply per-layer c_k range: {c_k_range_list}')
    apply_per_layer_c_k_range(model, c_k_range_list, seed=args.c_k_seed)

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f'[Task #304 Arm B] Best Loss = {best_loss}, Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()