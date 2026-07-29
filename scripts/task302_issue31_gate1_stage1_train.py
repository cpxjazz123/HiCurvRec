#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #302 / Issue #31 — Gate 1 Stage 1 训练 wrapper

背景 (Issue #31 body §Gate 1):
  Stage 1 100 epoch 训练 (per-layer encoder regularization):
    - β_l = [0.1, 0.3, 0.5]   (per-layer 异构 commit loss 权重, baseline 0.5 → per-layer 异构)
    - α_l = [0.01, 0.005, 0.001]   (per-layer 异构 codebook center anchor)
    - γ_l = [0.001, 0.0005, 0.0001]   (per-layer 异构 encoder L2 正则)
    - c_k_range_list = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)]   (task242 Arm A 协同)

通过条件 (Issue #31 §Gate 1, 五条同时):
  (a) L0 utilization ≥ 90% at any evaluation step ≥ ep50
  (b) L1 utilization ≥ 90% at any evaluation step ≥ ep50
  (c) L2 utilization ≥ 90% at any evaluation step ≥ ep50
  (d) collision_rate ≤ 0.20
  (e) ‖x‖_E ≥ 0.3 at any evaluation step ≥ ep50

实施方式 (R11.4 + R11.5):
  - 不修改 HG-Rec/model/ 上游源码
  - 使用 task302_issue31_gate0_wrapper.EncoderRegHRQVAE (继承 HRQVAE)
  - 通过 per-layer β_l patch + forward 重写加 anchor + encoder L2 正则
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
sys.path.insert(0, f'{REPO}/scripts')

from model.hrqvae_trainer import Trainer
from model.utils import EmbDataset

from task302_issue31_gate0_wrapper import EncoderRegHRQVAE


def parse_args():
    parser = argparse.ArgumentParser(description='Task #302 / Issue #31 train_hrqvae_encoder_reg')

    # baseline args (跟 task301 一致)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100, help='Issue #31 Gate 1 = 100 epoch')
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
    parser.add_argument('--beta', type=float, default=0.5, help='baseline beta (被 beta_list 覆盖)')
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--save_limit', type=int, default=1, help='R12: 只保留最新')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/Instruments_encoder_reg')
    parser.add_argument('--seed', type=int, default=42)

    # Issue #31 专属参数 (per-layer 异构 encoder regularization)
    parser.add_argument('--beta_list', type=float, nargs='+', default=[0.1, 0.3, 0.5],
                        help='per-layer commit loss 权重. 默认 [0.1, 0.3, 0.5] (L0 commit loss 降权).')
    parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.01, 0.005, 0.001],
                        help='per-layer codebook center anchor. 默认 [0.01, 0.005, 0.001].')
    parser.add_argument('--gamma_list', type=float, nargs='+', default=[0.001, 0.0005, 0.0001],
                        help='per-layer encoder L2 正则. 默认 [0.001, 0.0005, 0.0001].')
    parser.add_argument('--radii', type=float, nargs='+', default=None,
                        help='(可选) per-layer radii, 留 None 用 baseline c=1.')
    parser.add_argument('--c_k_range_list', type=str, default='1.0:5.0,0.5:20.0,0.5:20.0',
                        help='per-layer c_k range (Issue #31 §D, 沿用 task242 Arm A). 格式 "low1:high1,...".')
    parser.add_argument('--c_k_seed', type=int, default=42)
    parser.add_argument('--curvature_list', type=float, nargs='+', default=None,
                        help='(可选) per-layer 固定 c_k, 默认 None 用 c_k_range 随机.')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    c_k_range_list = [tuple(map(float, r.split(':'))) for r in args.c_k_range_list.split(',')]
    n_layers = len(args.num_emb_list)
    assert len(c_k_range_list) == n_layers, \
        f"c_k_range_list len {len(c_k_range_list)} != num_emb_list len {n_layers}"
    assert len(args.beta_list) == n_layers, \
        f"beta_list len {len(args.beta_list)} != n_layers {n_layers}"
    assert len(args.alpha_list) == n_layers, \
        f"alpha_list len {len(args.alpha_list)} != n_layers {n_layers}"
    assert len(args.gamma_list) == n_layers, \
        f"gamma_list len {len(args.gamma_list)} != n_layers {n_layers}"

    print(f'[Issue #31] beta_list = {args.beta_list}')
    print(f'[Issue #31] alpha_list = {args.alpha_list}')
    print(f'[Issue #31] gamma_list = {args.gamma_list}')
    print(f'[Issue #31] c_k_range_list = {c_k_range_list}')

    # baseline 数据 + EncoderRegHRQVAE wrapper
    data = EmbDataset(args.data_path)
    print(f'[Issue #31] Dataset: {len(data)} items, dim={data.dim}')

    # 兼容 baseline trainer: 不传 curvature_list (baseline HRQVAE 不接受), 改用 wrapper 内的 c_k_range_list 逻辑.
    # baseline 训练时的 c_k_range 通过外部 wrapper 处理 (后续 task 可加 per-layer curvature_list patch).
    # 当前实现: 暂不传 curvature_list (baseline HRQVAE 兼容), per-layer β_l + α_l + γ_l 是核心改动.
    model = EncoderRegHRQVAE(
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
        beta_list=args.beta_list,
        alpha_list=args.alpha_list,
        gamma_list=args.gamma_list,
    )

    # baseline Trainer 沿用, 加 R12 save_limit=1
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f'[Issue #31] Best Loss = {best_loss}, Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()