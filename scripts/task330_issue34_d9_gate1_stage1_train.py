#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #330 / Issue #34 D9 — Gate 1 Stage 1 训练 wrapper (100 epoch)

背景 (Issue #34 body Gate 1):
  Stage 1 100 epoch 训练 (per-layer Codebook Transforms + per-layer 异构 hash 函数族):
    - r_l = [0.1, 1.0, 10.0]       (Issue #30 GO 端点)
    - s_l = [2.0, 2.0, 2.0]       (Issue #30 GO 端点)
    - R_l = I identity            (Issue #30 GO 端点)
    - hash_top_k_list = [3, 5, 7] (Issue #34 新增)
    - κ 不 frozen (baseline 默认)

通过条件 (Issue #34 §Gate 1, 6 条全过):
  (a) L0 utilization ≥ 90% at any evaluation step ≥ ep50
  (b) L1 utilization ≥ 90% at any evaluation step ≥ ep50
  (c) L2 utilization ≥ 90% at any evaluation step ≥ ep50
  (d) collision_rate ≤ 0.25
  (e) norm 健康区: ‖x‖_E ∈ [0.7, 0.95] for L0/L1/L2 at any evaluation step ≥ ep50
  (f) hash candidates 有效性: L0 top-3 unique, L1 top-5 unique, L2 top-7 unique

实施方式 (R11.4 + R11.5):
  - 不修改 HG-Rec/model/ 上游源码
  - 复用 Issue #30 PerLayerCodebookTransformHRQVAE wrapper + Issue #34 D9 hash 模块
  - 训练 loop = baseline trainer.fit(), 加载 hash modules 在每层 forward 后
  - hard argmin commitment 保留 (与 #30 一致, 不引入 expected-loss)
"""
import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
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
    (复用 Issue #30 实施)
    """
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

        print(f'[Issue #34 D9] Layer {li}: r_l={r}, s_l={s}, R_l=I, weight_norm_after={q.embeddings.weight.norm().item():.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Task #330 / Issue #34 D9 train_hrqvae_perlayer_hash')

    # baseline args (跟 Issue #30 一致)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100, help='Issue #34 Gate 1 = 100 epoch')
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
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/Instruments_perlayer_hash')
    parser.add_argument('--seed', type=int, default=42)

    # Issue #30 专属参数 (Issue #34 沿用)
    parser.add_argument('--radius_list', type=float, nargs='+', default=[0.1, 1.0, 10.0])
    parser.add_argument('--scale_list', type=float, nargs='+', default=[2.0, 2.0, 2.0])

    # Issue #34 D9 专属参数
    parser.add_argument('--hash_top_k_list', type=int, nargs='+', default=[3, 5, 7],
                        help='per-layer hash top-k candidates. L0=3, L1=5, L2=7.')
    parser.add_argument('--hash_seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert len(args.radius_list) == len(args.num_emb_list)
    assert len(args.scale_list) == len(args.num_emb_list)
    assert len(args.hash_top_k_list) == len(args.num_emb_list), \
        f"hash_top_k_list len {len(args.hash_top_k_list)} != num_emb_list len {len(args.num_emb_list)}"

    print(f'[Issue #34 D9] radius_list = {args.radius_list}')
    print(f'[Issue #34 D9] scale_list = {args.scale_list}')
    print(f'[Issue #34 D9] hash_top_k_list = {args.hash_top_k_list}')

    # baseline HRQVAE 构造
    data = EmbDataset(args.data_path)
    print(f'[Issue #34 D9] Dataset: {len(data)} items, dim={data.dim}')

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
        curvature_list=None,
        euclidean_qloss=False,
        loss_mult_codebook=args.loss_mult_codebook,
        radii=args.radii,
        c_k_range_list=None,  # Issue #34 D9 不引入 c_k range (沿用 #30 但跟 #32 区分)
        c_k_seed=args.hash_seed,
        assignment_mode='shared',
    )

    # 应用 per-layer codebook transforms (Issue #30 核心, Issue #34 沿用)
    rotation_list = build_identity_rotation_list(args.num_emb_list, args.e_dim)
    print(f'[Issue #34 D9] Apply per-layer codebook transforms: r_l={args.radius_list}, s_l={args.scale_list}, R_l=I')
    apply_per_layer_codebook_transforms(
        model=model,
        radius_list=args.radius_list,
        rotation_list=rotation_list,
        scale_list=args.scale_list,
    )

    # Issue #34 D9 新增: 初始化 per-layer hash modules
    print(f'[Issue #34 D9] Initializing per-layer hash modules...')
    from task330_issue34_d9_gate0_perlayer_hash import (
        SparseRandomProjectionHash, LSHMultiProbe, KMeansMultiBucket
    )
    hash_modules = []
    for li, K in enumerate(args.num_emb_list):
        top_k = args.hash_top_k_list[li]
        cb_weight = model.hrq.vq_layers[li].embeddings.weight.data.cpu().numpy()
        if li == 0:
            mod = SparseRandomProjectionHash(K, args.e_dim, top_k=top_k, n_proj=8)
        elif li == 1:
            mod = LSHMultiProbe(K, args.e_dim, top_k=top_k, n_tables=16, n_probes=3)
        elif li == 2:
            mod = KMeansMultiBucket(K, args.e_dim, top_k=top_k, n_buckets=8)
        mod.fit(cb_weight)
        hash_modules.append(mod)
        print(f'[Issue #34 D9] Layer {li} hash module initialized: top_k={top_k}, K={K}')

    # baseline Trainer 沿用, 加 R12 save_limit=1
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f'[Issue #34 D9] Best Loss = {best_loss}, Best Collision Rate = {best_collision_rate}')

    # Issue #34 D9 Gate 1 (e) norm 健康区验证: 在 ckpt 落盘后, 计算每层 norm
    print(f'[Issue #34 D9] Gate 1 (e) norm 健康区验证:')
    for li, q in enumerate(model.hrq.vq_layers):
        w_norm = q.embeddings.weight.norm().item()
        # Issue #34 spec: ‖x‖_E ∈ [0.7, 0.95]
        norm_healthy = 0.7 <= (w_norm / np.sqrt(q.embeddings.weight.shape[-1])) <= 0.95
        print(f'  Layer {li}: weight_norm={w_norm:.4f}, mean_dim_norm={w_norm/np.sqrt(q.embeddings.weight.shape[-1]):.4f}, healthy={norm_healthy}')

    # Issue #34 D9 Gate 1 (f) hash candidates 验证: 对 batch input 计算 hash candidates unique count
    print(f'[Issue #34 D9] Gate 1 (f) hash candidates 验证:')
    model.eval()
    sample = torch.stack([data[i] for i in range(16)]).float().to(args.device)
    with torch.no_grad():
        x_enc = model.encoder(sample)
        residual = x_enc
        for li, q in enumerate(model.hrq.vq_layers):
            x_res, _, _ = q(residual, use_sk=False)
            residual = residual - x_res
            # Apply hash to current residual
            residual_np = residual.cpu().numpy()
            top_k_idx = hash_modules[li].query(residual_np)  # (B, top_k)
            unique_per_item = [len(set(top_k_idx[i].tolist())) for i in range(top_k_idx.shape[0])]
            min_unique = min(unique_per_item)
            top_k_expected = args.hash_top_k_list[li]
            hash_valid = min_unique >= top_k_expected
            print(f'  Layer {li}: top_k={top_k_expected}, per-item unique min={min_unique}, valid={hash_valid}')

    print('[Issue #34 D9] Gate 1 training done.')


if __name__ == '__main__':
    main()