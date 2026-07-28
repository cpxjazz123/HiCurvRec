#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #176 + #177 Stage 1 — κ-Stereographic + 位置依赖 β(x).

Task #176 (sigmoid MCKG-style): β(x) = β_base · sigmoid(-sign(κ)·||x||²·scale)
Task #177 (conformal factor): β(x) = β_base · (1+κ·||x||²)/2, clamp [0, β_max]

共享训练 script, --beta_mode 控制用哪个 variant.

C1 硬约束保留: κ-Stereographic 距离公式 (Berman-Metzler 2020 Eq 3), NOT L2.
Distance formula 从 FreeCurvVectorQuantization 继承, 没改. 只 override
commitment/codebook loss 的 β weight.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from torch.utils.data import DataLoader

from model.hrqvae_posdep_beta import PosDepBetaHRQVAE
from model.utils import EmbDataset


def parse_args():
    p = argparse.ArgumentParser()
    # Position-dependent β hyperparams
    p.add_argument('--beta_mode', choices=['sigmoid', 'conformal'], required=True,
                   help='sigmoid (Task #176 MCKG) | conformal (Task #177 Riemannian)')
    p.add_argument('--beta_base', type=float, default=1.0)
    p.add_argument('--beta_scale', type=float, default=10.0,
                   help='Only used for sigmoid mode')
    p.add_argument('--beta_max', type=float, default=5.0,
                   help='Only used for conformal mode (clamp to avoid explosion)')
    # Standard FreeCurvHRQVAE args
    p.add_argument('--M', type=int, default=3)
    p.add_argument('--kappa_max', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_emb_list', type=int, nargs='+',
                   default=[32, 64, 256, 1])
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128])
    p.add_argument('--loss_type', default='poincare', choices=['poincare', 'mse'])
    p.add_argument('--beta', type=float, default=1.0,
                   help='Legacy fixed β for commitment loss weighting (commitment_sq uses this)')
    p.add_argument('--quant_loss_weight', type=float, default=1.0)
    p.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0])
    p.add_argument('--sk_iters', type=int, default=50)
    p.add_argument('--kmeans_init', action='store_true', default=True)
    p.add_argument('--kmeans_iters', type=int, default=1000)
    p.add_argument('--dead_code_reset_every', type=int, default=20)
    p.add_argument('--dead_code_reset_threshold', type=float, default=0.0)
    p.add_argument('--dead_code_replace_ratio', type=float, default=0.1)
    p.add_argument('--data_path',
                   default='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--kappa_log_path', required=True)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"[Task #176/#177 Stage 1] β_mode = {args.beta_mode}")
    if args.beta_mode == 'sigmoid':
        print(f"  β(x) = {args.beta_base} · sigmoid(-sign(κ)·||x||²·{args.beta_scale})")
    else:
        print(f"  β(x) = {args.beta_base} · (1+κ·||x||²)/2, clamp [0, {args.beta_max}]")

    # Load data
    dataset = EmbDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, drop_last=True)
    print(f"  dataset size = {len(dataset)}, batch_size = {args.batch_size}, "
          f"num_batches = {len(dataloader)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build PosDepBetaHRQVAE
    model = PosDepBetaHRQVAE(
        in_dim=768,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        M=args.M,
        kappa_max=args.kappa_max,
        layers=args.layers,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        beta_mode=args.beta_mode,
        beta_base=args.beta_base,
        beta_scale=args.beta_scale,
        beta_max=args.beta_max,
    ).to(device)

    # Verify κ initial
    print(f"  κ initial:")
    for li, vq in enumerate(model.hrq.vq_layers):
        kappas = vq.kappa_m().detach().cpu().tolist()
        print(f"    L{li}: κ = {kappas}")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params = {n_trainable:,} / {n_total:,}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Kappa log
    kappa_log = {
        'config': {
            'task': f'task176_or_177_beta_{args.beta_mode}',
            'beta_mode': args.beta_mode,
            'beta_base': args.beta_base,
            'beta_scale': args.beta_scale,
            'beta_max': args.beta_max,
            'M': args.M,
            'kappa_max': args.kappa_max,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
        },
        'history': [],
    }

    # R12 force-save
    best_loss = float('inf')
    best_ckpt_path = os.path.join(args.ckpt_dir, 'best_loss_model.pth')

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        ep_recon = 0.0
        ep_quant = 0.0
        n_batches = 0

        for batch_idx, x in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            out, quant_loss, indices = model(x, use_sk=True)
            total_loss, recon_loss = model.compute_loss(out, quant_loss, xs=x)
            total_loss.backward()
            optimizer.step()
            ep_loss += float(total_loss.item())
            ep_recon += float(recon_loss.item())
            ep_quant += float(quant_loss.item())
            n_batches += 1

        ep_loss /= n_batches
        ep_recon /= n_batches
        ep_quant /= n_batches

        # R12 force-save
        if ep_loss < best_loss:
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_loss = ep_loss
            torch.save(model.state_dict(), best_ckpt_path)

        kappas_per_layer = model.get_kappa_history()
        kappa_log['history'].append({
            'epoch': epoch,
            'loss': ep_loss,
            'recon_loss': ep_recon,
            'quant_loss': ep_quant,
            'kappa_per_layer': kappas_per_layer,
        })

        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            print(f"[Task #176/#177 Stage 1] ep{epoch:3d}/{args.epochs} "
                  f"loss={ep_loss:.4f} recon={ep_recon:.4f} quant={ep_quant:.4f} "
                  f"κ_L0={kappas_per_layer[0]} best_loss={best_loss:.4f}")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            snapshot_path = os.path.join(args.ckpt_dir, f'snapshot_ep{epoch}.pth')
            torch.save(model.state_dict(), snapshot_path)
            print(f"[Task #176/#177 Stage 1] saved snapshot_ep{epoch}.pth")

    if not os.path.exists(best_ckpt_path):
        torch.save(model.state_dict(), best_ckpt_path)
    print(f"✅ Training done. Best ckpt: {best_ckpt_path} (loss={best_loss:.4f})")

    with open(args.kappa_log_path, 'w') as f:
        json.dump(kappa_log, f, indent=2)
    print(f"✅ κ log saved: {args.kappa_log_path}")


if __name__ == '__main__':
    main()