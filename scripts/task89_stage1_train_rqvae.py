#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #89 Stage 1 — FreeCurvHRQVAE training (A/B/C arms).

A (M=1): κ_m free learned, single 32-dim block.
B (M=2): 16+16 split, two independent κ_m.
C (M=3): 11+11+10 split, three independent κ_m.

Trains on Musical_Instruments item_emb.parquet (768d).
Reports per-layer per-component κ_m history every K epochs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from torch.utils.data import DataLoader

from model.hrqvae_free_curv import FreeCurvHRQVAE
from model.utils import EmbDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, required=True, choices=[1, 2, 3],
                   help='M=1 (A 臂), M=2 (B 臂), M=3 (C 臂)')
    p.add_argument('--kappa_max', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_theta', type=float, default=5e-3,
                   help='separate lr for θ_m (κ_m learning rate), per user spec')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128])
    p.add_argument('--loss_type', default='poincare', choices=['poincare', 'mse'])
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--quant_loss_weight', type=float, default=1.0)
    p.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.000])
    p.add_argument('--sk_iters', type=int, default=50)
    p.add_argument('--kmeans_init', action='store_true', default=True)
    p.add_argument('--kmeans_iters', type=int, default=1000)
    p.add_argument('--data_path',
                   default='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ckpt_dir', required=True,
                   help='output dir (e.g. products/task89/train/arm_A_M1)')
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_every', type=int, default=200)
    p.add_argument('--kappa_log_path', required=True,
                   help='JSON path to log per-layer κ_m history')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #89 Stage 1] FreeCurvHRQVAE training (M={args.M}, κ_max={args.kappa_max})")
    print(f"  ckpt_dir: {args.ckpt_dir}")
    print(f"  data: {args.data_path}")
    print(f"  device: {args.device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    # Dataset
    dataset = EmbDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    print(f"  dataset: {len(dataset)} items, dim={dataset.dim}")

    # Model
    model = FreeCurvHRQVAE(
        in_dim=dataset.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        M=args.M,
        kappa_max=args.kappa_max,
        layers=args.layers,
        dropout_prob=0.0,
        bn=False,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    ).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters())}")
    print(f"  num codebook layers: {len(model.hrq.vq_layers)}")
    print(f"  per-component θ_m init (zeros): {[0.0]*args.M}")

    # Optimizer: separate lr for codebook vs θ_m
    codebook_params = []
    theta_params = []
    other_params = []
    for name, p in model.named_parameters():
        if 'embeddings' in name:
            codebook_params.append(p)
        elif 'theta_m' in name:
            theta_params.append(p)
        else:
            other_params.append(p)

    optim = torch.optim.Adam([
        {'params': codebook_params + other_params, 'lr': args.lr},
        {'params': theta_params, 'lr': args.lr_theta},
    ], weight_decay=args.weight_decay)

    # κ_m history
    kappa_history = {
        'M': args.M,
        'kappa_max': args.kappa_max,
        'epochs': [],
        'per_layer_per_m': [],   # list of [layer_idx, [κ_m for m in 0..M]]
    }
    def log_kappa(epoch):
        per_layer = []
        for li, vq in enumerate(model.hrq.vq_layers):
            kappas = vq.kappa_m().detach().cpu().tolist()
            per_layer.append([li, kappas])
        kappa_history['epochs'].append(epoch)
        kappa_history['per_layer_per_m'].append(per_layer)

    # Training loop
    model.train()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_quant = 0.0
        n_batches = 0

        for batch in loader:
            x = batch.to(device, non_blocking=True).float()
            optim.zero_grad()

            out, quant_loss, indices = model(x, use_sk=True)
            total_loss, recon_loss = model.compute_loss(out, quant_loss, xs=x)

            total_loss.backward()
            optim.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_quant += quant_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_quant = epoch_quant / n_batches

        if epoch % args.log_interval == 0 or epoch == 1:
            elapsed = time.time() - t_start
            # Print κ_m per layer
            kappa_str = ""
            for li, vq in enumerate(model.hrq.vq_layers):
                kappas = vq.kappa_m().detach().cpu().tolist()
                kappa_str += f"L{li}κ=[{','.join(f'{k:+.4f}' for k in kappas)}] "
            print(f"  ep{epoch:4d}/{args.epochs} loss={avg_loss:.4f} recon={avg_recon:.4f} "
                  f"quant={avg_quant:.4f} {kappa_str} ({elapsed:.1f}s)")
            log_kappa(epoch)

        # Save ckpt periodically (R12)
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.ckpt_dir, 'best_loss_model.pth')
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            torch.save({
                'state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'best_loss': best_loss,
                'kappa_history': kappa_history,
            }, ckpt_path)
            if epoch % args.log_interval == 0:
                print(f"    [R12] saved best_loss_model.pth (loss={best_loss:.4f})")

        # Always save κ history (so Stage 1 verdict can use it even if interrupted)
        if epoch % args.save_every == 0:
            with open(args.kappa_log_path, 'w') as f:
                json.dump(kappa_history, f, indent=2)

    # Final save
    with open(args.kappa_log_path, 'w') as f:
        json.dump(kappa_history, f, indent=2)
    print(f"\n[Done] Final ckpt: {ckpt_path}, best_loss={best_loss:.4f}")
    print(f"[Done] κ_m history: {args.kappa_log_path}")
    print(f"[Done] Final κ_m values:")
    for li, vq in enumerate(model.hrq.vq_layers):
        kappas = vq.kappa_m().detach().cpu().tolist()
        print(f"    L{li}: κ = [{', '.join(f'{k:+.4f}' for k in kappas)}]")


if __name__ == '__main__':
    main()