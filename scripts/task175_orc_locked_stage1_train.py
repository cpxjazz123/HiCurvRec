#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #175 Stage 1 — κ-Stereographic + κ LOCKED at Ollivier ORC 实测值.

Hypothesis (用户 2026-07-25): 让 κ_m 学 → 学回 ≈ 0. 强制锁定在
真实 ORC 几何 (-0.65 ~ -0.84), codebook 在 hyperbolic 空间聚类,
下游 T5 能否超 baseline 0.1058.

Diff vs task174 (D 臂):
  - import ORCLockedHRQVAE instead of MCKGGatingHRQVAE
  - theta_m frozen (requires_grad=False) via ORCLockedVectorQuantization
  - κ_m 永远 = LOCKED ORC tensor, override kappa_m()
  - single Phase 200 epoch (no Phase A/B split, 因为 κ 不需要 freeze→unfreeze)
  - new flag: --kappa_orc_values (3 floats: -0.653 -0.829 -0.840)

C1 硬约束保留: κ-Stereographic 距离公式 (Berman-Metzler 2020 Eq 3),
NOT L2. 距离公式实现从 FreeCurvVectorQuantization 继承, 没改.
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

from model.hrqvae_orc_locked import ORCLockedHRQVAE
from model.utils import EmbDataset


def parse_args():
    p = argparse.ArgumentParser()
    # ORC LOCKED specific
    p.add_argument('--kappa_orc_values', type=float, nargs='+',
                   default=[-0.653, -0.829, -0.840],
                   help='Per-component ORC κ values to LOCK. Length must == M.')
    # Standard FreeCurvHRQVAE args (跟 task174 对齐, 但移除 gating/M=2 特有)
    p.add_argument('--M', type=int, default=3,
                   help='M=3 → e_dim 32 / 3 = 11+11+10 split per ORC value.')
    p.add_argument('--kappa_max', type=float, default=2.0,
                   help='κ_max 重参数化上限. LOCKED 时仅用于初始化 theta_m.')
    p.add_argument('--epochs', type=int, default=200,
                   help='Single Phase (no A/B split).')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_emb_list', type=int, nargs='+',
                   default=[32, 64, 256, 1],
                   help='Stage 2 SID 4 cols (跟 baseline 一致).')
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128])
    p.add_argument('--loss_type', default='poincare', choices=['poincare', 'mse'])
    p.add_argument('--beta', type=float, default=1.0)
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


def build_orc_locked_per_layer(num_layers: int, M: int, orc_values: List[float]) -> List[List[float]]:
    """Per-layer per-component ORC κ. All 4 layers get same config (per-layer 独立).

    For Stage 2 SID 4-col codebook: [32, 64, 256, 1]. L0/L1/L2 learn, L3 dummy (n_e=1).
    ORC values length = M (e.g. M=3 → 3 values per layer).
    """
    assert len(orc_values) == M, \
        f"--kappa_orc_values length {len(orc_values)} != M {M}"
    # Same config for all 4 layers (per-layer 独立 but 同样的 ORC 配置)
    return [list(orc_values) for _ in range(num_layers)]


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Build kappa_locked_per_layer for all 4 layers
    n_layers = len(args.num_emb_list)
    kappa_locked_per_layer = build_orc_locked_per_layer(
        n_layers, args.M, args.kappa_orc_values
    )

    print(f"[Task #175 Stage 1] ORC LOCKED κ config:")
    for li, kappas in enumerate(kappa_locked_per_layer):
        print(f"  L{li}: κ = {kappas}")

    # Load data
    dataset = EmbDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, drop_last=True)
    print(f"[Task #175 Stage 1] dataset size = {len(dataset)}, "
          f"batch_size = {args.batch_size}, num_batches = {len(dataloader)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build ORC-locked HRQVAE
    model = ORCLockedHRQVAE(
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
        kappa_locked_per_layer=kappa_locked_per_layer,
    ).to(device)

    # Verify κ LOCKED
    print(f"[Task #175 Stage 1] Verify κ LOCKED:")
    for li, vq in enumerate(model.hrq.vq_layers):
        kappas = vq.kappa_m().detach().cpu().tolist()
        print(f"  L{li}: κ = {kappas}  (theta_m requires_grad = {vq.theta_m.requires_grad})")

    # Optimizer: only trainable params (theta_m is frozen, won't appear)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Task #175 Stage 1] trainable params = {n_trainable:,} / {n_total:,} "
          f"(θ_m FROZEN → not in trainable)")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Kappa log
    kappa_log = {
        'config': {
            'task': 'task175',
            'M': args.M,
            'kappa_max': args.kappa_max,
            'kappa_orc_values_locked': args.kappa_orc_values,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
        },
        'history': [],
    }

    # R12 force-save: keep only 1 best ckpt
    best_loss = float('inf')
    best_ckpt_path = os.path.join(args.ckpt_dir, 'best_loss_model.pth')

    # Training loop (single Phase, κ LOCKED 全程)
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

        # R12 force-save (delete old best ckpt before saving new one)
        if ep_loss < best_loss:
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_loss = ep_loss
            torch.save(model.state_dict(), best_ckpt_path)

        # Log κ per epoch (LOCKED values constant)
        kappas_per_layer = model.get_kappa_history()
        kappa_log['history'].append({
            'epoch': epoch,
            'loss': ep_loss,
            'recon_loss': ep_recon,
            'quant_loss': ep_quant,
            'kappa_per_layer': kappas_per_layer,
        })

        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            print(f"[Task #175 Stage 1] ep{epoch:3d}/{args.epochs} "
                  f"loss={ep_loss:.4f} recon={ep_recon:.4f} quant={ep_quant:.4f} "
                  f"κ_L0={kappas_per_layer[0]} best_loss={best_loss:.4f}")

        # Periodic ckpt save per R12 (each save_every epochs)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            # Save snapshot for debugging (DO NOT replace best)
            snapshot_path = os.path.join(args.ckpt_dir, f'snapshot_ep{epoch}.pth')
            torch.save(model.state_dict(), snapshot_path)
            print(f"[Task #175 Stage 1] saved snapshot_ep{epoch}.pth")

    # Final save (replace best with final model — should already be best since we
    # tracked best_loss every epoch)
    if not os.path.exists(best_ckpt_path):
        torch.save(model.state_dict(), best_ckpt_path)
    print(f"[Task #175 Stage 1] ✅ Training done. Best ckpt: {best_ckpt_path} "
          f"(loss={best_loss:.4f})")

    # Save final κ + log
    with open(args.kappa_log_path, 'w') as f:
        json.dump(kappa_log, f, indent=2)
    print(f"[Task #175 Stage 1] ✅ κ log saved: {args.kappa_log_path}")


if __name__ == '__main__':
    main()