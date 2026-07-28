#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #138 — A-arm quick kmeans smoke test (10 epoch, --geodesic_kmeans ON).

Goal: verify --geodesic_kmeans / --re_kmeans_every / --dead_code_reset_every
flags work end-to-end WITHOUT crashing. This is NOT the full 1000-epoch run;
it's a 10-epoch sanity check that:
  1. geodesic_reinit() runs on first batch (no NaN, no crash)
  2. periodic re-kmeans at epoch 5 (re_kmeans_every=5) doesn't crash
  3. dead_code_reset at every 100 batches runs without error
  4. best_loss_model.pth is saved at least once

Output: products/task138/smoke_test/arm_A_M1/
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from torch.utils.data import DataLoader

from model.hrqvae_free_curv import FreeCurvHRQVAE
from model.utils import EmbDataset


def main():
    """10-epoch smoke test of geodesic_kmeans + dead_code_reset flags."""
    CKPT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/products/task138/smoke_test/arm_A_M1'
    KAPPA_LOG = os.path.join(CKPT_DIR, 'kappa_history.json')
    DATA_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Seed (match task137 baseline)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:0')

    # Hyperparams (mirror task137 A-arm baseline + new flags)
    M = 1
    kappa_max = 0.5
    epochs = 10
    batch_size = 256
    lr = 1e-3
    lr_theta = 1e-3
    theta_init = 0.01  # R137: escape Euclidean fixed point
    num_emb_list = [64, 128, 256]
    e_dim = 32
    layers = [512, 256, 128]
    re_kmeans_every = 5   # smoke test: re-kmeans at epoch 5 (only 1 re-kmeans event)
    dead_code_reset_every = 100  # smoke test: roughly 1 reset call total

    print(f"[Task #138 smoke] geodesic_kmeans + dead_code_reset quick test")
    print(f"  ckpt_dir: {CKPT_DIR}")
    print(f"  epochs={epochs}, M={M}, kappa_max={kappa_max}, θ_init={theta_init}")
    print(f"  re_kmeans_every={re_kmeans_every}, dead_code_reset_every={dead_code_reset_every}")

    dataset = EmbDataset(DATA_PATH)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    print(f"  dataset: {len(dataset)} items, dim={dataset.dim}")

    model = FreeCurvHRQVAE(
        in_dim=dataset.dim,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        M=M,
        kappa_max=kappa_max,
        layers=layers,
        dropout_prob=0.0,
        bn=False,
        loss_type='poincare',
        quant_loss_weight=1.0,
        beta=1.0,
        kmeans_init=True,
        kmeans_iters=1000,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=50,
    ).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters())}")

    # R137: override θ_m init to escape Euclidean fixed point
    if theta_init != 0.0:
        with torch.no_grad():
            for vq in model.hrq.vq_layers:
                vq.theta_m.data = torch.full(
                    (M,), theta_init, dtype=torch.float32, device=device
                )
        kappa_init_val = kappa_max * np.tanh(theta_init)
        print(f"  θ_m init={theta_init} → κ_m={kappa_init_val:.6f}")

    # Build optimizer
    codebook_params, theta_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if 'embeddings' in name:
            codebook_params.append(p)
        elif 'theta_m' in name:
            theta_params.append(p)
        else:
            other_params.append(p)
    optim = torch.optim.Adam([
        {'params': codebook_params + other_params, 'lr': lr},
        {'params': theta_params, 'lr': lr_theta},
    ])

    # Import helpers from task89 script
    from task89_stage1_train_rqvae import (
        _collect_encoder_latents,
        _per_layer_residuals,
        geodesic_reinit,
    )

    num_layers = len(model.hrq.vq_layers)
    geo_initialized = False
    global_batch = 0
    usage_accum = [
        torch.zeros(vq.n_e, dtype=torch.long, device=device)
        for vq in model.hrq.vq_layers
    ]
    kappa_history = {'epochs': [], 'per_layer_per_m': [], 'M': M, 'kappa_max': kappa_max}

    t_start = time.time()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            x = batch.to(device, non_blocking=True).float()

            # A方案: lazy geodesic init on first real batch
            if not geo_initialized:
                with torch.no_grad():
                    z0 = model.encoder(x)
                geodesic_reinit(model, z0)
                for vq in model.hrq.vq_layers:
                    vq.initted = True
                geo_initialized = True
                print(f"  [A方案] geodesic init done at epoch {epoch} (first batch)")

            optim.zero_grad()
            out, quant_loss, indices = model(x, use_sk=True)
            total_loss, recon_loss = model.compute_loss(out, quant_loss, xs=x)

            if not torch.isfinite(total_loss):
                raise ValueError(
                    f"[smoke] NaN/Inf loss at epoch {epoch}, batch {n_batches}. "
                    f"kappa={[vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]}"
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            global_batch += 1
            for li, vq in enumerate(model.hrq.vq_layers):
                _, uc = vq.get_codebook_usage(indices[..., li])
                usage_accum[li] = usage_accum[li] + uc

            # B方案: dead_code_reset at every 100 batches
            if global_batch % dead_code_reset_every == 0:
                with torch.no_grad():
                    z_cur = model.encoder(x)
                    res_list = _per_layer_residuals(model, z_cur)
                n_reset_total = 0
                for li, vq in enumerate(model.hrq.vq_layers):
                    n_reset = vq.dead_code_reset(
                        usage_accum[li], res_list[li],
                        replace_ratio=0.1,
                        threshold=0.0,
                    )
                    n_reset_total += n_reset
                    usage_accum[li] = torch.zeros(
                        vq.n_e, dtype=torch.long, device=device
                    )
                if n_reset_total > 0:
                    print(f"    [B方案] batch {global_batch}: reset {n_reset_total} dead codes")

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # A方案: periodic geodesic re-kmeans at epoch % re_kmeans_every == 0
        if epoch % re_kmeans_every == 0:
            latents = _collect_encoder_latents(model, loader, device)
            geodesic_reinit(model, latents)
            for li in range(num_layers):
                usage_accum[li] = torch.zeros(
                    model.hrq.vq_layers[li].n_e, dtype=torch.long, device=device
                )
            print(f"  [A方案] re-kmeans at epoch {epoch} (on {latents.shape[0]} latents)")

        elapsed = time.time() - t_start
        kappa_str = ""
        for li, vq in enumerate(model.hrq.vq_layers):
            kappas = vq.kappa_m().detach().cpu().tolist()
            kappa_str += f"L{li}κ=[{','.join(f'{k:+.4f}' for k in kappas)}] "
        print(f"  ep{epoch:4d}/{epochs} loss={avg_loss:.4f} {kappa_str} ({elapsed:.1f}s)")

        # Record κ_m
        per_layer = []
        for li, vq in enumerate(model.hrq.vq_layers):
            kappas = vq.kappa_m().detach().cpu().tolist()
            per_layer.append([li, kappas])
        kappa_history['epochs'].append(epoch)
        kappa_history['per_layer_per_m'].append(per_layer)

        # R12: save best ckpt
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(CKPT_DIR, 'best_loss_model.pth')
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'kappa_history': kappa_history,
                'args': {
                    'M': M, 'kappa_max': kappa_max, 'theta_init': theta_init,
                    'num_emb_list': num_emb_list, 'e_dim': e_dim, 'layers': layers,
                    'lr': lr, 'lr_theta': lr_theta,
                    'geodesic_kmeans': True,
                    're_kmeans_every': re_kmeans_every,
                    'dead_code_reset_every': dead_code_reset_every,
                },
            }, ckpt_path)
            print(f"    [R12] saved best_loss_model.pth (loss={best_loss:.4f})")

    # Save kappa history
    with open(KAPPA_LOG, 'w') as f:
        json.dump(kappa_history, f, indent=2)

    print(f"\n[smoke OK] 10 epochs done in {time.time() - t_start:.1f}s")
    print(f"  best_loss={best_loss:.4f}")
    print(f"  ckpt: {CKPT_DIR}/best_loss_model.pth")
    print(f"  κ_history: {KAPPA_LOG}")
    print(f"  Final κ_m:")
    for li, vq in enumerate(model.hrq.vq_layers):
        kappas = vq.kappa_m().detach().cpu().tolist()
        print(f"    L{li}: κ = [{', '.join(f'{k:+.4f}' for k in kappas)}]")
    print("\n[smoke PASS] --geodesic_kmeans / --re_kmeans_every / --dead_code_reset_every all work.")


if __name__ == '__main__':
    main()