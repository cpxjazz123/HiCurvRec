#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #174 Stage 1 — MCKG门控融合HRQVAE (用户 design §5 D 臂).

D 臂 = M=2 components + 门控网络 w_m = softmax(MLP(x))_m
     dist(x, c) = Σ_m w_m(x, c) · d_{κ_m}(x_m, c_m)

严格保留 κ-Stereographic 距离公式 (满足用户 hard constraint C1).
只改变距离合成机制 (A/B/C 臂用 baseline sqrt(sum sq), D 臂用 gating).

Diff vs task89:
  - import MCKGGatingHRQVAE instead of FreeCurvHRQVAE
  - new flags: --gating_enabled (True), --gate_hidden (32), --M (强制 2)
  - --theta_init_list: 2 values per layer (跟 M=2 对齐, 每层独立)
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

from model.hrqvae_mckg_gating import MCKGGatingHRQVAE
from model.utils import EmbDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gating_enabled', action='store_true', default=True,
                   help='Task #174 D 臂: enable gating fusion. Disable → baseline sqrt(sum sq) (B 臂).')
    p.add_argument('--gate_hidden', type=int, default=32,
                   help='D 臂 gating MLP hidden size.')
    p.add_argument('--M', type=int, default=2,
                   help='D 臂 M=2 (用户 design §5). M=1 → 退化到 A 臂.')
    p.add_argument('--kappa_max', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_theta', type=float, default=5e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128])
    p.add_argument('--loss_type', default='poincare', choices=['poincare', 'mse'])
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--quant_loss_weight', type=float, default=1.0)
    p.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    p.add_argument('--sk_iters', type=int, default=50)
    p.add_argument('--kmeans_init', action='store_true', default=True)
    p.add_argument('--kmeans_iters', type=int, default=1000)
    p.add_argument('--theta_init', type=float, default=0.0)
    p.add_argument('--theta_init_list', type=float, nargs='+', default=None,
                   help='per-layer θ_m init, length must equal num_emb_list layers.')
    p.add_argument('--geodesic_kmeans', action='store_true', default=False)
    p.add_argument('--re_kmeans_every', type=int, default=50)
    p.add_argument('--dead_code_reset_every', type=int, default=0)
    p.add_argument('--dead_code_reset_threshold', type=float, default=0.0)
    p.add_argument('--dead_code_replace_ratio', type=float, default=0.1)
    # Task #144: κ+codebook 解耦训练
    p.add_argument('--kappa_freeze_epochs', type=int, default=100,
                   help='Phase A length. D 臂默认 100 (跟 #169-#172 一致).')
    p.add_argument('--lr_theta_post_unfreeze', type=float, default=1e-5)
    p.add_argument('--utilization_freeze_threshold', type=float, default=0.05)
    p.add_argument('--phase_a_baseline_util_path', type=str, default=None)
    p.add_argument('--soft_vq_enabled', action='store_true')
    p.add_argument('--soft_vq_tau_start', type=float, default=1.0)
    p.add_argument('--soft_vq_tau_end', type=float, default=0.01)
    p.add_argument('--soft_vq_anneal_type', choices=['linear', 'cosine'], default='linear')
    p.add_argument('--data_path',
                   default='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_every', type=int, default=200)
    p.add_argument('--kappa_log_path', required=True)
    return p.parse_args()


@torch.no_grad()
def _per_layer_residuals(model, latents):
    residual = latents
    res_list = []
    for vq in model.hrq.vq_layers:
        res_list.append(residual)
        d = vq._per_component_dist_sq(residual, vq.embeddings.weight)
        idx = torch.argmin(d, dim=-1)
        x_q = vq.embeddings.weight.index_select(0, idx)
        residual = residual - x_q
    return res_list


def main():
    args = parse_args()
    mode = "D 臂 (gating ON)" if args.gating_enabled else "B 臂 fallback (gating OFF)"
    print(f"[Task #174 Stage 1] MCKGGatingHRQVAE training ({mode}, M={args.M}, "
          f"gate_hidden={args.gate_hidden}, κ_max={args.kappa_max})")
    print(f"  ckpt_dir: {args.ckpt_dir}")
    print(f"  data: {args.data_path}")
    print(f"  device: {args.device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    dataset = EmbDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    print(f"  dataset: {len(dataset)} items, dim={dataset.dim}")

    model = MCKGGatingHRQVAE(
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
        soft_vq_enabled=args.soft_vq_enabled,
        soft_vq_tau_start=args.soft_vq_tau_start,
        soft_vq_tau_end=args.soft_vq_tau_end,
        soft_vq_anneal_type=args.soft_vq_anneal_type,
        use_gating=args.gating_enabled,
        gate_hidden=args.gate_hidden,
    ).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters())}")
    print(f"  num codebook layers: {len(model.hrq.vq_layers)}")
    if args.gating_enabled:
        gate_params = sum(p.numel() for vq in model.hrq.vq_layers for p in vq.gate_net.parameters())
        print(f"  D 臂 gating params total: {gate_params} "
              f"({gate_params // len(model.hrq.vq_layers)} per layer)")

    if args.theta_init_list is not None:
        n_layers = len(model.hrq.vq_layers)
        if len(args.theta_init_list) != n_layers:
            raise ValueError(
                f"[Task #174 theta_init_list] got {len(args.theta_init_list)} values "
                f"but model has {n_layers} layers. Must match."
            )
        with torch.no_grad():
            for li, vq in enumerate(model.hrq.vq_layers):
                vq.theta_m.data = torch.full(
                    (args.M,), args.theta_init_list[li], dtype=torch.float32, device=device
                )
        kappa_inits = [args.kappa_max * np.tanh(t) for t in args.theta_init_list]
        print(f"  per-layer θ_m init (diverse): {args.theta_init_list} "
              f"→ κ_m={['%.4f' % k for k in kappa_inits]}")
    elif args.theta_init != 0.0:
        with torch.no_grad():
            for vq in model.hrq.vq_layers:
                vq.theta_m.data = torch.full(
                    (args.M,), args.theta_init, dtype=torch.float32, device=device
                )
        kappa_init = (args.kappa_max * np.tanh(args.theta_init))
        print(f"  per-component θ_m init: {[args.theta_init]*args.M} → κ_m={kappa_init:.6f}")
    else:
        print(f"  per-component θ_m init (default zeros): {[0.0]*args.M}")

    codebook_params, theta_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if 'embeddings' in name:
            codebook_params.append(p)
        elif 'theta_m' in name:
            theta_params.append(p)
        else:
            other_params.append(p)
    print(f"  param groups: codebook={len(codebook_params)}, theta={len(theta_params)}, "
          f"other={len(other_params)}")

    optim = torch.optim.Adam([
        {'params': codebook_params + other_params, 'lr': args.lr},
        {'params': theta_params, 'lr': args.lr_theta},
    ], weight_decay=args.weight_decay)

    kappa_history = {
        'M': args.M,
        'kappa_max': args.kappa_max,
        'gating_enabled': args.gating_enabled,
        'gate_hidden': args.gate_hidden,
        'epochs': [],
        'per_layer_per_m': [],
    }
    def log_kappa(epoch):
        per_layer = []
        for li, vq in enumerate(model.hrq.vq_layers):
            kappas = vq.kappa_m().detach().cpu().tolist()
            per_layer.append([li, kappas])
        kappa_history['epochs'].append(epoch)
        kappa_history['per_layer_per_m'].append(per_layer)

    model.train()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_loss = float('inf')
    t_start = time.time()

    num_layers = len(model.hrq.vq_layers)
    usage_accum = [torch.zeros(vq.n_e, dtype=torch.long, device=device)
                   for vq in model.hrq.vq_layers]
    epoch_usage = [torch.zeros(vq.n_e, dtype=torch.long, device=device)
                   for vq in model.hrq.vq_layers]

    kappa_decouple_enabled = args.kappa_freeze_epochs > 0
    phase_a_done = False
    kappa_frozen_for_good = False
    phase_a_baseline_util = None

    if kappa_decouple_enabled:
        for pg in optim.param_groups:
            if pg['params'] == theta_params:
                pg['lr'] = 0.0
        print(f"  [Phase A] κ FROZEN at 0 for first {args.kappa_freeze_epochs} epochs.")
    else:
        print(f"  [κ-decouple disabled] κ learned freely.")

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

            if not torch.isfinite(total_loss):
                raise ValueError(
                    f"[NaN guard] ep{epoch} batch{n_batches}: loss={total_loss.item()}, "
                    f"kappa={[vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]}"
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            for li, vq in enumerate(model.hrq.vq_layers):
                _, uc = vq.get_codebook_usage(indices[..., li])
                usage_accum[li] = usage_accum[li] + uc
                epoch_usage[li] = epoch_usage[li] + uc

            if args.dead_code_reset_every > 0 and (n_batches + 1) % args.dead_code_reset_every == 0:
                with torch.no_grad():
                    z_cur = model.encoder(x)
                    res_list = _per_layer_residuals(model, z_cur)
                n_reset_total = 0
                for li, vq in enumerate(model.hrq.vq_layers):
                    n_reset = vq.dead_code_reset(
                        usage_accum[li], res_list[li],
                        replace_ratio=args.dead_code_replace_ratio,
                        threshold=args.dead_code_reset_threshold,
                    )
                    n_reset_total += n_reset
                    usage_accum[li] = torch.zeros(vq.n_e, dtype=torch.long, device=device)
                if n_reset_total > 0:
                    print(f"    [dead_code] ep{epoch} batch{n_batches+1}: reset {n_reset_total}")

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_quant += quant_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_quant = epoch_quant / n_batches

        epoch_util_per_layer = []
        for li, vq in enumerate(model.hrq.vq_layers):
            used_codes = (epoch_usage[li] > 0).sum().item()
            util_frac = used_codes / vq.n_e
            epoch_util_per_layer.append(util_frac)
            epoch_usage[li] = torch.zeros(vq.n_e, dtype=torch.long, device=device)
        util_str = " ".join(f"L{li}u={u:.2%}" for li, u in enumerate(epoch_util_per_layer))

        if epoch % args.log_interval == 0 or epoch == 1:
            elapsed = time.time() - t_start
            kappa_str = ""
            for li, vq in enumerate(model.hrq.vq_layers):
                kappas = vq.kappa_m().detach().cpu().tolist()
                kappa_str += f"L{li}κ=[{','.join(f'{k:+.4f}' for k in kappas)}] "
            gate_w_str = ""
            if args.gating_enabled and epoch % (args.log_interval * 5) == 0:
                # Sample gating weights distribution every 50 epochs
                with torch.no_grad():
                    z0 = model.encoder(x)
                    w = model.hrq.vq_layers[0].get_gating_weights(z0)
                    gate_w_str = f" L0w_mean=[{','.join(f'{v:.3f}' for v in w.mean(0).tolist())}]"
            print(f"  ep{epoch:4d}/{args.epochs} loss={avg_loss:.4f} recon={avg_recon:.4f} "
                  f"quant={avg_quant:.4f} {kappa_str}{util_str}{gate_w_str} ({elapsed:.1f}s)")
            log_kappa(epoch)

        if kappa_decouple_enabled and not phase_a_done and epoch == args.kappa_freeze_epochs:
            phase_a_baseline_util = list(epoch_util_per_layer)
            phase_a_done = True
            if args.phase_a_baseline_util_path:
                with open(args.phase_a_baseline_util_path, 'w') as f:
                    json.dump({
                        'phase_a_end_epoch': epoch,
                        'per_layer_baseline_util': phase_a_baseline_util,
                        'num_layers': num_layers,
                    }, f, indent=2)
            print(f"  [Phase A END] ep{epoch}: baseline util = {phase_a_baseline_util}")

        if kappa_decouple_enabled and phase_a_done and not kappa_frozen_for_good \
                and epoch == args.kappa_freeze_epochs + 1:
            for pg in optim.param_groups:
                if pg['params'] == theta_params:
                    pg['lr'] = args.lr_theta_post_unfreeze
            print(f"  [Phase B START] ep{epoch}: κ UNFROZEN, lr_theta={args.lr_theta_post_unfreeze}")

        if kappa_decouple_enabled and phase_a_done and not kappa_frozen_for_good \
                and phase_a_baseline_util is not None:
            for li, util_now in enumerate(epoch_util_per_layer):
                util_baseline = phase_a_baseline_util[li]
                util_drop = util_baseline - util_now
                if util_drop > args.utilization_freeze_threshold:
                    for pg in optim.param_groups:
                        if pg['params'] == theta_params:
                            pg['lr'] = 0.0
                    kappa_frozen_for_good = True
                    print(f"  [FREEZE-ON-COLLAPSE] ep{epoch} L{li}: util {util_baseline:.2%} → {util_now:.2%}")
                    break

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

        if epoch % args.save_every == 0:
            with open(args.kappa_log_path, 'w') as f:
                json.dump(kappa_history, f, indent=2)

        if args.soft_vq_enabled:
            model.hrq.update_tau(epoch, args.epochs)

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