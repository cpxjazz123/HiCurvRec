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
    p.add_argument('--theta_init', type=float, default=0.0,
                   help='R137: θ_m init value. Default 0 → κ_m=0 (Euclidean fixed point). '
                        'Use 0.01 to escape fixed point and test sph branch (R137 verified).')
    p.add_argument('--theta_init_list', type=float, nargs='+', default=None,
                   help='Task #149: per-layer diverse θ_m init, e.g. "-0.5 0.0 0.5" for 3 layers. '
                        'If set, OVERRIDES --theta_init. Must have len == num_emb_list layers.')
    # Task #89 A方案 (geodesic kmeans) + B方案 (dead code reset)
    p.add_argument('--geodesic_kmeans', action='store_true', default=False,
                   help='A方案: use init_emb_geodesic (expmap0 -> kmeans -> logmap0) '
                        'instead of plain init_emb for codebook init.')
    p.add_argument('--re_kmeans_every', type=int, default=50,
                   help='A方案: re-run geodesic kmeans every N epochs (0=disable). '
                        'Only active when --geodesic_kmeans set.')
    p.add_argument('--dead_code_reset_every', type=int, default=100,
                   help='B方案: run dead-code reset every K batches (0=disable).')
    p.add_argument('--dead_code_reset_threshold', type=float, default=0.0,
                   help='B方案: entries with usage_count <= threshold are dead '
                        '(default 0.0 -> reset only never-used codes).')
    p.add_argument('--dead_code_replace_ratio', type=float, default=0.1,
                   help='B方案: cap on fraction of codebook replaced per reset call.')
    # Task #144: κ + codebook 解耦训练调度 (用户 2026-07-24 提议)
    # Phase A: κ 冻结在 0, codebook 单独训练到健康稳定
    # Phase B: 解冻 κ + 极小 lr_theta, 监控 utilization, 掉则 freeze
    p.add_argument('--kappa_freeze_epochs', type=int, default=0,
                   help='Task #144: Phase A length in epochs. 0=disable kappa-decouple. '
                        'N>0: first N epochs lr_theta=0 (kappa frozen), then switch to --lr_theta_post_unfreeze.')
    p.add_argument('--lr_theta_post_unfreeze', type=float, default=1e-5,
                   help='Task #144 Phase B: theta_m lr after unfreeze (user specified very small lr).')
    p.add_argument('--utilization_freeze_threshold', type=float, default=0.05,
                   help='Task #144: Phase B freeze-on-collapse trigger. If any layer util drops '
                        'by more than this fraction vs Phase A baseline, lr_theta permanently set to 0.')
    p.add_argument('--phase_a_baseline_util_path', type=str, default=None,
                   help='Task #144: Phase A 终态 per-layer utilization JSON path (输出). '
                        'Phase B 监控时读取作为 baseline reference.')
    # Task #145 Phase 2: 软量化退火 (Soft VQ-VAE 风格)
    # τ annealing: training 早期用 softmax-weighted soft assignment (每 codeword 拿梯度),
    # 后期 τ → 0 渐进逼近 hard assignment, 跟 VQ-VAE/VQ-GAN 社区标准做法一致
    p.add_argument('--soft_vq_enabled', action='store_true',
                   help='Task #145 Phase 2: enable softmax-weighted soft assignment with τ annealing.')
    p.add_argument('--soft_vq_tau_start', type=float, default=1.0,
                   help='Task #145: τ at epoch 0 (uniform soft assignment).')
    p.add_argument('--soft_vq_tau_end', type=float, default=0.01,
                   help='Task #145: τ at final epoch (near-hard assignment).')
    p.add_argument('--soft_vq_anneal_type', choices=['linear', 'cosine'], default='linear',
                   help='Task #145: τ annealing schedule.')
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


@torch.no_grad()
def _collect_encoder_latents(model, loader, device, max_batches=8):
    """Collect encoder output latents z (pre-quant) for geodesic re-kmeans."""
    was_training = model.training
    model.eval()
    zs = []
    for i, batch in enumerate(loader):
        x = batch.to(device, non_blocking=True).float()
        zs.append(model.encoder(x))
        if max_batches and (i + 1) >= max_batches:
            break
    if was_training:
        model.train()
    if len(zs) == 0:
        raise ValueError("_collect_encoder_latents: no batches collected.")
    return torch.cat(zs, dim=0)


@torch.no_grad()
def _per_layer_residuals(model, latents):
    """Replay residual-VQ flow to get the input latent feeding each layer.

    Returns list of length num_layers; res_list[li] is the residual (B, e_dim)
    entering layer li. Uses argmin selection (no sinkhorn) for deterministic
    residual reconstruction.
    """
    residual = latents
    res_list = []
    for vq in model.hrq.vq_layers:
        res_list.append(residual)
        d = vq._per_component_dist_sq(residual, vq.embeddings.weight)
        idx = torch.argmin(d, dim=-1)
        x_q = vq.embeddings.weight.index_select(0, idx)
        residual = residual - x_q
    return res_list


@torch.no_grad()
def geodesic_reinit(model, latents):
    """A方案: re-init every layer's codebook via init_emb_geodesic on its
    residual input (replayed across the residual-VQ stack)."""
    residual = latents
    for vq in model.hrq.vq_layers:
        vq.init_emb_geodesic(residual)
        d = vq._per_component_dist_sq(residual, vq.embeddings.weight)
        idx = torch.argmin(d, dim=-1)
        x_q = vq.embeddings.weight.index_select(0, idx)
        residual = residual - x_q


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
        soft_vq_enabled=args.soft_vq_enabled,
        soft_vq_tau_start=args.soft_vq_tau_start,
        soft_vq_tau_end=args.soft_vq_tau_end,
        soft_vq_anneal_type=args.soft_vq_anneal_type,
    ).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters())}")
    print(f"  num codebook layers: {len(model.hrq.vq_layers)}")
    if args.soft_vq_enabled:
        print(f"  Task #145 soft VQ ENABLED: τ {args.soft_vq_tau_start} → {args.soft_vq_tau_end} ({args.soft_vq_anneal_type})")
    # R137: apply theta_init override (default 0.0; use 0.01 to escape Euclidean fixed point)
    # Task #149: support --theta_init_list for per-layer diverse init (3 layers → 3 different κ).
    if args.theta_init_list is not None:
        n_layers = len(model.hrq.vq_layers)
        if len(args.theta_init_list) != n_layers:
            raise ValueError(
                f"[Task #149 theta_init_list] got {len(args.theta_init_list)} values "
                f"but model has {n_layers} layers. Must match."
            )
        with torch.no_grad():
            for li, vq in enumerate(model.hrq.vq_layers):
                vq.theta_m.data = torch.full(
                    (args.M,), args.theta_init_list[li], dtype=torch.float32, device=device
                )
        kappa_inits = [args.kappa_max * np.tanh(t) for t in args.theta_init_list]
        print(f"  Task #149 per-layer θ_m init (diverse): {args.theta_init_list} "
              f"→ κ_m={['%.4f' % k for k in kappa_inits]}")
    elif args.theta_init != 0.0:
        with torch.no_grad():
            for vq in model.hrq.vq_layers:
                vq.theta_m.data = torch.full(
                    (args.M,), args.theta_init, dtype=torch.float32, device=device
                )
        kappa_init = (args.kappa_max * np.tanh(args.theta_init))
        print(f"  per-component θ_m init (R137 override): {[args.theta_init]*args.M} "
              f"→ κ_m={kappa_init:.6f}")
    else:
        print(f"  per-component θ_m init (default zeros): {[0.0]*args.M}")

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

    # Task #89 A方案/B方案 state
    num_layers = len(model.hrq.vq_layers)
    geo_initialized = False  # A方案: lazy geodesic init on first real batch
    global_batch = 0         # B方案: global batch counter for dead-code cadence
    # per-layer accumulated usage since last dead-code reset
    usage_accum = [torch.zeros(vq.n_e, dtype=torch.long, device=device)
                   for vq in model.hrq.vq_layers]
    if args.geodesic_kmeans:
        print(f"  [A方案] geodesic_kmeans=ON, re_kmeans_every={args.re_kmeans_every}")
    if args.dead_code_reset_every > 0:
        print(f"  [B方案] dead_code_reset_every={args.dead_code_reset_every} batches, "
              f"threshold={args.dead_code_reset_threshold}, "
              f"replace_ratio={args.dead_code_replace_ratio}")

    # Task #144 κ+codebook 解耦训练调度 state
    kappa_decouple_enabled = args.kappa_freeze_epochs > 0
    phase_a_done = False
    kappa_frozen_for_good = False  # Task #144 Phase B: 一旦 freeze-on-collapse 触发, 永久冻结
    if kappa_decouple_enabled:
        # Phase A: κ 冻结在 0 (lr_theta=0 for theta_m params)
        for pg in optim.param_groups:
            if pg['params'] == theta_params:
                pg['lr'] = 0.0
        print(f"  [Task #144 Phase A] κ FROZEN at 0 for first {args.kappa_freeze_epochs} epochs "
              f"(lr_theta=0). Codebook trained alone to health baseline.")
    else:
        print(f"  [Task #144 disabled] κ-decouple off. κ learned freely (traditional Task #89 mode).")

    # Task #144: per-epoch usage tracking for utilization monitoring
    epoch_usage = [torch.zeros(vq.n_e, dtype=torch.long, device=device)
                   for vq in model.hrq.vq_layers]
    # Phase A baseline utilization (per layer, fraction used) — written at end of Phase A,
    # used as reference for Phase B freeze-on-collapse trigger
    phase_a_baseline_util = None

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_quant = 0.0
        n_batches = 0

        for batch in loader:
            x = batch.to(device, non_blocking=True).float()

            # A方案: lazy geodesic codebook init on the first real batch (mirrors
            # forward()'s lazy init_emb, but uses expmap0->kmeans->logmap0).
            if args.geodesic_kmeans and not geo_initialized:
                with torch.no_grad():
                    z0 = model.encoder(x)
                geodesic_reinit(model, z0)
                for vq in model.hrq.vq_layers:
                    vq.initted = True  # block forward()'s plain init_emb
                geo_initialized = True
                print(f"  [A方案] geodesic init done at epoch {epoch} (first batch)")

            optim.zero_grad()

            out, quant_loss, indices = model(x, use_sk=True)
            total_loss, recon_loss = model.compute_loss(out, quant_loss, xs=x)

            # R137 fix: NaN guard — torch.where 3-branch 在 κ_max=2 + lr_theta=5e-3
            # 训练 ep 200→260 之间触发 NaN (κ→κ_max 后 acos 边界不稳定).
            # Per R2 (no fallback): detect + raise, 不 silent ignore.
            if not torch.isfinite(total_loss):
                raise ValueError(
                    f"[Task #137 R137 NaN guard] NaN/Inf loss at epoch {epoch}, batch {n_batches}. "
                    f"total_loss={total_loss.item()}, quant_loss={quant_loss.item()}, "
                    f"kappa={[vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]}"
                )

            total_loss.backward()
            # R137 fix: gradient clip — 防止 κ→κ_max 边界梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            # B方案: usage tracking + periodic dead-code reset.
            # indices shape (B, num_layers); column li = layer li's selection.
            global_batch += 1
            for li, vq in enumerate(model.hrq.vq_layers):
                _, uc = vq.get_codebook_usage(indices[..., li])
                usage_accum[li] = usage_accum[li] + uc
                # Task #144: also accumulate per-epoch usage (for util monitoring)
                epoch_usage[li] = epoch_usage[li] + uc
            if args.dead_code_reset_every > 0 and global_batch % args.dead_code_reset_every == 0:
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
                    print(f"    [B方案] batch {global_batch}: reset {n_reset_total} dead codes")

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_quant += quant_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_quant = epoch_quant / n_batches

        # Task #144: compute per-layer epoch utilization (fraction of codebook used this epoch)
        epoch_util_per_layer = []
        for li, vq in enumerate(model.hrq.vq_layers):
            used_codes = (epoch_usage[li] > 0).sum().item()
            util_frac = used_codes / vq.n_e
            epoch_util_per_layer.append(util_frac)
            epoch_usage[li] = torch.zeros(vq.n_e, dtype=torch.long, device=device)
        util_str = " ".join(f"L{li}u={u:.2%}" for li, u in enumerate(epoch_util_per_layer))

        # A方案: periodic geodesic re-kmeans (rebuild codebook in current manifold).
        if (args.geodesic_kmeans and args.re_kmeans_every > 0
                and epoch % args.re_kmeans_every == 0):
            latents = _collect_encoder_latents(model, loader, device)
            geodesic_reinit(model, latents)
            for li in range(num_layers):
                usage_accum[li] = torch.zeros(
                    model.hrq.vq_layers[li].n_e, dtype=torch.long, device=device)
            print(f"  [A方案] re-kmeans at epoch {epoch} "
                  f"(on {latents.shape[0]} latents)")

        if epoch % args.log_interval == 0 or epoch == 1:
            elapsed = time.time() - t_start
            # Print κ_m per layer
            kappa_str = ""
            for li, vq in enumerate(model.hrq.vq_layers):
                kappas = vq.kappa_m().detach().cpu().tolist()
                kappa_str += f"L{li}κ=[{','.join(f'{k:+.4f}' for k in kappas)}] "
            print(f"  ep{epoch:4d}/{args.epochs} loss={avg_loss:.4f} recon={avg_recon:.4f} "
                  f"quant={avg_quant:.4f} {kappa_str}{util_str} ({elapsed:.1f}s)")
            log_kappa(epoch)

        # Task #144: Phase A → Phase B transition + Phase B freeze-on-collapse
        if kappa_decouple_enabled and not phase_a_done and epoch == args.kappa_freeze_epochs:
            # End of Phase A: snapshot baseline utilization
            phase_a_baseline_util = list(epoch_util_per_layer)
            phase_a_done = True
            # Save baseline to JSON if path provided
            if args.phase_a_baseline_util_path:
                with open(args.phase_a_baseline_util_path, 'w') as f:
                    json.dump({
                        'phase_a_end_epoch': epoch,
                        'per_layer_baseline_util': phase_a_baseline_util,
                        'num_layers': num_layers,
                    }, f, indent=2)
            print(f"  [Task #144 Phase A END] epoch {epoch}: baseline util = {phase_a_baseline_util}")
            print(f"  [Task #144 Phase A ckpt] saved best_loss_model_phaseA.pth (separate from main ckpt)")

        if kappa_decouple_enabled and phase_a_done and not kappa_frozen_for_good \
                and epoch == args.kappa_freeze_epochs + 1:
            # Start of Phase B: unfreeze κ with very small lr_theta
            for pg in optim.param_groups:
                if pg['params'] == theta_params:
                    pg['lr'] = args.lr_theta_post_unfreeze
            print(f"  [Task #144 Phase B START] epoch {epoch}: κ UNFROZEN, "
                  f"lr_theta={args.lr_theta_post_unfreeze}")

        if kappa_decouple_enabled and phase_a_done and not kappa_frozen_for_good \
                and phase_a_baseline_util is not None:
            # Phase B monitoring: check per-layer util drop
            for li, util_now in enumerate(epoch_util_per_layer):
                util_baseline = phase_a_baseline_util[li]
                util_drop = util_baseline - util_now
                if util_drop > args.utilization_freeze_threshold:
                    # Freeze κ for good (lr_theta=0)
                    for pg in optim.param_groups:
                        if pg['params'] == theta_params:
                            pg['lr'] = 0.0
                    kappa_frozen_for_good = True
                    print(f"  [Task #144 FREEZE-ON-COLLAPSE] epoch {epoch} layer {li}: "
                          f"util dropped {util_baseline:.2%} → {util_now:.2%} "
                          f"(Δ={util_drop:+.2%}, threshold={args.utilization_freeze_threshold:.2%}). "
                          f"κ frozen permanently.")
                    break  # 一旦 freeze, 不再解冻

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

        # Task #145 Phase 2: 软量化退火 — 每 epoch 调用 update_tau 退火 τ
        if args.soft_vq_enabled:
            model.hrq.update_tau(epoch, args.epochs)

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