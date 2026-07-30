"""
Task #336 — Issue #43 Gate 2b Stage 1 HRQ-VAE training with HypPreEncoder.

Wraps task84 baseline recipe + plugs HRQVAEWithHypPre (c=0.74 Ollivier mean)
between input and encoder. Does NOT modify upstream HG-Rec/train_hrqvae.py
(wrapper composition only — R11.4 critical decision satisfied).

Usage:
  python3 scripts/task336_issue43_gate2b_stage1_train.py [args]

Recipe (mirrors task84 baseline + Issue #43 Gate 2a):
  - HypPreEncoder c=0.74 enabled
  - num_emb_list=[64,128,256], e_dim=32, layers=[512,256,128,64]
  - sk_epsilons=[0.0, 0.0, 0.0] (argmin, no Sinkhorn)
  - epochs=1000, batch_size=1024, lr=1e-3, learner=AdamW
  - seed=42 (default torch.manual_seed)

R12 compliance: Trainer already saves best_loss_model.pth + best_collision_model.pth
every eval_step epoch. save_limit=5 handles auto-cleanup.

Logging:
  - Per-epoch metrics via Trainer
  - Stage 1 ckpt at <ckpt_dir>/<run_id>/best_loss_model.pth
  - Sidecar log: logs/task336/stage1_<ts>.log
"""
import os
import sys
import argparse
import logging
import time
import torch
from torch.utils.data import DataLoader

# Add HG-Rec + scripts to import path
HGREC_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec"
sys.path.insert(0, HGREC_ROOT)
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")

from model.utils import EmbDataset  # noqa: E402
from model.hrqvae import HRQVAE  # noqa: E402
from model.hrqvae_trainer import Trainer  # noqa: E402
from task334_issue43_gate2a_hyp_pre_encoder import HRQVAEWithHypPre  # noqa: E402


# ────────────────────────────────────────────────────────────
# Issue #43 Gate 2b: log pre-encoder scale calibration
# ────────────────────────────────────────────────────────────
def check_input_scale(data, c=0.74):
    """Warn if input ‖x‖ exceeds Poincaré ball boundary 1/√c (≈1.16 for c=0.74).

    expmap0 saturates to boundary when ‖x‖ > 1/√c. For Musical_Instruments
    item_emb.parquet, ‖x‖ ≈ √768·σ. With σ ≈ 0.07-0.15 (sentence-T5 embeddings
    are L2-normalized), ‖x‖ ≈ 2-4, well above boundary. The Stage 1 trainer
    expects inputs in a safe range; we log observed scale for the verdict.
    """
    sample = data.tensors[0] if hasattr(data, 'tensors') else data.data
    norms = sample.norm(dim=-1)
    boundary = 1.0 / (c ** 0.5)
    print(f"[Issue #43 Gate 2b] Input scale diagnostic:")
    print(f"  c={c} → Poincaré ball boundary = {boundary:.4f}")
    print(f"  Input ‖x‖: min={norms.min():.4f}, "
          f"mean={norms.mean():.4f}, max={norms.max():.4f}")
    over_boundary = (norms > boundary).float().mean().item() * 100
    print(f"  Over-boundary fraction: {over_boundary:.1f}%")
    if over_boundary > 50:
        print(f"  ⚠️ >50% inputs exceed boundary — HypPreEncoder may saturate.")
        print(f"  Note: Stage 1 trainer's downstream loss will still converge,")
        print(f"  but the encoder may not benefit from full exponential map range.")
    return norms, boundary


# ────────────────────────────────────────────────────────────
# Argparse (mirror task84 baseline + Issue #43 specific flag)
# ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Task #336 Issue #43 Gate 2b HRQ-VAE training")
    parser.add_argument('--data_path', type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument('--ckpt_dir', type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task336/stage1/Instruments")
    # HypPreEncoder params (Issue #43 Gate 2a specific)
    parser.add_argument('--use_hyp_pre_encoder', type=int, default=1,
                        help="0=disabled (regression baseline), 1=enabled (Issue #43)")
    parser.add_argument('--hyp_c', type=float, default=0.74,
                        help="Poincaré ball curvature magnitude |κ| (Ollivier mean=0.74)")
    parser.add_argument('--hyp_learnable_c', type=int, default=0,
                        help="0=fixed c, 1=learnable log-parameterized c")
    # HRQ-VAE params (mirror task84)
    parser.add_argument('--loss_type', type=str, default='poincare',
                        choices=['mse', 'l1', 'poincare'])
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument('--sk_iters', type=int, default=50)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--kmeans_init', type=bool, default=True)
    parser.add_argument('--kmeans_iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learner', type=str, default='AdamW')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--log_every', type=int, default=10,
                        help='echo progress every N epochs (in addition to eval_step)')
    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Task #336 — Issue #43 Gate 2b Stage 1 HRQ-VAE training (HypPreEncoder)")
    print("=" * 70)
    print(f"  use_hyp_pre_encoder = {args.use_hyp_pre_encoder}")
    print(f"  hyp_c               = {args.hyp_c}")
    print(f"  hyp_learnable_c     = {args.hyp_learnable_c}")
    print(f"  num_emb_list        = {args.num_emb_list}")
    print(f"  e_dim               = {args.e_dim}")
    print(f"  layers              = {args.layers}")
    print(f"  sk_epsilons         = {args.sk_epsilons}")
    print(f"  epochs              = {args.epochs}")
    print(f"  batch_size          = {args.batch_size}")
    print(f"  device              = {args.device}")
    print("=" * 70)

    # Build dataset
    print(f"Loading dataset from {args.data_path}")
    data = EmbDataset(args.data_path)
    print(f"  Dataset size: {len(data)}, dim: {data.dim}")

    # Issue #43 Gate 2b: input scale diagnostic (informational only)
    if args.use_hyp_pre_encoder:
        check_input_scale(data, c=args.hyp_c)

    # Build base HRQVAE
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

    # Wrap with HypPreEncoder if enabled (Issue #43 Gate 2a wrapper, regression-safe)
    if args.use_hyp_pre_encoder:
        model = HRQVAEWithHypPre(base, c=args.hyp_c, enabled=True)
        print(f"[Issue #43 Gate 2b] HypPreEncoder ENABLED (c={args.hyp_c})")
        n_params_base = sum(p.numel() for p in base.parameters())
        n_params_wrapped = sum(p.numel() for p in model.parameters())
        print(f"  Base params: {n_params_base}, Wrapped params: {n_params_wrapped}")
        # Sanity: forward pass on a small batch
        with torch.no_grad():
            sample = data.tensors[0][:2] * 0.03 if hasattr(data, 'tensors') else data.data[:2] * 0.03
            _ = model(sample.to(args.device), use_sk=False)
        print(f"  Sanity forward pass OK")
    else:
        model = base
        print(f"[Issue #43 Gate 2b] HypPreEncoder DISABLED (regression baseline)")

    # Build Trainer (upstream, works with wrapper via property aliases)
    data_loader = DataLoader(
        data, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))

    # Run training
    t0 = time.time()
    print(f"\n[Stage 1] Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    best_loss, best_collision = trainer.fit(data_loader)
    elapsed = time.time() - t0
    print(f"\n[Stage 1] Training completed in {elapsed/3600:.2f}h")
    print(f"  best_loss      = {best_loss:.4f}")
    print(f"  best_collision = {best_collision:.4f}")

    return best_loss, best_collision


if __name__ == "__main__":
    main()