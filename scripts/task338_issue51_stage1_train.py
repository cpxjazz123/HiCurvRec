"""
Task #338 — Issue #51 Stage 1 — Combined HypPreEncoder + per-layer learned κ training.

Wraps FreeCurvHRQVAE (Issue #49 Arm B recipe) with HRQVAEWithHypPre (Issue #43 wrapper)
to combine both:
- Issue #43: HypPreEncoder(expmap0, c=0.74) before encoder → preserves hyperbolic
  structure in residual space (counter Task #80's flattening)
- Issue #49: per-layer learned κ (Issue #47 fixed formula) + κ-codebook decoupled
  scheduling (Phase A κ frozen + Phase B κ unfrozen lr_theta=1e-5)

Architecture (Issue #51 combined):
    x (768d) → HypPreEncoder(c=0.74) → encoder (MLP) → hrq (FreeCurvHRQVAE) → decoder

Stage 1 only. Stage 2-4 in separate scripts.

Usage:
  python3 scripts/task338_issue51_stage1_train.py \\
    --theta_init -0.02 --lr_theta 1e-5 \\
    --ckpt_dir products/task338/stage1 \\
    --kappa_log_path products/task338/kappa_log.json
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.utils import EmbDataset
from model.hrqvae_free_curv import FreeCurvHRQVAE
from task334_issue43_gate2a_hyp_pre_encoder import HRQVAEWithHypPre


def parse_args():
    p = argparse.ArgumentParser()
    # Issue #51 specific
    p.add_argument('--theta_init', type=float, default=-0.02,
                   help='θ init value (Issue #49 Arm B recipe: -0.02)')
    p.add_argument('--lr_theta', type=float, default=1e-5,
                   help='θ learning rate in Phase B')
    p.add_argument('--phase_a_epochs', type=int, default=200,
                   help='Phase A: κ frozen, codebook training epochs')
    p.add_argument('--phase_b_epochs', type=int, default=200,
                   help='Phase B: κ unfrozen, joint training epochs')

    # Issue #43 HypPreEncoder
    p.add_argument('--hyp_c', type=float, default=0.74,
                   help='HypPreEncoder curvature magnitude (Ollivier mean)')
    p.add_argument('--use_hyp_pre_encoder', type=int, default=1,
                   help='0=disabled, 1=enabled (Issue #51 default)')

    # Model
    p.add_argument('--M', type=int, default=1)
    p.add_argument('--kappa_max', type=float, default=2.0)
    p.add_argument('--num_emb_list', type=int, nargs='+',
                   default=[64, 128, 256])
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+',
                   default=[512, 256, 128, 64])
    p.add_argument('--loss_type', default='poincare')
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--quant_loss_weight', type=float, default=1.0)

    # Training
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_path', default=None)
    p.add_argument('--device', default='cuda:0')

    # Kmeans init
    p.add_argument('--kmeans_init', action='store_true', default=True)
    p.add_argument('--kmeans_iters', type=int, default=1000)
    p.add_argument('--geodesic_kmeans', action='store_true', default=True)

    # Sinkhorn (essential per Issue #49)
    p.add_argument('--sk_epsilons', type=float, nargs='+',
                   default=[0.01, 0.01, 0.01])
    p.add_argument('--sk_iters', type=int, default=50)

    # Output
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--kappa_log_path', required=True)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_every', type=int, default=50)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, in_dim):
    """Build FreeCurvHRQVAE + (optional) wrap with HRQVAEWithHypPre.

    Issue #51 combined recipe:
      - FreeCurvHRQVAE (Issue #49 per-layer κ)
      - HRQVAEWithHypPre wrapper (Issue #43 c=0.74 pre-mapping)
    """
    base = FreeCurvHRQVAE(
        in_dim=in_dim,
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
        kmeans_init=False,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    if args.use_hyp_pre_encoder:
        model = HRQVAEWithHypPre(base, c=args.hyp_c, enabled=True)
        print(f"[Issue #51] HypPreEncoder ENABLED (c={args.hyp_c})")
        n_base = sum(p.numel() for p in base.parameters())
        n_wrap = sum(p.numel() for p in model.parameters())
        print(f"  base params={n_base}, wrapped params={n_wrap}")
    else:
        model = base
        print(f"[Issue #51] HypPreEncoder DISABLED (regression baseline)")

    return model


def init_theta(model, theta_init_value):
    """Set all θ_m to the same init value."""
    # Access the underlying hrq via wrapper property alias
    hrq = model.hrq if hasattr(model, 'hrq') else model.base.hrq
    with torch.no_grad():
        for vq in hrq.vq_layers:
            vq.theta_m.data.fill_(theta_init_value)
            vq.initted = False


def geodesic_kmeans_init(model, data_loader, device):
    """Run geodesic kmeans init for all VQ layers."""
    hrq = model.hrq if hasattr(model, 'hrq') else model.base.hrq
    print("  Geodesic kmeans init...")
    all_z = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Wrapper forward: applies HypPreEncoder first
            if hasattr(model, 'hyp_pre'):
                batch_in = model.hyp_pre(batch)
            else:
                batch_in = batch
            z = model.encoder(batch_in)  # via property alias
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    print(f"  Collected latents: {all_z.shape}")

    for vq in hrq.vq_layers:
        if not vq.initted:
            z_device = all_z.to(vq.embeddings.weight.device)
            if hasattr(vq, 'init_emb_geodesic'):
                print(f"    Init layer (geodesic, n_e={vq.n_e})")
                vq.init_emb_geodesic(z_device)
            else:
                print(f"    Init layer (Euclidean kmeans, n_e={vq.n_e})")
                vq.init_emb(z_device)
    model.train()


@torch.no_grad()
def evaluate(model, data_loader, device, num_emb_list, use_sk=True):
    """Compute utilization and collision rate."""
    model.eval()
    all_indices = []
    for batch in data_loader:
        batch = batch.to(device)
        # Wrapper forward: applies HypPreEncoder first
        if hasattr(model, 'hyp_pre'):
            batch_in = model.hyp_pre(batch)
        else:
            batch_in = batch
        indices = model.get_indices(batch_in, use_sk=use_sk)  # (B, 3)
        all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices, dim=0).numpy()  # (N, 3)

    utils = []
    for lyr_idx in range(len(num_emb_list)):
        K = num_emb_list[lyr_idx]
        sid_l = all_indices[:, lyr_idx]
        unique = len(np.unique(sid_l))
        utils.append(unique / K * 100)

    from collections import Counter
    sid_tuples = [tuple(row) for row in all_indices]
    counts = Counter(sid_tuples)
    collisions = sum(c - 1 for c in counts.values())
    collision_rate = collisions / len(sid_tuples)

    return utils, collision_rate, all_indices


def train_epoch(model, data_loader, optimizer, device, phase='A', use_sk=True):
    """Single training epoch (shared by Phase A and B)."""
    total_loss = 0
    n_batches = 0
    # Access base for attrs not aliased on wrapper
    base = model.base if hasattr(model, 'base') else model
    quant_loss_weight = getattr(model, 'quant_loss_weight',
                                 base.quant_loss_weight)
    for batch in data_loader:
        batch = batch.to(device)
        # Wrapper forward: applies HypPreEncoder first
        if hasattr(model, 'hyp_pre'):
            batch_in = model.hyp_pre(batch)
        else:
            batch_in = batch
        z = model.encoder(batch_in)  # via property alias
        out, rq_loss, indices = model.hrq(z, use_sk=use_sk)
        out = model.decoder(out)
        recon_loss = F.mse_loss(out, batch)
        loss = recon_loss + quant_loss_weight * rq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def save_checkpoint(model, optimizer, epoch, phase, path, extra_args=None):
    """Save checkpoint with R12 cleanup (delete old before saving new)."""
    if path.exists():
        os.remove(path)
    # Access base attrs through wrapper if needed
    base = model.base if hasattr(model, 'base') else model
    ckpt_args = {
        'num_emb_list': base.num_emb_list,
        'e_dim': base.e_dim,
        'layers': base.layers,
        'loss_type': base.loss_type,
        'quant_loss_weight': base.quant_loss_weight,
        'beta': base.beta,
        'sk_epsilons': [vq.sk_eps for vq in base.hrq.vq_layers],
        'sk_iters': base.hrq.vq_layers[0].sk_iters if base.hrq.vq_layers else 50,
    }
    if extra_args:
        ckpt_args.update(extra_args)
    torch.save({
        'epoch': epoch, 'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': ckpt_args,
    }, path)
    print(f"  Checkpoint saved: {path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"Task #338 — Issue #51 — Combined HypPreEncoder + per-layer learned κ")
    print(f"{'='*70}")
    print(f"  HypPreEncoder  = {'ENABLED c=' + str(args.hyp_c) if args.use_hyp_pre_encoder else 'DISABLED'}")
    print(f"  θ_init          = {args.theta_init}")
    print(f"  num_emb_list    = {args.num_emb_list}")
    print(f"  Phase A epochs  = {args.phase_a_epochs}")
    print(f"  Phase B epochs  = {args.phase_b_epochs}")
    print(f"  device          = {args.device}")
    print(f"{'='*70}")

    if args.data_path is None:
        args.data_path = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

    data = EmbDataset(args.data_path)
    in_dim = data.dim
    print(f"Data: {len(data)} items, dim={in_dim}")

    loader = DataLoader(data, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(data, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Issue #43 input scale diagnostic
    if args.use_hyp_pre_encoder:
        sample = torch.as_tensor(data.embeddings, dtype=torch.float32)
        norms = sample.norm(dim=-1)
        boundary = 1.0 / (args.hyp_c ** 0.5)
        print(f"\n[Issue #51 HypPre scale diagnostic]")
        print(f"  c={args.hyp_c} → Poincaré ball boundary = {boundary:.4f}")
        print(f"  Input ‖x‖: min={norms.min():.4f}, "
              f"mean={norms.mean():.4f}, max={norms.max():.4f}")
        over_boundary = (norms > boundary).float().mean().item() * 100
        print(f"  Over-boundary fraction: {over_boundary:.1f}%")

    model = build_model(args, in_dim).to(device)

    init_theta(model, args.theta_init)
    print(f"\nθ init: {args.theta_init} "
          f"(κ=κ_max·tanh({args.theta_init})"
          f"={args.kappa_max * np.tanh(args.theta_init):.4f})")

    if args.geodesic_kmeans:
        geodesic_kmeans_init(model, loader, device)

    # ─── Phase A: κ frozen (codebook only) ───
    print(f"\nPhase A ({args.phase_a_epochs} epochs, θ frozen)")
    theta_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'theta_m' in name:
            param.requires_grad = False
            theta_params.append(param)
        else:
            param.requires_grad = True
            other_params.append(param)

    optimizer_a = torch.optim.AdamW(other_params, lr=args.lr,
                                    weight_decay=args.weight_decay)

    best_collision = 1.0
    best_epoch = -1
    kappa_log = []

    for epoch in range(1, args.phase_a_epochs + 1):
        loss = train_epoch(model, loader, optimizer_a, device, phase='A')

        if epoch % args.log_interval == 0 or epoch == 1:
            utils, collision, indices = evaluate(
                model, eval_loader, device, args.num_emb_list)
            print(f"  [A {epoch}/{args.phase_a_epochs}] loss={loss:.4f} "
                  f"util={utils} collision={collision:.4f}")

            if collision < best_collision:
                best_collision = collision
                best_epoch = epoch
                save_checkpoint(model, optimizer_a, epoch, 'A',
                                Path(args.ckpt_dir) / "best_collision_model.pth")

            hrq = model.hrq if hasattr(model, 'hrq') else model.base.hrq
            kappas = []
            for vq in hrq.vq_layers:
                kappas.append(vq.kappa_m().detach().cpu().tolist())
            kappa_log.append({
                'phase': 'A', 'epoch': epoch, 'loss': loss,
                'utilization': utils, 'collision': collision,
                'kappa_per_layer': kappas,
            })

    save_checkpoint(model, optimizer_a, args.phase_a_epochs, 'A',
                    Path(args.ckpt_dir) / "phase_a_final.pth")
    print(f"Phase A done. Best collision={best_collision:.4f} @ epoch {best_epoch}")

    # ─── Phase B: κ unfrozen ───
    print(f"\nPhase B ({args.phase_b_epochs} epochs, θ unfrozen, lr_θ={args.lr_theta})")

    for name, param in model.named_parameters():
        if 'theta_m' in name:
            param.requires_grad = True

    theta_params = [p for n, p in model.named_parameters()
                    if 'theta_m' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if 'theta_m' not in n and p.requires_grad]

    optimizer_b_other = torch.optim.AdamW(other_params, lr=args.lr,
                                          weight_decay=args.weight_decay)
    optimizer_b_theta = torch.optim.AdamW(theta_params, lr=args.lr_theta,
                                          weight_decay=args.weight_decay)

    for epoch in range(1, args.phase_b_epochs + 1):
        loss = train_epoch(model, loader, optimizer_b_other, device, phase='B')
        # θ step separately (smaller lr)
        # Get current θ grad from model
        hrq = model.hrq if hasattr(model, 'hrq') else model.base.hrq
        for vq in hrq.vq_layers:
            if vq.theta_m.grad is not None:
                optimizer_b_theta.step()
                break

        if epoch % args.log_interval == 0 or epoch == 1:
            utils, collision, indices = evaluate(
                model, eval_loader, device, args.num_emb_list)
            print(f"  [B {epoch}/{args.phase_b_epochs}] loss={loss:.4f} "
                  f"util={utils} collision={collision:.4f}")

            if collision < best_collision:
                best_collision = collision
                best_epoch = epoch
                save_checkpoint(model, optimizer_b_other, epoch, 'B',
                                Path(args.ckpt_dir) / "best_collision_model.pth")

            kappas = []
            for vq in hrq.vq_layers:
                kappas.append(vq.kappa_m().detach().cpu().tolist())
            kappa_log.append({
                'phase': 'B', 'epoch': epoch, 'loss': loss,
                'utilization': utils, 'collision': collision,
                'kappa_per_layer': kappas,
            })

    save_checkpoint(model, optimizer_b_other, args.phase_b_epochs, 'B',
                    Path(args.ckpt_dir) / "phase_b_final.pth")
    print(f"\nPhase B done. Best collision overall={best_collision:.4f}")

    # Save kappa log
    Path(args.kappa_log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.kappa_log_path, 'w') as f:
        json.dump(kappa_log, f, indent=2)
    print(f"\nKappa log saved: {args.kappa_log_path}")

    print(f"\n{'='*70}")
    print(f"Issue #51 Stage 1 complete.")
    print(f"  Best collision = {best_collision:.4f}")
    print(f"  ckpt_dir       = {args.ckpt_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()