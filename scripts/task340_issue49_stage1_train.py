"""
Task #340 — Issue #49 Stage 1 — FreeCurvHRQVAE 训练 (统一公式修复版).

基于 Issue #47 修复的统一 κ-stereographic 公式 + 测地 kmeans + κ-codebook 解耦调度.

Phase A: κ 冻结在 0 (θ_init=给定值, 但 κ=κ_max·tanh(θ) 在 tiny θ 时≈0),
          codebook 正常训练至收敛 (collision ≤ 0.20, util ≥ 90%).
Phase B: θ 解冻 (lr_theta << lr), κ 逐渐学习非零值.

3 臂: θ_init = +δ, -δ, 0 (δ=0.02 per Issue #48 噪声底线).

Usage:
  # Arm A (+δ)
  python3 scripts/task340_issue49_stage1_train.py \\
    --theta_init 0.02 --lr_theta 1e-5 \\
    --ckpt_dir products/task340/arm_plus \\
    --kappa_log_path products/task340/arm_plus/kappa_log.json

  # Arm B (-δ)
  python3 scripts/task340_issue49_stage1_train.py \\
    --theta_init -0.02 --lr_theta 1e-5 \\
    --ckpt_dir products/task340/arm_minus \\
    --kappa_log_path products/task340/arm_minus/kappa_log.json

  # Arm C (0)
  python3 scripts/task340_issue49_stage1_train.py \\
    --theta_init 0.0 --lr_theta 1e-5 \\
    --ckpt_dir products/task340/arm_zero \\
    --kappa_log_path products/task340/arm_zero/kappa_log.json
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


def parse_args():
    p = argparse.ArgumentParser()
    # Issue #49 specific
    p.add_argument('--theta_init', type=float, default=0.02,
                   help='θ init value (3 arms: +0.02, -0.02, 0.0)')
    p.add_argument('--lr_theta', type=float, default=1e-5,
                   help='θ learning rate in Phase B (much smaller than lr)')
    p.add_argument('--phase_a_epochs', type=int, default=200,
                   help='Phase A: κ frozen, codebook training epochs')
    p.add_argument('--phase_b_epochs', type=int, default=200,
                   help='Phase B: κ unfrozen, joint training epochs')

    # Model
    p.add_argument('--M', type=int, default=1,
                   help='M=1 (per-layer full e_dim), not component-split')
    p.add_argument('--kappa_max', type=float, default=2.0)
    p.add_argument('--num_emb_list', type=int, nargs='+',
                   default=[64, 128, 256])
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+',
                   default=[512, 256, 128, 64])
    p.add_argument('--loss_type', default='poincare')
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--quant_loss_weight', type=float, default=1.0)
    # Note: Issue #47 fix is baked into hrqvae_free_curv.py formulas, no flag needed

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
    p.add_argument('--geodesic_kmeans', action='store_true', default=True,
                   help='Use geodesic kmeans init (Task #138)')

    # Sinkhorn (for training, but we use argmin for Phase A/B consistency)
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
    """Build FreeCurvHRQVAE with Issue #47 fixed formula."""
    model = FreeCurvHRQVAE(
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
        kmeans_init=False,  # We'll do init manually
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    return model


def init_theta(model, theta_init_value):
    """Set all θ_m to the same init value."""
    with torch.no_grad():
        for vq in model.hrq.vq_layers:
            vq.theta_m.data.fill_(theta_init_value)
            vq.initted = False


def geodesic_kmeans_init(model, data_loader, device):
    """Run geodesic kmeans init for all VQ layers."""
    print("  Geodesic kmeans init...")
    # Collect all latents
    all_z = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            z = model.encoder(batch)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    print(f"  Collected latents: {all_z.shape}")

    for vq in model.hrq.vq_layers:
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
        indices = model.get_indices(batch, use_sk=use_sk)  # (B, 3)
        all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices, dim=0).numpy()  # (N, 3)

    utils = []
    for lyr_idx in range(len(num_emb_list)):
        K = num_emb_list[lyr_idx]
        sid_l = all_indices[:, lyr_idx]
        unique = len(np.unique(sid_l))
        utils.append(unique / K * 100)

    # Collision rate: (N - unique SID rows with 3 digits) / N
    from collections import Counter
    sid_tuples = [tuple(row) for row in all_indices]
    counts = Counter(sid_tuples)
    collisions = sum(c - 1 for c in counts.values())
    collision_rate = collisions / len(sid_tuples)

    return utils, collision_rate, all_indices


def train_epoch(model, data_loader, optimizer, device, phase='A', use_sk=True):
    """Single training epoch (shared by Phase A and B).

    use_sk=True 匹配 HG-Rec 健康 baseline recipe (Sinkhorn 防止坍缩).
    Phase A 冻结 θ, Phase B 解冻 θ — 这是 κ-codebook 解耦调度的核心.
    """
    total_loss = 0
    n_batches = 0
    for batch in data_loader:
        batch = batch.to(device)
        z = model.encoder(batch)
        out, rq_loss, indices = model.hrq(z, use_sk=use_sk)
        out = model.decoder(out)
        recon_loss = F.mse_loss(out, batch)
        loss = recon_loss + model.quant_loss_weight * rq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def save_checkpoint(model, optimizer, epoch, phase, path, extra_args=None):
    """Save checkpoint with R12 cleanup."""
    if path.exists():
        os.remove(path)
    ckpt_args = {
        'num_emb_list': model.num_emb_list,
        'e_dim': model.e_dim,
        'layers': model.layers,
        'loss_type': model.loss_type,
        'quant_loss_weight': model.quant_loss_weight,
        'beta': model.beta,
        'sk_epsilons': [vq.sk_eps for vq in model.hrq.vq_layers],
        'sk_iters': model.hrq.vq_layers[0].sk_iters if model.hrq.vq_layers else 50,
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

    print(f"\n{'='*60}")
    print(f"Issue #49 Stage 1 — θ_init={args.theta_init}")
    print(f"{'='*60}")

    # Determine data path
    if args.data_path is None:
        args.data_path = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

    # Load data
    data = EmbDataset(args.data_path)
    in_dim = data.dim
    print(f"Data: {len(data)} items, dim={in_dim}")

    loader = DataLoader(data, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(data, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    model = build_model(args, in_dim).to(device)
    print(f"Model: M={args.M}, num_emb_list={args.num_emb_list}, "
          f"e_dim={args.e_dim}")

    # Init θ
    init_theta(model, args.theta_init)
    print(f"θ init: {args.theta_init} (κ=κ_max·tanh({args.theta_init})"
          f"={args.kappa_max * np.tanh(args.theta_init):.4f})")

    # Geodesic kmeans init
    if args.geodesic_kmeans:
        geodesic_kmeans_init(model, loader, device)

    # ─── Phase A: κ frozen (codebook only) ───
    print(f"\nPhase A ({args.phase_a_epochs} epochs, θ frozen)")
    # Freeze θ
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

            # Log κ values
            kappas = []
            for vq in model.hrq.vq_layers:
                kappas.append(vq.kappa_m().detach().cpu().tolist())
            kappa_log.append({
                'phase': 'A', 'epoch': epoch, 'loss': loss,
                'utilization': utils, 'collision': collision,
                'kappa_per_layer': kappas,
            })

    # Save Phase A final
    save_checkpoint(model, optimizer_a, args.phase_a_epochs, 'A',
                    Path(args.ckpt_dir) / "phase_a_final.pth")
    print(f"Phase A done. Best collision={best_collision:.4f} @ epoch {best_epoch}")

    # ─── Phase B: κ unfrozen ───
    print(f"\nPhase B ({args.phase_b_epochs} epochs, θ unfrozen, lr_θ={args.lr_theta})")

    # Unfreeze θ
    for name, param in model.named_parameters():
        if 'theta_m' in name:
            param.requires_grad = True

    # Separate optimizers: low lr for θ, normal lr for rest
    theta_params = [p for n, p in model.named_parameters()
                    if 'theta_m' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if 'theta_m' not in n and p.requires_grad]

    optimizer_b = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr,
         'weight_decay': args.weight_decay},
        {'params': theta_params, 'lr': args.lr_theta,
         'weight_decay': 0.0},
    ])

    for epoch in range(1, args.phase_b_epochs + 1):
        loss = train_epoch(model, loader, optimizer_b, device, phase='B')

        if epoch % args.log_interval == 0 or epoch == 1:
            utils, collision, indices = evaluate(
                model, eval_loader, device, args.num_emb_list)
            kappas = []
            for vq in model.hrq.vq_layers:
                kappas.append(vq.kappa_m().detach().cpu().tolist())
            print(f"  [B {epoch}/{args.phase_b_epochs}] loss={loss:.4f} "
                  f"util={utils} collision={collision:.4f} "
                  f"κ={[f'{k[0]:.4f}' for k in kappas]}")

            if collision < best_collision:
                best_collision = collision
                best_epoch = args.phase_a_epochs + epoch
                save_checkpoint(model, optimizer_b, best_epoch, 'B',
                                Path(args.ckpt_dir) / "best_collision_model.pth")

            # Gate 1 check: early stop if utilization drops below 90%
            if any(u < 90.0 for u in utils):
                print(f"  ⚠️ Utilization drop below 90%: {utils}")
                print(f"  HALT: Phase B codebook collapse detected")
                # Save this state for diagnosis
                save_checkpoint(model, optimizer_b, epoch, 'B_COLLAPSED',
                                Path(args.ckpt_dir) / "collapsed_model.pth")
                break

            kappa_log.append({
                'phase': 'B', 'epoch': args.phase_a_epochs + epoch,
                'loss': loss, 'utilization': utils, 'collision': collision,
                'kappa_per_layer': kappas,
            })

    # Save final
    save_checkpoint(model, optimizer_b, epoch, 'B_FINAL',
                    Path(args.ckpt_dir) / "model.pth")

    # ─── Final report ───
    print(f"\n{'='*60}")
    print(f"Stage 1 Complete — θ_init={args.theta_init}")
    print(f"{'='*60}")
    print(f"  Best collision: {best_collision:.4f} @ epoch {best_epoch}")
    print(f"  Final κ values:")
    for lyr_idx, vq in enumerate(model.hrq.vq_layers):
        print(f"    Layer {lyr_idx}: {vq.kappa_m().detach().cpu().tolist()}")

    # Gate 1 pass/fail
    utils_final, collision_final, _ = evaluate(
        model, eval_loader, device, args.num_emb_list)
    kappas_final = [vq.kappa_m().detach().cpu().tolist()
                    for vq in model.hrq.vq_layers]

    gate1a = all(u >= 90.0 for u in utils_final)
    gate1b = collision_final <= 0.20
    gate1c = any(abs(k[0]) > 0.01 for k in kappas_final)

    print(f"\n  Gate 1a (util ≥ 90%): {'✅ PASS' if gate1a else '❌ FAIL'} {utils_final}")
    print(f"  Gate 1b (collision ≤ 0.20): {'✅ PASS' if gate1b else '❌ FAIL'} {collision_final:.4f}")
    print(f"  Gate 1c (κ learns non-trivial): {'✅ PASS' if gate1c else '❌ FAIL'} {kappas_final}")

    if gate1a and gate1b and gate1c:
        print(f"\n  ✅ ALL GATE 1 PASS — proceeding to Stage 2 recommended")
    else:
        print(f"\n  ❌ GATE 1 FAIL — STOP, do NOT proceed to Stage 2")

    # Save kappa log
    os.makedirs(os.path.dirname(args.kappa_log_path), exist_ok=True)
    with open(args.kappa_log_path, 'w') as f:
        json.dump(kappa_log, f, indent=2)
    print(f"  κ log saved: {args.kappa_log_path}")


if __name__ == "__main__":
    main()
