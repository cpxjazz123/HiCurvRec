"""
Task #339 — Issue #52 Stage 1 — per-layer κ 自由度对照 3 臂训练.

复用 Issue #49 统一公式 + FreeCurvHRQVAE + 解耦调度,
3 臂对照 (per-layer lr_theta / 拓宽 κ_max / 延长 Phase B).

用法:
  Arm A: --per_layer_lr_theta 1   # 每层独立 lr_theta
  Arm B: --kappa_max 1.0           # 拓宽 κ_max 范围
  Arm C: --phase_b_epochs 500      # 延长 Phase B
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.utils import EmbDataset
from model.hrqvae_free_curv import FreeCurvHRQVAE


def parse_args():
    p = argparse.ArgumentParser()
    # Issue #52 关键差异 (vs Issue #49)
    p.add_argument('--per_layer_lr_theta', type=int, default=0,
                   help='Arm A: 每层独立 lr_theta (1=启用)')
    p.add_argument('--kappa_max_override', type=float, default=None,
                   help='Arm B: 拓宽 κ_max 范围 (e.g. 1.0)')
    p.add_argument('--phase_b_epochs_override', type=int, default=None,
                   help='Arm C: 延长 Phase B epochs')

    # Issue #49 标准 (复用)
    p.add_argument('--theta_init', type=float, default=-0.02)
    p.add_argument('--lr_theta', type=float, default=1e-5)
    p.add_argument('--phase_a_epochs', type=int, default=200)
    p.add_argument('--phase_b_epochs', type=int, default=200)

    p.add_argument('--M', type=int, default=1)
    p.add_argument('--kappa_max', type=float, default=2.0)
    p.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    p.add_argument('--loss_type', default='poincare')
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--quant_loss_weight', type=float, default=1.0)

    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_path', default=None)
    p.add_argument('--device', default='cuda:0')

    p.add_argument('--kmeans_init', action='store_true', default=True)
    p.add_argument('--kmeans_iters', type=int, default=1000)
    p.add_argument('--geodesic_kmeans', action='store_true', default=True)

    p.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.01, 0.01, 0.01])
    p.add_argument('--sk_iters', type=int, default=50)

    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--kappa_log_path', required=True)
    p.add_argument('--log_interval', type=int, default=10)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, in_dim):
    kappa_max = args.kappa_max_override if args.kappa_max_override else args.kappa_max
    return FreeCurvHRQVAE(
        in_dim=in_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        M=args.M,
        kappa_max=kappa_max,
        layers=args.layers,
        dropout_prob=0.0, bn=False,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=False,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )


def init_theta(model, theta_init_value):
    with torch.no_grad():
        for vq in model.hrq.vq_layers:
            vq.theta_m.data.fill_(theta_init_value)
            vq.initted = False


def geodesic_kmeans_init(model, data_loader, device):
    print("  Geodesic kmeans init...")
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
                vq.init_emb_geodesic(z_device)
            else:
                vq.init_emb(z_device)
    model.train()


@torch.no_grad()
def evaluate(model, data_loader, device, num_emb_list, use_sk=True):
    model.eval()
    all_indices = []
    for batch in data_loader:
        batch = batch.to(device)
        indices = model.get_indices(batch, use_sk=use_sk)
        all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices, dim=0).numpy()

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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"Task #339 — Issue #52 — per-layer κ 自由度对照")
    print(f"{'='*70}")
    print(f"  per_layer_lr_theta = {args.per_layer_lr_theta}")
    print(f"  kappa_max_override = {args.kappa_max_override}")
    print(f"  phase_b_epochs_override = {args.phase_b_epochs_override}")
    print(f"  theta_init = {args.theta_init}")
    print(f"  device = {args.device}")
    print(f"{'='*70}")

    phase_b_epochs = args.phase_b_epochs_override if args.phase_b_epochs_override else args.phase_b_epochs

    if args.data_path is None:
        args.data_path = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

    data = EmbDataset(args.data_path)
    in_dim = data.dim
    print(f"Data: {len(data)} items, dim={in_dim}")

    loader = DataLoader(data, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(data, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args, in_dim).to(device)

    init_theta(model, args.theta_init)
    print(f"θ init: {args.theta_init}")

    if args.geodesic_kmeans:
        geodesic_kmeans_init(model, loader, device)

    # Phase A
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
    kappa_log = []

    for epoch in range(1, args.phase_a_epochs + 1):
        loss = train_epoch(model, loader, optimizer_a, device, phase='A')
        if epoch % args.log_interval == 0 or epoch == 1:
            utils, collision, indices = evaluate(model, eval_loader, device, args.num_emb_list)
            print(f"  [A {epoch}/{args.phase_a_epochs}] loss={loss:.4f} util={utils} collision={collision:.4f}")
            if collision < best_collision:
                best_collision = collision
                save_checkpoint(model, optimizer_a, epoch, 'A',
                                Path(args.ckpt_dir) / "best_collision_model.pth")
            kappas = [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]
            kappa_log.append({
                'phase': 'A', 'epoch': epoch, 'loss': loss,
                'utilization': utils, 'collision': collision,
                'kappa_per_layer': kappas,
            })

    save_checkpoint(model, optimizer_a, args.phase_a_epochs, 'A',
                    Path(args.ckpt_dir) / "phase_a_final.pth")

    # Phase B
    print(f"\nPhase B ({phase_b_epochs} epochs, θ unfrozen)")
    for name, param in model.named_parameters():
        if 'theta_m' in name:
            param.requires_grad = True

    theta_params = [p for n, p in model.named_parameters() if 'theta_m' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'theta_m' not in n and p.requires_grad]

    optimizer_b_other = torch.optim.AdamW(other_params, lr=args.lr, weight_decay=args.weight_decay)

    # Per-layer lr_theta (Issue #52 Arm A)
    if args.per_layer_lr_theta:
        # Build per-layer optimizer groups: L0:1e-4, L1:1e-5, L2:1e-6
        param_groups = []
        layer_lrs = [1e-4, 1e-5, 1e-6]
        for lyr_idx, p in enumerate(theta_params):
            if lyr_idx < len(layer_lrs):
                param_groups.append({'params': [p], 'lr': layer_lrs[lyr_idx]})
            else:
                param_groups.append({'params': [p], 'lr': layer_lrs[-1]})
        optimizer_b_theta = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"  Per-layer lr_theta: {layer_lrs}")
    else:
        optimizer_b_theta = torch.optim.AdamW(theta_params, lr=args.lr_theta, weight_decay=args.weight_decay)

    for epoch in range(1, phase_b_epochs + 1):
        loss = train_epoch(model, loader, optimizer_b_other, device, phase='B')
        # θ step separately
        for vq in model.hrq.vq_layers:
            if vq.theta_m.grad is not None:
                optimizer_b_theta.step()
                break

        if epoch % args.log_interval == 0 or epoch == 1:
            utils, collision, indices = evaluate(model, eval_loader, device, args.num_emb_list)
            print(f"  [B {epoch}/{phase_b_epochs}] loss={loss:.4f} util={utils} collision={collision:.4f}")
            if collision < best_collision:
                best_collision = collision
                save_checkpoint(model, optimizer_b_other, epoch, 'B',
                                Path(args.ckpt_dir) / "best_collision_model.pth")
            kappas = [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]
            kappa_log.append({
                'phase': 'B', 'epoch': epoch, 'loss': loss,
                'utilization': utils, 'collision': collision,
                'kappa_per_layer': kappas,
            })

    save_checkpoint(model, optimizer_b_other, phase_b_epochs, 'B',
                    Path(args.ckpt_dir) / "phase_b_final.pth")

    Path(args.kappa_log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.kappa_log_path, 'w') as f:
        json.dump(kappa_log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Issue #52 Stage 1 complete. Best collision = {best_collision:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()