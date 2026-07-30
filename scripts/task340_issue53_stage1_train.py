"""
Task #340 — Issue #53 Stage 1 — κ-codebook 解冻节奏对照 4 臂训练.

4 臂对照:
  Arm A: 硬性两阶段 (=#49 原配置, 第 200 epoch 解冻) — 对照组
  Arm B: 渐进式解冻, lr_theta 从 0 按指数曲线上升
  Arm C: 更早解冻 (第 50 epoch), 更小初始 lr_theta + 更长总训练
  Arm D: 振荡式 (多次冻结/小幅解冻切换)

用法:
  # Arm A (对照)
  python3 scripts/task340_issue53_stage1_train.py --arm_mode hard_two_stage --ckpt_dir products/task340/arm_a ...

  # Arm B (渐进式)
  python3 scripts/task340_issue53_stage1_train.py --arm_mode progressive --ckpt_dir products/task340/arm_b ...

  # Arm C (早解冻)
  python3 scripts/task340_issue53_stage1_train.py --arm_mode early_unfreeze --phase_b_epochs 350 --unfreeze_epoch 50 ...

  # Arm D (振荡)
  python3 scripts/task340_issue53_stage1_train.py --arm_mode oscillating --num_oscillations 4 ...
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
    # Issue #53 关键差异
    p.add_argument('--arm_mode', type=str, default='hard_two_stage',
                   choices=['hard_two_stage', 'progressive', 'early_unfreeze', 'oscillating'])
    p.add_argument('--unfreeze_epoch', type=int, default=200,
                   help='Arm C/D: 解冻开始 epoch')
    p.add_argument('--num_oscillations', type=int, default=4,
                   help='Arm D: 振荡次数')

    # Issue #49 标准
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
    return FreeCurvHRQVAE(
        in_dim=in_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        M=args.M,
        kappa_max=args.kappa_max,
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


def save_checkpoint(model, optimizer, epoch, phase, path):
    if path.exists():
        os.remove(path)
    torch.save({
        'epoch': epoch, 'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': {
            'num_emb_list': model.num_emb_list,
            'e_dim': model.e_dim,
            'layers': model.layers,
            'loss_type': model.loss_type,
            'quant_loss_weight': model.quant_loss_weight,
            'beta': model.beta,
            'sk_epsilons': [vq.sk_eps for vq in model.hrq.vq_layers],
        },
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"Task #340 — Issue #53 — κ-codebook 解冻节奏对照 ({args.arm_mode})")
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

    model = build_model(args, in_dim).to(device)
    init_theta(model, args.theta_init)
    if args.geodesic_kmeans:
        geodesic_kmeans_init(model, loader, device)

    # Freeze θ initially
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
    optimizer_b_other = torch.optim.AdamW(other_params, lr=args.lr,
                                          weight_decay=args.weight_decay)
    optimizer_b_theta = torch.optim.AdamW(theta_params, lr=args.lr_theta,
                                          weight_decay=args.weight_decay)

    best_collision = 1.0
    kappa_log = []
    total_epochs = args.phase_a_epochs + args.phase_b_epochs

    # Mode-specific logic
    unfreeze_epoch = args.unfreeze_epoch
    if args.arm_mode == 'early_unfreeze':
        unfreeze_epoch = 50  # Override for Arm C

    # Progressive lr_theta schedule (Arm B)
    if args.arm_mode == 'progressive':
        # Exponential growth: lr_theta(epoch) = lr_theta_target * (epoch/total_epochs)^2
        target_lr = args.lr_theta
        warmup = total_epochs
    else:
        target_lr = args.lr_theta

    # Oscillation schedule (Arm D)
    if args.arm_mode == 'oscillating':
        # num_oscillations cycles: freeze 30 ep, unfreeze 20 ep, repeat
        cycle_len = total_epochs // args.num_oscillations
        osc_freeze_dur = int(cycle_len * 0.6)
        osc_unfreeze_dur = cycle_len - osc_freeze_dur

    for epoch in range(1, total_epochs + 1):
        in_phase_b = epoch > args.phase_a_epochs

        # ── Determine θ grad state per mode ──
        if args.arm_mode == 'hard_two_stage':
            theta_grad = in_phase_b
        elif args.arm_mode == 'progressive':
            # Always θ unfrozen, lr_theta grows
            theta_grad = True
            progress = epoch / total_epochs
            cur_lr = target_lr * (progress ** 2)
            for pg in optimizer_b_theta.param_groups:
                pg['lr'] = cur_lr
        elif args.arm_mode == 'early_unfreeze':
            theta_grad = epoch >= unfreeze_epoch
        elif args.arm_mode == 'oscillating':
            # Within current oscillation cycle, determine if frozen or unfrozen
            cyc_idx = (epoch - 1) // cycle_len
            cyc_pos = (epoch - 1) % cycle_len
            theta_grad = cyc_pos >= osc_freeze_dur

        # Apply θ grad state
        for name, param in model.named_parameters():
            if 'theta_m' in name:
                param.requires_grad = theta_grad

        # Train
        optimizer = optimizer_b_other if in_phase_b else optimizer_a
        loss = train_epoch(model, loader, optimizer, device,
                          phase='B' if in_phase_b else 'A')

        # θ step if unfrozen
        if theta_grad:
            for vq in model.hrq.vq_layers:
                if vq.theta_m.grad is not None:
                    optimizer_b_theta.step()
                    break

        # Eval + log
        if epoch % args.log_interval == 0 or epoch == 1 or epoch == unfreeze_epoch:
            utils, collision, indices = evaluate(model, eval_loader, device, args.num_emb_list)
            print(f"  [ep {epoch}/{total_epochs} {args.arm_mode}] "
                  f"loss={loss:.4f} util={utils} collision={collision:.4f} "
                  f"θ_grad={theta_grad}")
            if collision < best_collision:
                best_collision = collision
                save_checkpoint(model, optimizer_b_other if in_phase_b else optimizer_a,
                              epoch, 'B' if in_phase_b else 'A',
                              Path(args.ckpt_dir) / "best_collision_model.pth")
            kappas = [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]
            kappa_log.append({
                'mode': args.arm_mode, 'epoch': epoch,
                'theta_grad': theta_grad, 'loss': loss,
                'utilization': utils, 'collision': collision,
                'kappa_per_layer': kappas,
            })

    save_checkpoint(model, optimizer_b_other, total_epochs, 'B',
                    Path(args.ckpt_dir) / "phase_b_final.pth")

    Path(args.kappa_log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.kappa_log_path, 'w') as f:
        json.dump(kappa_log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Issue #53 ({args.arm_mode}) complete. Best collision = {best_collision:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()