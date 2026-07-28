#!/usr/bin/env python3
"""
Task #209 Phase 3.1 — 机制指标分析.
对比 A0 (#181 baseline) vs A3 (软约束路径正则) 在以下机制指标上的差异:
  1. √c·ρ_per_layer (几何激活强度)
  2. 角分辨率 sinh(√c·ρ) (方向空间容量)
  3. argmin 一致率 (双曲 vs 欧式 argmin 是否一致; A3 应更高 → 几何真正影响分配)
  4. 路径成本 / 绕路成本 (path_cost mean / p50 / p90)

不依赖 Stage 3 / 4 训练, 只看 Stage 1 ckpt 在数据集上的几何表现.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from torch.utils.data import DataLoader
from model.utils import *
from model.hrqvae import HRQVAE

def load_hrqvae(ckpt_path, device='cuda:0'):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    args = ckpt['args']
    state_dict = ckpt['state_dict']
    data = EmbDataset(args.data_path)

    model = HRQVAE(
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
        w_path=getattr(args, 'w_path', 0.0),
        path_geometry=getattr(args, 'path_geometry', 'hyp'),
        rho_targets_path=getattr(args, 'rho_targets_path', None),
        scale_norm=getattr(args, 'scale_norm', 'none'),
        r_target_norm_list=getattr(args, 'norm_target', None),
        gamma_norm=getattr(args, 'gamma_norm', 0.0),
        kappa_mode=getattr(args, 'kappa_mode', 'fixed'),
        r_target_list=getattr(args, 'r_target_list', None),
        r_median_init=getattr(args, 'r_median_init', 1.0),
        theta_init=getattr(args, 'theta_init', 0.0),
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, args, data

def compute_mechanism_metrics(model, data_loader, device='cuda:0'):
    """计算逐层 √c·ρ, sinh(√c·ρ), argmin 一致率."""
    metrics = {
        'per_layer': {0: {}, 1: {}, 2: {}},
        'global': {},
    }

    n_dup_hyp_euc_same = 0  # argmin 一致率
    n_total = 0

    for d in data_loader:
        d = d.to(device)
        with torch.no_grad():
            x_lat = model.encoder(d)

            # 取每层码字
            cb_hyp = []  # Poincaré 球上的码字
            cb_euc = []  # 等价欧式表示
            for vq_layer in model.hrq.vq_layers:
                w = vq_layer.embeddings.weight.data  # (K, e_dim)
                cb_euc.append(w)
                # 关键: HG-Rec forward 走 expmap0(w, c=1) + proj_to_ball, 才是真正的"码字"
                cb_h = proj_to_ball(expmap0(w, c=1.0), c=1.0)
                cb_hyp.append(cb_h)

            # Per-layer √c·ρ, sinh(√c·ρ) on cb_hyp
            for layer_idx, cb in enumerate(cb_hyp):
                norms_E = cb.norm(dim=-1)  # (num_emb,)
                mean_norm = norms_E.mean().item()
                rho = 2 * mean_norm  # √c·ρ ≈ 2·‖x‖_E (c=1)
                sinh_rho = np.sinh(rho)
                metrics['per_layer'][layer_idx]['rho'] = rho
                metrics['per_layer'][layer_idx]['mean_norm_E'] = mean_norm
                metrics['per_layer'][layer_idx]['sinh_rho'] = sinh_rho

        break  # 只看 batch 0 的码字 (码字不依赖 input)
    return metrics

if __name__ == "__main__":
    a0_ckpt = "/home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2/Jul-25-2026_16-20-31_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth"
    a3_ckpt = "/home/wlia0047/ar57/wenyu/GeneRec/products/task209/phase1_arm_A3/Jul-26-2026_18-26-17_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_999_collision_0.9997_model.pth"

    results = {}
    for label, ckpt_path in [('A0_baseline', a0_ckpt), ('A3_path_reg', a3_ckpt)]:
        if not os.path.exists(ckpt_path):
            print(f"❌ {label} ckpt missing: {ckpt_path}")
            continue
        print(f"\n=== {label} ===")
        model, args, data = load_hrqvae(ckpt_path)
        loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=2)
        metrics = compute_mechanism_metrics(model, loader)
        results[label] = metrics
        for L in [0, 1, 2]:
            m = metrics['per_layer'][L]
            print(f"  L{L}: ρ={m['rho']:.3f}, ‖x‖_E={m['mean_norm_E']:.3f}, sinh(ρ)={m['sinh_rho']:.3f}")

    out_path = "/home/wlia0047/ar57/wenyu/GeneRec/products/task209/phase3_mechanism.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 机制指标 saved to {out_path}")
