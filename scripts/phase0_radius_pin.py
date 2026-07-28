"""5 min 检查: 钉 latent 半径到 r ∈ {2.0, 2.7, 3.4}, 测 κ 几何激活.

目的: 看"钉大半径"是否能激活球面/双曲侧几何信号, 跟"范数小"对照.
公式: ρ = 2·atanh(r). 钉 r 等价钉 ρ.
  - r=2.0: 数学上无效 (tanh 单调 ∈ (-1,1), 但 ‖x‖ 可以 > 1 在切空间; 球面/双曲用 ‖x‖∈(0,1) 球内坐标)
  - r=2.7, 3.4: 同样在 (0,1) 区间外的 latent ‖x‖, 但模拟"如果不限制 NormCap"

实测中:
  - ‖x‖=2.0 → 数值上 d_κ 在 κ<0 不会饱和 (因 ‖x‖ < 1/√c 边界很远)
  - ‖x‖=3.4 → 大范数, 球面/双曲都几何可见
"""
import sys, json
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.hrqvae import HRQVAE

CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task218/stage1_pck_spread/Jul-27-2026_01-25-33_beta_0.500_codebook_[64,128,256]_sk_0.003/best_loss_model.pth"
DATA = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

KAPPA_LIST = [-10.0, -3.0, -1.0, 0.0, 0.3, 1.0, 3.0, 10.0]
RADII = [2.0, 2.7, 3.4]  # 用户指定
LAYERS = [(0, 64), (1, 128), (2, 256)]

# 加载 ckpt
ckpt = torch.load(CKPT, weights_only=False, map_location='cpu')
args = ckpt['args']
state_dict = ckpt['state_dict']

# 重建模型 (跟 ckpt args 一致)
model = HRQVAE(in_dim=768,
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
               product_manifold=getattr(args, 'product_manifold', False),
               angular_dim=getattr(args, 'angular_dim', None),
               radial_dim=getattr(args, 'radial_dim', None),
               assignment_mode=getattr(args, 'assignment_mode', 'shared'),
               c_k_min=getattr(args, 'c_k_min', 0.5),
               c_k_max=getattr(args, 'c_k_max', 20.0),
               c_k_seed=getattr(args, 'c_k_seed', 42),
               c_k_update_mode=getattr(args, 'c_k_update_mode', 'fixed'),
               distance_mode=getattr(args, 'distance_mode', 'sphere'),
               use_normcap=getattr(args, 'use_normcap', False),
               normcap_target=getattr(args, 'normcap_target', 0.4),
               w_anchor=getattr(args, 'w_anchor', 0.0),
               rho_min=getattr(args, 'rho_min', 1.5),
               rho_max=getattr(args, 'rho_max', 2.9),
               anti_collapse=getattr(args, 'anti_collapse', 'none'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# 加载数据 + encoder 拿 latent
df = pd.read_parquet(DATA)
emb = np.stack(df['embedding'].values, axis=0)
with torch.no_grad():
    z = model.encoder(torch.from_numpy(emb).float())  # (N, e_dim)
print(f"E1 latent: shape={z.shape}, ‖z‖ mean={z.norm(dim=-1).mean():.4f} max={z.norm(dim=-1).max():.4f}")

# 把 latent 投到 2D + 钉半径 (R11.3: 简单做法 — 取前 2 维方向 + 钉半径)
# 取每个 sample 的方向 (前 2 维 normalize), 然后乘以目标半径
print()
print("=" * 90)
print(f"{'layer':>5} {'radius':>7} {'kappa':>7} {'side':>7} {'agreement':>10} {'dyn':>8} {'usage':>8} {'assign_H':>9}")
print("-" * 90)

results = []
for li, K in LAYERS:
    cb = model.hrq.vq_layers[li].embeddings.weight.detach()  # (K, e_dim)
    cb_dir = cb[:, :2] / cb[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (K, 2)
    cb_pinned = cb_dir * RADII[0]  # 用最大半径 normalize, 后改
    for radius in RADII:
        # 投 latent 到 2D + 钉半径
        z_dir = z[:, :2] / z[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (N, 2)
        z_pinned = z_dir * radius  # (N, 2) - 钉半径
        cb_pinned = cb_dir * radius  # (K, 2) - 同样钉半径
        # 欧式 argmin (baseline)
        eucl_d = torch.cdist(z_pinned, cb_pinned)
        eucl_idx = eucl_d.argmin(dim=-1)
        for kappa_val in KAPPA_LIST:
            kappa = torch.tensor(kappa_val)
            side = "sphere" if kappa_val > 0 else ("hyper" if kappa_val < 0 else "euclid")
            # κ-Stereographic distance
            from model.hrqvae_free_curv import geodesic_distance_sq
            d = geodesic_distance_sq(z_pinned.unsqueeze(1), cb_pinned.unsqueeze(0), kappa).squeeze(-1).sqrt()
            k_idx = d.argmin(dim=-1)
            agreement = (k_idx == eucl_idx).float().mean().item()
            dyn = (d.max(dim=-1).values / (d.min(dim=-1).values + 1e-8)).mean().item()
            usage_count = torch.bincount(k_idx, minlength=K)
            usage = (usage_count > 0).sum().item() / K
            p = usage_count.float() / (usage_count.sum() + 1e-8)
            p_nz = p[p > 0]
            assign_H = -(p_nz * (p_nz + 1e-9).log()).sum().item() if len(p_nz) > 0 else 0.0
            print(f"  L{li}  {radius:>7.1f} {kappa_val:>7.2f} {side:>7} {agreement:>10.4f} {dyn:>8.4f} {usage:>8.4f} {assign_H:>9.4f}")
            results.append({
                "layer": li, "radius": radius, "kappa": kappa_val, "side": side,
                "agreement": agreement, "dyn_ratio": dyn, "usage": usage,
                "assign_entropy": assign_H,
            })
    print()

# 找甜区: agreement ∈ [0.6, 0.9] AND usage ≥ 0.8
print("=" * 90)
print("甜区: agreement ∈ [0.6, 0.9] AND usage ≥ 0.8 (用户判据)")
print("=" * 90)
hits = [r for r in results if 0.6 <= r["agreement"] <= 0.9 and r["usage"] >= 0.8]
if hits:
    for h in hits[:20]:
        print(f"  ✅ L{h['layer']} r={h['radius']:.1f} κ={h['kappa']:+.2f}  agr={h['agreement']:.3f} usage={h['usage']:.3f}")
else:
    print("  ❌ 无甜区 — 钉半径也不能同时满足 agreement ∈ 60-90% AND usage ≥ 80%")
print()

# 保存
out_path = "/home/wlia0047/ar57/wenyu/GeneRec/verdicts/_phase0_radius_pin.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
