"""Phase 0 重测 (R11.3 选项 C): 把 E1 latent 范数放大到无 NormCap 范围, 重测 κ 几何.

E1 normcap=0.4 → z norm ≈ 0.19
无 NormCap 期望: z norm ≈ 0.5-1.0 (跟用户方案 ρ=0.847 一致)
放大 5x → norm ≈ 1.0; 放大 10x → norm ≈ 2.0
"""
import sys, os, json
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.hrqvae_free_curv import FreeCurvHRQVAE, geodesic_distance_sq

CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task226/e1_normcap/Jul-27-2026_00-09-44_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_199_collision_0.0953_model.pth"
DATA = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

KAPPA_LIST = [-10.0, -3.0, -1.0, 0.0, 0.3, 1.0, 3.0, 10.0]
SCALES = [1.0, 5.0, 10.0, 20.0]  # 1x = E1 baseline, 5/10/20x = 模拟无 NormCap
LAYERS = [(0, 64), (1, 128), (2, 256)]

config = {
    "in_dim": 768, "num_emb_list": [64, 128, 256], "e_dim": 32, "M": 1,
    "kappa_max": 2.0, "layers": [512, 256, 128], "dropout_prob": 0.0, "bn": False,
    "loss_type": "poincare", "quant_loss_weight": 1.0, "beta": 0.5,
    "kmeans_init": False, "kmeans_iters": 100,
    "sk_eps": [0.003, 0.003, 0.003], "sk_iters": 3,
}
model = FreeCurvHRQVAE(**config)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt, strict=False)
model.eval()

df = pd.read_parquet(DATA)
emb = np.stack(df['embedding'].values, axis=0)

print("=" * 70)
print("Phase 0 重测: latent 范数放大, 看 κ 几何是否激活")
print("=" * 70)
with torch.no_grad():
    z = model.encoder(torch.from_numpy(emb).float())
print(f"E1 baseline latent: ‖z‖ mean={z.norm(dim=-1).mean():.4f} max={z.norm(dim=-1).max():.4f}")

print()
for li, K in LAYERS:
    cb = model.hrq.vq_layers[li].embeddings.weight.detach()
    eucl_d = torch.cdist(z, cb)
    eucl_idx = eucl_d.argmin(dim=-1)
    print(f"--- L{li} (K={K}) ---")
    print(f"{'scale':>6} {'z_norm':>8} {'kappa':>7} {'side':>7} {'agreement':>10} {'dyn':>8} {'usage':>8} {'assign_H':>9}")
    for scale in SCALES:
        z_scaled = z * scale
        c_scaled = cb * scale
        eucl_d_s = torch.cdist(z_scaled, c_scaled)
        eucl_idx_s = eucl_d_s.argmin(dim=-1)
        for kappa_val in KAPPA_LIST:
            kappa = torch.tensor(kappa_val)
            side = "sphere" if kappa_val > 0 else ("hyper" if kappa_val < 0 else "euclid")
            d = geodesic_distance_sq(z_scaled.unsqueeze(1), c_scaled.unsqueeze(0), kappa).squeeze(-1).sqrt()
            k_idx = d.argmin(dim=-1)
            agreement = (k_idx == eucl_idx_s).float().mean().item()
            dyn = (d.max(dim=-1).values / (d.min(dim=-1).values + 1e-8)).mean().item()
            usage_count = torch.bincount(k_idx, minlength=K)
            usage = (usage_count > 0).sum().item() / K
            p = usage_count.float() / (usage_count.sum() + 1e-8)
            p_nz = p[p > 0]
            assign_H = -(p_nz * (p_nz + 1e-9).log()).sum().item() if len(p_nz) > 0 else 0.0
            print(f"{scale:>6.1f} {z_scaled.norm(dim=-1).mean().item():>8.3f} {kappa_val:>7.2f} {side:>7} {agreement:>10.4f} {dyn:>8.4f} {usage:>8.4f} {assign_H:>9.4f}")
    print()
