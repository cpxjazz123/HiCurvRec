"""Phase 0 选项 D: 用 task220 hrqvae_pck (Per-Codeword κ, 无 NormCap) ckpt 跑 κ 扫描.

不带 NormCap, latent 范数天然大, 球面侧几何可能激活.
"""
import sys, os, json
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.hrqvae_free_curv import FreeCurvHRQVAE, geodesic_distance_sq

CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task220/hrqvae_pck/Jul-26-2026_22-54-05_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth"
DATA = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

KAPPA_LIST = [-10.0, -3.0, -1.0, 0.0, 0.3, 1.0, 3.0, 10.0]
LAYERS = [(0, 64), (1, 128), (2, 256)]

# task220 config (Per-Codeword κ, e_dim=32, [64,128,256], 无 NormCap)
config = {
    "in_dim": 768, "num_emb_list": [64, 128, 256], "e_dim": 32, "M": 1,
    "kappa_max": 5.0, "layers": [512, 256, 128], "dropout_prob": 0.0, "bn": False,
    "loss_type": "poincare", "quant_loss_weight": 1.0, "beta": 0.5,
    "kmeans_init": False, "kmeans_iters": 100,
    "sk_eps": [0.003, 0.003, 0.003], "sk_iters": 3,
}
model = FreeCurvHRQVAE(**config)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
result = model.load_state_dict(ckpt, strict=False)
print(f"Loaded task220 ckpt, missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
model.eval()

df = pd.read_parquet(DATA)
emb = np.stack(df['embedding'].values, axis=0)

print("=" * 70)
print("Phase 0 选项 D: task220 hrqvae_pck ckpt (Per-Codeword κ, 无 NormCap)")
print("=" * 70)
with torch.no_grad():
    z = model.encoder(torch.from_numpy(emb).float())
print(f"latent z: shape={z.shape}, ‖z‖ mean={z.norm(dim=-1).mean():.4f} max={z.norm(dim=-1).max():.4f}")
print(f"  (vs E1 normcap: mean=0.131 max=0.19)")

# 看 task220 codebook
for li, vq in enumerate(model.hrq.vq_layers):
    cb = vq.embeddings.weight.detach()
    print(f"  L{li} codebook: shape={cb.shape}, ‖c‖ mean={cb.norm(dim=-1).mean():.4f} max={cb.norm(dim=-1).max():.4f}")

print()
for li, K in LAYERS:
    cb = model.hrq.vq_layers[li].embeddings.weight.detach()
    eucl_d = torch.cdist(z, cb)
    eucl_idx = eucl_d.argmin(dim=-1)
    print(f"--- L{li} (K={K}, ‖z‖={z.norm(dim=-1).mean().item():.3f}) ---")
    print(f"{'kappa':>7} {'side':>7} {'agreement':>10} {'dyn':>8} {'usage':>8} {'assign_H':>9}")
    for kappa_val in KAPPA_LIST:
        kappa = torch.tensor(kappa_val)
        side = "sphere" if kappa_val > 0 else ("hyper" if kappa_val < 0 else "euclid")
        d = geodesic_distance_sq(z.unsqueeze(1), cb.unsqueeze(0), kappa).squeeze(-1).sqrt()
        k_idx = d.argmin(dim=-1)
        agreement = (k_idx == eucl_idx).float().mean().item()
        dyn = (d.max(dim=-1).values / (d.min(dim=-1).values + 1e-8)).mean().item()
        usage_count = torch.bincount(k_idx, minlength=K)
        usage = (usage_count > 0).sum().item() / K
        p = usage_count.float() / (usage_count.sum() + 1e-8)
        p_nz = p[p > 0]
        assign_H = -(p_nz * (p_nz + 1e-9).log()).sum().item() if len(p_nz) > 0 else 0.0
        print(f"{kappa_val:>7.2f} {side:>7} {agreement:>10.4f} {dyn:>8.4f} {usage:>8.4f} {assign_H:>9.4f}")
    print()
