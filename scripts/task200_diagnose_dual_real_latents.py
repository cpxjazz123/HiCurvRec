#!/usr/bin/env python3
"""用真实 latent (从 task181 baseline 收集) 验证 rec_grad 是否真在长训中被削."""
import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.utils import HVectorQuantization

torch.manual_seed(42)

# 加载真实 L0 latent (来自隔离测试 v3)
lat = torch.load(REPO / "logs/task200/isolation_test_v3/lat_L0.pt")
print(f"latent shape: {lat.shape}, norm mean: {lat.norm(dim=-1).mean():.3f}")

e_dim = lat.shape[-1]
K = 64

vq = HVectorQuantization(
    n_e=K, e_dim=e_dim, sk_eps=0.0, beta=0.5,
    kmeans_init=False, kmeans_iters=100, sk_iters=50,
    curvature=1.0, euclidean_qloss=False, loss_mult_codebook=1.0,
    dual_codebook=True,
)
vq.use_centering = True
vq.z_mean = torch.zeros(e_dim)
vq.z_mean_ema = 0.99

vq.train()
vq.init_emb(lat)

# 直接走 5 步,不通过 opt (避免 AdamW 状态混淆)
print(f"\n=== 5 步 forward-backward (no opt) ===")
for step in range(5):
    vq.zero_grad()
    x_q, loss, indices = vq(lat)
    loss.backward()
    rg = vq.emb_rec.weight.grad.norm().item() if vq.emb_rec.weight.grad is not None else 0
    gg = vq.emb_geo.weight.grad.norm().item() if vq.emb_geo.weight.grad is not None else 0
    print(f"step {step}: loss={loss.item():.6f}, "
          f"emb_rec grad={rg:.6f}, emb_geo grad={gg:.6f}, "
          f"unique_sids={len(set(indices.tolist()))}")

# 用 AdamW 跑 5 步
print(f"\n=== 5 步 AdamW ===")
opt = torch.optim.AdamW(vq.parameters(), lr=1e-3)
vq.train()
for step in range(5):
    opt.zero_grad()
    x_q, loss, indices = vq(lat)
    loss.backward()
    rg_before = vq.emb_rec.weight.grad.norm().item() if vq.emb_rec.weight.grad is not None else 0
    gg_before = vq.emb_geo.weight.grad.norm().item() if vq.emb_geo.weight.grad is not None else 0
    opt.step()
    print(f"step {step}: loss={loss.item():.6f}, "
          f"emb_rec grad={rg_before:.6f}, emb_geo grad={gg_before:.6f}, "
          f"unique_sids={len(set(indices.tolist()))}")