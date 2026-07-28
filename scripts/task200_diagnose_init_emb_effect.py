#!/usr/bin/env python3
"""直接验证 init_emb 是否破坏 emb_rec 梯度链路."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.utils import HVectorQuantization

torch.manual_seed(42)
e_dim, K, B = 32, 64, 200

vq = HVectorQuantization(
    n_e=K, e_dim=e_dim, sk_eps=0.0, beta=0.5,
    kmeans_init=False, kmeans_iters=100, sk_iters=50,
    curvature=1.0, euclidean_qloss=False, loss_mult_codebook=1.0,
    dual_codebook=True,
)
vq.use_centering = True
vq.z_mean = torch.zeros(e_dim)
vq.z_mean_ema = 0.99

# 直接调用 forward 不调 init_emb
latent = torch.randn(B, e_dim) * 0.3
vq.train()

print(f"=== 直接 forward (无 init_emb) ===")
vq.initted = True  # 跳过 init_emb
vq.emb_geo.weight.data.uniform_(-0.1, 0.1)  # 简单 init
vq.emb_rec.weight.data.uniform_(-0.1, 0.1)

x_q, loss, indices = vq(latent)
loss.backward()
print(f"emb_rec.grad norm: {vq.emb_rec.weight.grad.norm().item():.6f}")
print(f"emb_geo.grad norm: {vq.emb_geo.weight.grad.norm().item():.6f}")

# 现在调 init_emb
print(f"\n=== 调 init_emb 后再 forward ===")
vq.zero_grad()
vq.initted = False
latent2 = torch.randn(B, e_dim) * 0.3
vq.init_emb(latent2)
vq.initted = True  # 确保 forward 不再调 init_emb

print(f"init_emb 后 emb_rec.weight.requires_grad: {vq.emb_rec.weight.requires_grad}")
print(f"init_emb 后 emb_rec.weight.is_leaf: {vq.emb_rec.weight.is_leaf}")

x_q, loss, indices = vq(latent)
loss.backward()
print(f"emb_rec.grad norm: {vq.emb_rec.weight.grad.norm().item():.6f}")
print(f"emb_geo.grad norm: {vq.emb_geo.weight.grad.norm().item():.6f}")

# 检查 emb_rec.weight 是否被替换(变成 detached Tensor)
print(f"\n=== 验证 emb_rec.weight 类型 ===")
print(f"type(emb_rec.weight): {type(vq.emb_rec.weight)}")
print(f"isinstance Parameter: {isinstance(vq.emb_rec.weight, torch.nn.Parameter)}")