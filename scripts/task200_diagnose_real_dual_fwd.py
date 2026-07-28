#!/usr/bin/env python3
"""隔离测试 rec_grad=0 的真正原因 — 在 HVectorQuantization 双码本 forward 后."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.utils import HVectorQuantization

torch.manual_seed(42)

e_dim, K, B = 32, 64, 200

# 完全复刻 isolation_test 创建 vq 的方式
vq = HVectorQuantization(
    n_e=K, e_dim=e_dim, sk_eps=0.0, beta=0.5,
    kmeans_init=False, kmeans_iters=100, sk_iters=50,
    curvature=1.0, euclidean_qloss=False, loss_mult_codebook=1.0,
    dual_codebook=True,
)
vq.use_centering = True
vq.z_mean = torch.zeros(e_dim)
vq.z_mean_ema = 0.99

# 看 nn.Embedding 默认 weight 状态
print(f"=== 初始 ===")
print(f"emb_rec.weight.requires_grad: {vq.emb_rec.weight.requires_grad}")
print(f"emb_rec.weight.is_leaf: {vq.emb_rec.weight.is_leaf}")

# 模拟 init_emb: in-place copy
latent = torch.randn(B, e_dim) * 0.3
vq.train()
vq.init_emb(latent)
print(f"\n=== init_emb 后 ===")
print(f"emb_rec.weight.requires_grad: {vq.emb_rec.weight.requires_grad}")
print(f"emb_rec.weight.is_leaf: {vq.emb_rec.weight.is_leaf}")
print(f"emb_rec.weight.grad_fn: {vq.emb_rec.weight.grad_fn}")

# 跑 forward
vq.eval()
x_q, loss, indices = vq(latent)
print(f"\n=== forward 后 ===")
print(f"x_q.requires_grad: {x_q.requires_grad}")
print(f"x_q.grad_fn: {x_q.grad_fn}")
print(f"loss.requires_grad: {loss.requires_grad}")
print(f"loss.grad_fn type: {type(loss.grad_fn).__name__ if loss.grad_fn else None}")

# 看 x_q 是否含 emb_rec 的连接
# 简单方式: 用 torch.autograd.grad 检查 emb_rec.weight 是否在 loss 的依赖图中
grad_check = torch.autograd.grad(loss, [vq.emb_rec.weight], retain_graph=True,
                                  allow_unused=True)
print(f"\n[autograd.grad] emb_rec.weight grad is None: {grad_check[0] is None}")
if grad_check[0] is not None:
    print(f"  grad norm: {grad_check[0].norm().item():.6f}")
else:
    print(f"  → emb_rec.weight 不在 loss 的依赖图中")

# 看 emb_geo 是否在
grad_check_geo = torch.autograd.grad(loss, [vq.emb_geo.weight], retain_graph=True,
                                      allow_unused=True)
print(f"\n[autograd.grad] emb_geo.weight grad is None: {grad_check_geo[0] is None}")
if grad_check_geo[0] is not None:
    print(f"  grad norm: {grad_check_geo[0].norm().item():.6f}")

# 尝试 loss.backward 看 emb_rec.weight.grad
vq.zero_grad()
loss.backward()
print(f"\n[after loss.backward()]")
print(f"emb_rec.weight.grad is None: {vq.emb_rec.weight.grad is None}")
print(f"emb_rec.weight.grad norm: {vq.emb_rec.weight.grad.norm().item() if vq.emb_rec.weight.grad is not None else 'None'}")
print(f"emb_geo.weight.grad norm: {vq.emb_geo.weight.grad.norm().item() if vq.emb_geo.weight.grad is not None else 'None'}")