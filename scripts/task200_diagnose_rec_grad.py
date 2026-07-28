#!/usr/bin/env python3
"""Mini-test: 直接验证 F.mse_loss(emb_rec(indices), z.detach()) 是否真传 grad."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

e_dim, K = 32, 64
B = 100

emb_rec = nn.Embedding(K, e_dim)
emb_rec.weight.data.uniform_(-0.1, 0.1)

z = torch.randn(B, e_dim) * 0.5  # 切空间向量
indices = torch.randint(0, K, (B,))

x_q = emb_rec(indices)
print(f"x_q.requires_grad: {x_q.requires_grad}")
print(f"x_q.grad_fn: {x_q.grad_fn}")

commit = F.mse_loss(x_q.detach(), z)
code = F.mse_loss(x_q, z.detach())
geo = torch.tensor(0.0)
loss = commit + 0.5 * code + 0.1 * geo

print(f"\n[loss] commit={commit.item():.6f}, code={code.item():.6f}")

loss.backward()

print(f"\n[after backward]")
print(f"emb_rec.weight.grad norm: {emb_rec.weight.grad.norm().item() if emb_rec.weight.grad is not None else 'None'}")
print(f"emb_rec.weight.grad shape: {emb_rec.weight.grad.shape if emb_rec.weight.grad is not None else 'None'}")

# 看哪个 indices 有梯度
grad_per_idx = emb_rec.weight.grad.norm(dim=-1)
print(f"\n[grad per index] max={grad_per_idx.max().item():.6f}, "
      f"mean={grad_per_idx.mean().item():.6f}, "
      f"min={grad_per_idx.min().item():.6f}")
print(f"grad nonzero indices: {(grad_per_idx > 1e-10).sum().item()} / {K}")
print(f"indices used in this batch: {set(indices.tolist())}")
print(f"grad nonzero at indices: {sum(1 for i in indices.tolist() if grad_per_idx[i].item() > 1e-10)}")