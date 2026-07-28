#!/usr/bin/env python3
"""诊断: nn.Embedding.weight.data.copy_() 是否破坏 emb_rec 梯度连接."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

e_dim, K, B = 32, 64, 100

emb_rec = nn.Embedding(K, e_dim)
emb_rec.weight.data.uniform_(-0.1, 0.1)

# 模拟 init_emb 的 in-place copy
new_w = torch.randn(K, e_dim) * 0.3
emb_rec.weight.data.copy_(new_w)

# 看 requires_grad
print(f"emb_rec.weight.requires_grad: {emb_rec.weight.requires_grad}")
print(f"emb_rec.weight.is_leaf: {emb_rec.weight.is_leaf}")
print(f"emb_rec.weight.grad_fn: {emb_rec.weight.grad_fn}")

z = torch.randn(B, e_dim) * 0.5
indices = torch.randint(0, K, (B,))

x_q = emb_rec(indices)
print(f"\nx_q.requires_grad: {x_q.requires_grad}")
print(f"x_q.grad_fn: {x_q.grad_fn}")

code = F.mse_loss(x_q, z.detach())
code.backward()

print(f"\n[after code.backward()]")
print(f"emb_rec.weight.grad is None: {emb_rec.weight.grad is None}")
print(f"emb_rec.weight.grad norm: "
      f"{emb_rec.weight.grad.norm().item() if emb_rec.weight.grad is not None else 'None'}")

# 再测 commit
emb_rec.zero_grad()
commit = F.mse_loss(x_q.detach(), z)
commit.backward()
print(f"\n[after commit.backward()] (commit 应该无 grad 因为 x_q detached)")
print(f"emb_rec.weight.grad norm: "
      f"{emb_rec.weight.grad.norm().item() if emb_rec.weight.grad is not None else 'None'}")