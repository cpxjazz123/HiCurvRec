"""诊断 κ 梯度为何卡死."""
import torch
import sys, os
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
from model.hg_rec_curv_v3 import PoincareEmbeddingV3, HG_Rec_Curv_V3

torch.manual_seed(42)
vocab_size, d_model = 1025, 128
pe = PoincareEmbeddingV3(vocab_size, d_model, padding_idx=0, kappa_init=1.0, init_std=0.1)
ids = torch.randint(1, 1025, (4, 20))
y = pe(ids)
print(f"input ids shape: {ids.shape}")
print(f"output y shape: {y.shape}, norm mean: {y.norm(dim=-1).mean().item():.4f}")

# 假 loss: sum of squares (just to test gradient flow)
loss = (y ** 2).sum()
loss.backward()

print(f"log_kappa: {pe.log_kappa.item():.6f}, kappa: {pe.kappa.item():.6f}")
print(f"log_kappa.grad: {pe.log_kappa.grad.item():.6e}")
print(f"embedding.weight.grad norm: {pe.embedding.weight.grad.norm().item():.4f}")
print(f"embedding.weight.grad mean abs: {pe.embedding.weight.grad.abs().mean().item():.6e}")

# 模拟 Adam step
with torch.no_grad():
    pe.log_kappa -= 1e-4 * pe.log_kappa.grad
print(f"after adam step (lr=1e-4) log_kappa: {pe.log_kappa.item():.6f}")