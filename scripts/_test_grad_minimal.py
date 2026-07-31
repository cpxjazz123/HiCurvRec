#!/usr/bin/env python3
"""Minimal test: does BoundedKappaAdapter produce non-zero gradient for u_l?"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")
from task413_issue120_bounded_kappa_adapter import BoundedKappaAdapter

torch.manual_seed(42)

# Test 1: Direct test on adapter
print("=" * 60)
print("Test 1: Direct adapter forward + backward")
print("=" * 60)
adapter = BoundedKappaAdapter(d_model=128, kappa_min=0.1, eps=1e-3, position_count=3)
adapter.train()
x = torch.randn(4, 8, 128)  # (B, T, D)
position_ids = torch.zeros(4, 8, dtype=torch.long)
position_ids[:, 4:] = 1  # Half L0, half L1
y = adapter(x, position_ids)
print(f"  y.shape={y.shape}, y.norm()={y.norm().item():.3f}")
loss = F.mse_loss(y, torch.zeros_like(y))
print(f"  loss={loss.item():.6f}")
loss.backward()
print(f"  u_l.grad.norm() = {adapter.u_l_per_position.grad.norm().item():.6e}")
print(f"  scale.grad.norm() = {adapter.scale_per_position.grad.norm().item():.6e}")
print(f"  gate.grad.norm() = {adapter.gate_logits.grad.norm().item():.6e}")

# Test 2: With T5 loaded
print("\n" + "=" * 60)
print("Test 2: Through T5 encoder block + adapter")
print("=" * 60)
import torch
from transformers import T5ForConditionalGeneration, T5Config
DEVICE = "cuda:0"
config = T5Config(
    vocab_size=1025,
    d_model=128,
    d_kv=64,
    d_ff=1024,
    num_layers=4,
    num_decoder_layers=4,
    num_heads=3,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    dropout_rate=0.0,
)
t5 = T5ForConditionalGeneration(config).to(DEVICE)
adapter = BoundedKappaAdapter(d_model=128, kappa_min=0.1, eps=1e-3, position_count=3).to(DEVICE)
adapter.train()

sample_sid = torch.randint(0, 1024, (4, 4)).long().to(DEVICE)  # (4, 4)
embed = t5.shared(sample_sid)
print(f"  embed.requires_grad={embed.requires_grad}")
print(f"  t5.shared.weight.requires_grad={t5.shared.weight.requires_grad}")

h = embed
for i, block in enumerate(t5.encoder.block):
    h = block(h)[0]
    if i == 0:
        position_ids = torch.zeros(4, 4, dtype=torch.long).to(DEVICE)
        h = adapter(h, position_ids)

print(f"  h.requires_grad={h.requires_grad}, h.norm()={h.norm().item():.3f}")
loss = F.mse_loss(h, torch.zeros_like(h))
print(f"  loss={loss.item():.6f}")
loss.backward()
print(f"  adapter.u_l.grad.norm() = {adapter.u_l_per_position.grad.norm().item():.6e}")
print(f"  adapter.scale.grad.norm() = {adapter.scale_per_position.grad.norm().item():.6e}")
print(f"  adapter.gate.grad.norm() = {adapter.gate_logits.grad.norm().item():.6e}")

# Test 3: Check if encoder blocks have requires_grad=False
print("\n" + "=" * 60)
print("Test 3: Check requires_grad on all params")
print("=" * 60)
print(f"  t5.encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad = {t5.encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad}")
print(f"  t5.shared.weight.requires_grad = {t5.shared.weight.requires_grad}")