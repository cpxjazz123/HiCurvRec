"""诊断 κ 梯度在真实 CE loss 下的大小."""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
from model.hg_rec_curv_v3 import HG_Rec_Curv_V3
from transformers import T5Config

torch.manual_seed(42)
config = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.0, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, feed_forward_proj="relu",
)
model = HG_Rec_Curv_V3(config)
# 模拟一次 batch (B=4, L=20 input, L=4 target)
B = 4
input_ids = torch.randint(1, 1000, (B, 20))
attention_mask = torch.ones(B, 20, dtype=torch.long)
labels = torch.randint(1, 1000, (B, 4))
labels[0, 3] = 0  # 一个 padding

# 前向
loss, logits = model(input_ids, attention_mask, labels)
print(f"loss: {loss.item():.4f}, logits shape: {logits.shape}")
print(f"log_kappa: {model.model.shared.log_kappa.item():.6f}")

loss.backward()
print(f"log_kappa.grad: {model.model.shared.log_kappa.grad.item():.6e}")
print(f"log_kappa.grad (in scientific): {model.model.shared.log_kappa.grad.item():.6e}")
# 模拟 Adam 更新
with torch.no_grad():
    model.model.shared.log_kappa -= 1e-4 * model.model.shared.log_kappa.grad
print(f"after adam step (lr=1e-4) log_kappa: {model.model.shared.log_kappa.item():.6f}")