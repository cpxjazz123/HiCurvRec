"""深入诊断 κ 梯度断链."""
import torch
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

print(f"shared type: {type(model.model.shared).__name__}")
print(f"shared.log_kappa: {model.model.shared.log_kappa}")
print(f"shared.log_kappa.requires_grad: {model.model.shared.log_kappa.requires_grad}")
print(f"shared.log_kappa.is_leaf: {model.model.shared.log_kappa.is_leaf}")
print(f"shared.embedding.weight.requires_grad: {model.model.shared.embedding.weight.requires_grad}")
print(f"lm_head.weight is shared.embedding.weight: {model.model.lm_head.weight is model.model.shared.embedding.weight}")
print(f"lm_head.weight.data_ptr() == shared.embedding.weight.data_ptr(): "
      f"{model.model.lm_head.weight.data_ptr() == model.model.shared.embedding.weight.data_ptr()}")

# 手动验证 kappa 属性计算
test_log_kappa = torch.tensor(0.5, requires_grad=True)
test_kappa = test_log_kappa.exp().clamp(min=0.05, max=8.0)
test_loss = (test_kappa ** 2)
test_loss.backward()
print(f"\nIsolated .kappa property test:")
print(f"  test_log_kappa.grad: {test_log_kappa.grad.item():.6e}")
print(f"  → κ属性正常返回梯度")

# 跑真实 forward
B = 4
input_ids = torch.randint(1, 1000, (B, 20))
attention_mask = torch.ones(B, 20, dtype=torch.long)
labels = torch.randint(1, 1000, (B, 4))

loss, logits = model(input_ids, attention_mask, labels)
print(f"\nReal forward:")
print(f"  loss: {loss.item():.4f}")
print(f"  shared(log_kappa).grad_fn: {model.model.shared.log_kappa.grad_fn}")

loss.backward()
print(f"\nAfter backward:")
print(f"  shared.log_kappa.grad: {model.model.shared.log_kappa.grad}")
print(f"  shared.embedding.weight.grad sum: {model.model.shared.embedding.weight.grad.sum().item():.4f}")
print(f"  lm_head.weight.grad sum: {model.model.lm_head.weight.grad.sum().item() if model.model.lm_head.weight.grad is not None else 'None'}")

# 检查中间节点: 走一遍 PoincareEmbeddingV3.forward 看 expmap 输出
v = model.model.shared.embedding(input_ids)
print(f"\nv (raw embedding) shape: {v.shape}, grad_fn: {v.grad_fn}")
k = model.model.shared.kappa
print(f"k shape: {k.shape}, grad_fn: {k.grad_fn}, requires_grad: {k.requires_grad}")
sqrt_k = k.sqrt()
v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
factor = torch.tanh(sqrt_k * v_norm / 2.0) / (sqrt_k * v_norm)
expmap = v * factor
print(f"expmap grad_fn: {expmap.grad_fn}")