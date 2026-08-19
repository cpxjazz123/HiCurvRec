"""验证 hypothesis: T5 encoder/decoder.embed_tokens 独立属性, 需显式更新."""
import torch
import sys, os
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
from model.hg_rec_curv_v3 import PoincareEmbeddingV3, HG_Rec_Curv_V3
from transformers import T5Config

torch.manual_seed(42)
config = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.0, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, feed_forward_proj="relu",
)
model = HG_Rec_Curv_V3(config)
print(f"=== BEFORE FIX ===")
print(f"model.model.shared: {type(model.model.shared).__name__}")
print(f"model.model.encoder.embed_tokens: {type(model.model.encoder.embed_tokens).__name__}")
print(f"model.model.decoder.embed_tokens: {type(model.model.decoder.embed_tokens).__name__}")
print(f"encoder.embed_tokens is shared: {model.model.encoder.embed_tokens is model.model.shared}")

# === THE FIX ===
new_shared = model.model.shared  # PoincareEmbeddingV3 instance
model.model.encoder.embed_tokens = new_shared
model.model.decoder.embed_tokens = new_shared
print(f"\n=== AFTER FIX ===")
print(f"encoder.embed_tokens is shared: {model.model.encoder.embed_tokens is model.model.shared}")

# Re-run forward/backward
B = 4
input_ids = torch.randint(1, 1000, (B, 20))
attention_mask = torch.ones(B, 20, dtype=torch.long)
labels = torch.randint(1, 1000, (B, 4))

loss, logits = model(input_ids, attention_mask, labels)
loss.backward()
print(f"\nAfter fix, real forward:")
print(f"  loss: {loss.item():.4f}")
print(f"  shared.log_kappa.grad: {model.model.shared.log_kappa.grad}")
if model.model.shared.log_kappa.grad is not None:
    print(f"  shared.log_kappa.grad magnitude: {model.model.shared.log_kappa.grad.item():.6e}")
print(f"  shared.embedding.weight.grad sum: {model.model.shared.embedding.weight.grad.sum().item():.4f}")