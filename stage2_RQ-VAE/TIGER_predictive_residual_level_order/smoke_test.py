"""Minimal CPU smoke: RQ-VAE gradients plus train-only order and unique SID export."""
import numpy as np
import pandas as pd
import torch
from train_rqvae import RQVAE, _tokenizer_config, _extend_collisions, predictive_level_order

torch.manual_seed(42)
torch.set_num_threads(1)
model = RQVAE(_tokenizer_config(), in_dim=32)
model.init_codebook(torch.randn(256, 32))
batch = torch.randn(8, 32)
reconstructed, quant_loss, _, _ = model(batch)
loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
assert torch.isfinite(loss)
loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())

tokens = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=np.int64)
frame = pd.DataFrame({"history": [[], [0], [0], [1], [2], [2]], "target": [0, 1, 2, 3, 1, 3]})
ordered, permutation, scores = predictive_level_order(tokens, frame)
assert sorted(permutation) == [0, 1, 2] and len(scores) == 3
sid = _extend_collisions(ordered, [256, 256, 256])
assert sid.shape == (4, 4) and len(np.unique(sid, axis=0)) == 4
print(f"PASS loss={loss.item():.6f} grad=finite permutation={permutation} sid_shape={sid.shape} unique=4")
