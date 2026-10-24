"""Minimal finite-gradient, weighted-loss, and unique-SID smoke test."""
from types import SimpleNamespace
import numpy as np
import torch
from model import RQVAE
from train_rqvae import _extend_collisions

cfg = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3,
                      codebook_size=(256, 256, 256), codebook_dim=32,
                      dropout=0., beta=.25, vq_type="vq", ema_decay=.99,
                      fix_code_embs=False, sk_epsilon=.003, sk_iters=5)
torch.manual_seed(7)
model = RQVAE(cfg, in_dim=32)
weights = torch.linspace(.5, 1.5, 32)
model.axis_weights.copy_(weights)
x = torch.randn(12, 32)
recon, qloss, _, _ = model(x)
loss, recon_loss = model.compute_loss(x, recon, qloss)
assert torch.isfinite(loss) and torch.isfinite(recon_loss)
loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
           for p in model.parameters())
tokens = np.array([[0, 1, 2], [0, 1, 2], [2, 1, 0]], dtype=np.int64)
sid = _extend_collisions(tokens, [256, 256, 256])
assert sid.shape == (3, 4) and len(np.unique(sid, axis=0)) == 3
print(f"PASS loss={loss.item():.6f} recon={recon_loss.item():.6f} gradient=finite sid_shape={sid.shape} unique=3")
