from types import SimpleNamespace
import numpy as np
import torch

from model import RQVAE
from position_role import position_roles
from train_rqvae import _extend_collisions

torch.manual_seed(7)
cfg = SimpleNamespace(hidden_sizes=(16,), codebook_num=3,
    codebook_size=(256, 256, 256), codebook_dim=32, dropout=0.0,
    beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
    sk_epsilon=0.003, sk_iters=5)
model = RQVAE(cfg, in_dim=8)
x = torch.randn(12, 8)
reconstruction, qloss, _, _ = model(x)
loss, _ = model.compute_loss(x, reconstruction, qloss)
assert torch.isfinite(loss)
loss.backward()
grad = next(model.encoder.parameters()).grad
assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

roles, stats = position_roles([[0, 1, 2], [2, 3, 4]], 6)
assert set(roles.tolist()) <= {0, 1, 2, 3} and roles[5] == np.searchsorted(stats["cuts"], stats["mu0"], side="right")
raw = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2]])
baseline = _extend_collisions(raw, [256, 256, 256])
sid = np.column_stack((baseline, 768 + len(raw) + roles))
assert sid.shape == (6, 5) and len(np.unique(sid, axis=0)) == 6
assert np.isfinite(loss.item())
print("PASS: finite forward/backward, encoder gradient, train-only role bins, valid unique SID export")
