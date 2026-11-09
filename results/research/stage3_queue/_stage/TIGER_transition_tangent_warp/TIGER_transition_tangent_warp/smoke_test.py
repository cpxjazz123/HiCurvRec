"""Small forward/backward and SID-export contract test; performs no file writes."""
from types import SimpleNamespace
import numpy as np
import torch
from model import RQVAE

torch.manual_seed(7)
cfg = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3,
    codebook_size=(256, 256, 256), codebook_dim=32, dropout=0.0,
    beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
    sk_epsilon=0.003, sk_iters=5)
model = RQVAE(cfg, in_dim=20)
x, profile = torch.randn(12, 20), torch.randn(12, 64)
recon, qloss, _, _ = model(x, profile)
loss = torch.nn.functional.mse_loss(recon, x) + qloss
assert torch.isfinite(loss)
loss.backward()
grad = model.behavior_map.weight.grad
assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
with torch.no_grad():
    warped = model._warp(model.encoder(x), profile)
    no_behavior = model._warp(model.encoder(x), profile, behavior_on=False)
    euclidean = model._warp(model.encoder(x), profile, geometry_on=False)
    assert not torch.allclose(warped, no_behavior)
    assert not torch.allclose(warped, euclidean)
    ids = model.get_indices(x, profile=profile).cpu().numpy()
    assert ids.shape == (len(x), 3) and np.isfinite(ids).all()
    extended = np.column_stack((ids, np.arange(len(ids))))
    assert extended.shape == (len(x), 4)
print(f"PASS finite_loss={loss.item():.6f} behavior_grad={grad.norm().item():.6f} sid_shape={ids.shape} geometry_and_behavior_controls=distinct")
