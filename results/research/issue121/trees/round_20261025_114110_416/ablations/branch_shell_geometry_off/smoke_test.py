"""Small CPU proof of finite coupled forward/backward and SID export."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model import RQVAE
from train_rqvae import _extend_collisions

cfg = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3, codebook_size=(8, 8, 8),
                      codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq",
                      ema_decay=0.99, fix_code_embs=False, sk_epsilon=-1.0, sk_iters=-1)
model = RQVAE(cfg, in_dim=6)
x = torch.randn(12, 6)
r = torch.linspace(0.1, 1.0, 12)
y, qloss, _, _ = model(x, branch_radii=r)
loss, _ = model.compute_loss(x, y, qloss)
assert torch.isfinite(loss).all(), "loss is not finite"
loss.backward()
encoder_param = model.encoder.mlp[1].weight
codebook_param = model.rq.vq_layers[0].embed.weight
for name, parameter in (("encoder", encoder_param), ("codebook", codebook_param)):
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), f"{name} gradient missing/nonfinite"
    assert parameter.grad.abs().sum() > 0, f"{name} gradient is zero"
encoder_before, codebook_before = encoder_param.detach().clone(), codebook_param.detach().clone()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer.step()
assert not torch.equal(encoder_before, encoder_param) and not torch.equal(codebook_before, codebook_param), "optimizer did not update encoder and codebook"
with torch.no_grad():
    encoded = model.encoder(x)
    curved_branch = model.rq(encoded, branch_radii=r)[3]
    curved_behavior_off = model.rq(encoded, branch_radii=r, behavior_on=False)[3]
    euclidean_branch = model.rq(encoded, branch_radii=r, geometry_on=False)[3]
    euclidean_behavior_off = model.rq(encoded, branch_radii=r, behavior_on=False, geometry_on=False)[3]
assert not torch.equal(curved_branch, euclidean_branch), "Euclidean geometry ablation left assignments unchanged"
assert not torch.equal(curved_branch, curved_behavior_off), "curved behavior-off ablation left assignments unchanged"
assert not torch.equal(euclidean_branch, euclidean_behavior_off), "Euclidean behavior-off ablation left assignments unchanged"
assert not torch.equal(curved_behavior_off, euclidean_behavior_off), "geometry comparison without behavior is not controlled"

# Near-boundary float32 path: distance, selected loss, and gradients stay finite.
from model.utils import poincare_distance, proj_to_ball
boundary = torch.tensor([[0.9999999, 0.0]], dtype=torch.float32, requires_grad=True)
other = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
near_distance = poincare_distance(proj_to_ball(boundary, 1.0), other, 1.0)
near_loss = near_distance.square().mean()
near_loss.backward()
assert torch.isfinite(near_distance).all() and torch.isfinite(near_loss)
assert boundary.grad is not None and torch.isfinite(boundary.grad).all()
assert other.grad is not None and torch.isfinite(other.grad).all()
with torch.no_grad():
    tokens = model.get_indices(x, branch_radii=r).numpy()
sid = _extend_collisions(tokens, list(cfg.codebook_size))
assert sid.shape == (12, 4) and len(np.unique(sid, axis=0)) == 12
with tempfile.TemporaryDirectory() as d:
    Path(d, "sid.json").write_text(json.dumps(sid.tolist()))
print(f"PASS loss={loss.item():.6f} encoder_grad=finite codebook_grad=finite near_boundary_distance={near_distance.item():.6f} near_boundary_loss_grad=finite ablations=curved_on_off+euclidean_on_off+geometry_control sid_shape={sid.shape} unique={len(np.unique(sid, axis=0))}")
