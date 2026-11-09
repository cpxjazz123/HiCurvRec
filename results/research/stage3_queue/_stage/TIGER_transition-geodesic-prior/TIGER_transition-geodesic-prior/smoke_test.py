"""Minimal forward/backward, coupled assignment, and SID export smoke test."""
from types import SimpleNamespace
import numpy as np
import torch
from model import RQVAE

torch.manual_seed(7)
cfg = SimpleNamespace(hidden_sizes=(16,), codebook_num=3, codebook_size=(8, 8, 8),
                      codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq",
                      ema_decay=0.99, fix_code_embs=False, sk_epsilon=0.003, sk_iters=5)
m = RQVAE(cfg, in_dim=6)
x = torch.randn(12, 6)
prior = torch.zeros(12, 8); prior[:, 1] = 1
y, qloss, _, codes = m(x, transition_prior=prior, transition_matrix=torch.eye(8).mul(0.9).add(0.1 / 8))
loss = torch.nn.functional.mse_loss(y, x) + qloss
assert torch.isfinite(loss)
loss.backward()
assert m.rq.vq_layers[0].embed.weight.grad is not None
assert torch.isfinite(m.rq.vq_layers[0].embed.weight.grad).all()

# Deterministic boundary test: the query is fractionally closer to code 0,
# while the predecessor's Poincare region prior favors code 1. The hard
# argmin itself is nondifferentiable; verify its selected-code VQ gradient.
layer = m.rq.vq_layers[0]
with torch.no_grad():
    layer.embed.weight.zero_()
    layer.embed.weight[1, 0] = 0.1
    layer.embed.weight[2, 0] = 1.0
query = torch.tensor([[0.047, 0., 0., 0.]])
dist = layer._distances(query)
region_prior = torch.zeros(1, 8); region_prior[0, 1] = 1
euclidean = layer._indices(dist, False, None)
no_prior = layer._indices(dist, False, torch.zeros_like(region_prior))
table_a = torch.full((8, 8), 1.0 / 8); table_a[1] = 0.0; table_a[1, 2] = 1.0
table_b = table_a.clone(); table_b[1] = 0.0; table_b[1, 0] = 1.0
coupled = layer._indices(dist, False, region_prior, table_a)
original_kernel = layer._poincare_kernel
try:
    layer._poincare_kernel = lambda: -torch.cdist(layer.get_code_embs(), layer.get_code_embs()).pow(2) / 2
    geometry_off = layer._indices(dist, False, region_prior, table_a)
finally:
    layer._poincare_kernel = original_kernel
assert euclidean.item() == no_prior.item() == geometry_off.item() == 0
assert coupled.item() == 1, (euclidean.item(), coupled.item())
assert layer._indices(dist, False, region_prior, table_b).item() == 0
# A zero-mass predecessor row must not affect expected compatibility or rank.
base_kernel = torch.tensor([[0.0] * 8, [0.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]] + [[0.0] * 8 for _ in range(6)])
changed_kernel = base_kernel.clone()
changed_kernel[0] = torch.tensor([-100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0])
zero_mass = torch.zeros(1, 8); zero_mass[0, 1] = 1
try:
    layer._poincare_kernel = lambda: base_kernel
    assignment_a = layer._indices(dist, False, zero_mass)
    layer._poincare_kernel = lambda: changed_kernel
    assignment_b = layer._indices(dist, False, zero_mass)
finally:
    layer._poincare_kernel = original_kernel
assert torch.equal(assignment_a, assignment_b)
layer.zero_grad(set_to_none=True)
_, selected_loss, _, _ = layer(query, transition_prior=region_prior)
selected_loss.backward()
assert torch.isfinite(layer.embed.weight.grad).all()
assert layer.embed.weight.grad[1].abs().sum() > 0
# The index output is integer and has no gradient. Verify the selected code's
# differentiable VQ path through the trainer's real RQVAE forward/backward.
torch.manual_seed(7)
m = RQVAE(cfg, in_dim=6)
with torch.no_grad():
    sid = m.get_indices(x).cpu().numpy()
assert sid.shape == (12, 3) and np.isfinite(sid).all()
print(f"forward_backward=ok loss={loss.item():.6f} grad=finite sid_shape={sid.shape} coupled_assignment={euclidean.item()}->{coupled.item()} controls={no_prior.item()},{geometry_off.item()}")
