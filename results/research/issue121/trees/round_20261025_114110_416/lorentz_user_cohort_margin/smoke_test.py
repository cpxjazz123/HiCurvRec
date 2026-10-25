"""CPU smoke test for pair-margin gradients and SID export contract."""
from types import SimpleNamespace
import torch
from model import RQVAE
from train_rqvae import pair_margin_loss, _extend_collisions, lorentz_distance

torch.manual_seed(7)
cfg = SimpleNamespace(hidden_sizes=(16,), codebook_num=3, codebook_size=(8, 8, 8),
                      codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq",
                      ema_decay=0.99, fix_code_embs=False, sk_epsilon=0.003, sk_iters=5)
model = RQVAE(cfg, in_dim=6)
x = torch.randn(5, 6)
recon, qloss, _, tokens, _ = model(x)
za = model.cohort_code_vectors(x[:2], 0)
zb = model.cohort_code_vectors(x[2:4], 0)
loss = torch.nn.functional.mse_loss(recon, x) + qloss + 0.05 * pair_margin_loss(za, zb, torch.tensor([1, 0]))
assert torch.isfinite(loss)
pair_only = pair_margin_loss(za, zb, torch.tensor([1, 0]))
pair_encoder_grad = torch.autograd.grad(pair_only, model.encoder.mlp[1].weight, retain_graph=True)[0]
assert torch.isfinite(pair_encoder_grad).all() and pair_encoder_grad.abs().sum() > 0
loss.backward()
assert model.encoder.mlp[1].weight.grad is not None
assert model.rq.vq_layers[0].embed.weight.grad is not None
sid = _extend_collisions(tokens.detach().numpy(), list(cfg.codebook_size))
assert sid.shape == (5, 4) and len(set(map(tuple, sid))) == 5
print(f"forward_backward=ok loss={loss.item():.6f} encoder_grad=ok codebook_grad=ok pair_encoder_grad=ok sid_shape={sid.shape} sid_unique={len(set(map(tuple, sid)))}")

# Deterministic pair-only checks: coincident Lorentz codes, behaviour off,
# and Euclidean geometry all use the same fixed minibatch.
a = torch.tensor([[0.2, -0.1], [0.2, -0.1]], requires_grad=True)
b = a.detach().clone().requires_grad_(True)
labels = torch.tensor([1, 1])
both = pair_margin_loss(a, b, labels, "lorentz", True)
g_both = torch.autograd.grad(both, (a, b), retain_graph=True)
assert torch.isfinite(both) and all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in g_both)
off = pair_margin_loss(a, b, labels, "lorentz", False)
euclidean = pair_margin_loss(a, b, labels, "euclidean", True)
assert not torch.equal(both.detach(), off.detach())
assert not torch.equal(both.detach(), euclidean.detach())
g_off = torch.autograd.grad(off, (a, b), retain_graph=True)
g_euclidean = torch.autograd.grad(euclidean, (a, b))
assert any(not torch.allclose(x, y) for x, y in zip(g_both, g_off))
assert any(not torch.allclose(x, y) for x, y in zip(g_both, g_euclidean))
print(f"pair_comparisons=ok both={both.item():.6f} behaviour_off={off.item():.6f} euclidean={euclidean.item():.6f} coincident_grad=finite_nonzero")

# Inputs beyond float32 cosh/sinh range are projected before exp0. Exercise
# the coupled pair loss and its gradients at the projection boundary.
large_a = torch.tensor([[100.0, 0.0]], requires_grad=True)
large_b = torch.tensor([[99.0, 0.0]], requires_grad=True)
large_loss = pair_margin_loss(large_a, large_b, torch.tensor([1]), "lorentz", True)
large_grads = torch.autograd.grad(large_loss, (large_a, large_b))
assert torch.isfinite(large_loss)
assert all(torch.isfinite(g).all() for g in large_grads)
assert all(g.abs().max() <= 1e3 for g in large_grads)
print(f"large_norm_boundary=ok loss={large_loss.item():.6f} gradients=finite_bounded max_grad={max(g.abs().max().item() for g in large_grads):.6f}")
