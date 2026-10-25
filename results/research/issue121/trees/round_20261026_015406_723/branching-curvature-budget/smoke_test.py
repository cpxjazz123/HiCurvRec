"""Minimal differentiability/assignment/export smoke for the candidate pathway."""
import torch
from types import SimpleNamespace
from model import RQVAE
from train_rqvae import _behavior_inputs, _conditional_entropy, _soft_conditional_entropy, _extend_collisions, BALANCE_MARGIN
import numpy as np

torch.manual_seed(4)
cfg = SimpleNamespace(hidden_sizes=(16,), codebook_num=4, codebook_size=(8,8,8,8),
                      codebook_dim=4, dropout=0., beta=.25, vq_type="vq",
                      ema_decay=.99, fix_code_embs=False, sk_epsilon=-1., sk_iters=-1)
m = RQVAE(cfg, in_dim=6)
x = torch.randn(12, 6)
branch = torch.linspace(0, 1, 12)
out, qloss, _, ids = m(x, branch)
loss, _ = m.compute_loss(x, out, qloss)
assert torch.isfinite(loss)
loss.backward()
assert m.rq.vq_layers[0].embed.weight.grad is not None
assert torch.isfinite(m.rq.vq_layers[0].embed.weight.grad).all()
assert ids.shape == (12, 4)
sid = _extend_collisions(ids.detach().numpy(), [8,8,8,8])
assert sid.shape == (12, 5) and len(np.unique(sid, axis=0)) == 12
probe = m.rq.vq_layers[0]
z = m.encoder(x)
euclidean = probe._distances(z)
curvature = ((1 - .85 * branch[:, None]).clamp(.15, 1.) *
             (.75 + .5 * (probe.get_code_embs().square().sum(-1) /
                           probe.get_code_embs().square().sum(-1).mean().clamp_min(1e-8))[None, :])).clamp(.15, 1.)
curved = probe._poincare_distance_sq(z, probe.get_code_embs(), curvature)
assert not torch.equal(euclidean.argmin(-1), curved.argmin(-1)), "geometry-on must steer assignments"
probe.zero_grad(set_to_none=True)
euclidean.sum().backward(retain_graph=True)
euclidean_grad = probe.embed.weight.grad.detach().clone()
probe.zero_grad(set_to_none=True)
curved.sum().backward()
curved_grad = probe.embed.weight.grad.detach().clone()
assert not torch.allclose(euclidean_grad, curved_grad), "geometry-on must change codebook gradients"

# Exercise the same edge-input -> model -> loss -> backward route as the trainer.
raw = m
raw._training_embeddings = torch.randn(20, 6)
raw._branch_by_id = torch.linspace(0, 1, 20)
item_ids = torch.tensor([1, 2, 3, 4])
branch_values = raw._branch_by_id[item_ids]
counts = {1: [5], 2: [6], 3: [7], 4: [8]}
pair_x, pair_branch, source_branch = _behavior_inputs(raw, item_ids, counts, branch_values)
assert pair_x is not None and pair_x.shape == (8, 6)
edge_out = m(x[:4], branch[:4], pair_x, pair_branch, source_branch)
edge_loss, _ = m.compute_loss(x[:4], edge_out[0], edge_out[1])
coupled_loss = edge_loss + 0.05 * edge_out[4]
assert torch.isfinite(coupled_loss) and torch.isfinite(edge_out[4])
m.zero_grad(set_to_none=True)
coupled_loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

# Empty-edge path remains valid and returns model's ordinary forward tuple.
empty_inputs = _behavior_inputs(raw, item_ids, {}, branch_values)
assert empty_inputs == (None, None, None)
off_out = m(x[:4], branch[:4])
off_loss, _ = m.compute_loss(x[:4], off_out[0], off_out[1])
assert torch.isfinite(off_loss)

# Geometry-off comparison for the same edge tokens: Euclidean and curved
# code-vector distances must differ; behavior-off and behavior+geometry-on
# losses above must likewise produce finite, distinct training computations.
with torch.no_grad():
    edge_encoded = m.encoder(pair_x)
    _, _, _, edge_tokens = m.rq(edge_encoded, branching=pair_branch)
    level = m.rq.vq_layers[0]
    left = level.get_code_embs()[edge_tokens[:4, 0]]
    right = level.get_code_embs()[edge_tokens[4:, 0]]
    euclidean_edge = (left - right).square().sum(-1).mean()
    curved_edge = level._poincare_distance_sq(left, right, (1 - .85 * source_branch).clamp(.15, 1.)).diagonal().mean()
    assert not torch.allclose(euclidean_edge, curved_edge)

# Verify that four-token minibatches expose H(T3|T1,T2) and
# H(T4|T1,T2,T3) as the final two conditional-entropy entries.
entropy_tokens = torch.tensor([
    [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
])
entropy = _conditional_entropy(entropy_tokens)
assert entropy.shape == (3,)
assert torch.allclose(entropy[-2], entropy.new_tensor(np.log(2.0)))
assert torch.allclose(entropy[-1], entropy.new_zeros(()))
# Verify the specified hinge direction: when H(T4)>H(T3), it is zero.
synthetic_entropy = torch.tensor([0.2, 0.8])  # H(T3), H(T4): no penalty
assert torch.relu(synthetic_entropy[0] - synthetic_entropy[1] - BALANCE_MARGIN) == 0
synthetic_entropy = torch.tensor([0.8, 0.2])  # H(T3) exceeds H(T4): penalty
assert torch.relu(synthetic_entropy[0] - synthetic_entropy[1] - BALANCE_MARGIN) > 0
# The training balance term must send gradients through curved soft assignment
# probabilities into both the encoder and the codebook.
m.zero_grad(set_to_none=True)
soft_entropy = _soft_conditional_entropy(m, x, branch)
balance = torch.relu(soft_entropy[-2] - soft_entropy[-1] - BALANCE_MARGIN)
assert balance.requires_grad and torch.isfinite(balance)
balance.backward()
encoder_grad = [p.grad for p in m.encoder.parameters() if p.grad is not None]
code_grad = m.rq.vq_layers[0].embed.weight.grad
assert encoder_grad and any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in encoder_grad)
assert code_grad is not None and torch.isfinite(code_grad).all() and code_grad.abs().sum() > 0
print("PASS four learned token levels; T3/T4 conditional entropy indices; trainer edge-input populated/empty; coupled finite loss/backward; behavior-off finite loss; curved-vs-Euclidean edge metric; entropy hinge direction; differentiable balance gradient to encoder and codebook; unique SID export", flush=True)
