"""Small forward/backward and SID-export contract check; no shared outputs."""
import numpy as np
import torch
from types import SimpleNamespace
from model import RQVAE
from train_rqvae import incoming_assignment_loss, _extend_collisions, _stable_squared_poincare_from_arg

# Exact coincidence is a reachable code/barycentre state; its squared
# Poincare distance and backward gradients must remain finite.
coincident_a = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
coincident_b = torch.zeros((1, 3), dtype=torch.float32)
delta_sq = (coincident_a - coincident_b).square().sum(-1)
arg = (2.0 * delta_sq / ((1.0 - coincident_a.square().sum(-1)) *
                         (1.0 - coincident_b.square().sum(-1)))).clamp_min(0)
coincident_squared = _stable_squared_poincare_from_arg(arg)
coincident_squared.sum().backward()
assert coincident_squared.item() == 0.0
assert torch.isfinite(coincident_a.grad).all()

torch.manual_seed(9)
cfg = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3,
    codebook_size=(8, 8, 8), codebook_dim=4, dropout=0.0, beta=0.25,
    vq_type="vq", ema_decay=0.99, fix_code_embs=False,
    sk_epsilon=0.003, sk_iters=5)
model = RQVAE(cfg, in_dim=6)
emb = torch.randn(12, 6)
model.rq.init_codebook(model.encoder(emb).detach(), torch.device("cpu"))
ids = torch.tensor([4, 5, 6, 7])
incoming = {4: [(0, 3), (1, 1)], 5: [(1, 2)], 6: [(2, 1)], 7: [(3, 1)]}
negative = {4: 8, 5: 9, 6: 10, 7: 11}
bary = torch.randn(12, 4) * 0.1
neg_bary = torch.randn(12, 4) * 0.1
concentration = torch.ones(12)
behaviour_off = incoming_assignment_loss(model, ids, bary, neg_bary, emb, behaviour_on=False)
euclidean = incoming_assignment_loss(model, ids, bary, neg_bary, emb, geometry="euclidean", concentration=concentration)
hyperbolic = incoming_assignment_loss(model, ids, bary, neg_bary, emb, concentration=concentration)
assert not torch.equal(behaviour_off.argmax(-1), hyperbolic.argmax(-1))
assert not torch.allclose(euclidean, hyperbolic)
assignments = {}
for name, logits in (("both_on", hyperbolic), ("behaviour_off", behaviour_off), ("euclidean", euclidean)):
    model.zero_grad(set_to_none=True)
    recon, qloss, _, assigned = model(emb[ids], first_assignment_logits=logits)
    assignments[name] = assigned[:, 0].detach().clone()
    assert torch.equal(assigned[:, 0], logits.argmax(-1)), name
    loss = torch.nn.functional.mse_loss(recon, emb[ids]) + qloss
    assert torch.isfinite(loss), name
    loss.backward()
    grad = model.rq.vq_layers[0].embed.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
assert not torch.equal(assignments["both_on"], assignments["behaviour_off"])
assert not torch.allclose(hyperbolic, euclidean)
tokens = model.get_indices(emb)
sid = _extend_collisions(tokens.detach().numpy(), [8, 8, 8])
assert sid.shape == (12, 4) and len(np.unique(sid, axis=0)) == 12
print("PASS finite forward/backward; coincident Poincare gradient; first-codebook gradient; unique SID export", flush=True)
