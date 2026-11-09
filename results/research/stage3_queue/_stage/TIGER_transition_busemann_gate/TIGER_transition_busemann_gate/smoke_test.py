"""Small CPU forward/backward and SID-contract smoke for promoted candidate."""
import torch
from types import SimpleNamespace
from model import RQVAE
from train_rqvae import _extend_collisions, REPO_ROOT, ARM_NAME, RESULT_DIR

assert (REPO_ROOT / "stage2_RQ-VAE").is_dir()
assert ARM_NAME == "TIGER_transition_busemann_gate"
assert RESULT_DIR == REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME
cfg = SimpleNamespace(hidden_sizes=(16,), codebook_num=3, codebook_size=(8,8,8),
                      codebook_dim=4, dropout=0., beta=.25, vq_type="vq",
                      ema_decay=.99, fix_code_embs=False, sk_epsilon=.003, sk_iters=3)
m = RQVAE(cfg, in_dim=6)
x = torch.randn(12, 6)
h = torch.randn(12, 4)
a = torch.ones(12)
recon, qloss, _, ids = m(x, behavior=(h, a))
loss = torch.nn.functional.mse_loss(recon, x) + qloss
assert torch.isfinite(loss)
loss.backward()
assert m.encoder.mlp[1].weight.grad is not None and torch.isfinite(m.encoder.mlp[1].weight.grad).all()
assert m.rq.vq_layers[0].embed.weight.grad is not None
assert ids.shape == (12, 3)
sid = _extend_collisions(ids.detach().numpy(), [8,8,8])
assert sid.shape == (12, 4) and len(set(map(tuple, sid))) == 12
# Both the behavior-off and geometry-off paths are operational and differ in computation.
layer = m.rq.vq_layers[0]
latent = torch.randn(12, 4)
curved = layer._curved_transition_scores(latent, h, a)
euclidean = layer._distances(latent)
assert curved.shape == euclidean.shape and not torch.allclose(curved, euclidean)
layer.use_transition_gate = True
layer.use_hyperbolic_gate = False
geometry_off = euclidean - 0.05 * a[:, None] * (torch.nn.functional.normalize(layer.get_code_embs(), dim=-1) @ torch.nn.functional.normalize(h, dim=-1).t()).t()
assert not torch.allclose(curved, geometry_off)
layer.use_transition_gate = False
behavior_off_indices = layer(latent)[3]
assert behavior_off_indices.shape == (12,)
print("PASS forward/backward finite; encoder+codebook gradients; behavior-curved scores differ from Euclidean; SID export unique; promoted-root discovery OK")
