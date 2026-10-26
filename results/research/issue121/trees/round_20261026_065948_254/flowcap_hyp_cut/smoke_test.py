"""CPU smoke of the exact shared training flow/capacity objective and export."""
from types import SimpleNamespace
import numpy as np
import torch
from model import RQVAE
from train_rqvae import _extend_collisions, balanced_assignments, flow_capacity_loss, hardened_prefixes
import train_rqvae as trainer

torch.manual_seed(7)
trainer.BEHAVIOR_ON = True
cfg = SimpleNamespace(hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 96, 256),
    codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
    fix_code_embs=False, sk_epsilon=0.003, sk_iters=5)
model = RQVAE(cfg, in_dim=16)
x = torch.randn(8, 16)
encoded = model.encoder(x)
ids = torch.arange(8)
neighbors = {i: ((int((i + 1) % 8), 1.0),) for i in range(8)}
all_emb = torch.randn(8, 16)
loss_on = flow_capacity_loss(encoded, ids, all_emb, neighbors, model)
loss_on.backward()
assert torch.isfinite(loss_on) and model.encoder.mlp[1].weight.grad is not None
grad_on = model.encoder.mlp[1].weight.grad.detach().clone()
model.zero_grad(set_to_none=True)
trainer.BEHAVIOR_ON = False
loss_off = flow_capacity_loss(model.encoder(x), ids, all_emb, neighbors, model)
(loss_off + model.encoder(x).square().mean()).backward()
grad_off = model.encoder.mlp[1].weight.grad.detach().clone()
assert loss_off.item() == 0 and not torch.allclose(grad_on, grad_off)
trainer.BEHAVIOR_ON = True
trainer.LORENTZ_GEOMETRY_ON = False
loss_euclidean = flow_capacity_loss(model.encoder(x), ids, all_emb, neighbors, model)
trainer.LORENTZ_GEOMETRY_ON = True
assert torch.isfinite(loss_euclidean) and not torch.allclose(loss_on, loss_euclidean)
prefix = hardened_prefixes(model.encoder(x), model)
assert prefix.shape == (8, 2) and int(prefix[:, 0].max()) < 16 and int(prefix[:, 1].max()) < 96
sid = _extend_collisions(np.column_stack((prefix.numpy(), np.zeros(8, dtype=np.int64))), [16, 96, 256])
assert sid.shape == (8, 4) and len(np.unique(sid, axis=0)) == 8
print(f"PASS shared_flow={loss_on.item():.6f} behavior_off={loss_off.item():.6f} euclidean={loss_euclidean.item():.6f} grad_delta={(grad_on-grad_off).norm().item():.6f} prefix_shape={tuple(prefix.shape)} sid_shape={sid.shape} unique={len(np.unique(sid,axis=0))}")
