"""Deterministic forward/backward and assignment-switch regression check."""
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from model import RQVAE
from model.layers import VQLayer, _exp0
from train_rqvae import _extend_collisions


def finite_gradients(model):
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    return sum(float(g.norm()) for g in grads)


def main():
    torch.manual_seed(11)
    layer = VQLayer(2, 2)
    with torch.no_grad():
        layer.embed.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    z = torch.tensor([[0.05, 0.0]])
    layer.set_flow(_exp0(torch.tensor([[-1.5, 0.0]])), torch.ones(1))
    layer.flow_eta = 0.9
    layer.flow_enabled = False
    baseline = layer.assignment_distances(z, torch.tensor([0])).argmin(-1)
    layer.flow_enabled = True
    flowed = layer.assignment_distances(z, torch.tensor([0])).argmin(-1)
    assert not torch.equal(baseline, flowed), (baseline, flowed)
    layer.flow_enabled = False
    restored = layer.assignment_distances(z, torch.tensor([0])).argmin(-1)
    assert torch.equal(baseline, restored)

    hyper = layer.assignment_distances(torch.tensor([[0.2, 0.3]]))
    layer.hyperbolic_enabled = False
    euclidean = layer.assignment_distances(torch.tensor([[0.2, 0.3]]))
    assert not torch.allclose(hyper, euclidean)
    layer.hyperbolic_enabled = True
    assert torch.allclose(hyper, layer.assignment_distances(torch.tensor([[0.2, 0.3]])))

    cfg = SimpleNamespace(hidden_sizes=(16,), codebook_num=3,
        codebook_size=(8, 8, 8), codebook_dim=4, dropout=0.0,
        beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
        sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(cfg, in_dim=6)
    x = torch.randn(5, 6)
    ids = torch.arange(5)
    bary = _exp0(torch.randn(5, 4) * 0.7)
    for vq in model.rq.vq_layers:
        vq.set_flow(bary, torch.ones(5))
    reconstruction, qloss, _, tokens = model(x, ids)
    loss = F.mse_loss(reconstruction, x) + qloss
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = finite_gradients(model)
    sid = _extend_collisions(tokens.detach().numpy(), list(cfg.codebook_size))
    assert sid.shape == (5, 4) and len(np.unique(sid, axis=0)) == 5
    print(f"PASS assignment_off={baseline.tolist()} assignment_on={flowed.tolist()} restored={restored.tolist()} finite_loss={loss.item():.6f} finite_grad_norm={grad_norm:.6f} sid_shape={sid.shape} unique={len(np.unique(sid, axis=0))}")


if __name__ == "__main__":
    main()
