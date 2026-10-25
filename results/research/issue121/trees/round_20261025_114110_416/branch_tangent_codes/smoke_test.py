"""Deterministic CPU checks of curved assignments, ablations and training step."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from model import RQVAE
from model.layers import VQLayer
from train_rqvae import _extend_collisions, _single_gpu_requested, RESULT_DIR, REPO_ROOT


def main():
    # Zero latent rows have the exp_0 limit of the zero vector and must remain
    # finite through curved distance, quantization, and backward.
    zero_layer = VQLayer(4, 2)
    zero_latent = torch.zeros(1, 2, requires_grad=True)
    zero_distances = zero_layer._distances(zero_latent, geometry=True, curvature=0.5)
    assert torch.isfinite(zero_distances).all()
    zero_quantized, zero_loss, _, _ = zero_layer(zero_latent, geometry=True, curvature=0.5)
    assert torch.isfinite(zero_quantized).all() and torch.isfinite(zero_loss)
    zero_loss.backward()
    assert zero_latent.grad is not None and torch.isfinite(zero_latent.grad).all()

    # Check scalar and non-collinear logarithms against Mobius subtraction
    # (-z) (+_c) b, whose two coefficients multiply -z and b respectively.
    c = 0.5
    for z, b in (
        (torch.tensor([[0.2]], dtype=torch.float64), torch.tensor([[0.4]], dtype=torch.float64)),
        (torch.tensor([[0.2, 0.0]], dtype=torch.float64), torch.tensor([[0.0, 0.2]], dtype=torch.float64)),
    ):
        actual = RQVAE._log_at(z, b, c)
        dot = (z * b).sum(-1, keepdim=True)
        z2, b2 = z.square().sum(-1, keepdim=True), b.square().sum(-1, keepdim=True)
        delta = (-(1 - 2*c*dot + c*b2)*z + (1-c*z2)*b) / (1 - 2*c*dot + c*c*z2*b2)
        norm = delta.norm(dim=-1, keepdim=True)
        expected = (2 / ((c**0.5) * (2 / (1-c*z2)))) * torch.atanh((c**0.5)*norm) * delta / norm
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)

    # exp_0/log_0 must round-trip tangent code vectors, including nonzero codes.
    codes = torch.tensor([[0.2, 0.0], [0.0, -0.7]], dtype=torch.float64, requires_grad=True)
    projected = RQVAE._exp0(codes, c)
    recovered = RQVAE._log0(projected, c)
    assert torch.allclose(recovered, codes, atol=1e-9, rtol=1e-9)
    recovered.square().sum().backward()
    assert torch.isfinite(codes.grad).all()

    # exp_z(alpha*v) has geodesic distance alpha*||v||_z from z.
    alpha = 0.35
    v = RQVAE._log_at(z, b, c)
    endpoint = RQVAE._exp_at(z, alpha * v, c)
    relative = RQVAE._mobius_add(-z, endpoint, c)
    distance = (2 / (c ** 0.5)) * torch.atanh((c ** 0.5) * relative.norm(dim=-1, keepdim=True))
    lambda_z = 2 / (1 - c * z.square().sum(-1, keepdim=True))
    expected_distance = alpha * lambda_z * v.norm(dim=-1, keepdim=True)
    assert torch.allclose(distance, expected_distance, atol=1e-9, rtol=1e-9)
    z_grad = z.clone().requires_grad_()
    b_grad = b.clone().requires_grad_()
    curved = RQVAE._log0(RQVAE._exp_at(z_grad, alpha * RQVAE._log_at(z_grad, b_grad, c), c), c)
    curved.square().sum().backward()
    assert torch.isfinite(z_grad.grad).all() and torch.isfinite(b_grad.grad).all()

    cfg = SimpleNamespace(hidden_sizes=(2,), codebook_num=2, codebook_size=(4, 4),
                          codebook_dim=2, dropout=0.0, beta=0.25, vq_type="vq",
                          sk_epsilon=-1.0, sk_iters=-1, ema_decay=0.99, fix_code_embs=False)
    torch.manual_seed(17)
    model = RQVAE(cfg, in_dim=2)
    with torch.no_grad():
        model.encoder.mlp[1].weight.copy_(torch.eye(2))
        model.encoder.mlp[1].bias.zero_()
        model.context_w.weight.copy_(torch.eye(2))
        model.context_u.weight.copy_(torch.eye(2))
        model.branch_alpha = 1.0
        model.rq.vq_layers[0].embed.weight.copy_(torch.tensor([[0.,0.],[.2,0.],[0.,.2],[.2,.2]]))
        model.rq.vq_layers[1].embed.weight.copy_(torch.tensor([[0.,0.],[.2,0.],[0.,.2],[.2,.2]]))
    x = torch.tensor([[.03,.17],[.17,.03],[.12,.18],[.18,.12]])
    contexts = torch.tensor([[.18,.02],[.02,.18],[.18,.02],[.02,.18]])

    def ids(behavior, geometry):
        model.behavior_enabled, model.geometry_enabled = behavior, geometry
        return model.get_indices(x, contexts)

    both_ids = ids(True, True)
    behavior_off_ids = ids(False, True)
    geometry_off_ids = ids(True, False)
    assert not torch.equal(both_ids, behavior_off_ids), "behavior ablation did not alter assignments"
    assert not torch.equal(both_ids, geometry_off_ids), "geometry ablation did not alter assignments"
    assert torch.equal(both_ids, ids(True, True)), "both-on assignments were not restored"

    # Compare the behaviour-projection gradient in each assignment mode too;
    # this catches a connected-looking branch whose contribution is dead.
    mode_grads = {}
    for label, behavior, geometry in (("both", True, True),
                                      ("behavior_off", False, True),
                                      ("geometry_off", True, False)):
        model.behavior_enabled, model.geometry_enabled = behavior, geometry
        model.zero_grad(set_to_none=True)
        reconstruction, qloss, _, _ = model(x, contexts)
        mode_loss, _ = model.compute_loss(x, reconstruction, qloss)
        mode_loss.backward()
        grad = model.context_u.weight.grad
        if grad is None:
            assert label == "behavior_off", f"missing {label} gradient"
            grad = torch.zeros_like(model.context_u.weight)
        assert torch.isfinite(grad).all(), f"invalid {label} gradient"
        mode_grads[label] = grad.detach().clone()
    assert mode_grads["both"].abs().sum() > 0
    assert not torch.equal(mode_grads["both"], mode_grads["behavior_off"]), "behavior ablation did not alter gradient"
    assert not torch.equal(mode_grads["both"], mode_grads["geometry_off"]), "geometry ablation did not alter gradient"

    model.behavior_enabled = model.geometry_enabled = True
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = model.context_u.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    reconstruction, qloss, _, train_ids = model(x, contexts)
    loss, _ = model.compute_loss(x, reconstruction, qloss)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("context_u.weight", "context_w.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, f"invalid {name} gradient"
    optimizer.step()
    assert torch.isfinite(model.context_u.weight).all()
    assert not torch.equal(before, model.context_u.weight), "optimizer did not update context projection"

    sid = _extend_collisions(train_ids.detach().numpy(), [4, 4])
    assert sid.shape == (4, 3) and len(np.unique(sid, axis=0)) == 4
    assert RESULT_DIR.parts[-2:] == ("stage2_RQ-VAE", "TIGER_branch_tangent_codes")
    assert RESULT_DIR == REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_branch_tangent_codes"
    assert _single_gpu_requested("1") and not _single_gpu_requested("0")
    print(f"PASS logmap=matched assignments=both_on:{both_ids.tolist()} behavior_off:{behavior_off_ids.tolist()} geometry_off:{geometry_off_ids.tolist()} gradient_modes=distinct_nonzero loss={loss.item():.6f} gradients=finite_nonzero optimizer=updated sid_shape={sid.shape} unique={len(np.unique(sid, axis=0))}")


if __name__ == "__main__":
    main()
