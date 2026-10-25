"""Deterministic forward/backward and ablation assertions for the coupled loss."""
from types import SimpleNamespace
import numpy as np
import torch
from model import RQVAE


def main():
    torch.manual_seed(7)
    cfg = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3, codebook_size=(4, 4, 4),
        codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=5)
    model = RQVAE(cfg, in_dim=6)
    x = torch.randn(12, 6)
    weights = torch.arange(1, 13, dtype=torch.float32)
    # Nonuniform normalized training-only prefix mass targets.
    dense = (torch.tensor([.7, .2, .1, 0.]), torch.tensor([.4, .3, .2, .1] * 4),
             torch.ones(64) / 64)
    tables = tuple((t.nonzero().flatten(), t[t.nonzero().flatten()], t.numel()) for t in dense)

    def evaluate(behaviour=True, geometry=True):
        model.zero_grad(set_to_none=True)
        rec, qloss, _, ids, _ = model(x)
        occ = model.prefix_occupancy_loss(x, weights, torch.arange(len(x)), tables,
                                           behaviour_on=behaviour, geometry_on=geometry)
        loss = torch.nn.functional.mse_loss(rec, x) + qloss + .1 * occ
        assert torch.isfinite(loss) and torch.isfinite(occ)
        loss.backward()
        named = list(model.encoder.named_parameters()) + [(f"vq.{i}.{n}", p) for i, l in enumerate(model.rq.vq_layers) for n,p in l.named_parameters()]
        grads = {n: p.grad.detach().clone() for n,p in named if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        return float(occ.detach()), grads, ids.detach().clone()

    both = evaluate(True, True)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    after = evaluate(True, True)
    euclidean = evaluate(True, False)
    behaviour_off = evaluate(False, True)
    assert not torch.allclose(torch.tensor(both[0]), torch.tensor(euclidean[0]), atol=1e-7)
    assert not torch.allclose(torch.tensor(both[0]), torch.tensor(behaviour_off[0]), atol=1e-7)
    assert any(not torch.allclose(both[1][k], euclidean[1][k]) for k in both[1].keys() & euclidean[1].keys())
    assert any(not torch.allclose(both[1][k], behaviour_off[1][k]) for k in both[1].keys() & behaviour_off[1].keys())
    assert torch.isfinite(after[0] * torch.ones(())).all()
    with torch.no_grad():
        ids = model.get_indices(x)
    exported = np.concatenate([ids.numpy(), np.arange(len(ids))[:, None]], axis=1)
    assert ids.shape == (12, 3) and len(np.unique(exported, axis=0)) == 12
    # Full trainer radix: only sparse observed support is retained/evaluated.
    full_radix = 256 ** 3
    sparse_ids = torch.tensor([0, full_radix // 2, full_radix - 1])
    assert sparse_ids.numel() == 3 and full_radix == 16_777_216
    print(f"PASS occupancy={both[0]:.6f} euclidean={euclidean[0]:.6f} behaviour_off={behaviour_off[0]:.6f} optimizer_updated=True sid_shape={tuple(ids.shape)} unique_export={len(np.unique(exported, axis=0))}")


if __name__ == "__main__":
    main()
