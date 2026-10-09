"""CPU/production-geometry smoke tests for Stage2 semantic hierarchy loss.

Run from repository root:
    python -m bench_semantic_sid.test_loss
Requires production Stage2 layers and torch/sklearn, no dataset or GPU.
"""
import torch
import torch.nn.functional as F

from bench_hier_behavior.models import RQVAE
from .hierarchy_losses import SemanticHierarchyLoss, _xi


def main():
    torch.manual_seed(42)
    coarse = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    fine = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    parent = torch.tensor([0, 0, 1, 1])
    x = torch.randn(8, 768) * 0.1
    for name in ("euclid", "poincare"):
        model = RQVAE(name, 768, 32, normalize_latent=True,
                      hidden_sizes=(128, 64))
        loss_fn = SemanticHierarchyLoss(2, parent, 32)
        # Production-style three-level hard RQ, not toy quantization.
        reconstructed, qloss, tokens, latent, prefixes = model(
            x, return_prefixes=True)
        c = loss_fn(model, latent, tokens, prefixes, coarse, fine)
        total = F.mse_loss(reconstructed, x) + qloss + 0.05 * c["cone"] + 0.01 * c["prefix"]
        assert torch.isfinite(total), name
        total.backward()
        assert model.encoder.state_dict()
        for param in model.encoder.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
        for i, codebook in enumerate(model.rq.codes):
            assert codebook.grad is not None and torch.isfinite(codebook.grad).all(), (name, i)
        for param in loss_fn.parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        with torch.no_grad():
            p = model.geometry.to_point(torch.tensor([[0.3, 0.]]))
            child = model.geometry.to_point(torch.tensor([[0.4, 0.]]))
            ancestor = model.geometry.to_point(torch.tensor([[0.2, 0.]]))
            assert _xi(name, p, child).item() < 0.005
            assert _xi(name, p, ancestor).item() > 3.13
        print(f"PASS {name}: finite loss, gradients to all 3 codebooks, correct cone direction")
    print("PASS: Stage2 only; no Stage3")


if __name__ == "__main__":
    main()
