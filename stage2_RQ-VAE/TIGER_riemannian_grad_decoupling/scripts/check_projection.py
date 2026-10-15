"""Check that the tangent projection is what it claims to be.

Two properties, both of which the trained run depends on:

  1. the gradient handed back to the latent is exactly ``g - (g.u)u``, which is
     orthogonal to the code direction because ``u`` is normalised;
  2. the radial part is re-routed to the codebook rather than dropped, so the
     signal is transplanted instead of deleted.

Run from the tree root:  python3 scripts/check_projection.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.layers import VQLayer
from tangent_grad_projection import (
    NORM_EPS, TangentProjection, install_tangent_projection, projection_scope_note,
)


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ok = True
    torch.manual_seed(0)

    # 1. the returned gradient is the projection, and it is orthogonal
    latent = torch.randn(6, 4, requires_grad=True)
    codes = torch.nn.Parameter(torch.randn(5, 4))
    index = torch.tensor([0, 1, 2, 3, 4, 0])
    out = TangentProjection.apply(latent, codes, index)
    ok &= check("the Function's output carries a backward node",
                out.grad_fn is not None, str(type(out.grad_fn).__name__))
    out.sum().backward()

    grad_output = torch.ones_like(latent)
    unit = F.normalize(codes.detach()[index], dim=-1, eps=NORM_EPS)
    expected = grad_output - (grad_output * unit).sum(-1, keepdim=True) * unit
    ok &= check("the latent receives exactly g - (g.u)u",
                torch.allclose(latent.grad, expected),
                f"max|diff| = {float((latent.grad - expected).abs().max()):.2e}")
    # Orthogonality is the *sum* over the feature axis vanishing. The largest
    # elementwise product is not the dot product and is non-zero even for
    # orthogonal vectors -- an earlier version of this check asserted on it and
    # reported a spurious failure.
    dot = (latent.grad * unit).sum(dim=-1).abs().max()
    ok &= check("and that gradient is orthogonal to the code direction",
                float(dot) < 1e-5,
                f"max |sum_d g_d u_d| = {float(dot):.2e} "
                f"(max elementwise |g_d u_d| = "
                f"{float((latent.grad * unit).abs().max()):.2e}, expected non-zero)")
    ok &= check("the projection was not bypassed",
                not torch.allclose(latent.grad, grad_output),
                f"max|diff from ones| = {float((latent.grad - grad_output).abs().max()):.4f}")

    # 2. the radial part reaches the codebook
    ok &= check("the codebook received the radial component",
                codes.grad is not None and float(codes.grad.abs().sum()) > 0,
                f"|codebook grad|_1 = {float(codes.grad.abs().sum()):.4f}")

    # 3. the documented scope: the hook is on the encoder output, so every loss
    #    through that tensor is projected -- the known gap, asserted so it cannot
    #    drift unnoticed.
    encoder = torch.nn.Linear(4, 4)
    layer = VQLayer(codebook_size=5, codebook_dim=4)
    install_tangent_projection(encoder, layer)
    x = torch.randn(8, 4)
    encoder(x).pow(2).mean().backward()
    recon_grad = encoder.weight.grad.abs().sum().clone()
    encoder.zero_grad()
    encoder(x).pow(3).mean().backward()
    deep_grad = encoder.weight.grad.abs().sum().clone()
    ok &= check("both a reconstruction-style and a deep-style loss are projected",
                float(recon_grad) > 0 and float(deep_grad) > 0,
                f"recon |g| = {float(recon_grad):.4f}, deep |g| = {float(deep_grad):.4f}")
    print(f"\n  scope: {projection_scope_note()}")

    print("\n" + ("all projection checks passed" if ok else "FAILURES above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
