"""Independent H reference for the cumulative native readback."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import native_product

E = 16


def ball(h):
    norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return h * torch.tanh(norm) / norm


def log0(y):
    norm = y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return y * torch.atanh(norm) / norm


def mobius_add(a, b):
    a2 = a.pow(2).sum(-1, keepdim=True)
    b2 = b.pow(2).sum(-1, keepdim=True)
    ab = (a * b).sum(-1, keepdim=True)
    return (((1 + 2 * ab + b2) * a + (1 - a2) * b)
            / (1 + 2 * ab + a2 * b2))


def reference(codes):
    euclid = sum(c[..., :E] for c in codes)
    ballp = ball(codes[-1][..., E:])
    for code in reversed(codes[:-1]):
        ballp = mobius_add(ball(code[..., E:]), ballp)
    return torch.cat([euclid, log0(ballp)], dim=-1)


def test_cumulative_readback_matches_the_independent_reference():
    torch.manual_seed(0)
    codes = [torch.randn(5, 32, dtype=torch.float64) * 0.2 for _ in range(3)]
    # non-collinear H directions in the ball
    codes = [torch.cat([c[..., :E], torch.tanh(c[..., E:])], dim=-1) for c in codes]
    for level in (1, 2, 3):
        got = native_product.reconstruct(codes[:level])
        want = reference(codes[:level])
        assert torch.allclose(got, want, rtol=1e-10, atol=1e-12), (level, (got - want).abs().max())

    grad_codes = [c.clone().requires_grad_(True) for c in codes]
    got_grads = torch.autograd.grad(native_product.reconstruct(grad_codes).sum(),
                                    grad_codes, allow_unused=True)
    ref_codes = [c.clone().requires_grad_(True) for c in codes]
    ref_grads = torch.autograd.grad(reference(ref_codes).sum(-1).sum(), ref_codes)
    for a, b in zip(got_grads, ref_grads):
        assert a is not None and torch.allclose(a, b, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    test_cumulative_readback_matches_the_independent_reference()
    print("ok")
