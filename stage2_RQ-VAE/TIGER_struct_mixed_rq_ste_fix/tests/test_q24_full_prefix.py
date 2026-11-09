"""The full E16xH16 prefix readback: value, collision and non-collinear gradients."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mixed_geometry as G
import structure as S

D = torch.float64


def _phi(t):
    t = t.clamp_min(0.0)
    root = torch.sqrt(t.clamp_min(1e-30))
    return torch.where(t < 1e-6, t - t * t / 3, torch.asinh(root) ** 2)


def _manual(codes, h, s):
    e = 0.0
    cum = []
    for level in codes:
        e = e + level[..., :16]
        r = torch.sqrt((level[..., 16:] ** 2).sum(-1, keepdim=True) + 1e-24)
        sr = r.clamp_min(1e-12)
        b = torch.where(r > 0, torch.tanh(sr) / sr, torch.ones_like(r)) * level[..., 16:]
        cum.append(b)
        nested = cum[-1]
        for earlier in reversed(cum[:-1]):
            a2 = (earlier * earlier).sum(-1, keepdim=True)
            b2 = (nested * nested).sum(-1, keepdim=True)
            ab = (earlier * nested).sum(-1, keepdim=True)
            nested = ((1 + 2 * ab + b2) * earlier + (1 - a2) * nested) / (
                1 + 2 * ab + a2 * b2)
        nr = torch.sqrt((nested * nested).sum(-1, keepdim=True) + 1e-24)
        sn = nr.clamp(1e-12, 1 - 1e-12)
        logh = torch.where(nr > 0, torch.atanh(sn) / sn, torch.ones_like(nr)) * nested
        cum_q = torch.cat([e, logh], dim=-1) if len(cum) > 0 else None
        if len(cum) == 1:
            layers = [cum_q]
        else:
            layers.append(cum_q)
    total = 0.0
    for q in layers:
        terms = []
        for i, j, k in ((0, 1, 0), (0, 2, 1), (1, 2, 2)):
            bi, bj = q[:, i, 16:], q[:, j, 16:]
            ri = torch.sqrt((bi * bi).sum(-1, keepdim=True) + 1e-24)
            rj = torch.sqrt((bj * bj).sum(-1, keepdim=True) + 1e-24)
            si, sj = ri.clamp_min(1e-12), rj.clamp_min(1e-12)
            bi = torch.where(ri > 0, torch.tanh(si) / si, torch.ones_like(ri)) * bi
            bj = torch.where(rj > 0, torch.tanh(sj) / sj, torch.ones_like(rj)) * bj
            num = ((bi - bj) ** 2).sum(-1)
            den = (1 - (bi ** 2).sum(-1)) * (1 - (bj ** 2).sum(-1))
            d = ((q[:, i, :16] - q[:, j, :16]) ** 2).sum(-1) + _phi(num / den)
            terms.append((d / (s[:, 0] ** 2 * h[:, k] ** 2) - 1) ** 2)
        total = total + (terms[0] + terms[1] + terms[2]) / 3
    return total / 3


def _codes(scale=0.1):
    g = torch.Generator().manual_seed(7)
    return [(torch.rand(2, 3, 32, generator=g, dtype=D) - 0.5) * scale for _ in range(3)]


def test_the_full_prefix_matches_the_manual_readback():
    codes = _codes()
    h = torch.ones(2, 3, dtype=D)
    h[:, 2] = 2.0
    s = torch.ones(2, 1, dtype=D)
    got = S.reference_prefix_native(codes, h, s)
    exp = _manual(codes, h, s)
    assert torch.allclose(got, exp, rtol=0.0, atol=1e-10)


def test_the_full_prefix_is_finite_at_the_collision_and_near_collision():
    h = torch.ones(1, 3, dtype=D)
    h[:, 2] = 2.0
    s = torch.ones(1, 1, dtype=D)
    for scale in (0.0, 1e-5):
        codes = [torch.full((1, 3, 32), scale, dtype=D) for _ in range(3)]
        for c in codes:
            c.requires_grad_(True)
        loss = S.reference_prefix_native(codes, h, s)
        g = torch.autograd.grad(loss.sum(), codes)
        assert torch.isfinite(loss).all()
        assert all(torch.isfinite(x).all() for x in g)


def test_the_full_prefix_gradchecks_off_the_collision():
    codes = _codes(0.05)
    for c in codes:
        c.requires_grad_(True)
    h = torch.ones(1, 3, dtype=D)
    h[:, 2] = 2.0
    s = torch.ones(1, 1, dtype=D)
    tiny = [c[:1].detach().clone().requires_grad_(True) for c in codes]
    assert torch.autograd.gradcheck(
        lambda *cs: S.reference_prefix_native(list(cs), h, s), tuple(tiny),
        eps=1e-6, atol=1e-8, rtol=1e-6)


def test_the_reference_radial_map_matches_production_over_the_measured_norms():
    for norm in (0.16, 1.0, 5.0, 10.0, 15.0, 19.0, 20.0, 30.0):
        t = torch.zeros(1, 16, dtype=D)
        t[0, 0] = norm
        got = S._log0(S._ball(t))
        exp = G.log_zero(G.to_ball(t))
        assert torch.allclose(got, exp, rtol=0.0, atol=1e-9), (norm, got[0, 0], exp[0, 0])


def test_the_near_collision_uses_distinct_products_and_covers_the_h_domain():
    torch.manual_seed(3)
    h = torch.ones(1, 3, dtype=D)
    h[:, 2] = 2.0
    s = torch.ones(1, 1, dtype=D)
    base = torch.zeros(1, 3, 32, dtype=D)
    base[0, 0, 16] = 0.5
    base[0, 1, 16] = 0.5 + 1e-5
    base[0, 2, 16] = 0.5 - 2e-5
    base[0, 0, 0] = 0.01
    base[0, 2, 0] = -0.02
    codes = [base.clone() for _ in range(3)]
    for c in codes:
        c.requires_grad_(True)
    loss = S.reference_prefix_native(codes, h, s)
    assert 0.98 <= float(loss) <= 1.0
    g = torch.autograd.grad(loss.sum(), codes)
    assert all(torch.isfinite(x).all() for x in g)

    wide = [base.clone() * 1.8 for _ in range(3)]
    for c in wide:
        c.requires_grad_(True)
    loss = S.reference_prefix_native(wide, h, s)
    assert torch.isfinite(loss).all()
    g = torch.autograd.grad(loss.sum(), wide)
    assert all(torch.isfinite(x).all() for x in g)


if __name__ == "__main__":
    test_the_full_prefix_matches_the_manual_readback()
    test_the_full_prefix_is_finite_at_the_collision_and_near_collision()
    test_the_full_prefix_gradchecks_off_the_collision()
    test_the_reference_radial_map_matches_production_over_the_measured_norms()
    test_the_near_collision_uses_distinct_products_and_covers_the_h_domain()
    print("ok")
