"""Native residual reference, and H/STE combination on the real RQLayer."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import native_product
from model.layers import RQLayer

E = 16


def ball(h):
    n = h.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return h * torch.tanh(n) / n


def log0(y):
    n = y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return y * torch.atanh(n) / n


def madd(a, b):
    a2 = a.pow(2).sum(-1, keepdim=True)
    b2 = b.pow(2).sum(-1, keepdim=True)
    ab = (a * b).sum(-1, keepdim=True)
    return (((1 + 2 * ab + b2) * a + (1 - a2) * b) / (1 + 2 * ab + a2 * b2))


def ref_residuals(latent, codes):
    re_ = latent[..., :E]
    rh = ball(latent[..., E:])
    out = []
    for code in codes:
        out.append(torch.cat([re_, log0(rh)], dim=-1))
        re_ = re_ - code[..., :E]
        rh = madd(-ball(code[..., E:]), rh)         # (-c) (+) r, the left inverse
    out.append(torch.cat([re_, log0(rh)], dim=-1))
    return out


def ref_reconstruct(codes):
    euclid = sum(c[..., :E] for c in codes)
    bh = ball(codes[-1][..., E:])
    for code in reversed(codes[:-1]):
        bh = madd(ball(code[..., E:]), bh)
    return torch.cat([euclid, log0(bh)], dim=-1)


def make_codes(level, seed, zero=False, rows=4):
    torch.manual_seed(seed)
    codes = []
    for _ in range(level):
        if zero:
            c = torch.zeros(rows, 32, dtype=torch.float64)
        else:
            c = torch.randn(rows, 32, dtype=torch.float64) * 0.2
            c = torch.cat([c[..., :E], torch.tanh(c[..., E:])], dim=-1)
        codes.append(c.clone().requires_grad_(True))
    return codes


def test_native_residual_matches_its_reference():
    for level in (1, 2, 3):
        for zero in (False, True):
            latent = torch.cat([torch.randn(4, E, dtype=torch.float64) * 0.2,
                                torch.tanh(torch.randn(4, E, dtype=torch.float64) * 0.2)],
                               dim=-1).requires_grad_(True)
            codes = make_codes(level, seed=level, zero=zero)
            got = native_product.residuals(latent, codes)
            want = ref_residuals(latent, codes)
            for lvl, (a, b) in enumerate(zip(got, want)):
                assert torch.allclose(a, b, rtol=1e-9, atol=1e-11), (level, zero, lvl)


def test_native_residual_gradients_match_and_gradcheck_both():
    latent = torch.cat([torch.randn(3, E, dtype=torch.float64) * 0.2,
                        torch.tanh(torch.randn(3, E, dtype=torch.float64) * 0.2)],
                       dim=-1).requires_grad_(True)
    codes = make_codes(2, seed=7, rows=3)
    g_got = torch.autograd.grad(native_product.residuals(latent, codes)[-1].sum(),
                                [latent] + codes, retain_graph=True, allow_unused=True)
    ref_lat = latent.clone().requires_grad_(True)
    ref_codes = [c.clone().requires_grad_(True) for c in codes]
    g_ref = torch.autograd.grad(ref_residuals(ref_lat, ref_codes)[-1].sum(),
                                [ref_lat] + ref_codes)
    for a, b in zip(g_got, g_ref):
        assert a is not None and torch.allclose(a, b, rtol=1e-8, atol=1e-10)
    assert torch.autograd.gradcheck(lambda z, *c: native_product.residuals(z, list(c))[-1],
                                    (latent.clone().requires_grad_(True),
                                     *[c.clone().requires_grad_(True) for c in codes]),
                                    eps=1e-6, atol=1e-6)
    assert torch.autograd.gradcheck(lambda z, *c: ref_residuals(z, list(c))[-1],
                                    (latent.clone().requires_grad_(True),
                                     *[c.clone().requires_grad_(True) for c in codes]),
                                    eps=1e-6, atol=1e-6)


def build_layer(seed=0):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(codebook_num=3, codebook_dim=32, codebook_size=256,
                          vq_type="vq", beta=0.25, sk_epsilon=-1.0, sk_iters=-1)
    layer = RQLayer(cfg).double()
    with torch.no_grad():
        for level, vq in enumerate(layer.vq_layers):
            embs = vq.get_code_embs()
            embs.normal_(0, 50.0)
            embs[level + 1] = 0.0
            embs[level + 1, :E] = 0.1 * (level + 1)
            h = torch.zeros(E, dtype=torch.float64)
            h[0], h[1] = 0.15 * (level + 1), 0.1 * (level + 1)
            embs[level + 1, E:] = torch.tanh(h)
        for level, vq in enumerate(layer.vq_layers):
            idx = level + 1
            vq._indices = lambda d, infer, idx=idx: torch.full((d.shape[0],), idx, dtype=torch.long)
    return layer


def test_real_rqlayer_h_and_encoder_ste_per_level():
    layer = build_layer()
    z = torch.empty(1, 32, dtype=torch.float64).normal_(0, 0.3).requires_grad_(True)
    out = layer(z, return_levels=True)
    levels = out[4]
    codes = [vq.embed_code(torch.tensor([i + 1])) for i, vq in enumerate(layer.vq_layers)]
    for k in range(3):
        got = levels[k]
        want = ref_reconstruct([c.detach() for c in codes[:k + 1]])
        assert torch.allclose(got.detach(), want, rtol=1e-8, atol=1e-10), k
        for j, vq in enumerate(layer.vq_layers):
            embs = vq.get_code_embs()
            g = torch.autograd.grad(got.sum(), embs, allow_unused=True, retain_graph=True)[0]
            if j > k:
                assert g is None or float(g.abs().sum()) == 0.0, (k, j)
            else:
                assert g is not None and float(g.abs().sum()) > 0.0, (k, j)
        jac = torch.autograd.functional.jacobian(
            lambda zz: layer(zz.unsqueeze(0), return_levels=True)[4][k][0], z[0])
        assert torch.allclose(jac, torch.eye(32, dtype=torch.float64), atol=1e-9), (k, (jac - torch.eye(32)).abs().max())


if __name__ == "__main__":
    test_native_residual_matches_its_reference()
    test_native_residual_gradients_match_and_gradcheck_both()
    test_real_rqlayer_h_and_encoder_ste_per_level()
    print("ok")


def test_native_residual_full_jacobian_each_level():
    for level in (1, 2, 3):
        latent = torch.cat([torch.randn(2, E, dtype=torch.float64) * 0.2,
                            torch.tanh(torch.randn(2, E, dtype=torch.float64) * 0.2)],
                           dim=-1).requires_grad_(True)
        codes = make_codes(level, seed=100 + level, rows=2)
        ref_codes = [c.clone().requires_grad_(True) for c in codes]
        for out_lvl in range(level + 1):
            j_got = torch.autograd.functional.jacobian(
                lambda z, *cs: native_product.residuals(z, list(cs))[out_lvl],
                (latent.clone().requires_grad_(True), *[c.clone().requires_grad_(True) for c in codes]),
                vectorize=True)
            j_ref = torch.autograd.functional.jacobian(
                lambda z, *cs: ref_residuals(z, list(cs))[out_lvl],
                (latent.clone().requires_grad_(True), *[c.clone().requires_grad_(True) for c in ref_codes]),
                vectorize=True)
            for a, b in zip(j_got, j_ref):
                assert torch.allclose(a, b, rtol=1e-7, atol=1e-9), (level, out_lvl)
        assert torch.autograd.gradcheck(
            lambda z, *cs: native_product.residuals(z, list(cs))[level],
            (latent.clone().requires_grad_(True), *[c.clone().requires_grad_(True) for c in codes]),
            eps=1e-6, atol=1e-6)
        assert torch.autograd.gradcheck(
            lambda z, *cs: ref_residuals(z, list(cs))[level],
            (latent.clone().requires_grad_(True), *[c.clone().requires_grad_(True) for c in ref_codes]),
            eps=1e-6, atol=1e-6)


def _level_code_jac(layer, z, k, j):
    level = layer(z, return_levels=True)[4][k][0]
    embs = layer.vq_layers[j].get_code_embs()
    index = j + 1
    jac = []
    for i in range(32):
        g = torch.autograd.grad(level[i], embs, retain_graph=True, allow_unused=True)[0]
        jac.append(torch.zeros(32, dtype=torch.float64) if g is None else g[index].clone())
    return torch.stack(jac), level


def test_real_rqlayer_codebook_jacobian_matches_reference():
    layer = build_layer(seed=3)
    z = torch.empty(1, 32, dtype=torch.float64).normal_(0, 0.3).requires_grad_(True)
    ref_codes = []
    for k in range(3):
        for j in range(len(layer.vq_layers)):
            got, level = _level_code_jac(layer, z, k, j)
            if j > k:
                assert float(got.abs().sum()) == 0.0, (k, j)
                continue
            rows = [layer.vq_layers[i].get_code_embs()[i + 1].detach().requires_grad_(True)
                    for i in range(k + 1)]
            jac_ref = torch.autograd.functional.jacobian(
                lambda *cs: ref_reconstruct(list(cs)), tuple(rows), vectorize=True)
            want = jac_ref[j].reshape(32, 32)
            assert torch.allclose(got, want, rtol=1e-6, atol=1e-8), (k, j, (got - want).abs().max())



if __name__ == "__main__":
    test_native_residual_matches_its_reference()
    test_native_residual_gradients_match_and_gradcheck_both()
    test_native_residual_full_jacobian_each_level()
    test_real_rqlayer_h_and_encoder_ste_per_level()
    test_real_rqlayer_codebook_jacobian_matches_reference()
    print("ok")
