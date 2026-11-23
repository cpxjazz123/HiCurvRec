"""Production-vs-reference comparison (issue #243 C4/C6/C7/C12).

Every production operator is compared against the numpy float64 reference on random
inputs, and against the hand-computed values from CONTRACT §1.7, so an error shared by
both implementations cannot pass unnoticed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import kappa_gcn, manifold  # noqa: E402
from tools.graph_infra import reference as ref  # noqa: E402

CURVATURES = (-3.0, -1.0, 0.0, 1.0, 3.0)
TOL = 1e-4


def _pair(rng, k, radius_fraction=0.25, dim=4, count=16):
    radius = 1.0 if k == 0 else radius_fraction / math.sqrt(abs(k))
    x = rng.normal(size=(count, dim)) * radius
    y = rng.normal(size=(count, dim)) * radius
    return x, y


def test_proju_matches_reference():
    rng = np.random.default_rng(10)
    for k in CURVATURES:
        x, _ = _pair(rng, k, radius_fraction=3.0)  # deliberately outside the ball
        assert np.allclose(manifold.proju(torch.tensor(x), k).numpy(), ref.proju(x, k), atol=TOL)
    zero = torch.zeros(1, 4)
    for k in (-1.0, 1.0):
        assert torch.allclose(manifold.proju(zero, k), zero), "zero must stay the origin"


def test_exp_log_match_reference():
    rng = np.random.default_rng(11)
    for k in CURVATURES:
        x, _ = _pair(rng, k, radius_fraction=0.3)
        assert np.allclose(manifold.expmap0(torch.tensor(x), k).numpy(), ref.expmap0(x, k), atol=TOL)
        y, _ = _pair(rng, k, radius_fraction=0.5)
        assert np.allclose(manifold.logmap0(torch.tensor(y), k).numpy(), ref.logmap0(y, k), atol=TOL)


def test_mobius_add_matches_reference():
    rng = np.random.default_rng(12)
    for k in CURVATURES:
        x, y = _pair(rng, k)
        actual = manifold.mobius_add(torch.tensor(x), torch.tensor(y), k).numpy()
        assert np.allclose(actual, ref.mobius_add(x, y, k), atol=TOL), f"mobius k={k}"


def test_distance_matches_reference_and_known_answers():
    rng = np.random.default_rng(13)
    for k in CURVATURES:
        x, y = _pair(rng, k)
        actual = manifold.dist(torch.tensor(x), torch.tensor(y), k).numpy()
        assert np.allclose(actual, ref.dist(x, y, k), atol=TOL), f"dist k={k}"
    # hand values from CONTRACT §1.7, checked on the PRODUCTION function
    e1 = torch.tensor([[1.0, 0.0, 0.0]])
    zero = torch.zeros(1, 3)
    assert abs(float(manifold.dist(zero, 0.5 * e1, -1.0)) - math.log(3.0)) < 1e-6
    assert abs(float(manifold.dist(zero, 0.9 * e1, -1.0)) - math.log(19.0)) < 1e-5
    assert abs(float(manifold.dist(zero, 0.5 * e1, 1.0)) - 2.0 * math.atan(0.5)) < 1e-6
    assert abs(float(manifold.dist(zero, e1, 0.0)) - 1.0) < 1e-6


def test_sphere_distance_is_not_the_poincare_formula():
    """The exact #243 bug: the Poincare formula returns pi/2 here; the sphere returns 4*atan(0.9)."""
    x = torch.tensor([[0.9]])
    y = torch.tensor([[-0.9]])
    actual = float(manifold.dist(x, y, 1.0))
    assert abs(actual - 4.0 * math.atan(0.9)) < 1e-6, actual
    assert abs(actual - math.pi / 2) > 1.0, "must not collapse to the Poincare value"
    assert abs(float(manifold.dist(x, x, 1.0))) < 1e-9


def test_sphere_matches_reference_over_a_grid():
    rng = np.random.default_rng(77)
    for k in (0.5, 1.0, 3.0):
        radius = 1.0 / math.sqrt(k)
        x = rng.normal(size=(16, 3)) * radius * 3.0
        y = rng.normal(size=(16, 3)) * radius * 3.0
        produced = manifold.dist(torch.tensor(x), torch.tensor(y), k).numpy()
        assert np.allclose(produced, ref.dist(x, y, k), atol=TOL), f"sphere k={k}"
        assert np.allclose(manifold.expmap0(torch.tensor(x), k).numpy(), ref.expmap0(x, k), atol=TOL)
        assert np.allclose(manifold.logmap0(torch.tensor(x), k).numpy(), ref.logmap0(x, k), atol=TOL)


def test_sphere_projection_does_not_use_the_hyperbolic_radius():
    """k>0 coordinates are unbounded; a 5/sqrt(k) radius must survive projection intact."""
    k = 1.0
    far = torch.tensor([[5.0, 0.0]])
    assert torch.allclose(manifold.proju(far, k), far), "positive curvature must not be ball-clamped"
    # and the distance to a far point is finite and below pi*R
    assert float(manifold.dist(torch.zeros(1, 2), far, k)) < math.pi / math.sqrt(k)


def test_distance_is_not_double_scaled():
    """The #242 bug: a 2x coefficient. This pins k=-3 against the hand value."""
    e1 = torch.tensor([[1.0, 0.0, 0.0]])
    zero = torch.zeros(1, 3)
    r = 0.1
    k = -3.0
    expected = 2.0 * math.atanh(math.sqrt(3.0) * r) / math.sqrt(3.0)
    actual = float(manifold.dist(zero, r * e1, k))
    assert abs(actual - expected) < 1e-6, (actual, expected)
    wrong = 2.0 * expected
    assert abs(actual - wrong) > 1e-3, "the distance must not carry the #242 factor of two"


def test_kappa_layer_identity_weights_matches_reference():
    """With identity weights the layer is exp(log(h_u) + mean(log(h_v))); check by hand."""
    torch.manual_seed(0)
    k = -1.0
    layer = kappa_gcn.KappaGCNLayer(3, 3, k)
    with torch.no_grad():
        layer.w_self.weight.copy_(torch.eye(3))
        layer.w_self.bias.zero_()
        layer.w_in.weight.copy_(torch.eye(3))
        layer.w_out.weight.copy_(torch.eye(3))
        layer.bias.zero_()
    h = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.15]])
    # 0 -> 1 and 2 -> 1 : node 1 has two in-edges, so its aggregate is the mean
    in_edges = torch.tensor([[0, 1], [2, 1]])
    out_edges = torch.tensor([[1, 0], [1, 2]])
    produced = layer(h, in_edges, out_edges).detach().numpy()
    tangent = ref.logmap0(h.numpy(), k)
    expected = []
    for node in range(3):
        incoming = [int(src) for src, dst in in_edges.tolist() if int(dst) == node]
        outgoing = [int(src) for src, dst in out_edges.tolist() if int(dst) == node]
        aggregate = np.zeros(3)
        if incoming:
            aggregate += tangent[incoming].mean(axis=0)
        if outgoing:
            aggregate += tangent[outgoing].mean(axis=0)
        expected.append(ref.expmap0(tangent[node] + aggregate, k))
    assert np.allclose(produced, np.stack(expected), atol=TOL), "kappa layer arithmetic"


def test_kappa_layer_direction_sensitivity():
    """Flipping every edge must change the output: direction is actually used."""
    torch.manual_seed(1)
    layer = kappa_gcn.KappaGCNLayer(4, 4, -1.0)
    h = torch.randn(5, 4) * 0.1
    in_edges = torch.tensor([[0, 1], [2, 3], [1, 4]])
    out_edges = torch.tensor([[1, 0], [3, 2], [4, 1]])
    a = layer(h, in_edges, out_edges)
    b = layer(h, out_edges, in_edges)  # swap in/out
    assert not torch.allclose(a, b, atol=1e-6), "swapping direction must change the layer"


def test_expert_output_lives_in_its_manifold():
    torch.manual_seed(2)
    for k in CURVATURES:
        expert = kappa_gcn.KappaExpert(6, 8, 4, k)
        x = torch.randn(7, 6)
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        z = expert(x, edges, edges)
        assert z.shape == (7, 4)
        # k<0 lives in a ball of radius 1/sqrt|k|; k>0 is a stereographic plane and has no
        # radius bound at all, so only finiteness (and a valid sphere point) can be asserted
        if k < 0:
            radius = 1.0 / math.sqrt(abs(k))
            assert float(z.norm(dim=-1).max()) <= radius + 1e-5, f"expert escaped its ball k={k}"
        assert torch.isfinite(z).all()


def test_gradients_flow_through_the_manifold_ops():
    """Every parameter must receive a finite gradient; a detached expmap would cut layer 1 off."""
    torch.manual_seed(3)
    expert = kappa_gcn.KappaExpert(6, 8, 4, -1.0)
    x = torch.randn(7, 6)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    z = expert(x, edges, edges)
    z.sum().backward()
    missing = [name for name, parameter in expert.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    finite = all(bool(torch.isfinite(p.grad).all()) for p in expert.parameters())
    assert finite, "non-finite gradient"
    moved = sum(float(p.grad.abs().sum()) for p in expert.parameters())
    assert moved > 0.0, "all gradients are zero"


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures.append((name, str(error)))
            print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
