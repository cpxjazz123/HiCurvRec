"""Red-first checks for the closure/fork structure term."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import structure as S
import mixed_geometry as G


def _tri(h):
    # three points on the Euclidean half, 0.2 apart roughly
    q = torch.zeros(1, 3, 32)
    q[0, 1, 0] = 1.0
    q[0, 2, 1] = 1.0
    return q, torch.tensor([h], dtype=torch.float64)


def test_loss_is_zero_when_distances_match_the_targets():
    q = torch.zeros(1, 3, 32, dtype=torch.float64)
    q[0, 1, 0] = 1.0      # d(0,1) = 1
    q[0, 2, 0] = 2.0      # d(0,2) = 2, d(1,2) = 1
    d = G.mixed_sq(q[:, 0], q[:, 1]).item()
    h = torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float64)
    # set the third pair distance so that each pair equals s*h with s = d
    loss = S.structure_loss(q, h, scale=d, geometry_on=False)
    assert loss.item() < 1e-9


def test_collapse_is_penalised():
    q = torch.zeros(1, 3, 32)
    loss = S.structure_loss(q, torch.tensor([[1.0, 1.0, 1.0]]), scale=1.0, geometry_on=False)
    assert loss.item() > 0.5


def test_wedge_pair_is_penalised_less_than_a_closed_pair():
    # d(0,1)=1, d(0,2)=1, d(1,2)=2 with scale 1: the closed target of 1 doubles the
    # last pair, the wedge target of 2 matches it exactly.
    q = torch.zeros(1, 3, 32)
    q[0, 1, 0] = 1.0
    q[0, 2, 0] = -1.0
    closed = S.structure_loss(q, torch.tensor([[1.0, 1.0, 1.0]]), scale=1.0, geometry_on=False)
    wedge = S.structure_loss(q, torch.tensor([[1.0, 1.0, 2.0]]), scale=1.0, geometry_on=False)
    assert wedge.item() < closed.item()


def test_scale_calibration_is_deterministic_and_positive():
    q = torch.randn(4, 3, 32, generator=torch.Generator().manual_seed(0))
    h = torch.ones(4, 3)
    first = S.calibrate_scale(q, h)
    second = S.calibrate_scale(q, h)
    assert first == second and first > 0


def test_sampler_respects_edges_and_wedges():
    # Two separate centres: 0 owns a real triangle (1-2 edge exists), 1 owns a
    # real wedge (3-4 edge does not exist).  The fixture itself is checked below.
    nodes = np.arange(6)
    neighbours = {0: {1, 2, 3, 4}, 1: {0, 2}, 2: {0, 1}, 3: {0}, 4: {0}, 5: set()}
    assert 2 in neighbours[1] and 1 in neighbours[2]      # closed: 0-1, 0-2, 1-2
    assert 3 in neighbours[0] and 4 in neighbours[0] and 4 not in neighbours[3]
    graph = S.Graph(nodes=nodes, neighbours=neighbours,
                    triangle_pairs={0: [(1, 2)]}, wedge_pairs={0: [(3, 4)]})
    triples = S.sample_triples(graph, 64, random.Random(0))
    kinds = {kind for *_, kind in triples}
    assert triples and kinds == {"triangle", "wedge"}
    for centre, partner_a, partner_b, kind in triples:
        assert partner_a in neighbours[centre] and partner_b in neighbours[centre]
        if kind == "triangle":
            assert partner_b in neighbours[partner_a]
        else:
            assert partner_b not in neighbours[partner_a]


def test_hub_centres_do_not_dominate():
    neighbours = {0: set(range(1, 200)), 1: {0}, 2: {0}}
    graph = S.Graph(nodes=np.arange(200), neighbours=neighbours,
                    wedge_pairs={0: [(1, 2)]}, triangle_pairs={})
    triples = S.sample_triples(graph, 50, random.Random(0))
    centres = [t[0] for t in triples]
    assert centres.count(0) <= 5


def test_stress_gradient_is_exactly_zero_at_a_collision():
    # analytic: d/dx (||x||^2/a^2 - 1)^2 = 4x/a^2 (||x||^2/a^2 - 1) -> 0 at x = 0
    left = torch.zeros(4, 32, dtype=torch.float64, requires_grad=True)
    right = torch.zeros(4, 32, dtype=torch.float64)
    loss = S.pair_stress(left, right, scale=1.3, h=1.0, geometry_on=False)
    loss.sum().backward()
    assert torch.equal(left.grad, torch.zeros_like(left.grad))


def test_analytic_gradient_of_one_pair_matches_the_formula():
    left = torch.tensor([[0.4, 0.0]], dtype=torch.float64, requires_grad=True)
    right = torch.zeros(1, 2, dtype=torch.float64)
    scale, h = 2.0, 1.0
    # geometry-off on a width that is not 32: use the full 32-d layout instead
    left = torch.zeros(1, 32, dtype=torch.float64, requires_grad=True)
    left.data[0, 0] = 0.4
    value = S.pair_stress(left, torch.zeros(1, 32, dtype=torch.float64),
                          scale=scale, h=h, geometry_on=False)
    value.backward()
    a2 = (scale * h) ** 2
    expected = 4 * 0.4 / a2 * (0.4 ** 2 / a2 - 1)
    assert abs(left.grad[0, 0].item() - expected) < 1e-9


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, type(exc).__name__, exc)
    raise SystemExit(1 if failures else 0)
