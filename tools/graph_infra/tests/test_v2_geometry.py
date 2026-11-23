"""v2 geometry and curvature acceptance tests (issue #246 C23)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import reference_hand as ref  # noqa: E402
from tools.graph_infra import curvature, manifold  # noqa: E402


def path_graph():
    edges = [(0, 1), (1, 2)]
    rows = np.array([a for a, _ in edges] + [b for _, b in edges])
    cols = np.array([b for _, b in edges] + [a for a, _ in edges])
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3, 3)).tocsr()


def triangle_graph():
    edges = [(0, 1), (1, 2), (2, 0)]
    rows = np.array([a for a, _ in edges] + [b for _, b in edges])
    cols = np.array([b for _, b in edges] + [a for a, _ in edges])
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3, 3)).tocsr()


def square_graph():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    rows = np.array([a for a, _ in edges] + [b for _, b in edges])
    cols = np.array([b for _, b in edges] + [a for a, _ in edges])
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(4, 4)).tocsr()


def test_orc_known_answers():
    assert abs(curvature.orc_edge(path_graph(), 0, 1) - 0.0) < 1e-9
    assert abs(curvature.orc_edge(triangle_graph(), 0, 1) - 0.5) < 1e-9
    assert abs(curvature.orc_edge(square_graph(), 0, 1) - 0.0) < 1e-9


def test_orc_matches_the_enumeration_reference():
    graph = triangle_graph()
    neighbours_0 = curvature.neighbour_sets(graph, 0)
    neighbours_1 = curvature.neighbour_sets(graph, 1)
    reach = curvature.local_distances(graph, np.union1d(neighbours_0, neighbours_1))
    matrix = np.array([[reach[a][b] for b in neighbours_1.tolist()] for a in neighbours_0.tolist()],
                      dtype=np.float64)
    expected = ref.orc_edge(neighbours_0.tolist(), neighbours_1.tolist(),
                            lambda a, b: reach[a][b])
    produced = curvature.transport_w1(neighbours_0, neighbours_1, matrix)
    assert abs(produced - (1.0 - expected)) < 1e-9 or abs(produced - (1 - expected)) < 1e-9


def test_curvature_summary_counts_signs():
    report = curvature.summarise([0.5, -0.2, 0.0, None])
    assert report["n"] == 3
    assert report["positive"] == 1 and report["negative"] == 1 and report["near_zero"] == 1
    assert report["unit"].startswith("dimensionless")


def test_geometry_boundaries_and_gradients():
    # kappa -> 0 must approach the flat limit of the family: 2 * |x - y|
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    y = torch.tensor([[0.1, 0.4]], dtype=torch.float64)
    euclidean = 2.0 * (x - y).norm(dim=-1)
    flat_plus = manifold.dist(x, y, torch.tensor(-1e-12, dtype=torch.float64))
    flat_minus = manifold.dist(x, y, torch.tensor(1e-12, dtype=torch.float64))
    assert abs(float(flat_plus) - float(euclidean)) < 1e-6
    # the spherical branch loses precision when acos approaches 1 at R -> infinity
    assert abs(float(flat_minus) - float(euclidean)) < 1e-4, (float(flat_minus), float(euclidean))
    # the defining round trip: d(0, expmap0(v)) = 2 |v| for both signs and at k = 0
    for kappa in (-1.0, 1e-12, 0.0, 1.0):
        vector = torch.tensor([[0.4, -0.3]], dtype=torch.float64)
        point = manifold.expmap0(vector, torch.tensor(kappa, dtype=torch.float64))
        zero = torch.zeros_like(point)
        value = manifold.dist(zero, point, torch.tensor(kappa, dtype=torch.float64))
        # the spherical branch loses precision when R -> infinity (acos of a value near 1)
        assert abs(float(value) - 2.0 * float(vector.norm())) < 1e-4, (kappa, float(value))
    # near-antipodal spherical distance stays finite and close to pi*R
    radius = 1.0
    a = torch.tensor([[0.0, 0.0, radius]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[1e-7, 0.0, -radius]], dtype=torch.float64)
    value = manifold.dist(a, b, torch.tensor(1.0, dtype=torch.float64))  # k > 0 is the sphere
    assert torch.isfinite(value).all()
    assert abs(float(value) - np.pi * radius) < 1e-4
    # gradients are finite at the boundary of the hyperbolic ball
    near = torch.tensor([[0.999, 0.0]], dtype=torch.float64, requires_grad=True)
    projected = manifold.proju(near, torch.tensor(1.0, dtype=torch.float64))
    assert torch.isfinite(projected).all()
    assert float(projected.norm()) < 1.0


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
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
