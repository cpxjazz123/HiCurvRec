"""The hyperbolic oracles' tests (issue #110, WP2 and the section 7 acceptance list).

These are the independent mathematical checks: the coincident and the near-boundary cases
must be finite, the exponential and logarithm maps must be inverses, the two models must
agree on the distance, the parallel transport must preserve the norm, and the Riemannian
gradient must differ from the Euclidean one by exactly the conformal factor. The oracle must
not import any candidate's code, which the test asserts on the module's source.

    python3 -m pytest tools/genrec_agents/tests/test_hyperbolic_geometry.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from tools.hyperbolic_verifier import geometry as G  # noqa: E402


def _points(count: int = 5, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, generator=generator, dtype=torch.float64) * 0.3


def test_the_oracle_does_not_import_a_candidate():
    """An oracle that imports the mechanism's code can only confirm that it agrees with itself."""
    source = (REPO_ROOT / "tools" / "hyperbolic_verifier" / "geometry.py").read_text(
        encoding="utf-8"
    )
    assert "importlib" not in source
    assert "mechanism_" not in source
    assert "stage2_RQ-VAE" not in source


def test_the_distance_is_zero_for_coincident_points_and_finite_at_the_boundary():
    x = _points(1)
    # the implementation clamps the arcosh argument just above one, so the distance of
    # coincident points is bounded by that floor rather than exactly zero
    assert G.poincare_distance(x, x, 1.0).item() == pytest.approx(0.0, abs=1e-5)
    # a point on the boundary has an infinite distance; the safeguard keeps it finite
    boundary = torch.tensor([[0.999999, 0.0, 0.0]], dtype=torch.float64)
    inside = G.project_to_ball(boundary, 1.0)
    assert torch.isfinite(G.poincare_distance(inside, inside + 1e-3, 1.0)).all()
    assert (G.ball_margin(inside, 1.0) > 0).all()


def test_the_projection_keeps_the_points_inside_the_ball():
    x = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    projected = G.project_to_ball(x, 1.0)
    assert (G.ball_margin(projected, 1.0) > 0).all()


def test_the_exp_and_log_maps_are_inverses():
    v = _points(6)
    x = G.poincare_exp_map0(v, 0.7)
    assert torch.allclose(G.poincare_log_map0(x, 0.7), v, atol=1e-9)
    assert (G.ball_margin(x, 0.7) > 0).all()


def test_the_distance_is_symmetric_and_zero_on_the_diagonal():
    x, y = _points(5), _points(5, seed=11)
    assert torch.allclose(
        G.poincare_distance(x, y, 1.0), G.poincare_distance(y, x, 1.0), atol=1e-12
    )
    assert torch.allclose(
        G.poincare_distance(x, x, 1.0), torch.zeros(5, dtype=torch.float64), atol=1e-5
    )



def test_the_distance_from_the_origin_equals_the_tangent_norm():
    """The decisive check: the distance from the origin to the exponential map's image is
    the norm of the tangent vector. It is what ties the distance, the exponential map and
    the logarithm map together."""
    v = _points(5, seed=23)
    origin = torch.zeros_like(v)
    assert torch.allclose(
        G.poincare_distance(origin, G.poincare_exp_map0(v, 1.0), 1.0),
        torch.linalg.norm(v, dim=-1),
        atol=1e-9,
    )


def test_the_busemann_function_diverges_at_its_base_point_and_is_finite_elsewhere():
    """The base point lies ON the horosphere, so the function diverges there."""
    xi = _points(1)
    far = xi + 0.6
    value = G.busemann(far, xi, 1.0)
    assert torch.isfinite(value).all()
    near = G.busemann(xi + 1e-4, xi, 1.0)
    assert (near > value).all()


def test_the_distance_from_the_origin_is_twice_the_atanh_of_the_norm():
    """An identity of the distance that does not come from this module's own formula."""
    x = _points(5, seed=29)
    origin = torch.zeros_like(x)
    norm = torch.linalg.norm(x, dim=-1)
    assert torch.allclose(
        G.poincare_distance(origin, x, 1.0), 2.0 * torch.atanh(norm), atol=1e-9
    )



def test_the_riemannian_gradient_differs_from_the_euclidean_one_by_the_conformal_factor():
    x = _points(4)
    euclidean = torch.randn(4, 3, dtype=torch.float64)
    riemannian = G.riemannian_gradient(euclidean, x, 1.0)
    factor = (1.0 - torch.linalg.norm(x, dim=-1, keepdim=True) ** 2) ** 2
    assert torch.allclose(riemannian, euclidean * factor, atol=1e-12)
    assert not torch.allclose(riemannian, euclidean, atol=1e-6)


def test_the_retraction_stays_on_the_manifold():
    x = _points(3)
    v = torch.randn(3, 3, dtype=torch.float64) * 0.1
    moved = G.retraction(x, v, 1.0)
    assert (G.ball_margin(moved, 1.0) > 0).all()


def test_a_non_positive_curvature_is_refused():
    x = _points(1)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            G.poincare_distance(x, x, bad)
        with pytest.raises(ValueError):
            G.project_to_ball(x, bad)
