"""Known-answer tests for issue #243, written from CONTRACT.md §1.7.

Expected values are hand-derived (log(3), log(19), pi/12 ...), never produced by the code
under test.  This file checks the reference implementation; the production code is checked
against the reference in ``test_reference.py``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import reference as ref  # noqa: E402

CURVATURES = (-3.0, -1.0, 0.0, 1.0, 3.0)


def approx(actual, expected, tol=1e-9, label=""):
    assert abs(float(actual) - float(expected)) <= tol, f"{label}: {actual} != {expected}"


def test_self_distance_is_zero():
    rng = np.random.default_rng(0)
    for k in CURVATURES:
        for _ in range(10):
            x = ref.proju(rng.normal(size=4) * 0.3, k)
            d = ref.dist(x, x, k)[0]
            approx(d, 0.0, 1e-9, f"self-distance k={k}")


def test_hyperbolic_radial_known_answers():
    k = -1.0
    e1 = np.array([[1.0, 0.0, 0.0]])
    # 2*artanh(r) = log((1+r)/(1-r))
    approx(ref.dist(np.zeros((1, 3)), 0.5 * e1, k)[0], math.log(3.0), 1e-12, "r=0.5")
    approx(ref.dist(np.zeros((1, 3)), 0.9 * e1, k)[0], math.log(19.0), 1e-12, "r=0.9")


def test_spherical_radial_known_answers():
    k = 1.0
    e1 = np.array([[1.0, 0.0, 0.0]])
    # 2*arctan(r)
    approx(ref.dist(np.zeros((1, 3)), 0.5 * e1, k)[0], 2.0 * math.atan(0.5), 1e-12, "r=0.5")
    approx(ref.dist(np.zeros((1, 3)), 0.25 * e1, k)[0], 2.0 * math.atan(0.25), 1e-12, "r=0.25")


def test_euclidean_known_answer():
    a = np.array([[1.0, 2.0]])
    b = np.array([[4.0, 6.0]])
    approx(ref.dist(a, b, 0.0)[0], 5.0, 1e-12, "euclid 3-4-5")


def test_symmetry():
    rng = np.random.default_rng(1)
    for k in CURVATURES:
        x = ref.proju(rng.normal(size=3) * 0.2, k)
        y = ref.proju(rng.normal(size=3) * 0.2, k)
        approx(ref.dist(x, y, k)[0], ref.dist(y, x, k)[0], 1e-12, f"symmetry k={k}")


def test_mobius_and_arcosh_agree():
    """CONTRACT §1.6: the two hyperbolic closed forms must agree in float64."""
    rng = np.random.default_rng(2)
    for k in (-3.0, -1.0):
        # stay at most a quarter of the ball radius: proju's eps clamp would otherwise
        # truncate the Mobius sum near the boundary and break the identity artificially
        radius = 0.25 / math.sqrt(abs(k))
        for _ in range(20):
            x = ref.proju(rng.normal(size=3) * radius, k)
            y = ref.proju(rng.normal(size=3) * radius, k)
            approx(ref.dist(x, y, k)[0], ref.dist_arcosh(x, y, k)[0], 1e-9, f"k={k}")


def test_stereographic_matches_sphere_geometry():
    """Model points must reproduce the true sphere geodesic after inverse stereographic mapping.

    With radius R = 1/sqrt(k), a sphere point at polar angle theta maps to model radius
    r = R * tan(theta/2).  Two such points at polar angles theta1/theta2 and azimuths
    phi1/phi2 must have model distance == R * arccos(<s1,s2>/R^2).
    """
    k = 1.0
    radius = 1.0 / math.sqrt(k)
    for theta1, theta2, phi in ((math.pi / 3, math.pi / 4, 0.0),
                                (math.pi / 2.5, math.pi / 6, 0.7),
                                (math.pi / 8, math.pi / 3, 1.9)):
        r1, r2 = radius * math.tan(theta1 / 2), radius * math.tan(theta2 / 2)
        p1 = np.array([[r1, 0.0]])
        p2 = np.array([[r2 * math.cos(phi), r2 * math.sin(phi)]])
        s1 = np.array([math.sin(theta1), 0.0, math.cos(theta1)])
        s2 = np.array([math.sin(theta2) * math.cos(phi),
                       math.sin(theta2) * math.sin(phi),
                       math.cos(theta2)])
        expected = ref.dist_sphere_geodesic(s1, s2, k)[0]
        approx(ref.dist(p1, p2, k)[0], expected, 1e-9, f"theta=({theta1:.3f},{theta2:.3f})")


def test_collinear_sphere_distance():
    """Same azimuth, polar angles pi/3 and pi/4 on the unit sphere: distance pi/12."""
    k = 1.0
    radius = 1.0 / math.sqrt(k)
    r_a = radius * math.tan((math.pi / 3) / 2)
    r_b = radius * math.tan((math.pi / 4) / 2)
    approx(ref.dist(np.array([[r_a, 0.0]]), np.array([[r_b, 0.0]]), k)[0],
           math.pi / 3 - math.pi / 4, 1e-9, "pi/12")


def test_exp_log_round_trip():
    rng = np.random.default_rng(3)
    for k in CURVATURES:
        # keep the tangent norm well inside the injectivity/ball limit for |k| = 3
        scale = 1.0 if k == 0 else 0.2 / math.sqrt(abs(k))
        tangent = rng.normal(size=(5, 4)) * scale
        manifold = ref.expmap0(tangent, k)
        back = ref.logmap0(manifold, k)
        assert np.allclose(back, tangent, atol=1e-8), f"round trip k={k}"


def test_projection_keeps_origin_and_clamps():
    for k in (-1.0, 1.0):
        assert np.allclose(ref.proju(np.zeros((1, 3)), k), 0.0), "zero must stay zero"
        far = ref.proju(np.array([[10.0, 0.0, 0.0]]), k)
        assert np.linalg.norm(far) < 1.0 / math.sqrt(abs(k)), "radius must be clamped"


def test_spearman_with_ties():
    # hand-computed: x = [1,2,2,3] -> ranks [0, 1.5, 1.5, 3]; y = [1,2,3,4] -> [0,1,2,3]
    x = [1.0, 2.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0, 4.0]
    rx = np.array([0.0, 1.5, 1.5, 3.0])
    ry = np.array([0.0, 1.0, 2.0, 3.0])
    rx -= rx.mean()
    ry -= ry.mean()
    expected = float((rx * ry).sum() / (np.linalg.norm(rx) * np.linalg.norm(ry)))
    approx(ref.spearman_with_ties(x, y), expected, 1e-12, "spearman ties")
    assert ref.spearman_with_ties([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None, "constant -> NA"
    # monotone but strongly non-linear: Spearman is exactly 1, Pearson is not.
    # Without this case a Pearson substitution survives the tie test above.
    mono = ref.spearman_with_ties([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 60.0, 120.0])
    approx(mono, 1.0, 1e-12, "monotone -> spearman 1")
    pearson = float(np.corrcoef([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 60.0, 120.0])[0, 1])
    assert abs(mono - pearson) > 1e-3, "the case must separate Spearman from Pearson"


def test_jaccard_uses_intersection_over_union():
    approx(ref.jaccard({1, 2, 3}, {2, 3, 4}), 2.0 / 4.0, 1e-12, "jaccard")
    approx(ref.jaccard(set(), set()), 0.0, 1e-12, "empty jaccard")
    approx(ref.jaccard({1}, {1}), 1.0, 1e-12, "identical jaccard")


def test_bhattacharyya_known_answer():
    p = {1: 0.5, 2: 0.5}
    q = {1: 0.5, 2: 0.5}
    approx(ref.bhattacharyya(p, q), 1.0, 1e-12, "identical distributions")
    q2 = {1: 1.0, 3: 0.0}
    approx(ref.bhattacharyya(p, q2), math.sqrt(0.5), 1e-12, "half overlap")
    assert ref.bhattacharyya({1: 0.5, 2: 0.5}, {3: 1.0}) == 0.0, "disjoint support"


def test_ego_and_induced_edges():
    # 0 -> 1 -> 2, 3 -> 1 ; radius 1 from 0 reaches {0,1}; radius 2 reaches {0,1,2,3}
    edges = [(0, 1), (1, 2), (3, 1)]
    assert ref.ego_nodes(edges, 4, 0, 1) == {0, 1}
    assert ref.ego_nodes(edges, 4, 0, 2) == {0, 1, 2, 3}
    induced = ref.induced_edges(edges, {0, 1, 2, 3})
    assert induced == {(0, 1), (1, 2), (3, 1)}, "all edges inside the node set must be kept"
    partial = ref.induced_edges(edges, {0, 1})
    assert partial == {(0, 1)}, "edges leaving the set are dropped"


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
