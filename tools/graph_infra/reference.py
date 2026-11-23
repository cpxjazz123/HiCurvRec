"""Independent reference implementation for issue #243 (numpy, float64).

Rules of this file:

* it imports nothing from the production modules and nothing from torch/geoopt;
* every formula is written from ``CONTRACT.md`` §1, not from the code it checks;
* where two closed forms exist (Möbius/artanh vs arcosh) both are implemented so
  they can check each other.

Nothing here is used at training time; it exists to be compared against.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------- κ-stereographic
def tan_k(v, k: float):
    """tan_k from CONTRACT §1.2."""
    v = np.asarray(v, dtype=np.float64)
    if k > 0:
        return np.tan(math.sqrt(k) * v) / math.sqrt(k)
    if k < 0:
        return np.tanh(math.sqrt(-k) * v) / math.sqrt(-k)
    return v


def tan_k_inv(u, k: float):
    """tan^{-1}_k from CONTRACT §1.3."""
    u = np.asarray(u, dtype=np.float64)
    if k > 0:
        return np.arctan(math.sqrt(k) * u) / math.sqrt(k)
    if k < 0:
        scale = math.sqrt(-k)
        clipped = np.clip(scale * u, -1.0 + EPS, 1.0 - EPS)
        return np.arctanh(clipped) / scale
    return u


def proju(x: np.ndarray, k: float, eps: float = 1e-3) -> np.ndarray:
    """Clamp into the coordinate domain of the model.

    k < 0: the open Poincare ball of radius (1-eps)/sqrt|k|.
    k > 0: the stereographic plane has **no hyperbolic radius bound** -- every finite
           coordinate is a valid sphere point (the projection pole sits at infinity), so
           only a large numerical guard is applied.  #243 used the hyperbolic clamp here,
           which is one of the reasons the spherical distance was wrong.
    k = 0: identity.
    """
    x = np.asarray(x, dtype=np.float64)
    if k == 0:
        return x.copy()
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if k > 0:
        guard = 1e6 / math.sqrt(k)
        return np.where(norm > guard, x * (guard / np.maximum(norm, EPS)), x)
    maxnorm = (1.0 - eps) / math.sqrt(abs(k))
    return np.where(norm > maxnorm, x * (maxnorm / np.maximum(norm, EPS)), x)


def expmap0(x: np.ndarray, k: float) -> np.ndarray:
    """k > 0: r = R * tan(|v| / R) with R = 1/sqrt(k), so that dist(0, expmap0(v)) = 2 |v|.

    The 2x normalisation matches the Poincare branch; with it k = 0 is the continuous flat limit.
    """
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if k > 0:
        radius = 1.0 / math.sqrt(k)
        mapped = radius * np.tan(norm / radius)
    else:
        mapped = tan_k(norm, k)
    scale = np.where(norm > EPS, mapped / np.maximum(norm, EPS), 1.0)
    return proju(scale * x, k)


def logmap0(y: np.ndarray, k: float) -> np.ndarray:
    """k > 0: |v| = R * arctan(r / R); inverse of expmap0 above."""
    y = proju(y, k)
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    if k > 0:
        radius = 1.0 / math.sqrt(k)
        mapped = radius * np.arctan(norm / radius)
    else:
        mapped = tan_k_inv(norm, k)
    scale = np.where(norm > EPS, mapped / np.maximum(norm, EPS), 1.0)
    return scale * y


def mobius_add(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x2 = (x * x).sum(-1, keepdims=True)
    y2 = (y * y).sum(-1, keepdims=True)
    xy = (x * y).sum(-1, keepdims=True)
    denominator = 1.0 - 2.0 * k * xy + k * k * x2 * y2
    if np.any(np.abs(denominator) < 1e-9):
        raise FloatingPointError("mobius_add denominator vanished")
    numerator = (1.0 - 2.0 * k * xy - k * y2) * x + (1.0 + k * x2) * y
    return proju(numerator / denominator, k)


def mobius_matvec(matrix: np.ndarray, x: np.ndarray, k: float) -> np.ndarray:
    return expmap0(matrix @ logmap0(x, k), k)


def stereo_to_sphere(u: np.ndarray, k: float) -> np.ndarray:
    """Inverse stereographic projection onto S^(d+1) of radius R = 1/sqrt(k).

    This is the *definition* of the positive-curvature model, so the reference distance
    below uses the three-dimensional points and an angle, not a stereographic closed form.
    """
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    radius = 1.0 / math.sqrt(k)
    squared = (u * u).sum(-1, keepdims=True)
    # The radius is NOT a global scale factor: the stereographic coordinate itself is
    # measured in the same units, so R must appear inside the construction.
    r2 = radius * radius
    planar = 2.0 * r2 * u / (r2 + squared)
    height = radius * (squared - r2) / (r2 + squared)
    return np.concatenate([planar, height], axis=-1)


def dist(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    """Distance in the model of curvature ``k``.

    k > 0: computed through the inverse stereographic projection -- R * arccos(<P,Q>/R^2).
           The Mobius/tan_k^-1 form is the **Poincare** formula and is not valid here.
    k < 0: Poincare ball, 2 * tan_k^-1(|(-x) +_k y|).
    k = 0: the flat limit of the same family, 2 * |x - y| (the v1 value |x - y| was a
           factor-two discontinuity with the curved branches).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if k == 0:
        return 2.0 * np.linalg.norm(x - y, axis=-1)
    if k > 0:
        radius = 1.0 / math.sqrt(k)
        left = stereo_to_sphere(x, k)
        right = stereo_to_sphere(y, k)
        cosine = (left * right).sum(-1) / (radius * radius)
        return radius * np.arccos(np.clip(cosine, -1.0, 1.0))
    moved = mobius_add(-x, y, k)
    return 2.0 * tan_k_inv(np.linalg.norm(moved, axis=-1), k)


def dist_arcosh(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    """Closed form for k<0 only: (1/sqrt|k|)*arcosh(...) -- must match ``dist``."""
    if k >= 0:
        raise ValueError("the arcosh form is hyperbolic-only")
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    c = abs(k)
    xx = (x * x).sum(-1)
    yy = (y * y).sum(-1)
    diff = ((x - y) ** 2).sum(-1)
    inner = 1.0 + 2.0 * c * diff / ((1.0 - c * xx) * (1.0 - c * yy))
    return np.arccosh(np.maximum(inner, 1.0 + 1e-15)) / math.sqrt(c)


def dist_sphere_geodesic(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    """Sphere of radius R=1/sqrt(k): R * arccos(<x,y>/R^2), valid only for |x|=|y|=R."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    radius = 1.0 / math.sqrt(k)
    cosine = (x * y).sum(-1) / (radius * radius)
    return radius * np.arccos(np.clip(cosine, -1.0, 1.0))


# --------------------------------------------------------------------- graph reference
def experimental_shortest_paths(edges, n_nodes: int, root: int, depth: int) -> dict:
    """Plain BFS on the undirected projection, written independently of scipy."""
    adjacency = [[] for _ in range(n_nodes)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        if int(right) != int(left):
            adjacency[int(right)].append(int(left))
    seen = {int(root): 0}
    frontier = [int(root)]
    for hop in range(1, depth + 1):
        nxt = []
        for node in frontier:
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen[neighbour] = hop
                    nxt.append(neighbour)
        frontier = nxt
        if not frontier:
            break
    return seen


def induced_edges(edges, nodes) -> set:
    """Every edge whose two endpoints are in ``nodes`` -- the induced subgraph."""
    keep = {int(node) for node in nodes}
    return {(int(a), int(b)) for a, b in edges if int(a) in keep and int(b) in keep}


def ego_nodes(edges, n_nodes: int, root: int, radius: int) -> set:
    return set(experimental_shortest_paths(edges, n_nodes, root, radius))


# --------------------------------------------------------------------- metrics reference
def spearman_with_ties(x, y):
    """Spearman with average ranks; a constant input returns None (NA), never 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None

    def ranks(values):
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        rank = np.empty(values.size, dtype=np.float64)
        i = 0
        while i < values.size:
            j = i
            while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
                j += 1
            rank[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return rank

    rx = ranks(x) - ranks(x).mean()
    ry = ranks(y) - ranks(y).mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denominator == 0:
        return None
    return float((rx * ry).sum() / denominator)


def jaccard(left, right) -> float:
    """Real intersection over union, not a proxy."""
    left, right = set(left), set(right)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def bhattacharyya(p: dict, q: dict) -> float:
    """Σ_k sqrt(p_k q_k) over the shared support; missing keys contribute nothing."""
    return float(sum(math.sqrt(p[k] * q[k]) for k in set(p) & set(q)))
