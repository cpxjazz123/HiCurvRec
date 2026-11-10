"""The prefix-order supervision of issue #154: AF3 edge sets constrain prefix distances.

The structure enters as a differentiable ordering requirement on the *prefix embedding*
geometry, not as a change to the codebook, the assignment or the residual recursion: high
AF3 edges must sit closer in the Poincare ball than low AF3 edges, and the triangles whose
three edges are all high must close (their three prefixes cannot be pairwise far).
"""
from __future__ import annotations

import torch

#: The switches of the issue: OFF reproduces the baseline bit for bit; GEOMETRY_OFF takes
#: the c = 0 Euclidean limit of every hyperbolic quantity.
STRUCTURE_OFF = False
GEOMETRY_OFF = False

#: The MAD floor, so the spread never divides by zero on a degenerate batch.
TAU_FLOOR = 1e-6


def af3_per_edge(edges: torch.Tensor, triangles: torch.Tensor, nodes: int) -> torch.Tensor:
    """``AF3(e) = 4 - deg(u) - deg(v) + 3*m_e`` over the frozen training graph edges."""
    degree = torch.bincount(edges.reshape(-1), minlength=int(nodes)).to(torch.float64)
    multiplicity = torch.zeros(edges.shape[0], dtype=torch.float64)
    if triangles.numel():
        keys = {}
        for index, (tail, head) in enumerate(edges.tolist()):
            keys[(min(tail, head), max(tail, head))] = index
        for triangle in triangles.tolist():
            for tail, head in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                               (triangle[0], triangle[2])):
                index = keys.get((min(tail, head), max(tail, head)))
                if index is not None:
                    multiplicity[index] += 1.0
    return (4.0 - degree[edges[:, 0]] - degree[edges[:, 1]] + 3.0 * multiplicity)


def centre_and_spread(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``d_bar`` and ``tau``: the batch median and its median absolute deviation."""
    flat = distances.reshape(-1).to(torch.float64)
    d_bar = flat.median()
    tau = (flat - d_bar).abs().median().clamp_min(TAU_FLOOR)
    return d_bar, tau


def attribution(distance: torch.Tensor, d_bar: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """``s = sigmoid((d_bar - d) / tau)``: close prefixes score high."""
    return torch.sigmoid((d_bar.to(distance.dtype) - distance) / tau.to(distance.dtype))


def ball_distance(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    """``d_c`` in the Poincare ball; the c = 0 limit is the Euclidean distance."""
    if GEOMETRY_OFF or curvature <= 0:
        return (left - right).norm(dim=-1)
    from .poincare_contract import mobius_add
    moved = mobius_add(-left, right, float(curvature)).norm(dim=-1).clamp(max=1.0 - 1e-7)
    return 2.0 * torch.atanh(moved * (float(curvature) ** 0.5)) / (float(curvature) ** 0.5)


def mixed_distance(left_e: torch.Tensor, left_h: torch.Tensor, right_e: torch.Tensor,
                   right_h: torch.Tensor, curvature: float) -> torch.Tensor:
    """One number per pair: the mean of the Euclidean and the hyperbolic term."""
    return 0.5 * ((left_e - right_e).norm(dim=-1)
                  + ball_distance(left_h, right_h, curvature))


def gap_loss(high: torch.Tensor, low: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """``softplus(delta - (mean_H s - mean_B s))`` -- zero once the gap is at least delta."""
    return torch.nn.functional.softplus(delta - (high.mean() - low.mean()))


def triangle_loss(edge_attributions: torch.Tensor) -> torch.Tensor:
    """``mean(1 - min over a triangle's three edges)`` over the triangles given."""
    if edge_attributions.numel() == 0:
        return torch.zeros((), dtype=torch.float64)
    return (1.0 - edge_attributions.min(dim=-1).values).mean()


def structure_losses(prefix_e: torch.Tensor, prefix_h: torch.Tensor,
                     edges_high: torch.Tensor, edges_low: torch.Tensor,
                     triples_high: torch.Tensor, delta: torch.Tensor,
                     curvature: float) -> dict:
    """The two supervised terms on one shared prefix geometry.

    ``prefix_e``/``prefix_h`` are ``[n, d]`` prefix embeddings for the batch's items; the
    edges index into them. Both terms read the same ``s`` field, so they are one path.
    """
    if STRUCTURE_OFF:
        zero = torch.zeros((), dtype=torch.float64)
        return {"gap": zero, "tri": zero, "delta_bar": torch.zeros(()), "tau": torch.zeros(())}

    def distance(pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.zeros(0, dtype=torch.float64)
        tail, head = pairs[:, 0], pairs[:, 1]
        return mixed_distance(prefix_e[tail], prefix_h[tail], prefix_e[head], prefix_h[head],
                              curvature)

    high_distance = distance(edges_high)
    low_distance = distance(edges_low)
    both = torch.cat([high_distance, low_distance]) if high_distance.numel() or low_distance.numel() \
        else torch.zeros(1, dtype=torch.float64)
    d_bar, tau = centre_and_spread(both)
    high_s = attribution(high_distance, d_bar, tau)
    low_s = attribution(low_distance, d_bar, tau)
    gap = gap_loss(high_s, low_s, delta.to(gap_dtype := torch.float64))

    if triples_high.numel():
        corners = triples_high
        tri_pairs = torch.stack([corners[:, [0, 1]], corners[:, [1, 2]],
                                 corners[:, [0, 2]]], dim=1)
        flat = tri_pairs.reshape(-1, 2)
        tri_distance = distance(flat).reshape(tri_pairs.shape[0], 3)
        tri = triangle_loss(attribution(tri_distance, d_bar, tau))
    else:
        tri = torch.zeros((), dtype=torch.float64)
    return {"gap": gap, "tri": tri, "delta_bar": d_bar, "tau": tau}
