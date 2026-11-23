"""κ-GCN layer (CONTRACT.md §2): log-map, weighted aggregation, exp-map.

The official implementation (`kappaGCNConv` in RingBDStack/GraphMoRE) does
`mobius_matvec` → `logmap0` → degree-normalised aggregation → `expmap0(project=True)`.
This is the same computation with two deliberate differences, both written into the
contract: aggregation is a **mean over in- and out-edges together** rather than
`deg^{-1/2}` symmetric normalisation, and the directional weights are separate because
the item graph is directed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import manifold


def _segment_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if values.shape[0] == 0:
        return values.new_zeros(size, values.shape[1])
    out = values.new_zeros(size, values.shape[1])
    weight = values.new_zeros(size, 1)
    out.index_add_(0, index, values)
    weight.index_add_(0, index, values.new_ones(values.shape[0], 1))
    return out / weight.clamp(min=1.0)


class KappaGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, curvature: float):
        super().__init__()
        self.curvature = float(curvature)
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.w_out = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h: torch.Tensor, in_edges: torch.Tensor, out_edges: torch.Tensor) -> torch.Tensor:
        tangent = manifold.logmap0(h, self.curvature)
        # the aggregate lives in the OUTPUT width, not the tangent width
        aggregate = h.new_zeros(h.shape[0], self.w_in.out_features)
        if in_edges.numel():
            src, dst = in_edges[:, 0], in_edges[:, 1]
            aggregate = aggregate + _segment_mean(self.w_in(tangent[src]), dst, h.shape[0])
        if out_edges.numel():
            src, dst = out_edges[:, 0], out_edges[:, 1]
            aggregate = aggregate + _segment_mean(self.w_out(tangent[src]), dst, h.shape[0])
        combined = self.w_self(tangent) + aggregate + self.bias
        return manifold.expmap0(combined, self.curvature)


class KappaExpert(nn.Module):
    """Two κ-GCN layers in one curvature space; output is a manifold point per node."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, curvature: float):
        super().__init__()
        self.curvature = float(curvature)
        self.layer1 = KappaGCNLayer(in_dim, hidden, curvature)
        self.layer2 = KappaGCNLayer(hidden, out_dim, curvature)

    def forward(self, x: torch.Tensor, in_edges: torch.Tensor, out_edges: torch.Tensor) -> torch.Tensor:
        h = manifold.expmap0(manifold.proju(x, self.curvature), self.curvature)
        h = self.layer1(h, in_edges, out_edges)
        return self.layer2(h, in_edges, out_edges)
