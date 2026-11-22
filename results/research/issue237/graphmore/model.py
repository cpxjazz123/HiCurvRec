"""GraphMoRE: κ-GCN experts with node-level gating, adapted to the item graph.

Faithful points (verified against the official code, `models.py` / `exp.py`):

* ``Encoder`` — two κ-GCN convolutions, input pushed onto the manifold with
  ``expmap0``, curvature-0 expert is a plain GCN with a frozen curvature.
* ``kappaGCNConv`` — the aggregate is ``expmap0(D^-1/2 (A+I) D^-1/2 · logmap0(x))``,
  i.e. the manifold points are pulled to the tangent space at the origin, GCN-normalised
  and summed there, then pushed back.  This is exactly what the official code does, so
  the aggregation is manifold-valued, not a Euclidean weighted sum.
* ``Experts`` — five encoders concatenated and layer-normalised.
* ``Gating`` — two GCN layers, mean-pooled over the 2-hop and 3-hop ego neighbourhoods,
  concatenated and passed through a linear layer to a softmax over the five experts.
* Distortion / link-prediction score — per-expert squared distance, weighted by
  ``softmax(w_u ⊙ w_v)`` and summed, i.e. the paper's "weight-aligned distance".

Deviations, all recorded in the report:

* ``geoopt`` is not installed, so ``manifold.py`` implements the κ-stereographic
  operations.  ``mobius_matvec`` is the tangent-space linear map rather than geoopt's
  algebraically equivalent closed form.
* The optimizer is Adam on tangent-space parameters rather than ``RiemannianAdam``.
* Ego neighbourhoods are capped (64 nodes at 2 hops, 128 at 3) because the frozen graph
  has 24,587 nodes where the paper's benchmarks have a few thousand; mean-pooling with a
  capped indicator equals pooling the capped subgraph.
* The 768-dim head is a projection from the concatenated tangent-space experts, which
  the issue requires (C16) and the paper does not have.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold import Stereographic

CURVATURES = [-3.0, -1.0, 0.0, 1.0, 3.0]


def sparse_adjacency(edge_index: torch.Tensor, n: int, device) -> torch.Tensor:
    """``D^-1/2 (A + I) D^-1/2`` as a coalesced sparse tensor."""
    loop = torch.arange(n, device=device)
    rows = torch.cat([edge_index[0], edge_index[1], loop])
    cols = torch.cat([edge_index[1], edge_index[0], loop])
    values = torch.ones(rows.numel(), device=device)
    adjacency = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n)).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1.0)
    inv_sqrt = degree.pow(-0.5)
    scaled = adjacency.values() * inv_sqrt[adjacency.indices()[0]] * inv_sqrt[adjacency.indices()[1]]
    return torch.sparse_coo_tensor(adjacency.indices(), scaled, (n, n)).coalesce()


class KappaLinear(nn.Module):
    """``expmap0(W · logmap0(x) + b_tangent)`` on the manifold of curvature k."""

    def __init__(self, manifold: Stereographic, in_dim: int, out_dim: int):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.manifold.mobius_matvec(self.weight, x)
        bias = self.manifold.expmap0(self.bias)
        return self.manifold.mobius_add(result, bias)


class KappaGCNConv(nn.Module):
    """One κ-GCN layer: manifold linear, tangent-space propagation, back to the manifold."""

    def __init__(self, manifold: Stereographic, in_dim: int, out_dim: int):
        super().__init__()
        self.manifold = manifold
        self.linear = KappaLinear(manifold, in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        tangent = self.manifold.logmap0(x)
        propagated = torch.sparse.mm(adjacency, tangent)
        return self.manifold.expmap0(propagated)


class Encoder(nn.Module):
    """Two-layer κ-GCN; ``encode`` returns the tangent-space coordinates at the origin."""

    def __init__(self, k: float, in_dim: int, hidden_dim: int, out_dim: int,
                 learnable: bool = True):
        super().__init__()
        self.manifold = Stereographic(k, learnable=learnable and k != 0.0)
        self.layer1 = KappaGCNConv(self.manifold, in_dim, hidden_dim)
        self.layer2 = KappaGCNConv(self.manifold, hidden_dim, out_dim)

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise, scale into the ball, then push onto the manifold."""
        x = F.normalize(x, dim=-1)
        k = float(torch.abs(self.manifold.k))
        if k > 0.0:
            x = x * (0.5 / (k ** 0.5))
        return self.manifold.expmap0(x)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.layer1(self._prepare(x), adjacency)
        return self.layer2(h, adjacency)

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.manifold.logmap0(self.forward(x, adjacency))


class Experts(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 curvatures: list = CURVATURES):
        super().__init__()
        self.curvatures = list(curvatures)
        self.experts = nn.ModuleList([
            Encoder(k, in_dim, hidden_dim, out_dim, learnable=(k != 0.0))
            for k in self.curvatures
        ])
        self.norm = nn.LayerNorm(len(self.curvatures) * out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.cat([expert(x, adjacency) for expert in self.experts], -1))

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.cat([expert.encode(x, adjacency) for expert in self.experts], -1)

    def curvature_values(self) -> list:
        return [float(expert.manifold.k) for expert in self.experts]


class Gating(nn.Module):
    """Ego-pooled GCN features -> softmax over experts."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_experts: int,
                 num_scales: int = 2, temperature: float = 1.0):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_dim)
        self.classifier = nn.Linear(out_dim * num_scales, num_experts)
        self.temperature = temperature

    def features(self, x: torch.Tensor, adjacency: torch.Tensor,
                 egos: list) -> torch.Tensor:
        h = F.relu(torch.sparse.mm(adjacency, self.conv1(x)))
        h = torch.sparse.mm(adjacency, self.conv2(h))
        pooled = [torch.sparse.mm(ego, h) for ego in egos]
        return torch.cat(pooled, -1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                egos: list) -> torch.Tensor:
        logits = self.classifier(self.features(x, adjacency, egos))
        return F.softmax(logits / self.temperature, dim=-1)


class GraphMoRE(nn.Module):
    """Experts + gating, with the weight-aligned distance and the 768-dim head."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, expert_dim: int = 128,
                 gating_hidden: int = 64, gating_out: int = 32,
                 out_dim: int = 768, curvatures: list = CURVATURES,
                 num_scales: int = 2):
        super().__init__()
        self.experts = Experts(in_dim, hidden_dim, expert_dim, curvatures)
        self.gating = Gating(in_dim, gating_hidden, gating_out, len(curvatures), num_scales)
        self.head = nn.Linear(expert_dim * len(curvatures), out_dim)
        self.expert_dim = expert_dim

    # -- shared pieces -------------------------------------------------------
    def weights(self, x, adjacency, egos):
        return self.gating(x, adjacency, egos)

    def expert_embedding(self, x, adjacency):
        return self.experts.encode(x, adjacency)

    def align(self, embedding: torch.Tensor, weights: torch.Tensor,
              left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Weight-aligned squared distance between two node sets (paper's form)."""
        diff = (embedding[left] - embedding[right]) ** 2
        diff = diff.reshape(diff.shape[0], -1, self.expert_dim).sum(dim=2)
        pair = F.softmax(weights[left] * weights[right], dim=1)
        return torch.sum(diff * pair, dim=-1)

    def forward(self, x, adjacency, egos):
        weights = self.weights(x, adjacency, egos)
        embedding = self.expert_embedding(x, adjacency)
        return embedding, weights


class FermiDirac(nn.Module):
    """``sigmoid((r - d) / t)`` on a distance, as in the official code."""

    def __init__(self, r: float = 2.0, t: float = 1.0):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(float(r)), requires_grad=False)
        self.t = nn.Parameter(torch.tensor(float(t)), requires_grad=False)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((self.r - distance) / self.t)
