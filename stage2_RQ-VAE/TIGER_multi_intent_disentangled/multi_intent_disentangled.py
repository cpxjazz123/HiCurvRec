"""Multi-Intent Disentangled Hyperbolic Item Encoder (Issue #25).

The item text feeds several low-dimensional hyperbolic intent branches at once.
Each branch is supervised by a *different* behaviour-teacher view, so the branches
have something genuinely distinct to encode rather than three copies of one
objective:

* ``transition``  -- directed last-item -> target successors, i.e. what follows;
* ``cooccurrence`` -- items bought by the same user, order ignored;
* ``collaborative`` -- the same pairs with the item popularity divided out, so
  two popular items stop looking similar just because both are popular.

On the encoder side (step 2-4 of the issue):

1. every branch takes its tangent feature from the same text latent and maps it
   with its own exponential map, giving ``h_i^(a) = Exp_o(u_i^(a))``;
2. branch ``a`` is scored against its own view with a hyperbolic ranking hinge;
3. the branches' tangent features are pushed apart by a decorrelation constraint,
   which is what stops them from encoding the same thing twice;
4. ``Log_o`` brings the branches back to the tangent space -- exactly the
   ``u_i^(a)`` that produced them -- where a gate read off the *text embedding*
   fuses them into one tangent vector and one ``z_i = Exp_o(...)`` is handed to the
   unchanged three-level quantizer.

On the decoder side (step 5) one head per intent replays that intent's own view on
the quantizer's continuous output through a straight-through proxy that refuses
gradient to every codebook, so the behaviour signal never moves a codeword. The
output stays a single three-level semantic id with no per-intent codebook.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURVATURE = 1.0
BALL_EPS = 1e-5
RANK_MARGIN = 0.5
DECORRELATION_WEIGHT = 0.1
TANGENT_SCALE = 0.5
MIN_TRIPLES_PER_STEP = 64


def _expmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Origin exponential map with a unit-speed tangent: ``||v||`` is the arc length."""
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm) * x


def _logmap0(y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Origin logarithm map, the exact inverse of :func:`_expmap0`.

    ``d(0, Exp_0(v)) = (2 / sqrt(c)) * atanh(sqrt(c) * ||v|| / 2)``, so the inverse
    scales by ``2 * atanh(sqrt(c) ||y||) / (sqrt(c) ||y||)``; dropping that factor
    of two returns ``v / 2``, which the check asserts against.
    """
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-12)
    radius = torch.atanh(sqrt_c * norm.clamp(max=(1.0 - BALL_EPS) ** 0.5))
    return 2.0 * radius / (sqrt_c * norm) * y


def _geodesic(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Poincare geodesic distance; the acosh reduction runs in double precision."""
    sqrt_c = c ** 0.5
    limit = (1.0 - BALL_EPS) / c
    norm_x = torch.linalg.vector_norm(x, dim=-1).clamp_max(limit ** 0.5)
    norm_y = torch.linalg.vector_norm(y, dim=-1).clamp_max(limit ** 0.5)
    squared = (x - y).square().sum(dim=-1, dtype=torch.float64)
    conform = ((1.0 - c * norm_x.double().square())
               * (1.0 - c * norm_y.double().square())).clamp_min(1e-7)
    argument = 1.0 + 2.0 * c * squared / conform
    return torch.acosh(argument.clamp_min(1.0 + 1e-12)).to(x.dtype) / sqrt_c


class _BlockCodebookGradient(torch.autograd.Function):
    """Identity forward that refuses to pass gradient back into the codebooks."""

    @staticmethod
    def forward(ctx, quantized: torch.Tensor, *codebooks: torch.Tensor) -> torch.Tensor:
        ctx.codebook_count = len(codebooks)
        return quantized.view_as(quantized)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (grad_output,) + (None,) * ctx.codebook_count


def _ste_without_codebook_gradient(quantized: torch.Tensor,
                                   codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    live = [codebook for codebook in codebooks if codebook.requires_grad]
    if not live:
        return quantized
    return _BlockCodebookGradient.apply(quantized, *live)


def build_intent_views(frame_pairs: dict[str, np.ndarray], n_items: int) -> dict[str, dict]:
    """Turn each behaviour view into (anchor, positive) pairs, popularity-aware.

    ``frame_pairs`` holds one ``(left, right)`` array per view. A view keeps only
    the pairs whose left item has at least ``MIN_NEIGHBOURS`` partners, so every
    anchor in the returned index has a real neighbour to be ranked against.
    """
    views: dict[str, dict] = {}
    popularity = np.zeros(n_items, dtype=np.float64)
    for pairs in frame_pairs.values():
        if pairs.size:
            np.add.at(popularity, pairs[:, 0], 1.0)
            np.add.at(popularity, pairs[:, 1], 1.0)
    for name, pairs in frame_pairs.items():
        if pairs.size == 0:
            raise RuntimeError(f"behaviour view {name!r} produced no pair")
        degree = np.zeros(n_items, dtype=np.int64)
        np.add.at(degree, pairs[:, 0], 1)
        np.add.at(degree, pairs[:, 1], 1)
        order = np.argsort(degree, kind="stable")
        keep = degree[order] >= MIN_NEIGHBOURS
        views[name] = {
            "pairs": pairs,
            "anchors": order[keep].astype(np.int64),
            "item_space": np.int64(n_items),
        }
    return views


MIN_NEIGHBOURS = 2


def view_triples(view: dict, batch: int, generator: torch.Generator,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw ``batch`` (anchor, positive, negative) triples from one view."""
    pairs = view["pairs"]
    anchors = view["anchors"]
    pick = torch.randint(anchors.shape[0], (batch,), generator=generator).numpy()
    left = anchors[pick]
    # The positive is any partner of that anchor; the neighbours are small, so a
    # short scan over the anchor's own row is enough.
    starts = np.searchsorted(pairs[:, 0], pairs[:, 0], side="left")
    ends = np.searchsorted(pairs[:, 0], pairs[:, 0], side="right")
    positive = np.empty(batch, dtype=np.int64)
    for index, item in enumerate(left):
        row = pairs[starts[item]:ends[item], 1]
        positive[index] = row[int(torch.randint(row.shape[0], (1,), generator=generator))]
    negative = torch.randint(
        int(view["item_space"]), (batch,), generator=generator).numpy()
    return (
        torch.as_tensor(left, device=device),
        torch.as_tensor(positive, device=device),
        torch.as_tensor(negative, device=device),
    )


def intent_ranking_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                        margin: float = RANK_MARGIN, curvature: float = CURVATURE) -> torch.Tensor:
    """``max(0, m + d_H(h_i, h_j) - d_H(h_i, h_k))`` for one intent branch."""
    return F.relu(
        margin + _geodesic(anchor, positive, curvature)
        - _geodesic(anchor, negative, curvature)
    ).mean()


def intent_decorrelation_loss(stacked: torch.Tensor) -> torch.Tensor:
    """Push the intent branches apart in tangent space.

    ``stacked`` is ``(intents, rows, dim)``. Squared cosine, so parallel *and*
    antiparallel duplication are both penalised: two intents that encode the same
    direction up to a sign still encode the same relation. It takes one stacked
    tensor rather than a list of them so the launcher's gradient probe can see the
    term it is checking.
    """
    intents = int(stacked.shape[0])
    if intents < 2:
        return stacked.sum() * 0.0
    unit = F.normalize(stacked, dim=-1)
    terms: list[torch.Tensor] = []
    for first in range(intents):
        for second in range(first + 1, intents):
            terms.append((unit[first] * unit[second]).sum(-1).square().mean())
    return torch.stack(terms).mean()


def _intent_tangents(latent: torch.Tensor, gate: nn.Module) -> list[torch.Tensor]:
    return [branch(latent) for branch in gate.branches]


class MultiIntentGate(nn.Module):
    """The per-intent tangent branches plus the text-conditioned mixing weights."""

    def __init__(self, latent_dim: int, text_dim: int, intents: int = 3) -> None:
        super().__init__()
        self.intents = int(intents)
        self.branches = nn.ModuleList(
            nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(self.intents)
        )
        for branch in self.branches:
            nn.init.normal_(branch.weight, std=1.0 / latent_dim ** 0.5)
        self.weight = nn.Linear(text_dim, self.intents)

    def forward(self, latent: torch.Tensor, text_embedding: torch.Tensor):
        tangents = _intent_tangents(latent, self)
        weights = F.softmax(self.weight(text_embedding), dim=-1)
        mixed = sum(weight.unsqueeze(-1) * tangent
                    for weight, tangent in zip(weights.T, tangents))
        return mixed, weights, tangents

    def intent_points(self, tangents: list[torch.Tensor],
                      curvature: float = CURVATURE) -> list[torch.Tensor]:
        return [_expmap0(tangent, curvature) for tangent in tangents]

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def fuse_intents(points: list[torch.Tensor], weights: torch.Tensor,
                 curvature: float = CURVATURE) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """``Log_o`` back to the tangent space, gate, then ``Exp_o`` once.

    ``Log_o`` of each branch's own point is the tangent that produced it, so this
    is the issue's step 4 made explicit: the mixture happens in the tangent space
    and the item leaves as a single hyperbolic point.
    """
    tangents = [_logmap0(point, curvature) for point in points]
    mixed = sum(weight.unsqueeze(-1) * tangent
                for weight, tangent in zip(weights.T, tangents))
    rms = mixed.square().sum(-1).mean().sqrt().clamp_min(1e-12)
    return _expmap0(mixed / rms * TANGENT_SCALE, curvature), tangents


def install_multi_intent_encoder(model: nn.Module, gate: nn.Module,
                                 curvature: float = CURVATURE) -> nn.Module:
    """Send the fused ``z_i`` to the quantizer; the branch itself is trainer-owned.

    The encoder module is not replaced and nothing is attached to it, so
    ``model.state_dict()`` keeps exactly the keys the checkpoint probe expects.
    """
    encoder = model.encoder
    original_forward = encoder.forward

    def patched_forward(embeddings: torch.Tensor) -> torch.Tensor:
        latent = original_forward(embeddings)
        _mixed, weights, tangents = gate(latent, embeddings)
        fused, _back = fuse_intents(gate.intent_points(tangents, curvature), weights, curvature)
        gate.last_points = gate.intent_points(tangents, curvature)
        gate.last_weights = weights
        gate.last_tangents = tangents
        return fused

    encoder.forward = patched_forward
    return gate


class IntentRecoveryHeads(nn.Module):
    """One decoder-side head per intent, replaying that intent's own view."""

    def __init__(self, dim: int, intents: int = 3,
                 curvature: float = CURVATURE) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.heads = nn.ModuleList(nn.Linear(dim, dim) for _ in range(intents))

    def forward(self, quantized: torch.Tensor,
                codebooks: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        continuous = _ste_without_codebook_gradient(quantized, codebooks)
        return [_expmap0(head(continuous), self.curvature) for head in self.heads]

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
