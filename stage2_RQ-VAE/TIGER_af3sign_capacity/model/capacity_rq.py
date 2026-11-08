"""Issue #145: static capacity allocation from the TRAIN graph's AF3 sign mix.

The frozen rule reads only the TRAIN behaviour graph: the AF3 edge values of the
canonical graph are counted by sign (threshold 0), and the three proportions are
turned into per-level dimensions by largest remainder. Level 1 is fixed at E32;
levels 2 and 3 split their 32 coordinates into H, E and S components.

The measured TRAIN counts are frozen in the tree:

    edges 315391   minus 315390   zero 1   plus 0

so the largest-remainder allocation is H32 x E0 x S0 for levels 2 and 3. The
sphere dimension is zero and therefore has no code path; the rule itself is
still exercised by the tests on synthetic counts.

The geometry is the repository's frozen one (`poincare_contract`): curvature
c = 1, distance d_c, d^2/4 in the cost, native Moebius residual, and the readout

    phi(q) = q_E(L1) + log_0(q_H(L2) (+) q_H(L3))

in the 32-dimensional tangent space the decoder reads. At c = 0 the whole
expression is the Euclidean sum, which is the geometry-off switch.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .poincare_contract import expmap0, logmap0, mobius_add, project

#: The frozen TRAIN edge counts, by sign of F_ij = 4 - d_i - d_j + 3 T_ij.
SIGN_COUNTS = {"edges": 315391, "minus": 315390, "zero": 1, "plus": 0}
#: Curvature of the hyperbolic components; the static rule never learns or scans it.
CURVATURE = 1.0
#: The frozen recipe's commitment weight.
BETA = 0.25
#: The total codeword width.
TOTAL_DIM = 32
#: Level 1 is fixed at E32 by the issue.
LEVEL1 = (0, 32, 0)  # (H, E, S)


def sign_proportions(counts: dict | None = None) -> tuple[float, float, float]:
    """``(p_minus, p_zero, p_plus)`` from the frozen or supplied edge counts."""
    counts = counts or SIGN_COUNTS
    total = float(counts["edges"])
    if total <= 0:
        raise ValueError("the TRAIN graph has no edges; the allocation is undefined")
    return (float(counts["minus"]) / total,
            float(counts["zero"]) / total,
            float(counts["plus"]) / total)


def allocate(proportions: tuple[float, float, float], width: int = 32) -> tuple[int, int, int]:
    """Largest remainder, in the order (H, E, S), ties to the lower index.

    The rule is deterministic and uses no data beyond the three proportions.
    """
    if width <= 0:
        raise ValueError("allocation width must be positive")
    raw = [float(value) * width for value in proportions]
    base = [int(value) for value in raw]
    remainder = width - sum(base)
    order = sorted(range(3), key=lambda index: (-(raw[index] - base[index]), index))
    for index in order[:remainder]:
        base[index] += 1
    return tuple(base)


def layouts() -> tuple[tuple[int, int, int], ...]:
    """The frozen per-level (H, E, S) split derived from the frozen counts."""
    return (LEVEL1, allocate(sign_proportions()), allocate(sign_proportions()))


def hyperbolic_cost(residual: torch.Tensor, code: torch.Tensor,
                    curvature: float = CURVATURE) -> torch.Tensor:
    """``d_c(residual, exp_0(code))^2 / 4`` for one pair, any width.

    At ``c = 1`` this is ``atanh(||(-exp_0(code)) (+) residual||)^2``: the
    repository's distance convention gives ``d = 2 atanh(...)``. The same
    expression is what ``reference_contract.json`` states for width 1.
    """
    ball = expmap0(code, curvature)
    delta = mobius_add(-ball, residual, curvature)
    return torch.atanh(delta.norm(dim=-1).clamp(max=1 - 1e-7)).pow(2)


def _to_space(residual: torch.Tensor, current: str, target: str,
              curvature: float) -> torch.Tensor:
    """Move a residual into the next level's space by the canonical maps."""
    if current == target:
        return residual
    if current == "E" and target == "H":
        return expmap0(residual, curvature)
    if current == "H" and target == "E":
        return logmap0(residual, curvature)
    raise NotImplementedError("the frozen allocation contains no sphere component; "
                              "an S branch is outside this candidate")


def _space_of(layout: tuple[int, int, int]) -> str:
    hyperbolic, euclidean, spherical = layout
    if spherical:
        raise NotImplementedError("the frozen allocation has no sphere dimension")
    if hyperbolic and euclidean:
        raise ValueError("the frozen allocation is per-level, not per-coordinate mixed")
    if hyperbolic:
        return "H"
    if euclidean:
        return "E"
    raise ValueError("a level with no dimensions cannot hold a code")


class CapacityVQLevel(nn.Module):
    """One 256-code level in the space its allocated dimensions define.

    The codeword is stored as 32 tangent-space coordinates. A hyperbolic level
    maps it with ``exp_0``; a Euclidean level uses it as is. The residual update
    is native: Euclidean subtraction, or ``(-code) (+) residual`` in the ball.
    """

    def __init__(self, size: int, layout: tuple[int, int, int],
                 curvature: float = CURVATURE, beta: float = BETA,
                 sk_epsilon: float = -1.0, sk_iters: int = -1):
        super().__init__()
        self.layout = tuple(int(value) for value in layout)
        self.space = _space_of(self.layout)
        self.dim = sum(self.layout)
        self.n_embed = int(size)
        self.curvature = float(curvature)
        self.beta = float(beta)
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        self.embed = nn.Embedding(self.n_embed, self.dim)

    def get_code_embs(self) -> nn.Parameter:
        return self.embed.weight

    # -- the baseline's assignment rule ------------------------------------
    @staticmethod
    def center_distance(distances: torch.Tensor) -> torch.Tensor:
        maximum, minimum = distances.max(), distances.min()
        middle = (maximum + minimum) / 2
        amplitude = maximum - middle + 1e-5
        if not bool(amplitude > 0):
            raise ValueError("Cannot center a constant distance matrix.")
        return (distances - middle) / amplitude

    @torch.no_grad()
    def sinkhorn(self, distances: torch.Tensor, epsilon: float = 0.003,
                 iterations: int = 5) -> torch.Tensor:
        q = torch.exp(-distances / epsilon)
        batch, codes = q.shape
        q /= q.sum(-1, keepdim=True).sum(-2, keepdim=True)
        for _ in range(iterations):
            q /= q.sum(0, keepdim=True)
            q /= codes
            q /= q.sum(1, keepdim=True)
            q /= batch
        q *= batch
        return q

    def _indices(self, distances: torch.Tensor, infer_use_sk: bool) -> torch.Tensor:
        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            centered = self.center_distance(distances).double()
            q = self.sinkhorn(centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                raise RuntimeError("Sinkhorn assignment returned NaN or infinity.")
            return torch.argmax(q, dim=-1)
        return torch.argmin(distances, dim=-1)

    # -- the allocated cost ------------------------------------------------
    def distances(self, residual: torch.Tensor) -> torch.Tensor:
        codes = self.get_code_embs()
        if self.space == "E":
            return (residual.unsqueeze(1) - codes.unsqueeze(0)).pow(2).sum(-1)
        ball = expmap0(codes, self.curvature)
        delta = mobius_add(-ball.unsqueeze(0), residual.unsqueeze(1), self.curvature)
        distance = 2.0 * torch.atanh(delta.norm(dim=-1).clamp(max=1 - 1e-7))
        return distance.pow(2) / 4.0

    def _point_cost(self, residual: torch.Tensor, picked: torch.Tensor) -> torch.Tensor:
        if self.space == "E":
            return (residual - picked).pow(2).sum(-1)
        ball = expmap0(picked, self.curvature)
        delta = mobius_add(-ball, residual, self.curvature)
        return (2.0 * torch.atanh(delta.norm(dim=-1).clamp(max=1 - 1e-7))).pow(2) / 4.0

    def forward(self, residual: torch.Tensor, infer_use_sk: bool = False):
        cost = self.distances(residual)
        index = self._indices(cost, infer_use_sk)
        codes = self.get_code_embs()
        picked = codes[index]
        point = expmap0(picked, self.curvature) if self.space == "H" else picked
        onehot = F.one_hot(index, self.n_embed)
        used = onehot.sum(0).float()
        unused = int((used == 0).sum().item())
        if (torch.distributed.is_available() and torch.distributed.is_initialized()
                and infer_use_sk is False
                and not getattr(self, "_skip_ddp_reduce", False)):
            torch.distributed.all_reduce(used, op=torch.distributed.ReduceOp.SUM)
            unused = int((used == 0).sum().item())
        commitment = self._point_cost(residual, point.detach())
        codebook_term = self._point_cost(residual.detach(), point)
        quant_loss = (codebook_term + self.beta * commitment).mean() / float(self.dim)
        det = point.detach()
        if self.space == "E":
            next_residual = residual - det
        else:
            next_residual = project(mobius_add(-det, residual, self.curvature),
                                    self.curvature)
        return next_residual, quant_loss, unused, index, picked

    @torch.no_grad()
    def init_codebook(self, residual: torch.Tensor, device: torch.device) -> None:
        from .utils import kmeans
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed or torch.distributed.get_rank() == 0:
            coordinates = (residual.detach().float() if self.space == "E"
                           else logmap0(residual.detach().float(), self.curvature))
            self.embed.weight.copy_(kmeans(coordinates, self.n_embed))
        if distributed:
            torch.distributed.broadcast(self.embed.weight, src=0)


class CapacityRQ(nn.Module):
    """Three levels whose spaces are the frozen sign allocation."""

    def __init__(self, sizes: Iterable[int], *, curvature: float = CURVATURE,
                 beta: float = BETA, sk_epsilon: float = -1.0, sk_iters: int = -1):
        super().__init__()
        sizes = [int(size) for size in sizes]
        layouts = globals()["layouts"]()
        if len(sizes) != len(layouts):
            raise ValueError("the frozen allocation has one layout per level")
        self.curvature = float(curvature)
        self.layouts = layouts
        self.levels = nn.ModuleList()
        for index, (size, layout) in enumerate(zip(sizes, layouts)):
            last = index == len(sizes) - 1
            self.levels.append(CapacityVQLevel(
                size, layout, curvature=curvature, beta=beta,
                sk_epsilon=sk_epsilon if last else -1.0,
                sk_iters=sk_iters if last else -1,
            ))

    @property
    def vq_layers(self) -> nn.ModuleList:
        return self.levels

    def forward(self, z: torch.Tensor, infer_use_sk: bool = False):
        residual = z
        spaces = [_space_of(layout) for layout in self.layouts]
        codes: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        total_loss: torch.Tensor | float = 0.0
        unused_total = 0
        for level, layer in enumerate(self.levels):
            residual = _to_space(residual, spaces[level - 1] if level else spaces[0],
                                 spaces[level], self.curvature)
            residual, quant_loss, unused, index, picked = layer(
                residual, infer_use_sk=infer_use_sk)
            codes.append(picked)
            tokens.append(index)
            total_loss = total_loss + quant_loss
            unused_total += int(unused)
        readout = self.readout(codes, spaces)
        stacked = torch.stack(tokens, dim=1)
        return readout, total_loss / len(self.levels), unused_total, stacked

    def readout(self, codes: list[torch.Tensor], spaces: list[str]) -> torch.Tensor:
        """``phi(q) = q_E(L1) + log_0(nested H code points)`` in the common tangent."""
        euclidean = sum(code for code, space in zip(codes, spaces) if space == "E")
        hyperbolic = [code for code, space in zip(codes, spaces) if space == "H"]
        if not hyperbolic:
            return euclidean
        points = [expmap0(code, self.curvature) for code in hyperbolic]
        q_h = points[-1]
        for point in reversed(points[:-1]):
            q_h = mobius_add(point, q_h, self.curvature)
        return euclidean + logmap0(q_h, self.curvature)

    @torch.no_grad()
    def init_codebook(self, z: torch.Tensor, device: torch.device) -> None:
        residual = z
        spaces = [_space_of(layout) for layout in self.layouts]
        for level, layer in enumerate(self.levels):
            residual = _to_space(residual, spaces[level - 1] if level else spaces[0],
                                 spaces[level], self.curvature)
            layer.init_codebook(residual, device)
            residual, _, _, _, _ = layer(residual)


def encode_tokens(z: torch.Tensor, rq: CapacityRQ) -> torch.Tensor:
    """The deterministic nearest code at every level, without gradients."""
    with torch.no_grad():
        residual = z
        spaces = [_space_of(layout) for layout in rq.layouts]
        tokens: list[torch.Tensor] = []
        for level, layer in enumerate(rq.levels):
            residual = _to_space(residual, spaces[level - 1] if level else spaces[0],
                                 spaces[level], rq.curvature)
            cost = layer.distances(residual)
            index = torch.argmin(cost, dim=-1)
            picked = layer.get_code_embs()[index]
            point = expmap0(picked, rq.curvature) if spaces[level] == "H" else picked
            if spaces[level] == "E":
                residual = residual - point
            else:
                residual = project(mobius_add(-point, residual, rq.curvature),
                                   rq.curvature)
            tokens.append(index)
        return torch.stack(tokens, dim=1)
