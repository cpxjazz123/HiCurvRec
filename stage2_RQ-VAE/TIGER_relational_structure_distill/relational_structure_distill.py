"""Hyperbolic Relational Structure Distillation for Item Encodings (Issue #23).

The behaviour teacher is built once from the existing user-item interactions: an
item-item co-occurrence count turned into a cosine-normalised collaborative
similarity. Nothing here regresses onto CF embedding coordinates -- the teacher
only ever supplies *relations*: which neighbour is strong, which is weak, which
is negative, and the three-point shape of each triangle.

What the mechanism puts on the encoder side, all on the continuous side of the
unchanged three-level quantizer:

1. ``z_i = Exp_o(f_theta(x_i))`` -- the encoder's latent read as a ball point,
   through the tree's own ``encode_ball`` convention so the raw MLP latent cannot
   saturate tanh against the boundary. The residual/assignment/codebook code is
   not touched.
2. A geodesic ranking hinge ``max(0, m + d_H(z_i, z_j) - d_H(z_i, z_k))`` over the
   (anchor, strong, negative) triple. It never touches a codeword, so the encoder
   is the only thing the behaviour gradient can reach.
3. A three-point structure term over the (anchor, strong, weak) triple. The
   teacher's similarity becomes a normalised distance ``delta = 1 - similarity``;
   the law of cosines on the teacher's triangle gives a target angle and the
   ratio of its two sides a target distance ratio. In the ball the term matches
   the model's geodesic ratio and the angle at the anchor -- which is the angle
   between ``Log_{z_i}(z_j)`` and ``Log_{z_i}(z_k)``, read off the side lengths so
   no logarithm map has to subtract nearly equal ball points.
4. The decoder side reads the quantizer's continuous output through a
   straight-through proxy that refuses gradient to any codebook and replays the
   same two objectives, so the reconstruction path is asked to carry the same
   relations while no codebook can be moved by behaviour supervision.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import encode_ball

CURVATURE = 1.0
BALL_EPS = 1e-5
RANK_MARGIN = 0.5
ANGLE_WEIGHT = 0.5
NEIGHBOURS_PER_ITEM = 16
STRONG_NEIGHBOURS = 8
TEACHER_FLOOR = 1e-6
TANGENT_SCALE = 0.5        # fixed working radius for z_i, per hyperbolic.encode_ball
MIN_TRIANGLE_SIDE = 1e-2   # below this a triple's shape carries no gradient safely
RECOVERY_RADIUS_CAP = 0.9


# --------------------------------------------------------------------------- #
# Poincare ball geometry
# --------------------------------------------------------------------------- #

def _expmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Origin exponential map with a unit-speed tangent: ``||v||`` is the arc length.

    ``d(0, Exp_0(v)) = 2 * atanh(sqrt(c) * ||v|| / 2) / sqrt(c)``, so the tangent
    norm is the travelled distance. The mechanism check asserts the isometry
    rather than trusting the formula.
    """
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm) * x


def _to_hyperboloid(point: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Ball point -> hyperboloid point, curvature -c, in double precision."""
    if abs(c - 1.0) > 1e-12:
        raise ValueError(f"the hyperboloid bridge here is derived for c = 1, got {c}")
    squared = point.double().square().sum(-1, keepdim=True)
    scale = (1.0 - squared).clamp_min(BALL_EPS)
    return torch.cat([1.0 + squared, 2.0 * point.double()], dim=-1) / scale


def _from_hyperboloid(lifted: torch.Tensor) -> torch.Tensor:
    """Hyperboloid point -> ball point."""
    return lifted[..., 1:] / (lifted[..., :1] + 1.0)


def _lorentz_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product; negative on the hyperboloid."""
    return -left[..., :1] * right[..., :1] + (
        left[..., 1:] * right[..., 1:]
    ).sum(-1, keepdim=True)


def _expmap_x(x: torch.Tensor, v: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Exponential map at a base point: a geodesic of length ``||v||`` along ``v``.

    Taken in the hyperboloid model, where the map is exactly
    ``cosh(L) * p + sinh(L) * u`` for a unit Lorentz-tangent ``u``. The tangent's
    time component follows from the tangency condition ``<p, v> = 0``, and ``L`` is
    the *ambient* norm of ``v``; normalising the tangent to unit Lorentz length
    first is what makes the travelled distance equal ``||v||``. Verified to 1e-6
    (float32 output) from the origin out to anchor radius 0.95 by the check.
    """
    base = _to_hyperboloid(x, c)
    time = (base[..., 1:] * v.double()).sum(-1, keepdim=True) / base[..., :1]
    tangent = torch.cat([time, v.double()], dim=-1)
    unit = tangent / _lorentz_inner(tangent, tangent).clamp_min(1e-24).sqrt()
    length = torch.linalg.vector_norm(v.double(), dim=-1, keepdim=True)
    return _from_hyperboloid(torch.cosh(length) * base + torch.sinh(length) * unit).to(x.dtype)


def _geodesic(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Poincare geodesic distance, broadcast over the leading dimensions.

    ``acosh`` takes ``1 + O(squared distance)``, a cancellation float32 cannot
    resolve below ~1e-7, so the reduction runs in double precision and casts back.
    """
    sqrt_c = c ** 0.5
    limit = (1.0 - BALL_EPS) / c
    norm_x = torch.linalg.vector_norm(x, dim=-1).clamp_max(limit ** 0.5)
    norm_y = torch.linalg.vector_norm(y, dim=-1).clamp_max(limit ** 0.5)
    squared = (x - y).square().sum(dim=-1, dtype=torch.float64)
    conform = ((1.0 - c * norm_x.double().square())
               * (1.0 - c * norm_y.double().square())).clamp_min(1e-7)
    argument = 1.0 + 2.0 * c * squared / conform
    # 1 + 1e-12 keeps d(acosh)/d(argument) finite for coincident points while
    # sitting far under the float32 resolution of the ball's geometry.
    return torch.acosh(argument.clamp_min(1.0 + 1e-12)).to(x.dtype) / sqrt_c


def _degenerate(*sides: torch.Tensor) -> torch.Tensor:
    """True where a side is too short for the triangle's shape to be defined.

    Both the law of cosines and the distance ratio are ratios of quantities that
    vanish together on a degenerate triangle, so their gradient is unbounded
    there. Such a triple carries no shape information anyway -- the anchor, its
    strong and its weak neighbour are the same point -- so it is masked out
    rather than clamped into a finite but meaningless value.
    """
    mask = torch.zeros_like(sides[0], dtype=torch.bool)
    for side in sides:
        mask = mask | (side <= MIN_TRIANGLE_SIDE)
    return mask


def _hyperbolic_angle(first: torch.Tensor, second: torch.Tensor, opposite: torch.Tensor,
                      c: float = CURVATURE) -> torch.Tensor:
    """Angle at the anchor between the geodesics to ``second`` and ``opposite``.

    This is the hyperbolic law of cosines on the triangle's three side lengths,
    which is the same angle as the one between ``Log_x(second)`` and
    ``Log_x(opposite)``. Degenerate triangles return a neutral right angle with
    no gradient.
    """
    near = _geodesic(first, second, c)
    far = _geodesic(first, opposite, c)
    across = _geodesic(second, opposite, c)
    degenerate = _degenerate(near, far, across)
    a, b, d = near.double(), far.double(), across.double()
    denominator = torch.sinh(a) * torch.sinh(b)
    cosine = (torch.cosh(a) * torch.cosh(b) - torch.cosh(d)) / denominator.clamp_min(1e-12)
    angle = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    neutral = torch.full_like(angle, math.pi / 2.0)
    return torch.where(degenerate, neutral, angle).to(first.dtype)


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


# --------------------------------------------------------------------------- #
# The behaviour teacher
# --------------------------------------------------------------------------- #

def build_relational_teacher(histories: list[list[int]], n_items: int) -> dict[str, np.ndarray]:
    """Co-occurrence teacher: strong neighbours, weak neighbours, triangle shape.

    ``histories`` is one item list per user. The collaborative similarity is the
    cosine-normalised co-occurrence count, which is the standard item-CF strength
    and needs no embedding coordinates. Only items with enough neighbours get a
    teacher; the count is reported so the caller can see the coverage.
    """
    counts: dict[tuple[int, int], int] = {}
    for history in histories:
        items = np.unique(np.asarray(history, dtype=np.int64))
        if items.size < 2:
            continue
        for a in range(items.size):
            left = items[a]
            for b in range(a + 1, items.size):
                right = items[b]
                counts[(int(left), int(right))] = counts.get((int(left), int(right)), 0) + 1
    popularity = np.zeros(n_items, dtype=np.float64)
    for (left, right), value in counts.items():
        popularity[left] += value
        popularity[right] += value
    neighbours: dict[int, dict[int, float]] = {}
    for (left, right), value in counts.items():
        denominator = np.sqrt(popularity[left] * popularity[right]) + 1.0
        similarity = value / denominator
        neighbours.setdefault(left, {})[right] = similarity
        neighbours.setdefault(right, {})[left] = similarity
    anchors: list[int] = []
    strong: list[int] = []
    weak: list[int] = []
    for anchor, table in neighbours.items():
        ranked = sorted(table.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < STRONG_NEIGHBOURS + 1:
            continue
        band = ranked[:NEIGHBOURS_PER_ITEM]
        for rank in range(STRONG_NEIGHBOURS):
            anchors.append(anchor)
            strong.append(band[rank][0])
            weak.append(
                band[STRONG_NEIGHBOURS + rank][0]
                if rank < len(band) - STRONG_NEIGHBOURS else band[-1][0]
            )
    anchors_array = np.asarray(anchors, dtype=np.int64)
    strong_array = np.asarray(strong, dtype=np.int64)
    weak_array = np.asarray(weak, dtype=np.int64)
    if anchors_array.size == 0:
        raise RuntimeError("behaviour teacher produced no triple")
    delta_strong = 1.0 - np.asarray(
        [neighbours[int(a)][int(b)] for a, b in zip(anchors_array, strong_array)],
        dtype=np.float64,
    )
    delta_weak = 1.0 - np.asarray(
        [neighbours[int(a)][int(b)] for a, b in zip(anchors_array, weak_array)],
        dtype=np.float64,
    )
    delta_pair = np.ones_like(delta_weak)
    for index, (anchor, best, worst) in enumerate(
            zip(anchors_array, strong_array, weak_array)):
        delta_pair[index] = 1.0 - float(neighbours.get(int(best), {}).get(int(worst), 0.0))
    ratio_target = np.clip(
        delta_weak / np.maximum(delta_strong, TEACHER_FLOOR), 1.0, 4.0,
    ).astype(np.float32)
    cosine = (delta_strong ** 2 + delta_weak ** 2 - delta_pair ** 2) / np.maximum(
        2.0 * delta_strong * delta_weak, TEACHER_FLOOR,
    )
    angle_target = np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
    return {
        "anchor": anchors_array,
        "strong": strong_array,
        "weak": weak_array,
        "ratio_target": ratio_target,
        "angle_target": angle_target,
        "delta_strong": delta_strong.astype(np.float32),
        "delta_weak": delta_weak.astype(np.float32),
        "item_space": np.int64(n_items),
    }


def teacher_triples(teacher: dict[str, np.ndarray], batch: int, generator: torch.Generator,
                    device: torch.device) -> dict[str, torch.Tensor]:
    """Draw a batch of (anchor, strong, weak) triples plus their negative partners."""
    available = int(teacher["anchor"].shape[0])
    picks = torch.randint(available, (batch,), generator=generator)
    negative = torch.randint(int(teacher["item_space"]), (batch,), generator=generator)
    return {
        "anchor": torch.as_tensor(teacher["anchor"][picks.numpy()], device=device),
        "strong": torch.as_tensor(teacher["strong"][picks.numpy()], device=device),
        "weak": torch.as_tensor(teacher["weak"][picks.numpy()], device=device),
        "negative": negative.to(device),
        "ratio_target": torch.as_tensor(teacher["ratio_target"][picks.numpy()], device=device),
        "angle_target": torch.as_tensor(teacher["angle_target"][picks.numpy()], device=device),
    }


# --------------------------------------------------------------------------- #
# Wiring and objectives
# --------------------------------------------------------------------------- #

def install_relational_encoder(model: nn.Module,
                               curvature: float = CURVATURE) -> None:
    """Read the encoder's latent as a ball point, leaving the model alone.

    The module is not replaced and nothing is attached to it, so
    ``model.state_dict()`` keeps exactly the keys the checkpoint probe expects;
    only the encoder's forward now ends in ``Exp_o``. The tree's own
    ``encode_ball`` convention is used because a raw 32-d MLP latent saturates
    tanh and pins every point to the ball boundary.
    """
    encoder = model.encoder
    original_forward = encoder.forward

    def patched_forward(embeddings: torch.Tensor) -> torch.Tensor:
        return encode_ball(original_forward(embeddings), curvature, TANGENT_SCALE)

    encoder.forward = patched_forward


class RelationalRecoveryHead(nn.Module):
    """Decoder-side head replaying the relations on the reconstructed item.

    The head's tangent is capped before the exponential map: without it a head that
    overshoots parks its output on the ball boundary, where every relational
    distance is huge and the gradient of the ranking hinge explodes. The cap sits
    well above the radii real encoders reach, and when it binds that item simply
    stops supplying a replay gradient.
    """

    def __init__(self, dim: int, curvature: float = CURVATURE,
                 radius_cap: float = RECOVERY_RADIUS_CAP) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.radius_cap = float(radius_cap)
        self.head = nn.Linear(dim, dim)

    def forward(self, quantized: torch.Tensor,
                codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        continuous = _ste_without_codebook_gradient(quantized, codebooks)
        tangent = self.head(continuous)
        norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-12)
        cap = torch.as_tensor(self.radius_cap, dtype=continuous.dtype,
                              device=continuous.device)
        allowed = 2.0 * torch.atanh(cap) / self.curvature ** 0.5
        capped = torch.where(norm > allowed, allowed * tangent / norm, tangent)
        return _expmap0(capped, self.curvature)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def relational_ranking_loss(anchor: torch.Tensor, strong: torch.Tensor,
                            negative: torch.Tensor, margin: float = RANK_MARGIN,
                            curvature: float = CURVATURE) -> torch.Tensor:
    """``max(0, m + d_H(z_i, z_j) - d_H(z_i, z_k))`` over behaviour triples."""
    return F.relu(
        margin + _geodesic(anchor, strong, curvature)
        - _geodesic(anchor, negative, curvature)
    ).mean()


def relational_structure_loss(anchor: torch.Tensor, strong: torch.Tensor,
                              weak: torch.Tensor, ratio_target: torch.Tensor,
                              angle_target: torch.Tensor,
                              angle_weight: float = ANGLE_WEIGHT,
                              curvature: float = CURVATURE) -> torch.Tensor:
    """Match the teacher's three-point shape with geodesic ratios and a tangent angle."""
    near = _geodesic(anchor, strong, curvature)
    far = _geodesic(anchor, weak, curvature)
    across = _geodesic(strong, weak, curvature)
    degenerate = _degenerate(near, far, across)
    ratio_error = (far / near.clamp_min(MIN_TRIANGLE_SIDE) - ratio_target).square()
    angle_error = (_hyperbolic_angle(anchor, strong, weak, curvature) - angle_target).square()
    keep = ~degenerate
    if not bool(keep.any()):
        return anchor.sum() * 0.0
    return ratio_error[keep].mean() + angle_weight * angle_error[keep].mean()


def measure_relational_structure(anchor: torch.Tensor, strong: torch.Tensor,
                                 weak: torch.Tensor,
                                 curvature: float = CURVATURE) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's own distance ratio and anchor angle, for verification."""
    near = _geodesic(anchor, strong, curvature)
    far = _geodesic(anchor, weak, curvature)
    return far / near.clamp_min(MIN_TRIANGLE_SIDE), _hyperbolic_angle(
        anchor, strong, weak, curvature,
    )
