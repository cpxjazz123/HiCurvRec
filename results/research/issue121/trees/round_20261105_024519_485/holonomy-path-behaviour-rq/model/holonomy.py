"""Behaviour-chosen two-leg parallel transport (curvature holonomy) in the RQ recursion.

The mechanism moves the quantizer from the Euclidean ball into the Lorentz
model of the same curvature, and chooses, for every level, a codebook that has
been transported through TWO paths: the origin to the behaviour anchor, and the
anchor to the point the walk currently stands on. The difference between the
two paths is the holonomy of the curvature, and the assignment is made in the
frame the behaviour chose.

Why the two sides are one computation and not two:

* the behaviour side is the ANCHOR: a per-prefix vector built from the
  training-only transition mass, frozen between refreshes, detached from
  autograd. It enters through ``m = exp_o(v_prefix)``;
* the curve side is the TRANSPORT: the codeword is carried through that anchor
  and into the current frame, and the residual is carried with it;
* the assignment is the arg-min of the MINKOWSKI squared distance to the
  TRANSPORTED codewords - so removing the behaviour removes the second path and
  changes the arg-min, and removing the curvature makes the transports the
  identity, which changes it as well. Neither switch can be removed without
  changing the tokens.

The Lorentz operators are the framework's (``model.poincare_contract``): the
ball and the hyperboloid are two charts of one curvature, and the framework
owns both.

The recursion below is the one two INDEPENDENT derivations agreed on (the
framework's differential verifier: the same values to 5.6e-17, the same
gradients to 2.2e-16, gradcheck true on both sides). That agreed instance is
level one without an anchor (there is no prefix yet), the walk moving to the
chosen codeword, the residual carried into that frame, and level two assigning
against the codewords transported through the anchor of the prefix that level
one chose. The reference contract states exactly that instance, so the code
and the agreed derivation are compared by the framework rather than by
assertion.
"""
from __future__ import annotations

import torch

from .poincare_contract import (
    lorentz_exp_origin,
    lorentz_lift,
    lorentz_parallel_transport,
    minkowski_dot,
)

#: The two switches of the mechanism. They exist in the code and behave; the
#: experiment policy trains the both-on arm only.
USE_BEHAVIOUR = True
USE_CURVATURE = True


def lift(vector: torch.Tensor) -> torch.Tensor:
    """A spatial vector as a tangent vector at the origin: ``(0, vector)``."""
    return lorentz_lift(vector)


def origin_like(reference: torch.Tensor) -> torch.Tensor:
    """The origin of the hyperboloid, shaped like ``reference``."""
    origin = torch.zeros_like(reference)
    origin[..., 0] = 1.0
    return origin


def transport_codewords(codebook: torch.Tensor, anchor: torch.Tensor,
                        point: torch.Tensor) -> torch.Tensor:
    """``W_k = PT_{m -> a} (PT_{o -> m} (theta_hat_k))`` for every codeword.

    ``codebook`` is ``(K, D)`` (spatial), ``anchor`` is ``(B, D)`` (spatial, one
    per row) and ``point`` is ``(B, D + 1)`` (a Lorentz point per row): the
    frame each walk stands on. The result is ``(B, K, D + 1)``: every row gets
    the codebook in its own frame. The two legs are the behaviour leg (origin
    to the anchor) and the geometric leg (the anchor to the current frame).
    """
    lifted = lift(codebook)
    if not USE_CURVATURE:
        return lifted.unsqueeze(0).expand(point.shape[0], -1, -1)
    middle = lorentz_exp_origin(anchor)
    rows = middle.shape[0]
    first = _transport(origin_like(middle).unsqueeze(1),
                       middle.unsqueeze(1),
                       lifted.unsqueeze(0).expand(rows, -1, -1))
    return _transport(middle.unsqueeze(1), point.unsqueeze(1), first)


def _transport(source: torch.Tensor, target: torch.Tensor,
               vector: torch.Tensor) -> torch.Tensor:
    """``PT_{source -> target}(vector)`` with the three broadcast together.

    ``source`` and ``target`` are ``(B, 1, D + 1)`` and ``vector`` is
    ``(B, K, D + 1)``: one frame per row, every codeword moved in it.
    """
    inner = minkowski_dot(source, target)
    factor = minkowski_dot(target, vector) / (1.0 - inner)
    return vector + factor.unsqueeze(-1) * (source + target)


def transported_distance(codeword: torch.Tensor, anchor: torch.Tensor,
                         frame: torch.Tensor) -> torch.Tensor:
    """The Minkowski squared distance between the TRANSPORTED codeword and the frame.

    The rows are independent: every row carries its own codeword, its own
    anchor and its own frame, and the result is one number per row. This is
    the continuous surface of the mechanism: the two legs of the transport, the
    anchor and the frame, in one number. The discrete choice (which codeword)
    is not part of it and is excluded from the contract.
    """
    point = lorentz_exp_origin(frame)
    lifted = lift(codeword)
    middle = lorentz_exp_origin(anchor)
    origin = origin_like(middle).unsqueeze(1)
    first = _transport(origin, middle.unsqueeze(1), lifted.unsqueeze(1))
    moved = _transport(middle.unsqueeze(1), point.unsqueeze(1), first).squeeze(1)
    return squared_separation(point, moved)


def minkowski_square(residual: torch.Tensor, codeword: torch.Tensor) -> torch.Tensor:
    """``<r - w, r - w>_L``, broadcast over the rows."""
    difference = residual - codeword
    return minkowski_dot(difference, difference)


def squared_separation(residual: torch.Tensor, codeword: torch.Tensor) -> torch.Tensor:
    """``-<r, w>_L``: the negative Minkowski product of the residual with the
    transported codeword.

    For two points of the hyperboloid this is ``cosh`` of their hyperbolic
    distance, so its arg-min is the NEAREST codeword. The squared difference
    ``<r - w, r - w>_L`` does NOT work here: the residual is not a point of the
    hyperboloid, so its sign is not fixed, and the first run of this candidate
    took the same codeword for every item (the SID collapsed).
    """
    return -minkowski_dot(residual, codeword)


def assign(residual: torch.Tensor, codewords: torch.Tensor) -> tuple:
    """The nearest transported codeword: the index and its Minkowski distance.

    ``residual`` is ``(B, D + 1)`` and ``codewords`` is ``(B, K, D + 1)`` (or
    ``(K, D + 1)``, the same frame for every row).
    """
    scores = squared_separation(residual.unsqueeze(1), codewords)
    choice = scores.argmin(dim=-1)
    return choice, scores.gather(1, choice.unsqueeze(1)).squeeze(1)


def transport_residual(residual: torch.Tensor, source: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    """``PT_{source -> target}(e)``; the identity when the curve is switched off."""
    if not USE_CURVATURE:
        return residual
    return lorentz_parallel_transport(source, target, residual)


def two_level_loss(z: torch.Tensor,
                   theta_1_1: torch.Tensor, theta_1_2: torch.Tensor,
                   theta_2_1: torch.Tensor, theta_2_2: torch.Tensor,
                   s: float,
                   M_1: torch.Tensor, M_2: torch.Tensor) -> torch.Tensor:
    """The verified two-level instance of the mechanism.

    Every row is independent and carries its own codeword per level: ``z`` is
    ``(B, D)``, the four codewords are ``(B, D)`` and the two anchors are
    ``(B, D)``. Level one is the plain assignment (there is no prefix yet, so
    there is no anchor). The walk moves to the chosen codeword, the residual is
    carried into that frame, and level two assigns against the codewords
    transported through the anchor of the prefix that level one chose. The
    returned value is the sum of the two selected Minkowski distances: the
    quantity this mechanism adds to the objective.

    This function is the one the reference contract compares against the
    framework's own rebuild of the same SymPy statements, and the one the
    coupling surface is measured on.
    """
    residual = lift(z / s)
    first = lift(torch.stack([theta_1_1, theta_1_2], dim=1))
    first_choice, first_distance = assign(residual, first)
    code = first.gather(1, first_choice.unsqueeze(1).unsqueeze(-1).expand(
        -1, -1, first.shape[-1])).squeeze(1)
    point = lorentz_exp_origin(code[..., 1:])
    residual = transport_residual(residual - code, origin_like(residual), point)

    anchor = torch.where(first_choice.unsqueeze(-1) == 0, M_1, M_2)
    second = lift(torch.stack([theta_2_1, theta_2_2], dim=1))
    middle = lorentz_exp_origin(anchor)
    first_leg = _transport(origin_like(middle).unsqueeze(1),
                           middle.unsqueeze(1), second)
    moved = _transport(middle.unsqueeze(1), point.unsqueeze(1), first_leg)
    second_choice, second_distance = assign(residual, moved)
    return first_distance + second_distance


def holonomy_forward(z: torch.Tensor,
                     codebooks: list,
                     anchors: list,
                     scale: float) -> tuple:
    """The recursion of the mechanism over every level.

    ``codebooks`` is one ``(K, D)`` tensor per level, ``anchors`` one
    ``(K, D)`` tensor per level holding the anchor of every possible prefix
    code of that level (an empty tensor when the behaviour is switched off).
    Returns the tokens, the per-level distances and the final walk point.
    """
    residual = lift(z / scale)
    point = origin_like(residual)
    tokens, distances, walk = [], [], []
    for level, codebook in enumerate(codebooks):
        anchor_table = anchors[level] if level < len(anchors) else None
        anchor = torch.zeros_like(codebook[0]).unsqueeze(0).expand(
            residual.shape[0], -1)
        if anchor_table is not None and anchor_table.numel() and USE_BEHAVIOUR \
                and tokens:
            # The anchor of the prefix the walk stands in: the code chosen at
            # the previous level.
            anchor = anchor_table[tokens[-1]]
        moved = transport_codewords(codebook, anchor, point)
        choice, distance = assign(residual, moved)
        code = moved.gather(1, choice.unsqueeze(1).unsqueeze(-1).expand(
            -1, -1, moved.shape[-1])).squeeze(1)
        if level == 0:
            # The first level's codewords are spatial: the walk moves to the
            # codeword lifted onto the hyperboloid.
            new_point = lorentz_exp_origin(code[..., 1:])
        else:
            new_point = code
        residual = transport_residual(residual - code, point, new_point)
        point = new_point
        tokens.append(choice)
        distances.append(distance)
        walk.append(point)
    return torch.stack(tokens, dim=-1), torch.stack(distances, dim=-1), point
