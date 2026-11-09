"""The differentiable surface shared by this tree's structure mechanism.

``soft_readout`` is the exported readout extended to soft assignments,
``structural_delta`` is the frozen structural metric (a cosine dissimilarity on
the readout, the same one the report measures), ``mixed_delta`` is the quantizer's
own cost distance, and the two penalty primitives are the objective's atoms.
"""
from __future__ import annotations

import torch

from .mixed_rq import split


def soft_readout(assignments, codebooks, curvature: float = 1.0) -> torch.Tensor:
    """``phi`` for soft assignments: the hard readout's differentiable extension."""
    from .mixed_rq import EUCLIDEAN_DIM
    expected = [assignment @ codebook for assignment, codebook in zip(assignments, codebooks)]
    tangent = torch.stack([part[..., EUCLIDEAN_DIM:] for part in expected], dim=0)
    euclidean = torch.stack([part[..., :EUCLIDEAN_DIM] for part in expected], dim=0).sum(0)
    composed = tangent[-1]
    for part in reversed(tangent[:-1]):
        composed = _compose(part, composed, curvature)
    # The exported readout projects and takes log_0 ONCE, after the whole chain;
    # doing it per step would change the value.
    if float(curvature) > 0:
        from .poincare_contract import logmap0, project
        composed = logmap0(project(composed, curvature), curvature)
    return torch.cat([euclidean, composed], dim=-1)



def _compose(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    """The right-nested composition of the EXPORTED readout, verbatim.

    ``mixed_rq.quantized_readout`` composes the three tangent halves with
    ``mobius_add`` directly -- it does not map them into the ball first -- and then
    projects and takes ``log_0``. The soft readout must follow the same convention
    or a one-hot assignment would not reproduce the exported value.
    """
    from .poincare_contract import mobius_add
    if float(curvature) <= 0:
        return left + right
    return mobius_add(left, right, curvature)



def structural_delta(phi_left: torch.Tensor, phi_right: torch.Tensor) -> torch.Tensor:
    """The frozen structural metric of the readout: ``1 - cos(phi_i, phi_j)``.

    Issue #143 requires the same convention as the existing ``D_AF3`` and
    neighbour retention, and that convention is ``af3_structure.readout_delta`` --
    a cosine dissimilarity on the readout, not the quantizer's own mixed cost.
    Using anything else here would make the training objective and the reported
    metric two different quantities.
    """
    left = phi_left.to(torch.float64)
    right = phi_right.to(torch.float64)
    denominator = (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8)
    return 1.0 - (left * right).sum(-1) / denominator



def mixed_delta(phi_left: torch.Tensor, phi_right: torch.Tensor,
                curvature: float = 1.0) -> torch.Tensor:
    """``delta`` between two readouts: the frozen mixed distance on a pair.

    The Euclidean halves contribute their squared norm and the tangent halves are
    mapped into the ball before the canonical distance, which is what the hard
    readout's ``D_AF3`` measures.
    """
    from .poincare_contract import expmap0, squared_distance
    left_e, left_t = split(phi_left)
    right_e, right_t = split(phi_right)
    euclidean = (left_e - right_e).pow(2).sum(-1)
    if float(curvature) > 0:
        curved = squared_distance(expmap0(left_t, curvature), expmap0(right_t, curvature),
                                  curvature) / 4.0
    else:
        curved = left_t.sub(right_t).pow(2).sum(-1) / 4.0
    return euclidean + curved



def point_probability(delta: torch.Tensor, intercept: float, tau: float) -> torch.Tensor:
    """``p = sigmoid(b - delta / tau)`` written as the reference contract states it.

    ``1 / (1 + exp(delta/tau - b))`` is algebraically the same function and
    numerically better behaved on the large-shift side.
    """
    return 1.0 / (1.0 + torch.exp(delta / float(tau) - float(intercept)))



def huber_penalty(residual: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber, as the quadratic minus its excised tail, so it stays differentiable.

    ``0.5 r^2 - max(0, |r| - delta)^2 / (2 delta)`` equals ``delta |r| - delta^2/2``
    for ``|r| > delta`` and ``0.5 r^2`` otherwise, and unlike a Python branch the
    form has a gradient everywhere in ``r``.
    """
    magnitude = residual.abs()
    tail = torch.clamp(magnitude - float(delta), min=0.0)
    return 0.5 * residual.pow(2) - tail.pow(2) / (2.0 * float(delta))


