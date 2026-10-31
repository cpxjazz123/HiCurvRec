"""The three auxiliary losses of the SID-alignment study, and the geometry they share.

One file, one switch. ``AUX_MODE`` names the single loss a tree adds to the frozen
baseline objective; the other two stay implemented (and unit-tested) but inactive,
so the comparison is exactly "baseline + one term".

Everything here is constant-free by construction:

* the soft assignment temperature is the median distance of the batch itself,
  detached, so it follows the geometry instead of being tuned;
* the ball scale is ``1 / max ||latent||`` of the batch, detached;
* every loss is standardized over its pair set before it is compared, so its
  magnitude is O(1) and one unit of weight is the natural scale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: Which single auxiliary term this tree adds: "none", "gp", "tree", "cluster".
AUX_MODE = "gp"

#: The weight of the term. The losses are standardized, so 1.0 is the natural
#: scale: any other value would be a tuned constant.
AUX_WEIGHT = 1.0

#: How many rows of a batch take part in a pair loss. A compute bound, not a
#: tuned parameter: 128 x 128 pairs carry the batch's rank structure and keep the
#: cost of the auxiliary term under a few milliseconds.
PAIR_ROWS = 128

#: The unit ball of the contract, with the usual epsilon away from the boundary.
BALL_EPS = 1e-6

#: The diameter of the unit ball, used to put a distance in [0, 1].
BALL_DIAMETER = 2.0 * torch.atanh(torch.tensor(1.0 - BALL_EPS)).item()


def project(x: torch.Tensor) -> torch.Tensor:
    """Map a latent into the unit ball with a data-derived scale."""
    norm = x.norm(dim=-1, keepdim=True)
    scale = norm.detach().max().clamp_min(1e-8)
    return torch.tanh(norm / scale) * x / norm.clamp_min(1e-8)


def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The Poincare distance between broadcastable point tensors."""
    difference = (a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2).sum(-1)
    a_sq = a.pow(2).sum(-1).unsqueeze(-1)
    b_sq = b.pow(2).sum(-1).unsqueeze(-2)
    argument = 1.0 + 2.0 * difference / (
        (1.0 - a_sq).clamp_min(BALL_EPS) * (1.0 - b_sq).clamp_min(BALL_EPS))
    return torch.acosh(argument.clamp_min(1.0 + BALL_EPS))


def soft_assignment(distances: torch.Tensor) -> torch.Tensor:
    """``p = softmax(-d / median(d))``: a temperature from the batch itself."""
    scale = distances.detach().median().clamp_min(1e-8)
    return torch.softmax(-distances / scale, dim=-1)


def _standardize(x: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit variance over the pair set, with detached statistics."""
    mean = x.detach().mean()
    std = x.detach().std().clamp_min(1e-8)
    return (x - mean) / std


def _pair_mask(rows: int, device) -> torch.Tensor:
    mask = torch.ones(rows, rows, dtype=torch.bool, device=device)
    return ~torch.eye(rows, dtype=torch.bool, device=device)


def _rows(probs: list[torch.Tensor], latents: list[torch.Tensor]):
    rows = min(PAIR_ROWS, probs[0].shape[0])
    return [p[:rows] for p in probs], [x[:rows] for x in latents], rows


def gp_sid_alignment(probs: list[torch.Tensor], code_points: list[torch.Tensor]) -> torch.Tensor:
    """Align the shared-prefix depth with the shared-ancestor depth.

    Shared prefix depth counts the levels whose soft assignment agrees:
    ``A_ij = sum_l prod_{l' <= l} <p_l'(i), p_l'(j)>``.

    Shared ancestor depth is the same construction in the ball, on the expected
    code point of each level ``c_l(i) = sum_k p_l(i,k) * phi(e_lk)``:
    ``H_ij = sum_l prod_{l' <= l} (1 - d(c_l'(i), c_l'(j)) / D)``.

    The loss is the standardized squared difference of the two over the pairs of
    the batch. Nothing here is tuned: the temperature and the scale come from the
    batch, and D is the diameter of the unit ball.
    """
    probs, _, rows = _rows(probs, [torch.zeros(1)])
    agreement = None
    curved = None
    running_a = torch.ones(rows, rows, device=probs[0].device)
    running_h = torch.ones(rows, rows, device=probs[0].device)
    for level, p in enumerate(probs):
        running_a = running_a * (p @ p.t())
        centres = p @ code_points[level]
        normalised = 1.0 - distance(centres, centres) / BALL_DIAMETER
        running_h = running_h * normalised.clamp(-1.0, 1.0)
        agreement = running_a if agreement is None else agreement + running_a
        curved = running_h if curved is None else curved + running_h
    mask = _pair_mask(rows, probs[0].device)
    return _standardize(agreement[mask]).sub(_standardize(curved[mask])).pow(2).mean()


def tree_distortion(probs: list[torch.Tensor], latents: list[torch.Tensor]) -> torch.Tensor:
    """Make the SID tree distance reproduce the frozen hyperbolic distances.

    Tree distance from the soft assignment: ``T_ij = sum_l (1 - <p_l(i), p_l(j)>)``.
    Reference distance: the Poincare distance between the pre-quantization
    latents, projected into the ball with the batch scale. The loss is the
    stress ``mean (z(T) - z(R))^2`` over the pairs.
    """
    probs, latents, rows = _rows(probs, latents)
    tree = None
    for p in probs:
        level = 1.0 - (p @ p.t())
        tree = level if tree is None else tree + level
    points = project(latents[0])
    reference = distance(points, points) / BALL_DIAMETER
    mask = _pair_mask(rows, probs[0].device)
    return _standardize(tree[mask]).sub(_standardize(reference[mask])).pow(2).mean()


def frechet_cluster(probs: list[torch.Tensor], latents: list[torch.Tensor]) -> torch.Tensor:
    """Tighten items of one prefix and separate the prefixes, in the ball.

    The prefix is the first-level code (argmax, detached for grouping only: the
    positions and the centres carry the gradient). A prefix centre is the
    weighted mean of its members' ball points with the soft first-level mass as
    the weight, which is the one-step Frechet mean the contract's ball admits
    without an inner loop. The loss is the mean intra-prefix distance minus the
    mean inter-prefix distance, divided by the ball diameter.
    """
    probs, latents, rows = _rows(probs, latents)
    points = project(latents[0])
    first = probs[0]
    labels = first.detach().argmax(dim=-1)
    centres = []
    weights = []
    for code in labels.unique():
        mask = labels == code
        if int(mask.sum()) < 2:
            continue
        mass = first[:, code].detach()
        centre = (points[mask] * mass[mask].unsqueeze(-1)).sum(0) / mass[mask].sum().clamp_min(1e-8)
        centres.append(centre)
        weights.append(mask)
    if len(centres) < 2:
        return points.sum() * 0.0
    centres = torch.stack(centres)
    intra = []
    for centre, mask in zip(centres, weights):
        intra.append((distance(points[mask], centre.unsqueeze(0)) / BALL_DIAMETER).mean())
    intra = torch.stack(intra).mean()
    centre_distance = distance(centres, centres) / BALL_DIAMETER
    center_pair = ~torch.eye(len(centres), dtype=torch.bool, device=centres.device)
    inter = centre_distance[center_pair].mean()
    return intra - inter


LOSSES = {"gp": gp_sid_alignment, "tree": tree_distortion, "cluster": frechet_cluster}


def compute(mode: str, probs: list[torch.Tensor],
            code_points: list[torch.Tensor], latents: list[torch.Tensor]) -> torch.Tensor:
    """The single active term of this tree, or a differentiable zero."""
    if mode == "none":
        return latents[0].sum() * 0.0
    if mode == "gp":
        return LOSSES["gp"](probs, code_points)
    if mode == "tree":
        return LOSSES["tree"](probs, latents)
    if mode == "cluster":
        return LOSSES["cluster"](probs, latents)
    raise ValueError("unknown auxiliary mode: " + str(mode))
