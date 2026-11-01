"""The three auxiliary losses of the SID-alignment study, with their exact definitions.

One file, one switch: ``AUX_MODE`` names the single loss a tree adds to the frozen
baseline objective. The other two stay implemented and unit-tested but inactive,
so the comparison is exactly "baseline + one term".

Definitions (the objects the terms are about, not proxies for them):

* **Tree distance.** The SID is a tree over codes; the distance between two items
  is the number of levels they do NOT share, i.e. ``L - LCP`` where LCP is the
  longest common prefix. Under the soft assignment the expected LCP is
  ``sum_l prod_{l' <= l} <p_l'(i), p_l'(j)>`` -- a product over levels, because a
  level counts only if every level above it also agrees. Matching a code at
  level 2 while differing at level 1 does NOT shorten the tree distance.

* **Gromov product.** For a metric space ``X`` and a base point ``o``,
  ``G_X(i,j) = (d_X(o,i) + d_X(o,j) - d_X(i,j)) / 2``. The alignment term asks the
  Gromov product of the SID tree and the Gromov product of the FROZEN reference
  geometry to agree.

* **Frechet cluster.** The centre of a prefix group is the hyperbolic Frechet
  mean -- the minimiser of the sum of squared hyperbolic distances -- computed by
  Riemannian iterations, NOT the weighted arithmetic mean, which is a different
  point in general.

The reference geometry is FROZEN: the latents of the frozen baseline encoder,
mapped into the unit ball once. A term that compares against a frozen structure
cannot be minimised by collapsing the tree, because the other side of the
comparison does not move.
"""
from __future__ import annotations

import torch

#: Which single auxiliary term this tree adds: "none", "gp", "tree", "cluster".
AUX_MODE = "cluster"

#: The weight of the term. The losses are standardized over their pair set, so
#: 1.0 is the unit scale; the term is additionally rescaled at every step by the
#: measured reconstruction gradient (see model.compute_loss).
AUX_WEIGHT = 1.0

#: How many rows of a batch take part in a pair term: a compute bound, not a
#: tuned parameter.
PAIR_ROWS = 128

#: The unit ball of the contract, with the usual epsilon away from the boundary.
BALL_EPS = 1e-6

#: The diameter of the unit ball, so a distance divided by it lives in [0, 1].
BALL_DIAMETER = 2.0 * torch.atanh(torch.tensor(1.0 - BALL_EPS)).item()


# --------------------------------------------------------------------------
# the geometry of the contract
# --------------------------------------------------------------------------

def project(x: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
    """Map a latent into the unit ball with a data-derived scale."""
    norm = x.norm(dim=-1, keepdim=True)
    if scale is None:
        scale = norm.detach().max().clamp_min(1e-8)
    return torch.tanh(norm / scale) * x / norm.clamp_min(1e-8)


def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The Poincare distance between broadcastable point tensors."""
    difference = (a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2).sum(-1)
    a_sq = a.pow(2).sum(-1).unsqueeze(-1)
    b_sq = b.pow(2).sum(-1).unsqueeze(-2)
    argument = 1.0 + (2.0 * difference / (
        (1.0 - a_sq).clamp_min(BALL_EPS) * (1.0 - b_sq).clamp_min(BALL_EPS))
    ).clamp_min(1e-3)
    return torch.acosh(argument.clamp_min(1.0 + BALL_EPS))


def _mobius_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xy = (x * y).sum(-1, keepdim=True)
    xx = (x * x).sum(-1, keepdim=True)
    yy = (y * y).sum(-1, keepdim=True)
    return ((1 + 2 * xy + yy) * x + (1 - xx) * y) / (1 + 2 * xy + xx * yy)


def _log_map(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    delta = _mobius_add(-base, point)
    norm = delta.norm(dim=-1, keepdim=True)
    # log_p(q) = 2 (1 - |p|^2) atanh(|p (-) q|) (p (-) q)/|p (-) q|
    factor = 2.0 * (1.0 - (base * base).sum(-1, keepdim=True)).clamp_min(BALL_EPS)
    return factor * torch.arctanh(norm.clamp(max=1 - BALL_EPS)) * delta / norm.clamp_min(1e-12)


def _exp_map(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / (1.0 - (base * base).sum(-1, keepdim=True)).clamp_min(BALL_EPS)
    norm = tangent.norm(dim=-1, keepdim=True)
    return _mobius_add(base, torch.tanh(norm * scale / 2.0) * tangent / norm.clamp_min(1e-12))


def frechet_mean(points: list, weights: torch.Tensor = None, steps: int = 8) -> torch.Tensor:
    """The hyperbolic Frechet mean: the minimiser of ``sum_i w_i d_H(x_i, m)^2``.

    Riemannian gradient descent in the ball, which is a different point from the
    weighted arithmetic mean in general; the tests assert that difference. The
    weights enter the GRADIENT, never the points: scaling a point by a weight
    would push it out of the ball.
    """
    mean = points[0]
    if weights is None:
        weights = torch.ones(len(points), device=points[0].device, dtype=points[0].dtype)
    weights = weights / weights.sum().clamp_min(1e-12)
    for _ in range(steps):
        gradient = (torch.stack([_log_map(mean, point) for point in points])
                    * weights.unsqueeze(-1)).sum(dim=0)
        mean = _exp_map(mean, gradient)
        radius = mean.norm(dim=-1, keepdim=True)
        mean = mean * (radius.clamp(max=1 - 1e-6) / radius.clamp_min(1e-12))
    return mean


# --------------------------------------------------------------------------
# the soft assignment and the tree it defines
# --------------------------------------------------------------------------

def soft_assignment(distances: torch.Tensor) -> torch.Tensor:
    """``p = softmax(-d / median(d))``: the temperature comes from the batch."""
    scale = distances.detach().median().clamp_min(1e-8)
    return torch.softmax(-distances / scale, dim=-1)


def lcp_depth(probs: list) -> torch.Tensor:
    """The expected LONGEST COMMON PREFIX depth of the soft SID.

    A level counts only if every level above it agrees, so the levels enter as a
    product and not as a sum.
    """
    running = None
    depth = None
    for level, p in enumerate(probs):
        agreement = p @ p.t()
        running = agreement if running is None else running * agreement
        depth = running if depth is None else depth + running
    return depth


def tree_distance(probs: list) -> torch.Tensor:
    """``T(i,j) = L - LCP(i,j)``: the SID tree distance in levels."""
    levels = len(probs)
    return levels - lcp_depth(probs)


def gromov_product(matrix: torch.Tensor, base: int) -> torch.Tensor:
    """``G(i,j) = (d(o,i) + d(o,j) - d(i,j)) / 2`` of a distance matrix."""
    to_base = matrix[base].unsqueeze(0)
    return (to_base + to_base.t() - matrix) / 2.0


def _standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.detach().mean()
    std = x.detach().std().clamp_min(1e-8)
    return (x - mean) / std


def _rows(probs: list, others: list):
    rows = min(PAIR_ROWS, probs[0].shape[0])
    return [p[:rows] for p in probs], [o[:rows] for o in others], rows


def _pair_mask(rows: int, device) -> torch.Tensor:
    return ~torch.eye(rows, dtype=torch.bool, device=device)


def _base_index(reference: torch.Tensor) -> int:
    """A deterministic base point: the item whose reference norm is the median."""
    norms = reference.detach().norm(dim=-1)
    return int(torch.argsort(norms)[len(norms) // 2].item())


# --------------------------------------------------------------------------
# the three terms
# --------------------------------------------------------------------------

def gp_alignment(probs: list, reference: torch.Tensor) -> torch.Tensor:
    """The Gromov products of the SID tree and of the reference geometry must agree.

    ``L_GP = mean_{i != j} ( z(G_tree(i,j)) - z(G_ref(i,j)) )^2``

    The reference side is frozen, so a collapsed tree cannot make the two sides
    equal: this term has no collapse optimum.
    """
    probs, references, rows = _rows(probs, [reference])
    points = references[0]
    tree = tree_distance(probs)
    curved = distance(points, points)
    base = _base_index(points)
    g_tree = gromov_product(tree, base)
    g_ref = gromov_product(curved, base)
    mask = _pair_mask(rows, probs[0].device)
    return _standardize(g_tree[mask]).sub(_standardize(g_ref[mask])).pow(2).mean()


def tree_distortion(probs: list, reference: torch.Tensor) -> torch.Tensor:
    """The SID tree distance must reproduce the frozen hyperbolic distances.

    ``L_tree = mean_{i != j} ( z(T(i,j)) - z(R(i,j)) )^2`` with ``T`` the LCP tree
    distance and ``R`` the reference distance.
    """
    probs, references, rows = _rows(probs, [reference])
    points = references[0]
    tree = tree_distance(probs)
    curved = distance(points, points)
    mask = _pair_mask(rows, probs[0].device)
    return _standardize(tree[mask]).sub(_standardize(curved[mask])).pow(2).mean()


def frechet_cluster(probs: list, code_points: list, latents: list) -> torch.Tensor:
    """Compact inside a prefix, separated across prefixes, in the ball.

    Two positions are compared for every item: its PRE-QUANTIZATION ball position
    and the expected code point of its prefix, so the term acts on the item
    representation and on the codebook at the same time (the second part is what
    gives the codebooks a gradient).

    A group's centre is the hyperbolic Frechet mean of its members. The grouping
    falls back down the hierarchy when a level puts the whole batch on one code.
    The normalized ratio is 1 at collapse (intra = inter = 0) and 0 when the
    prefixes are compact and mutually separated.
    """
    probs, latents, rows = _rows(probs, latents)
    # the codebooks are NOT sliced: they are (codes, dim) and the rows index them
    item_points = project(latents[0])
    # The grouping level is the one that leaves the most NON-TRIVIAL groups (at
    # least two members). A level that scatters the batch over hundreds of
    # singleton codes leaves nothing to compare, and a level that puts the whole
    # batch on one code has no contrast at all.
    labels = None
    masses = None
    level_index = 0
    best_groups = 0
    for index, level_probs in enumerate(probs):
        candidate = level_probs.detach().argmax(dim=-1)
        if candidate.unique().numel() < 2:
            continue
        counts = torch.bincount(candidate)
        groups = int((counts >= 2).sum())
        if groups > best_groups:
            best_groups = groups
            labels = candidate
            masses = level_probs
            level_index = index
    if labels is None:
        # nothing separates the batch: the term is at its worst, and it carries no
        # gradient because there is no group to move
        return item_points.sum() * 0.0 + 1.0
    # the EXPECTED code point of every item: the soft mass of the GROUPING level
    # over that level's codebook. Using the same level for the grouping and for
    # the code point is what keeps the gradient connected to the codebook.
    prefix_points = probs[level_index] @ code_points[level_index]
    item_centres = []
    code_centres = []
    members = []
    for code in labels.unique():
        mask = labels == code
        if int(mask.sum()) < 2:
            continue
        weight = masses[:, code][mask].detach().clamp_min(1e-8)
        item_group = item_points[mask]
        code_group = prefix_points[mask]
        item_centres.append(frechet_mean([item_group[i] for i in range(item_group.shape[0])],
                                          weight))
        code_centres.append(frechet_mean([code_group[i] for i in range(code_group.shape[0])],
                                          weight))
        members.append(mask)
    if len(item_centres) < 2:
        return item_points.sum() * 0.0 + 1.0
    item_centres = torch.stack(item_centres)
    code_centres = torch.stack(code_centres)
    intra = []
    for item_centre, code_centre, mask in zip(item_centres, code_centres, members):
        intra.append((distance(item_points[mask], item_centre.unsqueeze(0)) ** 2).mean())
        intra.append((distance(prefix_points[mask], code_centre.unsqueeze(0)) ** 2).mean())
    intra = torch.stack(intra).mean()
    centre_distance = distance(code_centres, code_centres) ** 2
    off_diagonal = ~torch.eye(len(code_centres), dtype=torch.bool,
                               device=code_centres.device)
    inter = centre_distance[off_diagonal].mean()
    return 1.0 - (inter - intra) / (inter + intra).clamp_min(1e-8)

LOSSES = {"gp": gp_alignment, "tree": tree_distortion, "cluster": frechet_cluster}


def compute(mode: str, probs: list, reference: torch.Tensor,
            latents: list, code_points: list = None) -> torch.Tensor:
    """The single active term of this tree, or a differentiable zero."""
    if mode == "none":
        return latents[0].sum() * 0.0
    if mode == "gp":
        return LOSSES["gp"](probs, reference)
    if mode == "tree":
        return LOSSES["tree"](probs, reference)
    if mode == "cluster":
        return LOSSES["cluster"](probs, code_points, latents)
    raise ValueError("unknown auxiliary mode: " + str(mode))
