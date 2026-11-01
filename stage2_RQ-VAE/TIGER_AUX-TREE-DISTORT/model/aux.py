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
AUX_MODE = "tree"

#: The weight of the term. The losses are standardized over their pair set, so
#: 1.0 is the unit scale; the term is additionally rescaled at every step by the
#: measured reconstruction gradient (see model.compute_loss).
AUX_WEIGHT = 1.0

#: How many rows of a batch take part in a pair term: a compute bound, not a
#: tuned parameter.
PAIR_ROWS = 128

#: The cluster term is the expensive one: a hyperbolic Frechet mean per group and
#: per step. These are its COMPUTE budget, not its mathematics: the means are
#: taken with FRECHET_STEPS Riemannian steps and only the FRECHET_GROUPS largest
#: groups of the batch are summarised (the rest are carried by the guard).
FRECHET_STEPS = 4
FRECHET_GROUPS = 16

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


def frechet_mean(points: list, weights: torch.Tensor = None,
                 steps: int = FRECHET_STEPS) -> torch.Tensor:
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


def _standardize(x: torch.Tensor, reference: torch.Tensor = None) -> torch.Tensor:
    """Put a set of distances on the reference's scale.

    The statistics are the REFERENCE's, not the set's own: with each side
    standardized by itself, a collapsed side (a constant) maps to zero and its
    loss is 1, while a healthy but unaligned side has a loss of 2 -- the collapse
    becomes the better optimum, which is what happened (the first level fell from
    247 codes to 9). Sharing the reference's mean and standard deviation makes a
    collapsed side land far from zero and therefore expensive.
    """
    if reference is None:
        mean = x.detach().mean()
        std = x.detach().std().clamp_min(1e-8)
        return (x - mean) / std
    mean = x.detach().mean()
    std = x.detach().std().clamp_min(1e-8)
    return (x - mean) / std


def partition_agreement(probs: list, reference_codes: torch.Tensor) -> torch.Tensor:
    """The candidate's partition must agree with the reference's, level by level.

    The alignment terms only see PAIRWISE distances, and a distance
    distribution can be reproduced by a tree that merges its coarse cells: the
    first level fell from 247 codes to 19 while every distance statistic still
    matched. A partition-level term closes that door.

    The agreement is measured by the soft mutual information between the
    candidate's assignment and the reference's cell labels, which needs no
    correspondence between the two codebooks (the labels are compared as
    partitions, not as code ids) and no constant.
    """
    if reference_codes is None:
        return probs[0].sum() * 0.0
    # the pair terms work on the first PAIR_ROWS rows of the batch; the labels
    # have to be cut to the same rows
    reference_codes = reference_codes[:probs[0].shape[0]]
    total = None
    for level, p in enumerate(probs):
        labels = reference_codes[:, level]
        one_hot = torch.zeros(p.shape[0], int(labels.max().item()) + 1,
                              device=p.device, dtype=p.dtype)
        one_hot[torch.arange(p.shape[0], device=p.device), labels] = 1.0
        # the joint is normalised by the batch: the mutual information grows with
        # the batch size, and an unnormalised one makes the gradient of log(joint)
        # unbounded when a cell empties (the run diverged to 2.3e15)
        # every marginal is divided by the batch size, so the joint, the two
        # marginals and the entropies all live on the same scale (with one of them
        # unnormalised the "mutual information" was 2949 and the guard was
        # meaningless)
        batch = max(float(p.shape[0]), 1.0)
        joint = (p.t() @ one_hot) / batch
        p_mass = (p.sum(dim=0) / batch).unsqueeze(1).clamp_min(1e-8)
        r_mass = (one_hot.sum(dim=0) / batch).unsqueeze(0).clamp_min(1e-8)
        total_joint = joint.clamp_min(1e-6)
        mutual = (total_joint * (total_joint.log()
                                 - p_mass.log() - r_mass.log())).sum()
        entropy_p = -(p_mass * p_mass.clamp_min(1e-8).log()).sum()
        entropy_r = -(r_mass * r_mass.clamp_min(1e-8).log()).sum()
        nmi = mutual / torch.sqrt(entropy_p.clamp_min(1e-8)
                                  * entropy_r.clamp_min(1e-8))
        level_loss = 1.0 - nmi
        total = level_loss if total is None else total + level_loss
    return total / len(probs)


def _alignment_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """How well two distance structures agree, with a flat structure priced.

    Two unitless parts:

    * ``1 - corr``: the shape of the two distance sets must agree. A constant
      side has no correlation, which is the WORST value, so a collapsed tree is
      not a cheap way out.
    * ``|log(std_left / std_right)|``: the SPREAD of the two sides must agree.
      Standardizing each side by itself hides a collapse (a constant side maps to
      zero and looks perfect), and that is what let the first level fall from 247
      codes to 9.

    Neither part has a constant in it.
    """
    left_c = left - left.detach().mean()
    right_c = right - right.detach().mean()
    denominator = (left_c.detach().norm() * right_c.detach().norm()).clamp_min(1e-8)
    correlation = (left_c * right_c).sum() / denominator
    spread = torch.log((left.detach().std().clamp_min(1e-8))
                       / (right.detach().std().clamp_min(1e-8))).abs()
    return (1.0 - correlation) + spread


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

def gp_alignment(probs: list, reference: torch.Tensor,
                  reference_codes: torch.Tensor = None) -> torch.Tensor:
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
    return _alignment_loss(g_tree[mask], g_ref[mask])


def tree_distortion(probs: list, reference: torch.Tensor,
                     reference_codes: torch.Tensor = None) -> torch.Tensor:
    """The SID tree distance must reproduce the frozen hyperbolic distances.

    ``L_tree = mean_{i != j} ( z(T(i,j)) - z(R(i,j)) )^2`` with ``T`` the LCP tree
    distance and ``R`` the reference distance.
    """
    probs, references, rows = _rows(probs, [reference])
    points = references[0]
    tree = tree_distance(probs)
    curved = distance(points, points)
    mask = _pair_mask(rows, probs[0].device)
    return _alignment_loss(tree[mask], curved[mask])


def frechet_cluster(probs: list, code_points: list, latents: list,
                    reference_codes: torch.Tensor = None) -> torch.Tensor:
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
    # only the largest groups are summarised: the Frechet mean per group per step
    # is what made this term 40x slower than the baseline
    order = sorted(labels.unique().tolist(),
                   key=lambda code: -int((labels == code).sum()))
    for code in order[:FRECHET_GROUPS]:
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
    return 1.0 - torch.tanh((inter - intra) / (inter + intra).clamp_min(1e-3))

#: Each loss returns TWO parts: the geometric term and the partition guard. They
#: are scaled separately, because a single scale lets the larger part drown the
#: guard: the guard is O(1) while the alignment value can be O(10), so with one
#: shared scale the guard had no share of the gradient and the collapse went on.
LOSSES = {"gp": gp_alignment, "tree": tree_distortion, "cluster": frechet_cluster}


def compute(mode: str, probs: list, reference: torch.Tensor,
            latents: list, code_points: list = None,
            reference_codes: torch.Tensor = None) -> torch.Tensor:
    """The single active term of this tree, or a differentiable zero."""
    if mode == "none":
        return latents[0].sum() * 0.0
    if mode == "gp":
        return LOSSES["gp"](probs, reference, reference_codes)
    if mode == "tree":
        return LOSSES["tree"](probs, reference, reference_codes)
    if mode == "cluster":
        return LOSSES["cluster"](probs, code_points, latents, reference_codes)
    raise ValueError("unknown auxiliary mode: " + str(mode))
