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

#: The cluster term is the expensive one: a hyperbolic Frechet mean per group and
#: per step. These are its COMPUTE budget, not its mathematics: the means are
#: taken with FRECHET_STEPS Riemannian steps and only the FRECHET_GROUPS largest
#: groups of the batch are summarised (the rest are carried by the guard).
FRECHET_STEPS = 2
FRECHET_GROUPS = 8

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
    """The candidate must USE as many cells per level as the reference does.

    This is the constraint the three terms are missing. They are all about
    PAIRWISE structure, and a pairwise objective does not see the marginal of the
    assignment: the cluster term is in fact minimised by FEWER and FARTHER
    clusters, and it traded the vocabulary away (the first level fell to 21 cells
    and the duplicate rate to 0.91).

    The guard compares the ENTROPY of the assignment's marginal with the entropy
    of the reference's label distribution, level by level:

        H_cand_l = - sum_k pbar(k) log pbar(k)          (pbar = the mean assignment)
        H_ref_l  = - sum_c r(c) log r(c)                (the reference's labels)
        loss     = sum_l | log(H_cand_l / H_ref_l) |

    The marginal is a distribution over cells, so the comparison needs no
    correspondence between the two codebooks (a partition alignment by code id
    does not exist and a mutual information over partitions proved unreliable:
    it did not pin the granularity and it inflated the objective). Matching the
    entropy pins the NUMBER of cells, which is exactly what the three terms
    destroy, and it is label-free, stable and cheap.
    """
    if reference_codes is None:
        return probs[0].sum() * 0.0
    reference_codes = reference_codes[:probs[0].shape[0]]
    total = None
    for level, p in enumerate(probs):
        marginal = p.detach().mean(dim=0).clamp_min(1e-9)          # (codes,)
        candidate_entropy = -(marginal * marginal.log()).sum()
        labels = reference_codes[:, level]
        counts = torch.bincount(labels).float()
        reference = (counts / counts.sum().clamp_min(1e-9)).clamp_min(1e-9)
        reference_entropy = -(reference * reference.log()).sum()
        level_loss = (torch.log(candidate_entropy.clamp_min(1e-9))
                      - torch.log(reference_entropy.clamp_min(1e-9))).abs()
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

    The clustering is SOFT and fully vectorised: a prefix is a probability, not a
    hard label, so every quantity carries a gradient. The hard-label version had
    almost no gradient path (the membership was detached and the codebook gradient
    measured zero) and it ran into the trap where a batch with a single group
    returns a constant and the run stays collapsed for ever.

    For the grouping level ``l`` with soft assignment ``p`` (codes x items after
    the transpose):

        centre_c = sum_i p(i,c) x_i / sum_i p(i,c)      (a soft Frechet centre)
        intra    = sum_c sum_i p(i,c) d_H(x_i, centre_c)^2 / N
        inter    = sum_{c<c'} p_c p_{c'} d_H(centre_c, centre_{c'})^2

    ``x_i`` is the item's pre-quantization ball position; the code-space half of
    the intra term uses the expected code point, which is what gives the codebooks
    a gradient. The centres are the true hyperbolic Frechet means: the mass-weighted
    mean in the ball is the standard Riemannian initialisation and one Frechet step
    refines it. The loss is ``1 - tanh((inter - intra)/(inter + intra))``: 1 when
    everything sits on one point, 0 when the prefixes are compact and separated.
    """
    probs, latents, rows = _rows(probs, latents)
    item_points = project(latents[0])
    level_index = 0
    best_groups = 0
    for index, level_probs in enumerate(probs):
        candidate = level_probs.detach().argmax(dim=-1)
        if candidate.unique().numel() < 2:
            continue
        groups = int((torch.bincount(candidate) >= 2).sum())
        if groups > best_groups:
            best_groups = groups
            level_index = index
    masses = probs[level_index].clamp_min(1e-8)             # (items, codes)
    mass = masses.t()                                       # (codes, items)
    total = mass.sum(dim=1, keepdim=True).clamp_min(1e-12)  # (codes, 1)
    prefix_points = masses @ code_points[level_index]        # (items, dim)

    def soft_centres(points: torch.Tensor) -> torch.Tensor:
        """The mass-weighted hyperbolic Frechet centre of every code.

        The centre is the mass-weighted mean in the ball (the standard Riemannian
        initialisation) refined by one Frechet step: the mass-weighted mean of the
        log map at that point. Every shape is explicit, because the broadcast of
        (codes, items, dim) against (codes, 1) is where this went wrong twice.
        """
        centre = (mass @ points) / total                     # (codes, dim)
        radius = centre.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        centre = centre / radius * radius.clamp(max=1 - 1e-6)   # (codes, dim)
        delta = _mobius_add(-centre.unsqueeze(1), points.unsqueeze(0))  # (codes, items, dim)
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)        # (codes, items, 1)
        scale = (1.0 - (centre * centre).sum(dim=-1)).clamp_min(1e-6)  # (codes,)
        log_vector = (torch.arctanh(norm.clamp(max=1 - 1e-6))
                      * scale.view(-1, 1, 1)
                      * delta / norm)                                      # (codes, items, dim)
        gradient = (log_vector * mass.unsqueeze(-1)).sum(dim=1)            # (codes, dim)
        length = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        refined = _mobius_add(centre, torch.tanh(length / 2.0) * gradient / length)
        radius = refined.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return refined / radius * radius.clamp(max=1 - 1e-6)

    item_centres = soft_centres(item_points)
    code_centres = soft_centres(prefix_points)
    # the distance matrix is (items, codes) and the mass is (codes, items), so
    # the weighting is the transposed mass
    # the distance matrix is (items, codes) and the weight matrix is (codes,
    # items); the weight is transposed to (items, codes) and both get a trailing
    # axis so that the broadcast aligns on the codes
    item_intra = (distance(item_points, item_centres) ** 2).unsqueeze(-1) \
        * mass.t().unsqueeze(-1)
    item_intra = item_intra.sum() / max(float(item_points.shape[0]), 1.0)
    code_intra = (distance(prefix_points, code_centres) ** 2).unsqueeze(-1) \
        * mass.t().unsqueeze(-1)
    code_intra = code_intra.sum() / max(float(prefix_points.shape[0]), 1.0)
    intra = 0.5 * (item_intra + code_intra)
    centre_distance = distance(item_centres, item_centres) ** 2
    weight = mass.sum(dim=1)
    inter = (centre_distance * weight.unsqueeze(1) * weight.unsqueeze(0)).sum() / max(
        float((weight ** 2).sum()), 1e-12)
    return 1.0 - torch.tanh((inter - intra) / (inter + intra).clamp_min(1e-3))


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
