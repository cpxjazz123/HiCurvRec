"""The soft community model: a free branch, an optional hyperbolic branch.

Both branches receive the SAME frozen 768-dimensional input and learn their own
representation; the input embedding is never modified. Each branch owns 64
learnable community centres, and the two branch distances are combined into the
cost that the soft assignment sees.

Geometry: the hyperbolic branch uses the frozen contract operators of the
framework's geometry module (``tools/research/geometry/poincare_contract.py``),
reached through the origin exponential map, with the curvature fixed at c = 1
(sectional curvature -1). The operators are imported, not re-derived, and the
file's sha256 is recorded in the run report so a reader can tell which geometry
ran.

The distance combination has no free knob. Each branch's distance row (the 64
distances of one item to the centres) is standardised per item -- zero mean,
unit spread -- and the branches are averaged with equal weight. The soft
assignment is then softmax(-distance) over the centres: the standardisation
makes the temperature implicit and scale-free, so no temperature, weight or
margin had to be chosen by trying values.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[4]
#: The geometry contract is the framework's, and it lives in the framework. The
#: frozen baseline used to ship a byte-identical copy; that copy was removed with
#: the baseline's hyperbolic leftovers, so the canonical module is read directly.
CONTRACT_PATH = REPO / "tools" / "research" / "geometry" / "poincare_contract.py"


def load_contract():
    """The frozen geometry contract, with the sha256 of the file it came from."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("community_curvature_contract",
                                                  CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the geometry contract at {CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    return module, digest


def centre_range(distances: torch.Tensor) -> torch.Tensor:
    """The baseline's cost centring: the matrix is mapped onto [-1, 1].

    It is the centring of ``Quantizer.center_distance`` in the Stage-2
    baseline, which is the cost the baseline's Sinkhorn plan is computed on.
    """
    maximum = distances.max()
    minimum = distances.min()
    middle = (maximum + minimum) / 2
    amplitude = maximum - middle + 1e-5
    if not bool(amplitude > 0):
        raise ValueError("the cost matrix is constant; it cannot be centred")
    return (distances - middle) / amplitude


def sinkhorn_plan(distances: torch.Tensor, epsilon: float = 0.003,
                  iterations: int = 100) -> torch.Tensor:
    """The baseline's balanced assignment plan, on a centred cost.

    This is ``Quantizer.sinkhorn`` of the Stage-2 baseline: the plan's rows sum
    to one, so a row is a soft community distribution, and the column
    normalisation pulls the plan towards a balanced use of the communities
    (with five iterations on a near-decision cost the balance is approximate,
    not exact -- the baseline's own rule behaves the same way). The RQ-VAE
    quantizer assigns its codes with exactly this rule, and the test suite
    asserts the two implementations agree on the values.

    The iteration count differs from the baseline's five, and deliberately so:
    with a near-decision cost five iterations leave the columns of the plan
    unbalanced by tens of percent, and an unbalanced plan is exactly what lets
    a node-to-cluster term fold the whole batch onto one community. Running
    the iteration to convergence makes the column balance real; the cost is a
    solver setting, not a tuned quantity.
    """
    q = torch.exp(-distances / epsilon)
    batch_size, codebook_size = q.shape
    q = q / q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    for _ in range(iterations):
        # the same normalisation as the baseline, written out of place: the
        # baseline's version runs under no_grad because its plan is a decision,
        # while here the plan has to carry the gradient of the losses
        q = q / q.sum(0, keepdim=True)
        q = q / codebook_size
        q = q / q.sum(1, keepdim=True)
        q = q / batch_size
    return q * batch_size


class CurvatureGraphConvolution(nn.Module):
    """One curvature-aware propagation over the behaviour graph.

    The edge weights are fixed, because the curvature of an edge is a property
    of the graph and not of the model. The AF3 curvature of the edge is
    standardised over the graph's edges and turned into a multiplicative
    weight on the GCN's symmetric normalisation,

        alpha_ij  ~  exp(z_ij) / sqrt(deg_i * deg_j)

    and the row is normalised, so every node's message is a weighted average of
    its neighbours and the degree of a node cannot blow the message up. The
    curvature enters monotonically: the higher the AF3 curvature of an edge,
    the more of the neighbour it carries.

    With ``curvature_aware=False`` the curvature factor is dropped and the
    aggregator is the plain GCN mean, which is the control for the curvature
    weighting: it tells whether the curvature is what helps or merely the
    neighbourhood.
    """

    def __init__(self, rows: np.ndarray, cols: np.ndarray, z: np.ndarray,
                 degrees: np.ndarray, curvature_aware: bool = True):
        super().__init__()
        if rows.size == 0:
            raise ValueError("the graph has no edges")
        factor = np.exp(z) if curvature_aware else np.ones_like(z)
        normalisation = 1.0 / np.sqrt(np.maximum(degrees[rows], 1.0)
                                     * np.maximum(degrees[cols], 1.0))
        alpha = factor * normalisation
        totals = np.zeros(degrees.size, dtype=np.float64)
        np.add.at(totals, rows, alpha)
        alpha = alpha / np.maximum(totals[rows], 1e-12)
        indices = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
        values = torch.from_numpy(alpha.astype(np.float32))
        # a buffer, so the module's .to(device) carries the sparse adjacency
        # with it; a plain attribute would stay on the CPU
        self.register_buffer("adjacency", torch.sparse_coo_tensor(
            indices, values, (int(degrees.size), int(degrees.size))).coalesce())
        self.curvature_aware = bool(curvature_aware)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """The message every node receives: a weighted average of its neighbours."""
        return torch.sparse.mm(self.adjacency, h)


class CurvatureHead(nn.Module):
    """An MLP that reads an item's community point and predicts its local curvature.

    The target is the mean AF3 curvature of the item's incident edges, which is
    a property of the behaviour graph. If a representation can predict it, the
    representation has encoded the local graph structure the partition is
    supposed to follow.
    """

    def __init__(self, width: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ball_point: torch.Tensor) -> torch.Tensor:
        return self.net(ball_point).squeeze(-1)


class CommunityModel(nn.Module):
    """Soft community assignment over K communities from a frozen embedding."""

    def __init__(self, input_dim: int = 768, free_dim: int = 128, hyper_dim: int = 0,
                 n_communities: int = 64, curvature: float = 1.0, seed: int = 42,
                 graph: CurvatureGraphConvolution | None = None,
                 learn_curvature: bool = False, curvature_head: bool = False):
        super().__init__()
        if free_dim <= 0:
            raise ValueError("the free branch needs a positive width")
        self.free_dim = int(free_dim)
        self.hyper_dim = int(hyper_dim)
        self.n_communities = int(n_communities)
        self.curvature = float(curvature)
        generator = torch.Generator().manual_seed(int(seed))
        # The curvature of the community space can be learned. It is
        # parameterised as exp(raw) so that it stays positive and starts at the
        # fixed value the earlier rounds used (raw = 0 gives c = 1): a model with
        # learn_curvature=False is then exactly the fixed-curvature model, and
        # the two are comparable without a sweep.
        self.learn_curvature = bool(learn_curvature)
        self.raw_curvature = nn.Parameter(torch.tensor(0.0),
                                          requires_grad=self.learn_curvature)

        self.free_projection = nn.Linear(input_dim, self.free_dim, bias=False)
        self.free_centres = nn.Parameter(torch.randn(self.n_communities, self.free_dim,
                                                     generator=generator) * 0.01)
        if self.hyper_dim > 0:
            self.contract, self.contract_sha256 = load_contract()
            self.hyper_projection = nn.Linear(input_dim, self.hyper_dim, bias=False)
            self.hyper_centres = nn.Parameter(torch.randn(self.n_communities, self.hyper_dim,
                                                          generator=generator) * 0.01)
        else:
            self.contract, self.contract_sha256 = None, None

        # The node-to-cluster terms need an ABSOLUTE distance, and the only
        # absolute scale in the model is arbitrary (the projection's scale and
        # the centres' spread are both free parameters). The scales are read
        # once from the first batch and then frozen, so the terms mean
        # "as close as the model started" and "as far apart as the centres
        # started".
        self.graph = graph
        self.curvature_head = None
        if curvature_head:
            if self.hyper_dim <= 0:
                raise ValueError("the curvature head reads the ball point; the model "
                                 "needs a hyperbolic branch")
            self.curvature_head = CurvatureHead(self.hyper_dim)
        if graph is not None:
            # the propagation mixes the neighbourhood into the representation;
            # the strength is a learnable scalar, so the model can turn the
            # neighbour information off if it does not help
            self.gamma_free = nn.Parameter(torch.tensor(1.0))
            self.gamma_hyper = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("scale_free", torch.tensor(1.0))
        self.register_buffer("scale_hyper", torch.tensor(1.0))
        self.register_buffer("separation_threshold", torch.tensor(0.0))
        self.register_buffer("scales_frozen", torch.tensor(0.0))

    @property
    def curvature_value(self) -> torch.Tensor:
        """The curvature of the community ball, fixed or learned."""
        if not self.learn_curvature:
            return torch.tensor(self.curvature)
        return torch.exp(self.raw_curvature)

    # ------------------------------------------------------------------ branches
    def represent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The representation of every item, after the graph propagation.

        The propagation runs over every item, because a neighbour of a batch
        item need not be in the batch. The hyperbolic branch is propagated in
        the tangent space and mapped into the ball afterwards, so the geometry
        is applied to the propagated representation.
        """
        free = self.free_projection(x)
        if self.graph is not None:
            free = free + self.gamma_free * self.graph(free)
        if self.hyper_dim <= 0:
            return free, None
        tangent = self.hyper_projection(x)
        if self.graph is not None:
            tangent = tangent + self.gamma_hyper * self.graph(tangent)
        return free, self.contract.expmap0(tangent, self.curvature_value)

    def free_branch(self, x: torch.Tensor) -> torch.Tensor:
        """The item's coordinates in the free space."""
        return self.free_projection(x)

    def hyperbolic_branch(self, x: torch.Tensor) -> torch.Tensor:
        """The item's point in the Poincare ball, through the origin exponential."""
        if self.hyper_dim <= 0:
            raise RuntimeError("this model has no hyperbolic branch")
        tangent = self.hyper_projection(x)
        return self.contract.expmap0(tangent, self.curvature_value)

    def ball_limit(self) -> float:
        """The radius the contract keeps every ball point inside."""
        curvature = self.curvature_value
        return float((1.0 - self.contract.BALL_EPS)
                     / float(torch.sqrt(curvature.detach())))

    def centres(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The community centres: free coordinates and, if present, ball points."""
        free = self.free_centres
        if self.hyper_dim <= 0:
            return free, None
        return free, self.contract.expmap0(self.hyper_centres, self.curvature_value)

    # ------------------------------------------------------------------ distances
    def distances(self, free_items: torch.Tensor,
                  ball_items: torch.Tensor | None = None) -> torch.Tensor:
        """The combined distance of every item to every community centre.

        Each branch's row is standardised per item (zero mean, unit spread over
        the centres) and the branches are averaged with equal weight.
        """
        free_centres, hyper_centres = self.centres()
        # the matrix-multiplication mode of cdist cancels the squares of large
        # norms away, which would make the scale of a branch visible in its
        # standardised distances; the exact pairwise mode does not
        d_free = torch.cdist(free_items.double(), free_centres.double(),
                             compute_mode="donot_use_mm_for_euclid_dist")
        parts = [_standardise(d_free)]
        if ball_items is not None:
            # the contract's distance is elementwise, so the pair grid is built here
            d_hyper = self.contract.distance(ball_items.double()[:, None, :],
                                             hyper_centres.double()[None, :, :],
                                             self.curvature_value)
            parts.append(_standardise(d_hyper))
        combined = torch.stack(parts, dim=0).mean(dim=0)
        return combined

    def branch_logits(self, free_items: torch.Tensor,
                      ball_items: torch.Tensor | None = None):
        """Each branch's own score of every community, before they are combined.

        The combined distance standardises each branch and averages them; these
        are the branch's own logits, which is what the cross-space consistency
        term compares.
        """
        free_centres, hyper_centres = self.centres()
        d_free = torch.cdist(free_items.double(), free_centres.double(),
                             compute_mode="donot_use_mm_for_euclid_dist")
        if ball_items is None:
            return -d_free, None
        d_ball = self.contract.distance(ball_items.double()[:, None, :],
                                       hyper_centres.double()[None, :, :],
                                       self.curvature_value)
        return -d_free, -d_ball

    def assignment(self, free_items: torch.Tensor,
                   ball_items: torch.Tensor | None = None) -> torch.Tensor:
        """The soft community distribution of every item.

        The assignment is the baseline's plan on the centred cost, so the
        distribution is nearly a decision: the soft usage and the hard usage
        agree, which is what lets the usage constraint see the hard partition
        instead of a flat distribution that hides a collapse.
        """
        return sinkhorn_plan(centre_range(self.distances(free_items, ball_items)))

    # ------------------------------------------------------- node-to-cluster
    def branch_distances(self, free_items: torch.Tensor,
                         ball_items: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The raw distance of every item to every centre, per branch."""
        free_centres, hyper_centres = self.centres()
        d_free = torch.cdist(free_items.double(), free_centres.double(),
                             compute_mode="donot_use_mm_for_euclid_dist")
        if ball_items is None:
            return d_free, None
        d_hyper = self.contract.distance(ball_items.double()[:, None, :],
                                         hyper_centres.double()[None, :, :],
                                         self.curvature_value)
        return d_free, d_hyper

    def absolute_distances(self, free_items: torch.Tensor,
                           ball_items: torch.Tensor | None = None) -> torch.Tensor:
        """The distance of every item to every centre, in the frozen scale units.

        The standardised distances of ``distances`` cannot carry a node-to-
        cluster term: standardising per item rescales the row, so the term
        would never ask an item to move closer in absolute terms. Here each
        branch is divided by the scale it had when the scales were frozen.
        """
        d_free, d_hyper = self.branch_distances(free_items, ball_items)
        parts = [d_free / self.scale_free]
        if d_hyper is not None:
            parts.append(d_hyper / self.scale_hyper)
        return torch.stack(parts, dim=0).mean(dim=0)

    def centre_distances(self) -> torch.Tensor:
        """The distance between every pair of community centres."""
        free_centres, hyper_centres = self.centres()
        free = torch.cdist(free_centres.double(), free_centres.double(),
                           compute_mode="donot_use_mm_for_euclid_dist") / self.scale_free
        if self.hyper_dim <= 0:
            return free
        hyper = self.contract.distance(hyper_centres.double()[:, None, :],
                                       hyper_centres.double()[None, :, :],
                                       self.curvature_value) / self.scale_hyper
        return (free + hyper) / 2.0

    def assignment_and_absolute(self, free_items: torch.Tensor,
                                 ball_items: torch.Tensor | None = None
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """The soft assignment and the absolute distance, from one forward pass."""
        d_free, d_hyper = self.branch_distances(free_items, ball_items)
        standardised = [_standardise(d_free)]
        absolute = [d_free / self.scale_free]
        if d_hyper is not None:
            standardised.append(_standardise(d_hyper))
            absolute.append(d_hyper / self.scale_hyper)
        pi = sinkhorn_plan(centre_range(torch.stack(standardised, dim=0).mean(dim=0)))
        return pi, torch.stack(absolute, dim=0).mean(dim=0)

    @torch.no_grad()
    def freeze_scales(self, free_items: torch.Tensor,
                      ball_items: torch.Tensor | None = None) -> dict:
        """Read the branch scales and the separation threshold from one batch.

        The threshold is the mean distance from a centre to its nearest
        neighbour at that moment: the centres are asked to stay at least that
        far apart, and no further, so the term cannot run away.
        """
        if float(self.scales_frozen) == 1.0:
            return {"scale_free": float(self.scale_free),
                    "scale_hyper": float(self.scale_hyper),
                    "separation_threshold": float(self.separation_threshold)}
        d_free, d_hyper = self.branch_distances(free_items, ball_items)
        self.scale_free = d_free.mean().clamp_min(1e-8)
        if d_hyper is not None:
            self.scale_hyper = d_hyper.mean().clamp_min(1e-8)
        centres = self.centre_distances()
        eye = torch.eye(centres.shape[0], dtype=centres.dtype, device=centres.device)
        nearest = (centres + eye * 1e9).min(dim=1).values
        self.separation_threshold = nearest.mean()
        self.scales_frozen = torch.tensor(1.0)
        return {"scale_free": float(self.scale_free),
                "scale_hyper": float(self.scale_hyper),
                "separation_threshold": float(self.separation_threshold)}


def cross_space_divergence(free_logit: torch.Tensor, ball_logit: torch.Tensor) -> torch.Tensor:
    """How much the two subspaces disagree about an item's community.

    The free branch and the hyperbolic branch each give a distribution over the
    communities; the symmetric Jensen-Shannon divergence between the two rows is
    zero when they agree and maximal when they are disjoint. It is the
    cross-space consistency the plan listed as missing.
    """
    free = torch.softmax(free_logit, dim=1)
    ball = torch.softmax(ball_logit, dim=1)
    mixture = 0.5 * (free + ball)
    left = 0.5 * (free * (free.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(dim=1)
    right = 0.5 * (ball * (ball.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(dim=1)
    return (left + right).mean()


def _standardise(rows: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero mean and unit spread per row, so no branch can dominate by scale.

    The statistics are taken in double precision: the rows are differences of
    large coordinates, and a single-precision mean would cancel away the very
    spread the standardisation is supposed to measure.
    """
    values = rows.double()
    mean = values.mean(dim=1, keepdim=True)
    spread = values.std(dim=1, keepdim=True, unbiased=False)
    standardised = (values - mean) / (spread + eps)
    return standardised.to(rows.dtype)


# ------------------------------------------------------------------------ losses
def overlap(pi_left: torch.Tensor, pi_right: torch.Tensor) -> torch.Tensor:
    """S_ij = sum_k pi_ik pi_jk: the probability mass two items share."""
    return (pi_left * pi_right).sum(dim=1)


def ranking_loss(pi_anchor: torch.Tensor, pi_high: torch.Tensor, pi_low: torch.Tensor,
                 margin: float) -> torch.Tensor:
    """max(0, m - S_AB + S_AC) over the anchors of a batch."""
    s_high = overlap(pi_anchor, pi_high)
    s_low = overlap(pi_anchor, pi_low)
    return torch.clamp(margin - s_high + s_low, min=0.0).mean()


def usage_loss(pi_mean: torch.Tensor, n_communities: int) -> torch.Tensor:
    """1 - H(usage)/log K: zero when the communities are used equally.

    The usage distribution is the mean soft assignment over the items, so the
    term is exactly zero for a uniform usage and one for a fully collapsed one.
    """
    usage = pi_mean.clamp_min(1e-12)
    entropy = -(usage * usage.log()).sum()
    return 1.0 - entropy / float(torch.log(torch.tensor(float(n_communities))))


def commitment_loss(pi: torch.Tensor, absolute: torch.Tensor) -> torch.Tensor:
    """The node-to-cluster term: the distance an item keeps to the centre it leans on.

    ``sum_k pi_ik * d(i, k)`` is the expected distance to the community the
    item's soft assignment leans towards, so minimising it moves the item
    towards its own centre instead of only ranking the centres against each
    other.
    """
    return (pi * absolute).sum(dim=1).mean()


def separation_loss(centre_distances: torch.Tensor, threshold: float) -> torch.Tensor:
    """The distinctness term: every centre keeps at least ``threshold`` from its nearest neighbour.

    The hinge is zero once the centres are far enough apart, so the term
    cannot be won by pushing the centres to infinity; it only prevents the
    communities from folding into one another.
    """
    size = centre_distances.shape[0]
    eye = torch.eye(size, dtype=centre_distances.dtype, device=centre_distances.device)
    nearest = (centre_distances + eye * 1e9).min(dim=1).values
    return torch.clamp(threshold - nearest, min=0.0).mean()


def margin_for(n_communities: int) -> float:
    """The margin, derived from the community count and not chosen by trying values.

    Under a uniform soft assignment every pair shares S = 1/K. Requiring
    S_AB - S_AC >= 1/K therefore asks the high-curvature pair to share more
    community mass than a random pair would, which is the statement the
    experiment is about.
    """
    return 1.0 / float(n_communities)
