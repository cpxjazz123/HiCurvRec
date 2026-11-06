"""The soft community model: a free branch, an optional hyperbolic branch.

Both branches receive the SAME frozen 768-dimensional input and learn their own
representation; the input embedding is never modified. Each branch owns 64
learnable community centres, and the two branch distances are combined into the
cost that the soft assignment sees.

Geometry: the hyperbolic branch uses the frozen contract operators of the
Stage-2 baseline (``model/poincare_contract.py``), reached through the origin
exponential map, with the curvature fixed at c = 1 (sectional curvature -1). The
operators are imported, not re-derived, and the file's sha256 is recorded in the
run report so a reader can tell which geometry ran.

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
CONTRACT_PATH = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "model" / "poincare_contract.py"


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


class CommunityModel(nn.Module):
    """Soft community assignment over K communities from a frozen embedding."""

    def __init__(self, input_dim: int = 768, free_dim: int = 128, hyper_dim: int = 0,
                 n_communities: int = 64, curvature: float = 1.0, seed: int = 42):
        super().__init__()
        if free_dim <= 0:
            raise ValueError("the free branch needs a positive width")
        self.free_dim = int(free_dim)
        self.hyper_dim = int(hyper_dim)
        self.n_communities = int(n_communities)
        self.curvature = float(curvature)
        generator = torch.Generator().manual_seed(int(seed))

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

    # ------------------------------------------------------------------ branches
    def free_branch(self, x: torch.Tensor) -> torch.Tensor:
        """The item's coordinates in the free space."""
        return self.free_projection(x)

    def hyperbolic_branch(self, x: torch.Tensor) -> torch.Tensor:
        """The item's point in the Poincare ball, through the origin exponential."""
        if self.hyper_dim <= 0:
            raise RuntimeError("this model has no hyperbolic branch")
        tangent = self.hyper_projection(x)
        return self.contract.expmap0(tangent, self.curvature)

    def ball_limit(self) -> float:
        """The radius the contract keeps every ball point inside."""
        return float((1.0 - self.contract.BALL_EPS) / np.sqrt(self.curvature))

    def centres(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The community centres: free coordinates and, if present, ball points."""
        free = self.free_centres
        if self.hyper_dim <= 0:
            return free, None
        return free, self.contract.expmap0(self.hyper_centres, self.curvature)

    # ------------------------------------------------------------------ distances
    def distances(self, x: torch.Tensor) -> torch.Tensor:
        """The combined distance of every item to every community centre.

        Each branch's row is standardised per item (zero mean, unit spread over
        the centres) and the branches are averaged with equal weight.
        """
        free_items = self.free_branch(x).double()
        free_centres, hyper_centres = self.centres()
        # the matrix-multiplication mode of cdist cancels the squares of large
        # norms away, which would make the scale of a branch visible in its
        # standardised distances; the exact pairwise mode does not
        d_free = torch.cdist(free_items, free_centres.double(),
                             compute_mode="donot_use_mm_for_euclid_dist")
        parts = [_standardise(d_free)]
        if self.hyper_dim > 0:
            ball_items = self.hyperbolic_branch(x).double()
            # the contract's distance is elementwise, so the pair grid is built here
            d_hyper = self.contract.distance(ball_items[:, None, :],
                                             hyper_centres.double()[None, :, :],
                                             self.curvature)
            parts.append(_standardise(d_hyper))
        combined = torch.stack(parts, dim=0).mean(dim=0)
        return combined

    def assignment(self, x: torch.Tensor) -> torch.Tensor:
        """The soft community distribution of every item."""
        return torch.softmax(-self.distances(x), dim=1)


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


def margin_for(n_communities: int) -> float:
    """The margin, derived from the community count and not chosen by trying values.

    Under a uniform soft assignment every pair shares S = 1/K. Requiring
    S_AB - S_AC >= 1/K therefore asks the high-curvature pair to share more
    community mass than a random pair would, which is the statement the
    experiment is about.
    """
    return 1.0 / float(n_communities)
