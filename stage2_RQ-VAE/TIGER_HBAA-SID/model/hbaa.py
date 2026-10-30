"""The HBAA objective, as a training signal for the Stage-2 quantizer.

The metric of Issue #126's metric rounds asks whether the true next item's
level-1 prefix shares a DEEPER hyperbolic common ancestor with the user's
history than a negative's prefix does. As a training objective it needs a
gradient, and the assignment is a hard argmin, so the objective uses the frozen
proxy's reading of the same question:

* the level-1 assignment of an item is the softmax over the quantizer's own
  level-1 cost, ``p_ib = softmax(-cost_ib / tau_assign)``; the gradient reaches
  the codebook and the encoder through ``p_ib``;
* the prototype of a code is the weighted Frechet mean of the FROZEN reference's
  target representations, i.e. the hyperbolic barycentre of the items the code
  attracts, refreshed every ``REFRESH_STEPS`` steps and held fixed in between;
  it is a stop-gradient quantity, exactly as the Issue #126 proxy states;
* the query point is the Frechet mean of the frozen reference's SOURCE
  representations of the user's whole training history;
* the depth of a code for that query is the Gromov product at the origin,
  ``(u_h | mu_b)_0 = (d(0,u_h) + d(0,mu_b) - d(u_h,mu_b)) / 2``;
* the term is ``softplus(-(depth_negative - depth_positive) / tau)``, with the
  reference's own temperature.

Nothing here is fitted against a result: the temperature, the assignment
temperature and the refresh interval are the frozen values of Issue #126's proxy,
and the weight of the term is set once, from the value the term had on the first
measured batch.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
REFERENCE_NPZ = REPO / "results" / "research" / "issue126" / "reference" / "reference.npz"
REFERENCE_JSON = REPO / "results" / "research" / "issue126" / "reference" / "reference.json"
TRAIN_PARQUET = REPO / "results" / "stage0_build_parquet" / "train.parquet"

ASSIGN_TEMPERATURE = 0.1     # the frozen value of Issue #126's proxy
REFRESH_STEPS = 50           # steps between two prototype refreshes
NEGATIVES = 10
SEED = 42


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exp_0(tangent: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """The exponential map at the origin of the Poincare ball."""
    root = math.sqrt(curvature)
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.tanh(root * norm) * tangent / (root * norm)


def log_0(point: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    root = math.sqrt(curvature)
    norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.atanh((root * norm).clamp(max=1 - 1e-7)) * point / (root * norm)


def poincare_distance(x: torch.Tensor, y: torch.Tensor,
                      curvature: float = 1.0) -> torch.Tensor:
    """The geodesic distance in the Poincare ball."""
    root = math.sqrt(curvature)
    difference = (x - y).pow(2).sum(dim=-1)
    denominator = ((1 - curvature * x.pow(2).sum(dim=-1)).clamp_min(1e-7)
                   * (1 - curvature * y.pow(2).sum(dim=-1)).clamp_min(1e-7))
    return torch.acosh((1 + 2 * curvature * difference / denominator).clamp(min=1 + 1e-12)) / root


def gromov_at_origin(x: torch.Tensor, y: torch.Tensor,
                     curvature: float = 1.0) -> torch.Tensor:
    """The Gromov product at the origin: the shared depth of two points."""
    return 0.5 * (poincare_distance(x, torch.zeros_like(x), curvature)
                  + poincare_distance(y, torch.zeros_like(y), curvature)
                  - poincare_distance(x, y, curvature))


def frechet_mean(points: torch.Tensor, weights: torch.Tensor = None,
                 steps: int = 20, curvature: float = 1.0) -> torch.Tensor:
    """The Karcher mean of a set of points in the ball.

    With weights it is the weighted mean: the iteration is over the points and the
    weights enter the average. The result is used as a stop-gradient quantity.
    """
    if points.shape[0] == 1:
        return points[0]
    if weights is None:
        weights = torch.ones(points.shape[0], device=points.device)
    weights = weights / weights.sum().clamp_min(1e-9)
    centre = exp_0((log_0(points, curvature) * weights[:, None]).sum(dim=0), curvature)
    for _ in range(int(steps)):
        tangent = (log_0(points, curvature) - log_0(centre, curvature)) * weights[:, None]
        step = tangent.sum(dim=0)
        moved = exp_0(step, curvature)
        centre = moved
    return centre


class BehaviourTriples:
    """The training triples: a user, the item they buy next, and ten negatives.

    Built from the TRAINING split only: the user's history gives the query point
    and the negatives are sampled from the items the user did not touch.
    """

    def __init__(self, items: int, device: torch.device):
        import pandas as pd

        payload = np.load(REFERENCE_NPZ)
        record = json.loads(REFERENCE_JSON.read_text())
        self.source = torch.from_numpy(np.asarray(payload["source"], dtype=np.float32)).to(device)
        self.target = torch.from_numpy(np.asarray(payload["target"], dtype=np.float32)).to(device)
        self.temperature = float(record["temperature"])
        self.curvature = 1.0
        self.reference_sha256 = _sha256(REFERENCE_NPZ)

        frame = pd.read_parquet(TRAIN_PARQUET, columns=["user", "history", "target"])
        frame = frame.sort_values("user", kind="stable")
        last = frame.groupby("user", sort=False).tail(1)
        rng = np.random.default_rng(SEED)
        users, positives, negatives = [], [], []
        for history, target in zip(last["history"].to_numpy(), last["target"].to_numpy()):
            sequence = [int(item) for item in list(history)]
            if not sequence:
                continue
            seen = set(sequence) | {int(target)}
            chosen, attempts = [], 0
            while len(chosen) < NEGATIVES and attempts < NEGATIVES * 20:
                candidate = int(rng.integers(0, items))
                attempts += 1
                if candidate not in seen and candidate not in chosen:
                    chosen.append(candidate)
            if len(chosen) < NEGATIVES:
                continue
            users.append(sequence)
            positives.append(int(target))
            negatives.append(np.asarray(chosen, dtype=np.int64))
        self.histories = users
        self.positives = torch.tensor(positives, dtype=torch.int64, device=device)
        self.negatives = torch.from_numpy(np.stack(negatives)).to(device)
        # the query point of every user: the barycentre of its history's source
        # representations (a frozen quantity: the reference does not move)
        self.queries = torch.stack([
            frechet_mean(self.source[torch.tensor(history, dtype=torch.int64,
                                                   device=device)])
            for history in users])

    def __len__(self) -> int:
        return int(self.positives.shape[0])


class HBAAObjective:
    """The training term: the depth of the true prefix against the negatives'."""

    def __init__(self, triples: BehaviourTriples, codes: int, device: torch.device):
        self.triples = triples
        self.codes = int(codes)
        self.device = device
        self.prototypes = None
        self.refresh_step = 0
        self.last_accuracy = float("nan")

    # -- the prototypes: the weighted barycentre of the items each code attracts
    @torch.no_grad()
    def refresh_prototypes(self, assignment: torch.Tensor, targets: torch.Tensor,
                           codes: torch.Tensor) -> None:
        """Refresh the per-code prototypes from the current assignment.

        The prototypes are a stop-gradient quantity (the Issue #126 proxy says so
        too): the gradient of the term reaches the codebook through the
        assignment, not through the barycentre.
        """
        k = self.codes
        # the prototype of a code is the barycentre of the items it attracts,
        # approximated by ONE step of the Karcher iteration: the weighted mean of
        # the members in the log map. The approximation is stated, not hidden, and
        # it is what makes the prototype the SAME on every rank: the per-code
        # weighted sums are reduced across the ranks before the mean is taken.
        weighted = torch.zeros(k, targets.shape[1], device=targets.device)
        mass = torch.zeros(k, device=targets.device)
        weighted.index_add_(0, codes, assignment * log_0(targets, self.triples.curvature))
        mass.index_add_(0, codes, assignment.sum(dim=-1))
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
            dist.all_reduce(mass, op=dist.ReduceOp.SUM)
        prototypes = torch.zeros_like(weighted)
        for code in range(k):
            if float(mass[code]) <= 1e-9:
                continue
            prototypes[code] = exp_0(weighted[code] / mass[code], self.triples.curvature)
        self.prototypes = prototypes

    # -- the term
    def term(self, model, encoded: torch.Tensor, batch: dict) -> torch.Tensor:
        """The HBAA term for one batch of (positive, negatives) triples."""
        layer = model.rq.vq_layers[0]
        positives = batch["positives"]
        negatives = batch["negatives"]
        queries = batch["queries"]

        cost_positive = layer._distances(encoded.index_select(0, positives))
        assignment_positive = torch.softmax(-cost_positive / ASSIGN_TEMPERATURE, dim=-1)
        codes_positive = assignment_positive.argmax(dim=-1)

        cost_negative = layer._distances(encoded.index_select(0, negatives.reshape(-1)))
        assignment_negative = torch.softmax(-cost_negative / ASSIGN_TEMPERATURE, dim=-1)
        codes_negative = assignment_negative.argmax(dim=-1).reshape(negatives.shape)

        self.refresh_step += 1
        if self.prototypes is None or self.refresh_step % REFRESH_STEPS == 1:
            targets = self.triples.target
            self.refresh_prototypes(assignment_positive.detach(),
                                    targets.index_select(0, positives),
                                    codes_positive.detach())
        prototypes = self.prototypes

        def depth(assignment: torch.Tensor, codes: torch.Tensor,
                  query: torch.Tensor) -> torch.Tensor:
            """The expected common-ancestor depth of the assignment."""
            centres = prototypes.index_select(0, codes)
            depths = gromov_at_origin(query[:, None, :], centres, self.triples.curvature)
            return (assignment * depths).sum(dim=-1)

        positive_depth = depth(assignment_positive, codes_positive, queries)
        negative_depth = depth(assignment_negative, codes_negative,
                               queries[:, None, :].expand(-1, negatives.shape[1], -1).reshape(-1, queries.shape[-1]))
        negative_depth = negative_depth.reshape(queries.shape[0], negatives.shape[1])
        delta = positive_depth[:, None] - negative_depth
        self.last_accuracy = float((delta > 0).float().mean())
        return F.softplus(-delta / self.triples.temperature).mean()
