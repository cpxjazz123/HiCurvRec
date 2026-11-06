"""Behaviour pairs as supervision, contrasted in the Poincare ball.

This arm is the campaign's evidence pointing somewhere else. The six community
arms all failed for one measured reason: a partition of the behaviour graph
carries 1.3 to 2.8 times the behaviour information, while the level-1 codes
carry 6.2 (baseline) to 16.0 (the resample arm) -- so a community partition is a
strictly weaker re-encoding of what the identifiers already hold, and any term
that pulls the codes towards it dilutes them. The arm that reached 0.06374 did
not use a partition at all: it trained the encoder on the training split's own
**transitions**, resampled every epoch, and left the quantizer's assignment
alone.

So this arm takes that mechanism and changes two things:

* the **positives** are the training split's directed successor pairs, not
  community members -- the structure the codes' own behaviour discrimination
  comes from;
* the **kernel** is the geometry contract's Poincare distance rather than the
  Euclidean one, so the behaviour signal and an explicit curve operation are
  coupled on one path (the behaviour enters as the pairs, the curve as the
  metric the pairs are scored with, and the loss is one contrastive
  cross-entropy over the step's candidates).

Two details keep the signal honest. A candidate that is *also* a successor of
the anchor is masked out of the denominator: it is an unlabelled positive, not a
negative, and treating it as a negative would teach the encoder to separate
items the behaviour says belong together. And the term's weight is bounded by
the reconstruction's own encoder-gradient norm, measured every step, so the
supervision cannot dominate the objective it is added to -- the failure the
unbounded community contrast measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .poincare_contract import distance as contract_distance, expmap0


def successor_rows(train_frame) -> dict[int, list[int]]:
    """The training split's directed successors: item -> the items that follow it.

    The sequences are the ones the training parquet holds: a row's history plus
    its target is one training sequence, and consecutive items in it are a
    transition. Only the training split is read.
    """
    successors: dict[int, list[int]] = {}
    history_column = train_frame["history"]
    target_column = train_frame["target"]
    histories = (history_column.tolist() if hasattr(history_column, "tolist")
                 else list(history_column))
    targets = (target_column.tolist() if hasattr(target_column, "tolist")
               else list(target_column))
    for history, target in zip(histories, targets):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            bucket = successors.setdefault(first, [])
            if second not in bucket:
                bucket.append(second)
    return successors


class BehaviourPairs:
    """The training split's transitions, with a positive drawn fresh every step."""

    def __init__(self, train_frame, n_items: int, seed: int = 42):
        self.successors = successor_rows(train_frame)
        self.anchors = np.asarray(sorted(self.successors), dtype=np.int64)
        self.n_items = int(n_items)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.pool = []
        for item in self.anchors:
            for target in self.successors[int(item)]:
                self.pool.append((int(item), int(target)))
        if not self.pool:
            raise ValueError("the training split has no transitions")
        self.pool = np.asarray(self.pool, dtype=np.int64)
        # the membership of the transitions as a sparse matrix over the item ids,
        # so that the mask of the unlabelled positives is one slice instead of a
        # python loop over every successor of every anchor
        sources, targets = [], []
        for anchor, target in self.pool.tolist():
            sources.append(anchor)
            targets.append(target)
        self.matrix = sp.csr_matrix((np.ones(len(sources), dtype=np.int8),
                                      (sources, targets)),
                                     shape=(n_items, n_items))

    def __len__(self) -> int:
        return int(self.pool.shape[0])

    def draw(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """A batch of (anchor, successor) pairs, drawn without replacement."""
        index = torch.randint(len(self.pool), (count,), generator=self.generator)
        chosen = self.pool[index.numpy()]
        return chosen[:, 0].copy(), chosen[:, 1].copy()

    def successor_mask(self, anchors: torch.Tensor,
                       candidates: torch.Tensor) -> torch.Tensor:
        """The (anchors x candidates) mask of pairs the behaviour puts together.

        A candidate that is one of the anchor's successors is an unlabelled
        positive: it is neither the drawn positive nor a negative, so it must
        not be pushed away.
        """
        anchor_ids = anchors.detach().cpu().numpy()
        candidate_ids = candidates.detach().cpu().numpy()
        block = self.matrix[anchor_ids][:, candidate_ids].toarray()
        return torch.from_numpy(block.astype(bool)).to(anchors.device)


def pair_distance(source: torch.Tensor, positive: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """The contract's distance between two latents, lifted at the unit scale.

    This is the surface the reference contract declares. The run lifts at a
    frozen scale (the 95th percentile of the latents' norms) and divides by a
    measured temperature; both are constants of the run, and the distance itself
    is what the contract states.
    """
    left = expmap0(source.double(), c)
    right = expmap0(positive.double(), c)
    return contract_distance(left, right, c)


def curved_logit(source: torch.Tensor, positive: torch.Tensor, temperature: float,
                 scale: float, c: float = 1.0) -> torch.Tensor:
    """The kernel the contract declares: the Poincare distance over the temperature.

    The latents are lifted into the ball with the contract's origin exponential
    at the frozen scale, and the distance between the two ball points is the
    contract's distance.
    """
    left = expmap0(source.double() / scale, c)
    right = expmap0(positive.double() / scale, c)
    return -contract_distance(left, right, c) / temperature


def measure_temperature(source: torch.Tensor, positive: torch.Tensor, scale: float,
                        c: float = 1.0) -> float:
    """The median curved distance of the step's pairs: the term's own scale."""
    with torch.no_grad():
        distances = -curved_logit(source, positive, 1.0, scale, c)
        value = float(distances.median())
    return value if value > 0 else 1.0


def contrastive_loss(logits: torch.Tensor, same_behaviour: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the step's candidates, with the unlabelled positives masked out.

    The target of row i is column i, the drawn successor. Every other column
    that is also a successor of the anchor is excluded from the denominator
    instead of being pushed away.
    """
    size = int(logits.shape[0])
    target = torch.arange(size, device=logits.device)
    excluded = same_behaviour.clone()
    excluded[target, target] = False
    logits = logits.masked_fill(excluded, float("-inf"))
    total = torch.logsumexp(logits, dim=1)
    picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    return (total - picked).mean()


def encoder_gradient_norm(term: torch.Tensor, parameters) -> float:
    """The L2 norm of a term's gradient over the encoder's parameters."""
    gradients = torch.autograd.grad(term, [p for p in parameters if p.requires_grad],
                                    retain_graph=True, allow_unused=True)
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            total += float(gradient.detach().pow(2).sum())
    return total ** 0.5


def bounded_scale(reconstruction_norm: float, behaviour_norm: float) -> float:
    """How much of the behaviour term may reach the encoder.

    The bound is stated once and needs no constant: the behaviour supervision may
    not push the encoder harder than the reconstruction it is added to, so the
    scale is ``min(1, g_reconstruction / g_behaviour)``.
    """
    if behaviour_norm <= 0.0:
        return 1.0
    return float(min(1.0, reconstruction_norm / behaviour_norm))


def latent_scale(latent: torch.Tensor) -> float:
    """The frozen lift scale: the median of the latents' norms.

    The median, not a high percentile: the lift's tanh saturates towards the
    ball's boundary, and points on the boundary make the contract's distance
    saturate too, so the contrast would score every pair the same and carry no
    signal. At the median the lifted points sit well inside the ball.
    """
    with torch.no_grad():
        norms = latent.detach().norm(dim=1)
        scale = float(torch.quantile(norms.float(), 0.5))
    return scale if scale > 0 else 1.0
