"""Issue #274: the curvature-selected relative pair ranking on the final hard q.

The mechanism (see ``results/research/issue274/CONTRACT.md``) supervises a **two-pair distance
ranking** on the final hard representation ``q = sum_l c_l``:

    L_geo = mean over the groups of  max(0, ||q_B - q_C|| - ||q_D - q_E|| + m)

``(B, C)`` is the positive outer pair (the closed-triangle triplet with the highest raw graph
curvature in the centre's comparable group) and ``(D, E)`` the negative outer pair (the open-wedge
triplet with the lowest raw graph curvature in the same group).  The pairs are selected **offline**
and frozen; the curvature value never enters this module.

The gradient of ``L_geo`` reaches the three codebooks through the embedding lookup.  It does not
reach the encoder: the level selection is an argmin over the codebook distances, so the selected
codes are constants with respect to the encoder parameters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

#: the frozen margin, calibrated in advance on the frozen TIGER's training-pair distance scale
#: (results/research/issue274/baselines.json).  Identical for every valid sample.
MARGIN = 0.2529585361480713


class PairIndex:
    """The frozen positive/negative pair index (read-only for the model)."""

    def __init__(self, centre, pos_b, pos_c, neg_d, neg_e, pos_curvature, neg_curvature,
                 is_holdout, train_mask=None):
        self.centre = np.asarray(centre, dtype=np.int64)
        self.pos_b = np.asarray(pos_b, dtype=np.int64)
        self.pos_c = np.asarray(pos_c, dtype=np.int64)
        self.neg_d = np.asarray(neg_d, dtype=np.int64)
        self.neg_e = np.asarray(neg_e, dtype=np.int64)
        self.pos_curvature = np.asarray(pos_curvature, dtype=np.float64)
        self.neg_curvature = np.asarray(neg_curvature, dtype=np.float64)
        self.is_holdout = np.asarray(is_holdout, dtype=bool)
        if train_mask is None:
            train_mask = ~self.is_holdout
        self.train_mask = np.asarray(train_mask, dtype=bool)
        self.train_groups = np.flatnonzero(self.train_mask)
        if self.train_groups.size == 0:
            raise ValueError("the pair index has no training group")

    @classmethod
    def load(cls, path) -> "PairIndex":
        data = np.load(Path(path))
        return cls(data["centre"], data["pos_b"], data["pos_c"], data["neg_d"], data["neg_e"],
                   data["pos_curvature"], data["neg_curvature"], data["is_holdout"])

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """A deterministic sample of training group positions."""
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        if count >= self.train_groups.size:
            return self.train_groups
        picked = rng.choice(self.train_groups.size, size=count, replace=False)
        return self.train_groups[np.sort(picked)]

    def items_of(self, groups: np.ndarray) -> np.ndarray:
        """The unique items referenced by the sampled groups."""
        if groups.size == 0:
            return np.zeros(0, dtype=np.int64)
        stacked = np.concatenate([self.pos_b[groups], self.pos_c[groups],
                                  self.neg_d[groups], self.neg_e[groups]])
        return np.unique(stacked)

    def rows_of(self, groups: np.ndarray, items: np.ndarray):
        """The row of every group's four items inside the gathered item array."""
        lookup = {int(item): row for row, item in enumerate(items)}
        pos_b = np.array([lookup[int(v)] for v in self.pos_b[groups]], dtype=np.int64)
        pos_c = np.array([lookup[int(v)] for v in self.pos_c[groups]], dtype=np.int64)
        neg_d = np.array([lookup[int(v)] for v in self.neg_d[groups]], dtype=np.int64)
        neg_e = np.array([lookup[int(v)] for v in self.neg_e[groups]], dtype=np.int64)
        return pos_b, pos_c, neg_d, neg_e


def hard_quantized_sum(tokens: torch.Tensor, vq_layers) -> torch.Tensor:
    """The final hard representation ``q = sum_l c_l`` of the given code tokens.

    ``tokens`` is ``[items, levels]`` of the selected code indices; the result is
    ``[items, dim]`` and is differentiable with respect to the codebook embeddings only.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must be [items, levels]")
    if tokens.shape[1] != len(vq_layers):
        raise ValueError("one token column per VQ level is required")
    total = None
    for level, layer in enumerate(vq_layers):
        codebook = layer.get_code_embs()
        selected = F.embedding(tokens[:, level], codebook)
        total = selected if total is None else total + selected
    if total is None:
        raise ValueError("at least one VQ level is required")
    return total


def pair_distances(q: torch.Tensor, pos_b, pos_c, neg_d, neg_e):
    """The two pair distances of every group."""
    pos_b = torch.as_tensor(pos_b, device=q.device, dtype=torch.long)
    pos_c = torch.as_tensor(pos_c, device=q.device, dtype=torch.long)
    neg_d = torch.as_tensor(neg_d, device=q.device, dtype=torch.long)
    neg_e = torch.as_tensor(neg_e, device=q.device, dtype=torch.long)
    if q.ndim != 2:
        raise ValueError("q must be [items, dim]")
    lengths = {int(torch.as_tensor(v).numel()) for v in (pos_b, pos_c, neg_d, neg_e)}
    if len(lengths) != 1:
        raise ValueError("the four pair index arrays must have the same length")
    if int(torch.max(pos_b)) >= q.shape[0] or int(torch.max(pos_c)) >= q.shape[0] \
            or int(torch.max(neg_d)) >= q.shape[0] or int(torch.max(neg_e)) >= q.shape[0]:
        raise IndexError("a pair index points outside the representation")
    d_pos = torch.linalg.vector_norm(q[pos_b] - q[pos_c], dim=-1)
    d_neg = torch.linalg.vector_norm(q[neg_d] - q[neg_e], dim=-1)
    return d_pos, d_neg


def pair_ranking_loss(q: torch.Tensor, pos_b, pos_c, neg_d, neg_e, margin: float = MARGIN) -> torch.Tensor:
    """``mean over the groups of max(0, d_pos - d_neg + margin)``."""
    if not (margin > 0):
        raise ValueError("the margin must be positive")
    d_pos, d_neg = pair_distances(q, pos_b, pos_c, neg_d, neg_e)
    if d_pos.numel() == 0:
        return q.sum() * 0.0
    return torch.relu(d_pos - d_neg + float(margin)).mean()


def pair_ranking_diagnostics(q: torch.Tensor, pos_b, pos_c, neg_d, neg_e,
                              margin: float = MARGIN) -> dict:
    """The loss plus the diagnostics the issue asks the training log to carry."""
    d_pos, d_neg = pair_distances(q, pos_b, pos_c, neg_d, neg_e)
    hinge = torch.relu(d_pos - d_neg + float(margin))
    with torch.no_grad():
        return {
            "pair_loss": float(hinge.mean()) if hinge.numel() else 0.0,
            "ranking_accuracy": float((d_pos < d_neg).float().mean()) if hinge.numel() else float("nan"),
            "margin_satisfaction_rate": float((d_pos + float(margin) <= d_neg).float().mean())
            if hinge.numel() else float("nan"),
            "mean_distance_gap": float((d_neg - d_pos).mean()) if hinge.numel() else float("nan"),
            "mean_positive_distance": float(d_pos.mean()) if hinge.numel() else float("nan"),
            "mean_negative_distance": float(d_neg.mean()) if hinge.numel() else float("nan"),
            "hinge_active_fraction": float((hinge > 0).float().mean()) if hinge.numel() else float("nan"),
        }
