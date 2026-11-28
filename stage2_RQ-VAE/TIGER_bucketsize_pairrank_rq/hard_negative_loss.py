"""Issue #287 round 6: the hard-negative same-anchor ranking in the item representation space.

The measured obstacle of rounds 1-5 is a monotone trade-off between the L1 positive sharing rate P
and the L1 negative sharing probability N: every mechanism that pulls the behaviourally related items
together also lets the unrelated items into the same L1 bucket, because the ranking never pushes the
items that *already* share a code away.  Two measured facts make the old form unable to fix that:

* the distance between the **selected L1 codewords** of two items that share their L1 code is
  exactly zero, so a codeword-distance hinge between them has **no gradient** (the codebook gradient
  is proportional to ``c_anchor - c_negative = 0``): a "push the false positive away" term written on
  the codewords cannot exist;
* the module that decides *which* items share an L1 code is the **encoder**, and the codeword form
  never reaches it.

Round 6 therefore states the same-anchor ranking on the **item representation** (the encoder's
bottleneck), which is the space in which "keep these two items apart" is a well-posed instruction:

* the **positive** is round 2's frozen, curvature-screened behavioural partner;
* the **negative** is a **measured false positive**: an item that already shares the anchor's L1 code
  and has no observed training-graph relation with it.  The negatives are mined from the frozen
  TIGER export before the run and refreshed from the current assignment during the run.

The loss stays the frozen ranking hinge ``mean relu(d_pos - d_neg + m)`` with the frozen margin rule
(the median of ``d_negative - d_positive`` on the frozen TIGER checkpoint, computed once before the
run).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#: the frozen margin, calibrated once on the frozen TIGER checkpoint (MARGIN_287_R6.json)
MARGIN = 0.04035949200819747

#: the VQ level whose code sharing the mechanism protects (0 = L1).
LEVEL = 0


class HardNegativeIndex:
    """The frozen hard-negative index (the negatives are refreshed from the current partition)."""

    def __init__(self, anchor, pos, neg, is_holdout, nbr_ptr, nbr_items, bucket_size=None):
        self.anchor = np.asarray(anchor, dtype=np.int64)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.neg = np.asarray(neg, dtype=np.int64)
        self.is_holdout = np.asarray(is_holdout, dtype=bool)
        self.nbr_ptr = np.asarray(nbr_ptr, dtype=np.int64)
        self.nbr_items = np.asarray(nbr_items, dtype=np.int64)
        if not (len(self.anchor) == len(self.pos) == len(self.neg) == len(self.is_holdout)):
            raise ValueError("the index columns must have the same length")
        if bucket_size is None:
            bucket_size = np.ones(len(self.anchor), dtype=np.int64)
        self.bucket_size = np.asarray(bucket_size, dtype=np.int64)
        if self.bucket_size.shape != self.anchor.shape:
            raise ValueError("the bucket sizes must have one entry per group")
        if int(self.bucket_size.min()) < 1:
            raise ValueError("every group needs a positive bucket size")
        if int(self.neg.min()) < 0:
            raise ValueError("every group needs a mined negative")
        self.train_mask = ~self.is_holdout
        self.train_groups = np.flatnonzero(self.train_mask)
        if self.train_groups.size == 0:
            raise ValueError("the pair index has no training group")

    @classmethod
    def load(cls, path) -> "HardNegativeIndex":
        data = np.load(Path(path))
        bucket = data["bucket_size"] if "bucket_size" in data.files else None
        return cls(data["anchor"], data["pos"], data["neg"], data["is_holdout"],
                   data["nbr_ptr"], data["nbr_items"], bucket)

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        if count >= self.train_groups.size:
            return self.train_groups
        picked = rng.choice(self.train_groups.size, size=count, replace=False)
        return self.train_groups[np.sort(picked)]

    def items_of(self, groups: np.ndarray) -> np.ndarray:
        if groups.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate([self.anchor[groups], self.pos[groups], self.neg[groups]]))

    def rows_of(self, groups: np.ndarray, items: np.ndarray):
        lookup = {int(item): row for row, item in enumerate(items)}
        anchor = np.array([lookup[int(v)] for v in self.anchor[groups]], dtype=np.int64)
        pos = np.array([lookup[int(v)] for v in self.pos[groups]], dtype=np.int64)
        neg = np.array([lookup[int(v)] for v in self.neg[groups]], dtype=np.int64)
        return anchor, pos, neg

    def neighbours_of(self, item: int) -> np.ndarray:
        item = int(item)
        return self.nbr_items[self.nbr_ptr[item]:self.nbr_ptr[item + 1]]

    def set_negatives(self, neg: np.ndarray) -> None:
        neg = np.asarray(neg, dtype=np.int64)
        if neg.shape != self.neg.shape:
            raise ValueError("the refreshed negatives must have the index's shape")
        if int(neg.min()) < 0:
            raise ValueError("every group needs a mined negative")
        self.neg = neg

    def set_bucket_size(self, bucket_size: np.ndarray) -> None:
        """Replace the per-group L1 bucket sizes (round 7's separation weight)."""
        bucket_size = np.asarray(bucket_size, dtype=np.int64)
        if bucket_size.shape != self.bucket_size.shape:
            raise ValueError("the refreshed bucket sizes must have the index's shape")
        if int(bucket_size.min()) < 1:
            raise ValueError("every group needs a positive bucket size")
        self.bucket_size = bucket_size


def mine_hard_negatives(index: HardNegativeIndex, codes: np.ndarray,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Re-mine the negatives and their bucket weights from the current L1 partition.

    ``codes[item]`` is the item's current L1 code.  The candidates of an anchor are the items that
    share the anchor's code and are **not** one of the anchor's direct neighbours (the observed
    training-graph relation), so the negative is always a measured false positive.  A group without a
    candidate keeps its previous negative (the group is never dropped mid-run).  The weight of a
    group is the size of the anchor's bucket: the separation targets the buckets that dominate N.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if codes.min() < 0:
        raise ValueError("the codes must be non-negative")
    by_code: dict[int, list[int]] = {}
    for item, code in enumerate(codes.tolist()):
        by_code.setdefault(int(code), []).append(int(item))
    counts = np.bincount(codes, minlength=max(1, int(codes.max()) + 1)).astype(np.int64)
    bucket_size = counts[codes[index.anchor]].astype(np.int64)
    negatives = index.neg.copy()
    pools: list[int] = []
    refreshed = 0
    for row, anchor in enumerate(index.anchor.tolist()):
        neighbours = set(index.neighbours_of(anchor).tolist())
        neighbours.add(int(index.pos[row]))
        candidates = [item for item in by_code.get(int(codes[anchor]), [])
                      if item != int(anchor) and item not in neighbours]
        pools.append(len(candidates))
        if not candidates:
            continue
        negatives[row] = int(rng.choice(np.asarray(sorted(candidates), dtype=np.int64)))
        refreshed += 1
    stats = {'groups': int(len(index.anchor)), 'refreshed': int(refreshed),
             'mean_pool': round(float(np.mean(pools)), 3) if pools else 0.0,
             'median_pool': float(np.median(pools)) if pools else 0.0,
             'mean_bucket_size': round(float(bucket_size.mean()), 3),
             'max_bucket_size': int(bucket_size.max())}
    return negatives, bucket_size, stats


def encoder_distances(z: torch.Tensor, anchor, pos, neg):
    """The positive and negative distances of every group (both share the anchor)."""
    anchor = torch.as_tensor(anchor, device=z.device, dtype=torch.long)
    pos = torch.as_tensor(pos, device=z.device, dtype=torch.long)
    neg = torch.as_tensor(neg, device=z.device, dtype=torch.long)
    if z.ndim != 2:
        raise ValueError("z must be [items, dim]")
    lengths = {int(torch.as_tensor(v).numel()) for v in (anchor, pos, neg)}
    if len(lengths) != 1:
        raise ValueError("the three index arrays must have the same length")
    for v in (anchor, pos, neg):
        if int(torch.max(v)) >= z.shape[0]:
            raise IndexError("a pair index points outside the representation")
    d_pos = torch.linalg.vector_norm(z[pos] - z[anchor], dim=-1)
    d_neg = torch.linalg.vector_norm(z[neg] - z[anchor], dim=-1)
    return d_pos, d_neg


def hard_negative_loss(z: torch.Tensor, anchor, pos, neg, margin: float = MARGIN,
                       weight=None) -> torch.Tensor:
    """``sum_g w_g * max(0, d_pos - d_neg + margin) / sum_g w_g``.

    Without ``weight`` the reduction is the plain mean (round 6's form).  With the bucket sizes as
    the weights, a group inside a large L1 bucket counts more: N is dominated by those buckets, so the
    separation spends its budget where the mass is.
    """
    if not (margin > 0):
        raise ValueError("the margin must be positive")
    d_pos, d_neg = encoder_distances(z, anchor, pos, neg)
    if d_pos.numel() == 0:
        return z.sum() * 0.0
    hinge = torch.relu(d_pos - d_neg + float(margin))
    if weight is None:
        return hinge.mean()
    weight = torch.as_tensor(weight, device=hinge.device, dtype=hinge.dtype)
    if weight.shape != hinge.shape:
        raise ValueError("the weights must have one entry per group")
    if float(weight.sum()) <= 0.0:
        raise ValueError("the weights must have a positive sum")
    return (hinge * weight).sum() / weight.sum()


def hard_negative_diagnostics(z: torch.Tensor, anchor, pos, neg, margin: float = MARGIN,
                              weight=None) -> dict:
    d_pos, d_neg = encoder_distances(z, anchor, pos, neg)
    hinge = torch.relu(d_pos - d_neg + float(margin))
    with torch.no_grad():
        if weight is not None and hinge.numel():
            w = torch.as_tensor(weight, device=hinge.device, dtype=hinge.dtype)
            loss_value = float((hinge * w).sum() / w.sum())
        else:
            loss_value = float(hinge.mean()) if hinge.numel() else 0.0
        return {
            "pair_loss": loss_value,
            "ranking_accuracy": float((d_pos < d_neg).float().mean()) if hinge.numel() else float("nan"),
            "margin_satisfaction_rate": float((d_pos + float(margin) <= d_neg).float().mean())
            if hinge.numel() else float("nan"),
            "mean_distance_gap": float((d_neg - d_pos).mean()) if hinge.numel() else float("nan"),
            "mean_positive_distance": float(d_pos.mean()) if d_pos.numel() else float("nan"),
            "mean_negative_distance": float(d_neg.mean()) if d_neg.numel() else float("nan"),
            "hinge_active_fraction": float((hinge > 0).float().mean()) if hinge.numel() else float("nan"),
        }
