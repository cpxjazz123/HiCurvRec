"""The behaviour ranking objective of BRad-RQ v4.

Three pieces, all read-only with respect to the training data:

* the FROZEN #126 reference (per-item hyperbolic representations, learned on the
  training pairs only) and the user points it induces: the Frechet mean of the
  reference representations of the items in a user's training history;
* the per-level prefix centroids of the CURRENT hard assignment: the Frechet mean
  of the item points of the items that share a prefix;
* the negatives, per level, with the level's own rule: global hard negatives for
  level 1, same-prefix negatives for levels 2 and 3.

The loss is softplus(-Delta) with Delta = d_H^2(u_h, mu_{P_l(i-)}) - d_H^2(u_h,
mu_{P_l(i+)}). With one code for every item the two centroids coincide, Delta is
zero and the term is at its maximum, so a collapse cannot satisfy it.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
REFERENCE_NPZ = REPO / "results" / "research" / "issue126" / "reference" / "reference.npz"
REFERENCE_JSON = REPO / "results" / "research" / "issue126" / "reference" / "reference.json"
TRAIN_PAIRS = REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_s42.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frechet_mean(points: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """The Frechet (Karcher) mean of points in the Poincare ball.

    One fixed-point iteration scheme with a fixed number of steps: the mean is a
    statistic of the assignment, not a tuned parameter.
    """
    if points.numel() == 0:
        raise ValueError("no points to average")
    c = float(curvature)
    mean = points.mean(dim=0)
    for _ in range(5):
        norm = mean.norm().clamp_min(1e-8)
        scale = 2.0 * torch.atanh((norm * c).clamp(max=1 - 1e-7)) / (c * norm)
        target = points.mean(dim=0)
        difference = target - mean
        mean = mean + difference * (1.0 / max(1.0, scale))
        mean = mean / (1.0 + 0.5 * mean.norm() ** 2)  # stay inside the ball
    return mean


def hyperbolic_distance(a: torch.Tensor, b: torch.Tensor,
                         curvature: float = 1.0) -> torch.Tensor:
    """``d^2(a, b)`` in the Poincare ball of curvature ``-curvature``."""
    c = float(curvature)
    difference = a - b
    numerator = (c * difference.norm(dim=-1) ** 2).clamp_min(0.0)
    denominator = ((1.0 - c * a.norm(dim=-1) ** 2).clamp_min(1e-7)
                   * (1.0 - c * b.norm(dim=-1) ** 2).clamp_min(1e-7))
    return numerator / denominator


class BehaviourReference:
    """The frozen #126 reference and the user points it induces."""

    def __init__(self, reference_path: Path = REFERENCE_NPZ,
                 pairs_path: Path = TRAIN_PAIRS):
        self.reference_path = Path(reference_path)
        self.pairs_path = Path(pairs_path)
        payload = np.load(self.reference_path)
        self.source = torch.from_numpy(np.asarray(payload["source"], dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(payload["target"], dtype=np.float32))
        self.sha256 = _sha256(self.reference_path)
        pairs = np.load(self.pairs_path)
        self.anchors = torch.from_numpy(np.asarray(pairs["anchors"], dtype=np.int64))
        self.positives = torch.from_numpy(np.asarray(pairs["positives"], dtype=np.int64))
        self.negatives = torch.from_numpy(np.asarray(pairs["negatives"], dtype=np.int64))

    def user_points(self, histories: list) -> torch.Tensor:
        """``u_h`` for every interaction: the Frechet mean over the history items."""
        points = []
        for history in histories:
            ids = [int(item) for item in history]
            if not ids:
                points.append(torch.zeros(self.source.shape[1], dtype=torch.float32))
                continue
            points.append(frechet_mean(self.source[torch.tensor(ids, dtype=torch.int64)]))
        return torch.stack(points) if points else torch.zeros(0, self.source.shape[1])


class PrefixCentroids:
    """The per-level prefix centroids of the current hard assignment."""

    def __init__(self, tokens: torch.Tensor, points: torch.Tensor,
                 codebook_num: int):
        self.codebook_num = int(codebook_num)
        self.centroids: list[torch.Tensor] = []
        self.sizes: list[dict] = []
        for level in range(self.codebook_num):
            codes = tokens[:, level]
            groups: dict[int, list] = defaultdict(list)
            for item, code in enumerate(codes.tolist()):
                groups[int(code)].append(item)
            table = torch.zeros(self.codebook_num, points.shape[1], dtype=points.dtype)
            sizes: dict[int, int] = {}
            for code, members in groups.items():
                if not members:
                    continue
                index = torch.tensor(members, dtype=torch.int64)
                table[code] = frechet_mean(points[index])
                sizes[code] = len(members)
            self.centroids.append(table)
            self.sizes.append(sizes)

    def centroid(self, level: int, code: int) -> torch.Tensor:
        return self.centroids[level][int(code)]


def level_negatives(tokens: torch.Tensor, positives: torch.Tensor,
                    radius_band: torch.Tensor, popularity: torch.Tensor,
                    radius_buckets: int) -> tuple:
    """The negative pool of each level, by the level's own rule.

    Level 1 takes global hard negatives: the same radius bucket and the same
    popularity band as the positive, a different item. Levels 2 and 3 take the
    items that share the positive's prefix, which is what makes each level carry
    its own granularity.
    """
    popularity_band = popularity
    buckets: dict[tuple, list] = defaultdict(list)
    for item in range(tokens.shape[0]):
        key = (int(radius_band[item]), int(popularity_band[item]))
        buckets[key].append(item)
    level1: list = []
    for item in positives.tolist():
        key = (int(radius_band[item]), int(popularity_band[item]))
        pool = buckets.get(key, [])
        candidates = [candidate for candidate in pool if candidate != item]
        level1.append(candidates)
    level2: dict[int, list] = defaultdict(list)
    level3: dict[tuple, list] = defaultdict(list)
    for item in range(tokens.shape[0]):
        level2[int(tokens[item, 0])].append(item)
        level3[(int(tokens[item, 0]), int(tokens[item, 1]))].append(item)
    return level1, level2, level3


def behaviour_loss(user: torch.Tensor, positive_code: torch.Tensor,
                   negative_code: torch.Tensor, centroids: PrefixCentroids,
                   level: int) -> torch.Tensor:
    """``softplus(-Delta)`` for one level, over the pairs with a legal negative."""
    positive_points = centroids.centroids[level].index_select(
        0, positive_code.to(torch.int64))
    negative_points = centroids.centroids[level].index_select(
        0, negative_code.to(torch.int64))
    positive_distance = hyperbolic_distance(user, positive_points)
    negative_distance = hyperbolic_distance(user, negative_points)
    delta = negative_distance - positive_distance
    return torch.nn.functional.softplus(-delta).mean()
