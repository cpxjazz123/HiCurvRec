"""The community partition as per-step pair supervision.

The standalone community experiment (issue 121) partitions the catalogue into
64 communities from the TRAINING split's behaviour graph and its AF3 curvature.
This module turns that frozen partition into the supervision the encoder sees:

* every step, every item of the batch is an **anchor**;
* its **positive** is another item drawn from the same community, freshly drawn
  on every step, so the encoder cannot memorise a fixed positive set;
* the negatives are the other positives of the step, and a candidate that sits
  in the anchor's own community is **masked out**: it is not a negative but an
  unlabelled positive, and treating it as a negative would teach the encoder to
  separate items the partition says belong together;
* the loss is the contrastive cross-entropy over the latents entering the
  quantiser, so the quantiser's own structure and codebooks are untouched.

The scale of the term is measured, not chosen: the temperature is the median
latent distance of the first step, and the weight is the ratio that makes the
term equal to the reconstruction loss at that step. Neither is swept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class CommunityPartition:
    """The frozen hard partition, with a per-step positive sampler."""

    def __init__(self, path: str | Path, n_items: int, seed: int = 42):
        payload = np.load(str(path))
        # the artifact is a plain array (the assignment the community experiment
        # saved); a keyed archive is accepted so the loader survives a re-export
        if isinstance(payload, np.ndarray):
            probabilities = payload
        else:
            probabilities = payload["probabilities"]
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.shape[0] != n_items:
            raise ValueError(f"the partition covers {probabilities.shape[0]} items, "
                             f"the quantiser has {n_items}")
        self.labels = probabilities.argmax(axis=1).astype(np.int64)
        self.members = [np.nonzero(self.labels == community)[0]
                        for community in range(probabilities.shape[1])]
        if any(member.size == 0 for member in self.members):
            raise ValueError("the partition has an empty community")
        self.n_communities = int(probabilities.shape[1])
        self.generator = torch.Generator().manual_seed(int(seed))

    def positive(self, anchors: torch.Tensor) -> torch.Tensor:
        """Another item of each anchor's community, drawn fresh every call."""
        anchors_cpu = anchors.detach().to("cpu")
        labels = self.labels[anchors_cpu.numpy()]
        out = np.empty(labels.size, dtype=np.int64)
        for index, community in enumerate(labels):
            member = self.members[int(community)]
            if member.size == 1:
                out[index] = int(member[0])
                continue
            # a member other than the anchor itself
            choice = int(torch.randint(member.size - 1, (1,),
                                       generator=self.generator).item())
            candidate = int(member[choice])
            if candidate == int(anchors_cpu[index]):
                candidate = int(member[-1] if choice == 0 else member[0])
            out[index] = candidate
        return torch.from_numpy(out).to(anchors.device)

    def same_community(self, anchors: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """The (anchors x candidates) mask of pairs the partition puts together."""
        left = torch.from_numpy(self.labels)[anchors.detach().to("cpu")].to(anchors.device)
        right = torch.from_numpy(self.labels)[candidates.detach().to("cpu")].to(candidates.device)
        return left.unsqueeze(1) == right.unsqueeze(0)


def contrastive_loss(logits: torch.Tensor, same_community: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the step's candidates, with the unlabelled positives masked out.

    The target of row i is column i, the drawn positive. Every other column
    that belongs to the anchor's own community is neither a positive nor a
    negative, so it is removed from the denominator instead of being pushed
    away: the softmax is taken over the negatives and the target only.
    """
    size = int(logits.shape[0])
    target = torch.arange(size, device=logits.device)
    excluded = same_community.clone()
    excluded[target, target] = False
    logits = logits.masked_fill(excluded, float("-inf"))
    total = torch.logsumexp(logits, dim=1)
    picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    return (total - picked).mean()


def pair_logit(source: torch.Tensor, positive: torch.Tensor,
               temperature: float) -> torch.Tensor:
    """The kernel the contract declares: the row-wise distance over the temperature."""
    return -pair_distance(source, positive).sqrt() / temperature


def logit_matrix(source: torch.Tensor, positive: torch.Tensor,
                 temperature: float) -> torch.Tensor:
    """The step's logits: every anchor against every drawn positive."""
    return -torch.cdist(source, positive, p=2) / temperature


def pair_distance(source: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """The squared row-wise latent distance of the community contrast."""
    difference = source - positive
    return (difference * difference).sum(dim=1)


def measure_temperature(latent: torch.Tensor) -> float:
    """The median pairwise latent distance: the term's own scale, measured once."""
    with torch.no_grad():
        distances = torch.cdist(latent.detach(), latent.detach(), p=2)
        size = distances.shape[0]
        if size < 2:
            return 1.0
        values = distances[torch.triu(torch.ones(size, size, dtype=torch.bool,
                                                 device=distances.device), diagonal=1)]
        median = float(values.median())
    return median if median > 0 else 1.0
