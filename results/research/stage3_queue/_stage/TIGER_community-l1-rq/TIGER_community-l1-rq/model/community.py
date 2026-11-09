"""The community-guided level-1 cost of the RQ-VAE.

The standalone community experiment (issue 121) learns, from the TRAINING
split's behaviour graph and its AF3 curvature, a soft community distribution
per item and a point per item in the community model's Poincare ball. This
module turns that frozen representation into supervision for the level-1
assignment of the quantiser:

* every level-1 code carries a **community profile** -- the mean of the
  community representations of the items the code has been assigned to,
  accumulated over the epoch and frozen at its end;
* the level-1 cost gains the **community term**: the contract's distance
  between the item's community representation and the code's profile, so a code
  whose items live in the item's behaviour community becomes cheaper for that
  item and a code whose items live elsewhere becomes more expensive.

Two properties keep this a cost and not a second objective:

* the term enters the **cost matrix** that the baseline's assignment rule
  consumes -- the rule itself is untouched;
* the term is **detached** (the community representation is frozen and the
  profile is a statistic of past assignments), so it moves the assignment and
  not the gradient of the losses.

The balance between the two terms is **measured, not chosen**: the community
term is standardised per item, so its spread is one, and the weight is the
baseline cost's own per-item spread, read once from the first epoch that has
a profile.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .poincare_contract import distance as contract_distance, expmap0, logmap0


def community_distance(item: torch.Tensor, profile: torch.Tensor,
                       c: float) -> torch.Tensor:
    """The contract's distance between an item's community point and a code's profile.

    This is the surface the reference contract declares: the Poincare distance
    of the community model, elementwise over the pairs, with the contract's
    artanh clip.
    """
    return contract_distance(item, profile, c)


class CommunitySupervisor:
    """The frozen community representation and the codebook's community profiles."""

    def __init__(self, path: str | Path, n_items: int, codebook_size: int,
                 device: torch.device, curvature: float = 1.0):
        payload = np.load(str(path))
        free = np.asarray(payload["free"], dtype=np.float32)
        ball = np.asarray(payload["ball"], dtype=np.float32)
        if free.shape[0] != n_items:
            raise ValueError(f"the community representation has {free.shape[0]} items, "
                             f"the quantiser has {n_items}")
        self.curvature = float(curvature)
        self.codebook_size = int(codebook_size)
        self.device = device
        self.free = torch.from_numpy(free).to(device)
        self.ball = torch.from_numpy(ball).to(device)
        self.width = int(self.ball.shape[1])
        # the tangent vector of every item, read once: the community
        # representation is frozen, so the tangents never change, and a mean of
        # ball points is not a ball point while a mean of tangents is
        self.tangent = logmap0(self.ball.double(), self.curvature).float()
        # the profiles are accumulated in the tangent space, because a mean of
        # ball points is not a ball point; the tangent mean is mapped back with
        # the contract's origin exponential at the end of the epoch
        self.free_sum = torch.zeros(self.codebook_size, self.free.shape[1],
                                    device=device, dtype=torch.float32)
        self.tangent_sum = torch.zeros(self.codebook_size, self.width,
                                       device=device, dtype=torch.float32)
        self.counts = torch.zeros(self.codebook_size, device=device, dtype=torch.float32)
        self.profiles_ready = False
        self.weight: float | None = None

    # ------------------------------------------------------------------ profile
    @torch.no_grad()
    def observe(self, indices: torch.Tensor, item_ids: torch.Tensor) -> None:
        """Record the community representation of the items a code received."""
        if indices.numel() == 0:
            return
        flat = indices.reshape(-1)
        ids = item_ids.reshape(-1)
        self.free_sum.index_add_(0, flat, self.free[ids])
        self.tangent_sum.index_add_(0, flat, self.tangent[ids])
        self.counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

    @torch.no_grad()
    def finish_epoch(self) -> None:
        """Freeze the epoch's profiles and start the next epoch's accumulation.

        The ranks each see a shard of the items, so the accumulation is summed
        across them first: the profiles are then the same on every rank, and
        the cost of an item does not depend on which rank it happened to land
        on.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for tensor in (self.free_sum, self.tangent_sum, self.counts):
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        counts = self.counts.clamp_min(1.0).unsqueeze(1)
        self.free_profile = self.free_sum / counts
        self.tangent_profile = self.tangent_sum / counts
        self.profiles_ready = True
        self.free_sum.zero_()
        self.tangent_sum.zero_()
        self.counts.zero_()

    # --------------------------------------------------------------------- term
    @torch.no_grad()
    def term(self, item_ids: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
        """The community term for the level-1 cost, standardised per item.

        The term is the community model's own distance -- the Euclidean
        distance in the free branch and the contract's Poincare distance in the
        ball -- between the item and the code's profile, standardised per item
        over the codes, so that its spread is one whatever the codebooks look
        like. It is added to the cost, never multiplied by it.
        """
        if not self.profiles_ready:
            return torch.zeros_like(cost)
        ids = item_ids.reshape(-1)
        item_free = self.free[ids].double()
        item_ball = self.ball[ids].double()
        profile_free = self.free_profile.double()
        profile_ball = expmap0(self.tangent_profile.double(), self.curvature)
        d_free = torch.cdist(item_free, profile_free,
                             compute_mode="donot_use_mm_for_euclid_dist")
        d_ball = contract_distance(item_ball[:, None, :], profile_ball[None, :, :],
                                   self.curvature)
        parts = [_standardise(d_free), _standardise(d_ball)]
        combined = torch.stack(parts, dim=0).mean(dim=0)
        return _standardise(combined).to(cost.dtype)

    @torch.no_grad()
    def calibrate(self, cost: torch.Tensor) -> None:
        """Read the weight once: the baseline cost's own per-item spread.

        The community term is standardised, so its spread is one; the weight
        that puts it on the same footing is the cost's spread over the codes of
        an item. Nothing here is chosen by trying values.
        """
        if self.weight is not None:
            return
        spread = cost.std(dim=1, unbiased=False).mean()
        self.weight = float(spread) if float(spread) > 0 else 1.0


def _standardise(rows: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero mean and unit spread per row, in double precision."""
    values = rows.double()
    mean = values.mean(dim=1, keepdim=True)
    spread = values.std(dim=1, keepdim=True, unbiased=False)
    return (values - mean) / (spread + eps)
