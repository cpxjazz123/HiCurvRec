"""HiD-VAE's Uniqueness Loss, reproduced on the RQ-VAE (issue #175).

``L_unique = mean over pairs P of relu(cos(z_i, z_j) - m)``, where P holds the pairs *inside one training batch* whose
three-level codes are identical and whose items differ, and z is the encoder's pre-quantisation continuous
representation. The code selection is detached, so the gradient reaches the encoder only through the cosine.

Direction note required by the issue: in this formula a larger m is *more permissive* (a pair whose cosine is already
below m contributes nothing), which is the opposite of what the paper's prose says. The formula is implemented as written
and the contradiction is recorded here rather than silently "fixed".
"""
from __future__ import annotations

import torch

#: beta and m from the paper (arXiv:2508.04618v2); not tuned.
BETA_UNIQUE = 2.0
MARGIN = 0.9

#: Switches: OFF must reproduce the baseline loss bit for bit; the random source must still exercise the code path.
UNIQUE_OFF = False
PAIR_SOURCE_OFF = False

#: Bookkeeping for the run report: how many colliding pairs a batch had and how many of them were above the margin.
LAST_STATS: dict = {}


def collision_pairs(tokens: torch.Tensor) -> list[torch.Tensor]:
    """Rows of one batch grouped by their three-level code; groups with two or more rows are the pair sources."""
    if tokens.dim() != 2 or tokens.shape[1] < 3:
        return []
    codes = tokens[:, :3].detach()
    keys = codes[:, 0].to(torch.int64) * 65536 + codes[:, 1].to(torch.int64) * 256 + codes[:, 2].to(torch.int64)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    boundaries = torch.nonzero(sorted_keys[1:] != sorted_keys[:-1], as_tuple=False).flatten() + 1
    groups = torch.split(order, torch.diff(torch.cat([torch.zeros(1, dtype=torch.long, device=keys.device),
                                                      boundaries,
                                                      torch.tensor([order.numel()], device=keys.device)])).tolist())
    return [group for group in groups if group.numel() >= 2]


def uniqueness_loss(z: torch.Tensor, tokens: torch.Tensor, margin: float | None = None,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """``mean relu(cos(z_i, z_j) - m)`` over the colliding pairs of this batch."""
    global LAST_STATS
    if UNIQUE_OFF or z.shape[0] < 2:
        LAST_STATS = {"pairs": 0, "above_margin": 0, "loss": 0.0, "note": "UNIQUE_OFF"}
        return torch.zeros((), dtype=z.dtype, device=z.device)
    margin = MARGIN if margin is None else float(margin)
    if PAIR_SOURCE_OFF:
        # the wiring switch: same code path, but the pairs are random items rather than code collisions
        rows = z.shape[0]
        index = torch.randint(0, rows, (max(rows // 2, 1), 2), generator=generator, device=z.device)
        groups = [pair for pair in index if pair[0] != pair[1]]
    else:
        groups = collision_pairs(tokens)
    if not groups:
        LAST_STATS = {"pairs": 0, "above_margin": 0, "loss": 0.0, "note": "no colliding pair in this batch"}
        return torch.zeros((), dtype=z.dtype, device=z.device)
    unit = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    total = torch.zeros((), dtype=z.dtype, device=z.device)
    count = 0
    above = 0
    for group in groups:
        if group.numel() == 2 and group.dim() == 1 and not torch.is_tensor(group.numel()):
            pass
        members = group if group.dim() == 1 else group.flatten()
        vect = unit[members]
        cosine = vect @ vect.t()
        mask = torch.ones_like(cosine, dtype=torch.bool).triu(diagonal=1)
        entries = cosine[mask]
        if entries.numel() == 0:
            continue
        total = total + torch.clamp(entries - margin, min=0.0).sum()
        count += int(entries.numel())
        above += int((entries > margin).sum())
    if count == 0:
        LAST_STATS = {"pairs": 0, "above_margin": 0, "loss": 0.0, "note": "empty pair set"}
        return torch.zeros((), dtype=z.dtype, device=z.device)
    loss = total / float(count)
    LAST_STATS = {"pairs": count, "above_margin": above, "loss": float(loss.detach())}
    return loss
