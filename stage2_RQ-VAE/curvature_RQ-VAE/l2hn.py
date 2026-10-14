"""Within-L1 hard-negative supervision on the q2 prefix, for the curvature tree.

Motivation, measured on this tree's own baseline SID. The behaviour positives that the
quantised SID cannot express are the ones whose L1 codeword matches the anchor's:
7.94% of positives against 0.50% of negatives in the behaviour-sample tree, a sixteen
fold enrichment. The conditional term that was aimed at that slice missed it, because
its negatives came from the batch's other targets and almost none of those share the
anchor's codeword -- the gain leaked onto pairs L1 already separates. This version draws
its negatives from inside the anchor's own L1 codeword, so the only way to score is to
separate items L1 cannot tell apart.

No substrate had to be imported for this. ``RQLayer.forward(return_prefixes=True)``
already returns the per-level running sums, and entry 1 of that list is exactly
``q2 = e1 + e2``. Measured here on the real model:

    d(prefixes[1]) / d(encoder)     8.114662e+04
    d(prefixes[1]) / d(codebooks)  [6.251e+02, 1.253e+04, 0.000e+00]

so the prefix also carries codebook gradient, which this condition must not use. The
straight-through form keeps the forward value exact and routes the gradient to the
encoder alone, verified on the same probe: forward value equal to ``prefixes[1]``,
encoder 4.423151e+03, all three codebooks exactly 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
BASELINE_SID_PATH = (
    REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE/item_sids.json"
)
HARD_NEGATIVE_WIDTH = 32
WEIGHT = 0.005
TEMPERATURE = 0.07


def cumulative_q2(encoded: torch.Tensor, prefixes: list[torch.Tensor]) -> torch.Tensor:
    """``q2 = e1 + e2`` in value, identity gradient into ``encoded``, none into codebooks.

    ``prefixes[1]`` is the accumulation the decoder would receive after two levels, so
    using it keeps the forward value identical to what the quantiser produces while the
    detach keeps the codebook out of the backward pass.
    """
    return encoded + (prefixes[1] - encoded).detach()


def load_l1_partition(path: Path = BASELINE_SID_PATH):
    """Frozen L1 partition of the baseline as (code per item, padded groups, sizes)."""
    with path.open(encoding="utf-8") as stream:
        table = json.load(stream)
    n_items = max(int(key) for key in table) + 1
    code_of_item = np.full(n_items, -1, dtype=np.int64)
    for key, row in table.items():
        code_of_item[int(key)] = int(row[0])
    members: dict[int, list[int]] = {}
    for item, code in enumerate(code_of_item):
        if code >= 0:
            members.setdefault(int(code), []).append(item)
    codes = sorted(members)
    width = max(len(v) for v in members.values())
    groups = np.full((len(codes), width), -1, dtype=np.int64)
    sizes = np.zeros(len(codes), dtype=np.int64)
    for index, code in enumerate(codes):
        row = members[code]
        groups[index, :len(row)] = row
        sizes[index] = len(row)
    return code_of_item, groups, sizes, {code: i for i, code in enumerate(codes)}


def sample_within_l1_negatives(
    anchor_items: torch.Tensor,
    exclude_items: torch.Tensor,
    code_of_item: np.ndarray,
    groups: np.ndarray,
    sizes: np.ndarray,
    code_index: dict[int, int],
    width: int = HARD_NEGATIVE_WIDTH,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``width`` same-codeword items per anchor, never the anchor or its true positive."""
    rows = []
    for anchor, positive in zip(anchor_items.tolist(), exclude_items.tolist()):
        code = int(code_of_item[anchor])
        slot = code_index[code]
        pool = groups[slot, : int(sizes[slot])]
        pool = pool[(pool != anchor) & (pool != positive)]
        if pool.size < width:
            raise RuntimeError(
                f"L1 codeword {code} holds only {pool.size} usable hard negatives, "
                f"below the width of {width}"
            )
        drawn = torch.randperm(pool.size, generator=generator)[:width].numpy()
        rows.append(pool[drawn])
    return torch.from_numpy(np.stack(rows)).to(anchor_items.device)


def within_l1_ranking_loss(
    anchor_q2: torch.Tensor,
    positive_q2: torch.Tensor,
    negative_q2: torch.Tensor,
    *,
    weight: float = WEIGHT,
) -> torch.Tensor:
    """Contrastive loss with the positive against same-codeword negatives only.

    ``negative_q2`` is [anchors, width, dim]; the positive sits at logit index 0.
    cdist rather than a hand-rolled pow/sum/sqrt: two different items can have
    identical q2 -- that is what sharing both codewords means -- and sqrt's derivative
    diverges at zero.
    """
    anchor = anchor_q2[:, None, :]
    candidates = torch.cat([positive_q2[:, None, :], negative_q2], dim=1)
    logits = -torch.cdist(anchor, candidates).squeeze(1) / TEMPERATURE
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return float(weight) * F.cross_entropy(logits, labels)
