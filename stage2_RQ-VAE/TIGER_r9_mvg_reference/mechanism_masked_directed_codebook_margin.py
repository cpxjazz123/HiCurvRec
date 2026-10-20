"""Masked directed codebook margin -- the reference mechanism of the MVG programme.

This tree exists to give the verification gates something faithful to check. The mechanism
is deliberately small, fully specified, and written so that every part of it can be
recomputed by an independent reference implementation that never imports this file
(``tools/mvg/reference/directed_codebook_margin.py``).

The objective, per level-0 residual pair ``(z_src, z_tgt)`` and the level-0 codebook ``E``:

    A[b, c]   = ||z_src[b] - E[c]||_1            the source's distance to every codebook entry
    B[b, c]   = ||z_tgt[b] - E[c]||_1            the target's distance to every codebook entry
    c*_src    = argmin_c A[b, c]                the source's nearest codebook entry
    c*_tgt    = argmin_c B[b, c]                the target's nearest codebook entry
    r[b]      = max(0, margin + B[b, c*_tgt] - A[b, c*_src])
    p[b]      = softmax_c(-A[b, c] / temperature)
    q[b]      = softmax_c(-B[b, c] / temperature)
    G[i, j]   = ||E[i] - E[j]||_1               the codebook's own relational structure
    s[b]      = sum_{i,j} p[b, i] q[b, j] G[i, j]
    L         = sum_b mask[b] (r[b] + structural_weight * s[b]) / sum_b mask[b]

``r`` is the directed part: the target's nearest entry must be at least ``margin`` closer to
the target than the source's nearest entry is, otherwise the pair is penalised. ``s`` is the
relational part: it measures how far apart the source's and the target's codebook
assignments sit *through the codebook's own geometry*, which is a different quantity from a
per-element distance error.

Three properties are deliberate and are the ones the gates check:

* the loss is **directed** -- swapping the source and the target changes it;
* the loss is **masked** -- a masked-out pair contributes nothing, and the denominator is the
  number of valid pairs, not the batch size;
* the loss reaches three gradient targets -- the source residual, the target residual and the
  codebook -- and none of them is detached.
"""

from __future__ import annotations

import torch

#: Softmax temperature of the codebook assignment.
TEMPERATURE = 0.1
#: How much closer the target's nearest entry must be, in L1 units.
MARGIN = 0.5
#: Weight of the relational term relative to the directed margin term.
STRUCTURAL_WEIGHT = 0.1
#: The residual level the mechanism reads.
LEVEL = 0


def masked_directed_codebook_margin_intermediates(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    codebook: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = TEMPERATURE,
    margin: float = MARGIN,
    structural_weight: float = STRUCTURAL_WEIGHT,
) -> dict:
    """Every quantity the loss is built from, in the production code path.

    Exposed so that the independent reference can be compared on the intermediates and not
    only on the scalar: a mechanism that computes the right number by the wrong route is
    exactly what the M1 gate exists to catch.
    """
    if z_src.dim() != 2 or z_tgt.dim() != 2:
        raise ValueError(f"z_src/z_tgt must be [B, D], got {tuple(z_src.shape)} / {tuple(z_tgt.shape)}")
    if z_src.shape != z_tgt.shape:
        raise ValueError(f"z_src {tuple(z_src.shape)} != z_tgt {tuple(z_tgt.shape)}")
    if codebook.dim() != 2:
        raise ValueError(f"codebook must be [C, D], got {tuple(codebook.shape)}")
    if codebook.shape[1] != z_src.shape[1]:
        raise ValueError(
            f"codebook dim {codebook.shape[1]} != residual dim {z_src.shape[1]}"
        )
    if mask.dim() != 1 or mask.shape[0] != z_src.shape[0]:
        raise ValueError(f"mask must be [B]={z_src.shape[0]}, got {tuple(mask.shape)}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative, got {margin!r}")
    if structural_weight < 0.0:
        raise ValueError(f"structural_weight must be non-negative, got {structural_weight!r}")

    # A[b, c]: the source's L1 distance to every codebook entry.
    src_delta = torch.abs(z_src.unsqueeze(1) - codebook.unsqueeze(0))
    src_dist = src_delta.sum(dim=-1)
    # B[b, c]: the target's L1 distance to every codebook entry.
    tgt_delta = torch.abs(z_tgt.unsqueeze(1) - codebook.unsqueeze(0))
    tgt_dist = tgt_delta.sum(dim=-1)

    src_index = src_dist.argmin(dim=1)
    tgt_index = tgt_dist.argmin(dim=1)
    src_nearest = src_dist.gather(1, src_index.unsqueeze(1)).squeeze(1)
    tgt_nearest = tgt_dist.gather(1, tgt_index.unsqueeze(1)).squeeze(1)

    # The directed margin: the target's entry must beat the source's entry by `margin`.
    margin_term = torch.clamp(margin + tgt_nearest - src_nearest, min=0.0)

    # The relational term: the two assignments, compared through the codebook's geometry.
    src_assign = torch.softmax(-src_dist / temperature, dim=1)
    tgt_assign = torch.softmax(-tgt_dist / temperature, dim=1)
    codebook_delta = torch.abs(codebook.unsqueeze(1) - codebook.unsqueeze(0))
    codebook_geometry = codebook_delta.sum(dim=-1)
    structural_term = (
        src_assign.unsqueeze(2) * tgt_assign.unsqueeze(1) * codebook_geometry.unsqueeze(0)
    ).sum(dim=(1, 2))

    per_pair = margin_term + structural_weight * structural_term

    weights = mask.to(per_pair.dtype)
    valid = weights.sum()
    return {
        "src_dist": src_dist,
        "tgt_dist": tgt_dist,
        "src_index": src_index,
        "tgt_index": tgt_index,
        "src_nearest": src_nearest,
        "tgt_nearest": tgt_nearest,
        "margin_term": margin_term,
        "structural_term": structural_term,
        "per_pair": per_pair,
        "weights": weights,
        "valid": valid,
    }


def masked_directed_codebook_margin(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    codebook: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = TEMPERATURE,
    margin: float = MARGIN,
    structural_weight: float = STRUCTURAL_WEIGHT,
) -> torch.Tensor:
    """The masked directed codebook margin loss.

    ``z_src`` and ``z_tgt`` are ``[B, D]``, ``codebook`` is ``[C, D]`` and ``mask`` is
    ``[B]``. The returned scalar is the masked mean of the per-pair terms.
    """
    parts = masked_directed_codebook_margin_intermediates(
        z_src,
        z_tgt,
        codebook,
        mask,
        temperature=temperature,
        margin=margin,
        structural_weight=structural_weight,
    )
    per_pair = parts["per_pair"]
    weights = parts["weights"]
    valid = parts["valid"]
    if float(valid) <= 0.0:
        # Every pair is masked out: the term contributes nothing, and it must not produce a
        # NaN from a division by zero.
        return per_pair.sum() * 0.0
    return (per_pair * weights).sum() / valid


__all__ = [
    "LEVEL",
    "MARGIN",
    "STRUCTURAL_WEIGHT",
    "TEMPERATURE",
    "masked_directed_codebook_margin",
    "masked_directed_codebook_margin_intermediates",
]
