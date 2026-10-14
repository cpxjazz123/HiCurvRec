"""Restrict one RQ level to a fixed subset of its codebook (crowding by capacity).

Common layer of the crowding programme: this file is copied verbatim into every
crowding tree (``TIGER_CROWDING_CTRL`` and the ten mechanism trees) and no
mechanism tree may change it. Only the L1 capacity differs from the baseline:
L1 keeps ``CROWDING_L1_ALLOWED = 128`` selectable codewords out of its 256, so
twice as many items have to share the same L1 prefix.
"""
from types import MethodType

import torch


def install_level_capacity(layer, allowed: int) -> None:
    """Keep only the first ``allowed`` codewords selectable on ``layer``.

    Wraps whatever ``_distances`` is currently installed, so it composes with the
    geodesic assignment patch in scripts/hyp_losses.py. Distance to a disabled
    codeword becomes +inf, which both argmin and Sinkhorn treat as unreachable.
    Codebook parameters keep their original shape: the Stage3 token vocabulary is
    unchanged.

    Installation must happen *after* any other ``_distances`` patch (geodesic
    assignment, fan-out, radial factors, usage redistribution, ...) because this
    wrapper captures the function that is installed at that moment. Disabled
    codewords are the trailing ``n_embed - allowed`` indices, and the same subset
    for every tree: KMeans initialises the L1 codebook identically under seed 42.
    """
    if getattr(layer, "_capacity_mask_active", False):
        return
    original = layer._distances
    keep = torch.arange(int(allowed), device=layer.get_code_embs().device)

    def distances(self, latent):
        matrix = original(latent)
        mask = torch.ones(matrix.shape[-1], dtype=torch.bool, device=matrix.device)
        mask[keep] = False
        return matrix.masked_fill(mask, float("inf"))

    layer._distances = MethodType(distances, layer)
    layer._capacity_mask_active = True
    layer._capacity_allowed = int(allowed)
