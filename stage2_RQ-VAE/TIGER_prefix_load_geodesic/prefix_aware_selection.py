"""M10: Hyperbolic Prefix-Aware Quantization.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
Levels 2 and 3 choose their codeword with knowledge of the prefix that has already
been generated: the load counter is kept *per prefix*, not per level, so two items
that share ``l1`` (or ``l1 l2``) do not both pile onto the same child codeword while
the sibling children of that prefix sit idle. This is the prefix-local version of
M9's global load balancing, and it is the level where L1 crowding actually
propagates: one crowded prefix contains many items that then compete for one child.

Math
----
For a row whose generated prefix key is ``k`` (``k = l1`` at level 2,
``k = l1 * 256 + l2`` at level 3) and codeword ``i``:

    counts[k, i] <- decay * counts[k, i] + (1 - decay) * count_{k,i}(batch)
    d(x, i)      <- d(x, i) + strength * ln(1 + counts[k, i])

``d`` is whatever distance the level already uses -- the control's assignment
metric. The mechanism adds the prefix-local load term to it and nothing else;
whether the fixed strength moves any assignment at that scale is reported as
measured rather than tuned.

Level 1 is untouched (the crowded condition itself), and the counter is updated
from the argmin of the penalised matrix in training mode only, per rank, exactly
like the control's unused-code accounting.

Composition with the shared layer
---------------------------------
Installed on levels 2 and 3 only, so ``scripts/capacity_mask.py`` keeps owning
level 1 and the 128-codeword capacity is untouched. The prefix key is handed down
by ``RQLayer.forward`` (the same plumbing M1 uses for its parent conditioning),
and the residual trace is delegated to this module so the behaviour term sees the
same assignment path the mechanism changes.

What is completely unchanged relative to the control
---------------------------------------------------
L1 capacity mask and diagnostics, level-1 assignment, all losses, pair sampling,
codeword count and codebook sizes, optimiser, schedule, seed, epochs, evaluation
interval, Stage 3 model.

Fixed parameters (no sweep): ``decay = 0.9``, ``strength = 0.5``.
"""
from __future__ import annotations

from types import MethodType

import torch

PREFIX_DECAY = 0.9
PREFIX_STRENGTH = 0.5
PREFIX_BASE = 256


def install_prefix_aware(layer, level: int, decay: float = PREFIX_DECAY,
                         strength: float = PREFIX_STRENGTH,
                         base: int = PREFIX_BASE) -> None:
    """Per-prefix load penalty for ``layer`` (level 2 or 3).

    ``base`` is the codebook size used to pack the prefix into one integer key, so
    level 3's key is ``l1 * base + l2``. The counter table is allocated once per
    layer: ``base ** (level - 1)`` prefixes x ``n_embed`` codewords.

    Order of operations, which used to be wrong in two ways:

    * the counter was updated from ``penalised.argmin(-1)`` inside ``distances``,
      while the level's real index came from ``_indices`` (Sinkhorn on the last
      level while training), so the histogram described codes that were not
      chosen;
    * the EMA was written as ``flat[unique] = decay * flat[unique] + ...``, which
      only touches the slots this batch hit. Slots the batch never selected kept
      their old value forever, so a stale prefix load could dominate the penalty
      for the rest of training. The decay now applies to the whole table every
      step and the batch counts are added on top.

    The table is a persistent buffer, so ``state_dict()`` carries it and a
    reloaded checkpoint continues the same load history. ``_indices`` is wrapped,
    not replaced: the wrapper counts and delegates, so Sinkhorn still decides.
    """
    if getattr(layer, "_prefix_aware_active", False):
        return
    n_embed = int(layer.n_embed)
    n_prefix = int(base) ** max(int(level) - 1, 1)
    if not hasattr(layer, "_prefix_counts"):
        layer.register_buffer(
            "_prefix_counts",
            torch.zeros(n_prefix, n_embed, dtype=torch.float32,
                        device=layer.get_code_embs().device),
        )
    counts = layer._prefix_counts
    flat = counts.view(-1)
    original = layer._distances
    original_indices = layer._indices

    def distances(self, latent):
        # Pre-registered operator: the prefix-local load term is added to whatever
        # distance this level already uses, with the fixed strength.
        matrix = original(latent)
        keys = getattr(self, "_prefix_key", None)
        if keys is None:
            return matrix
        return matrix + float(strength) * torch.log1p(counts[keys.to(torch.long)])

    def indices(self, distance_matrix, infer_use_sk):
        chosen = original_indices(distance_matrix, infer_use_sk)
        keys = getattr(self, "_prefix_key", None)
        if self.training and keys is not None:
            with torch.no_grad():
                keys = keys.to(torch.long)
                slot = keys.reshape(-1) * n_embed + chosen.reshape(-1).to(torch.long)
                unique, inverse = torch.unique(slot, return_inverse=True)
                batch_counts = torch.bincount(
                    inverse, minlength=unique.numel()
                ).to(flat.dtype)
                # Decay every slot first, then add this batch's counts, so a slot
                # the batch never hits still fades with time.
                flat.mul_(float(decay))
                flat[unique] += (1.0 - float(decay)) * batch_counts
        return chosen

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)
    layer._prefix_aware_active = True
    layer._prefix_aware_level = int(level)
    layer._prefix_aware_strength = float(strength)
    layer._prefix_aware_decay = float(decay)
    layer.prefix_counts = counts


def _trace_with_prefix_keys(module, embeddings: torch.Tensor, with_tokens: bool = False):
    """Residual trace that hands each level its running prefix key, like ``RQLayer``."""
    residual = module.encoder(embeddings)
    traces = [residual]
    tokens = []
    key = None
    for level, vq_layer in enumerate(module.rq.vq_layers):
        quant, _, _, indices = vq_layer(residual, parent_indices=key)
        tokens.append(indices)
        residual = residual - quant
        traces.append(residual)
        key = indices if key is None else key * int(module.rq.codebook_sizes[level]) + indices
    stacked = torch.stack(traces[: module.rq.codebook_num], dim=0)
    if with_tokens:
        return stacked, torch.stack(tokens, dim=0)
    return stacked
