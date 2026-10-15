"""M9: Hyperbolic Capacity Redistribution.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
Levels 2 and 3 keep their codewords and their total capacity, but the *load* is
redistributed: a codeword that has recently absorbed more of the stream is made
slightly less attractive, so the level's usage spread widens instead of the first
codewords winning everything. Level 1 is untouched -- the crowded capacity
constraint is the condition under test and must not be relaxed.

Math
----
Per level, with an EMA of the codeword usage over the training stream,

    usage_i <- decay * usage_i + (1 - decay) * count_i(batch)
    d(x, i) <- d(x, i) + strength * ln(1 + usage_i)

``d`` is whatever distance the level already uses -- the control's assignment
metric. This mechanism adds a load term to it and nothing else: no new metric, no
new loss, no new codeword. Whether the fixed strength moves any assignment at that
scale is a property of the experiment and is reported as measured (see
``penalty_scale``), not adjusted after the fact. The usage histogram is the *argmin* usage of the penalised matrix, i.e.
exactly the codewords the assignment is about to select, and it is updated only in
training mode. As in the control's own unused-code accounting, the histogram is a
per-rank statistic (no ``all_reduce``), so no extra synchronisation is added.

Composition with the shared layer
---------------------------------
Levels 2 and 3 only; ``scripts/capacity_mask.py`` still owns level 1, so the
128-codeword crowding condition is untouched by construction.

What is completely unchanged relative to the control
---------------------------------------------------
L1 capacity mask and diagnostics, level-1 assignment, all losses, pair sampling,
codeword count and codebook sizes, optimiser, schedule, seed, epochs, evaluation
interval.

Fixed parameters (no sweep): ``decay = 0.9``, ``strength = 0.5``.
"""
from __future__ import annotations

from types import MethodType

import torch

USAGE_DECAY = 0.9
USAGE_STRENGTH = 0.5


def penalty_scale(matrix: torch.Tensor, usage: torch.Tensor,
                  strength: float = USAGE_STRENGTH) -> dict:
    """Measure the load term against the distance margin it has to overcome.

    ``margin`` is the gap between each row's best and second-best base distance;
    ``penalty`` is the largest load term in play. When the penalty is far below the
    margin the term can only break near-ties, which is what the report must say.
    """
    top2 = matrix.topk(2, dim=-1, largest=False).values
    margin = (top2[:, 1] - top2[:, 0]).median()
    term = float(strength) * torch.log1p(usage)
    return {
        "median_margin": float(margin),
        "max_penalty": float(term.max()),
        "ratio": float(term.max() / margin.clamp_min(1e-12)),
    }


def install_usage_redistribution(layer, decay: float = USAGE_DECAY,
                                 strength: float = USAGE_STRENGTH) -> None:
    """Add a load penalty to the layer's distance and keep its usage EMA current.

    Order of operations, which used to be wrong:

    * before: the counter was updated inside ``distances`` from
      ``penalised.argmin(-1)``, while the level's real index came from
      ``_indices`` -- Sinkhorn on the last level during training. The histogram
      therefore described codes that were not chosen, and on the level where
      Sinkhorn is active the two disagree on most rows.
    * now: ``distances`` only *shapes* the matrix, and a wrapper around
      ``_indices`` counts the indices the level actually assigned. The wrapper
      delegates to the original ``_indices``, so Sinkhorn still decides; nothing
      here replaces that decision with an argmin.

    The counter is registered as a persistent buffer, so ``state_dict()`` carries
    it and a reloaded checkpoint continues the same load history instead of
    restarting from zero usage.
    """
    if getattr(layer, "_usage_redistribution_active", False):
        return
    n_embed = int(layer.n_embed)
    original = layer._distances
    original_indices = layer._indices
    if not hasattr(layer, "_usage_ema"):
        layer.register_buffer(
            "_usage_ema",
            torch.zeros(n_embed, dtype=torch.float32,
                        device=layer.get_code_embs().device),
        )
    usage = layer._usage_ema

    def distances(self, latent):
        # Pre-registered operator: the load term is added to whatever distance this
        # level already uses (the control's assignment metric), with the fixed
        # strength. The term's magnitude relative to that metric's own margin is
        # measured in ``penalty_scale`` and reported, not tuned.
        matrix = original(latent)
        return matrix + float(strength) * torch.log1p(usage).unsqueeze(0)

    def indices(self, distance_matrix, infer_use_sk):
        chosen = original_indices(distance_matrix, infer_use_sk)
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(
                    chosen.reshape(-1).to(torch.long), minlength=n_embed
                ).to(usage.dtype)
                # Decay every slot, not only the ones this batch touched: the EMA
                # is over time, so an unused codeword must fade as well.
                usage.mul_(float(decay)).add_(counts, alpha=1.0 - float(decay))
        return chosen

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)
    layer._usage_redistribution_active = True
    layer._usage_decay = float(decay)
    layer._usage_strength = float(strength)
    layer.usage_ema = usage
