"""The measurement adapter the mechanism contract names.

This is not a test: it exposes the two observations the contract's templates compare
— the radius the behaviour statistic produces, and the level-1 assignment cost the
model computes for a given radius. The templates decide what counts; this file only
measures, and it may not compute an expectation.
"""
from __future__ import annotations

import torch

from types import SimpleNamespace

from model.model import RQVAE
from model.behaviour_radius import rank_radii


def _config():
    return SimpleNamespace(hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16),
                           codebook_dim=8, dropout=0.0, beta=0.25, vq_type="vq",
                           ema_decay=0.99, fix_code_embs=False, sk_epsilon=0.003,
                           sk_iters=5, frechet_codebook=False, use_behaviour=True,
                           use_curvature=True, frechet_eta=1.0, radial=True)


def radius_for(scale: float) -> torch.Tensor:
    """The radii of a small item population whose behaviour scores are `scale`."""
    values = torch.linspace(0.1, float(scale) * 3.0, 24)
    return rank_radii(values)


def observe_cost(radius: torch.Tensor) -> torch.Tensor:
    """The level-1 assignment costs of one item at the given radii."""
    torch.manual_seed(0)
    model = RQVAE(_config(), in_dim=16)
    embeddings = torch.randn(24, 16)
    with torch.no_grad():
        model.init_codebook(embeddings)
        encoded = model.encoder(embeddings)
        scaled = model.scale_by_radius(encoded, radius)
        layer = model.rq.vq_layers[0]
        return layer.radial_cost(scaled)
