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
    """The radii of a small item population whose behaviour scores are `scale`.

    The mechanism uses the RANK of the statistic, so a change of magnitude alone
    changes nothing; the perturbation therefore reorders the scores, which is what
    a different behaviour distribution does.
    """
    values = torch.linspace(0.1, 3.0, 24)
    if float(scale) >= 1.0:
        values = values.flip(0)
    return rank_radii(values)


def _fixed_codebook(layer) -> None:
    """A deterministic codebook.

    The KMeans start is not reproducible across calls (the cluster count moved
    between two runs of the same seed), and a measurement whose noise is larger
    than the effect it looks for proves nothing.
    """
    with torch.no_grad():
        count = int(layer.get_code_embs().shape[0])
        dim = int(layer.get_code_embs().shape[1])
        grid = torch.linspace(0.05, 0.85, count)
        table = torch.zeros(count, dim)
        table[:, 0] = grid
        for index in range(count):
            table[index, 1] = 0.1 * ((index % 3) - 1)
        layer.get_code_embs().copy_(table)


def assignment_path(scale: float = 1.0) -> dict:
    """The points, the codes and the cost matrix of the REAL assignment path.

    The cost comes from the layer the trainer uses, through the same mapping the
    assignment uses, so a check on it is a check on the mechanism and not on a
    helper that the mechanism happens to call.
    """
    torch.manual_seed(0)
    model = RQVAE(_config(), in_dim=16)
    with torch.no_grad():
        model.set_radii(torch.linspace(0.1, 1.2, 24))
        model.init_codebook(torch.randn(24, 16), item_ids=torch.arange(24))
        layer = model.rq.vq_layers[0]
        _fixed_codebook(layer)
        dimension = int(layer.get_code_embs().shape[1])
        tangents = torch.zeros(3, dimension)
        tangents[:, 0] = torch.tensor([0.05, 0.2, 0.5]) * float(scale)
        mapped = layer.point_of(tangents)
        codes = layer.point_of(layer.get_code_embs())
        cost = layer.mapped_cost(mapped)
    return {"points": mapped, "codes": codes, "cost": cost}


def observe_cost(radius: torch.Tensor) -> torch.Tensor:
    """The level-1 assignment costs of one item at the given radii."""
    torch.manual_seed(0)
    model = RQVAE(_config(), in_dim=16)
    embeddings = torch.randn(24, 16)
    with torch.no_grad():
        model.init_codebook(embeddings)
        _fixed_codebook(model.rq.vq_layers[0])
        encoded = model.encoder(embeddings)
        scaled = model.scale_by_radius(encoded, radius)
        layer = model.rq.vq_layers[0]
        return layer.radial_cost(scaled)
