"""Glue between the trainer and the structure term, kept small and testable."""
from __future__ import annotations

import random

import torch

import structure as S


def _positions(model, embeddings: torch.Tensor, item_ids: list[int]) -> torch.Tensor:
    """Positions rebuilt from the selected codes, differentiable on both sides.

    The straight-through path carries the encoder gradient; the codebook path
    (a plain gather of the selected codeword vectors) carries the codebook
    gradient, which the detach inside the straight-through estimator would
    otherwise block.
    """
    gathered = embeddings[torch.tensor(item_ids, dtype=torch.long)]
    _, _, _, tokens, levels = model.forward_with_codes(gathered)
    straight_through = levels[-1]
    codebook = model.codes_to_positions(tokens)
    return straight_through + (codebook - codebook.detach())


def structure_term(model, embeddings: torch.Tensor, graph, rng: random.Random, *,
                   scale: float, size: int, behaviour_on: bool = True,
                   geometry_on: bool | None = None) -> torch.Tensor:
    """(1/3) mean pairwise term over sampled triples, on quantized positions."""
    if not behaviour_on:
        return torch.zeros((), device=embeddings.device)
    triples = S.sample_triples(graph, size, rng)
    if not triples:
        return torch.zeros((), device=embeddings.device)
    item_ids = [item for centre, a, b, _ in triples for item in (centre, a, b)]
    positions = _positions(model, embeddings, item_ids).reshape(len(triples), 3, -1)
    h = S.triple_targets([kind for *_, kind in triples], device=positions.device)
    return S.structure_loss(positions, h, scale=scale, geometry_on=geometry_on)


def calibrate_structure_scale(model, embeddings: torch.Tensor, graph, rng: random.Random, *,
                              size: int, geometry_on: bool | None = None) -> float:
    """Median target-side distance at initialisation; deterministic for a given rng."""
    triples = S.sample_triples(graph, size, rng)
    if not triples:
        return 1.0
    item_ids = [item for centre, a, b, _ in triples for item in (centre, a, b)]
    positions = _positions(model, embeddings, item_ids).reshape(len(triples), 3, -1)
    h = S.triple_targets([kind for *_, kind in triples], device=positions.device)
    return S.calibrate_scale(positions, h)
