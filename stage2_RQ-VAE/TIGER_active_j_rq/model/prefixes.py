"""Prefix and aggregation helpers shared by the legacy diagnostic path."""
from __future__ import annotations

import torch


def cumulative_prefixes(level_codes):
    """Running sum of the selected level values: q^(l) = sum_{j<=l} e_j."""
    codes = list(level_codes)
    if not codes:
        raise ValueError("at least one level is required")
    shape = codes[0].shape
    running = torch.zeros_like(codes[0])
    prefixes = []
    for code in codes:
        if code.shape != shape:
            raise ValueError("every level must contribute the same shape")
        running = running + code
        prefixes.append(running.clone())
    return torch.stack(prefixes, dim=0)


def aggregate_layer_losses(values, weights=None):
    """Equal alpha weights by default; explicit so a change cannot be silent."""
    terms = [value for value in values]
    if not terms:
        raise ValueError("at least one level loss is required")
    if weights is None:
        weights = (1.0 / len(terms),) * len(terms)
    if len(weights) != len(terms):
        raise ValueError("one weight per level is required")
    total = terms[0] * float(weights[0])
    for term, weight in zip(terms[1:], weights[1:]):
        total = total + term * float(weight)
    return total
