"""The codeword-level geometric bias.

Why a codeword and not an item: the SID's partition is what the downstream model
consumes, and a per-item weighting fragments one shared codebook without moving
that partition.  A codeword's weight moves the partition itself -- the items a
code covers change, and with them which items share a prefix.

The bounds are measured, not tuned.  On F0's frozen readout the q-kNN behaviour
edge recovery stays within 2.5% of its peak for ``alpha`` in ``[0.3, 0.7]`` and
loses outside it (``results/research/graphlet_geometry_conclusion``,
``readout_knn_geometry.json``).  The schedule therefore uses exactly that band:
the least dense codeword takes ``ALPHA_MIN``, the densest ``ALPHA_MAX``, and the
rest follow a linear ramp by average rank.  A codeword with no assigned item
keeps the population weight, so an empty code never acquires a bias it did not
earn.
"""
from __future__ import annotations

import numpy as np

#: The measured plateau of the mixed readout: recovery within 2.5% of its peak.
ALPHA_MIN = 0.3
ALPHA_MAX = 0.7
#: The weight of a codeword with no assigned item, and of the global control.
POPULATION_ALPHA = 0.5


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so equal densities get equal weights."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def allocate(density: np.ndarray, assigned: np.ndarray) -> np.ndarray:
    """One weight per codeword: a monotone ramp over the assigned codes.

    ``density`` is the mean structural density of the items a code covers and
    ``assigned`` marks the codes that cover at least one training item.  The
    result is deterministic, bounded in ``[ALPHA_MIN, ALPHA_MAX]`` and constant
    on ties.
    """
    density = np.asarray(density, dtype=np.float64)
    assigned = np.asarray(assigned, dtype=bool)
    if density.shape != assigned.shape:
        raise ValueError("density and assigned must have the same shape")
    weights = np.full(density.shape, POPULATION_ALPHA, dtype=np.float64)
    count = int(assigned.sum())
    if count == 0:
        return weights
    values = density[assigned]
    if count == 1:
        weights[assigned] = (ALPHA_MIN + ALPHA_MAX) / 2.0
        return weights
    ranks = _average_ranks(values)
    weights[assigned] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * ranks / (count - 1)
    return weights
