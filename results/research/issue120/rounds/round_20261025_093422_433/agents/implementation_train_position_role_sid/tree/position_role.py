"""Train-only smoothed marginal sequence-position roles."""
from __future__ import annotations

import numpy as np


def position_roles(sequences, n_items: int, alpha: float = 5.0):
    """Return role bin per item plus frozen-profile diagnostics.

    ``sequences`` must contain training-only ordered item-ID sequences.
    """
    counts = np.zeros(n_items, dtype=np.int64)
    sums = np.zeros(n_items, dtype=np.float64)
    total_sum = 0.0
    total_count = 0
    for sequence in sequences:
        seq = np.asarray(sequence, dtype=np.int64)
        if len(seq) < 2:
            continue
        if np.any(seq < 0) or np.any(seq >= n_items):
            raise ValueError("sequence item ID outside embedding row space")
        xs = np.linspace(0.0, 1.0, num=len(seq), dtype=np.float64)
        np.add.at(counts, seq, 1)
        np.add.at(sums, seq, xs)
        total_sum += float(xs.sum())
        total_count += len(seq)
    mu0 = total_sum / total_count if total_count else 0.5
    means = (sums + alpha * mu0) / (counts + alpha)
    observed = means[counts > 0]
    cuts = np.quantile(observed, [0.25, 0.5, 0.75]) if len(observed) else np.array([.25, .5, .75])
    roles = np.searchsorted(cuts, means, side="right").astype(np.int64)
    # n=0 items use the global-median bin (the second quartile bin).
    roles[counts == 0] = int(np.searchsorted(cuts, mu0, side="right"))
    return roles, {"counts": counts, "means": means, "mu0": mu0, "cuts": cuts}
