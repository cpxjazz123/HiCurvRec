"""The independent reference for issue #274's pair-ranking loss.

This module never imports the production code.  It answers the same question by a **different
derivation**:

* the pair distances come from the algebraic identity
  ``||a - b|| = sqrt(||a||^2 + ||b||^2 - 2 a . b)`` (the squared-norm expansion), not from the
  production's ``torch.linalg.vector_norm(a - b)``;
* the hinge mean comes from the layer-cake identity
  ``mean_i max(0, g_i) = (1/N) * sum_j max(0, g_(j) - g_(j-1))`` over the ascending sorted gaps
  ``g_(1) <= ... <= g_(N)`` with ``g_(0) = 0`` (the telescoping sum of the positive parts), not
  from the production's ``torch.relu(...).mean()``;
* the hard representation is built by an explicit per-item loop over the codebook rows, not by a
  vectorised embedding lookup.

The two paths therefore agree only if the mathematics agrees, not if the code was copied.
"""
from __future__ import annotations

import numpy as np


def hard_quantized_sum_reference(tokens: np.ndarray, codebooks: list) -> np.ndarray:
    """``q = sum_l c_l`` built by an explicit loop over the items."""
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError("tokens must be [items, levels]")
    if tokens.shape[1] != len(codebooks):
        raise ValueError("one token column per level is required")
    items = tokens.shape[0]
    dim = np.asarray(codebooks[0]).shape[1]
    q = np.zeros((items, dim), dtype=np.float64)
    for item in range(items):
        for level in range(len(codebooks)):
            q[item] += np.asarray(codebooks[level])[tokens[item, level]]
    return q


def _distance_expansion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``||a - b||`` through the squared-norm expansion."""
    squared = (a * a).sum(axis=-1) + (b * b).sum(axis=-1) - 2.0 * (a * b).sum(axis=-1)
    return np.sqrt(np.maximum(squared, 0.0))


def pair_ranking_loss_reference(q: np.ndarray, pos_b, pos_c, neg_d, neg_e, margin: float) -> float:
    """The hinge mean through the layer-cake identity.

    ``sum_i max(0, g_i) = integral_0^inf #{g_i > t} dt``.  With the gaps clamped at zero and
    sorted ascending, the integral is the weighted sum of the increments
    ``sum_j (c_(j) - c_(j-1)) * (N - j + 1)``: every gap above the previous one stays counted
    for all the smaller gaps below it.  This is a different numerical route from the production's
    ``torch.relu(...).mean()``.
    """
    pos_b = np.asarray(pos_b, dtype=np.int64)
    pos_c = np.asarray(pos_c, dtype=np.int64)
    neg_d = np.asarray(neg_d, dtype=np.int64)
    neg_e = np.asarray(neg_e, dtype=np.int64)
    d_pos = _distance_expansion(q[pos_b], q[pos_c])
    d_neg = _distance_expansion(q[neg_d], q[neg_e])
    gaps = np.sort(np.maximum(0.0, d_pos - d_neg + float(margin)))
    count = gaps.size
    total = 0.0
    previous = 0.0
    for position, value in enumerate(gaps):
        total += (float(value) - previous) * (count - position)
        previous = float(value)
    return total / count


def pair_ranking_terms_reference(q: np.ndarray, pos_b, pos_c, neg_d, neg_e, margin: float) -> dict:
    """The diagnostics through the same independent route."""
    pos_b = np.asarray(pos_b, dtype=np.int64)
    pos_c = np.asarray(pos_c, dtype=np.int64)
    neg_d = np.asarray(neg_d, dtype=np.int64)
    neg_e = np.asarray(neg_e, dtype=np.int64)
    d_pos = _distance_expansion(q[pos_b], q[pos_c])
    d_neg = _distance_expansion(q[neg_d], q[neg_e])
    hinge = np.maximum(0.0, d_pos - d_neg + float(margin))
    return {
        "pair_loss": float(hinge.mean()),
        "ranking_accuracy": float((d_pos < d_neg).mean()),
        "margin_satisfaction_rate": float((d_pos + float(margin) <= d_neg).mean()),
        "mean_distance_gap": float((d_neg - d_pos).mean()),
        "mean_positive_distance": float(d_pos.mean()),
        "mean_negative_distance": float(d_neg.mean()),
        "hinge_active_fraction": float((hinge > 0).mean()),
    }
