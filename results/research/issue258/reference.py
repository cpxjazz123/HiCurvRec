"""Independent scalar reference for the R05 level-to-mixture routing.

This is not a translation of the torch module: it has no parameters, no
broadcasting, no module state and no autograd.  Every item and level is walked
in plain Python, softmax rows are normalised with ``math.exp`` one scalar at a
time, and the squared Euclidean cost is an explicit double loop.  It exists to
answer one question from the R05 contract: which curvature mixture does each
level consume, and does the level-2 cost equal the mixture-weighted expert
distance (Euclidean when the curvature switch is off)?
"""
from __future__ import annotations

import math


def softmax_rows(logits):
    """Row-wise softmax of a list of rows, one scalar exponent at a time."""
    rows = []
    for row in logits:
        peak = max(row)
        exponents = [math.exp(float(value) - float(peak)) for value in row]
        total = sum(exponents)
        rows.append([value / total for value in exponents])
    return rows


def routing(roles, logits, gate, *, behavior_on=True, global_index=None):
    """Per-item, per-level mixtures ``[N][3][3]`` from explicit selection.

    ``roles`` is the frozen per-item role label (``-1`` = uncovered),
    ``logits`` the role table rows, ``gate`` the item's learned ``[N][3][3]``
    mixtures.  Level index 1 (the second semantic level) is the only level that
    may read the role table; levels 0 and 2 copy the gate unchanged.
    """
    mixtures = softmax_rows(logits)
    if global_index is None:
        global_index = len(mixtures) - 1
    uncovered = len(mixtures) - 2
    out = []
    for item, role in enumerate(roles):
        per_level = []
        for level in range(3):
            if level == 1:
                row = global_index if not behavior_on else (uncovered if role < 0 else int(role))
                per_level.append(list(mixtures[row]))
            else:
                per_level.append([float(value) for value in gate[item][level]])
        out.append(per_level)
    return out


def euclidean_cost(residual, codewords, weights, scale):
    """``sum_e w_e * d_e`` with every expert Euclidean at the common scale.

    With the curvature switch off all three experts are the same squared
    Euclidean distance, so the mixture collapses to its normalised sum (one).
    """
    out = []
    for item, x in enumerate(residual):
        row = []
        for code in codewords:
            distance = 0.0
            for a, b in zip(x, code):
                delta = float(a) / float(scale) - float(b) / float(scale)
                distance += delta * delta
            row.append(distance * sum(float(w) for w in weights[item]))
        out.append(row)
    return out
