"""The one guard the pairwise losses share: same shape, not empty, floating point.

`torch.nn.functional.mse_loss` broadcasts a `(B, 1)` against a `(B, D)` and returns a
number that looks like a loss; it only warns. It also returns NaN for a zero-element
batch. Both shapes of mistake would reach the optimiser as a plausible scalar, so the
pairwise terms refuse them here instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pair_error(a: torch.Tensor, b: torch.Tensor, *, term: str) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("%s: shapes differ (%r vs %r); broadcasting is not a loss"
                         % (term, tuple(a.shape), tuple(b.shape)))
    if a.numel() == 0:
        raise ValueError("%s: the batch is empty, so the mean is undefined" % term)
    if not (a.is_floating_point() and b.is_floating_point()):
        raise TypeError("%s: both sides must be floating point, got %r and %r"
                        % (term, a.dtype, b.dtype))
    return F.mse_loss(a, b)
