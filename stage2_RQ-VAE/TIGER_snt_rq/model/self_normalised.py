"""Self-normalised structure stress for issue #191.

#190 replaced the numerator's metric and left the denominator as struct_mixed's fixed ``scale``, which is the Euclidean median; a curvature-
conditioned distance whose magnitude varies with kappa therefore could not satisfy the ratio and drove the first-level codebook onto a single
shell. This version deletes the constant and divides each pair's distance by the **median of that same batch**, computed inside the same forward
pass and detached:

    term_i = d_hat(a, b)_i / median_batch( d_hat(a, b) )

so the target is dimensionless and invariant to an overall rescaling of the positions. No scale, no epsilon and no clip threshold is introduced;
``finfo.eps`` under the square root and the asinh clamp are numerical, not tunable.
"""
from __future__ import annotations

import torch

#: ``NORM_OFF`` keeps #190's behaviour (conditioned distance over the inherited scale); ``GEOMETRY_OFF`` keeps struct_mixed's Euclidean pair.
NORM_OFF = False
GEOMETRY_OFF = False
BEHAVIOUR_OFF = False


def median_normalise(values: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Divide by the batch median, returning zeros and a count when the median vanishes."""
    median = values.detach().median()
    if float(median) <= 0.0:
        return torch.zeros_like(values), {"degenerate": 1, "median": float(median)}
    return values / median, {"degenerate": 0, "median": float(median)}
