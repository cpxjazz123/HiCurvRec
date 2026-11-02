"""The pre-registered profile-to-curvature mapping (Issue #129, WP4).

The constants are fixed before any training and must not be tuned against a test
recall. `z_l` is the size-preserving-null-standardised curvature of the frozen
pilot SID on the TRAIN-only graph.
"""
from __future__ import annotations

C0 = 1.0
ALPHA = 0.5
C_MIN = 0.05
C_MAX = 4.0


def mapping(z: float) -> float:
    """c_l = clip(c0 * exp(-alpha * z), c_min, c_max)."""
    import math

    return float(min(max(C0 * math.exp(-ALPHA * z), C_MIN), C_MAX))


def profile(zs: list[float]) -> tuple:
    return tuple(mapping(z) for z in zs)
