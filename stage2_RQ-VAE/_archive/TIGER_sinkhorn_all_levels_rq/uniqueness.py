"""E3: keep distinct items apart inside their (L1, L2) prefix.

    L_unique = mean over same-prefix pairs of  <p3_i, p3_j>

``p3`` is the level-3 soft assignment (the level-3 quantiser's own distances,
centred as ``VQLayer`` centres them).  Two different items that already share the
L1 and L2 codes should not also point at the same L3 codeword, because that is
exactly the three-code collision the SID is measured on; minimising their overlap
spreads the level-3 mass inside each prefix.

The prefix comes from the model's current hard assignment and is rebuilt every
``REFRESH_EPOCHS`` epochs; only groups with at least two items contribute.
"""
from __future__ import annotations

import numpy as np
import torch

#: Share of the total loss the uniqueness term takes at the first step (0 = off).
UNIQUE_SHARE = 0.05
#: Same-prefix pairs drawn per step.
PAIRS_PER_STEP = 512
#: Pairs frozen per prefix refresh.
POOL_TARGET = 20000
REFRESH_EPOCHS = 50
EPS = 1e-12


def group_pairs(prefix_codes: np.ndarray, target: int = POOL_TARGET) -> np.ndarray:
    """Pairs of distinct items that share one (L1, L2) prefix."""
    groups = {}
    for item, code in enumerate(prefix_codes):
        if code < 0:
            continue
        groups.setdefault(int(code), []).append(item)
    pairs = []
    for items in groups.values():
        if len(items) < 2:
            continue
        for index in range(len(items) - 1):
            pairs.append((items[index], items[(index + 1) % len(items)]))
            if len(pairs) >= target:
                return np.asarray(pairs, dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def uniqueness_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Mean overlap of the two level-3 soft assignments (lower is more distinct)."""
    return (first * second).sum(dim=1).mean()


def calibrate_share(recon: float, term: float, share: float = UNIQUE_SHARE) -> float:
    share = float(share)
    return share * float(recon) / max((1.0 - share) * float(term), EPS)


def soft_assignment(distances: torch.Tensor, epsilon: float) -> torch.Tensor:
    maximum = distances.max()
    minimum = distances.min()
    middle = (maximum + minimum) / 2
    amplitude = (maximum - middle) + 1e-5
    return torch.softmax(-((distances - middle) / amplitude) / epsilon, dim=1)
