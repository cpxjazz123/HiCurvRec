#!/usr/bin/env python3
"""FCCR-1 gradient-check compatibility note.

The old generic checker is retired because it:
- required a CLI iter_id;
- assumed learnable curvature flags;
- could incorrectly require gradients/updates on fixed curvature.

Under FCCR-1, gradient health is checked inside the iteration-local no-CLI
scripts/mvg_check.py after scripts/preflight_contract.py passes.

Required model-gradient checks:
- total loss requires grad;
- non-curvature RQ-VAE/model parameters receive finite nonzero gradients;
- fixed curvature buffers require no gradient and must never be optimized.

This file exits deliberately to prevent accidental use of the obsolete checker.
"""

raise RuntimeError(
    "Generic grad_check.py is retired under FCCR-1. "
    "Use preflight_contract.py and the iteration-local no-CLI mvg_check.py."
)
