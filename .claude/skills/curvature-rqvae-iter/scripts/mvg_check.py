#!/usr/bin/env python3
"""FCCR-1 MVG guidance.

The skill-level MVG is not a generic executable experiment gate because each
condition has a different structural mechanism. Run
stage2_RQ-VAE/curvature_RQ-VAE/scripts/mvg_check.py directly from that
directory; there is exactly one such directory now.

That check should verify only the minimum runtime properties needed for the
chosen mechanism:
- the intended equation/path is actually executed;
- fixed curvature remains non-trainable and time-invariant;
- tensor shapes and values are finite;
- intended gradients are finite/non-zero where applicable;
- the intervention has a measurable direct computational effect.

It must report 'MVG PASS' only when those checks succeed.

No hypothesis/protocol/manifest/preflight file is required.
"""

raise RuntimeError(
    "Run stage2_RQ-VAE/curvature_RQ-VAE/scripts/mvg_check.py directly. "
    "No preflight step or workflow record is required."
)
