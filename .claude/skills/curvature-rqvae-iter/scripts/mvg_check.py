#!/usr/bin/env python3
"""FCCR-1 MVG guidance.

The skill-level MVG is not a generic executable experiment gate because each
iteration has a different structural mechanism. Run the iteration-local
scripts/mvg_check.py directly from the iteration directory.

The iteration-local MVG should verify only the minimum runtime properties needed
for the chosen mechanism:
- the intended equation/path is actually executed;
- fixed curvature remains non-trainable and time-invariant;
- tensor shapes and values are finite;
- intended gradients are finite/non-zero where applicable;
- the intervention has a measurable direct computational effect.

It must print exactly 'MVG PASS' only when those checks succeed.

No hypothesis/protocol/manifest/preflight file is required.
"""

raise RuntimeError(
    "Use the iteration-local scripts/mvg_check.py directly. "
    "No preflight step or workflow record is required."
)
