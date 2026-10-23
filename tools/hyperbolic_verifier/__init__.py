"""The independent hyperbolic verifier for issue #110 (WP2).

The package holds the verifier's own geometry, written from the definitions rather than
taken from any mechanism tree, and the H1-H7 gates that use it. The oracle never imports a
candidate's code: a candidate is compared against this implementation, not against itself.
"""

from __future__ import annotations

__all__ = ["geometry"]
