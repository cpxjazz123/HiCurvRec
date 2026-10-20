"""Harness² integration for GeneRec (issue #101).

The framework lives outside the repository (CPython 3.12 venv, ``harness2==0.1.0``);
everything in this package is stdlib-only and talks to it through the documented
seams -- ``BenchmarkAdapter``, ``Substrate``, ``SurfaceSpec``, ``Proposer`` -- with
no fork and no patch of the framework core.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
