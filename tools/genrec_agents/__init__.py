"""Harness² integration for GeneRec (issue #101).

The package is first-party and imports nothing outside this repository;
everything in this package is stdlib-only and talks to it through the documented
seams -- ``BenchmarkAdapter``, ``Substrate``, ``SurfaceSpec``, ``Proposer`` -- with
no fork and no patch of the framework core.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
